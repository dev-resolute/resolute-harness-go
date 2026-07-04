# resolute-harness-go — Architecture

Design document for v1. Decisions were made in a structured design review on 2026-07-04; each load-bearing decision has an ADR in `docs/adr/`. This document is the integrated picture.

Reference implementation studied: [flue](https://github.com/withastro/flue) `packages/runtime` (the TS harness built on upstream pi). File references of the form `flue:session.ts` point there for implementers who want the proven semantics.

---

## 1. Position in the stack

```
┌──────────────────────────────────────────────────────────┐
│ app (your main.go)                                       │
│   harness.NewRuntime(Config{Agents, Store, …})           │
├──────────────────────────────────────────────────────────┤
│ resolute-harness-go                                      │
│   transport (net/http)   ← POST 202 / GET SSE            │
│   coordinator            ← claim loop, leases, recovery  │
│   session engine         ← operations, turn recovery     │
│   conversation           ← records, reducer, projection  │
│   stores                 ← Submission/Conversation/Attach│
├──────────────────────────────────────────────────────────┤
│ resolute-agent-core-go (pi)   agent loop, tools, skills  │
│ resolute-llm-go (llm)         providers, streaming       │
└──────────────────────────────────────────────────────────┘
```

The harness **owns durability and topology**; agent-core **owns the loop**. The harness never re-implements turn mechanics, tool dispatch, steering, or skill rendering — it drives `pi.Agent` and observes its event stream.

## 2. Object hierarchy

Mirrors flue's (`flue:AGENTS.md`):

```
AgentDefinition   registered by name in Runtime config; an initializer fn
  AgentInstance   one durable environment, addressed /agents/{name}/{id}
    Session       named conversation inside an instance ("default", …)
      Operation   Prompt | Compact   (v1.5: Task; later: Shell)
        Turn      one pi.Agent LLM round-trip
Submission        one admitted unit of work for a session (durable)
```

- **`AgentDefinition`** is not a config value but a function: `Initialize(ctx, InstanceID, Env) (AgentRuntimeConfig, error)` (ADR-0009). Per-instance dynamic setup — tenant prompts, per-user tools — is first-class. `AgentRuntimeConfig` declares: model ref (`provider/model`), context window + token budgets (ADR-0007: no catalog), system prompt, tools, skills, compaction settings, durability budget (max attempts, timeout).
- **`AgentInstance`** is materialized on first dispatch: instance identity + its conversations exist only in the stores.
- **`Session`** is a named conversation with a persisted conversation ID. Its public surface is small and typed: `Prompt`, `Compact` (+ `Steer`/`FollowUp` passthrough to a live run). `Prompt` optionally takes a JSON Schema for a validated structured result (ADR-0010 §ops).

## 3. The canonical conversation (source of truth)

**Event-sourced, append-only, tree-shaped** (ADR-0003; `flue:conversation-records.ts`, `conversation-reducer.ts`).

### 3.1 Records

Every record shares an envelope:

```go
type RecordEnvelope struct {
    ID             string // ULID; also the SSE event id / stream offset
    Kind           RecordKind
    ConversationID string
    Session        string
    SubmissionID   string // correlation to the durable submission
    TurnID         string
    AttemptID      string // which attempt authored this record
    Time           time.Time
}
```

v1 record kinds:

| Kind | Payload highlights |
|---|---|
| `conversation_created` | agent name, instance id, session name |
| `user_message` | body, `[]AttachmentRef` |
| `signal` | type, body, sender attributes, optional tag (ADR-0005) |
| `assistant_message_started` | model ref, turn correlation |
| `assistant_text_delta` / `assistant_thinking_delta` | batched delta text |
| `assistant_tool_call` | call id, tool name, args |
| `tool_outcome` | call id, `pi.ToolResult` (content, data, isError) |
| `assistant_message_completed` | final `pi.Message` |
| `compaction` | summary body, `firstKeptEntryID` (re-parent point) |
| `submission_settled` | status, error, structured result if requested |

Attachments are stored out-of-line in the AttachmentStore, keyed by content digest; records carry only `AttachmentRef{Digest, MediaType, Size}` (ADR-0006). This is in the schema from day one even though v1 has no image ingestion path — the schema is the expensive thing to change.

Delta records are **batched** before append (flush on size/interval/message-end; `flue:session.ts enqueueCanonical/flushCanonical`) so the log doesn't take one row per token.

### 3.2 Reducer

A **pure function** `Reduce([]Record) ConversationTree`. Every reduced entry has a `ParentID`; the tree supports:

- **Recovery** — agent state is never trusted in RAM; on every claim the coordinator re-reduces the log and rebuilds context.
- **Compaction** — a `compaction` record becomes a summary node; entries after `firstKeptEntryID` re-parent onto it. Nothing is deleted.
- **Branching / child sessions** — future `task` sub-agents attach as child branches without schema change (envelope already carries session + turn correlation).

The reducer is pure so it is trivially property-tested: `Reduce(log)` must be deterministic, and `Reduce(log[:n])` must be a prefix-consistent tree for all n.

### 3.3 Projection into agent-core

agent-core is configured with a `pi.SessionRepo` implementation backed by the reduced tree — the **projection adapter** (ADR-0003):

| `pi.SessionRepo` method | Adapter behavior |
|---|---|
| `Create` | maps to `conversation_created` (or no-op when conversation pre-exists) |
| `Append(msgs…)` | **no-op** — canonical records are authored from the event stream, not from the repo (see §5.3); the adapter is a read-side view |
| `Load` | returns the active-leaf-path `[]pi.Message` from the reduced tree |
| `List` | enumerates conversations from the ConversationStore |
| `AppendBranchSummary` | writes a `compaction` record |
| `LoadBranchSummaries` | serves summaries from reduced `compaction` nodes, so agent-core's own `BuildLLMContext` substitution keeps working |
| `Delete` | conversation deletion via the store |

Verified against agent-core v0.6.0 source: `SessionRepo` is exactly these seven methods; `ToolCallEndEvent` carries the full `ToolResult`; `MessageEndEvent`/`AgentEndEvent` carry complete messages — so the event stream suffices to author every record and `Append` can be a no-op without losing fidelity. If drift appears (an event missing data a record needs), the fix is a small additive change in agent-core, which we own.

## 4. The durable submission engine

Hand-rolled, flue-exact semantics (ADR-0002, ADR-0010; `flue:agent-execution-store.ts`, `node/agent-coordinator.ts`). No Temporal, no DBOS: the conversation log is already the checkpoint, so the engine only has to guarantee *don't lose or double-run a submission*.

### 4.1 Submission lifecycle

```
Dispatch ──admit──▶ queued ──claim──▶ running ──▶ terminalizing ──▶ settled
                              ▲                       │
                              └── lease expiry / crash┘  (new attempt)
```

```go
type Submission struct {
    ID             string // = dispatch id → idempotency key
    SessionKey     string // agent/instance/session
    Status         SubmissionStatus
    Input          DispatchMessage // user or signal
    AttemptCount   int
    AttemptID      string
    OwnerID        string
    LeaseExpiresAt time.Time
    CreatedAt      time.Time
}
```

### 4.2 Invariants (the conformance suite pins all of these)

1. **Idempotent admission** — re-admitting the same dispatch id with identical payload returns the same submission; with a different payload returns a conflict error.
2. **Per-session head-of-line serialization** — `ListRunnable` returns only the *oldest* unsettled submission per session key; different sessions run concurrently.
3. **Lease-based ownership** — `Claim` is an atomic CAS `queued→running` recording attempt id, owner id, lease expiry. A heartbeat renews; a lease scan reclaims expired work.
4. **Attempt markers** — an attempt inserts a marker before doing work, so reconciliation can distinguish "started then died" from "never started."
5. **Two-phase settlement** — `ReserveSettlement` then `FinalizeSettlement`, so a durable terminal record exists exactly once even if the process dies between phases.
6. **Startup reconciliation** — on boot, interrupted `running` submissions are handed to a fresh attempt (`AttemptCount++`) up to the durability budget.
7. **Budgets recomputed from history** — max attempts (default 10) and timeout (default 1h) are evaluated from durable state, so a restart resumes with the same remaining budget, not a fresh one.

These semantics are implemented in full from day one even though v1 runs single-process (ADR-0010): `kill -9` mid-turn must resume cleanly, and multi-node later is a Postgres store adapter, not an engine change.

### 4.3 Turn-loop recovery (inside one attempt)

Owned by the harness session engine (`flue:session.ts runModelTurnWithRecovery`), because agent-core has **no in-loop auto-compaction** (verified: `Agent.Compact` is idle-only; `ShouldCompact` is an exported helper):

- **Context overflow** → run compaction (via `Agent.Compact`, which lands as a `compaction` record through the projection), then retry the turn.
- **Transient model errors** → budgeted backoff retry; the consecutive-failure count is recomputed from the durable record history, so a crash mid-backoff doesn't reset the budget.
- **Cooperative halt** — between turns, check the durability deadline and lease validity.

## 5. Runtime flow, end to end

1. **Admission.** `POST /agents/{name}/{id}` (or in-process `rt.Dispatch`) validates the inbound `DispatchMessage` (`User` or `Signal`), admits it idempotently, and returns **202** `{submissionId, conversationId}`. Request lifetime ends here.
2. **Claim.** The coordinator's claim loop picks the runnable head submission for some session, CASes it to `running`, starts heartbeating.
3. **Materialize.** Load + reduce the conversation log. Call `AgentDefinition.Initialize(ctx, id, env)` → `AgentRuntimeConfig`. Construct `pi.Agent` with: the config's providers/model/tools/skills, `Session: projectionAdapter`, and the config's compaction settings.
4. **Drive.** Append the input record (`user_message`/`signal`). Call `agent.Prompt(...)` and consume its per-prompt `EventStream`:
   - each event → (a) an ephemeral `HarnessEvent` to registered Observers, and (b) zero or more canonical records (deltas batched).
   - turn boundaries run the recovery ladder (§4.3).
5. **Settle.** On `PromptResult`: if a structured result schema was requested, validate the final message against it (retry-with-feedback up to a small budget). Two-phase settle; append `submission_settled`.
6. **Read.** Any number of clients `GET /agents/{name}/{id}` with `Last-Event-ID`: the transport replays canonical records from that offset as SSE, then tails live appends. Reconnect/resume/multi-reader are free because the read side is just "replay the log from N."

## 6. Transport contract (ADR-0004)

```
POST /agents/{name}/{id}            → 202 {submissionId, conversationId}
POST /agents/{name}/{id}?wait=true  → 200 settled result (blocking convenience)
GET  /agents/{name}/{id}            → SSE canonical records; Last-Event-ID = record ID (replay offset)
POST /agents/{name}/{id}/steer      → steer the in-flight run
GET  /healthz                       → liveness
```

Exposed as an `http.Handler` from the Runtime — the app mounts it wherever it wants (its own mux, middleware, auth). The harness ships no auth in v1; the seam is standard `net/http` middleware.

SSE framing: `id:` = record ID (ULID, orderable), `event:` = record kind, `data:` = the record JSON. The wire format **is** the record schema — no second serialization layer.

## 7. Providers and models (ADR-0007)

Catalog-free, consistent with resolute-llm-go ADR-0008. No registry type. `AgentRuntimeConfig` declares:

```go
Model         string // "gemini/gemini-3.1-pro-preview"
ContextWindow int    // required; drives compaction thresholds
MaxTokens     int
Providers     []llm.LLMProvider // plain values, constructed in Go
```

agent-core already resolves `provider/model` refs by scanning `Providers` by name; the harness adds nothing.

## 8. Observability (ADR-0008)

Two seams, both in v1; the OTel adapter is a later, purely additive package:

```go
type Observer func(HarnessEvent)                    // data: typed union — submission/operation/turn/delta/tool/compaction lifecycle
type Interceptor func(ctx context.Context, op OpInfo, next func(context.Context) error) error
```

- **Observer**: synchronous, read-only, cheap; panics/errors are logged, never fatal. Carries the same correlation IDs as record envelopes.
- **Interceptor**: onion middleware wrapped around every operation boundary — submission attempt, operation, model turn, tool execution — so trace context propagates through `context.Context` natively (no AsyncLocalStorage gymnastics needed in Go). Call sites exist from day one because retrofitting them is invasive; implementations can come later.

agent-core's `Hooks.BeforeProviderRequest/AfterProviderResponse` remain available for wire-level concerns; the harness does not wrap them.

## 9. Composition API (ADR-0009)

```go
rt, err := harness.NewRuntime(harness.Config{
    Agents: map[string]harness.AgentDefinition{
        "support": {Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
            return harness.AgentRuntimeConfig{
                Model:         "gemini/gemini-3.1-pro-preview",
                ContextWindow: 1_000_000,
                Providers:     []llm.LLMProvider{gemini.New(gemini.Config{APIKey: env.Secret("GEMINI_API_KEY")})},
                SystemPrompt:  loadPromptFor(id),
                Tools:         supportTools,
            }, nil
        }},
    },
    Store:        sqlite.Open(dataDir), // or memory.New()
    Observers:    []harness.Observer{logObserver},
    Interceptors: []harness.Interceptor{traceInterceptor},
})
// rt.Start(ctx) / rt.Handler() / rt.Dispatch(ctx, …) / rt.Close()
```

Explicit registration, no init()-magic, no codegen. A future CLI layers scaffolding on top of this API without changing it.

## 10. Scope

**v1 (this repo, now):** conversation records + reducer + projection adapter; submission engine with full lease/attempt/reconciliation semantics; session engine with Prompt/Compact + structured results + turn recovery; memory + SQLite stores (modernc.org/sqlite, in-tree, one module — ADR-0006) + conformance suite; HTTP transport; Observer/Interceptor seams; one runnable example app.

**Deferred, in rough order:** `task`/subagents (child sessions as tree branches — schema already supports it); first channel adapter (Slack) + Go client SDK (validates the two external seams); OTel adapter; Postgres store (nested module, bring-your-own-`*sql.DB`); sandbox abstraction + `shell` op; workflows (`defineWorkflow` analog; would add RunStore); CLI + dev console; React/browser bindings; studio rebuilt on this harness.

## 11. Testing strategy

Test-driven throughout (project CLAUDE.md rule 4):

- **Store conformance suite** — exported package (like flue's `defineStoreContractTests` and the Temporal-era `storagetest`); memory and SQLite must pass identically; future adapters get correctness for free.
- **Reducer property tests** — determinism, prefix consistency, re-parent correctness under compaction.
- **Engine crash tests** — first-class: kill the coordinator between every pair of engine phases (post-admit, post-claim, mid-turn, post-reserve-settlement) and assert clean resume with correct attempt/budget accounting.
- **Loop tests** — `resolute-llm-go/mock.MockProvider` for deterministic agent behavior; no live-provider dependency in CI.
- **Transport tests** — SSE replay-from-offset, reconnect mid-stream, `?wait` settlement, idempotent re-POST.

## 12. Open questions — resolved by v0.1.0 implementation

What implementation actually chose (each remains a tracked default, revisitable):

- **Delta-batch flush policy** — flush on 1024 bytes, 200ms staleness, and every message boundary (message end always flushes; any non-delta record flushes pending deltas first so the log stays ordered). Configurable via `Config.DeltaFlushBytes` / `Config.DeltaFlushInterval`.
- **Structured-result retry budget and feedback shape** — default 2 corrective turns, per-Prompt override via `DispatchMessage.ResultRetries`. The feedback prompt is a canonical `user_message` (visible in the stream) carrying the validation error, the schema, and an instruction to reply with only conforming JSON (`correctiveMessage`). Terminal failure settles `failed/result_schema_invalid`.
- **`Signal` → LLM-context rendering** — a custom `"signal"`-typed transcript message whose body is the signal JSON (type, body, sender, tag); agent-core's default conversion surfaces custom types as text, so sender identity stays distinguishable from the principal. Deliberately minimal and **provisional** until the first channel adapter validates it.
- **Steer/FollowUp durability** — live-run-only passthrough shipped (structured `ErrNoRunInFlight` otherwise); durable steer remains deferred until channels.
- **Compaction projection (deviation from §3.3's letter, same outcome)** — `Load` serves the re-parented active leaf path with the summary **inline** as a `branch_summary` message, and `LoadBranchSummaries` returns nil. Because `Agent.Compact` computes its cut over whatever `Load` returns, this keeps indices self-consistent and makes repeated compactions incremental (each sees the already-compacted view); serving summaries separately as §3.3's table described would double-count. Requires `resolute-agent-core-go` **v0.6.1** (`CompactOpts.SessionID`, additive) for the manual idle-session `Compact`.
- **Transient model errors** — retried as fresh attempts with backoff (base = `ClaimInterval`, doubling per durable `AttemptCount`, capped 5s), so the budget is the max-attempts budget and survives restarts. `llm.ErrProviderFatal` stays terminal; context overflow gets compact-and-retry (2 per attempt) before failing.
