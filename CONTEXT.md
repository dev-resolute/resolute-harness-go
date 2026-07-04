# resolute-harness-go — Vocabulary

Fixes the domain terms for this repo so design and code reviews use the same words. Terms deliberately diverge from the older Temporal-based `resolute-sh` workspace vocabulary — do not import that CONTEXT.md's meanings here.

## Hierarchy

**Agent definition** — a named initializer registered in `Runtime` config: `Initialize(ctx, InstanceID, Env) (AgentRuntimeConfig, error)`. Not a config value; a function, so per-instance setup is first-class.
_Avoid:_ agent config (that's the function's *result*), agent template.

**Agent instance** — one durable agent environment, addressed `/agents/{name}/{id}`. Exists only in the stores; materialized on claim.

**Session** — a named conversation inside an instance (default name `"default"`). Public surface: `Prompt`, `Compact`, `Steer`, `FollowUp`.
_Avoid:_ thread, chat.

**Operation** — one typed unit of session work (`Prompt` | `Compact`; later `Task`, `Shell`). An operation spans one or more turns.

**Turn** — one `pi.Agent` LLM round-trip inside an operation. The harness never implements turn mechanics; agent-core owns them.

## Durability

**Dispatch** — an inbound request to run work: `{agent, instance id, DispatchMessage}`. Admission, not execution.

**Submission** — the durable record of one admitted dispatch: `queued → running → terminalizing → settled`. The unit of leasing, attempts, and settlement. Idempotency key = dispatch id.
_Avoid:_ job, task (conflicts with the future `Task` operation).

**Coordinator** — the claim loop that leases runnable submissions and drives sessions. v1: one per process.

**Lease** — time-bounded ownership of a running submission, renewed by heartbeat, reclaimed on expiry.

**Attempt** — one execution try of a submission. Attempt markers prove an attempt started; budgets (max attempts, timeout) are recomputed from durable history.

**Settlement** — the two-phase terminal transition (`ReserveSettlement` → `FinalizeSettlement`) guaranteeing exactly one durable terminal record.

**Reconciliation** — startup scan that hands interrupted `running` submissions to fresh attempts.

## Conversation

**Canonical record** — one append-only entry in the durable conversation log (`user_message`, `signal`, `assistant_text_delta`, `tool_outcome`, `compaction`, `submission_settled`, …). The wire format of the SSE stream **is** the record schema.
_Avoid:_ event (that's the ephemeral observer stream), message (that's a pi/llm type).

**Envelope** — the correlation header every record carries (ID/kind/conversation/session/submission/turn/attempt/time). Record ID doubles as the SSE offset.

**Reducer** — the pure function `Reduce([]Record) ConversationTree`. Deterministic and prefix-consistent; the only way conversation state is derived.

**Conversation tree** — the reduced, parent-linked projection. Compaction re-parents; branches and child sessions attach without schema change.

**Active leaf path** — the current branch of the tree, root → leaf; what the projection adapter serves to agent-core as `[]pi.Message`.

**Projection adapter** — the `pi.SessionRepo` implementation backed by the reduced tree. Read-side view: `Load`/`LoadBranchSummaries` serve projections; `Append` is a no-op (records are authored from the event stream); `AppendBranchSummary` writes a `compaction` record.
_Avoid:_ session repo bridge, repo shim.

**User message vs Signal** — the two inbound `DispatchMessage` kinds. `User` = a direct 1:1 exchange with the agent's principal. `Signal` = one participant's activity (sender attributes, type, optional tag) in a multi-party conversation the agent participates in (a Slack thread, a GitHub issue). Never force a signal into `User` — it conflates other participants with the assistant's own user.

**Attachment ref** — the in-record pointer `{digest, media type, size}` to bytes stored out-of-line in the AttachmentStore.

## Stores

**Store contract** — the single narrow persistence interface every backend implements: SubmissionStore + ConversationStore + AttachmentStore. One tier for all backends; conformance-tested.
_Avoid:_ adapter tiers, "expert" stores.

**Conformance suite** — the exported test suite every store implementation must pass (memory and SQLite in-tree; future adapters run the same suite).

## Observability

**Observer** — a synchronous subscriber of ephemeral `HarnessEvent`s (typed union: submission/operation/turn/delta/tool/compaction lifecycle). Data only; failures logged, never fatal.
_Avoid:_ event sink, run observer (Temporal-workspace terms).

**Interceptor** — onion middleware `func(ctx, OpInfo, next)` wrapped around every operation boundary for trace-context propagation. The pair Observer+Interceptor is the whole observability surface; OTel is an adapter over both.

## Recovery

**Turn recovery** — the in-attempt ladder: context overflow → compact + retry; transient model error → budgeted backoff (budget recomputed from record history); cooperative halt between turns. Harness-owned — agent-core has no in-loop auto-compaction.
