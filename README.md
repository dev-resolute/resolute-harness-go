# resolute-harness-go

The agent harness framework for Go — sessions, durable execution, event-sourced conversations, and HTTP exposure for agents built on [`resolute-agent-core-go`](https://github.com/dev-resolute/resolute-agent-core-go) and [`resolute-llm-go`](https://github.com/dev-resolute/resolute-llm-go).

Design docs: [`docs/architecture.md`](docs/architecture.md) and the ADRs in [`docs/adr/`](docs/adr/). Vocabulary: [`CONTEXT.md`](CONTEXT.md).

## What this is

`resolute-agent-core-go` is a single-conversation agent loop: one `Agent`, one session, per-prompt event streams, in-process only. It deliberately (ADR-0006/0007 in that repo) does not solve multi-session management, durability, persistence beyond JSONL files, or network transport.

`resolute-harness-go` is that missing outer layer. It is a Go port of the architecture of [flue](https://github.com/withastro/flue) — the TypeScript "Agent Harness Framework" built on `@earendil-works/pi-agent-core` and `@earendil-works/pi-ai`, the exact upstream libraries the resolute Go libraries were ported from. Flue is the proven harness for pi; this is the harness for the pi ports.

What the harness adds on top of agent-core:

- **Durable submission engine** — idempotent admission, lease-based ownership, attempt tracking, per-session head-of-line serialization, two-phase settlement, crash reconciliation, durability budgets. `kill -9` mid-turn resumes cleanly. No external orchestrator (no Temporal, no DBOS) — see ADR-0002.
- **Event-sourced conversation** — an append-only log of canonical records, reduced into a parent-linked message *tree*. Recovery, compaction (re-parenting), branching, and future sub-agent sessions all fall out of one structure. The `Agent`'s in-RAM state is rebuilt from this log; a `pi.SessionRepo` adapter projects the active leaf path into agent-core unchanged — see ADR-0003.
- **HTTP transport** — `POST` = 202 admission, `GET` = SSE replay-from-offset then live tail, `?wait=true` blocking convenience, `POST …/steer` and `…/followup` for mid-run control — see ADR-0004.
- **`user` vs `signal` inbound kinds** — direct exchanges vs. one participant's activity in a multi-party conversation (a Slack thread, a GitHub issue), in the record schema from day one — see ADR-0005.
- **Structured results** — attach a JSON Schema to a prompt (`resultSchema`) and get validated JSON on the settled record, with corrective-turn retries.
- **Pluggable storage** — one narrow adapter contract (Submission + Conversation + Attachment stores), memory and SQLite in-tree, shipped conformance test suite — see ADR-0006 and "Writing a store adapter" below.
- **Observability seams** — a typed event `Observer` plus an execution `Interceptor` at every operation boundary (attempt, operation, model turn, tool); the future OTel adapter needs no engine changes — see ADR-0008.

## Quickstart

```bash
go run ./examples/basic
```

runs keyless (a deterministic local provider stands in for the model) with a SQLite store in `./harness-data`. Set `GEMINI_API_KEY` to run against Gemini instead. Then:

```bash
# fire-and-forget: 202 with {submissionId, conversationId}
curl -s localhost:8484/agents/assistant/demo -d '{"kind":"user","body":"what time is it?"}'

# or block until the durable result:
curl -s 'localhost:8484/agents/assistant/demo?wait=true' -d '{"kind":"user","body":"hello there"}'

# watch the conversation as canonical records over SSE (replay + live tail;
# resume from any offset with -H "Last-Event-ID: <record id>"):
curl -N localhost:8484/agents/assistant/demo

# steer an in-flight run:
curl -s localhost:8484/agents/assistant/demo/steer -d '{"body":"answer in French"}'
```

**Durability walkthrough:** dispatch a prompt, `kill -9` the process before it settles, and start it again. The interrupted submission's lease expires, a fresh attempt reclaims it over the same SQLite file, and the run settles; the SSE replay shows records from both attempts, distinguished by `attemptId`.

Composition is explicit (ADR-0009) — no discovery, no codegen:

```go
rt, _ := harness.NewRuntime(harness.Config{
    Agents: map[string]harness.AgentDefinition{
        "assistant": {Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
            provider, err := gemini.New(gemini.Config{APIKey: env.Secret("GEMINI_API_KEY")})
            if err != nil {
                return harness.AgentRuntimeConfig{}, err
            }
            return harness.AgentRuntimeConfig{
                Model:         "gemini/gemini-3.1-pro-preview",
                ContextWindow: 1_000_000,
                Providers:     []llm.LLMProvider{provider},
                SystemPrompt:  promptFor(id), // per-instance setup is first-class
                Tools:         myTools,
            }, nil
        }},
    },
    Store: store, // sqlite.Open(dir) or memory.New()
})
rt.Start(ctx)
http.ListenAndServe(":8484", rt.Handler()) // auth = your middleware
```

`rt.Dispatch` / `rt.Wait` / `rt.Steer` / `rt.FollowUp` / `rt.Compact` expose the same operations in-process.

## Writing a store adapter

The store contract (`harness.Store` = `SubmissionStore` + `ConversationStore` + `AttachmentStore`) is one tier for every backend — no SQL-only extensions. The **exported conformance suite is the contract**: a third-party adapter (Postgres, Mongo, …) is correct exactly when it passes the same suite the in-tree memory and SQLite backends pass:

```go
import "github.com/dev-resolute/resolute-harness-go/storetest"

func TestConformance(t *testing.T) {
    storetest.Run(t, func(t *testing.T) harness.Store { return myStore(t) })
}
```

The suite pins every engine-visible invariant — admission idempotency and payload conflict, runnable-head-per-session, claim CAS, attempt markers, lease renew/expiry, two-phase settlement, record ordering and offset reads, digest-keyed attachments — so a subtly wrong adapter fails tests instead of corrupting production.

## What this is not (v1)

Channel adapters (Slack, Discord, …), a client SDK, React bindings, a dev console, a CLI, workflows, sandboxes/`shell`, and sub-agent `task` delegation are all deferred. The seams for each exist in the design; the packages do not. See `docs/architecture.md` §10.

## Layering

```
resolute-llm-go        LLM providers, streaming        (v0.8.0)
resolute-agent-core-go agent loop, tools, skills       (v0.6.1)
resolute-harness-go    sessions, durability, transport (this repo)
```

## License / lineage

Architecture derived from flue (Apache-2.0, © the flue authors). This is an independent Go implementation, not a source port.
