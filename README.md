# resolute-harness-go

The agent harness framework for Go — sessions, durable execution, event-sourced conversations, and HTTP exposure for agents built on [`resolute-agent-core-go`](https://github.com/dev-resolute/resolute-agent-core-go) and [`resolute-llm-go`](https://github.com/dev-resolute/resolute-llm-go).

**Status: design phase.** No code yet. Read [`docs/architecture.md`](docs/architecture.md) first, then the ADRs in [`docs/adr/`](docs/adr/).

## What this is

`resolute-agent-core-go` is a single-conversation agent loop: one `Agent`, one session, per-prompt event streams, in-process only. It deliberately (ADR-0006/0007 in that repo) does not solve multi-session management, durability, persistence beyond JSONL files, or network transport.

`resolute-harness-go` is that missing outer layer. It is a Go port of the architecture of [flue](https://github.com/withastro/flue) — the TypeScript "Agent Harness Framework" built on `@earendil-works/pi-agent-core` and `@earendil-works/pi-ai`, the exact upstream libraries the resolute Go libraries were ported from. Flue is the proven harness for pi; this is the harness for the pi ports.

What the harness adds on top of agent-core:

- **Durable submission engine** — idempotent admission, lease-based ownership, attempt tracking, per-session head-of-line serialization, two-phase settlement, crash reconciliation. `kill -9` mid-turn resumes cleanly. No external orchestrator (no Temporal, no DBOS) — see ADR-0002.
- **Event-sourced conversation** — an append-only log of canonical records, reduced into a parent-linked message *tree*. Recovery, compaction (re-parenting), branching, and future sub-agent sessions all fall out of one structure. The `Agent`'s in-RAM state is rebuilt from this log; a `pi.SessionRepo` adapter projects the active leaf path into agent-core unchanged — see ADR-0003.
- **HTTP transport** — `POST` = 202 admission, `GET` = SSE replay-from-offset then live tail, `?wait=true` blocking convenience — see ADR-0004.
- **`user` vs `signal` inbound kinds** — direct exchanges vs. one participant's activity in a multi-party conversation (a Slack thread, a GitHub issue), in the record schema from day one — see ADR-0005.
- **Pluggable storage** — one narrow adapter contract (Submission + Conversation + Attachment stores), memory and SQLite in-tree, shipped conformance test suite — see ADR-0006.
- **Observability seams** — a typed event `Observer` plus an execution `Interceptor` at every operation boundary; OTel adapter later — see ADR-0008.

## What this is not (v1)

Channel adapters (Slack, Discord, …), a client SDK, React bindings, a dev console, a CLI, workflows, sandboxes/`shell`, and sub-agent `task` delegation are all deferred. The seams for each exist in the design; the packages do not. See `docs/architecture.md` §10.

## Layering

```
resolute-llm-go        LLM providers, streaming        (v0.8.0)
resolute-agent-core-go agent loop, tools, skills       (v0.6.0)
resolute-harness-go    sessions, durability, transport (this repo)
```

## License / lineage

Architecture derived from flue (Apache-2.0, © the flue authors). This is an independent Go implementation, not a source port.
