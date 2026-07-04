# ADR-0002: Hand-rolled durable submission engine (no Temporal, no DBOS)

**Status:** Accepted (2026-07-04)

## Context

The harness needs durable admission, crash recovery, and per-session serialization. Candidates: Temporal, DBOS (`dbos-transact-golang`), or flue's approach — a small hand-rolled engine on plain SQL (idempotent admission, leases + heartbeats, attempt markers, two-phase settlement, startup reconciliation).

Key observation from flue: **the event-sourced conversation log is already the checkpoint.** Agent state is rebuilt by reducing the durable record log on every claim, so the engine only has to guarantee "don't lose or double-run a submission" — a few hundred lines over the same storage adapter the conversation store already needs.

Against Temporal: external server dependency; determinism tax on the agent loop (the prior `resolute-sh` workspace built an entire Flow framework to cope); streaming deltas can't flow through workflow histories, forcing observer side-channels. Its intra-workflow replay buys little when recovery is "re-drive the turn from the log."

Against DBOS: attractive (library-embedded, no server), but it pins the system-of-record to Postgres — breaking the SQLite-default, any-backend-first-class storage contract (ADR-0006) — and splits durable state between DBOS system tables and the conversation store (two sources of truth). Streaming still needs the event-log side-channel, so we'd carry two durability mechanisms.

## Decision

Port flue's engine semantics wholesale, behind a `SubmissionStore` interface that is part of the storage adapter contract and covered by the conformance suite. See `architecture.md` §4 for the seven invariants.

## Consequences

- Zero-infra default: `go run` + SQLite, no server.
- One source of truth: submissions and conversation records live in the same store, same transaction domain.
- We own lease/reconciliation correctness — mitigated by porting known-good semantics (not inventing them) and pinning every invariant in the conformance suite plus crash tests.
- The coordinator is swappable: if cross-machine workers are ever needed, a DBOS- or Temporal-backed coordinator becomes an adapter, not a rewrite.
