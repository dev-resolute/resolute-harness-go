# ADR-0003: Harness owns the event-sourced conversation tree; SessionRepo is a projection

**Status:** Accepted (2026-07-04)

## Context

Two persistence models compete for source-of-truth status:

- flue's: a harness-owned, append-only log of canonical records (deltas, tool outcomes, signals, compaction, submission envelopes), reduced by a pure function into a parent-linked **tree**; pi's in-RAM agent state is rebuilt from it on every claim.
- agent-core's: the `pi.SessionRepo` seam (linear `[]Message` + `BranchSummary`, JSONL/memory impls) — which cannot express a tree, deltas, signals, or submission correlation.

They cannot both be canonical. Dual-writing them drifts. Making `SessionRepo` canonical forecloses branching, delta replay, and flue's recovery model.

Bridge available: agent-core reloads a session's transcript through whatever `SessionRepo` is injected — so the harness can own the log and expose a **projection adapter**.

## Decision

The harness-owned event-sourced tree is the single source of truth. agent-core is configured with a `SessionRepo` adapter over the reduced tree:

- `Load` → active-leaf-path `[]pi.Message`; `LoadBranchSummaries` → reduced compaction nodes (agent-core's `BuildLLMContext` substitution keeps working unchanged).
- `AppendBranchSummary` → writes a `compaction` record (re-parent point).
- `Append` → **no-op**: canonical records are authored from the per-prompt event stream, verified sufficient against v0.6.0 (`ToolCallEndEvent` carries the full `ToolResult`; `MessageEndEvent`/`AgentEndEvent` carry complete messages).

## Consequences

- One write path (event subscription), one read path (reducer + projection). No drift.
- Recovery, compaction re-parenting, branching, and future child sessions fall out of one structure.
- agent-core needs no changes today; if an event ever lacks data a record needs, the fix is a small additive change there (we own it).
- The reducer must be pure and prefix-consistent — property-tested (architecture.md §11).
