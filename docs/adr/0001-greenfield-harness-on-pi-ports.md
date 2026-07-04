# ADR-0001: Greenfield harness on the pi ports, not the Temporal lineage

**Status:** Accepted (2026-07-04)

## Context

Two agent lineages exist locally:

1. `dev-resolute/resolute-llm-go` (v0.8.0) + `dev-resolute/resolute-agent-core-go` (v0.6.0) — Go ports of upstream `@earendil-works/pi-ai` / `pi-agent-core`.
2. The Temporal workspace (`~/playground-ai/resolute`): `resolute-sh/resolute` (Flow framework), `resolute-sh/resolute-agent`, and `resolute-sh/resolute-harness` (Manager, session trees, event serialization, storage contracts) — all built on Temporal.

[flue](https://github.com/withastro/flue) — the TS "Agent Harness Framework" — is built directly on the same upstream pi libraries the Go ports came from. A Go harness following flue's architecture on the pi ports is therefore a like-for-like rebuild of a proven stack, not speculation.

Options considered: greenfield on the pi ports; re-target the existing `resolute-harness` onto agent-core; replace the Temporal stack wholesale; permanent coexistence.

## Decision

Greenfield repo `resolute-harness-go` (module `github.com/dev-resolute/resolute-harness-go`) built on the two pi ports. The Temporal workspace stays untouched as a separate lineage. Its proven designs — Manager/TTL-eviction, tree entries, `storagetest` conformance suites — are ported as *design* inspiration only, never as a code dependency.

## Consequences

- No Temporal-era assumptions (determinism constraints, workflow/activity split, observer plumbing around workflow histories) leak into the new harness.
- Duplication with `resolute-sh/resolute-harness` is accepted short-term; if the new harness matures, the studio UI can be rebuilt on it later (deferred, not decided).
- The rebrand milestone's repo naming (`dev-resolute/*`) is followed.
