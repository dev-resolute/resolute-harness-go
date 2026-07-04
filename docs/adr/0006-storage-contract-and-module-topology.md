# ADR-0006: Three-store contract; single module with SQLite in-tree

**Status:** Accepted (2026-07-04)

## Context

flue's `PersistenceAdapter` bundles five stores (submissions, runs, event streams, conversation, attachments) behind **one contract for every backend** — no SQL-only tiers — verified by shipped contract-test suites, with a bring-your-own-driver seam so the framework bundles no DB driver. Its zero-dep SQLite default works because `node:sqlite` is stdlib; Go has no stdlib SQLite.

v1 has no workflows (no RunStore needed) and keeps telemetry events ephemeral (the durable conversation log already serves reconnecting clients — ADR-0004), so a durable EventStreamStore is also unneeded. Attachments are the judgment call: without `AttachmentRef` in the schema now, adding vision later forces a record migration.

## Decision

- **Contract:** `SubmissionStore` + `ConversationStore` + `AttachmentStore` (minimal digest-keyed blob store), one adapter interface, one tier.
- **Conformance suite:** exported test package (in the spirit of flue's `defineStoreContractTests` and the Temporal-era `storagetest`); every implementation — in-tree and third-party — runs the same suite.
- **Topology:** single repo, single `go.mod` (`github.com/dev-resolute/resolute-harness-go`). Memory and SQLite backends in-tree; SQLite via `modernc.org/sqlite` (pure Go, no cgo) so the batteries-included default is `go run`-clean.
- **Future heavy adapters** (Postgres, Redis) become nested modules with bring-your-own-driver seams (`*sql.DB` in, never a bundled driver) — preserving flue's rule where it actually bites.

## Consequences

- Getting started is one import; `modernc.org/sqlite` lands in every consumer's dependency graph (compile-time cost accepted).
- RunStore/EventStreamStore are added when workflows/telemetry-persistence exist; the adapter interface will grow then — acceptable because in-tree implementations and the conformance suite move in lockstep.
- Mongo-style non-SQL backends remain first-class: the contract is store-shaped, not SQL-shaped.
