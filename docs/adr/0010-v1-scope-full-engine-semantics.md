# ADR-0010: v1 scope — core runtime + HTTP; full engine semantics despite single-process

**Status:** Accepted (2026-07-04)

## Context

flue's full surface is runtime + CLI + SDK + React + dev-console + ~25 adapters. Building all of it up front dilutes the core. Separately, a tempting Go shortcut for a single-process v1 is to skip leases ("a mutex and per-session goroutines suffice") — but leases/attempts/reconciliation are the engine's *core invariants*, the most expensive place to retrofit, and they are what make `kill -9` recovery and future multi-node work.

## Decision

**v1 scope:** the harness library only — Harness/Session hierarchy with `Prompt` (incl. schema-validated structured results) + `Compact` + Steer/FollowUp passthrough; conversation records + reducer + projection adapter; the durable submission engine with **full flue semantics from day one** (idempotent admission, leases + heartbeats, attempt markers, two-phase settlement, startup reconciliation, budgets-from-history); memory + SQLite stores + conformance suite; HTTP transport per ADR-0004; Observer/Interceptor seams per ADR-0008; one runnable example app.

**Deferred** (rough order): `task`/subagents; Slack channel adapter + Go client SDK; OTel adapter; Postgres store; sandbox + `shell`; workflows (adds RunStore); CLI + dev console; React bindings; studio rebuild.

## Consequences

- Crash-restart recovery works in v1 and is pinned by crash tests (kill between every pair of engine phases).
- Multi-node later is a Postgres store adapter, not an engine change.
- Single-process SQLite pays a small awkwardness tax (leases guarding against your own restarts) — accepted as the price of never retrofitting invariants.
- Every deferred item has a designed seam (channel model, store contract growth, tree branches for task) so deferral is scheduling, not redesign.
