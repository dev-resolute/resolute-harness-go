# ADR-0008: Observability = Observer + Interceptor seams in v1; OTel adapter later

**Status:** Accepted (2026-07-04)

## Context

flue's observability is a duality: `observe(subscriber)` supplies event *data*; an `ExecutionInterceptor` (onion middleware around workflow/agent/model/tool operations) establishes ambient *trace context*; `instrument()` installs both, and the OTel package proves the pair scales to full GenAI semconv + W3C traceparent. In Go the interceptor is cheaper than in TS — `context.Context` already flows everywhere — but interceptor **call sites** (around submission attempts, operations, model turns, tool executions) are invasive to retrofit, while adapters over the seams are purely additive.

## Decision

v1 ships both seams and no adapters:

- `Observer func(HarnessEvent)` — synchronous, read-only subscription to a typed event union (submission/operation/turn/delta/tool/compaction lifecycle) carrying the same correlation IDs as record envelopes. Observer failures are logged, never fatal.
- `Interceptor func(ctx context.Context, op OpInfo, next func(context.Context) error) error` — wrapped at every operation boundary from day one.

The OTel GenAI-semconv adapter is a later, separate package over these seams. agent-core's `Hooks.BeforeProviderRequest/AfterProviderResponse` remain the wire-level seam; the harness does not duplicate them.

## Consequences

- No blind-flying: logging/metrics observers work immediately in v1 (the example app installs one).
- The expensive part (call-site coverage) is paid once, up front; OTel arrives without touching the engine.
- Two extension vocabularies must stay distinct in docs: harness Observer/Interceptor vs agent-core Hooks (see CONTEXT.md).
