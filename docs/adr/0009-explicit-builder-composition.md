# ADR-0009: Explicit Runtime builder with initializer-function agent definitions

**Status:** Accepted (2026-07-04)

## Context

flue's DX rests on file discovery + Vite codegen (`agents/<name>.ts` default-exporting `defineAgent(...)`; the CLI generates the server entry). Go has no bundler step and v1 ships no CLI. Candidate shapes: explicit builder; static config structs; `init()`-side-effect registration via blank imports.

flue's `defineAgent` takes a *closure*, not a value — `initialize({id, env})` returns the runtime config — which is what makes per-instance dynamic setup (tenant prompts, per-user tools) first-class. Static structs lose that; `init()` registration is hidden-global, import-order-coupled magic the project guidelines would reject.

## Decision

Explicit composition, no discovery:

```go
rt, err := harness.NewRuntime(harness.Config{
    Agents: map[string]harness.AgentDefinition{ "support": {Initialize: …} },
    Store:  sqlite.Open(dir),
    Observers: …, Interceptors: …,
})
```

`AgentDefinition.Initialize(ctx, InstanceID, Env) (AgentRuntimeConfig, error)` is flue's defineAgent closure minus the discovery magic. Runtime exposes `Start`/`Close`, `Handler() http.Handler`, and in-process `Dispatch`.

## Consequences

- Idiomatic, testable, zero magic; a future CLI layers scaffolding on top without changing this API.
- Registration verbosity accepted — one map literal in `main.go`.
- `Env` is the injection seam for secrets/config lookup, keeping definitions pure enough to unit-test.
