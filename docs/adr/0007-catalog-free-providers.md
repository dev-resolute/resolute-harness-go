# ADR-0007: Catalog-free providers; explicit per-agent model metadata

**Status:** Accepted (2026-07-04)

## Context

flue layers a provider registry with catalog hydration (cost, context windows, max tokens) over pi-ai and uses that metadata for compaction thresholds and budgeting. But `resolute-llm-go` ADR-0008/0010 deliberately went catalog-free — per-model behavior, no metadata tables, upstream's catalog/ProviderAuth overhaul declined. Rebuilding a catalog one layer up would reopen the same maintenance treadmill. Meanwhile agent-core already resolves `provider/model` refs by scanning `AgentConfig.Providers` by name — a registry adds no capability.

The harness genuinely needs one piece of metadata: the context window, to drive compaction thresholds (`ShouldCompact(contextTokens, contextWindow, settings)`).

## Decision

No catalog, no registry type. `AgentRuntimeConfig` declares model ref (`provider/model`), `ContextWindow` (required), `MaxTokens`, and token budgets explicitly; providers are constructed in plain Go and passed as `[]llm.LLMProvider`, exactly as agent-core consumes them.

## Consequences

- Consistent with the llm library's own ADRs; no model-metadata maintenance burden.
- Every agent definition states its context window by hand — accepted; the definition is an initializer function, so apps that want a lookup table can build their own.
- A model-discovery/selection UI (e.g. a future studio) must bring its own metadata source; that is deliberately not this layer's problem.
