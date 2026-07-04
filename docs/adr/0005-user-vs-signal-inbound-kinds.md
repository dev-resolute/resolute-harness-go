# ADR-0005: `user` and `signal` inbound kinds in the v1 record schema

**Status:** Accepted (2026-07-04)

## Context

flue's inbound `DeliveredMessage` is a discriminated union: `user` (a direct 1:1 exchange with the agent's principal) vs `signal` (one participant's activity — sender identity, structured attributes, optional tag — in a multi-party conversation the agent participates in: a Slack thread, a GitHub issue). Its design note is explicit: forcing channel events into `user` "conflates other participants with the assistant's own user."

v1 ships no channel adapters, so `signal` has no producer yet. But the record schema is the expensive thing to change: adding a second inbound kind later retroactively touches the reducer, the context renderer, the dispatch API, and leaves old conversations predating the kind.

## Decision

Both kinds exist in the v1 canonical record schema and the `Dispatch` API: `user_message{body, attachments}` and `signal{type, body, sender attributes, tag}`. The reducer and LLM-context rendering handle both from day one. agent-core absorbs signals cleanly — its `Message{Role, Type, Body}` already supports custom types.

## Consequences

- No record migration when the first channel adapter (Slack) lands; the adapter's only jobs are webhook verification and identity ⇄ conversation-key mapping, per flue's channel model.
- Cost accepted: the signal → prompt rendering template is designed without a live channel to validate it. Keep it minimal; revisit when Slack lands (architecture.md §12).
