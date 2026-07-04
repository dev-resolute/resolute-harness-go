# ADR-0004: Transport = 202 admission + SSE replay-from-offset, plus ?wait convenience

**Status:** Accepted (2026-07-04)

## Context

flue decouples request lifetime from work lifetime: `POST` = durable admission returning 202; `GET` = read the durable conversation stream (catch-up from offset + live tail). Multi-reader, resume-after-disconnect, and recovery streaming all reduce to "replay the log from N." A pure-async API is architecturally clean but hostile to curl/scripts/simple bots, and v1 ships no client SDK to paper over that.

## Decision

- `POST /agents/{name}/{id}` → **202** `{submissionId, conversationId}` after idempotent admission.
- `POST …?wait=true` → blocks until `submission_settled`, returns the settled result (convenience path only; same admission underneath).
- `GET /agents/{name}/{id}` → SSE; `Last-Event-ID` = canonical record ID (ULID, orderable) as the replay offset; replays records from there, then tails live appends. The SSE wire format **is** the record schema (`id:` = record ID, `event:` = kind, `data:` = record JSON) — no second serialization layer.
- `POST …/steer` → steers the in-flight run (live passthrough in v1; durability of steer revisited with channels — architecture.md §12).
- Exposed as an `http.Handler`; auth/middleware are the app's concern via standard `net/http` composition.

## Consequences

- One read endpoint covers catch-up, live streaming, and reconnect; clients need no protocol beyond SSE + Last-Event-ID.
- `?wait` re-couples one request to work lifetime by choice, not by architecture — it still survives coordinator crashes because it waits on the settled record, not on an in-process future.
- WebSocket deferred; steering via plain POST removes the bidirectional need in v1.
