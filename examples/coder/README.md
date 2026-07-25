# examples/coder

A coding assistant: one agent wired to the four built-in execution tools
from `resolute-agent-core-go`'s `tools` package — `read`, `write`, `edit`,
`bash` — rooted at a workspace directory, with a stdout observer that
narrates tool activity (start / streamed partial output / done) as it runs.

## Run it

```bash
go run ./examples/coder
```

runs keyless (a deterministic local provider echoes the last message — it
does not call tools) with a SQLite store in `./harness-data` and a workspace
at `./harness-data/workspace`. Set `GEMINI_API_KEY` to run against Gemini
instead, which is what actually drives the tools.

| Env var | Default | What it controls |
|---|---|---|
| `GEMINI_API_KEY` | unset | Present → Gemini provider; absent → keyless local echo provider. |
| `MODEL` | `gemini/gemini-3.1-pro-preview` | Model ref, only consulted when `GEMINI_API_KEY` is set. |
| `DATA_DIR` | `./harness-data` | SQLite store location. |
| `WORKSPACE` | `$DATA_DIR/workspace` | Root directory the read/write/edit/bash tools operate under. Created on first claim. |
| `ADDR` | `:8490` | HTTP listen address. |

```bash
# dispatch asynchronously (202 + ids), or block on the result:
curl -s localhost:8490/agents/coder/demo -d '{"kind":"user","body":"what files are in your workspace?"}'
curl -s 'localhost:8490/agents/coder/demo?wait=true' -d '{"kind":"user","body":"hello there"}'

# watch the conversation live (replay + tail; reconnect with Last-Event-ID),
# including each tool_outcome record's full result content:
curl -N localhost:8490/agents/coder/demo
```

## Live gate: three scenarios (needs `GEMINI_API_KEY`)

Start the server with `GEMINI_API_KEY` set, then place a fixture image for
scenario 2:

```bash
cp <any small png> "${DATA_DIR:-./harness-data}/workspace/logo.png"
```

All three prompts use the blocking `?wait=true` form so the reply comes back
in the curl response; the stdout observer's `[tool ...]` lines appear in the
server's own terminal as each scenario runs.

### 1. write + read + edit

```bash
curl -s 'localhost:8490/agents/coder/demo?wait=true' -d '{
  "kind": "user",
  "body": "Create hello.txt containing '\''hello world'\'', then change '\''world'\'' to '\''resolute'\'' using the edit tool, then read the file back."
}'
```

Expected: the stdout observer prints `[tool write] start` / `done (ok)`,
`[tool edit] start` / `done (ok)`, `[tool read] start` / `done (ok)` in
order; the settled reply's final assistant message contains `hello
resolute`. Verify on disk:

```bash
cat "${DATA_DIR:-./harness-data}/workspace/hello.txt"
# hello resolute
```

### 2. image read

```bash
curl -s 'localhost:8490/agents/coder/demo?wait=true' -d '{
  "kind": "user",
  "body": "Read logo.png and describe what you see."
}'
```

Expected: `[tool read] start` / `done (ok)` on stdout; the `tool_outcome`
record on the SSE stream (`curl -N localhost:8490/agents/coder/demo`) shows
`"content":"Read image file [image/png]"`; the settled reply is a text
description of the image (proof `ImageContent` reached Gemini and the turn
completed).

### 3. streamed bash

```bash
curl -s 'localhost:8490/agents/coder/demo?wait=true' -d '{
  "kind": "user",
  "body": "Run: for i in 1 2 3 4 5; do echo step $i; sleep 1; done"
}'
```

Expected: while the command runs (about 5 seconds), the stdout observer
prints a `[tool bash] step N` line as each `step N` reaches the tool's
output buffer — proof `ToolCallUpdatedEvent` carries partial results
end-to-end, not just the final result — followed by `[tool bash] done
(ok)` and a settled reply that includes all five `step N` lines.
