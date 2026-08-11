# ADR-0011: Durable subagents — transcript-level suspension, code-resident spawn policy, agent-core Suspend/Resume

**Status:** Accepted (2026-08-11)

## Context

HARNESS-15 lands the third `Operation` kind, `Task`: a running agent delegates to another registered definition through a harness-injected `task` tool, and the child is a durable submission leased, retried, and settled by the same engine as any dispatch. Three load-bearing decisions needed a record:

1. **Where the parent's suspension lives.** The naive shape keeps a goroutine (or async handle) parked per in-flight delegation — cheap to write, but it holds a worker for the child's whole lifetime, dies with the process, and caps fan-out at goroutine economics rather than policy.
2. **How spawn permission is granted.** flue-style sub-agent config could have drifted toward file- or content-driven grants (manifest files, tool annotations, per-dispatch grants). The harness is a library, not an application — it has no config-file surface anywhere else (ADR-0009).
3. **What agent-core had to grow.** The Go agent loop had no way to end a turn with a tool call pending and continue later: flue's `continue()` semantics had no port, and without one a "blocking tool call whose result arrives days later" is inexpressible.

## Decision

**(a) Transcript-level suspension.** The pending `assistant_tool_call` record *is* the suspension point. A successful `task` execute admits the child submission (exactly-once via `DispatchID = parentSubmissionID + ":" + callID`) and returns `Suspend`: the prompt ends with the call durably dangling, and the parent's submission CAS-transitions `running → waiting` — no lease, no heartbeat, no worker held, excluded from the interrupted-running reclaim. The **wake** is the settlement hand-off: the settling child authors the parent's `tool_outcome` (the second sanctioned out-of-band record author after the dangling-call reconciler) and requeues the parent once it is the last unsettled child. Resume drives don't consume the failure-attempt budget — a parent may legally suspend/resume more than `MaxAttempts` times across a multi-wave fan-out, and `transientBackoff` keys off `AttemptCount`, so the count must only ever reflect real failures. A lapsed bounded wait (`MaxWait`) cancels the live children (flag *and* in-process run cancel), lands an error outcome per outstanding call, and requeues the parent to see the failure as the call's result.

**(b) Spawn policy lives in code.** `Config.Subagents` is a `SubagentPolicy` — a map from parent definition name to the definitions it may spawn; an absent key means no task tool is injected at all. File- or content-driven grants are rejected for a library: the embedding application already expresses composition in code (ADR-0009), and a grant file would grow a parser, a validation surface, and a reload story for what is a map literal. Fan-out bounds ride along as `SubagentLimits`: `MaxChildrenPerRun` (excess calls get an immediate error result, never a suspension), `MaxDepth` (default 1 — children get no task tool, so cycles are impossible by construction), `MaxWait`, and `OnParentTerminal` (v1: `CancelChildren` only).

**(c) agent-core Suspend/Resume as the narrow restoration of `continue()`.** agent-core v0.11.0 (AGENT-25) adds `ToolResult.Suspend` and `Agent.Resume` — deliberately the *minimal* port of flue's `continue()`: the loop may end with a tool call pending (a Suspend-marked result persists nothing), and `Resume` continues from the session transcript only when its tail is a tool result, returning `ErrNothingToResume` otherwise. Narrower than a general continue-anytime on purpose: the tail check turns "resume against a transcript not ending in a tool result" into a hard error, which is exactly the invariant the harness's wake ordering maintains (outcome before requeue) — a miss signals a wake-ordering bug, never a retryable state.

## Consequences

- Suspension survives `kill -9` for free: the suspension point, the child row, and the wake are all durable, and the wait scan's backstop replays a lost wake idempotently.
- `waiting` is a first-class store state, so the SSE tail can (and must) treat it as busy: the promised stream shape — call → `task_spawned` → quiet → outcome → resume deltas — holds on a single connection.
- Policy changes are redeploys, not config edits; an embedding app that needs dynamic grants wraps `Config` construction itself.
- Agent-core's resume contract keeps the harness honest: the re-drive never has to guess whether resuming is legal, and `ErrNothingToResume` fails the submission loudly instead of replaying a bare tool call into the provider.
- v1's `MaxDepth=1` keeps the cancel cascade non-recursive; raising the depth budget later is a limits change, not a redesign.
