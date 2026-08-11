# Changelog

## [0.6.0] - 2026-08-09

> **Requires migration (SQLite):** the store schema moves to v3 — six new
> `submissions` columns (parent link, depth, pending-resume, wait bound,
> cancel flag). `sqlite.Open` migrates v1/v2 databases in place on first
> open; the migration is one-way, so builds older than v0.6.0 refuse a
> migrated database. Requires `resolute-agent-core-go` **v0.11.0** (the
> `ToolResult.Suspend` / `Agent.Resume` suspension primitives, AGENT-25).

### Added

- **Durable subagents (HARNESS-15): the `task` tool.** An agent with a
  `Config.Subagents` policy entry gets an injected `task` tool; calling it
  admits one durable child submission of a policy-allowed definition
  (`SubagentPolicy` maps parent name → spawnable names, validated at
  `NewRuntime`) and suspends the parent at the transcript level via
  agent-core's `ToolResult.Suspend`. Spawn is exactly-once via
  `DispatchID = parentSubmissionID + ":" + callID`. `SubagentLimits` bound
  the fan-out: `MaxChildrenPerRun` (default 8; overflow is an immediate
  error result, never a suspension), `MaxDepth` (default 1 — children get
  no task tool), `MaxWait` (default 0 = unbounded), and `OnParentTerminal`
  (v1: `CancelChildren`). `AgentDefinition` gains an optional `Description`
  field, surfaced to parent models in the task tool's schema as routing
  metadata.
- **`waiting` submission state.** A suspended parent CAS-transitions
  running→waiting: the lease is released, no heartbeat runs, and the
  startup reconciler's interrupted-running reclaim skips it. Resume drives
  are new attempts that do not consume the failure-attempt budget.
- **The settlement wake.** A settling child appends the parent's pending
  `tool_outcome` — the child's final answer, its validated structured
  result, or on failure an `isError` outcome carrying the child's error,
  error code, and partial final text — and requeues the parent once its
  last child settles. The wake is the second sanctioned out-of-band record
  author after the dangling-call reconciler, and is replayed on the
  recovery path when the in-reservation wake was missed.
- **Resume drives.** A woken parent is re-driven via agent-core's
  `Agent.Resume`, continuing the turn with the landed outcome in context;
  a resumed turn can suspend again. Observers see `OperationStarted/Ended`
  with operation `"resume"`.
- **Records: `task_spawned` and `ConversationCreatedPayload.ParentRef`.**
  The parent conversation records every spawn (`task_spawned`, authored
  after the `assistant_tool_call` by construction, with park-time and
  wake-time repair from durable child state); the child conversation's
  `conversation_created` carries the upward `ParentRef` link. `Submission`
  gains `ParentSubmissionID`/`ParentCallID`/`Depth`/`PendingResume`, and
  `Dispatch` gains a `Parent *SpawnParent` for admission-time linking.
- **Observer events:** `SubmissionSpawnedEvent` (a task call admitted a
  child), `SubmissionWaitingEvent` (a run parked), `SubmissionResumedEvent`
  (a wake requeued a parent).
- **Reconciler suspension exemption.** The dangling-call reconciler no
  longer error-synthesizes an outcome for a pending `assistant_tool_call`
  that names a spawned task call — that call is a suspension point, not a
  crash remnant.
- **Wait expiry.** `MaxWait`-bounded suspensions are scanned on the claim
  ticker; a lapsed wait lands an error `tool_outcome` for the pending call,
  cancels the children still in flight, and resumes the parent.
- **Orphan cascade.** A parent that settles terminally with live children
  cancels them (`OnParentTerminal: CancelChildren`): queued/waiting
  children settle cancelled directly, running children get
  `CancelRequested` and are cancelled by their owning coordinator.
- **Store contract additions:** `WaitSubmission` (running→waiting CAS),
  `ResumeSubmission` (waiting→queued CAS), `ListChildSubmissions`,
  `ListExpiredWaits`, and `CancelSubmission` (queued/waiting settle
  directly; running get `CancelRequested`). Memory and SQLite implement
  them; the exported conformance suite pins them, so third-party adapters
  must too.

### Changed

- **`ClaimSubmission` consumes `PendingResume`.** A wake sets
  `PendingResume` before requeueing; the flag survives the requeue, the
  claiming row carries it (the drive branches resume vs prompt on it), and
  the stored row clears it on claim — exactly one claimed drive sees the
  resume marker.
