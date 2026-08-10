package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
)

// Engine timing defaults; Config overrides them. The delta-flush defaults
// are the measured pick for "few records, low latency" on the SQLite store
// (architecture.md §12).
const (
	defaultClaimInterval      = 250 * time.Millisecond
	defaultLeaseDuration      = 30 * time.Second
	defaultDeltaFlushBytes    = 1024
	defaultDeltaFlushInterval = 200 * time.Millisecond
)

// errLeaseLost cancels a run whose heartbeat discovered another attempt owns
// the submission.
var errLeaseLost = errors.New("lease lost to another attempt")

// errDeadlineHalted stops a run whose durability timeout passed mid-flight
// (cooperative halt at a turn boundary).
var errDeadlineHalted = errors.New("durability timeout reached mid-run")

// errParentCancelled cancels a run whose parent settled terminally — the
// orphan cascade (HARNESS-15) interrupts the orphaned child's attempt.
var errParentCancelled = errors.New("parent settled; orphaned run cancelled")

// danglingToolCallMessage is the harness-owned synthesized tool_outcome
// content for a tool call recovered with no result: the process crashed
// between the durable assistant_tool_call record and its tool_outcome, so
// the outcome is genuinely unknown. Byte-exact (HARNESS-14; harness half of
// upstream #6285) — this string is not an upstream port.
const danglingToolCallMessage = "Tool call was interrupted before a result was recorded (the run was recovered). Re-issue the tool call if it is still needed."

// waitExpiredMessage is the harness-owned error tool_outcome content for a
// spawned task call whose parent's bounded wait lapsed before the child
// settled (HARNESS-15 wait expiry).
const waitExpiredMessage = "task wait expired"

// overflowCompactRetries bounds the in-attempt overflow ladder: each
// overflow triggers one compact-and-retry, at most this many times.
const overflowCompactRetries = 2

// transientRunError marks a model error worth a budgeted backoff retry (a
// fresh attempt) instead of terminal failure.
type transientRunError struct{ err error }

func (e *transientRunError) Error() string { return "transient model error: " + e.err.Error() }
func (e *transientRunError) Unwrap() error { return e.err }

// coordinator runs the claim loop: it reconciles interrupted work, leases
// runnable submissions, and drives their sessions. One per Runtime process
// (v1; multi-node is a store adapter concern, ADR-0010).
type coordinator struct {
	rt      *Runtime
	ownerID string

	mu     sync.Mutex
	active map[string]bool // session keys with a run in flight in this process

	// runs maps a live attempt's submission ID to its run-context cancel,
	// so the orphan cascade can interrupt a running child of this process
	// (v1 single-process; multi-node fencing is deferred, ADR-0010).
	runsMu sync.Mutex
	runs   map[string]context.CancelCauseFunc
}

func newCoordinator(rt *Runtime) *coordinator {
	return &coordinator{
		rt:      rt,
		ownerID: newULID(),
		active:  make(map[string]bool),
		runs:    make(map[string]context.CancelCauseFunc),
	}
}

// loop reconciles once at startup, then claims and reclaims until ctx is
// cancelled. It wakes on admission nudges and on a steady tick.
func (c *coordinator) loop(ctx context.Context) {
	c.reconcile(ctx)
	// One startup pass at reconcile cadence: a wait that lapsed while the
	// process was down must cancel its still-queued children before the
	// first claimRunnable can pick them up. Inside the loop the scans are
	// gated to ticker iterations (see below).
	c.expireWaits(ctx)
	ticker := time.NewTicker(c.rt.claimInterval)
	defer ticker.Stop()
	for {
		c.reclaimExpired(ctx)
		c.claimRunnable(ctx)
		select {
		case <-ctx.Done():
			return
		case <-c.rt.wake:
		case <-ticker.C:
			// The wait scans are O(waiting-parents × log-size) per run, so
			// they fire on ticker cadence only — on every nudge they would
			// serialize against run goroutines on single-connection stores
			// (sqlite SetMaxOpenConns(1)). reclaimExpired stays on nudges:
			// its queries are O(1).
			c.expireWaits(ctx)
		}
	}
}

// reconcile hands interrupted work to fresh attempts at startup: submissions
// stuck terminalizing are finalized (their outcome record either exists or
// is durably unknowable), and expired running leases are reclaimed by the
// regular loop.
func (c *coordinator) reconcile(ctx context.Context) {
	stuck, err := c.rt.store.ListByStatus(ctx, StatusTerminalizing)
	if err != nil {
		if ctx.Err() == nil {
			c.rt.logger.Error("reconcile: list terminalizing", "error", err)
		}
		return
	}
	for _, sub := range stuck {
		if err := c.finalizeInterrupted(ctx, sub); err != nil {
			c.rt.logger.Error("reconcile terminalizing submission", "submission", sub.ID, "error", err)
		}
	}
}

// finalizeInterrupted completes settlement for a submission that crashed
// between the two phases. If the terminal record landed before the crash it
// is honored; otherwise the outcome is unknowable and the submission settles
// failed with the indeterminate code.
func (c *coordinator) finalizeInterrupted(ctx context.Context, sub Submission) error {
	effective, err := c.appendSettledRecordOnce(ctx, sub, SettledPayload{
		Status:    SettledFailed,
		Error:     "process crashed during settlement; run outcome unknown",
		ErrorCode: SettledErrIndeterminate,
	})
	if err != nil {
		return err
	}
	if err := c.rt.store.FinalizeSettlement(ctx, sub.ID); err != nil {
		return fmt.Errorf("finalize settlement: %w", err)
	}
	if sub.ParentSubmissionID != "" {
		// Re-run the wake after finalization, as settle() does: a sibling
		// settling concurrently may have seen this submission still
		// terminalizing and skipped the requeue. Idempotent — the outcome
		// record exists by now, and a duplicate ResumeSubmission loses its
		// CAS.
		if err := c.wakeParent(ctx, sub, effective); err != nil {
			return fmt.Errorf("wake parent after finalize: %w", err)
		}
	}
	c.rt.notifySettled()
	return nil
}

// reclaimExpired releases running submissions whose lease expired — a
// crashed or wedged owner — so the normal claim path re-attempts them.
func (c *coordinator) reclaimExpired(ctx context.Context) {
	expired, err := c.rt.store.ListExpiredLeases(ctx, time.Now())
	if err != nil {
		if ctx.Err() == nil {
			c.rt.logger.Error("list expired leases", "error", err)
		}
		return
	}
	for _, sub := range expired {
		key := sub.SessionKey.String()
		c.mu.Lock()
		ownLive := c.active[key]
		c.mu.Unlock()
		if ownLive {
			// Our own run holds the session; its heartbeat owns the lease
			// question.
			continue
		}
		err := c.rt.store.ReleaseSubmission(ctx, SubmissionRelease{SubmissionID: sub.ID, AttemptID: sub.AttemptID})
		if err != nil && !errors.Is(err, ErrClaimLost) {
			c.rt.logger.Error("release expired lease", "submission", sub.ID, "error", err)
			continue
		}
		if err == nil {
			c.rt.logger.Info("reclaimed expired lease", "submission", sub.ID, "deadOwner", sub.OwnerID)
		}
	}
}

// expireWaits is the wait scan (HARNESS-15), gated to ticker iterations in
// the claim loop — never to nudges — because its per-parent log reads are
// O(waiting-parents × log-size). Two passes over parked parents:
//
//   - Expiry: a waiting submission whose WaitUntil lapsed can no longer pay
//     off — its live children are cancelled, an error tool_outcome lands
//     per outstanding spawned call, and it is requeued so the re-drive sees
//     the failure.
//   - Backstop: a waiting submission whose children ALL settled but which
//     still lacks an outcome for a spawned call lost its wake to a crash
//     window — the wake is re-run (wakeParent is idempotent; it no-ops
//     when the outcome exists).
func (c *coordinator) expireWaits(ctx context.Context) {
	expired, err := c.rt.store.ListExpiredWaits(ctx, time.Now())
	if err != nil {
		if ctx.Err() == nil {
			c.rt.logger.Error("list expired waits", "error", err)
		}
		return
	}
	for _, sub := range expired {
		if err := c.expireWait(ctx, sub); err != nil {
			c.rt.logger.Error("expire wait", "submission", sub.ID, "error", err)
		}
	}

	waiting, err := c.rt.store.ListByStatus(ctx, StatusWaiting)
	if err != nil {
		if ctx.Err() == nil {
			c.rt.logger.Error("list waiting submissions", "error", err)
		}
		return
	}
	for _, sub := range waiting {
		if err := c.backstopWake(ctx, sub); err != nil {
			c.rt.logger.Error("backstop wake", "submission", sub.ID, "error", err)
		}
	}
}

// expireWait ends one lapsed wait: the children still in flight are
// cancelled, each spawned call still lacking an outcome gets the
// wait-expired error outcome, and the parent is requeued so its re-drive
// sees the failure as the call's result. Cancelling a running child only
// sets CancelRequested (the orphan cascade, Task 11, owns running-child
// termination); either way the parent's outcome no longer waits on it.
func (c *coordinator) expireWait(ctx context.Context, sub Submission) error {
	children, err := c.rt.store.ListChildSubmissions(ctx, sub.ID)
	if err != nil {
		return fmt.Errorf("list children for wait expiry: %w", err)
	}
	for _, ch := range children {
		if ch.Status == StatusSettled {
			continue
		}
		if _, err := c.rt.store.CancelSubmission(ctx, ch.ID, "parent wait expired"); err != nil && !errors.Is(err, ErrClaimLost) {
			return fmt.Errorf("cancel child %s for wait expiry: %w", ch.ID, err)
		}
	}

	pending, err := c.pendingSpawnedCalls(ctx, sub)
	if err != nil {
		return err
	}
	if len(pending) > 0 {
		// Re-check outcome existence immediately before appending: the
		// natural wake (wakeParent) check-then-appends an outcome for the
		// same CallID from another goroutine, so one can have landed since
		// pendingSpawnedCalls read the log. This narrows the
		// duplicate-outcome race window but does not close it — the
		// eventual fix is a store-level guarded append (reject a second
		// tool_outcome per CallID).
		stillPending, err := c.pendingSpawnedCalls(ctx, sub)
		if err != nil {
			return err
		}
		open := make(map[string]bool, len(stillPending))
		for _, callID := range stillPending {
			open[callID] = true
		}
		outcomes := make([]Record, 0, len(pending))
		for _, callID := range pending {
			if !open[callID] {
				continue // the wake landed this outcome concurrently
			}
			outcomes = append(outcomes, Record{
				RecordEnvelope: RecordEnvelope{
					ID:             newULID(),
					Kind:           KindToolOutcome,
					ConversationID: sub.ConversationID,
					Session:        sub.SessionKey.Session,
					SubmissionID:   sub.ID,
					Time:           time.Now(),
				},
				Payload: mustPayload(&ToolOutcomePayload{
					CallID:   callID,
					ToolName: "task",
					IsError:  true,
					Content:  waitExpiredMessage,
				}),
			})
		}
		if len(outcomes) > 0 {
			if err := c.rt.store.AppendRecords(ctx, sub.ConversationID, outcomes); err != nil {
				return fmt.Errorf("append wait-expiry outcomes: %w", err)
			}
			c.rt.notifyAppend()
		}
	}

	if err := c.rt.store.ResumeSubmission(ctx, sub.ID); err != nil {
		if errors.Is(err, ErrClaimLost) {
			return nil // a wake already requeued it
		}
		return fmt.Errorf("requeue expired parent %s: %w", sub.ID, err)
	}
	// Nudge the claim loop without blocking (same as the wake).
	select {
	case c.rt.wake <- struct{}{}:
	default:
	}
	return nil
}

// backstopWake re-runs the settlement wake for a waiting submission whose
// children all settled — covering two lost-wake windows: the outcome for a
// spawned call never landed (crash mid-wake), or the outcome landed but the
// parent's ResumeSubmission was lost (the wake's requeue CAS repairs it).
// wakeParent is idempotent — an existing outcome wins, a duplicate requeue
// loses its CAS — so a submission whose wake fully landed costs only the
// reads.
func (c *coordinator) backstopWake(ctx context.Context, sub Submission) error {
	children, err := c.rt.store.ListChildSubmissions(ctx, sub.ID)
	if err != nil {
		return fmt.Errorf("list children for wake backstop: %w", err)
	}
	if len(children) == 0 {
		return nil
	}
	for _, ch := range children {
		if ch.Status != StatusSettled {
			return nil // still in flight — the wake fires at settlement
		}
	}
	pending, err := c.pendingSpawnedCalls(ctx, sub)
	if err != nil {
		return err
	}
	for _, ch := range children {
		// Wake unconditionally once all children are settled — the
		// ResumeSubmission CAS is what repairs a lost requeue even when the
		// outcome already landed. The payload (and its log read) is only
		// needed while the outcome work for this call is still pending.
		var payload SettledPayload
		if slices.Contains(pending, ch.ParentCallID) {
			var err error
			payload, err = c.settledPayload(ctx, ch)
			if err != nil {
				return err
			}
		}
		if err := c.wakeParent(ctx, ch, payload); err != nil {
			return fmt.Errorf("backstop wake for child %s: %w", ch.ID, err)
		}
	}
	return nil
}

// pendingSpawnedCalls returns, in spawn-record order, the call IDs of the
// conversation's task_spawned records that have no tool_outcome yet.
func (c *coordinator) pendingSpawnedCalls(ctx context.Context, sub Submission) ([]string, error) {
	recs, err := c.rt.store.ReadRecords(ctx, sub.ConversationID, "")
	if err != nil {
		return nil, fmt.Errorf("read records for wait scan: %w", err)
	}
	answered := make(map[string]bool)
	for _, rec := range recs {
		if rec.Kind != KindToolOutcome {
			continue
		}
		var p ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			return nil, fmt.Errorf("decode tool_outcome for wait scan: %w", err)
		}
		answered[p.CallID] = true
	}
	var pending []string
	seen := make(map[string]bool)
	for _, rec := range recs {
		if rec.Kind != KindTaskSpawned {
			continue
		}
		var sp TaskSpawnedPayload
		if err := rec.DecodePayload(&sp); err != nil {
			return nil, fmt.Errorf("decode task_spawned for wait scan: %w", err)
		}
		if answered[sp.CallID] || seen[sp.CallID] {
			continue
		}
		seen[sp.CallID] = true
		pending = append(pending, sp.CallID)
	}
	return pending, nil
}

// settledPayload reads back the submission_settled record of an
// already-settled submission — the payload a backstop wake replays.
func (c *coordinator) settledPayload(ctx context.Context, sub Submission) (SettledPayload, error) {
	recs, err := c.rt.store.ReadRecords(ctx, sub.ConversationID, "")
	if err != nil {
		return SettledPayload{}, fmt.Errorf("read records for settled payload: %w", err)
	}
	for _, rec := range recs {
		if rec.Kind != KindSubmissionSettled || rec.SubmissionID != sub.ID {
			continue
		}
		var p SettledPayload
		if err := rec.DecodePayload(&p); err != nil {
			return SettledPayload{}, fmt.Errorf("decode settled record: %w", err)
		}
		return p, nil
	}
	return SettledPayload{}, fmt.Errorf("settled submission %s has no settled record", sub.ID)
}

// claimRunnable claims every runnable submission whose session is not
// already active in this process and starts a run goroutine per claim.
func (c *coordinator) claimRunnable(ctx context.Context) {
	subs, err := c.rt.store.ListRunnable(ctx)
	if err != nil {
		if ctx.Err() == nil {
			c.rt.logger.Error("list runnable submissions", "error", err)
		}
		return
	}
	for _, sub := range subs {
		key := sub.SessionKey.String()
		c.mu.Lock()
		if c.active[key] {
			c.mu.Unlock()
			continue
		}
		c.active[key] = true
		c.mu.Unlock()

		claimed, err := c.rt.store.ClaimSubmission(ctx, SubmissionClaim{
			SubmissionID:   sub.ID,
			AttemptID:      newULID(),
			OwnerID:        c.ownerID,
			LeaseExpiresAt: time.Now().Add(c.rt.leaseDuration),
		})
		if err != nil {
			c.release(key)
			if ctx.Err() == nil && !errors.Is(err, ErrClaimLost) {
				c.rt.logger.Error("claim submission", "submission", sub.ID, "error", err)
			}
			continue
		}

		c.rt.observe(SubmissionClaimedEvent{
			Correlation:  claimed.correlation(),
			OwnerID:      c.ownerID,
			AttemptCount: claimed.AttemptCount,
		})

		// The attempt marker lands before any work so reconciliation can
		// distinguish "started then died" from "never started".
		if err := c.rt.store.StartAttempt(ctx, Attempt{
			ID:           claimed.AttemptID,
			SubmissionID: claimed.ID,
			OwnerID:      c.ownerID,
			StartedAt:    time.Now(),
		}); err != nil {
			c.release(key)
			if ctx.Err() == nil {
				c.rt.logger.Error("start attempt", "submission", claimed.ID, "error", err)
			}
			continue
		}
		c.rt.observe(AttemptStartedEvent{Correlation: claimed.correlation()})

		c.rt.running.Add(1)
		go func() {
			defer c.rt.running.Done()
			defer c.release(key)
			c.runSubmission(ctx, claimed)
		}()
	}
}

func (c *coordinator) release(sessionKey string) {
	c.mu.Lock()
	delete(c.active, sessionKey)
	c.mu.Unlock()
}

// runSubmission drives one claimed submission through one attempt:
// budgets, heartbeat, agent run, settlement or release.
func (c *coordinator) runSubmission(ctx context.Context, sub Submission) {
	logger := c.rt.logger.With("submission", sub.ID, "session", sub.SessionKey.String(), "attempt", sub.AttemptID)

	def := c.rt.agents[sub.SessionKey.Agent]
	cfg, err := def.Initialize(ctx, sub.SessionKey.Instance, c.rt.env)
	if err == nil {
		err = cfg.validate()
	}
	if err != nil {
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status: SettledFailed, Error: err.Error(), ErrorCode: SettledErrRunFailed,
		}, logger)
		return
	}

	// Durability budgets are evaluated from durable state on every attempt,
	// so a crash-restart loop exhausts them instead of retrying forever.
	maxAttempts := cfg.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = DefaultMaxAttempts
	}
	timeout := cfg.SubmissionTimeout
	if timeout <= 0 {
		timeout = DefaultSubmissionTimeout
	}
	if sub.AttemptCount > maxAttempts {
		budgetErr := fmt.Sprintf("attempt budget exhausted: attempt %d exceeds max %d", sub.AttemptCount, maxAttempts)
		if sub.LastError != "" {
			budgetErr += "; last error: " + sub.LastError
		}
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status:    SettledFailed,
			Error:     budgetErr,
			ErrorCode: SettledErrAttemptBudget,
		}, logger)
		return
	}
	deadline := sub.CreatedAt.Add(timeout)
	if time.Now().After(deadline) {
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status:    SettledFailed,
			Error:     fmt.Sprintf("durability timeout exceeded: admitted %s ago (budget %s)", time.Since(sub.CreatedAt).Round(time.Second), timeout),
			ErrorCode: SettledErrTimeout,
		}, logger)
		return
	}

	runCtx, cancelRun := context.WithCancelCause(ctx)
	defer cancelRun(nil)
	// Register the run's cancel so the orphan cascade can interrupt it when
	// this submission's parent settles; deleted on exit.
	c.runsMu.Lock()
	c.runs[sub.ID] = cancelRun
	c.runsMu.Unlock()
	defer func() {
		c.runsMu.Lock()
		delete(c.runs, sub.ID)
		c.runsMu.Unlock()
	}()
	heartbeatDone := c.startHeartbeat(runCtx, sub, cancelRun, logger)

	var result json.RawMessage
	var suspended bool
	runErr := c.rt.intercept(runCtx, OpInfo{Kind: OpAttempt, Correlation: sub.correlation()}, func(cctx context.Context) error {
		var derr error
		result, suspended, derr = c.driveAttempt(cctx, sub, cfg, deadline)
		return derr
	})
	cancelRun(nil)
	<-heartbeatDone

	var invalid *resultInvalidError
	var transient *transientRunError
	switch {
	case errors.Is(context.Cause(runCtx), errLeaseLost):
		// Another attempt owns the submission now; ours must not settle or
		// release.
		logger.Warn("lease lost mid-run; abandoning attempt")
	case errors.Is(context.Cause(runCtx), errParentCancelled):
		// The orphan cascade cancelled the run (the parent settled
		// terminally): settle cancelled — distinct from the shutdown arm
		// below, which RELEASES for retry; an orphan has nothing to retry
		// for.
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status:    SettledFailed,
			Error:     "cancelled: parent settled",
			ErrorCode: SettledErrCancelled,
		}, logger)
	case runErr != nil && ctx.Err() != nil:
		// Shutdown interrupted the attempt: release the claim so a fresh
		// Runtime (or this store's next owner) re-attempts immediately.
		releaseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := c.rt.store.ReleaseSubmission(releaseCtx, SubmissionRelease{SubmissionID: sub.ID, AttemptID: sub.AttemptID}); err != nil && !errors.Is(err, ErrClaimLost) {
			logger.Error("release on shutdown", "error", err)
		} else {
			logger.Info("released in-flight submission on shutdown")
		}
	case errors.Is(runErr, errDeadlineHalted):
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status:    SettledFailed,
			Error:     fmt.Sprintf("durability timeout exceeded mid-run (budget %s)", timeout),
			ErrorCode: SettledErrTimeout,
		}, logger)
	case errors.As(runErr, &transient):
		// Budgeted backoff retry: sleep, release, and let the claim path
		// re-attempt. The consecutive-failure count is the durable
		// AttemptCount, so a crash mid-backoff does not reset the budget.
		backoff := transientBackoff(c.rt.claimInterval, sub.AttemptCount)
		logger.Warn("transient model error; backing off before re-attempt", "error", runErr, "backoff", backoff, "attempt", sub.AttemptCount)
		c.rt.observe(RecoveryEvent{Correlation: sub.correlation(), Decision: "transient_backoff", Detail: runErr.Error()})
		select {
		case <-time.After(backoff):
		case <-ctx.Done():
		}
		releaseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := c.rt.store.ReleaseSubmission(releaseCtx, SubmissionRelease{
			SubmissionID: sub.ID,
			AttemptID:    sub.AttemptID,
			LastError:    runErr.Error(),
		}); err != nil && !errors.Is(err, ErrClaimLost) {
			logger.Error("release after transient failure", "error", err)
		}
	case errors.As(runErr, &invalid):
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status: SettledFailed, Error: invalid.Error(), ErrorCode: SettledErrResultInvalid,
		}, logger)
	case runErr != nil:
		logger.Error("attempt failed", "error", runErr)
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status: SettledFailed, Error: runErr.Error(), ErrorCode: SettledErrRunFailed,
		}, logger)
	case suspended:
		// The run suspended on a task call (HARNESS-15): park the
		// submission in waiting — the lease is released and the worker
		// freed — until a child settlement wakes it (Task 9).
		if err := c.rt.store.WaitSubmission(ctx, SubmissionWait{
			SubmissionID: sub.ID,
			AttemptID:    sub.AttemptID,
			WaitUntil:    c.rt.waitDeadline(),
		}); err != nil {
			if errors.Is(err, ErrClaimLost) {
				// Lost the race with a reclaimer; abandon the attempt.
				logger.Warn("wait transition lost; abandoning attempt", "error", err)
				return
			}
			// The submission stays running; it recovers via lease expiry.
			logger.Error("wait transition failed; submission stays running and recovers via lease expiry", "error", err)
			return
		}
		// Task 12: emit SubmissionWaitingEvent
	default:
		c.settleAndNotify(ctx, sub, SettledPayload{Status: SettledSucceeded, Result: result}, logger)
	}
}

// startHeartbeat renews the lease at a third of its duration until the run
// context ends. Discovering a lost lease cancels the run with errLeaseLost.
func (c *coordinator) startHeartbeat(runCtx context.Context, sub Submission, cancelRun context.CancelCauseFunc, logger *slog.Logger) <-chan struct{} {
	done := make(chan struct{})
	interval := c.rt.leaseDuration / 3
	if interval <= 0 {
		interval = defaultLeaseDuration / 3
	}
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-runCtx.Done():
				return
			case <-ticker.C:
			}
			err := c.rt.store.RenewLease(runCtx, LeaseRenewal{
				SubmissionID:   sub.ID,
				AttemptID:      sub.AttemptID,
				LeaseExpiresAt: time.Now().Add(c.rt.leaseDuration),
			})
			switch {
			case err == nil:
			case errors.Is(err, ErrClaimLost):
				cancelRun(errLeaseLost)
				return
			case runCtx.Err() != nil:
				return
			default:
				logger.Error("renew lease", "error", err)
			}
		}
	}()
	return done
}

func (c *coordinator) settleAndNotify(ctx context.Context, sub Submission, payload SettledPayload, logger *slog.Logger) {
	if err := c.settle(ctx, sub, payload); err != nil {
		logger.Error("settle submission", "error", err)
		return
	}
	c.rt.observe(SubmissionSettledEvent{Correlation: sub.correlation(), Payload: payload})
	c.rt.notifySettled()
	if c.rt.limits.OnParentTerminal == CancelChildren {
		c.cancelChildren(ctx, sub)
	}
}

// cancelChildren is the orphan cascade (HARNESS-15): a terminally settled
// parent's live children are cancelled. A running child is flagged via
// CancelSubmission and its in-process run context cancelled (the registry
// is v1-single-process, ADR-0010); its own post-drive switch settles it
// cancelled_by_parent. A queued or waiting child will never start an
// attempt: CancelSubmission settles the row outright, so the settled
// record lands here directly.
//
// No re-entry guard is needed at v1 MaxDepth=1: a settling child has no
// children of its own, so the cascade cannot recurse back into itself.
func (c *coordinator) cancelChildren(ctx context.Context, parent Submission) {
	children, err := c.rt.store.ListChildSubmissions(ctx, parent.ID)
	if err != nil {
		c.rt.logger.Warn("list children for cascade", "submission", parent.ID, "error", err)
		return
	}
	for _, ch := range children {
		if ch.Status == StatusSettled {
			continue
		}
		wasRunning, err := c.rt.store.CancelSubmission(ctx, ch.ID, "parent "+parent.ID+" settled")
		if err != nil {
			// ErrClaimLost included: the child went terminal between the
			// list and the cancel; its own settle owns the record.
			c.rt.logger.Warn("cancel child", "child", ch.ID, "error", err)
			continue
		}
		if wasRunning {
			c.runsMu.Lock()
			cancel, ok := c.runs[ch.ID]
			c.runsMu.Unlock()
			if ok {
				cancel(errParentCancelled)
			}
			continue
		}
		// Queued/waiting: no attempt will run — CancelSubmission settled
		// the row already, so land the settled record directly, bypassing
		// settle's reservation (which expects a running row). The wake
		// inside appendSettledRecordOnce is a guarded no-op: the parent is
		// settled by now.
		if _, err := c.appendSettledRecordOnce(ctx, ch, SettledPayload{
			Status:    SettledFailed,
			Error:     "cancelled: parent settled",
			ErrorCode: SettledErrCancelled,
		}); err != nil {
			c.rt.logger.Error("settle cancelled child", "child", ch.ID, "error", err)
			continue
		}
		c.rt.notifySettled()
	}
}

// driveAttempt runs the agent for one attempt, returning the validated
// structured result (nil when none was requested), whether the run
// suspended on a task call, and the run error.
func (c *coordinator) driveAttempt(ctx context.Context, sub Submission, cfg AgentRuntimeConfig, deadline time.Time) (json.RawMessage, bool, error) {
	conv, err := c.rt.store.GetConversation(ctx, sub.SessionKey)
	if err != nil {
		return nil, false, fmt.Errorf("resolve conversation for %s: %w", sub.SessionKey, err)
	}
	run := &submissionRun{
		rt:       c.rt,
		sub:      sub,
		conv:     conv,
		cfg:      cfg,
		deadline: deadline,
	}
	if err := run.drive(ctx); err != nil {
		return nil, false, err
	}
	return run.result, run.suspended, nil
}

// settle runs two-phase settlement: reserve the terminal transition, land
// the submission_settled record exactly once, then finalize. A crash between
// the phases is resolved by startup reconciliation.
func (c *coordinator) settle(ctx context.Context, sub Submission, payload SettledPayload) error {
	// Use a fresh context bound to the store, not the (possibly cancelled)
	// run context: settlement must land once the outcome is known.
	if ctx.Err() != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
	}
	if err := c.rt.store.ReserveSettlement(ctx, sub.ID, sub.AttemptID); err != nil {
		return fmt.Errorf("reserve settlement: %w", err)
	}
	effective, err := c.appendSettledRecordOnce(ctx, sub, payload)
	if err != nil {
		return err
	}
	if err := c.rt.store.FinalizeSettlement(ctx, sub.ID); err != nil {
		return fmt.Errorf("finalize settlement: %w", err)
	}
	if sub.ParentSubmissionID != "" {
		// Re-run the wake after finalization: a sibling settling
		// concurrently may have seen this submission still terminalizing
		// and skipped the requeue. Whichever child finalizes last sees
		// every sibling settled, so exactly one post-finalize wake
		// requeues the parent. Idempotent — the outcome record exists by
		// now, and a duplicate ResumeSubmission loses its CAS.
		if err := c.wakeParent(ctx, sub, effective); err != nil {
			return fmt.Errorf("wake parent after finalize: %w", err)
		}
	}
	return nil
}

// appendSettledRecordOnce appends the submission_settled record unless one
// already exists for the submission — the idempotency half of two-phase
// settlement. When the settling submission has a parent link, the wake
// (HARNESS-15 settlement hand-off) runs inside the same reservation,
// honoring an existing settled record's payload over the caller's (a
// crash-recovery replay passes the indeterminate failure even when the
// run's real outcome landed before the crash). It returns the effective
// payload so the caller's post-finalize wake replays with the same one.
func (c *coordinator) appendSettledRecordOnce(ctx context.Context, sub Submission, payload SettledPayload) (SettledPayload, error) {
	recs, err := c.rt.store.ReadRecords(ctx, sub.ConversationID, "")
	if err != nil {
		return payload, fmt.Errorf("read records before settle: %w", err)
	}
	effective := payload
	found := false
	for _, rec := range recs {
		if rec.Kind == KindSubmissionSettled && rec.SubmissionID == sub.ID {
			found = true
			if err := rec.DecodePayload(&effective); err != nil {
				return payload, fmt.Errorf("decode existing settled record: %w", err)
			}
			break
		}
	}
	if !found {
		rec := Record{
			RecordEnvelope: RecordEnvelope{
				ID:             newULID(),
				Kind:           KindSubmissionSettled,
				ConversationID: sub.ConversationID,
				Session:        sub.SessionKey.Session,
				SubmissionID:   sub.ID,
				AttemptID:      sub.AttemptID,
				Time:           time.Now(),
			},
			Payload: mustPayload(&payload),
		}
		if err := c.rt.store.AppendRecords(ctx, sub.ConversationID, []Record{rec}); err != nil {
			return payload, fmt.Errorf("append settled record: %w", err)
		}
		c.rt.notifyAppend()
	}
	if sub.ParentSubmissionID != "" {
		if err := c.wakeParent(ctx, sub, effective); err != nil {
			return payload, fmt.Errorf("wake parent: %w", err)
		}
	}
	return effective, nil
}

// submissionRun is the session engine for one attempt: it owns the pi.Agent,
// authors canonical records from the event stream, and tracks turn
// correlation.
type submissionRun struct {
	rt       *Runtime
	sub      Submission
	conv     Conversation
	cfg      AgentRuntimeConfig
	deadline time.Time

	mu     sync.Mutex
	turnID string
	halted bool

	// lastAssistantText is the most recent completed assistant text message
	// — the candidate for structured-result validation.
	lastAssistantText string

	// result is the validated structured result, set by drive when the
	// prompt requested one.
	result json.RawMessage

	// suspended reports the prompt ended on a Suspend-marked call
	// (HARNESS-15). Whether the attempt parks is decided in drive from
	// durable child state (reconcileSpawnRecords): parking leaves this set
	// and runSubmission waits the submission; a genuine bare Suspend from a
	// non-task tool clears it and the run settles normally. Single-goroutine
	// invariant: written in promptOnce/drive and read in drive/driveAttempt,
	// all on the drive goroutine — unlike the mutex-guarded fields above it
	// needs no lock.
	suspended bool

	// pendingSpawns counts the task_spawned records resolved for this
	// attempt (authored by the consumer, or already durable from a prior
	// admission). It feeds the park gate in reconcileSpawnRecords
	// (children-existence || pendingSpawns; HARNESS-15 re-review) as
	// belt-and-braces — the spawn payload rides agent-core's lossy event
	// channel, so this count alone cannot be trusted. Same
	// single-goroutine invariant as suspended.
	pendingSpawns int

	// Pending delta batch (accessed only from the event-consuming goroutine).
	deltaKind    RecordKind
	deltaBuf     []byte
	deltaFirstAt time.Time
}

func (r *submissionRun) currentTurnID() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.turnID
}

func (r *submissionRun) setTurnID(id string) {
	r.mu.Lock()
	r.turnID = id
	r.mu.Unlock()
}

// correlation snapshots this run's correlation ids for events and OpInfo.
func (r *submissionRun) correlation() Correlation {
	return Correlation{
		SessionKey:     r.sub.SessionKey,
		ConversationID: r.conv.ID,
		SubmissionID:   r.sub.ID,
		AttemptID:      r.sub.AttemptID,
		TurnID:         r.currentTurnID(),
	}
}

// record builds a canonical record stamped with this run's correlation ids.
func (r *submissionRun) record(kind RecordKind, payload interface{ payloadKind() RecordKind }) Record {
	return Record{
		RecordEnvelope: RecordEnvelope{
			ID:             newULID(),
			Kind:           kind,
			ConversationID: r.conv.ID,
			Session:        r.conv.Key.Session,
			SubmissionID:   r.sub.ID,
			TurnID:         r.currentTurnID(),
			AttemptID:      r.sub.AttemptID,
			Time:           time.Now(),
		},
		Payload: mustPayload(payload),
	}
}

func (r *submissionRun) append(ctx context.Context, recs ...Record) error {
	if err := r.rt.store.AppendRecords(ctx, r.conv.ID, recs); err != nil {
		return fmt.Errorf("append records: %w", err)
	}
	r.rt.notifyAppend()
	return nil
}

// drive executes the attempt: input record, prompt, event consumption,
// terminal result. Between turns it halts cooperatively when the durability
// deadline has passed or the run context ended.
func (r *submissionRun) drive(ctx context.Context) error {
	// Reconcile before authoring this submission's own input record — not
	// just before the prompt — so a synthesized outcome lands in log order
	// immediately after its dangling call, never after a newer user_message.
	// This runs on every drive, not just a re-claimed attempt of THIS
	// submission (AttemptCount > 1): the hazard is conversation-scoped. A
	// submission can settle failed — initialize failure, attempt-budget
	// exhaustion, durability timeout — with a dangling assistant_tool_call
	// still on the active leaf path; the next submission on the same
	// conversation starts at its own AttemptCount == 1 and would otherwise
	// skip the scan forever, replaying the bare call into every future
	// prompt. The scan is cheap: one read (ReadRecords) plus a path walk.
	if err := r.reconcileDanglingToolCalls(ctx); err != nil {
		return err
	}
	if err := r.appendInputRecord(ctx); err != nil {
		return err
	}

	proj := &projection{
		store:        r.rt.store,
		conv:         r.conv,
		systemPrompt: r.cfg.SystemPrompt,
		submissionID: r.sub.ID,
		attemptID:    r.sub.AttemptID,
		turnID:       r.currentTurnID,
	}
	tools := r.cfg.Tools
	if targets := r.rt.subagentTargets(r.conv.Key.Agent, r.sub.Depth); len(targets) > 0 {
		// Clone so the append cannot alias the definition's tool slice.
		tools = append(slices.Clone(r.cfg.Tools), newTaskTool(r.rt, r, targets))
	}
	agent, err := pi.NewAgent(pi.AgentConfig{
		Providers:          r.interceptedProviders(),
		DefaultModel:       r.cfg.Model,
		SystemPrompt:       r.cfg.SystemPrompt,
		Tools:              r.interceptedTools(tools),
		Skills:             r.cfg.Skills,
		ReserveTokens:      r.cfg.ReserveTokens,
		KeepRecentTokens:   r.cfg.KeepRecentTokens,
		SummarizationRetry: r.cfg.SummarizationRetry,
		Session:            proj,
		Hooks: pi.Hooks{
			OnSummarizationRetry: summarizationRetryObserver(r.rt.observe, r.correlation()),
			ShouldStopAfterTurn: func(hctx context.Context, c pi.AfterTurnCtx) bool {
				if ctx.Err() != nil {
					return true
				}
				if time.Now().After(r.deadline) {
					r.mu.Lock()
					r.halted = true
					r.mu.Unlock()
					return true
				}
				return false
			},
		},
	})
	if err != nil {
		return fmt.Errorf("construct agent: %w", err)
	}
	defer agent.Close()

	// Expose the run for Steer/FollowUp passthrough while it is in flight.
	r.rt.registerLiveRun(r.conv.Key, agent)
	defer r.rt.unregisterLiveRun(r.conv.Key)

	// A submission claimed with PendingResume was woken from a suspension
	// (HARNESS-15): the wake landed the pending call's tool_outcome before
	// the requeue, so the transcript tail is a tool result and the drive
	// Resumes instead of re-prompting the input. A resumed turn can suspend
	// again (more children) — the suspended flag propagates as on a prompt
	// drive.
	if r.sub.PendingResume {
		err = r.runRecovered(ctx, agent, r.runResume)
	} else {
		err = r.runRecovered(ctx, agent, func(c context.Context, a *pi.Agent) error {
			return r.runPrompt(c, a, inputToMessage(r.sub.Input))
		})
	}
	if err != nil {
		return err
	}
	if r.suspended {
		// Park-time reconciliation (HARNESS-15 re-review): repair any spawn
		// record the lossy event channel dropped, from durable child state,
		// then decide. Parking skips result validation — no final answer
		// this attempt; a genuine bare suspend falls through and settles
		// normally.
		park, err := r.reconcileSpawnRecords(ctx)
		if err != nil {
			return err
		}
		if park {
			return nil
		}
		r.suspended = false
	}
	if len(r.sub.Input.ResultSchema) > 0 {
		return r.validateResultLoop(ctx, agent)
	}
	return nil
}

// runRecovered is the turn-recovery ladder around one prompt or resume:
// context overflow compacts and retries under a small budget; other stream
// errors are classified fatal (llm.ErrProviderFatal) or transient (budgeted
// backoff via a fresh attempt).
func (r *submissionRun) runRecovered(ctx context.Context, agent *pi.Agent, promptFn func(context.Context, *pi.Agent) error) error {
	compactions := 0
	for {
		err := promptFn(ctx, agent)
		if err == nil {
			return nil
		}
		if errors.Is(err, errDeadlineHalted) || ctx.Err() != nil {
			return err
		}
		if errors.Is(err, pi.ErrNothingToResume) {
			// The wake lands the outcome before requeueing, so a resume
			// drive always finds a tool-result tail; a miss signals a
			// wake-ordering bug. Propagate: the catch-all arm settles the
			// submission failed (run_failed) — a hard fail is intended,
			// never a retry.
			return err
		}
		if errors.Is(llm.AsContextOverflow(err), llm.ErrContextOverflow) {
			if compactions >= overflowCompactRetries {
				return fmt.Errorf("context overflow persisted after %d compactions: %w", compactions, err)
			}
			compactions++
			r.rt.logger.Info("context overflow; compacting and retrying the turn",
				"submission", r.sub.ID, "compaction", compactions)
			r.rt.observe(RecoveryEvent{Correlation: r.correlation(), Decision: "overflow_compact_retry", Detail: err.Error()})
			cerr := r.rt.intercept(ctx, OpInfo{Kind: OpOperation, Operation: "compact", Correlation: r.correlation()}, func(c context.Context) error {
				_, e := agent.Compact(c, pi.CompactOpts{})
				return e
			})
			if cerr != nil {
				return fmt.Errorf("compact after overflow: %w", cerr)
			}
			r.rt.observe(CompactionEvent{Correlation: r.correlation(), Reason: "overflow"})
			r.rt.notifyAppend()
			continue
		}
		if errors.Is(err, llm.ErrProviderFatal) {
			return err
		}
		return &transientRunError{err: err}
	}
}

// reconcileDanglingToolCalls appends a synthesized error tool_outcome for
// every assistant_tool_call on the active leaf path that has no matching
// tool_outcome. A crash between the two records would otherwise replay a bare
// tool call straight into the provider on recovery, which deterministic-4xx
// providers reject (HARNESS-14; harness half of upstream #6285). Runs on
// every drive — the hazard is conversation-scoped (see drive's call site),
// not limited to a re-claimed attempt.
func (r *submissionRun) reconcileDanglingToolCalls(ctx context.Context) error {
	recs, err := r.rt.store.ReadRecords(ctx, r.conv.ID, "")
	if err != nil {
		return fmt.Errorf("read records for dangling tool call reconciliation: %w", err)
	}

	// order preserves path order so synthesized outcomes append in the same
	// order their calls were made; pending tracks which calls still lack an
	// outcome as the path is walked.
	//
	// Latent edge (disclosed, not defended against): pending is keyed by
	// CallID alone, so if a provider ever reused a CallID for two distinct
	// calls on the same active leaf path, the tool_outcome that satisfies the
	// first would also clear the second out of pending, leaving a
	// legitimately dangling second call unreconciled. Providers are expected
	// to mint unique call ids per turn, so this is believed unreachable.
	type danglingCall struct {
		callID   string
		toolName string
	}
	var order []danglingCall
	pending := make(map[string]bool)

	path := Reduce(recs).ActiveLeafPath()
	for _, rec := range path {
		switch rec.Kind {
		case KindAssistantToolCall:
			var p AssistantToolCallPayload
			if err := rec.DecodePayload(&p); err != nil {
				return fmt.Errorf("decode assistant_tool_call for reconciliation: %w", err)
			}
			order = append(order, danglingCall{callID: p.CallID, toolName: p.ToolName})
			pending[p.CallID] = true
		case KindToolOutcome:
			var p ToolOutcomePayload
			if err := rec.DecodePayload(&p); err != nil {
				return fmt.Errorf("decode tool_outcome for reconciliation: %w", err)
			}
			delete(pending, p.CallID)
		}
	}

	// Suspension exemption (HARNESS-15): a spawned task call is
	// intentionally outcome-less while its child runs — the wake authors
	// the outcome at settlement. Subtract those calls so a re-driven
	// parent never sees a fabricated error for a live child (which the
	// wake's existing-outcome-wins check would then honor forever). A
	// missing child row means the admission never landed — treat it as
	// settled and let the call reconcile. The lookup walks the same
	// active-path slice pending was built from (CallIDs are assumed unique
	// per path).
	for _, rec := range path {
		if rec.Kind != KindTaskSpawned {
			continue
		}
		var sp TaskSpawnedPayload
		if err := rec.DecodePayload(&sp); err != nil {
			return fmt.Errorf("decode task_spawned for reconciliation: %w", err)
		}
		if !pending[sp.CallID] {
			continue
		}
		child, err := r.rt.store.GetSubmission(ctx, sp.ChildSubmissionID)
		if err != nil {
			if errors.Is(err, ErrSubmissionNotFound) {
				continue
			}
			return fmt.Errorf("look up spawned child %s for reconciliation: %w", sp.ChildSubmissionID, err)
		}
		if child.Status != StatusSettled {
			delete(pending, sp.CallID)
		}
	}
	if len(pending) == 0 {
		return nil
	}

	var synthesized []Record
	for _, call := range order {
		if !pending[call.callID] {
			continue
		}
		synthesized = append(synthesized, r.record(KindToolOutcome, &ToolOutcomePayload{
			CallID:   call.callID,
			ToolName: call.toolName,
			IsError:  true,
			Content:  danglingToolCallMessage,
		}))
	}
	if err := r.append(ctx, synthesized...); err != nil {
		return err
	}
	r.rt.observe(RecoveryEvent{
		Correlation: r.correlation(),
		Decision:    "dangling_tool_call_reconciled",
		Detail:      fmt.Sprintf("synthesized %d error tool_outcome record(s) for dangling assistant_tool_call(s)", len(synthesized)),
	})
	return nil
}

// transientBackoff derives the retry delay from the durable attempt count:
// base doubling per attempt, capped at 5s. The base tracks ClaimInterval so
// tightened test engines back off proportionally.
func transientBackoff(base time.Duration, attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	d := base << (attempt - 1)
	if d > 5*time.Second || d <= 0 {
		return 5 * time.Second
	}
	return d
}

// runPrompt runs one prompt operation on the agent — wrapped in the
// OpOperation interceptor boundary and bounded by operation events — and
// consumes its event stream into canonical records.
func (r *submissionRun) runPrompt(ctx context.Context, agent *pi.Agent, msg pi.Message) error {
	corr := r.correlation()
	r.rt.observe(OperationStartedEvent{Correlation: corr, Operation: "prompt"})
	err := r.rt.intercept(ctx, OpInfo{Kind: OpOperation, Operation: "prompt", Correlation: corr}, func(c context.Context) error {
		return r.promptOnce(c, agent, msg)
	})
	r.rt.observe(OperationEndedEvent{Correlation: r.correlation(), Operation: "prompt", Err: errString(err)})
	return err
}

// runResume runs one resume operation on the agent — the re-drive of a
// woken suspension (HARNESS-15) — wrapped in the OpOperation interceptor
// boundary and bounded by operation events exactly like runPrompt.
func (r *submissionRun) runResume(ctx context.Context, agent *pi.Agent) error {
	corr := r.correlation()
	r.rt.observe(OperationStartedEvent{Correlation: corr, Operation: "resume"})
	err := r.rt.intercept(ctx, OpInfo{Kind: OpOperation, Operation: "resume", Correlation: corr}, func(c context.Context) error {
		return r.resumeOnce(c, agent)
	})
	r.rt.observe(OperationEndedEvent{Correlation: r.correlation(), Operation: "resume", Err: errString(err)})
	return err
}

// resumeOnce is the unwrapped resume body: agent.Resume continues from the
// transcript without appending input, so a wake re-drive lands no second
// user_message. ErrNothingToResume propagates (runRecovered fails the
// attempt on it).
func (r *submissionRun) resumeOnce(ctx context.Context, agent *pi.Agent) error {
	stream, err := agent.Resume(ctx, pi.PromptOpts{
		SessionID: pi.SessionID(r.conv.ID),
	})
	if err != nil {
		return fmt.Errorf("start resume: %w", err)
	}
	return r.consumeStream(ctx, agent, "resume", stream)
}

// promptOnce is the unwrapped prompt body.
func (r *submissionRun) promptOnce(ctx context.Context, agent *pi.Agent, msg pi.Message) error {
	stream, err := agent.Prompt(ctx, msg, pi.PromptOpts{
		SessionID: pi.SessionID(r.conv.ID),
	})
	if err != nil {
		return fmt.Errorf("start prompt: %w", err)
	}
	return r.consumeStream(ctx, agent, "prompt", stream)
}

// consumeStream drains one prompt or resume stream into canonical records
// and maps the terminal result: halt, interruption, suspension.
func (r *submissionRun) consumeStream(ctx context.Context, agent *pi.Agent, op string, stream *pi.EventStream) error {
	for ev := range stream.Events {
		if err := r.consumeEvent(ctx, ev); err != nil {
			// Record authoring must not lose events silently; stop the run.
			agent.Stop()
			r.rt.logger.Error("author record from event", "submission", r.sub.ID, "error", err)
		}
	}
	if err := r.flushDeltas(ctx); err != nil {
		r.rt.logger.Error("flush trailing deltas", "submission", r.sub.ID, "error", err)
	}
	result := <-stream.Done
	if result.Err != nil {
		return fmt.Errorf("%s: %w", op, result.Err)
	}
	r.mu.Lock()
	halted := r.halted
	r.mu.Unlock()
	if halted {
		return errDeadlineHalted
	}
	if ctx.Err() != nil {
		return fmt.Errorf("run interrupted: %w", context.Cause(ctx))
	}
	if result.Suspended {
		// Propagate the suspension regardless of pendingSpawns: the spawn
		// payload rides agent-core's lossy event channel, so a dropped
		// ToolCallEndEvent leaves pendingSpawns at zero with a live child.
		// drive's park-time reconciliation makes the park/settle decision
		// from durable child state.
		r.suspended = true
	}
	return nil
}

// validateResultLoop validates the final answer against the requested
// schema, feeding validation errors back as corrective turns under the
// per-prompt retry budget. The corrective turn is a canonical user_message,
// so it is visible in the record stream.
func (r *submissionRun) validateResultLoop(ctx context.Context, agent *pi.Agent) error {
	retries := r.sub.Input.ResultRetries
	if retries <= 0 {
		retries = DefaultResultRetries
	}
	for attempt := 0; ; attempt++ {
		r.mu.Lock()
		answer := r.lastAssistantText
		r.mu.Unlock()
		result, reason := validateStructuredResult(r.sub.Input.ResultSchema, answer)
		if reason == "" {
			r.result = result
			return nil
		}
		if attempt >= retries {
			return &resultInvalidError{reason: reason}
		}
		corrective := correctiveMessage(reason, r.sub.Input.ResultSchema)
		if err := r.append(ctx, r.record(KindUserMessage, &UserMessagePayload{Body: corrective})); err != nil {
			return err
		}
		if err := r.runPrompt(ctx, agent, pi.NewText("user", corrective)); err != nil {
			return err
		}
	}
}

// appendInputRecord authors the user_message (or signal) record for this
// submission unless a prior attempt already landed it.
func (r *submissionRun) appendInputRecord(ctx context.Context) error {
	recs, err := r.rt.store.ReadRecords(ctx, r.conv.ID, "")
	if err != nil {
		return fmt.Errorf("read records for input dedupe: %w", err)
	}
	for _, rec := range recs {
		if rec.SubmissionID == r.sub.ID && (rec.Kind == KindUserMessage || rec.Kind == KindSignal) {
			return nil // a prior attempt already authored the input
		}
	}
	if r.sub.Input.Kind == InboundSignal && r.sub.Input.Signal != nil {
		rec := r.record(KindSignal, &SignalPayload{
			Type:   r.sub.Input.Signal.Type,
			Body:   r.sub.Input.Body,
			Sender: r.sub.Input.Signal.Sender,
			Tag:    r.sub.Input.Signal.Tag,
		})
		return r.append(ctx, rec)
	}
	rec := r.record(KindUserMessage, &UserMessagePayload{
		Body:        r.sub.Input.Body,
		Attachments: r.sub.Input.Attachments,
	})
	return r.append(ctx, rec)
}

// consumeEvent authors canonical records from one agent event. Deltas are
// batched (flush on size, staleness, and every message boundary); any
// non-delta record flushes pending deltas first so the log stays ordered.
func (r *submissionRun) consumeEvent(ctx context.Context, ev pi.AgentEvent) error {
	switch e := ev.(type) {
	case pi.TextDeltaEvent:
		r.rt.observe(DeltaEvent{Correlation: r.correlation(), Kind: KindAssistantTextDelta, Text: e.Delta})
		return r.bufferDelta(ctx, KindAssistantTextDelta, e.Delta)
	case pi.ThinkingDeltaEvent:
		r.rt.observe(DeltaEvent{Correlation: r.correlation(), Kind: KindAssistantThinkingDelta, Text: e.Delta})
		return r.bufferDelta(ctx, KindAssistantThinkingDelta, e.Delta)
	case pi.TurnStartEvent:
		r.setTurnID(newULID())
		r.rt.observe(TurnStartedEvent{Correlation: r.correlation(), Turn: e.Turn})
	case pi.TurnEndEvent:
		r.rt.observe(TurnEndedEvent{Correlation: r.correlation(), Turn: e.Turn})
	case pi.SteerInjectedEvent:
		return r.appendInjected(ctx, e.Message)
	case pi.FollowUpInjectedEvent:
		return r.appendInjected(ctx, e.Message)
	case pi.MessageStartEvent:
		if e.Role != "assistant" {
			return nil
		}
		if err := r.flushDeltas(ctx); err != nil {
			return err
		}
		rec := r.record(KindAssistantMessageStarted, &AssistantMessageStartedPayload{
			Model:       r.cfg.Model,
			MessageType: e.MessageType,
		})
		return r.append(ctx, rec)
	case pi.ToolCallStartEvent:
		r.rt.observe(ToolCallStartedEvent{Correlation: r.correlation(), CallID: e.CallID, ToolName: e.ToolName})
		if err := r.flushDeltas(ctx); err != nil {
			return err
		}
		rec := r.record(KindAssistantToolCall, &AssistantToolCallPayload{
			CallID:           e.CallID,
			ToolName:         e.ToolName,
			Args:             e.Args,
			ThoughtSignature: e.ThoughtSignature,
		})
		return r.append(ctx, rec)
	case pi.ToolUpdateEvent:
		r.rt.observe(ToolCallUpdatedEvent{Correlation: r.correlation(), CallID: e.CallID, ToolName: e.Name, Result: e.Result})
		return nil
	case pi.ToolCallEndEvent:
		r.rt.observe(ToolCallEndedEvent{Correlation: r.correlation(), CallID: e.CallID, ToolName: e.ToolName, IsError: e.Result.IsError})
		if err := r.flushDeltas(ctx); err != nil {
			return err
		}
		if e.Result.Suspend {
			// A suspended task call carries its spawn payload in Data
			// (HARNESS-15). Author the task_spawned record HERE, on the
			// consumer goroutine, so it lands after the
			// assistant_tool_call record by construction. No outcome is
			// authored — the pending call is the suspension point; the
			// wake authors the outcome on child settlement.
			var sd spawnData
			if err := json.Unmarshal(e.Result.Data, &sd); err == nil && sd.SpawnRecordID != "" {
				exists, err := spawnRecordExists(ctx, r.rt.store, r.conv.ID, sd.CallID)
				if err != nil {
					return err
				}
				if !exists {
					rec := r.record(KindTaskSpawned, &TaskSpawnedPayload{
						CallID:              sd.CallID,
						Agent:               sd.Agent,
						ChildInstance:       sd.ChildInstance,
						ChildConversationID: sd.ChildConversationID,
						ChildSubmissionID:   sd.ChildSubmissionID,
						Prompt:              sd.Prompt,
					})
					rec.ID = sd.SpawnRecordID
					if err := r.append(ctx, rec); err != nil {
						// A failed append here orphans the admitted child: no
						// task_spawned names it. Two nets catch that — park-time
						// reconciliation re-authors the record from durable child
						// state (reconcileSpawnRecords), and the orphan cascade
						// (Task 11) cancels the child if the parent settles with
						// it still live.
						return err
					}
				}
				r.pendingSpawns++
				return nil
			}
			// A Suspend result without spawn data is not a task
			// suspension: author the outcome as if Suspend were unset so
			// the submission settles normally instead of parking forever.
			r.rt.logger.Warn("suspend result without spawn data; authoring outcome normally",
				"submission", r.sub.ID, "callId", e.CallID, "tool", e.ToolName)
		}
		rec := r.record(KindToolOutcome, &ToolOutcomePayload{
			CallID:   e.CallID,
			ToolName: e.ToolName,
			Content:  e.Result.Content,
			Data:     e.Result.Data,
			IsError:  e.Result.IsError,
		})
		return r.append(ctx, rec)
	case pi.MessageEndEvent:
		// Message end always flushes, even for non-assistant messages.
		if err := r.flushDeltas(ctx); err != nil {
			return err
		}
		if e.Message.Role != "assistant" {
			return nil
		}
		if text := e.Message.Text(); text != "" {
			r.mu.Lock()
			r.lastAssistantText = text
			r.mu.Unlock()
		}
		rec := r.record(KindAssistantMessageCompleted, &AssistantMessageCompletedPayload{
			Message: messageFromPi(e.Message),
		})
		return r.append(ctx, rec)
	}
	return nil
}

// appendInjected authors the canonical record for a steered or followed-up
// message, so readers see why the run changed course.
func (r *submissionRun) appendInjected(ctx context.Context, msg pi.Message) error {
	if err := r.flushDeltas(ctx); err != nil {
		return err
	}
	rec := r.record(KindUserMessage, &UserMessagePayload{Body: msg.Text()})
	return r.append(ctx, rec)
}

// bufferDelta accumulates one streamed fragment, flushing on kind change,
// size, or staleness.
func (r *submissionRun) bufferDelta(ctx context.Context, kind RecordKind, delta string) error {
	if len(r.deltaBuf) > 0 && r.deltaKind != kind {
		if err := r.flushDeltas(ctx); err != nil {
			return err
		}
	}
	if len(r.deltaBuf) == 0 {
		r.deltaKind = kind
		r.deltaFirstAt = time.Now()
	}
	r.deltaBuf = append(r.deltaBuf, delta...)
	if len(r.deltaBuf) >= r.rt.deltaFlushBytes || time.Since(r.deltaFirstAt) >= r.rt.deltaFlushInterval {
		return r.flushDeltas(ctx)
	}
	return nil
}

// flushDeltas appends the pending delta batch, if any, as one record.
func (r *submissionRun) flushDeltas(ctx context.Context) error {
	if len(r.deltaBuf) == 0 {
		return nil
	}
	text := string(r.deltaBuf)
	kind := r.deltaKind
	r.deltaBuf = r.deltaBuf[:0]

	var rec Record
	if kind == KindAssistantThinkingDelta {
		rec = r.record(kind, &ThinkingDeltaPayload{Text: text})
	} else {
		rec = r.record(kind, &TextDeltaPayload{Text: text})
	}
	return r.append(ctx, rec)
}

// correlation snapshots a submission's correlation ids (no live turn).
func (s Submission) correlation() Correlation {
	return Correlation{
		SessionKey:     s.SessionKey,
		ConversationID: s.ConversationID,
		SubmissionID:   s.ID,
		AttemptID:      s.AttemptID,
	}
}

// interceptedProviders wraps each configured provider so the OpTurn
// interceptor boundary covers every model round-trip.
func (r *submissionRun) interceptedProviders() []llm.LLMProvider {
	if len(r.rt.interceptors) == 0 {
		return r.cfg.Providers
	}
	out := make([]llm.LLMProvider, len(r.cfg.Providers))
	for i, p := range r.cfg.Providers {
		out[i] = &interceptedProvider{inner: p, run: r}
	}
	return out
}

// interceptedProvider wraps one provider's Stream call in the interceptor
// chain: next covers the full model round-trip (events drained, result
// delivered).
type interceptedProvider struct {
	inner llm.LLMProvider
	run   *submissionRun
}

func (p *interceptedProvider) Name() string { return p.inner.Name() }

func (p *interceptedProvider) Capabilities(model string) llm.ProviderCapabilities {
	return p.inner.Capabilities(model)
}

func (p *interceptedProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	events := make(chan llm.LLMEvent, 16)
	done := make(chan llm.StreamResult, 1)
	go func() {
		defer close(done)
		delivered := false
		err := p.run.rt.intercept(ctx, OpInfo{Kind: OpTurn, Correlation: p.run.correlation()}, func(c context.Context) error {
			es := p.inner.Stream(c, req)
			for ev := range es.Events {
				events <- ev
			}
			res := <-es.Done
			close(events)
			delivered = true
			done <- res
			return res.Err
		})
		if !delivered {
			// The chain aborted before (or instead of) running the model
			// call; surface the abort as the stream outcome.
			close(events)
			done <- llm.StreamResult{Err: err}
		}
	}()
	return llm.NewEventStream(events, done)
}

// interceptedTools wraps each registered tool so the OpTool interceptor
// boundary covers every execution — including the injected task tool.
func (r *submissionRun) interceptedTools(tools []pi.RegisteredTool) []pi.RegisteredTool {
	if len(r.rt.interceptors) == 0 || len(tools) == 0 {
		return tools
	}
	out := make([]pi.RegisteredTool, len(tools))
	for i, t := range tools {
		out[i] = &interceptedTool{inner: t, run: r}
	}
	return out
}

// interceptedTool wraps one tool's Execute in the interceptor chain.
type interceptedTool struct {
	inner pi.RegisteredTool
	run   *submissionRun
}

func (t *interceptedTool) Name() string            { return t.inner.Name() }
func (t *interceptedTool) Description() string     { return t.inner.Description() }
func (t *interceptedTool) Schema() json.RawMessage { return t.inner.Schema() }
func (t *interceptedTool) IsSequential() bool      { return t.inner.IsSequential() }

func (t *interceptedTool) Execute(ctx context.Context, callID string, args json.RawMessage) (pi.ToolResult, error) {
	op := OpInfo{Kind: OpTool, Correlation: t.run.correlation(), ToolName: t.inner.Name(), CallID: callID}
	var result pi.ToolResult
	err := t.run.rt.intercept(ctx, op, func(c context.Context) error {
		var xerr error
		result, xerr = t.inner.Execute(c, callID, args)
		return xerr
	})
	return result, err
}
