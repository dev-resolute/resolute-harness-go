package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
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
}

func newCoordinator(rt *Runtime) *coordinator {
	return &coordinator{
		rt:      rt,
		ownerID: newULID(),
		active:  make(map[string]bool),
	}
}

// loop reconciles once at startup, then claims and reclaims until ctx is
// cancelled. It wakes on admission nudges and on a steady tick.
func (c *coordinator) loop(ctx context.Context) {
	c.reconcile(ctx)
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
	if err := c.appendSettledRecordOnce(ctx, sub, SettledPayload{
		Status:    SettledFailed,
		Error:     "process crashed during settlement; run outcome unknown",
		ErrorCode: SettledErrIndeterminate,
	}); err != nil {
		return err
	}
	if err := c.rt.store.FinalizeSettlement(ctx, sub.ID); err != nil {
		return fmt.Errorf("finalize settlement: %w", err)
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
		err := c.rt.store.ReleaseSubmission(ctx, sub.ID, sub.AttemptID)
		if err != nil && !errors.Is(err, ErrClaimLost) {
			c.rt.logger.Error("release expired lease", "submission", sub.ID, "error", err)
			continue
		}
		if err == nil {
			c.rt.logger.Info("reclaimed expired lease", "submission", sub.ID, "deadOwner", sub.OwnerID)
		}
	}
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
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status:    SettledFailed,
			Error:     fmt.Sprintf("attempt budget exhausted: attempt %d exceeds max %d", sub.AttemptCount, maxAttempts),
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
	heartbeatDone := c.startHeartbeat(runCtx, sub, cancelRun, logger)

	var result json.RawMessage
	runErr := c.rt.intercept(runCtx, OpInfo{Kind: OpAttempt, Correlation: sub.correlation()}, func(cctx context.Context) error {
		var derr error
		result, derr = c.driveAttempt(cctx, sub, cfg, deadline)
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
	case runErr != nil && ctx.Err() != nil:
		// Shutdown interrupted the attempt: release the claim so a fresh
		// Runtime (or this store's next owner) re-attempts immediately.
		releaseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := c.rt.store.ReleaseSubmission(releaseCtx, sub.ID, sub.AttemptID); err != nil && !errors.Is(err, ErrClaimLost) {
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
		if err := c.rt.store.ReleaseSubmission(releaseCtx, sub.ID, sub.AttemptID); err != nil && !errors.Is(err, ErrClaimLost) {
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
}

// driveAttempt runs the agent for one attempt, returning the validated
// structured result (nil when none was requested) and the run error.
func (c *coordinator) driveAttempt(ctx context.Context, sub Submission, cfg AgentRuntimeConfig, deadline time.Time) (json.RawMessage, error) {
	conv, err := c.rt.store.GetConversation(ctx, sub.SessionKey)
	if err != nil {
		return nil, fmt.Errorf("resolve conversation for %s: %w", sub.SessionKey, err)
	}
	run := &submissionRun{
		rt:       c.rt,
		sub:      sub,
		conv:     conv,
		cfg:      cfg,
		deadline: deadline,
	}
	if err := run.drive(ctx); err != nil {
		return nil, err
	}
	return run.result, nil
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
	if err := c.appendSettledRecordOnce(ctx, sub, payload); err != nil {
		return err
	}
	if err := c.rt.store.FinalizeSettlement(ctx, sub.ID); err != nil {
		return fmt.Errorf("finalize settlement: %w", err)
	}
	return nil
}

// appendSettledRecordOnce appends the submission_settled record unless one
// already exists for the submission — the idempotency half of two-phase
// settlement.
func (c *coordinator) appendSettledRecordOnce(ctx context.Context, sub Submission, payload SettledPayload) error {
	recs, err := c.rt.store.ReadRecords(ctx, sub.ConversationID, "")
	if err != nil {
		return fmt.Errorf("read records before settle: %w", err)
	}
	for _, rec := range recs {
		if rec.Kind == KindSubmissionSettled && rec.SubmissionID == sub.ID {
			return nil
		}
	}
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
		return fmt.Errorf("append settled record: %w", err)
	}
	c.rt.notifyAppend()
	return nil
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
	agent, err := pi.NewAgent(pi.AgentConfig{
		Providers:        r.interceptedProviders(),
		DefaultModel:     r.cfg.Model,
		SystemPrompt:     r.cfg.SystemPrompt,
		Tools:            r.interceptedTools(),
		Skills:           r.cfg.Skills,
		ReserveTokens:    r.cfg.ReserveTokens,
		KeepRecentTokens: r.cfg.KeepRecentTokens,
		Session:          proj,
		Hooks: pi.Hooks{
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

	if err := r.runRecovered(ctx, agent, inputToMessage(r.sub.Input)); err != nil {
		return err
	}
	if len(r.sub.Input.ResultSchema) > 0 {
		return r.validateResultLoop(ctx, agent)
	}
	return nil
}

// runRecovered is the turn-recovery ladder around one prompt: context
// overflow compacts and retries under a small budget; other stream errors
// are classified fatal (llm.ErrProviderFatal) or transient (budgeted
// backoff via a fresh attempt).
func (r *submissionRun) runRecovered(ctx context.Context, agent *pi.Agent, msg pi.Message) error {
	compactions := 0
	for {
		err := r.runPrompt(ctx, agent, msg)
		if err == nil {
			return nil
		}
		if errors.Is(err, errDeadlineHalted) || ctx.Err() != nil {
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

// promptOnce is the unwrapped prompt body.
func (r *submissionRun) promptOnce(ctx context.Context, agent *pi.Agent, msg pi.Message) error {
	stream, err := agent.Prompt(ctx, msg, pi.PromptOpts{
		SessionID: pi.SessionID(r.conv.ID),
	})
	if err != nil {
		return fmt.Errorf("start prompt: %w", err)
	}

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
		return fmt.Errorf("prompt: %w", result.Err)
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
			CallID:   e.CallID,
			ToolName: e.ToolName,
			Args:     e.Args,
		})
		return r.append(ctx, rec)
	case pi.ToolCallEndEvent:
		r.rt.observe(ToolCallEndedEvent{Correlation: r.correlation(), CallID: e.CallID, ToolName: e.ToolName, IsError: e.Result.IsError})
		if err := r.flushDeltas(ctx); err != nil {
			return err
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
// boundary covers every execution.
func (r *submissionRun) interceptedTools() []pi.RegisteredTool {
	if len(r.rt.interceptors) == 0 || len(r.cfg.Tools) == 0 {
		return r.cfg.Tools
	}
	out := make([]pi.RegisteredTool, len(r.cfg.Tools))
	for i, t := range r.cfg.Tools {
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
