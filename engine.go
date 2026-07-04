package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// Engine timing defaults; Config.ClaimInterval/LeaseDuration override them.
const (
	defaultClaimInterval = 250 * time.Millisecond
	defaultLeaseDuration = 30 * time.Second
)

// errLeaseLost cancels a run whose heartbeat discovered another attempt owns
// the submission.
var errLeaseLost = errors.New("lease lost to another attempt")

// errDeadlineHalted stops a run whose durability timeout passed mid-flight
// (cooperative halt at a turn boundary).
var errDeadlineHalted = errors.New("durability timeout reached mid-run")

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

	runErr := c.driveAttempt(runCtx, sub, cfg, deadline)
	cancelRun(nil)
	<-heartbeatDone

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
	case runErr != nil:
		logger.Error("attempt failed", "error", runErr)
		c.settleAndNotify(ctx, sub, SettledPayload{
			Status: SettledFailed, Error: runErr.Error(), ErrorCode: SettledErrRunFailed,
		}, logger)
	default:
		c.settleAndNotify(ctx, sub, SettledPayload{Status: SettledSucceeded}, logger)
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
	c.rt.notifySettled()
}

// driveAttempt runs the agent for one attempt and returns the run error.
func (c *coordinator) driveAttempt(ctx context.Context, sub Submission, cfg AgentRuntimeConfig, deadline time.Time) error {
	conv, err := c.rt.store.GetConversation(ctx, sub.SessionKey)
	if err != nil {
		return fmt.Errorf("resolve conversation for %s: %w", sub.SessionKey, err)
	}
	run := &submissionRun{
		rt:       c.rt,
		sub:      sub,
		conv:     conv,
		cfg:      cfg,
		deadline: deadline,
	}
	return run.drive(ctx)
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
		Providers:    r.cfg.Providers,
		DefaultModel: r.cfg.Model,
		SystemPrompt: r.cfg.SystemPrompt,
		Tools:        r.cfg.Tools,
		Skills:       r.cfg.Skills,
		Session:      proj,
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

	stream, err := agent.Prompt(ctx, pi.NewText("user", r.sub.Input.Body), pi.PromptOpts{
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
	rec := r.record(KindUserMessage, &UserMessagePayload{
		Body:        r.sub.Input.Body,
		Attachments: r.sub.Input.Attachments,
	})
	return r.append(ctx, rec)
}

// consumeEvent authors canonical records from one agent event. Delta
// batching joins in the streaming slice (HARNESS-4).
func (r *submissionRun) consumeEvent(ctx context.Context, ev pi.AgentEvent) error {
	switch e := ev.(type) {
	case pi.TurnStartEvent:
		r.setTurnID(newULID())
	case pi.ToolCallStartEvent:
		rec := r.record(KindAssistantToolCall, &AssistantToolCallPayload{
			CallID:   e.CallID,
			ToolName: e.ToolName,
			Args:     e.Args,
		})
		return r.append(ctx, rec)
	case pi.ToolCallEndEvent:
		rec := r.record(KindToolOutcome, &ToolOutcomePayload{
			CallID:   e.CallID,
			ToolName: e.ToolName,
			Content:  e.Result.Content,
			Data:     e.Result.Data,
			IsError:  e.Result.IsError,
		})
		return r.append(ctx, rec)
	case pi.MessageEndEvent:
		if e.Message.Role != "assistant" {
			return nil
		}
		rec := r.record(KindAssistantMessageCompleted, &AssistantMessageCompletedPayload{
			Message: messageFromPi(e.Message),
		})
		return r.append(ctx, rec)
	}
	return nil
}
