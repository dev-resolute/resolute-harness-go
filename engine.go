package harness

import (
	"context"
	"fmt"
	"sync"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// Engine defaults for the walking skeleton. Full lease/heartbeat semantics
// arrive in the durable-engine slice (HARNESS-3).
const (
	defaultClaimInterval = 250 * time.Millisecond
	defaultLeaseDuration = 30 * time.Second
)

// coordinator runs the claim loop: it leases runnable submissions and drives
// their sessions. One per Runtime process (v1).
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

// loop claims and runs submissions until ctx is cancelled. It wakes on
// admission nudges and on a steady tick.
func (c *coordinator) loop(ctx context.Context) {
	ticker := time.NewTicker(defaultClaimInterval)
	defer ticker.Stop()
	for {
		c.claimRunnable(ctx)
		select {
		case <-ctx.Done():
			return
		case <-c.rt.wake:
		case <-ticker.C:
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
			LeaseExpiresAt: time.Now().Add(defaultLeaseDuration),
		})
		if err != nil {
			c.release(key)
			if ctx.Err() == nil {
				c.rt.logger.Error("claim submission", "submission", sub.ID, "error", err)
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
// materialize the agent, author the input record, run the prompt while
// authoring canonical records from its event stream, then settle.
func (c *coordinator) runSubmission(ctx context.Context, sub Submission) {
	logger := c.rt.logger.With("submission", sub.ID, "session", sub.SessionKey.String(), "attempt", sub.AttemptID)

	err := c.driveAttempt(ctx, sub)
	if ctx.Err() != nil && err != nil {
		// Shutdown interrupted the attempt: leave the submission running for
		// reconciliation (HARNESS-3) rather than settling a false failure.
		logger.Info("attempt interrupted by shutdown", "error", err)
		return
	}

	settledPayload := SettledPayload{Status: SettledSucceeded}
	if err != nil {
		logger.Error("attempt failed", "error", err)
		settledPayload = SettledPayload{Status: SettledFailed, Error: err.Error()}
	}
	if err := c.settle(ctx, sub, settledPayload); err != nil {
		logger.Error("settle submission", "error", err)
		return
	}
	c.rt.notifySettled()
}

// driveAttempt runs the agent for one attempt and returns the run error.
func (c *coordinator) driveAttempt(ctx context.Context, sub Submission) error {
	def := c.rt.agents[sub.SessionKey.Agent]
	cfg, err := def.Initialize(ctx, sub.SessionKey.Instance, c.rt.env)
	if err != nil {
		return fmt.Errorf("initialize agent %q: %w", sub.SessionKey.Agent, err)
	}
	if err := cfg.validate(); err != nil {
		return err
	}

	conv, err := c.rt.store.GetConversation(ctx, sub.SessionKey)
	if err != nil {
		return fmt.Errorf("resolve conversation for %s: %w", sub.SessionKey, err)
	}

	run := &submissionRun{
		rt:   c.rt,
		sub:  sub,
		conv: conv,
		cfg:  cfg,
	}
	return run.drive(ctx)
}

// settle appends the submission_settled record and marks the submission
// settled. Two-phase settlement semantics arrive with the durable engine
// slice (HARNESS-3).
func (c *coordinator) settle(ctx context.Context, sub Submission, payload SettledPayload) error {
	// Use a fresh context bound to the store, not the (possibly cancelled)
	// run context: settlement must land once the outcome is known.
	if ctx.Err() != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
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
	if err := c.rt.store.SettleSubmission(ctx, sub.ID); err != nil {
		return fmt.Errorf("mark submission settled: %w", err)
	}
	return nil
}

// submissionRun is the session engine for one attempt: it owns the pi.Agent,
// authors canonical records from the event stream, and tracks turn
// correlation.
type submissionRun struct {
	rt   *Runtime
	sub  Submission
	conv Conversation
	cfg  AgentRuntimeConfig

	mu     sync.Mutex
	turnID string
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
// terminal result.
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

// consumeEvent authors canonical records from one agent event. Walking
// skeleton set: turn correlation, tool outcomes, completed assistant
// messages. Delta batching and tool-call starts join in the streaming slice
// (HARNESS-4).
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
