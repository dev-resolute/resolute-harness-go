package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// Runtime-level sentinel errors.
var (
	// ErrUnknownAgent reports a dispatch to an agent name with no registered
	// definition.
	ErrUnknownAgent = errors.New("unknown agent")
	// ErrRuntimeClosed reports an operation on a closed Runtime.
	ErrRuntimeClosed = errors.New("runtime is closed")
)

// Runtime is the composed harness: agent definitions, store, coordinator,
// and transport. Construct with NewRuntime, then Start; mount Handler
// wherever the app wants it.
type Runtime struct {
	agents map[string]AgentDefinition
	store  Store
	env    Env
	logger *slog.Logger

	wake chan struct{} // nudges the coordinator claim loop

	mu        sync.Mutex
	started   bool
	closed    bool
	cancel    context.CancelFunc
	appendSub chan struct{} // closed and replaced on every record append
	settleSub chan struct{} // closed and replaced on every settlement

	running sync.WaitGroup
}

// NewRuntime validates cfg and builds a Runtime. The Runtime is inert until
// Start.
func NewRuntime(cfg Config) (*Runtime, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	env := cfg.Env
	if env == nil {
		env = OSEnv()
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Runtime{
		agents:    cfg.Agents,
		store:     cfg.Store,
		env:       env,
		logger:    logger,
		wake:      make(chan struct{}, 1),
		appendSub: make(chan struct{}),
		settleSub: make(chan struct{}),
	}, nil
}

// Start launches the coordinator. The supplied ctx bounds the coordinator's
// lifetime alongside Close.
func (rt *Runtime) Start(ctx context.Context) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return ErrRuntimeClosed
	}
	if rt.started {
		return errors.New("runtime already started")
	}
	rt.started = true
	runCtx, cancel := context.WithCancel(ctx)
	rt.cancel = cancel

	coord := newCoordinator(rt)
	rt.running.Add(1)
	go func() {
		defer rt.running.Done()
		coord.loop(runCtx)
	}()
	return nil
}

// Close stops the coordinator and waits for in-flight work to wind down,
// bounded by a shutdown timeout. Idempotent.
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	if rt.closed {
		rt.mu.Unlock()
		return nil
	}
	rt.closed = true
	cancel := rt.cancel
	rt.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	done := make(chan struct{})
	go func() {
		rt.running.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-time.After(10 * time.Second):
		return errors.New("close: timed out waiting for in-flight work")
	}
}

// Dispatch admits one unit of work in-process — the same admission path the
// HTTP transport uses. It returns as soon as the submission is durable.
func (rt *Runtime) Dispatch(ctx context.Context, d Dispatch) (DispatchResult, error) {
	if _, ok := rt.agents[d.Agent]; !ok {
		return DispatchResult{}, fmt.Errorf("%w: %q", ErrUnknownAgent, d.Agent)
	}
	if d.Instance == "" {
		return DispatchResult{}, fmt.Errorf("%w: instance id is required", ErrInvalidDispatch)
	}
	if err := d.Message.Validate(); err != nil {
		return DispatchResult{}, err
	}

	key := SessionKey{Agent: d.Agent, Instance: d.Instance, Session: d.Session}
	if key.Session == "" {
		key.Session = "default"
	}
	submissionID := d.DispatchID
	if submissionID == "" {
		submissionID = newULID()
	}

	conv, created, err := rt.store.EnsureConversation(ctx, Conversation{
		ID:        newULID(),
		Key:       key,
		CreatedAt: time.Now(),
	})
	if err != nil {
		return DispatchResult{}, fmt.Errorf("ensure conversation: %w", err)
	}
	if created {
		rec := Record{
			RecordEnvelope: RecordEnvelope{
				ID:             newULID(),
				Kind:           KindConversationCreated,
				ConversationID: conv.ID,
				Session:        key.Session,
				Time:           time.Now(),
			},
			Payload: mustPayload(&ConversationCreatedPayload{
				Agent:    key.Agent,
				Instance: key.Instance,
				Session:  key.Session,
			}),
		}
		if err := rt.store.AppendRecords(ctx, conv.ID, []Record{rec}); err != nil {
			return DispatchResult{}, fmt.Errorf("append conversation_created: %w", err)
		}
		rt.notifyAppend()
	}

	sub, err := rt.store.AdmitSubmission(ctx, Submission{
		ID:             submissionID,
		SessionKey:     key,
		ConversationID: conv.ID,
		Status:         StatusQueued,
		Input:          d.Message,
		CreatedAt:      time.Now(),
	})
	if err != nil {
		return DispatchResult{}, fmt.Errorf("admit submission: %w", err)
	}

	// Nudge the claim loop without blocking.
	select {
	case rt.wake <- struct{}{}:
	default:
	}
	return DispatchResult{SubmissionID: sub.ID, ConversationID: sub.ConversationID}, nil
}

// Wait blocks until the submission settles and returns the terminal payload.
// It waits on the durable submission_settled record, not an in-process
// future, so it survives a Runtime restart over the same store.
func (rt *Runtime) Wait(ctx context.Context, submissionID string) (SettledPayload, error) {
	for {
		sub, err := rt.store.GetSubmission(ctx, submissionID)
		if err != nil {
			return SettledPayload{}, err
		}
		if sub.Status == StatusSettled {
			return rt.settledPayload(ctx, sub)
		}

		rt.mu.Lock()
		settled := rt.settleSub
		rt.mu.Unlock()
		select {
		case <-ctx.Done():
			return SettledPayload{}, ctx.Err()
		case <-settled:
		case <-time.After(250 * time.Millisecond):
			// Fallback poll: settlement may have landed via another process
			// over the same store.
		}
	}
}

// settledPayload reads the submission_settled record for sub.
func (rt *Runtime) settledPayload(ctx context.Context, sub Submission) (SettledPayload, error) {
	recs, err := rt.store.ReadRecords(ctx, sub.ConversationID, "")
	if err != nil {
		return SettledPayload{}, fmt.Errorf("read settled record: %w", err)
	}
	for i := len(recs) - 1; i >= 0; i-- {
		rec := recs[i]
		if rec.Kind == KindSubmissionSettled && rec.SubmissionID == sub.ID {
			var p SettledPayload
			if err := rec.DecodePayload(&p); err != nil {
				return SettledPayload{}, err
			}
			return p, nil
		}
	}
	return SettledPayload{}, fmt.Errorf("submission %s settled without a submission_settled record", sub.ID)
}

// Records reads the conversation log from afterID (exclusive; "" reads from
// the start) — the same replay the SSE transport serves.
func (rt *Runtime) Records(ctx context.Context, conversationID string, afterID string) ([]Record, error) {
	return rt.store.ReadRecords(ctx, conversationID, afterID)
}

// Handler returns the HTTP transport (ADR-0004). Auth and other middleware
// are the app's concern; mount this wherever the app wants.
func (rt *Runtime) Handler() http.Handler {
	return newTransport(rt)
}

// notifyAppend wakes readers blocked on new records.
func (rt *Runtime) notifyAppend() {
	rt.mu.Lock()
	close(rt.appendSub)
	rt.appendSub = make(chan struct{})
	rt.mu.Unlock()
}

// notifySettled wakes Wait callers.
func (rt *Runtime) notifySettled() {
	rt.mu.Lock()
	close(rt.settleSub)
	rt.settleSub = make(chan struct{})
	rt.mu.Unlock()
}
