package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// Runtime-level sentinel errors.
var (
	// ErrUnknownAgent reports a dispatch to an agent name with no registered
	// definition.
	ErrUnknownAgent = errors.New("unknown agent")
	// ErrRuntimeClosed reports an operation on a closed Runtime.
	ErrRuntimeClosed = errors.New("runtime is closed")
	// ErrNoRunInFlight reports a steer or follow-up aimed at a session with
	// no live run. Steering is live-only in v1 (ADR-0004); nothing is
	// persisted.
	ErrNoRunInFlight = errors.New("no run in flight for the session")
)

// Runtime is the composed harness: agent definitions, store, coordinator,
// and transport. Construct with NewRuntime, then Start; mount Handler
// wherever the app wants it.
type Runtime struct {
	agents map[string]AgentDefinition
	store  Store
	env    Env
	logger *slog.Logger

	claimInterval      time.Duration
	leaseDuration      time.Duration
	deltaFlushBytes    int
	deltaFlushInterval time.Duration

	wake chan struct{} // nudges the coordinator claim loop

	liveMu   sync.Mutex
	liveRuns map[string]*pi.Agent // session key → in-flight agent

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
	claimInterval := cfg.ClaimInterval
	if claimInterval <= 0 {
		claimInterval = defaultClaimInterval
	}
	leaseDuration := cfg.LeaseDuration
	if leaseDuration <= 0 {
		leaseDuration = defaultLeaseDuration
	}
	deltaFlushBytes := cfg.DeltaFlushBytes
	if deltaFlushBytes <= 0 {
		deltaFlushBytes = defaultDeltaFlushBytes
	}
	deltaFlushInterval := cfg.DeltaFlushInterval
	if deltaFlushInterval <= 0 {
		deltaFlushInterval = defaultDeltaFlushInterval
	}
	return &Runtime{
		agents:             cfg.Agents,
		store:              cfg.Store,
		env:                env,
		logger:             logger,
		claimInterval:      claimInterval,
		leaseDuration:      leaseDuration,
		deltaFlushBytes:    deltaFlushBytes,
		deltaFlushInterval: deltaFlushInterval,
		wake:               make(chan struct{}, 1),
		liveRuns:           make(map[string]*pi.Agent),
		appendSub:          make(chan struct{}),
		settleSub:          make(chan struct{}),
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

// SteerRequest addresses a live run for Steer and FollowUp: the session key
// fields plus the message body.
type SteerRequest struct {
	Agent    string
	Instance InstanceID
	Session  string // empty means "default"
	Body     string
}

// Steer injects a message into the session's in-flight run at agent-core's
// next safe point (post-tool-batch). Live-only passthrough in v1: with no
// run in flight it returns ErrNoRunInFlight and persists nothing.
func (rt *Runtime) Steer(ctx context.Context, req SteerRequest) error {
	agent, err := rt.liveRun(req)
	if err != nil {
		return err
	}
	return agent.Steer(ctx, pi.NewText("user", req.Body))
}

// FollowUp enqueues a message for after the in-flight prompt's current
// exchange, producing the follow-up exchange within the same submission.
// Live-only, like Steer.
func (rt *Runtime) FollowUp(ctx context.Context, req SteerRequest) error {
	agent, err := rt.liveRun(req)
	if err != nil {
		return err
	}
	return agent.FollowUp(ctx, pi.NewText("user", req.Body))
}

func (rt *Runtime) liveRun(req SteerRequest) (*pi.Agent, error) {
	if _, ok := rt.agents[req.Agent]; !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownAgent, req.Agent)
	}
	key := SessionKey{Agent: req.Agent, Instance: req.Instance, Session: req.Session}
	if key.Session == "" {
		key.Session = "default"
	}
	rt.liveMu.Lock()
	agent, ok := rt.liveRuns[key.String()]
	rt.liveMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNoRunInFlight, key)
	}
	return agent, nil
}

// registerLiveRun makes the session's in-flight agent steerable.
func (rt *Runtime) registerLiveRun(key SessionKey, agent *pi.Agent) {
	rt.liveMu.Lock()
	rt.liveRuns[key.String()] = agent
	rt.liveMu.Unlock()
}

func (rt *Runtime) unregisterLiveRun(key SessionKey) {
	rt.liveMu.Lock()
	delete(rt.liveRuns, key.String())
	rt.liveMu.Unlock()
}

// appendWait returns a channel closed on the next record append — the live
// tail's wake signal.
func (rt *Runtime) appendWait() <-chan struct{} {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.appendSub
}

// sessionBusy reports whether the session has unsettled work — the live
// tail stays open while it does.
func (rt *Runtime) sessionBusy(ctx context.Context, key SessionKey) (bool, error) {
	for _, status := range []SubmissionStatus{StatusQueued, StatusRunning, StatusTerminalizing} {
		subs, err := rt.store.ListByStatus(ctx, status)
		if err != nil {
			return false, err
		}
		for _, sub := range subs {
			if sub.SessionKey == key {
				return true, nil
			}
		}
	}
	return false, nil
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
