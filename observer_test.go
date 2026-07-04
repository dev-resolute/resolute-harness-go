package harness_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// recordingObserver captures events with a lock (observers must be cheap
// and synchronous; the engine may call from several goroutines over time).
type recordingObserver struct {
	mu     sync.Mutex
	events []harness.HarnessEvent
}

func (o *recordingObserver) observe(ev harness.HarnessEvent) {
	o.mu.Lock()
	o.events = append(o.events, ev)
	o.mu.Unlock()
}

func (o *recordingObserver) snapshot() []harness.HarnessEvent {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make([]harness.HarnessEvent, len(o.events))
	copy(out, o.events)
	return out
}

func observedKinds(events []harness.HarnessEvent) []string {
	var out []string
	for _, ev := range events {
		switch ev.(type) {
		case harness.SubmissionAdmittedEvent:
			out = append(out, "admitted")
		case harness.SubmissionClaimedEvent:
			out = append(out, "claimed")
		case harness.AttemptStartedEvent:
			out = append(out, "attempt")
		case harness.OperationStartedEvent:
			out = append(out, "op_start")
		case harness.TurnStartedEvent:
			out = append(out, "turn_start")
		case harness.ToolCallStartedEvent:
			out = append(out, "tool_start")
		case harness.ToolCallEndedEvent:
			out = append(out, "tool_end")
		case harness.TurnEndedEvent:
			out = append(out, "turn_end")
		case harness.DeltaEvent:
			out = append(out, "delta")
		case harness.OperationEndedEvent:
			out = append(out, "op_end")
		case harness.CompactionEvent:
			out = append(out, "compaction")
		case harness.RecoveryEvent:
			out = append(out, "recovery")
		case harness.SubmissionSettledEvent:
			out = append(out, "settled")
		}
	}
	return out
}

func assertOrderedSubsequence(t *testing.T, got []string, want []string) {
	t.Helper()
	i := 0
	for _, k := range got {
		if i < len(want) && k == want[i] {
			i++
		}
	}
	if i != len(want) {
		t.Fatalf("observed %v missing ordered subsequence %v", got, want)
	}
}

// A recording Observer captures the full lifecycle of a multi-turn tool run
// in order, with correlation ids matching the canonical records.
func TestObserverCapturesLifecycleWithCorrelation(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondToolCall("get_weather", json.RawMessage(`{"city":"Berlin"}`)).Add()
	provider.OnAny().RespondText("sunny").Add()

	obs := &recordingObserver{}
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						Tools:         []pi.RegisteredTool{weatherTool()},
					}, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		Observers:     []harness.Observer{obs.observe},
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("weather in Berlin?"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if s, err := rt.Wait(ctx, res.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("settled %+v (%v), want success", s, err)
	}

	events := obs.snapshot()
	assertOrderedSubsequence(t, observedKinds(events), []string{
		"admitted", "claimed", "attempt", "op_start", "turn_start",
		"tool_start", "tool_end", "turn_end", "op_end", "settled",
	})

	// Correlation ids line up with the canonical records of the same run.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	recTurnIDs := map[string]bool{}
	attemptID := ""
	for _, rec := range recs {
		if rec.Kind == harness.KindToolOutcome {
			recTurnIDs[rec.TurnID] = true
			attemptID = rec.AttemptID
		}
	}
	for _, ev := range events {
		te, ok := ev.(harness.ToolCallEndedEvent)
		if !ok {
			continue
		}
		if te.SubmissionID != res.SubmissionID || te.AttemptID != attemptID || !recTurnIDs[te.TurnID] {
			t.Fatalf("tool event correlation %+v does not match records (attempt %s, turns %v)", te.Correlation, attemptID, recTurnIDs)
		}
	}
}

// A panicking Observer never affects execution.
func TestPanickingObserverDoesNotFailRun(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("survived").Add()
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model: "mock/test-model", ContextWindow: 200_000,
						Providers: []llm.LLMProvider{provider},
					}, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		Observers: []harness.Observer{func(harness.HarnessEvent) {
			panic("bad metrics sink")
		}},
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("hello"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled %+v, want success despite the panicking observer", settled)
	}
}

type ctxKey string

// A ctx-enriching interceptor's value is visible in nested boundaries
// (attempt → turn → tool), and multiple interceptors compose in
// registration order.
func TestInterceptorCtxPropagationAndOrder(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondToolCall("get_weather", json.RawMessage(`{"city":"Berlin"}`)).Add()
	provider.OnAny().RespondText("sunny").Add()

	var mu sync.Mutex
	sawValue := map[harness.OpKind]bool{}
	var order []string

	outer := func(ctx context.Context, op harness.OpInfo, next func(context.Context) error) error {
		mu.Lock()
		order = append(order, "outer:"+string(op.Kind))
		mu.Unlock()
		if op.Kind == harness.OpAttempt {
			ctx = context.WithValue(ctx, ctxKey("trace"), "trace-123")
		}
		if v, _ := ctx.Value(ctxKey("trace")).(string); v == "trace-123" {
			mu.Lock()
			sawValue[op.Kind] = true
			mu.Unlock()
		}
		return next(ctx)
	}
	inner := func(ctx context.Context, op harness.OpInfo, next func(context.Context) error) error {
		mu.Lock()
		order = append(order, "inner:"+string(op.Kind))
		mu.Unlock()
		return next(ctx)
	}

	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model: "mock/test-model", ContextWindow: 200_000,
						Providers: []llm.LLMProvider{provider},
						Tools:     []pi.RegisteredTool{weatherTool()},
					}, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		Interceptors:  []harness.Interceptor{outer, inner},
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("weather?"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if s, err := rt.Wait(ctx, res.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("settled %+v (%v), want success", s, err)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, kind := range []harness.OpKind{harness.OpAttempt, harness.OpOperation, harness.OpTurn, harness.OpTool} {
		if !sawValue[kind] {
			t.Fatalf("attempt-injected ctx value not visible at %s boundary (saw %v)", kind, sawValue)
		}
	}
	// Registration order: outer precedes inner at every boundary.
	for i := 0; i+1 < len(order); i++ {
		if strings.HasPrefix(order[i], "outer:") {
			kind := strings.TrimPrefix(order[i], "outer:")
			if order[i+1] != "inner:"+kind {
				t.Fatalf("interceptors out of order at %d: %v", i, order[i:i+2])
			}
		}
	}
}

// An aborting interceptor surfaces as a structured operational failure.
func TestAbortingInterceptorFailsSubmission(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock") // must never be called
	abort := func(ctx context.Context, op harness.OpInfo, next func(context.Context) error) error {
		if op.Kind == harness.OpAttempt {
			return errors.New("aborted by policy interceptor")
		}
		return next(ctx)
	}
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model: "mock/test-model", ContextWindow: 200_000,
						Providers: []llm.LLMProvider{provider},
					}, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		Interceptors:  []harness.Interceptor{abort},
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("never runs"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || !strings.Contains(settled.Error, "aborted by policy interceptor") {
		t.Fatalf("settled %+v, want failed with the interceptor's error", settled)
	}
	if provider.Called() != 0 {
		t.Fatalf("provider called %d times under an attempt-aborting interceptor, want 0", provider.Called())
	}
}

// Every boundary named in the design is covered: a compaction-and-retry
// scenario shows the Observer the recovery events too.
func TestObserverSeesRecoveryEvents(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnPrompt(mock.LastUser("build up history")).
		RespondText(strings.Repeat(longAnswer, 4)).Add()
	provider.OnPrompt(mock.LastUser("overflow now")).Error(overflowErr).Add()
	provider.OnAny().RespondText("## Goal\nsummary.").Add() // summarization
	provider.OnAny().RespondText("done after recovery").Add()

	obs := &recordingObserver{}
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model: "mock/test-model", ContextWindow: 8_000,
						Providers:        []llm.LLMProvider{provider},
						KeepRecentTokens: 60,
					}, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		Observers:     []harness.Observer{obs.observe},
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	first, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("build up history"),
	})
	if err != nil {
		t.Fatalf("Dispatch 1: %v", err)
	}
	if s, err := rt.Wait(ctx, first.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("first settled %+v (%v)", s, err)
	}
	second, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("overflow now"),
	})
	if err != nil {
		t.Fatalf("Dispatch 2: %v", err)
	}
	if s, err := rt.Wait(ctx, second.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("second settled %+v (%v), want recovery success", s, err)
	}

	kinds := observedKinds(obs.snapshot())
	assertOrderedSubsequence(t, kinds, []string{"recovery", "compaction", "settled"})
}
