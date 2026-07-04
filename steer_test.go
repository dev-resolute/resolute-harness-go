package harness_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// slowTool gives the test a window to steer while the run executes tools.
func slowTool(d time.Duration) pi.RegisteredTool {
	return pi.NewTool(pi.Tool[struct{}]{
		Name:        "slow_lookup",
		Description: "A lookup that takes a while",
		Execute: func(ctx context.Context, _ struct{}) (pi.ToolResult, error) {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return pi.ToolResult{}, ctx.Err()
			}
			return pi.ToolResult{Content: "lookup finished"}, nil
		},
	})
}

func steerRuntime(t *testing.T, provider llm.LLMProvider, tool pi.RegisteredTool) *harness.Runtime {
	t.Helper()
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					cfg := harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
					}
					if tool != nil {
						cfg.Tools = []pi.RegisteredTool{tool}
					}
					return cfg, nil
				},
			},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return rt
}

// Steer during a multi-turn run is drained at the post-tool-batch seam,
// visibly alters the next turn, and appears as a record in the stream.
func TestSteerAltersLiveRunOverHTTP(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondToolCall("slow_lookup", json.RawMessage(`{}`)).Add()
	// The post-steer turn only matches when the steered message is present.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "focus on Tokyo") {
				return true
			}
		}
		return false
	})).RespondText("Understood, focusing on Tokyo.").Add()

	rt := steerRuntime(t, provider, slowTool(500*time.Millisecond))
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", `{"kind":"user","body":"look something up"}`, http.StatusAccepted, &res)

	// Wait until the tool batch is running (the tool-call record is durable),
	// then steer while the slow tool holds the turn open.
	waitForRecordKindVia(t, rt, res.ConversationID, harness.KindAssistantToolCall)
	resp, err := http.Post(server.URL+"/agents/support/acme/steer", "application/json",
		strings.NewReader(`{"body":"focus on Tokyo instead"}`))
	if err != nil {
		t.Fatalf("POST steer: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("steer status = %d, want 200", resp.StatusCode)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded (mock only matches when the steer reached the model)", settled)
	}

	// The injection is visible in the canonical stream.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	found := false
	for _, rec := range recs {
		if rec.Kind != harness.KindUserMessage {
			continue
		}
		var p harness.UserMessagePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("decode user message: %v", err)
		}
		if strings.Contains(p.Body, "focus on Tokyo") {
			found = true
		}
	}
	if !found {
		t.Fatal("steer injection not visible as a record in the stream")
	}
}

// FollowUp queued during a text-only turn produces the follow-up exchange
// within the same submission.
func TestFollowUpWithinSameSubmission(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().Delay(500 * time.Millisecond).RespondText("first answer").Add()
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "also say goodbye") {
				return true
			}
		}
		return false
	})).RespondText("Goodbye!").Add()

	rt := steerRuntime(t, provider, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("say hello"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	// Queue the follow-up while the first (delayed) turn is in flight;
	// retry until the run registers as live.
	followUpDeadline := time.After(5 * time.Second)
	for {
		err := rt.FollowUp(ctx, harness.SteerRequest{
			Agent: "support", Instance: "acme", Body: "also say goodbye",
		})
		if err == nil {
			break
		}
		if !errors.Is(err, harness.ErrNoRunInFlight) {
			t.Fatalf("FollowUp: %v", err)
		}
		select {
		case <-followUpDeadline:
			t.Fatal("run never became steerable")
		case <-time.After(10 * time.Millisecond):
		}
	}

	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	assistants := 0
	for _, rec := range recs {
		if rec.Kind == harness.KindAssistantMessageCompleted && rec.SubmissionID == res.SubmissionID {
			assistants++
		}
	}
	if assistants != 2 {
		t.Fatalf("assistant messages in submission = %d, want 2 (answer + follow-up)", assistants)
	}
	if n := countKind(recs, harness.KindSubmissionSettled); n != 1 {
		t.Fatalf("settled records = %d, want 1 (one submission)", n)
	}
}

// Steer with no run in flight is a structured error and persists nothing.
func TestSteerNoRunInFlight(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	rt := steerRuntime(t, provider, nil)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	// In-process surface.
	err := rt.Steer(context.Background(), harness.SteerRequest{
		Agent: "support", Instance: "acme", Body: "too late",
	})
	if !errors.Is(err, harness.ErrNoRunInFlight) {
		t.Fatalf("Steer error = %v, want ErrNoRunInFlight", err)
	}

	// HTTP surface: conflict-shaped, nothing persisted.
	resp, err := http.Post(server.URL+"/agents/support/acme/steer", "application/json",
		strings.NewReader(`{"body":"too late"}`))
	if err != nil {
		t.Fatalf("POST steer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("steer status = %d, want 409", resp.StatusCode)
	}

	// Unknown agent is 404-shaped.
	resp2, err := http.Post(server.URL+"/agents/nope/acme/steer", "application/json",
		strings.NewReader(`{"body":"hello"}`))
	if err != nil {
		t.Fatalf("POST steer unknown agent: %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown-agent steer status = %d, want 404", resp2.StatusCode)
	}
}

// waitForRecordKindVia polls the Runtime read surface for a record kind.
func waitForRecordKindVia(t *testing.T, rt *harness.Runtime, conversationID string, kind harness.RecordKind) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		recs, err := rt.Records(context.Background(), conversationID, "")
		if err == nil && countKind(recs, kind) > 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("conversation %s never grew a %s record", conversationID, kind)
		case <-time.After(10 * time.Millisecond):
		}
	}
}
