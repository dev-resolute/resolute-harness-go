package harness_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// weatherTool is a deterministic tool for scripting multi-turn runs.
func weatherTool() pi.RegisteredTool {
	return pi.NewTool(pi.Tool[struct {
		City string `json:"city"`
	}]{
		Name:        "get_weather",
		Description: "Look up the weather for a city",
		Execute: func(ctx context.Context, params struct {
			City string `json:"city"`
		}) (pi.ToolResult, error) {
			return pi.ToolResult{Content: "sunny, 24C in " + params.City}, nil
		},
	})
}

func TestMultiTurnToolConversationAndResume(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// Turn 1: the model requests the weather tool.
	provider.OnPrompt(mock.LastUser("What's the weather in Berlin?")).
		RespondToolCall("get_weather", json.RawMessage(`{"city":"Berlin"}`)).Add()
	// Turn 2: after the tool result, it answers.
	provider.OnToolResult("get_weather", mock.Predicate(func([]llm.Message) bool { return true })).
		RespondText("It is sunny and 24C in Berlin.").Add()
	// Second prompt: answers only if the prior exchange is in context,
	// proving the projection adapter serves resumed conversations.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "sunny and 24C") {
				return true
			}
		}
		return false
	})).RespondText("As I said, sunny and 24C.").Add()

	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						SystemPrompt:  "You are a weather assistant.",
						Tools:         []pi.RegisteredTool{weatherTool()},
					}, nil
				},
			},
		},
		Store: memory.New(),
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

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// First prompt: tool-calling round trip.
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("What's the weather in Berlin?"),
	})
	if err != nil {
		t.Fatalf("Dispatch 1: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait 1: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("first prompt settled %q (error %q), want %q", settled.Status, settled.Error, harness.SettledSucceeded)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	assertKindSubsequence(t, recs, []harness.RecordKind{
		harness.KindUserMessage,
		harness.KindAssistantToolCall,
		harness.KindToolOutcome,
		harness.KindAssistantMessageCompleted,
		harness.KindSubmissionSettled,
	})

	// Tool call and outcome share a call id.
	var call harness.AssistantToolCallPayload
	var outcome harness.ToolOutcomePayload
	for _, rec := range recs {
		switch rec.Kind {
		case harness.KindAssistantToolCall:
			if err := rec.DecodePayload(&call); err != nil {
				t.Fatalf("decode tool call: %v", err)
			}
		case harness.KindToolOutcome:
			if err := rec.DecodePayload(&outcome); err != nil {
				t.Fatalf("decode tool outcome: %v", err)
			}
		}
	}
	if call.CallID == "" || call.CallID != outcome.CallID {
		t.Fatalf("tool call/outcome call ids = %q / %q, want matching non-empty", call.CallID, outcome.CallID)
	}
	if outcome.ToolName != "get_weather" || !strings.Contains(outcome.Content, "sunny, 24C in Berlin") {
		t.Fatalf("tool outcome = %+v, want get_weather sunny content", outcome)
	}

	// Second prompt on the same conversation resumes with prior context via
	// the projection adapter (the mock only answers when it sees the earlier
	// exchange in its request).
	res2, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("And now?"),
	})
	if err != nil {
		t.Fatalf("Dispatch 2: %v", err)
	}
	if res2.ConversationID != res.ConversationID {
		t.Fatalf("second dispatch conversation = %q, want same as first %q", res2.ConversationID, res.ConversationID)
	}
	settled2, err := rt.Wait(ctx, res2.SubmissionID)
	if err != nil {
		t.Fatalf("Wait 2: %v", err)
	}
	if settled2.Status != harness.SettledSucceeded {
		t.Fatalf("second prompt settled %q (error %q), want %q — projection likely served incomplete context", settled2.Status, settled2.Error, harness.SettledSucceeded)
	}
}

func TestDispatchIdempotentReplay(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("only once").Add()
	rt := newTestRuntime(t, provider)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	d := harness.Dispatch{
		Agent:      "support",
		Instance:   "acme",
		DispatchID: "webhook-delivery-42",
		Message:    harness.UserMessage("hello"),
	}
	first, err := rt.Dispatch(ctx, d)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	replay, err := rt.Dispatch(ctx, d)
	if err != nil {
		t.Fatalf("replayed Dispatch: %v", err)
	}
	if replay != first {
		t.Fatalf("replayed dispatch = %+v, want original %+v", replay, first)
	}

	d.Message = harness.UserMessage("something different")
	if _, err := rt.Dispatch(ctx, d); err == nil {
		t.Fatal("mutated payload on same dispatch id admitted, want conflict error")
	}
}
