package harness_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// streamerTool is a streaming tool whose ExecuteStream emits one partial
// result before returning its final result, driving a pi.ToolUpdateEvent
// through the engine's event consumption.
func streamerTool() pi.RegisteredTool {
	return pi.NewTool(pi.Tool[struct{}]{
		Name: "streamer",
		ExecuteStream: func(ctx context.Context, _ struct{}, emit func(pi.ToolResult)) (pi.ToolResult, error) {
			emit(pi.ToolResult{Content: "partial"})
			return pi.ToolResult{Content: "done"}, nil
		},
	})
}

// preExistingRecordKinds is the full v1 record kind set (record.go:18-28).
// ToolCallUpdatedEvent must never author a record, so every record kind
// observed after a run must be a member of this set.
func preExistingRecordKinds() map[harness.RecordKind]bool {
	return map[harness.RecordKind]bool{
		harness.KindConversationCreated:       true,
		harness.KindUserMessage:               true,
		harness.KindSignal:                    true,
		harness.KindAssistantMessageStarted:   true,
		harness.KindAssistantTextDelta:        true,
		harness.KindAssistantThinkingDelta:    true,
		harness.KindAssistantToolCall:         true,
		harness.KindToolOutcome:               true,
		harness.KindAssistantMessageCompleted: true,
		harness.KindCompaction:                true,
		harness.KindSubmissionSettled:         true,
	}
}

// A streaming tool's partial results surface as ephemeral
// ToolCallUpdatedEvents between ToolCallStartedEvent and ToolCallEndedEvent,
// and leave no durable trace: every record kind after the run is a
// pre-existing kind, none of them carrying the partial content.
func TestToolCallUpdatedEventObservedNotRecorded(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondToolCall("streamer", json.RawMessage(`{}`)).Add()
	provider.OnAny().RespondText("done streaming").Add()

	obs := &recordingObserver{}
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						Tools:         []pi.RegisteredTool{streamerTool()},
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
		Agent: "support", Instance: "acme", Message: harness.UserMessage("stream it"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if s, err := rt.Wait(ctx, res.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("settled %+v (%v), want success", s, err)
	}

	events := obs.snapshot()

	// Locate ToolCallStartedEvent / ToolCallEndedEvent for the streamer call
	// and assert at least one ToolCallUpdatedEvent for the same CallID lands
	// strictly between them, carrying the partial content.
	var startIdx, endIdx = -1, -1
	var callID string
	for i, ev := range events {
		if se, ok := ev.(harness.ToolCallStartedEvent); ok && se.ToolName == "streamer" {
			startIdx = i
			callID = se.CallID
		}
		if ee, ok := ev.(harness.ToolCallEndedEvent); ok && ee.ToolName == "streamer" {
			endIdx = i
		}
	}
	if startIdx == -1 || endIdx == -1 || endIdx <= startIdx {
		t.Fatalf("did not observe start/end bracket for streamer tool call: start=%d end=%d", startIdx, endIdx)
	}

	var updates []harness.ToolCallUpdatedEvent
	for i := startIdx + 1; i < endIdx; i++ {
		if ue, ok := events[i].(harness.ToolCallUpdatedEvent); ok {
			updates = append(updates, ue)
		}
	}
	if len(updates) == 0 {
		t.Fatalf("no ToolCallUpdatedEvent observed between start (%d) and end (%d): events=%v", startIdx, endIdx, events)
	}
	for _, u := range updates {
		if u.CallID != callID {
			t.Errorf("ToolCallUpdatedEvent.CallID = %q, want %q", u.CallID, callID)
		}
		if u.ToolName != "streamer" {
			t.Errorf("ToolCallUpdatedEvent.ToolName = %q, want %q", u.ToolName, "streamer")
		}
	}
	if updates[0].Result.Content != "partial" {
		t.Errorf("ToolCallUpdatedEvent.Result.Content = %q, want %q", updates[0].Result.Content, "partial")
	}
	// Correlation matches the bracketing started/ended events.
	se := events[startIdx].(harness.ToolCallStartedEvent)
	if updates[0].Correlation != se.Correlation {
		t.Errorf("ToolCallUpdatedEvent correlation = %+v, want %+v", updates[0].Correlation, se.Correlation)
	}

	// No new/unknown record kinds: updates left no durable trace.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	known := preExistingRecordKinds()
	for _, rec := range recs {
		if !known[rec.Kind] {
			t.Errorf("record %s has kind %q, not one of the pre-existing record.go kinds", rec.ID, rec.Kind)
		}
		if rec.Kind == harness.KindToolOutcome {
			var p harness.ToolOutcomePayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("decode tool outcome: %v", err)
			}
			if p.Content == "partial" {
				t.Errorf("partial result leaked into durable tool_outcome record: %+v", p)
			}
		}
	}
}
