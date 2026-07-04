package harness_test

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// batchProvider is a scripted llm.LLMProvider that emits one parallel
// tool-call batch with unique call ids on its first call and a text answer
// afterwards. MockProvider stamps every call in one step with the same id,
// so it cannot express this shape.
type batchProvider struct {
	mu    sync.Mutex
	calls int
}

func (p *batchProvider) Name() string { return "mock" }

func (p *batchProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{}
}

func (p *batchProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setMeta func(int, map[string]string)) ([]llm.Message, error) {
		p.mu.Lock()
		call := p.calls
		p.calls++
		p.mu.Unlock()

		if call == 0 {
			batch := []llm.ToolCallContent{
				{CallID: "call_berlin", ToolName: "get_weather", Args: json.RawMessage(`{"city":"Berlin"}`)},
				{CallID: "call_tokyo", ToolName: "get_weather", Args: json.RawMessage(`{"city":"Tokyo"}`)},
			}
			msgs := req.Messages
			for _, tc := range batch {
				if err := emit(llm.ToolCallStartEvent{CallID: tc.CallID, ToolName: tc.ToolName, Args: tc.Args}); err != nil {
					return nil, err
				}
				if err := emit(llm.ToolCallEndEvent{CallID: tc.CallID}); err != nil {
					return nil, err
				}
				msgs = append(msgs, llm.Message{Role: "assistant", Content: tc})
			}
			if err := emit(llm.MessageEndEvent{}); err != nil {
				return nil, err
			}
			return msgs, nil
		}
		if err := emit(llm.TextDeltaEvent{Delta: "Berlin and Tokyo are both sunny."}); err != nil {
			return nil, err
		}
		if err := emit(llm.MessageEndEvent{}); err != nil {
			return nil, err
		}
		return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: "Berlin and Tokyo are both sunny."}}), nil
	})
}

// A parallel tool batch yields correctly correlated assistant_tool_call /
// tool_outcome pairs: every call id appears exactly once on each side.
func TestParallelToolBatchCorrelation(t *testing.T) {
	t.Parallel()
	rt := newTestRuntimeWithTools(t, &batchProvider{})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("Weather in Berlin and Tokyo?"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
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
	calls := map[string]string{}    // call id → city arg
	outcomes := map[string]string{} // call id → content
	for _, rec := range recs {
		switch rec.Kind {
		case harness.KindAssistantToolCall:
			var p harness.AssistantToolCallPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("decode tool call: %v", err)
			}
			if _, dup := calls[p.CallID]; dup {
				t.Fatalf("duplicate assistant_tool_call for call id %s", p.CallID)
			}
			calls[p.CallID] = string(p.Args)
		case harness.KindToolOutcome:
			var p harness.ToolOutcomePayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("decode tool outcome: %v", err)
			}
			if _, dup := outcomes[p.CallID]; dup {
				t.Fatalf("duplicate tool_outcome for call id %s", p.CallID)
			}
			outcomes[p.CallID] = p.Content
		}
	}
	if len(calls) != 2 || len(outcomes) != 2 {
		t.Fatalf("calls = %d, outcomes = %d, want 2 and 2", len(calls), len(outcomes))
	}
	for id := range calls {
		if _, ok := outcomes[id]; !ok {
			t.Fatalf("call id %s has no correlated tool_outcome", id)
		}
	}
}
