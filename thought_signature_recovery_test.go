package harness_test

import (
	"bytes"
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// signatureProvider emits one signed tool call on its first call and a text
// answer afterwards, recording every request it receives (HARNESS-11).
type signatureProvider struct {
	signature []byte

	mu    sync.Mutex
	calls int
	reqs  []llm.LLMRequest
}

func (p *signatureProvider) Name() string { return "mock" }

func (p *signatureProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{}
}

func (p *signatureProvider) requests() []llm.LLMRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]llm.LLMRequest, len(p.reqs))
	copy(out, p.reqs)
	return out
}

func (p *signatureProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setMeta func(int, map[string]string)) ([]llm.Message, error) {
		p.mu.Lock()
		call := p.calls
		p.calls++
		p.reqs = append(p.reqs, req)
		p.mu.Unlock()

		if call == 0 {
			tc := llm.ToolCallContent{
				CallID:           "call_sig",
				ToolName:         "get_weather",
				Args:             json.RawMessage(`{"city":"Berlin"}`),
				ThoughtSignature: p.signature,
			}
			if err := emit(llm.ToolCallStartEvent{CallID: tc.CallID, ToolName: tc.ToolName, Args: tc.Args, ThoughtSignature: tc.ThoughtSignature}); err != nil {
				return nil, err
			}
			if err := emit(llm.ToolCallEndEvent{CallID: tc.CallID}); err != nil {
				return nil, err
			}
			if err := emit(llm.MessageEndEvent{}); err != nil {
				return nil, err
			}
			return append(req.Messages, llm.Message{Role: "assistant", Content: tc}), nil
		}
		if err := emit(llm.TextDeltaEvent{Delta: "sunny"}); err != nil {
			return nil, err
		}
		if err := emit(llm.MessageEndEvent{}); err != nil {
			return nil, err
		}
		return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: "sunny"}}), nil
	})
}

// The engine must persist the provider thought signature on the durable
// assistant_tool_call record (HARNESS-11: capture half).
func TestToolCallRecordPersistsThoughtSignature(t *testing.T) {
	t.Parallel()
	sig := []byte("opaque-signature-bytes")
	provider := &signatureProvider{signature: sig}
	rt := newTestRuntimeWithTools(t, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("Weather in Berlin?"),
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
	var found bool
	for _, rec := range recs {
		if rec.Kind != harness.KindAssistantToolCall {
			continue
		}
		found = true
		var p harness.AssistantToolCallPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if !bytes.Equal(p.ThoughtSignature, sig) {
			t.Errorf("assistant_tool_call ThoughtSignature = %q, want %q", p.ThoughtSignature, sig)
		}
	}
	if !found {
		t.Fatal("no assistant_tool_call record found")
	}
}

// Crash boundary: the tool call (with its signature) is durable, the tool
// result and the rest of the turn are not. Turn recovery must replay the
// persisted tool call WITH its thought signature, or Gemini 3 rejects every
// recovery re-prompt with 400 INVALID_ARGUMENT (HARNESS-11: replay half).
func TestCrashMidTurnRecoveryReplaysThoughtSignature(t *testing.T) {
	t.Parallel()
	sig := []byte("opaque-signature-bytes")
	store := memory.New()
	provider := &signatureProvider{signature: sig}
	// The recovery prompt is this provider's first call: it emits a fresh
	// signed tool call, the tool runs, and the follow-up call answers.
	conv, sub := seededSubmission(t, store, "weather please")
	claimed := claimSeeded(t, store, sub)
	ctx := context.Background()
	if err := store.StartAttempt(ctx, harness.Attempt{
		ID: claimed.AttemptID, SubmissionID: sub.ID, OwnerID: "dead-owner", StartedAt: time.Now(),
	}); err != nil {
		t.Fatalf("StartAttempt: %v", err)
	}

	// Seed the dead attempt's durable records: the user input, and a signed
	// tool call whose result never landed (crash while the tool ran).
	inputRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0SEEDREC000000000000001",
			Kind:           harness.KindUserMessage,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub.ID,
			AttemptID:      claimed.AttemptID,
			Time:           time.Now(),
		},
	}
	inputRec.Payload, _ = json.Marshal(map[string]string{"body": "weather please"})
	toolCallRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0SEEDREC000000000000002",
			Kind:           harness.KindAssistantToolCall,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub.ID,
			AttemptID:      claimed.AttemptID,
			Time:           time.Now(),
		},
	}
	toolCallRec.Payload, _ = json.Marshal(harness.AssistantToolCallPayload{
		CallID:           "call_dead",
		ToolName:         "get_weather",
		Args:             json.RawMessage(`{"city":"Berlin"}`),
		ThoughtSignature: sig,
	})
	toolOutcomeRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0SEEDREC000000000000003",
			Kind:           harness.KindToolOutcome,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub.ID,
			AttemptID:      claimed.AttemptID,
			Time:           time.Now(),
		},
	}
	toolOutcomeRec.Payload, _ = json.Marshal(harness.ToolOutcomePayload{
		CallID:   "call_dead",
		ToolName: "get_weather",
		Content:  "sunny, 21C",
	})
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{inputRec, toolCallRec, toolOutcomeRec}); err != nil {
		t.Fatalf("AppendRecords: %v", err)
	}

	rt := startRuntime(t, harness.Config{
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
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
	})
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	// The recovery prompt must replay the dead attempt's tool call with its
	// signature intact.
	var replayed bool
	for _, req := range provider.requests() {
		for _, m := range req.Messages {
			tc, ok := m.Content.(llm.ToolCallContent)
			if !ok || tc.CallID != "call_dead" {
				continue
			}
			replayed = true
			if !bytes.Equal(tc.ThoughtSignature, sig) {
				t.Errorf("replayed tool call ThoughtSignature = %q, want %q", tc.ThoughtSignature, sig)
			}
		}
	}
	if !replayed {
		t.Fatal("recovery prompt did not replay the persisted tool call")
	}
}
