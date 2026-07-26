package harness_test

import (
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

// danglingToolCallMessage mirrors the HARNESS-14 synthesized outcome string
// (engine.go); duplicated here (not imported) so the test pins the exact
// byte-exact wording as an independent assertion.
const danglingToolCallMessage = "Tool call was interrupted before a result was recorded (the run was recovered). Re-issue the tool call if it is still needed."

// recordingTextProvider answers every prompt with plain text and records
// every request it receives, so the test can inspect exactly what the
// recovery re-prompt sent to the model.
type recordingTextProvider struct {
	mu   sync.Mutex
	reqs []llm.LLMRequest
}

func (p *recordingTextProvider) Name() string { return "mock" }

func (p *recordingTextProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{}
}

func (p *recordingTextProvider) requests() []llm.LLMRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]llm.LLMRequest, len(p.reqs))
	copy(out, p.reqs)
	return out
}

func (p *recordingTextProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setMeta func(int, map[string]string)) ([]llm.Message, error) {
		p.mu.Lock()
		p.reqs = append(p.reqs, req)
		p.mu.Unlock()

		if err := emit(llm.TextDeltaEvent{Delta: "done"}); err != nil {
			return nil, err
		}
		if err := emit(llm.MessageEndEvent{}); err != nil {
			return nil, err
		}
		return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: "done"}}), nil
	})
}

// seedDanglingScenario seeds a dead attempt's durable records: the user
// input, and an assistant_tool_call whose result never landed (crash while
// the tool ran). When withOutcome is true, a matching tool_outcome record is
// also seeded — the normal-path / reconciliation-is-a-no-op scenario.
func seedDanglingScenario(t *testing.T, store harness.Store, withOutcome bool) (harness.Conversation, harness.Submission) {
	t.Helper()
	conv, sub := seededSubmission(t, store, "look up the weather")
	claimed := claimSeeded(t, store, sub)
	ctx := context.Background()
	if err := store.StartAttempt(ctx, harness.Attempt{
		ID: claimed.AttemptID, SubmissionID: sub.ID, OwnerID: "dead-owner", StartedAt: time.Now(),
	}); err != nil {
		t.Fatalf("StartAttempt: %v", err)
	}

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
	inputRec.Payload, _ = json.Marshal(map[string]string{"body": "look up the weather"})
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
		CallID:   "c1",
		ToolName: "get_weather",
		Args:     json.RawMessage(`{"city":"Berlin"}`),
	})
	recs := []harness.Record{inputRec, toolCallRec}
	if withOutcome {
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
			CallID:   "c1",
			ToolName: "get_weather",
			Content:  "sunny, 21C",
		})
		recs = append(recs, toolOutcomeRec)
	}
	if err := store.AppendRecords(ctx, conv.ID, recs); err != nil {
		t.Fatalf("AppendRecords: %v", err)
	}
	return conv, sub
}

func startDanglingRuntime(t *testing.T, store harness.Store, provider llm.LLMProvider) *harness.Runtime {
	t.Helper()
	return startRuntime(t, harness.Config{
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
}

// TestRecoveryReconcilesDanglingToolCall pins HARNESS-14: a crash between the
// durable assistant_tool_call record and its tool_outcome must not replay a
// bare tool call into the provider on recovery. The engine synthesizes an
// error tool_outcome for the dangling call before re-prompting.
func TestRecoveryReconcilesDanglingToolCall(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := &recordingTextProvider{}
	conv, sub := seedDanglingScenario(t, store, false)

	rt := startDanglingRuntime(t, store, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(ctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	recs, err := rt.Records(ctx, conv.ID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}

	// (a) the store gained a synthesized error tool_outcome for "c1", with
	// the HARNESS-14 message, before the run could settle.
	var found *harness.ToolOutcomePayload
	for _, rec := range recs {
		if rec.Kind != harness.KindToolOutcome {
			continue
		}
		var p harness.ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if p.CallID == "c1" {
			cp := p
			found = &cp
		}
	}
	if found == nil {
		t.Fatal("no synthesized tool_outcome record for call c1 — dangling call was never reconciled")
	}
	if !found.IsError {
		t.Errorf("synthesized tool_outcome IsError = false, want true")
	}
	if found.Content != danglingToolCallMessage {
		t.Errorf("synthesized tool_outcome Content = %q, want %q", found.Content, danglingToolCallMessage)
	}

	// (b) the provider's request replayed a ToolResultContent for "c1"
	// immediately after the matching ToolCallContent — no dangling tail.
	var sawCall, sawAdjacentResult bool
	for _, req := range provider.requests() {
		for i, m := range req.Messages {
			tc, ok := m.Content.(llm.ToolCallContent)
			if !ok || tc.CallID != "c1" {
				continue
			}
			sawCall = true
			if i+1 >= len(req.Messages) {
				t.Fatalf("tool call %q is the last message in the request — dangling tail", tc.CallID)
			}
			res, ok := req.Messages[i+1].Content.(llm.ToolResultContent)
			if !ok || res.CallID != "c1" {
				t.Fatalf("message after tool call %q = %#v, want a matching ToolResultContent", tc.CallID, req.Messages[i+1].Content)
			}
			sawAdjacentResult = true
		}
	}
	if !sawCall {
		t.Fatal("recovery prompt never replayed the seeded tool call c1")
	}
	if !sawAdjacentResult {
		t.Fatal("recovery prompt replayed tool call c1 without an adjacent tool result")
	}
}

// TestRecoveryDanglingToolCallReconciliationNoOpWithOutcome proves
// reconciliation is a no-op when the seeded call already has a matching
// outcome (the existing recovery suites, e.g. TestCrashMidTurnResumes and
// thought_signature_recovery_test.go, double as this proof — this test makes
// it explicit per the HARNESS-14 brief).
func TestRecoveryDanglingToolCallReconciliationNoOpWithOutcome(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := &recordingTextProvider{}
	conv, sub := seedDanglingScenario(t, store, true)

	rt := startDanglingRuntime(t, store, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(ctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	recs, err := rt.Records(ctx, conv.ID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	var outcomeCount int
	for _, rec := range recs {
		if rec.Kind == harness.KindToolOutcome {
			outcomeCount++
		}
	}
	if outcomeCount != 1 {
		t.Fatalf("tool_outcome records = %d, want exactly 1 (reconciliation must not synthesize an outcome when one already exists)", outcomeCount)
	}
}
