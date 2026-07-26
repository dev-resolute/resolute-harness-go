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

// TestRecoveryReconcilesDanglingToolCallAcrossSubmissions pins the
// conversation-scoped fix for HARNESS-14: the dangling-call hazard is not
// limited to a re-claimed attempt of the SAME submission. Submission 1
// settles failed (e.g. attempt-budget exhaustion, a durability timeout, or an
// initialize failure) with a dangling assistant_tool_call still on the
// active leaf path. Submission 2, admitted afterward on the SAME
// conversation, starts at its own fresh AttemptCount == 1 — the exact case
// an AttemptCount > 1 gate would skip, poisoning every later submission on
// the conversation forever. The engine must reconcile submission 1's
// dangling call before submission 2's own first prompt.
func TestRecoveryReconcilesDanglingToolCallAcrossSubmissions(t *testing.T) {
	t.Parallel()
	store := memory.New()
	ctx := context.Background()

	key := harness.SessionKey{Agent: "support", Instance: "acme", Session: "default"}
	conv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID:        "01A0XCONV0000000000000000",
		Key:       key,
		CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation: %v", err)
	}

	// Submission 1: claim it, author its input and a dangling tool call
	// (no matching tool_outcome — the crash window HARNESS-14 targets), then
	// settle it failed directly through the store, as a genuinely terminal
	// submission rather than a re-claimable running one.
	sub1, err := store.AdmitSubmission(ctx, harness.Submission{
		ID:             "01A0XSUB10000000000000001",
		SessionKey:     key,
		ConversationID: conv.ID,
		Status:         harness.StatusQueued,
		Input:          harness.UserMessage("look up the weather"),
		CreatedAt:      time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission sub1: %v", err)
	}
	claimed1, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   sub1.ID,
		AttemptID:      "01A0XATTEMPT0000000000001",
		OwnerID:        "owner-1",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("ClaimSubmission sub1: %v", err)
	}
	if err := store.StartAttempt(ctx, harness.Attempt{
		ID: claimed1.AttemptID, SubmissionID: sub1.ID, OwnerID: "owner-1", StartedAt: time.Now(),
	}); err != nil {
		t.Fatalf("StartAttempt sub1: %v", err)
	}

	userRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0XREC000000000000000001",
			Kind:           harness.KindUserMessage,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub1.ID,
			AttemptID:      claimed1.AttemptID,
			Time:           time.Now(),
		},
	}
	userRec.Payload, _ = json.Marshal(harness.UserMessagePayload{Body: "look up the weather"})
	toolCallRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0XREC000000000000000002",
			Kind:           harness.KindAssistantToolCall,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub1.ID,
			AttemptID:      claimed1.AttemptID,
			Time:           time.Now(),
		},
	}
	toolCallRec.Payload, _ = json.Marshal(harness.AssistantToolCallPayload{
		CallID:   "c1",
		ToolName: "get_weather",
		Args:     json.RawMessage(`{"city":"Berlin"}`),
	})
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{userRec, toolCallRec}); err != nil {
		t.Fatalf("AppendRecords sub1: %v", err)
	}

	if err := store.ReserveSettlement(ctx, sub1.ID, claimed1.AttemptID); err != nil {
		t.Fatalf("ReserveSettlement sub1: %v", err)
	}
	settledRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0XREC000000000000000003",
			Kind:           harness.KindSubmissionSettled,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub1.ID,
			AttemptID:      claimed1.AttemptID,
			Time:           time.Now(),
		},
	}
	settledRec.Payload, _ = json.Marshal(harness.SettledPayload{
		Status:    harness.SettledFailed,
		Error:     "attempt budget exhausted (test-simulated settle-failed with a dangling tool call)",
		ErrorCode: harness.SettledErrAttemptBudget,
	})
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{settledRec}); err != nil {
		t.Fatalf("AppendRecords settled sub1: %v", err)
	}
	if err := store.FinalizeSettlement(ctx, sub1.ID); err != nil {
		t.Fatalf("FinalizeSettlement sub1: %v", err)
	}

	// Submission 2: admitted fresh on the SAME conversation/session after
	// submission 1 settled. Its own AttemptCount will be 1 on its first
	// (only) claim — the case the old AttemptCount > 1 gate missed entirely.
	sub2, err := store.AdmitSubmission(ctx, harness.Submission{
		ID:             "01A0XSUB20000000000000002",
		SessionKey:     key,
		ConversationID: conv.ID,
		Status:         harness.StatusQueued,
		Input:          harness.UserMessage("what's next?"),
		CreatedAt:      time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission sub2: %v", err)
	}

	provider := &recordingTextProvider{}
	rt := startDanglingRuntime(t, store, provider)
	waitCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(waitCtx, sub2.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}
	if sub2Final, err := store.GetSubmission(waitCtx, sub2.ID); err != nil {
		t.Fatalf("GetSubmission sub2: %v", err)
	} else if sub2Final.AttemptCount != 1 {
		t.Fatalf("sub2 AttemptCount = %d, want 1 (reconciliation must fire on a submission's own first attempt)", sub2Final.AttemptCount)
	}

	recs, err := rt.Records(waitCtx, conv.ID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}

	// (a) submission 1's dangling call "c1" was reconciled — a synthesized
	// error tool_outcome exists — even though it was submission 2's drive
	// that produced it.
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
		t.Fatal("no synthesized tool_outcome for call c1 — submission 2 never reconciled submission 1's dangling call")
	}
	if !found.IsError {
		t.Errorf("synthesized tool_outcome IsError = false, want true")
	}
	if found.Content != danglingToolCallMessage {
		t.Errorf("synthesized tool_outcome Content = %q, want %q", found.Content, danglingToolCallMessage)
	}

	// (b) submission 2's OWN prompt request to the provider already carries
	// the synthesized result immediately after the seeded call — reconciled
	// before its first prompt, not after.
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
		t.Fatal("submission 2's prompt never replayed submission 1's tool call c1")
	}
	if !sawAdjacentResult {
		t.Fatal("submission 2's prompt replayed tool call c1 without an adjacent tool result")
	}
}
