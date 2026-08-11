package harness_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// overflowErr mimics a provider "maximum context length" rejection, which
// llm.AsContextOverflow classifies as a context overflow.
var overflowErr = errors.New("request too large: this model's maximum context length is 8000 tokens")

// recoveryRuntime builds a Runtime with compaction-friendly token budgets
// and tightened engine timings.
func recoveryRuntime(t *testing.T, provider llm.LLMProvider, maxAttempts int) (*harness.Runtime, harness.Store) {
	t.Helper()
	store := memory.New()
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:            "mock/test-model",
						ContextWindow:    8_000,
						Providers:        []llm.LLMProvider{provider},
						SystemPrompt:     "You are terse.",
						KeepRecentTokens: 60,
						MaxAttempts:      maxAttempts,
					}, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 2 * time.Second,
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
	return rt, store
}

const longAnswer = "The complete migration plan involves careful schema versioning, dual writes during the transition window, backfill of historical rows, and a verification pass comparing row counts and checksums before the cutover completes. "

// Overflow mid-run triggers compact-and-retry: the run settles, the
// compaction record is in the stream, and the retried turn sees the
// summary-substituted context.
func TestOverflowCompactsAndRetries(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// Prompt 1 builds up history.
	provider.OnPrompt(mock.LastUser("plan the migration")).
		RespondText(strings.Repeat(longAnswer, 4)).Add()
	// Prompt 2 first overflows...
	provider.OnPrompt(mock.LastUser("now execute it")).Error(overflowErr).Add()
	// ...then the harness compacts (the summarization call)...
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "conversation to summarize") {
				return true
			}
		}
		return false
	})).RespondText("## Goal\nCompact summary of the migration plan.").Add()
	// ...and the retried turn runs over the compacted context: the summary
	// is present.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "Compact summary of the migration plan") {
				return true
			}
		}
		return false
	})).RespondText("Executing now.").Add()

	rt, store := recoveryRuntime(t, provider, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	res1, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("plan the migration"),
	})
	if err != nil {
		t.Fatalf("Dispatch 1: %v", err)
	}
	if s, err := rt.Wait(ctx, res1.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("first prompt settled %+v (%v), want success", s, err)
	}

	before, err := rt.Records(ctx, res1.ConversationID, "")
	if err != nil {
		t.Fatalf("Records before: %v", err)
	}

	res2, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("now execute it"),
	})
	if err != nil {
		t.Fatalf("Dispatch 2: %v", err)
	}
	settled, err := rt.Wait(ctx, res2.SubmissionID)
	if err != nil {
		t.Fatalf("Wait 2: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("overflow run settled %+v, want success after compact-and-retry", settled)
	}

	after, err := rt.Records(ctx, res2.ConversationID, "")
	if err != nil {
		t.Fatalf("Records after: %v", err)
	}
	if countKind(after, harness.KindCompaction) == 0 {
		t.Fatal("no compaction record in the stream")
	}

	// History is append-only: every pre-existing record survives unchanged
	// in order.
	for i, rec := range before {
		if after[i].ID != rec.ID {
			t.Fatalf("record %d mutated by compaction: %s → %s", i, rec.ID, after[i].ID)
		}
	}
	_ = store
}

// Manual Compact on an idle session lands a compaction record; subsequent
// prompts resume on the re-parented leaf path (they see the summary, and the
// mock only answers when they do).
func TestManualCompactIdleSession(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnPrompt(mock.LastUser("remember the launch codes")).
		RespondText(strings.Repeat(longAnswer, 4)).Add()
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "conversation to summarize") {
				return true
			}
		}
		return false
	})).RespondText("## Goal\nManual compact summary.").Add()
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		sawSummary := false
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "Manual compact summary") {
				sawSummary = true
			}
		}
		return sawSummary
	})).RespondText("Resumed on the compacted path.").Add()

	rt, _ := recoveryRuntime(t, provider, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("remember the launch codes"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if s, err := rt.Wait(ctx, res.SubmissionID); err != nil || s.Status != harness.SettledSucceeded {
		t.Fatalf("prompt settled %+v (%v), want success", s, err)
	}

	if err := rt.Compact(ctx, harness.CompactRequest{Agent: "support", Instance: "acme"}); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if countKind(recs, harness.KindCompaction) != 1 {
		t.Fatalf("compaction records = %d, want 1", countKind(recs, harness.KindCompaction))
	}

	res2, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("continue"),
	})
	if err != nil {
		t.Fatalf("Dispatch 2: %v", err)
	}
	settled, err := rt.Wait(ctx, res2.SubmissionID)
	if err != nil {
		t.Fatalf("Wait 2: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("post-compact prompt settled %+v, want success (projection must serve the re-parented path)", settled)
	}
}

// Transient model errors retry with backoff as fresh attempts; the budget is
// durable, so the run eventually succeeds with the attempt count reflecting
// the retries.
func TestTransientErrorRetriesWithBackoff(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().Error(errors.New("upstream 503: temporarily unavailable")).Add()
	provider.OnAny().RespondText("recovered on retry").Add()

	rt, store := recoveryRuntime(t, provider, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("flaky please"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled %+v, want success after transient retry", settled)
	}
	sub, err := store.GetSubmission(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if sub.AttemptCount != 2 {
		t.Fatalf("attempt count = %d, want 2 (transient failure + recovery)", sub.AttemptCount)
	}
}

// An exhausted transient-retry budget settles as failed with a structured
// error; the budget is recomputed from durable history, so seeded prior
// attempts count (a restart mid-backoff cannot reset it).
func TestTransientBudgetRecomputedFromHistory(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// No steps needed: the budget check fires before any model call.

	store := memory.New()
	_, sub := seededSubmission(t, store, "doomed work")
	// Burn two attempts through the public store API — the durable history a
	// restart would find after crashing mid-backoff.
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		claimed, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
			SubmissionID:   sub.ID,
			AttemptID:      harness.SessionKey{Agent: "seed"}.String() + string(rune('a'+i)),
			OwnerID:        "dead-owner",
			LeaseExpiresAt: time.Now().Add(time.Minute),
		})
		if err != nil {
			t.Fatalf("ClaimSubmission %d: %v", i, err)
		}
		if err := store.ReleaseSubmission(ctx, harness.SubmissionRelease{SubmissionID: sub.ID, AttemptID: claimed.AttemptID}); err != nil {
			t.Fatalf("ReleaseSubmission %d: %v", i, err)
		}
	}

	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 8_000,
						Providers:     []llm.LLMProvider{provider},
						MaxAttempts:   2,
					}, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrAttemptBudget {
		t.Fatalf("settled = %+v, want failed/attempt_budget_exhausted from durable history", settled)
	}
	if provider.Called() != 0 {
		t.Fatalf("provider called %d times past the durable budget, want 0", provider.Called())
	}
}

// A spawned task call is intentionally outcome-less while its child runs —
// the wake authors the outcome at settlement. The reconciler must exempt it
// (HARNESS-15): a re-driven parent (crash mid-resume after a spawn, before
// parking) must not see a fabricated error for a live child, which the
// wake's existing-outcome-wins check would then honor forever. A genuinely
// dangling non-task call on the same path still gets the synthesized error.
func TestReconcilerExemptsSpawnedCall(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := memory.New()
	conv, sub := seededSubmission(t, store, "triage this")
	claimed := claimSeeded(t, store, sub)

	// Live child: running under a dead owner with an unexpired lease, so
	// neither the claim loop nor the lease reclaimer touches it.
	childKey := harness.SessionKey{Agent: "reviewer", Instance: harness.InstanceID("acme:" + safeCallID("task-1")), Session: "task-" + safeCallID("task-1")}
	childConv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000CHL", Key: childKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(child): %v", err)
	}
	child, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000CHL", SessionKey: childKey, ConversationID: childConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("check it"), CreatedAt: time.Now(),
		ParentSubmissionID: sub.ID, ParentCallID: "task-1", Depth: 1,
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(child): %v", err)
	}
	if _, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   child.ID,
		AttemptID:      "01A0DEADATTEMPT0000000000C",
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatalf("ClaimSubmission(child): %v", err)
	}

	// The crashed parent's conversation: a spawned task call (outcome-less
	// by design — the child above is still running) and a genuinely
	// dangling non-task call.
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000000EX1", harness.KindUserMessage, conv.ID, "default", sub.ID, claimed.AttemptID,
			&harness.UserMessagePayload{Body: "triage this"}),
		seedRecord("01A0SEEDREC0000000000000EX2", harness.KindAssistantToolCall, conv.ID, "default", sub.ID, claimed.AttemptID,
			&harness.AssistantToolCallPayload{CallID: "task-1", ToolName: "task", Args: taskArgs("reviewer", "check it")}),
		seedRecord("01A0SEEDREC0000000000000EX3", harness.KindTaskSpawned, conv.ID, "default", sub.ID, claimed.AttemptID,
			&harness.TaskSpawnedPayload{CallID: "task-1", Agent: "reviewer", ChildInstance: string(childKey.Instance),
				ChildConversationID: childConv.ID, ChildSubmissionID: child.ID, Prompt: "check it"}),
		seedRecord("01A0SEEDREC0000000000000EX4", harness.KindAssistantToolCall, conv.ID, "default", sub.ID, claimed.AttemptID,
			&harness.AssistantToolCallPayload{CallID: "c9", ToolName: "get_weather", Args: json.RawMessage(`{"city":"Berlin"}`)}),
	}); err != nil {
		t.Fatalf("AppendRecords: %v", err)
	}

	provider := &recordingTextProvider{}
	rt := startDanglingRuntime(t, store, provider)
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
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
	outcomes := map[string]harness.ToolOutcomePayload{}
	for _, rec := range recs {
		if rec.Kind != harness.KindToolOutcome {
			continue
		}
		var p harness.ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload(tool_outcome): %v", err)
		}
		outcomes[p.CallID] = p
	}
	if _, ok := outcomes["task-1"]; ok {
		t.Errorf("synthesized outcome for spawned call task-1 = %+v, want none — a live child exempts the call", outcomes["task-1"])
	}
	c9, ok := outcomes["c9"]
	if !ok {
		t.Fatal("no synthesized outcome for the genuinely dangling call c9")
	}
	if !c9.IsError || c9.Content != danglingToolCallMessage {
		t.Errorf("outcome for c9 = %+v, want the HARNESS-14 synthesized error", c9)
	}
}

// seedParkedParent seeds the crash state the wait scan recovers from: a
// parent claimed then parked in waiting (PendingResume set, wait bound as
// given), its conversation carrying one spawned task call, and a live child
// submission linked to that call. The child is left queued; the caller
// settles it directly (backstop test) or lets the scan cancel it (expiry
// test).
func seedParkedParent(t *testing.T, store harness.Store, waitUntil time.Time) (harness.Conversation, harness.Submission, harness.Conversation, harness.Submission) {
	t.Helper()
	ctx := context.Background()

	pKey := harness.SessionKey{Agent: "triage", Instance: "acme", Session: "default"}
	pConv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000WP", Key: pKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(parent): %v", err)
	}
	parent, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000WP", SessionKey: pKey, ConversationID: pConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("triage this"), CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(parent): %v", err)
	}
	const parentAttempt = "01A0DEADATTEMPT0000000000P"
	if _, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   parent.ID,
		AttemptID:      parentAttempt,
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	}); err != nil {
		t.Fatalf("ClaimSubmission(parent): %v", err)
	}
	if err := store.WaitSubmission(ctx, harness.SubmissionWait{
		SubmissionID: parent.ID, AttemptID: parentAttempt, WaitUntil: waitUntil,
	}); err != nil {
		t.Fatalf("WaitSubmission(parent): %v", err)
	}

	childKey := harness.SessionKey{Agent: "reviewer", Instance: harness.InstanceID("acme:" + safeCallID("call-1")), Session: "task-" + safeCallID("call-1")}
	childConv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000WC", Key: childKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(child): %v", err)
	}
	child, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000WC", SessionKey: childKey, ConversationID: childConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("check it"), CreatedAt: time.Now(),
		ParentSubmissionID: parent.ID, ParentCallID: "call-1", Depth: 1,
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(child): %v", err)
	}

	if err := store.AppendRecords(ctx, pConv.ID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000000WP1", harness.KindConversationCreated, pConv.ID, "default", "", "",
			&harness.ConversationCreatedPayload{Agent: "triage", Instance: "acme", Session: "default"}),
		seedRecord("01A0SEEDREC0000000000000WP2", harness.KindUserMessage, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.UserMessagePayload{Body: "triage this"}),
		seedRecord("01A0SEEDREC0000000000000WP3", harness.KindAssistantToolCall, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.AssistantToolCallPayload{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "check it")}),
		seedRecord("01A0SEEDREC0000000000000WP4", harness.KindTaskSpawned, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.TaskSpawnedPayload{CallID: "call-1", Agent: "reviewer", ChildInstance: string(childKey.Instance),
				ChildConversationID: childConv.ID, ChildSubmissionID: child.ID, Prompt: "check it"}),
	}); err != nil {
		t.Fatalf("AppendRecords(parent): %v", err)
	}
	return pConv, parent, childConv, child
}

// waitScanOutcomes flattens a conversation's tool_outcome payloads by call
// ID for the wait-scan assertions.
func waitScanOutcomes(t *testing.T, rt *harness.Runtime, conversationID string) map[string]harness.ToolOutcomePayload {
	t.Helper()
	recs, err := rt.Records(context.Background(), conversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	out := map[string]harness.ToolOutcomePayload{}
	for _, rec := range recs {
		if rec.Kind != harness.KindToolOutcome {
			continue
		}
		var p harness.ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload(tool_outcome): %v", err)
		}
		if _, dup := out[p.CallID]; dup {
			t.Fatalf("duplicate tool_outcome for %s", p.CallID)
		}
		out[p.CallID] = p
	}
	return out
}

// The child settled but its wake was lost to a crash window, leaving the
// spawned call outcome-less — the reconciler must NOT exempt it (the child
// is settled), yet the wake backstop in the wait scan lands the REAL
// outcome before the re-drive's reconciler can synthesize an error.
func TestReconcilerSynthesizesAfterChildSettledWithoutOutcome(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := memory.New()
	pConv, parent, childConv, child := seedParkedParent(t, store, time.Time{}) // unbounded wait: never expires

	// Settle the child directly through the store — no coordinator, so no
	// wake runs — with its final answer durable.
	const childAttempt = "01A0DEADATTEMPT0000000000K"
	if _, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   child.ID,
		AttemptID:      childAttempt,
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	}); err != nil {
		t.Fatalf("ClaimSubmission(child): %v", err)
	}
	if err := store.ReserveSettlement(ctx, child.ID, childAttempt); err != nil {
		t.Fatalf("ReserveSettlement(child): %v", err)
	}
	answerBody, _ := json.Marshal("child answer")
	if err := store.AppendRecords(ctx, childConv.ID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000000WC1", harness.KindAssistantMessageCompleted, childConv.ID, child.SessionKey.Session, child.ID, childAttempt,
			&harness.AssistantMessageCompletedPayload{Message: harness.MessagePayload{Role: "assistant", Type: "text", Body: answerBody}}),
		seedRecord("01A0SEEDREC0000000000000WC2", harness.KindSubmissionSettled, childConv.ID, child.SessionKey.Session, child.ID, childAttempt,
			&harness.SettledPayload{Status: harness.SettledSucceeded}),
	}); err != nil {
		t.Fatalf("AppendRecords(child): %v", err)
	}
	if err := store.FinalizeSettlement(ctx, child.ID); err != nil {
		t.Fatalf("FinalizeSettlement(child): %v", err)
	}

	parentProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("triage done")}}
	childProv := &scriptProvider{name: "mock"}
	rt := startRuntime(t, subagentConfig(store, parentProv, childProv, harness.SubagentLimits{}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// The backstop wake landed the child's real outcome and requeued the
	// parent; the resume drive settled it.
	parentSettled, err := rt.Wait(wctx, parent.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the backstop wake", parentSettled)
	}
	outcomes := waitScanOutcomes(t, rt, pConv.ID)
	outcome, ok := outcomes["call-1"]
	if !ok {
		t.Fatal("no tool_outcome for call-1 — the backstop wake never landed")
	}
	if outcome.IsError || outcome.Content != "child answer" {
		t.Errorf("outcome for call-1 = %+v, want the child's real answer (not a synthesized error)", outcome)
	}
	if n := len(parentProv.requests()); n != 1 {
		t.Errorf("parent provider calls = %d, want 1 (the resume drive)", n)
	}
	if n := len(childProv.requests()); n != 0 {
		t.Errorf("child provider calls = %d, want 0 (the child recovered from durable state)", n)
	}
}

// A bounded wait that lapses while the child is MID-RUN must not let the
// flagged child run to completion: the expiry scan cancels the in-process
// run context (errWaitExpired), the child's own post-drive switch settles
// it cancelled with the expiry reason, and the parent resumes on an error
// outcome for the outstanding call.
func TestWaitExpiryCancelsRunningChild(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := memory.New()

	parentProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("handled the expiry")}}
	childGate := make(chan struct{})
	childProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("child answer")}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parentProv, childProv, harness.SubagentLimits{}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Seed after startup so the full wait window is available for the claim:
	// the child is claimed and streaming (gated) before the wait lapses.
	_, parent, childConv, child := seedParkedParent(t, store, time.Now().Add(500*time.Millisecond))
	waitForRequests(t, childProv, 1)

	parentSettled, err := rt.Wait(wctx, parent.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded (the error outcome resumes, not fails, the parent)", parentSettled)
	}

	// The gated child never produced its answer: the expiry scan cancelled
	// its run context and its own settle landed cancelled with the reason.
	childSettled := waitForSettledRecord(t, rt, childConv.ID, child.ID)
	if childSettled.Status != harness.SettledFailed || childSettled.ErrorCode != harness.SettledErrCancelled {
		t.Errorf("child settled = %+v, want failed/cancelled", childSettled)
	}
	if childSettled.Error != "cancelled: parent wait expired" {
		t.Errorf("child settled error = %q, want %q", childSettled.Error, "cancelled: parent wait expired")
	}
	if n := len(childProv.requests()); n != 1 {
		t.Errorf("child provider calls = %d, want 1 (cancelled mid-run, never retried)", n)
	}
	if n := len(parentProv.requests()); n != 1 {
		t.Errorf("parent provider calls = %d, want 1 (the resume drive seeing the failure)", n)
	}
}

// A bounded wait that lapses with the child still in flight ends the
// suspension: the scan cancels the child, lands an error tool_outcome for
// the outstanding spawned call, and requeues the parent so its re-drive
// sees the failure as the call's result.
func TestWaitExpiry(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := memory.New()
	pConv, parent, _, child := seedParkedParent(t, store, time.Now().Add(-time.Minute)) // already expired

	parentProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("handled the expiry")}}
	childProv := &scriptProvider{name: "mock"}
	rt := startRuntime(t, subagentConfig(store, parentProv, childProv, harness.SubagentLimits{}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	parentSettled, err := rt.Wait(wctx, parent.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded (the error outcome resumes, not fails, the parent)", parentSettled)
	}

	// The scan cancelled the still-queued child before the claim loop could
	// ever run it.
	childFinal, err := store.GetSubmission(ctx, child.ID)
	if err != nil {
		t.Fatalf("GetSubmission(child): %v", err)
	}
	if childFinal.Status != harness.StatusSettled || childFinal.LastError != "parent wait expired" {
		t.Errorf("child = %+v, want settled via cancel with reason %q", childFinal, "parent wait expired")
	}
	if n := len(childProv.requests()); n != 0 {
		t.Errorf("child provider calls = %d, want 0 (cancelled before it ever ran)", n)
	}

	outcomes := waitScanOutcomes(t, rt, pConv.ID)
	outcome, ok := outcomes["call-1"]
	if !ok {
		t.Fatal("no tool_outcome for call-1 — the expiry scan never landed the failure")
	}
	if !outcome.IsError || outcome.Content != "task wait expired" {
		t.Errorf("outcome for call-1 = %+v, want the wait-expiry error", outcome)
	}
	if n := len(parentProv.requests()); n != 1 {
		t.Errorf("parent provider calls = %d, want 1 (the resume drive seeing the failure)", n)
	}
}
