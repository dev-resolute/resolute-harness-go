package harness_test

import (
	"context"
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
		if err := store.ReleaseSubmission(ctx, sub.ID, claimed.AttemptID); err != nil {
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
