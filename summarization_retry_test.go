package harness_test

import (
	"context"
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

// summarizationRetryRuntime builds a Runtime with compaction-friendly token
// budgets, summarization retries enabled, and an observer attached.
func summarizationRetryRuntime(t *testing.T, provider llm.LLMProvider, observe harness.Observer) *harness.Runtime {
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
						SummarizationRetry: pi.SummarizationRetryPolicy{
							MaxRetries: 3,
							BaseDelay:  time.Millisecond,
							MaxDelay:   10 * time.Millisecond,
						},
					}, nil
				},
			},
		},
		Store:         store,
		Observers:     []harness.Observer{observe},
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
	return rt
}

// TestSummarizationRetryRecoversOverflowCompaction drives the overflow ladder
// with a summarization call that fails transiently once: agent-core must
// retry it (configured via AgentRuntimeConfig.SummarizationRetry), the run
// must settle successfully, and observers must see the retry lifecycle as
// RecoveryEvents.
func TestSummarizationRetryRecoversOverflowCompaction(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// Prompt 1 builds up history.
	provider.OnPrompt(mock.LastUser("plan the migration")).
		RespondText(strings.Repeat(longAnswer, 4)).Add()
	// Prompt 2 first overflows...
	provider.OnPrompt(mock.LastUser("now execute it")).Error(overflowErr).Add()
	// ...the harness compacts, but the summarization call fails transiently...
	summarization := mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "conversation to summarize") {
				return true
			}
		}
		return false
	})
	provider.OnPrompt(summarization).Error(errors.New("503 service unavailable")).Add()
	// ...agent-core retries the summarization per SummarizationRetry...
	provider.OnPrompt(summarization).RespondText("## Goal\nCompact summary of the migration plan.").Add()
	// ...and the retried turn runs over the compacted context.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "Compact summary of the migration plan") {
				return true
			}
		}
		return false
	})).RespondText("Executing now.").Add()

	var mu sync.Mutex
	var decisions []string
	observe := func(ev harness.HarnessEvent) {
		if re, ok := ev.(harness.RecoveryEvent); ok {
			mu.Lock()
			decisions = append(decisions, re.Decision)
			mu.Unlock()
		}
	}

	rt := summarizationRetryRuntime(t, provider, observe)
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
		t.Fatalf("overflow run settled %+v, want success after summarization retry", settled)
	}

	records, err := rt.Records(ctx, res2.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if countKind(records, harness.KindCompaction) == 0 {
		t.Fatal("no compaction record in the stream")
	}

	mu.Lock()
	defer mu.Unlock()
	counts := map[string]int{}
	for _, d := range decisions {
		counts[d]++
	}
	for _, want := range []string{
		"summarization_retry_scheduled",
		"summarization_retry_attempt_start",
		"summarization_retry_finished",
	} {
		if counts[want] != 1 {
			t.Errorf("RecoveryEvent decision %q seen %d times, want 1 (all decisions: %v)", want, counts[want], decisions)
		}
	}
}
