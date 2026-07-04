package harness_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
	"github.com/dev-resolute/resolute-harness-go/sqlite"
)

// storeFactories enumerates the in-tree backends; behavioral tests run
// identically over each (ADR-0006: one contract, one tier).
func storeFactories() map[string]func(t *testing.T) harness.Store {
	return map[string]func(t *testing.T) harness.Store{
		"memory": func(t *testing.T) harness.Store {
			t.Helper()
			return memory.New()
		},
		"sqlite": func(t *testing.T) harness.Store {
			t.Helper()
			s, err := sqlite.Open(filepath.Join(t.TempDir(), "harness.db"))
			if err != nil {
				t.Fatalf("sqlite.Open: %v", err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Errorf("sqlite Close: %v", err)
				}
			})
			return s
		},
	}
}

// forEachStore runs fn as a subtest per in-tree store backend.
func forEachStore(t *testing.T, fn func(t *testing.T, store harness.Store)) {
	for name, factory := range storeFactories() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			fn(t, factory(t))
		})
	}
}

// newTestRuntime builds a started Runtime with one "support" agent wired to
// the given MockProvider over the memory store. Callers own the provider
// script; cleanup stops the Runtime.
func newTestRuntime(t *testing.T, provider *mock.MockProvider) *harness.Runtime {
	t.Helper()
	return newTestRuntimeOn(t, provider, memory.New())
}

// newTestRuntimeOn is newTestRuntime over an explicit store backend.
func newTestRuntimeOn(t *testing.T, provider *mock.MockProvider, store harness.Store) *harness.Runtime {
	t.Helper()
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						SystemPrompt:  "You are a support agent.",
					}, nil
				},
			},
		},
		Store: store,
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

// kinds extracts the record kinds in order.
func kinds(recs []harness.Record) []harness.RecordKind {
	out := make([]harness.RecordKind, len(recs))
	for i, r := range recs {
		out[i] = r.Kind
	}
	return out
}

// assertKindSubsequence asserts want appears in recs' kinds as an ordered
// subsequence (other kinds may interleave).
func assertKindSubsequence(t *testing.T, recs []harness.Record, want []harness.RecordKind) {
	t.Helper()
	i := 0
	for _, k := range kinds(recs) {
		if i < len(want) && k == want[i] {
			i++
		}
	}
	if i != len(want) {
		t.Fatalf("record kinds %v missing ordered subsequence %v", kinds(recs), want)
	}
}

func TestDispatchPromptSettles(t *testing.T) {
	t.Parallel()
	forEachStore(t, testDispatchPromptSettles)
}

func testDispatchPromptSettles(t *testing.T, store harness.Store) {
	provider := mock.New("mock")
	provider.OnAny().RespondText("hello from the mock").Add()
	rt := newTestRuntimeOn(t, provider, store)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("hi there"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if res.SubmissionID == "" || res.ConversationID == "" {
		t.Fatalf("Dispatch result missing ids: %+v", res)
	}

	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled status = %q (error %q), want %q", settled.Status, settled.Error, harness.SettledSucceeded)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	assertKindSubsequence(t, recs, []harness.RecordKind{
		harness.KindConversationCreated,
		harness.KindUserMessage,
		harness.KindAssistantMessageCompleted,
		harness.KindSubmissionSettled,
	})

	// Every record after conversation_created carries the submission id, and
	// all records share the conversation id.
	for _, r := range recs {
		if r.ConversationID != res.ConversationID {
			t.Errorf("record %s conversationId = %q, want %q", r.ID, r.ConversationID, res.ConversationID)
		}
		if r.Kind != harness.KindConversationCreated && r.SubmissionID != res.SubmissionID {
			t.Errorf("record %s (%s) submissionId = %q, want %q", r.ID, r.Kind, r.SubmissionID, res.SubmissionID)
		}
	}

	// Record IDs are strictly increasing (ULIDs double as SSE offsets).
	for i := 1; i < len(recs); i++ {
		if recs[i].ID <= recs[i-1].ID {
			t.Errorf("record ids not strictly increasing: %q then %q", recs[i-1].ID, recs[i].ID)
		}
	}
}
