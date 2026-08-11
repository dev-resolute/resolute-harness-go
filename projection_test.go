package harness

import (
	"bytes"
	"context"
	"reflect"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// fakeConvStore is a minimal ConversationStore fake for exercising the
// projection directly, without pulling in the memory package (which itself
// depends on this package).
type fakeConvStore struct {
	conv Conversation
	recs []Record
}

func (s *fakeConvStore) EnsureConversation(ctx context.Context, candidate Conversation) (Conversation, bool, error) {
	return s.conv, false, nil
}

func (s *fakeConvStore) GetConversation(ctx context.Context, key SessionKey) (Conversation, error) {
	return s.conv, nil
}

func (s *fakeConvStore) AppendRecords(ctx context.Context, conversationID string, recs []Record) error {
	s.recs = append(s.recs, recs...)
	return nil
}

func (s *fakeConvStore) ReadRecords(ctx context.Context, conversationID string, afterID string) ([]Record, error) {
	return s.recs, nil
}

func newTestProjection() (*projection, *fakeConvStore) {
	conv := Conversation{
		ID:        "conv-1",
		Key:       SessionKey{Agent: "support", Instance: "acme", Session: "default"},
		CreatedAt: time.Now(),
	}
	store := &fakeConvStore{conv: conv}
	proj := &projection{
		store:        store,
		conv:         conv,
		submissionID: "sub-1",
		attemptID:    "attempt-1",
	}
	return proj, store
}

// TestProjectionSkipsTaskSpawned pins the projection-safety rule (HARNESS-15): a task_spawned record — like any non-message record — never leaks into the LLM-facing []pi.Message.
func TestProjectionSkipsTaskSpawned(t *testing.T) {
	base := []Record{
		{RecordEnvelope: RecordEnvelope{ID: "r1", Kind: KindConversationCreated, ConversationID: "conv-1"},
			Payload: mustPayload(&ConversationCreatedPayload{Agent: "support", Instance: "acme", Session: "default"})},
		{RecordEnvelope: RecordEnvelope{ID: "r2", Kind: KindUserMessage, ConversationID: "conv-1"},
			Payload: mustPayload(&UserMessagePayload{Body: "hi there"})},
		{RecordEnvelope: RecordEnvelope{ID: "r3", Kind: KindAssistantToolCall, ConversationID: "conv-1"},
			Payload: mustPayload(&AssistantToolCallPayload{CallID: "call-1", ToolName: "task"})},
		{RecordEnvelope: RecordEnvelope{ID: "r5", Kind: KindToolOutcome, ConversationID: "conv-1"},
			Payload: mustPayload(&ToolOutcomePayload{CallID: "call-1", ToolName: "task", Content: "child answer"})},
	}
	spawned := []Record{
		base[0], base[1], base[2],
		{RecordEnvelope: RecordEnvelope{ID: "r4", Kind: KindTaskSpawned, ConversationID: "conv-1", SubmissionID: "sub-parent"},
			Payload: mustPayload(&TaskSpawnedPayload{
				CallID:              "call-1",
				Agent:               "researcher",
				ChildInstance:       "acme-call-1",
				ChildConversationID: "conv-child",
				ChildSubmissionID:   "sub-child",
				Prompt:              "summarize the thread",
			})},
		base[3],
	}

	load := func(recs []Record) []pi.Message {
		t.Helper()
		proj, store := newTestProjection()
		store.recs = recs
		msgs, err := proj.Load(context.Background(), pi.SessionID(proj.conv.ID))
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		return msgs
	}
	want, got := load(base), load(spawned)
	if len(got) != 3 {
		t.Fatalf("Load with task_spawned returned %d messages, want 3: %+v", len(got), got)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Load with task_spawned = %+v, want identical to load without it %+v", got, want)
	}
}

// TestAppendBranchSummaryUsageRoundTrip covers AGENT-20's harness half: the
// compaction record's Usage field must round-trip the summarization usage
// agent-core reports on pi.BranchSummary, and must stay absent (no "usage"
// key, decodes to nil) when the provider reported none.
func TestAppendBranchSummaryUsageRoundTrip(t *testing.T) {
	t.Run("usage present", func(t *testing.T) {
		proj, store := newTestProjection()
		usage := &pi.Usage{InputTokens: 3, OutputTokens: 4}
		err := proj.AppendBranchSummary(context.Background(), pi.SessionID(proj.conv.ID), pi.BranchSummary{
			StartIdx:  0,
			EndIdx:    0,
			Summary:   "the conversation so far",
			CreatedAt: time.Now(),
			Usage:     usage,
		})
		if err != nil {
			t.Fatalf("AppendBranchSummary: %v", err)
		}
		if len(store.recs) != 1 {
			t.Fatalf("recs = %d, want 1", len(store.recs))
		}
		rec := store.recs[0]
		if rec.Kind != KindCompaction {
			t.Fatalf("rec.Kind = %q, want %q", rec.Kind, KindCompaction)
		}
		var payload CompactionPayload
		if err := rec.DecodePayload(&payload); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if payload.Usage == nil {
			t.Fatal("payload.Usage = nil, want non-nil")
		}
		if *payload.Usage != *usage {
			t.Errorf("payload.Usage = %+v, want %+v", *payload.Usage, *usage)
		}
	})

	t.Run("usage absent", func(t *testing.T) {
		proj, store := newTestProjection()
		err := proj.AppendBranchSummary(context.Background(), pi.SessionID(proj.conv.ID), pi.BranchSummary{
			StartIdx:  0,
			EndIdx:    0,
			Summary:   "the conversation so far",
			CreatedAt: time.Now(),
		})
		if err != nil {
			t.Fatalf("AppendBranchSummary: %v", err)
		}
		if len(store.recs) != 1 {
			t.Fatalf("recs = %d, want 1", len(store.recs))
		}
		rec := store.recs[0]
		if bytes.Contains(rec.Payload, []byte(`"usage"`)) {
			t.Errorf("payload = %s, must not contain a \"usage\" key when Usage is nil", rec.Payload)
		}
		var payload CompactionPayload
		if err := rec.DecodePayload(&payload); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if payload.Usage != nil {
			t.Errorf("payload.Usage = %+v, want nil", *payload.Usage)
		}
	})
}
