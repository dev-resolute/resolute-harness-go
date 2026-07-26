package harness

import (
	"bytes"
	"context"
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
