package harness

import (
	"bytes"
	"testing"
	"time"
)

// TestTaskSpawnedPayloadRoundTrip pins the task_spawned record schema
// (HARNESS-15): envelope kind plus a full payload field round trip through
// mustPayload and DecodePayload.
func TestTaskSpawnedPayloadRoundTrip(t *testing.T) {
	rec := Record{
		RecordEnvelope: RecordEnvelope{
			ID:             newULID(),
			Kind:           KindTaskSpawned,
			ConversationID: "conv-parent",
			Session:        "default",
			SubmissionID:   "sub-parent",
			Time:           time.Now(),
		},
		Payload: mustPayload(&TaskSpawnedPayload{
			CallID:              "call-1",
			Agent:               "researcher",
			ChildInstance:       "acme-call-1",
			ChildConversationID: "conv-child",
			ChildSubmissionID:   "sub-child",
			Prompt:              "summarize the thread",
		}),
	}
	if rec.Kind != "task_spawned" {
		t.Fatalf("KindTaskSpawned = %q, want %q", rec.Kind, "task_spawned")
	}
	var p TaskSpawnedPayload
	if err := rec.DecodePayload(&p); err != nil {
		t.Fatalf("DecodePayload: %v", err)
	}
	want := TaskSpawnedPayload{
		CallID:              "call-1",
		Agent:               "researcher",
		ChildInstance:       "acme-call-1",
		ChildConversationID: "conv-child",
		ChildSubmissionID:   "sub-child",
		Prompt:              "summarize the thread",
	}
	if p != want {
		t.Errorf("decoded payload = %+v, want %+v", p, want)
	}
}

// TestConversationCreatedParentRefRoundTrip pins the additive ParentRef on
// conversation_created (HARNESS-15): present and round-tripping for spawned
// children, and absent from the wire form entirely for root conversations.
func TestConversationCreatedParentRefRoundTrip(t *testing.T) {
	t.Run("with parent ref", func(t *testing.T) {
		rec := Record{
			RecordEnvelope: RecordEnvelope{
				ID:             newULID(),
				Kind:           KindConversationCreated,
				ConversationID: "conv-child",
				Session:        "default",
				Time:           time.Now(),
			},
			Payload: mustPayload(&ConversationCreatedPayload{
				Agent:    "support",
				Instance: "acme-call-1",
				Session:  "default",
				ParentRef: &ParentRef{
					ConversationID: "conv-parent",
					SpawnRecordID:  "rec-spawn",
				},
			}),
		}
		var p ConversationCreatedPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if p.ParentRef == nil {
			t.Fatal("ParentRef = nil, want non-nil")
		}
		if *p.ParentRef != (ParentRef{ConversationID: "conv-parent", SpawnRecordID: "rec-spawn"}) {
			t.Errorf("ParentRef = %+v, want {conv-parent rec-spawn}", *p.ParentRef)
		}
	})

	t.Run("without parent ref", func(t *testing.T) {
		rec := Record{
			RecordEnvelope: RecordEnvelope{
				ID:             newULID(),
				Kind:           KindConversationCreated,
				ConversationID: "conv-root",
				Session:        "default",
				Time:           time.Now(),
			},
			Payload: mustPayload(&ConversationCreatedPayload{
				Agent:    "support",
				Instance: "acme",
				Session:  "default",
			}),
		}
		if bytes.Contains(rec.Payload, []byte(`"parentRef"`)) {
			t.Errorf("payload = %s, must not contain a \"parentRef\" key when ParentRef is nil", rec.Payload)
		}
		var p ConversationCreatedPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload: %v", err)
		}
		if p.ParentRef != nil {
			t.Errorf("ParentRef = %+v, want nil", *p.ParentRef)
		}
	})
}
