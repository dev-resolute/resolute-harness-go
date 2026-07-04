// Package storetest is the exported conformance suite for the harness store
// contract (ADR-0006). Every Store implementation — in-tree and third-party —
// must pass Run unchanged; it pins the engine's store-visible invariants
// (architecture.md §4.2) and doubles as crash-test infrastructure: any
// mid-crash engine state can be seeded through the public store API alone.
package storetest

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// Factory builds a fresh, empty store for one subtest. Implementations
// register cleanup on t.
type Factory func(t *testing.T) harness.Store

// Run executes the full conformance suite against stores built by factory.
func Run(t *testing.T, factory Factory) {
	tests := []struct {
		name string
		fn   func(t *testing.T, s harness.Store)
	}{
		{"AdmissionIdempotentReplay", testAdmissionIdempotentReplay},
		{"AdmissionPayloadConflict", testAdmissionPayloadConflict},
		{"GetSubmissionNotFound", testGetSubmissionNotFound},
		{"RunnableHeadPerSession", testRunnableHeadPerSession},
		{"RunnableExcludesBusySessions", testRunnableExcludesBusySessions},
		{"ListByStatus", testListByStatus},
		{"ClaimCAS", testClaimCAS},
		{"AttemptMarkers", testAttemptMarkers},
		{"LeaseRenewAndExpiry", testLeaseRenewAndExpiry},
		{"ReleaseSubmission", testReleaseSubmission},
		{"SettlementPhases", testSettlementPhases},
		{"FinalizeIdempotent", testFinalizeIdempotent},
		{"ConversationEnsureAndGet", testConversationEnsureAndGet},
		{"RecordAppendAndReadFromOffset", testRecordAppendAndReadFromOffset},
		{"AttachmentPutGet", testAttachmentPutGet},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(t, factory(t))
		})
	}
}

func ctxT(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

var idCounter int

// nextID fabricates lexically increasing ids so record offsets order
// correctly without depending on harness id generation.
func nextID(prefix string) string {
	idCounter++
	return fmt.Sprintf("%s-%010d", prefix, idCounter)
}

func newSubmission(session string) harness.Submission {
	id := nextID("sub")
	return harness.Submission{
		ID: id,
		SessionKey: harness.SessionKey{
			Agent:    "support",
			Instance: "acme",
			Session:  session,
		},
		ConversationID: "conv-" + session,
		Status:         harness.StatusQueued,
		Input:          harness.UserMessage("hello from " + id),
		CreatedAt:      time.Now(),
	}
}

// admit seeds a queued submission.
func admit(t *testing.T, s harness.Store, session string) harness.Submission {
	t.Helper()
	sub, err := s.AdmitSubmission(ctxT(t), newSubmission(session))
	if err != nil {
		t.Fatalf("AdmitSubmission: %v", err)
	}
	return sub
}

// claim moves a queued submission to running with a fresh attempt.
func claim(t *testing.T, s harness.Store, sub harness.Submission) harness.Submission {
	t.Helper()
	claimed, err := s.ClaimSubmission(ctxT(t), harness.SubmissionClaim{
		SubmissionID:   sub.ID,
		AttemptID:      nextID("att"),
		OwnerID:        "owner-1",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	})
	if err != nil {
		t.Fatalf("ClaimSubmission(%s): %v", sub.ID, err)
	}
	return claimed
}

func testAdmissionIdempotentReplay(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := newSubmission("default")
	first, err := s.AdmitSubmission(ctx, sub)
	if err != nil {
		t.Fatalf("AdmitSubmission: %v", err)
	}
	replayed, err := s.AdmitSubmission(ctx, sub)
	if err != nil {
		t.Fatalf("replayed AdmitSubmission: %v", err)
	}
	if replayed.ID != first.ID || replayed.Status != first.Status {
		t.Fatalf("replay returned %+v, want original %+v", replayed, first)
	}
}

func testAdmissionPayloadConflict(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := newSubmission("default")
	if _, err := s.AdmitSubmission(ctx, sub); err != nil {
		t.Fatalf("AdmitSubmission: %v", err)
	}
	mutated := sub
	mutated.Input = harness.UserMessage("a different payload")
	_, err := s.AdmitSubmission(ctx, mutated)
	if !errors.Is(err, harness.ErrDispatchConflict) {
		t.Fatalf("mutated re-admission error = %v, want ErrDispatchConflict", err)
	}
}

func testGetSubmissionNotFound(t *testing.T, s harness.Store) {
	_, err := s.GetSubmission(ctxT(t), "no-such-id")
	if !errors.Is(err, harness.ErrSubmissionNotFound) {
		t.Fatalf("GetSubmission(no-such-id) error = %v, want ErrSubmissionNotFound", err)
	}
}

func testRunnableHeadPerSession(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	a1 := admit(t, s, "session-a")
	admit(t, s, "session-a") // behind a1, never runnable while a1 is unsettled
	b1 := admit(t, s, "session-b")

	runnable, err := s.ListRunnable(ctx)
	if err != nil {
		t.Fatalf("ListRunnable: %v", err)
	}
	got := ids(runnable)
	if len(got) != 2 || !got[a1.ID] || !got[b1.ID] {
		t.Fatalf("runnable = %v, want exactly heads {%s, %s}", keys(got), a1.ID, b1.ID)
	}
}

func testRunnableExcludesBusySessions(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	a1 := admit(t, s, "session-a")
	admit(t, s, "session-a")
	claim(t, s, a1)

	runnable, err := s.ListRunnable(ctx)
	if err != nil {
		t.Fatalf("ListRunnable: %v", err)
	}
	if len(runnable) != 0 {
		t.Fatalf("runnable = %v, want none: session head is running", keys(ids(runnable)))
	}
}

func testListByStatus(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	q := admit(t, s, "session-a")
	r := claim(t, s, admit(t, s, "session-b"))

	queued, err := s.ListByStatus(ctx, harness.StatusQueued)
	if err != nil {
		t.Fatalf("ListByStatus(queued): %v", err)
	}
	if len(queued) != 1 || queued[0].ID != q.ID {
		t.Fatalf("queued = %v, want [%s]", keys(ids(queued)), q.ID)
	}
	running, err := s.ListByStatus(ctx, harness.StatusRunning)
	if err != nil {
		t.Fatalf("ListByStatus(running): %v", err)
	}
	if len(running) != 1 || running[0].ID != r.ID {
		t.Fatalf("running = %v, want [%s]", keys(ids(running)), r.ID)
	}
}

func testClaimCAS(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := admit(t, s, "default")

	claimed := claim(t, s, sub)
	if claimed.Status != harness.StatusRunning || claimed.AttemptCount != 1 || claimed.AttemptID == "" {
		t.Fatalf("claimed = %+v, want running with attempt count 1 and an attempt id", claimed)
	}

	// A second claim of a running submission must lose the CAS.
	_, err := s.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   sub.ID,
		AttemptID:      nextID("att"),
		OwnerID:        "owner-2",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	})
	if !errors.Is(err, harness.ErrClaimLost) {
		t.Fatalf("double claim error = %v, want ErrClaimLost", err)
	}
}

func testAttemptMarkers(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := claim(t, s, admit(t, s, "default"))

	first := harness.Attempt{ID: sub.AttemptID, SubmissionID: sub.ID, OwnerID: sub.OwnerID, StartedAt: time.Now()}
	if err := s.StartAttempt(ctx, first); err != nil {
		t.Fatalf("StartAttempt: %v", err)
	}
	second := harness.Attempt{ID: nextID("att"), SubmissionID: sub.ID, OwnerID: "owner-2", StartedAt: time.Now().Add(time.Second)}
	if err := s.StartAttempt(ctx, second); err != nil {
		t.Fatalf("StartAttempt(second): %v", err)
	}

	attempts, err := s.ListAttempts(ctx, sub.ID)
	if err != nil {
		t.Fatalf("ListAttempts: %v", err)
	}
	if len(attempts) != 2 || attempts[0].ID != first.ID || attempts[1].ID != second.ID {
		t.Fatalf("attempts = %+v, want [%s %s] in start order", attempts, first.ID, second.ID)
	}
}

func testLeaseRenewAndExpiry(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	now := time.Now()
	sub := admit(t, s, "default")
	claimed, err := s.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   sub.ID,
		AttemptID:      nextID("att"),
		OwnerID:        "owner-1",
		LeaseExpiresAt: now.Add(50 * time.Millisecond),
	})
	if err != nil {
		t.Fatalf("ClaimSubmission: %v", err)
	}

	// Not yet expired.
	expired, err := s.ListExpiredLeases(ctx, now)
	if err != nil {
		t.Fatalf("ListExpiredLeases: %v", err)
	}
	if len(expired) != 0 {
		t.Fatalf("expired before expiry = %v, want none", keys(ids(expired)))
	}

	// Expired once past the lease.
	expired, err = s.ListExpiredLeases(ctx, now.Add(time.Second))
	if err != nil {
		t.Fatalf("ListExpiredLeases(past): %v", err)
	}
	if len(expired) != 1 || expired[0].ID != sub.ID {
		t.Fatalf("expired past expiry = %v, want [%s]", keys(ids(expired)), sub.ID)
	}

	// Renewal pushes the expiry out.
	if err := s.RenewLease(ctx, harness.LeaseRenewal{
		SubmissionID:   sub.ID,
		AttemptID:      claimed.AttemptID,
		LeaseExpiresAt: now.Add(time.Hour),
	}); err != nil {
		t.Fatalf("RenewLease: %v", err)
	}
	expired, err = s.ListExpiredLeases(ctx, now.Add(time.Second))
	if err != nil {
		t.Fatalf("ListExpiredLeases(after renew): %v", err)
	}
	if len(expired) != 0 {
		t.Fatalf("expired after renew = %v, want none", keys(ids(expired)))
	}

	// A stale attempt cannot renew.
	err = s.RenewLease(ctx, harness.LeaseRenewal{
		SubmissionID:   sub.ID,
		AttemptID:      "stale-attempt",
		LeaseExpiresAt: now.Add(2 * time.Hour),
	})
	if !errors.Is(err, harness.ErrClaimLost) {
		t.Fatalf("stale renew error = %v, want ErrClaimLost", err)
	}
}

func testReleaseSubmission(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := claim(t, s, admit(t, s, "default"))

	// A stale attempt cannot release.
	if err := s.ReleaseSubmission(ctx, harness.SubmissionRelease{SubmissionID: sub.ID, AttemptID: "stale-attempt"}); !errors.Is(err, harness.ErrClaimLost) {
		t.Fatalf("stale release error = %v, want ErrClaimLost", err)
	}

	if err := s.ReleaseSubmission(ctx, harness.SubmissionRelease{
		SubmissionID: sub.ID,
		AttemptID:    sub.AttemptID,
		LastError:    "model exploded",
	}); err != nil {
		t.Fatalf("ReleaseSubmission: %v", err)
	}
	got, err := s.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.Status != harness.StatusQueued {
		t.Fatalf("released status = %q, want queued", got.Status)
	}
	if got.LastError != "model exploded" {
		t.Fatalf("released LastError = %q, want %q", got.LastError, "model exploded")
	}

	// Released work is claimable again, with the attempt count preserved.
	reclaimed := claim(t, s, got)
	if reclaimed.AttemptCount != 2 {
		t.Fatalf("reclaimed attempt count = %d, want 2", reclaimed.AttemptCount)
	}
	if reclaimed.LastError != "model exploded" {
		t.Fatalf("reclaimed LastError = %q, want it preserved across the re-claim", reclaimed.LastError)
	}

	// An error-less release (shutdown, lease reclaim) preserves the stored
	// last error instead of erasing it.
	if err := s.ReleaseSubmission(ctx, harness.SubmissionRelease{SubmissionID: sub.ID, AttemptID: reclaimed.AttemptID}); err != nil {
		t.Fatalf("ReleaseSubmission (no error): %v", err)
	}
	got, err = s.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.LastError != "model exploded" {
		t.Fatalf("LastError after empty release = %q, want %q preserved", got.LastError, "model exploded")
	}
}

func testSettlementPhases(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := claim(t, s, admit(t, s, "default"))

	// A stale attempt cannot reserve.
	if err := s.ReserveSettlement(ctx, sub.ID, "stale-attempt"); !errors.Is(err, harness.ErrClaimLost) {
		t.Fatalf("stale reserve error = %v, want ErrClaimLost", err)
	}

	if err := s.ReserveSettlement(ctx, sub.ID, sub.AttemptID); err != nil {
		t.Fatalf("ReserveSettlement: %v", err)
	}
	got, err := s.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.Status != harness.StatusTerminalizing {
		t.Fatalf("reserved status = %q, want terminalizing", got.Status)
	}

	// Reserving twice loses the CAS: the submission is no longer running.
	if err := s.ReserveSettlement(ctx, sub.ID, sub.AttemptID); !errors.Is(err, harness.ErrClaimLost) {
		t.Fatalf("double reserve error = %v, want ErrClaimLost", err)
	}

	if err := s.FinalizeSettlement(ctx, sub.ID); err != nil {
		t.Fatalf("FinalizeSettlement: %v", err)
	}
	got, err = s.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.Status != harness.StatusSettled {
		t.Fatalf("finalized status = %q, want settled", got.Status)
	}

	// A settled head frees the session for the next submission.
	next := admit(t, s, "default")
	runnable, err := s.ListRunnable(ctx)
	if err != nil {
		t.Fatalf("ListRunnable: %v", err)
	}
	if len(runnable) != 1 || runnable[0].ID != next.ID {
		t.Fatalf("runnable after settle = %v, want [%s]", keys(ids(runnable)), next.ID)
	}
}

func testFinalizeIdempotent(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	sub := claim(t, s, admit(t, s, "default"))
	if err := s.ReserveSettlement(ctx, sub.ID, sub.AttemptID); err != nil {
		t.Fatalf("ReserveSettlement: %v", err)
	}
	if err := s.FinalizeSettlement(ctx, sub.ID); err != nil {
		t.Fatalf("FinalizeSettlement: %v", err)
	}
	// A crash between the phases makes reconciliation finalize again.
	if err := s.FinalizeSettlement(ctx, sub.ID); err != nil {
		t.Fatalf("repeated FinalizeSettlement = %v, want nil (idempotent)", err)
	}
}

func testConversationEnsureAndGet(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	key := harness.SessionKey{Agent: "support", Instance: "acme", Session: "default"}

	if _, err := s.GetConversation(ctx, key); !errors.Is(err, harness.ErrConversationNotFound) {
		t.Fatalf("GetConversation(absent) error = %v, want ErrConversationNotFound", err)
	}

	candidate := harness.Conversation{ID: nextID("conv"), Key: key, CreatedAt: time.Now()}
	created, wasCreated, err := s.EnsureConversation(ctx, candidate)
	if err != nil {
		t.Fatalf("EnsureConversation: %v", err)
	}
	if !wasCreated || created.ID != candidate.ID {
		t.Fatalf("EnsureConversation = (%+v, %v), want created candidate", created, wasCreated)
	}

	// A second ensure returns the existing conversation, not the new
	// candidate.
	other := harness.Conversation{ID: nextID("conv"), Key: key, CreatedAt: time.Now()}
	existing, wasCreated, err := s.EnsureConversation(ctx, other)
	if err != nil {
		t.Fatalf("EnsureConversation(existing): %v", err)
	}
	if wasCreated || existing.ID != candidate.ID {
		t.Fatalf("re-ensure = (%+v, %v), want existing %s", existing, wasCreated, candidate.ID)
	}

	got, err := s.GetConversation(ctx, key)
	if err != nil {
		t.Fatalf("GetConversation: %v", err)
	}
	if got.ID != candidate.ID {
		t.Fatalf("GetConversation = %+v, want id %s", got, candidate.ID)
	}
}

func testRecordAppendAndReadFromOffset(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	convID := nextID("conv")
	var recs []harness.Record
	for i := 0; i < 5; i++ {
		recs = append(recs, harness.Record{
			RecordEnvelope: harness.RecordEnvelope{
				ID:             nextID("rec"),
				Kind:           harness.KindUserMessage,
				ConversationID: convID,
				Session:        "default",
				Time:           time.Now(),
			},
		})
	}
	if err := s.AppendRecords(ctx, convID, recs[:2]); err != nil {
		t.Fatalf("AppendRecords(first batch): %v", err)
	}
	if err := s.AppendRecords(ctx, convID, recs[2:]); err != nil {
		t.Fatalf("AppendRecords(second batch): %v", err)
	}

	all, err := s.ReadRecords(ctx, convID, "")
	if err != nil {
		t.Fatalf("ReadRecords: %v", err)
	}
	if len(all) != len(recs) {
		t.Fatalf("read %d records, want %d", len(all), len(recs))
	}
	for i, rec := range all {
		if rec.ID != recs[i].ID {
			t.Fatalf("record %d id = %s, want %s (append order)", i, rec.ID, recs[i].ID)
		}
	}

	tail, err := s.ReadRecords(ctx, convID, recs[1].ID)
	if err != nil {
		t.Fatalf("ReadRecords(after offset): %v", err)
	}
	if len(tail) != 3 || tail[0].ID != recs[2].ID {
		t.Fatalf("read from offset = %d records starting %s, want 3 starting %s", len(tail), tail[0].ID, recs[2].ID)
	}

	empty, err := s.ReadRecords(ctx, "no-such-conversation", "")
	if err != nil {
		t.Fatalf("ReadRecords(unknown conversation): %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("unknown conversation returned %d records, want 0", len(empty))
	}
}

func testAttachmentPutGet(t *testing.T, s harness.Store) {
	ctx := ctxT(t)
	data := []byte("attachment bytes")

	ref, err := s.PutAttachment(ctx, "text/plain", data)
	if err != nil {
		t.Fatalf("PutAttachment: %v", err)
	}
	if ref.Digest == "" || ref.Size != int64(len(data)) || ref.MediaType != "text/plain" {
		t.Fatalf("ref = %+v, want digest set, size %d, media type text/plain", ref, len(data))
	}

	// Identical bytes are idempotent and digest-stable.
	again, err := s.PutAttachment(ctx, "text/plain", data)
	if err != nil {
		t.Fatalf("PutAttachment(again): %v", err)
	}
	if again.Digest != ref.Digest {
		t.Fatalf("re-put digest = %s, want %s", again.Digest, ref.Digest)
	}

	got, err := s.GetAttachment(ctx, ref.Digest)
	if err != nil {
		t.Fatalf("GetAttachment: %v", err)
	}
	if string(got.Data) != string(data) || got.Ref != ref {
		t.Fatalf("attachment = %+v, want original bytes and ref", got)
	}

	if _, err := s.GetAttachment(ctx, "sha256:absent"); !errors.Is(err, harness.ErrAttachmentNotFound) {
		t.Fatalf("GetAttachment(absent) error = %v, want ErrAttachmentNotFound", err)
	}
}

func ids(subs []harness.Submission) map[string]bool {
	out := make(map[string]bool, len(subs))
	for _, s := range subs {
		out[s.ID] = true
	}
	return out
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
