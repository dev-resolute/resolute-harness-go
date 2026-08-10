package harness_test

import (
	"context"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// seedOrphanChild seeds a queued child submission linked to parentID, as
// the task tool's admission would have left it, with no runtime involved —
// the deterministic starting state for the orphan cascade tests. The child
// conversation carries no records: the cascade never reads it (the wake
// no-ops against the settled parent before the spawn repair runs).
func seedOrphanChild(t *testing.T, store harness.Store, parentID, callID string) harness.Submission {
	t.Helper()
	ctx := context.Background()
	key := harness.SessionKey{
		Agent:    "reviewer",
		Instance: harness.InstanceID("acme:" + safeCallID(callID)),
		Session:  "task-" + safeCallID(callID),
	}
	conv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000ORPH" + callID, Key: key, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(child): %v", err)
	}
	child, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB0000000ORPH" + callID, SessionKey: key, ConversationID: conv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("check it"), CreatedAt: time.Now(),
		ParentSubmissionID: parentID, ParentCallID: callID, Depth: 1,
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(child): %v", err)
	}
	return child
}

// newCascadeRuntime builds the runtime and dispatches the parent BEFORE
// Start, so a seeded child's parent link exists before any settle can fire
// the cascade. startCascadeRuntime starts the coordinator.
func newCascadeRuntime(t *testing.T, cfg harness.Config) (*harness.Runtime, harness.DispatchResult) {
	t.Helper()
	rt, err := harness.NewRuntime(cfg)
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	res, err := rt.Dispatch(context.Background(), harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch(parent): %v", err)
	}
	return rt, res
}

func startCascadeRuntime(t *testing.T, rt *harness.Runtime) {
	t.Helper()
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
}

// waitForRequests polls until the provider's Stream has been called n
// times — a run provably in flight (its cancel registered) before the test
// lets the parent fail.
func waitForRequests(t *testing.T, p *scriptProvider, n int) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for len(p.requests()) < n {
		select {
		case <-deadline:
			t.Fatalf("provider %q saw %d requests, want %d", p.name, len(p.requests()), n)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// waitForSettledRecord polls until the submission_settled record for subID
// exists on the conversation and returns its payload. The cascade's
// queued-child settle lands the record a store call after CancelSubmission
// settles the row, so rt.Wait can win the race and see the row settled
// without its record (the same torn state expireWait ships permanently);
// polling the record sidesteps the window.
func waitForSettledRecord(t *testing.T, rt *harness.Runtime, conversationID, submissionID string) harness.SettledPayload {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		recs, err := rt.Records(context.Background(), conversationID, "")
		if err != nil {
			t.Fatalf("Records: %v", err)
		}
		for _, rec := range recs {
			if rec.Kind != harness.KindSubmissionSettled || rec.SubmissionID != submissionID {
				continue
			}
			var p harness.SettledPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(submission_settled): %v", err)
			}
			return p
		}
		select {
		case <-deadline:
			t.Fatalf("no submission_settled record for %s on %s", submissionID, conversationID)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// claimDenyStore refuses every claim of one submission, keeping it queued
// forever — the deterministic shape of "admitted but never run" for the
// cascade's queued-child arm.
type claimDenyStore struct {
	harness.Store
	denyID string
}

func (s *claimDenyStore) ClaimSubmission(ctx context.Context, claim harness.SubmissionClaim) (harness.Submission, error) {
	if claim.SubmissionID == s.denyID {
		return harness.Submission{}, harness.ErrClaimLost
	}
	return s.Store.ClaimSubmission(ctx, claim)
}

// A parent settling failed with a RUNNING child triggers the orphan
// cascade: the store flags the child and its in-process run context is
// cancelled, so the child's own post-drive switch settles it
// failed/cancelled_by_parent — not run_failed, never released for retry.
func TestOrphanCascadeCancelsRunningChild(t *testing.T) {
	t.Parallel()
	store := memory.New()
	parentGate := make(chan struct{})
	childGate := make(chan struct{}) // never closed: the child stays in flight
	// The parent's first (only) model call waits until the child is
	// provably in flight, then fails fatally (empty script).
	parent := &scriptProvider{name: "mock", gates: []<-chan struct{}{parentGate}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt, res := newCascadeRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	childSub := seedOrphanChild(t, store, res.SubmissionID, "call-1")
	startCascadeRuntime(t, rt)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Hold the parent's failure until the child's run is mid-flight.
	waitForRequests(t, child, 1)
	close(parentGate)

	childSettled, err := rt.Wait(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if childSettled.Status != harness.SettledFailed || childSettled.ErrorCode != harness.SettledErrCancelled {
		t.Errorf("child settled = %+v, want failed/cancelled_by_parent", childSettled)
	}
	parentSettled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledFailed || parentSettled.ErrorCode != harness.SettledErrRunFailed {
		t.Errorf("parent settled = %+v, want failed/run_failed", parentSettled)
	}

	// The cancelled child landed no outcome of its own: the parent log
	// stays free of a tool_outcome for the never-answered call.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindToolOutcome); n != 0 {
		t.Errorf("tool_outcome records on the parent log = %d, want 0 (the seeded call was never answered)", n)
	}
}

// A parent settling failed with a QUEUED child settles the child directly:
// no attempt ever runs, and the settled record lands with
// failed/cancelled_by_parent.
func TestOrphanCascadeSettlesQueuedChild(t *testing.T) {
	t.Parallel()
	store := &claimDenyStore{Store: memory.New()}
	// The parent fails fatally on its first model call (empty script).
	parent := &scriptProvider{name: "mock"}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}}
	rt, res := newCascadeRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	childSub := seedOrphanChild(t, store, res.SubmissionID, "call-1")
	store.denyID = childSub.ID
	startCascadeRuntime(t, rt)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	childSettled := waitForSettledRecord(t, rt, childSub.ConversationID, childSub.ID)
	if childSettled.Status != harness.SettledFailed || childSettled.ErrorCode != harness.SettledErrCancelled {
		t.Errorf("child settled = %+v, want failed/cancelled_by_parent", childSettled)
	}
	if childSettled.Error != "cancelled: parent settled" {
		t.Errorf("child settled error = %q, want %q", childSettled.Error, "cancelled: parent settled")
	}
	if n := len(child.requests()); n != 0 {
		t.Errorf("child provider calls = %d, want 0 (settled without ever running)", n)
	}

	// The cancel is durable on the row: settled straight from queued, with
	// the cascade's reason recorded.
	childFinal, err := store.GetSubmission(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("GetSubmission(child): %v", err)
	}
	if childFinal.Status != harness.StatusSettled {
		t.Errorf("child status = %s, want settled", childFinal.Status)
	}
	if want := "parent " + res.SubmissionID + " settled"; childFinal.LastError != want {
		t.Errorf("child LastError = %q, want %q", childFinal.LastError, want)
	}

	parentSettled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledFailed || parentSettled.ErrorCode != harness.SettledErrRunFailed {
		t.Errorf("parent settled = %+v, want failed/run_failed", parentSettled)
	}
}

// An already-settled child is not re-cancelled: the cascade skips it,
// leaving its settled record and row untouched.
func TestOrphanCascadeSkipsSettledChild(t *testing.T) {
	t.Parallel()
	store := memory.New()
	// The parent fails fatally on its first model call (empty script).
	parent := &scriptProvider{name: "mock"}
	child := &scriptProvider{name: "mock"}
	rt, res := newCascadeRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	childSub := seedOrphanChild(t, store, res.SubmissionID, "call-1")

	// Settle the child directly through the store — succeeded, with its
	// record durable — before the runtime starts.
	ctx := context.Background()
	const childAttempt = "01A0DEADATTEMPT0000000ORP"
	if _, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   childSub.ID,
		AttemptID:      childAttempt,
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(time.Minute),
	}); err != nil {
		t.Fatalf("ClaimSubmission(child): %v", err)
	}
	if err := store.ReserveSettlement(ctx, childSub.ID, childAttempt); err != nil {
		t.Fatalf("ReserveSettlement(child): %v", err)
	}
	if err := store.AppendRecords(ctx, childSub.ConversationID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000ORPH", harness.KindSubmissionSettled,
			childSub.ConversationID, childSub.SessionKey.Session, childSub.ID, childAttempt,
			&harness.SettledPayload{Status: harness.SettledSucceeded}),
	}); err != nil {
		t.Fatalf("AppendRecords(child settled): %v", err)
	}
	if err := store.FinalizeSettlement(ctx, childSub.ID); err != nil {
		t.Fatalf("FinalizeSettlement(child): %v", err)
	}
	startCascadeRuntime(t, rt)
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// The parent settles failed; the cascade sees the child already
	// settled and skips it.
	parentSettled, err := rt.Wait(wctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledFailed {
		t.Fatalf("parent settled = %+v, want failed", parentSettled)
	}

	childFinal, err := store.GetSubmission(wctx, childSub.ID)
	if err != nil {
		t.Fatalf("GetSubmission(child): %v", err)
	}
	if childFinal.Status != harness.StatusSettled || childFinal.LastError != "" {
		t.Errorf("child = %+v, want the untouched succeeded settlement (no cancel reason)", childFinal)
	}
	settled, err := rt.Wait(wctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Errorf("child settled = %+v, want the original succeeded outcome, not a cascade overwrite", settled)
	}
	if n := countKind(mustRecords(t, rt, childSub.ConversationID), harness.KindSubmissionSettled); n != 1 {
		t.Errorf("child settled records = %d, want exactly 1 (the cascade must not re-author)", n)
	}
}

// The cascade is gated on the orphan policy: with OnParentTerminal set to
// anything other than CancelChildren (only one policy exists in v1, so an
// out-of-range value stands in for a future policy), a settling parent
// leaves its live children alone — the default resolves to CancelChildren
// (config_test.go pins the resolution; the tests above pin the cascade).
func TestOrphanCascadeGatedOnPolicy(t *testing.T) {
	t.Parallel()
	store := memory.New()
	parentGate := make(chan struct{})
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", gates: []<-chan struct{}{parentGate}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt, res := newCascadeRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{
		OnParentTerminal: harness.OrphanPolicy(99),
	}))
	childSub := seedOrphanChild(t, store, res.SubmissionID, "call-1")
	startCascadeRuntime(t, rt)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// The parent fails while the child is mid-flight; no cascade fires.
	waitForRequests(t, child, 1)
	close(parentGate)
	parentSettled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledFailed {
		t.Fatalf("parent settled = %+v, want failed", parentSettled)
	}

	// The child runs on undisturbed and settles succeeded — had the
	// cascade fired, its run context would have been cancelled into
	// failed/cancelled_by_parent.
	close(childGate)
	childSettled, err := rt.Wait(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if childSettled.Status != harness.SettledSucceeded {
		t.Errorf("child settled = %+v, want succeeded (the policy gate must hold the cascade back)", childSettled)
	}
	childFinal, err := store.GetSubmission(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("GetSubmission(child): %v", err)
	}
	if childFinal.CancelRequested {
		t.Error("child row flagged CancelRequested — the cascade touched it despite the policy gate")
	}
}

// mustRecords reads a conversation's full log.
func mustRecords(t *testing.T, rt *harness.Runtime, conversationID string) []harness.Record {
	t.Helper()
	recs, err := rt.Records(context.Background(), conversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	return recs
}
