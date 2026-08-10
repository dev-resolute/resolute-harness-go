package harness_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// userTexts flattens the text of the user messages in a provider request.
func userTexts(req llm.LLMRequest) []string {
	var out []string
	for _, m := range req.Messages {
		if m.Role != "user" {
			continue
		}
		if tc, ok := m.Content.(llm.TextContent); ok {
			out = append(out, tc.Text)
		}
	}
	return out
}

// A child that settles succeeded hands its answer to the parent: the wake
// appends the parent's tool_outcome for the pending call (a fresh ULID, so
// the append-ordered record IDs the stores rely on stay monotonic),
// requeues the parent, and the resume drive's provider request ends in
// that tool result — the parent settles succeeded.
func TestWakeLandsOutcomeAndRequeues(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "reviewer", "check it"),
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}
	childSub := children[0]

	close(childGate)
	childSettled, err := rt.Wait(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if childSettled.Status != harness.SettledSucceeded {
		t.Fatalf("child settled = %+v, want succeeded", childSettled)
	}

	// The wake landed the outcome before the requeue: the record carries a
	// fresh ULID ordered after every prior record on the parent
	// conversation (the stores rely on append-ordered IDs for
	// ReadRecords(afterID)), and the content is the child's final
	// assistant text (no structured result was requested).
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome — the wake did not land")
	}
	for _, prior := range recs[:outcomeIdx] {
		if recs[outcomeIdx].ID <= prior.ID {
			t.Errorf("outcome record ID %q not ordered after prior record %q — append-order invariant violated", recs[outcomeIdx].ID, prior.ID)
		}
	}
	var outcome harness.ToolOutcomePayload
	if err := recs[outcomeIdx].DecodePayload(&outcome); err != nil {
		t.Fatalf("DecodePayload(tool_outcome): %v", err)
	}
	if outcome.CallID != "call-1" || outcome.ToolName != "task" {
		t.Errorf("outcome = %+v, want the outcome of task call call-1", outcome)
	}
	if outcome.IsError {
		t.Error("outcome IsError = true, want false for a succeeded child")
	}
	if outcome.Content != "looks good" {
		t.Errorf("outcome content = %q, want the child's final text %q", outcome.Content, "looks good")
	}

	// The parent requeued and its resume drive settled it.
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}

	// The resume drive re-prompted nothing: the provider's second request
	// carries the original history ending in the wake outcome.
	reqs := parent.requests()
	if len(reqs) != 2 {
		t.Fatalf("parent provider calls = %d, want 2 (prompt + resume)", len(reqs))
	}
	resume := reqs[1]
	if n := len(userTexts(resume)); n != 1 {
		t.Errorf("resume request user messages = %d, want 1 (no re-appended input)", n)
	}
	last := resume.Messages[len(resume.Messages)-1]
	tr, ok := last.Content.(llm.ToolResultContent)
	if !ok {
		t.Fatalf("resume tail = %T, want a tool result (Resume requires it)", last.Content)
	}
	if tr.CallID != "call-1" || tr.Content != "looks good" {
		t.Errorf("resume tail = %+v, want the wake outcome for call-1", tr)
	}

	// The claim consumed PendingResume (the storetest pins the contract).
	stored, err := store.GetSubmission(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("GetSubmission(parent): %v", err)
	}
	if stored.PendingResume {
		t.Error("stored row still marked PendingResume after the resume drive claimed it")
	}
}

// A child that settles failed hands the failure to the parent: the wake
// outcome is IsError with the Error text and code, and the parent's resume
// drive still completes.
func TestWakeFailureDiagnostics(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "reviewer", "check it"),
		textTurn("handled the failure"),
	}}
	// No script: the child's first model call is a fatal provider error
	// (gated so the parked parent is observable before the failure lands).
	child := &scriptProvider{name: "mock", gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}

	close(childGate)
	childSettled, err := rt.Wait(ctx, children[0].ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if childSettled.Status != harness.SettledFailed {
		t.Fatalf("child settled = %+v, want failed", childSettled)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome — the wake did not land")
	}
	var outcome harness.ToolOutcomePayload
	if err := recs[outcomeIdx].DecodePayload(&outcome); err != nil {
		t.Fatalf("DecodePayload(tool_outcome): %v", err)
	}
	if !outcome.IsError {
		t.Error("outcome IsError = false, want true for a failed child")
	}
	if !strings.Contains(outcome.Content, "unexpected call") {
		t.Errorf("outcome content = %q, want the child's error text", outcome.Content)
	}
	if !strings.Contains(outcome.Content, "run_failed") {
		t.Errorf("outcome content = %q, want the settled error code", outcome.Content)
	}

	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded (the error outcome resumes, not fails, the parent)", parentSettled)
	}
}

// With two children in flight the first settlement does not requeue the
// parent; the last one does — all outcomes are in context on resume.
func TestWakeWaitsForLastChild(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "first")},
			llm.ToolCallEndEvent{CallID: "call-2", ToolName: "task", Args: taskArgs("reviewer", "second")},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn("both back"),
	}}
	// The second child to reach the model is gated, so exactly one child
	// can settle before the test releases it.
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("first done"),
		textTurn("second done"),
	}, gates: []<-chan struct{}{nil, childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 2 {
		t.Fatalf("children = %d, want 2", len(children))
	}

	// Wait for exactly one child to settle. Its wake has already run by the
	// time the settled status is observable (the wake lands inside the
	// settlement reservation), so the parked parent proves the wake waited
	// for the sibling.
	settled := 0
	for settled != 1 {
		subs, err := store.ListChildSubmissions(ctx, parentSub.ID)
		if err != nil {
			t.Fatalf("ListChildSubmissions: %v", err)
		}
		settled = 0
		for _, ch := range subs {
			if ch.Status == harness.StatusSettled {
				settled++
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("no child settled: %v", ctx.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
	if got, err := store.GetSubmission(ctx, parentSub.ID); err != nil {
		t.Fatalf("GetSubmission(parent): %v", err)
	} else if got.Status != harness.StatusWaiting {
		t.Fatalf("parent status after first child settle = %s, want waiting (a sibling is still in flight)", got.Status)
	}

	// The last child's settlement requeues the parent; both outcomes are in
	// context and the resume drive settles it.
	close(childGate)
	for _, ch := range children {
		if _, err := rt.Wait(ctx, ch.ID); err != nil {
			t.Fatalf("Wait(child %s): %v", ch.ID, err)
		}
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the last child's wake", parentSettled)
	}
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindToolOutcome); n != 2 {
		t.Errorf("tool_outcome records = %d, want 2 (one wake outcome per child)", n)
	}
}

// The wake runs twice per child settlement — inside the two-phase
// reservation and again after finalization (closing the concurrent-settle
// race between siblings); the CallID check keeps the replay from
// double-authoring: exactly one outcome lands.
func TestWakeIdempotent(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "reviewer", "check it"),
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}
	close(childGate)
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if _, err := rt.Wait(ctx, parentSub.ID); err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	outcomes := 0
	for _, rec := range recs {
		if rec.Kind == harness.KindToolOutcome {
			outcomes++
		}
	}
	if outcomes != 1 {
		t.Errorf("tool_outcome records = %d, want exactly 1 after two wake runs", outcomes)
	}
}

// A PendingResume re-drive Resumes instead of Prompting: no second
// user_message input record is authored (appendInputRecord dedupes by
// submission ID) and the provider sees the full history ending in the wake
// outcome — not a re-appended input.
func TestResumeDriveUsesResumeNotPrompt(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "reviewer", "check it"),
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}
	close(childGate)
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded", parentSettled)
	}

	// Exactly one user_message record across both drives.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	userMsgs := 0
	for _, rec := range recs {
		if rec.Kind == harness.KindUserMessage && rec.SubmissionID == parentSub.ID {
			userMsgs++
		}
	}
	if userMsgs != 1 {
		t.Errorf("user_message records for the parent submission = %d, want 1 (a Prompt re-drive would have authored a second)", userMsgs)
	}

	// The prompt request carries no tool result; the resume request carries
	// the full history ending in the wake outcome and no new input.
	reqs := parent.requests()
	if len(reqs) != 2 {
		t.Fatalf("parent provider calls = %d, want 2 (prompt + resume)", len(reqs))
	}
	for _, m := range reqs[0].Messages {
		if _, ok := m.Content.(llm.ToolResultContent); ok {
			t.Error("prompt request carries a tool result — history leaked into the first drive")
		}
	}
	resume := reqs[1]
	if n := len(userTexts(resume)); n != 1 {
		t.Errorf("resume request user messages = %d, want 1 (Prompt would re-append the input)", n)
	}
	last := resume.Messages[len(resume.Messages)-1]
	if _, ok := last.Content.(llm.ToolResultContent); !ok {
		t.Errorf("resume tail = %T, want the wake outcome (tool result)", last.Content)
	}
}

// seedRecord builds one durable record for crash-state seeding, in the
// style of engine_test.go's crash boundaries.
func seedRecord(id string, kind harness.RecordKind, convID, session, subID, attemptID string, payload any) harness.Record {
	blob, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	return harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             id,
			Kind:           kind,
			ConversationID: convID,
			Session:        session,
			SubmissionID:   subID,
			AttemptID:      attemptID,
			Time:           time.Now(),
		},
		Payload: blob,
	}
}

// A child can settle while the parent's conversation still lacks the
// task_spawned record for its call (a fast settle before the consumer
// authored it, or the crash window between admission and the spawn event).
// The wake repairs the record from durable child state BEFORE appending the
// outcome — recovering the record ID from the child conversation's
// ParentRef and the Agent/Instance/Prompt from the child row — so the log
// order is call→spawn→outcome universally, and the parent is requeued.
func TestWakeRepairsMissingSpawnRecord(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := memory.New()

	const (
		parentAttempt = "01A0DEADATTEMPT00000000PA"
		spawnRecID    = "01A0SEEDREC0000000000SPWN"
	)

	// The parent: claimed then parked in waiting with an unbounded wait,
	// its conversation carrying the task call but NO spawn record — the
	// crash/fast-settle window this test replays.
	pKey := harness.SessionKey{Agent: "triage", Instance: "acme", Session: "default"}
	pConv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000PAR", Key: pKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(parent): %v", err)
	}
	parent, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000PAR", SessionKey: pKey, ConversationID: pConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("triage this"), CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(parent): %v", err)
	}
	if _, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
		SubmissionID:   parent.ID,
		AttemptID:      parentAttempt,
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(-time.Minute),
	}); err != nil {
		t.Fatalf("ClaimSubmission(parent): %v", err)
	}
	if err := store.WaitSubmission(ctx, harness.SubmissionWait{SubmissionID: parent.ID, AttemptID: parentAttempt}); err != nil {
		t.Fatalf("WaitSubmission(parent): %v", err)
	}
	if err := store.AppendRecords(ctx, pConv.ID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000000R1", harness.KindConversationCreated, pConv.ID, "default", "", "",
			&harness.ConversationCreatedPayload{Agent: "triage", Instance: "acme", Session: "default"}),
		seedRecord("01A0SEEDREC0000000000000R2", harness.KindUserMessage, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.UserMessagePayload{Body: "triage this"}),
		seedRecord("01A0SEEDREC0000000000000R3", harness.KindAssistantToolCall, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.AssistantToolCallPayload{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "check it")}),
	}); err != nil {
		t.Fatalf("AppendRecords(parent): %v", err)
	}

	// The child: queued, linked back to the call; its conversation's
	// conversation_created carries the ParentRef with the spawn record ID
	// the wake must recover.
	cKey := harness.SessionKey{
		Agent:    "reviewer",
		Instance: harness.InstanceID("acme:" + safeCallID("call-1")),
		Session:  "task-" + safeCallID("call-1"),
	}
	cConv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000CHA", Key: cKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(child): %v", err)
	}
	child, err := store.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000CHA", SessionKey: cKey, ConversationID: cConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("check it"), CreatedAt: time.Now(),
		ParentSubmissionID: parent.ID, ParentCallID: "call-1", Depth: 1,
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(child): %v", err)
	}
	if err := store.AppendRecords(ctx, cConv.ID, []harness.Record{
		seedRecord("01A0SEEDREC0000000000000C1", harness.KindConversationCreated, cConv.ID, cKey.Session, "", "",
			&harness.ConversationCreatedPayload{Agent: "reviewer", Instance: cKey.Instance,
				Session: cKey.Session, ParentRef: &harness.ParentRef{ConversationID: pConv.ID, SpawnRecordID: spawnRecID}}),
	}); err != nil {
		t.Fatalf("AppendRecords(child): %v", err)
	}

	// The child runs and settles; its wake faces a parent log with no
	// spawn record for call-1. The parent's single scripted turn serves
	// the resume drive the requeue triggers.
	parentProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("triage done")}}
	childProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("looks good")}}
	rt := startRuntime(t, subagentConfig(store, parentProv, childProv, harness.SubagentLimits{}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	childSettled, err := rt.Wait(wctx, child.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if childSettled.Status != harness.SettledSucceeded {
		t.Fatalf("child settled = %+v, want succeeded", childSettled)
	}

	// The repaired spawn record landed with the recovered ID BETWEEN the
	// call and the wake outcome.
	recs, err := rt.Records(wctx, pConv.ID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	callIdx := findRecord(recs, harness.KindAssistantToolCall)
	spawnIdx := findRecord(recs, harness.KindTaskSpawned)
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if callIdx < 0 || spawnIdx < 0 || outcomeIdx < 0 {
		t.Fatalf("parent log: call at %d, spawn at %d, outcome at %d — all three must exist", callIdx, spawnIdx, outcomeIdx)
	}
	if !(callIdx < spawnIdx && spawnIdx < outcomeIdx) {
		t.Fatalf("record order: call at %d, spawn at %d, outcome at %d, want call→spawn→outcome", callIdx, spawnIdx, outcomeIdx)
	}
	if recs[spawnIdx].ID != spawnRecID {
		t.Errorf("repaired spawn record ID = %q, want %q recovered from the child conversation's ParentRef", recs[spawnIdx].ID, spawnRecID)
	}
	var spawn harness.TaskSpawnedPayload
	if err := recs[spawnIdx].DecodePayload(&spawn); err != nil {
		t.Fatalf("DecodePayload(task_spawned): %v", err)
	}
	wantSpawn := harness.TaskSpawnedPayload{
		CallID:              "call-1",
		Agent:               "reviewer",
		ChildInstance:       string(cKey.Instance),
		ChildConversationID: cConv.ID,
		ChildSubmissionID:   child.ID,
		Prompt:              "check it",
	}
	if spawn != wantSpawn {
		t.Errorf("repaired task_spawned payload = %+v, want %+v", spawn, wantSpawn)
	}

	// The wake requeued the parent; the resume drive settled it.
	parentSettled, err := rt.Wait(wctx, parent.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
	if n := len(parentProv.requests()); n != 1 {
		t.Errorf("parent provider calls = %d, want 1 (the resume drive)", n)
	}
}

// settleInterleaveStore deterministically replays the concurrent-settle race
// the recovery path's post-finalize wake (engine.go finalizeInterrupted)
// closes: when the crashed terminalizing child's in-reservation wake lists
// the parent's children, a sibling's full settle lands in the window before
// FinalizeSettlement — while both of the sibling's own wakes skipped the
// requeue because the crashed child was still terminalizing. The racing wake
// observes the pre-settle snapshot, exactly as a concurrent wake would.
type settleInterleaveStore struct {
	harness.Store
	parentID       string
	parentConvID   string
	parentSession  string
	siblingID      string
	siblingConvID  string
	siblingSession string
	siblingAttempt string
	siblingCallID  string

	once sync.Once
	mu   sync.Mutex
	err  error
}

func (s *settleInterleaveStore) ListChildSubmissions(ctx context.Context, parentID string) ([]harness.Submission, error) {
	children, err := s.Store.ListChildSubmissions(ctx, parentID)
	if err != nil {
		return nil, err
	}
	if parentID == s.parentID {
		s.once.Do(func() { s.setErr(s.settleSibling(ctx)) })
	}
	return children, nil
}

// settleSibling replays the sibling settle()'s end state through the raw
// store: reservation, settled record, the wake outcome its in-reservation
// wake appended to the parent before skipping the requeue, finalization.
func (s *settleInterleaveStore) settleSibling(ctx context.Context) error {
	if err := s.Store.ReserveSettlement(ctx, s.siblingID, s.siblingAttempt); err != nil {
		return fmt.Errorf("reserve sibling settlement: %w", err)
	}
	settled := seedRecord("01ZZSEEDHOOK00000000000SET", harness.KindSubmissionSettled,
		s.siblingConvID, s.siblingSession, s.siblingID, s.siblingAttempt,
		&harness.SettledPayload{Status: harness.SettledSucceeded})
	if err := s.Store.AppendRecords(ctx, s.siblingConvID, []harness.Record{settled}); err != nil {
		return fmt.Errorf("append sibling settled record: %w", err)
	}
	outcome := seedRecord("01ZZSEEDHOOK0000000000WAKE", harness.KindToolOutcome,
		s.parentConvID, s.parentSession, s.parentID, "",
		&harness.ToolOutcomePayload{CallID: s.siblingCallID, ToolName: "task", Content: "second answer"})
	if err := s.Store.AppendRecords(ctx, s.parentConvID, []harness.Record{outcome}); err != nil {
		return fmt.Errorf("append sibling wake outcome: %w", err)
	}
	if err := s.Store.FinalizeSettlement(ctx, s.siblingID); err != nil {
		return fmt.Errorf("finalize sibling settlement: %w", err)
	}
	return nil
}

func (s *settleInterleaveStore) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

func (s *settleInterleaveStore) settleErr() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// A child that crashes mid-terminalizing (settled record present, settlement
// never finalized) is recovered by startup reconcile; its post-finalize wake
// must requeue the waiting parent.
//
// The pin needs a sibling: the crashed child's in-reservation wake (inside
// appendSettledRecordOnce) already requeues a parent whose other children
// are settled — the post-finalize wake exists for the window in which a
// sibling settles between that wake and FinalizeSettlement, both of the
// sibling's wakes having skipped the requeue on the still-terminalizing
// crashed child. settleInterleaveStore replays that interleaving.
func TestRecoveryWakeAfterTerminalizingChild(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	base := memory.New()

	const (
		parentAttempt  = "01A0DEADATTEMPT00000000PA"
		childAttempt   = "01A0DEADATTEMPT00000000CA"
		siblingAttempt = "01A0DEADATTEMPT00000000CB"
	)
	claim := func(sub harness.Submission, attemptID string) {
		t.Helper()
		if _, err := base.ClaimSubmission(ctx, harness.SubmissionClaim{
			SubmissionID:   sub.ID,
			AttemptID:      attemptID,
			OwnerID:        "dead-owner",
			LeaseExpiresAt: time.Now().Add(-time.Minute),
		}); err != nil {
			t.Fatalf("ClaimSubmission(%s): %v", sub.ID, err)
		}
	}

	// The parent: admitted, claimed, then parked in waiting — WaitSubmission
	// releases the lease and marks the row PendingResume.
	pKey := harness.SessionKey{Agent: "triage", Instance: "acme", Session: "default"}
	pConv, _, err := base.EnsureConversation(ctx, harness.Conversation{
		ID: "01A0SEEDCONV00000000000PAR", Key: pKey, CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation(parent): %v", err)
	}
	parent, err := base.AdmitSubmission(ctx, harness.Submission{
		ID: "01A0SEEDSUB000000000000PAR", SessionKey: pKey, ConversationID: pConv.ID,
		Status: harness.StatusQueued, Input: harness.UserMessage("triage this"), CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission(parent): %v", err)
	}
	claim(parent, parentAttempt)
	if err := base.WaitSubmission(ctx, harness.SubmissionWait{SubmissionID: parent.ID, AttemptID: parentAttempt}); err != nil {
		t.Fatalf("WaitSubmission(parent): %v", err)
	}

	// Two children, keyed as the spawn path keys them. Child A crashed
	// mid-terminalizing; sibling B was still in flight (running, dead
	// owner) at the crash.
	seedChild := func(callID, prompt, convID, subID string) (harness.Conversation, harness.Submission) {
		t.Helper()
		key := harness.SessionKey{
			Agent:    "reviewer",
			Instance: harness.InstanceID("acme:" + safeCallID(callID)),
			Session:  "task-" + safeCallID(callID),
		}
		conv, _, err := base.EnsureConversation(ctx, harness.Conversation{ID: convID, Key: key, CreatedAt: time.Now()})
		if err != nil {
			t.Fatalf("EnsureConversation(%s): %v", callID, err)
		}
		sub, err := base.AdmitSubmission(ctx, harness.Submission{
			ID: subID, SessionKey: key, ConversationID: conv.ID,
			Status: harness.StatusQueued, Input: harness.UserMessage(prompt), CreatedAt: time.Now(),
			ParentSubmissionID: parent.ID, ParentCallID: callID, Depth: 1,
		})
		if err != nil {
			t.Fatalf("AdmitSubmission(%s): %v", callID, err)
		}
		return conv, sub
	}
	childConvA, childA := seedChild("call-1", "check it", "01A0SEEDCONV00000000000CHA", "01A0SEEDSUB000000000000CHA")
	childConvB, childB := seedChild("call-2", "review it", "01A0SEEDCONV00000000000CHB", "01A0SEEDSUB000000000000CHB")
	claim(childA, childAttempt)
	claim(childB, siblingAttempt) // left running: in flight at the crash

	// The parent conversation the parked run left behind: prompt, both task
	// calls, both spawn records. IDs stay below runtime-minted ULIDs and
	// increase in append order.
	const (
		spawnRecA = "01A0SEEDREC0000000000SPWNA"
		spawnRecB = "01A0SEEDREC0000000000SPWNB"
	)
	recID := func(n int) string { return fmt.Sprintf("01A0SEEDREC%015d", n) }
	appendSeed := func(convID string, recs []harness.Record) {
		t.Helper()
		if err := base.AppendRecords(ctx, convID, recs); err != nil {
			t.Fatalf("AppendRecords(%s): %v", convID, err)
		}
	}
	appendSeed(pConv.ID, []harness.Record{
		seedRecord(recID(1), harness.KindConversationCreated, pConv.ID, "default", "", "",
			&harness.ConversationCreatedPayload{Agent: "triage", Instance: "acme", Session: "default"}),
		seedRecord(recID(2), harness.KindUserMessage, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.UserMessagePayload{Body: "triage this"}),
		seedRecord(recID(3), harness.KindAssistantToolCall, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.AssistantToolCallPayload{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "check it")}),
		seedRecord(recID(4), harness.KindAssistantToolCall, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.AssistantToolCallPayload{CallID: "call-2", ToolName: "task", Args: taskArgs("reviewer", "review it")}),
		seedRecord(spawnRecA, harness.KindTaskSpawned, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.TaskSpawnedPayload{CallID: "call-1", Agent: "reviewer", ChildInstance: string(childA.SessionKey.Instance),
				ChildConversationID: childConvA.ID, ChildSubmissionID: childA.ID, Prompt: "check it"}),
		seedRecord(spawnRecB, harness.KindTaskSpawned, pConv.ID, "default", parent.ID, parentAttempt,
			&harness.TaskSpawnedPayload{CallID: "call-2", Agent: "reviewer", ChildInstance: string(childB.SessionKey.Instance),
				ChildConversationID: childConvB.ID, ChildSubmissionID: childB.ID, Prompt: "review it"}),
	})

	// Child A's conversation: its final answer and the settled record that
	// landed before the crash — settlement itself never finalized.
	answerBody, _ := json.Marshal("child answer")
	appendSeed(childConvA.ID, []harness.Record{
		seedRecord(recID(11), harness.KindConversationCreated, childConvA.ID, childA.SessionKey.Session, "", "",
			&harness.ConversationCreatedPayload{Agent: "reviewer", Instance: childA.SessionKey.Instance,
				Session: childA.SessionKey.Session, ParentRef: &harness.ParentRef{ConversationID: pConv.ID, SpawnRecordID: spawnRecA}}),
		seedRecord(recID(12), harness.KindUserMessage, childConvA.ID, childA.SessionKey.Session, childA.ID, childAttempt,
			&harness.UserMessagePayload{Body: "check it"}),
		seedRecord(recID(13), harness.KindAssistantMessageCompleted, childConvA.ID, childA.SessionKey.Session, childA.ID, childAttempt,
			&harness.AssistantMessageCompletedPayload{Message: harness.MessagePayload{Role: "assistant", Type: "text", Body: answerBody}}),
		seedRecord(recID(14), harness.KindSubmissionSettled, childConvA.ID, childA.SessionKey.Session, childA.ID, childAttempt,
			&harness.SettledPayload{Status: harness.SettledSucceeded}),
	})
	if err := base.ReserveSettlement(ctx, childA.ID, childAttempt); err != nil {
		t.Fatalf("ReserveSettlement(child A): %v", err)
	}

	// Sibling B's conversation: just the admission state — its settle is
	// replayed by the interleave store.
	appendSeed(childConvB.ID, []harness.Record{
		seedRecord(recID(15), harness.KindConversationCreated, childConvB.ID, childB.SessionKey.Session, "", "",
			&harness.ConversationCreatedPayload{Agent: "reviewer", Instance: childB.SessionKey.Instance,
				Session: childB.SessionKey.Session, ParentRef: &harness.ParentRef{ConversationID: pConv.ID, SpawnRecordID: spawnRecB}}),
		seedRecord(recID(16), harness.KindUserMessage, childConvB.ID, childB.SessionKey.Session, childB.ID, siblingAttempt,
			&harness.UserMessagePayload{Body: "review it"}),
	})

	store := &settleInterleaveStore{
		Store:          base,
		parentID:       parent.ID,
		parentConvID:   pConv.ID,
		parentSession:  "default",
		siblingID:      childB.ID,
		siblingConvID:  childConvB.ID,
		siblingSession: childB.SessionKey.Session,
		siblingAttempt: siblingAttempt,
		siblingCallID:  "call-2",
	}
	// The parent's single scripted turn serves the resume drive; the child
	// provider has no script — both children recover from durable state, so
	// any child run is a fatal error.
	parentProv := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{textTurn("triage done")}}
	childProv := &scriptProvider{name: "mock"}
	rt := startRuntime(t, subagentConfig(store, parentProv, childProv, harness.SubagentLimits{}))

	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Startup reconcile honored the seeded settled record and finalized the
	// crashed child without re-running it.
	childSettled, err := rt.Wait(wctx, childA.ID)
	if err != nil {
		t.Fatalf("Wait(child A): %v", err)
	}
	if childSettled.Status != harness.SettledSucceeded {
		t.Fatalf("child A settled = %+v, want the seeded succeeded outcome", childSettled)
	}
	if err := store.settleErr(); err != nil {
		t.Fatalf("interleaved sibling settle: %v", err)
	}

	// The post-finalize wake requeued the parent (the sibling settle landed
	// inside the crashed child's finalize window, so the in-reservation
	// wake had already skipped the requeue); the resume drive settled it.
	parentSettled, err := rt.Wait(wctx, parent.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the recovery wake", parentSettled)
	}

	// The parent conversation gained exactly one tool_outcome per call: the
	// crashed child's from its wake, the sibling's from its interleaved
	// settle.
	recs, err := rt.Records(ctx, pConv.ID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	contents := map[string]string{}
	for _, rec := range recs {
		if rec.Kind != harness.KindToolOutcome {
			continue
		}
		var p harness.ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload(tool_outcome): %v", err)
		}
		if p.IsError {
			t.Errorf("outcome for %s IsError = true, want false", p.CallID)
		}
		if _, dup := contents[p.CallID]; dup {
			t.Errorf("duplicate tool_outcome for %s — the wake replay must not double-author", p.CallID)
		}
		contents[p.CallID] = p.Content
	}
	if len(contents) != 2 || contents["call-1"] != "child answer" || contents["call-2"] != "second answer" {
		t.Errorf("outcomes = %v, want exactly {call-1: child answer, call-2: second answer}", contents)
	}

	// Only the parent's resume drive hit a provider; neither child re-ran.
	if n := len(parentProv.requests()); n != 1 {
		t.Errorf("parent provider calls = %d, want 1 (the resume drive)", n)
	}
	if n := len(childProv.requests()); n != 0 {
		t.Errorf("child provider calls = %d, want 0 (both children recovered from durable state)", n)
	}
}
