package harness_test

import (
	"context"
	"strings"
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
// appends the parent's tool_outcome for the pending call (record ID derived
// from the child submission ID), requeues the parent, and the resume
// drive's provider request ends in that tool result — the parent settles
// succeeded.
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

	// The wake landed the outcome before the requeue: the record ID derives
	// from the child submission ID, the content is the child's final
	// assistant text (no structured result was requested).
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome — the wake did not land")
	}
	if want := "wake-" + childSub.ID; recs[outcomeIdx].ID != want {
		t.Errorf("outcome record ID = %q, want %q (deterministic from the child submission)", recs[outcomeIdx].ID, want)
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
		if rec.Kind != harness.KindToolOutcome {
			continue
		}
		outcomes++
		if want := "wake-" + children[0].ID; rec.ID != want {
			t.Errorf("outcome record ID = %q, want %q", rec.ID, want)
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
