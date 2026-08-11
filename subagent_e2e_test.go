package harness_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// waitForCondition polls until cond holds, failing the test after 10s.
func waitForCondition(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for !cond() {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// callRecordIndex returns the index of the first record of the given kind
// naming callID in its payload, or -1.
func callRecordIndex(t *testing.T, recs []harness.Record, kind harness.RecordKind, callID string) int {
	t.Helper()
	for i, rec := range recs {
		if rec.Kind != kind {
			continue
		}
		var id string
		switch kind {
		case harness.KindAssistantToolCall:
			var p harness.AssistantToolCallPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(assistant_tool_call): %v", err)
			}
			id = p.CallID
		case harness.KindTaskSpawned:
			var p harness.TaskSpawnedPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(task_spawned): %v", err)
			}
			id = p.CallID
		case harness.KindToolOutcome:
			var p harness.ToolOutcomePayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(tool_outcome): %v", err)
			}
			id = p.CallID
		default:
			t.Fatalf("callRecordIndex: unsupported kind %s", kind)
		}
		if id == callID {
			return i
		}
	}
	return -1
}

// The full subagent lifecycle, end to end (HARNESS-15): a "triage" parent
// spawns two children (researcher + billing) in ONE turn via two task
// calls; the parent parks in waiting with no worker held; both children
// run and settle independently; the parent wakes once — the first settle
// does not requeue while a sibling is in flight — and the resume drive's
// provider request carries both tool results in original call order; the
// parent's final answer uses both; records and observer events show the
// full causal chain; child conversations link back via ParentRef.
//
// Determinism: both children are gated at their first (only) model call.
// The test parks the parent, asserts the parked state, then releases the
// researcher first and the billing child second, so the two wakes land in
// call order. The two task_spawned records may land in either relative
// order (same-turn tool calls execute in parallel, and the task tool's
// admission mutex picks no favorite), so spawn order is asserted pairwise
// — each spawn after BOTH calls (the consumer authors calls from stream
// events before any tool-end event arrives) and before BOTH outcomes (the
// gates hold every child until the parked parent's reconciliation has
// authored both spawn records).
func TestSubagentEndToEnd(t *testing.T) {
	t.Parallel()
	store := memory.New()
	researchGate := make(chan struct{})
	billingGate := make(chan struct{})

	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: taskArgs("researcher", "research the acme contract")},
			llm.ToolCallEndEvent{CallID: "call-2", ToolName: "task", Args: taskArgs("billing", "audit the acme invoice")},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn(`triage summary: research says "research answer"; billing says "billing answer"`),
	}}
	researcher := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("research answer"),
	}, gates: []<-chan struct{}{researchGate}}
	billing := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("billing answer"),
	}, gates: []<-chan struct{}{billingGate}}

	obs := &recordingObserver{}
	define := func(desc string, p llm.LLMProvider) harness.AgentDefinition {
		return harness.AgentDefinition{
			Description: desc,
			Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
				return harness.AgentRuntimeConfig{
					Model:         "mock/test-model",
					ContextWindow: 200_000,
					Providers:     []llm.LLMProvider{p},
				}, nil
			},
		}
	}
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"triage":     define("routes work to specialists", parent),
			"researcher": define("researches contracts", researcher),
			"billing":    define("audits invoices", billing),
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
		Subagents:     harness.SubagentPolicy{"triage": {"researcher", "billing"}},
		Observers:     []harness.Observer{obs.observe},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage the acme escalation"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	// (1) queued → running → waiting: the durable row parks (the event
	// chain below pins admitted → claimed → waiting).
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)

	// (2) While parked the parent holds no worker: its claim is released
	// and it is not runnable, yet both children stream independently —
	// each child provider saw its first (gated) request.
	if parentSub.OwnerID != "" || parentSub.AttemptID != "" {
		t.Errorf("parked parent still holds a claim: owner %q attempt %q", parentSub.OwnerID, parentSub.AttemptID)
	}
	runnable, err := store.ListRunnable(ctx)
	if err != nil {
		t.Fatalf("ListRunnable: %v", err)
	}
	for _, sub := range runnable {
		if sub.ID == parentSub.ID {
			t.Error("waiting parent is runnable — a waiting submission must free its worker")
		}
	}
	waitForCondition(t, "both children streaming while the parent is parked", func() bool {
		return len(researcher.requests()) == 1 && len(billing.requests()) == 1
	})

	// (3) Two child submissions, each linked back to the call that spawned
	// it, at depth 1, with the derived instance and dispatch IDs.
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 2 {
		t.Fatalf("children = %d, want 2", len(children))
	}
	childByCall := map[string]harness.Submission{}
	for _, ch := range children {
		childByCall[ch.ParentCallID] = ch
	}
	for callID, agent := range map[string]string{"call-1": "researcher", "call-2": "billing"} {
		ch, ok := childByCall[callID]
		if !ok {
			t.Fatalf("no child for %s", callID)
		}
		if ch.ParentSubmissionID != parentSub.ID {
			t.Errorf("%s child ParentSubmissionID = %q, want %q", callID, ch.ParentSubmissionID, parentSub.ID)
		}
		if ch.Depth != 1 {
			t.Errorf("%s child depth = %d, want 1", callID, ch.Depth)
		}
		if ch.SessionKey.Agent != agent {
			t.Errorf("%s child agent = %q, want %q", callID, ch.SessionKey.Agent, agent)
		}
		if want := "acme:" + safeCallID(callID); string(ch.SessionKey.Instance) != want {
			t.Errorf("%s child instance = %q, want %q", callID, ch.SessionKey.Instance, want)
		}
		if want := parentSub.ID + ":" + safeCallID(callID); ch.ID != want {
			t.Errorf("%s child dispatch id = %q, want %q", callID, ch.ID, want)
		}
	}

	// The parked parent's log: both calls, both spawns, no outcomes.
	parked, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parked parent): %v", err)
	}
	call1Idx := callRecordIndex(t, parked, harness.KindAssistantToolCall, "call-1")
	call2Idx := callRecordIndex(t, parked, harness.KindAssistantToolCall, "call-2")
	spawn1Idx := callRecordIndex(t, parked, harness.KindTaskSpawned, "call-1")
	spawn2Idx := callRecordIndex(t, parked, harness.KindTaskSpawned, "call-2")
	if call1Idx < 0 || call2Idx < 0 || spawn1Idx < 0 || spawn2Idx < 0 {
		t.Fatalf("parked parent log: call-1 at %d, call-2 at %d, spawn-1 at %d, spawn-2 at %d — all four must exist",
			call1Idx, call2Idx, spawn1Idx, spawn2Idx)
	}
	if call1Idx > call2Idx {
		t.Errorf("call-1 record at %d lands after call-2 at %d — stream order lost", call1Idx, call2Idx)
	}
	// Both calls precede both spawns (tool execution starts only after the
	// turn's message end); the two spawns' relative order is
	// nondeterministic under parallel tool execution.
	if spawn1Idx < call2Idx || spawn2Idx < call2Idx {
		t.Errorf("a spawn record precedes the second call record: calls at %d,%d spawns at %d,%d",
			call1Idx, call2Idx, spawn1Idx, spawn2Idx)
	}
	assertCallSpawnAdjacency(t, parked)
	if n := countKind(parked, harness.KindToolOutcome); n != 0 {
		t.Fatalf("tool_outcome records = %d, want 0 while both children are unsettled", n)
	}

	// (4, cont.) Release the researcher first: its wake lands outcome-1 but
	// does NOT requeue while billing is in flight. Then billing: the last
	// wake lands outcome-2 and requeues the parent.
	close(researchGate)
	researchSettled, err := rt.Wait(ctx, childByCall["call-1"].ID)
	if err != nil {
		t.Fatalf("Wait(researcher): %v", err)
	}
	if researchSettled.Status != harness.SettledSucceeded {
		t.Fatalf("researcher settled = %+v, want succeeded", researchSettled)
	}
	if got, err := store.GetSubmission(ctx, parentSub.ID); err != nil {
		t.Fatalf("GetSubmission(parent): %v", err)
	} else if got.Status != harness.StatusWaiting {
		t.Fatalf("parent status after researcher settle = %s, want waiting (billing still in flight)", got.Status)
	}

	close(billingGate)
	billingSettled, err := rt.Wait(ctx, childByCall["call-2"].ID)
	if err != nil {
		t.Fatalf("Wait(billing): %v", err)
	}
	if billingSettled.Status != harness.SettledSucceeded {
		t.Fatalf("billing settled = %+v, want succeeded", billingSettled)
	}

	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}

	// (4) The full causal chain on the parent conversation:
	// user_message → call-1 → call-2 → {spawn-1, spawn-2} → outcome-1 →
	// outcome-2 → assistant final text → settled.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	call1Idx = callRecordIndex(t, recs, harness.KindAssistantToolCall, "call-1")
	call2Idx = callRecordIndex(t, recs, harness.KindAssistantToolCall, "call-2")
	spawn1Idx = callRecordIndex(t, recs, harness.KindTaskSpawned, "call-1")
	spawn2Idx = callRecordIndex(t, recs, harness.KindTaskSpawned, "call-2")
	outcome1Idx := callRecordIndex(t, recs, harness.KindToolOutcome, "call-1")
	outcome2Idx := callRecordIndex(t, recs, harness.KindToolOutcome, "call-2")
	userIdx := findRecord(recs, harness.KindUserMessage)
	if outcome1Idx < 0 || outcome2Idx < 0 || userIdx < 0 {
		t.Fatalf("parent log: user at %d, outcome-1 at %d, outcome-2 at %d — all must exist", userIdx, outcome1Idx, outcome2Idx)
	}
	if n := countKind(recs, harness.KindToolOutcome); n != 2 {
		t.Errorf("tool_outcome records = %d, want exactly 2 (one wake outcome per child)", n)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 2 {
		t.Errorf("task_spawned records = %d, want exactly 2", n)
	}
	assertCallSpawnAdjacency(t, recs)
	switch {
	case userIdx > call1Idx:
		t.Errorf("user_message at %d lands after call-1 at %d", userIdx, call1Idx)
	case call1Idx > call2Idx:
		t.Errorf("call-1 at %d lands after call-2 at %d", call1Idx, call2Idx)
	case spawn1Idx < call2Idx || spawn2Idx < call2Idx:
		t.Errorf("a spawn precedes the second call: calls at %d,%d spawns at %d,%d", call1Idx, call2Idx, spawn1Idx, spawn2Idx)
	case outcome1Idx < spawn1Idx || outcome1Idx < spawn2Idx || outcome2Idx < spawn1Idx || outcome2Idx < spawn2Idx:
		t.Errorf("an outcome precedes a spawn: spawns at %d,%d outcomes at %d,%d", spawn1Idx, spawn2Idx, outcome1Idx, outcome2Idx)
	case outcome1Idx > outcome2Idx:
		t.Errorf("outcome-1 at %d lands after outcome-2 at %d — the gated wakes must land in call order", outcome1Idx, outcome2Idx)
	}
	var outcome1, outcome2 harness.ToolOutcomePayload
	if err := recs[outcome1Idx].DecodePayload(&outcome1); err != nil {
		t.Fatalf("DecodePayload(outcome-1): %v", err)
	}
	if err := recs[outcome2Idx].DecodePayload(&outcome2); err != nil {
		t.Fatalf("DecodePayload(outcome-2): %v", err)
	}
	if outcome1.IsError || outcome1.Content != "research answer" {
		t.Errorf("outcome-1 = %+v, want the researcher's final text", outcome1)
	}
	if outcome2.IsError || outcome2.Content != "billing answer" {
		t.Errorf("outcome-2 = %+v, want the billing child's final text", outcome2)
	}

	// (6) The parent's final answer incorporates both child answers and
	// lands after both outcomes, followed by the settled record.
	finalIdx := -1
	var finalBody string
	for i, rec := range recs {
		if rec.Kind != harness.KindAssistantMessageCompleted {
			continue
		}
		var p harness.AssistantMessageCompletedPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload(assistant_message_completed): %v", err)
		}
		var body string
		if err := json.Unmarshal(p.Message.Body, &body); err == nil && strings.Contains(body, "triage summary") {
			finalIdx, finalBody = i, body
		}
	}
	if finalIdx < 0 {
		t.Fatal("no final assistant message carrying the triage summary")
	}
	if !strings.Contains(finalBody, "research answer") || !strings.Contains(finalBody, "billing answer") {
		t.Errorf("final answer = %q, want it to incorporate both child answers", finalBody)
	}
	if finalIdx < outcome2Idx {
		t.Errorf("final assistant text at %d lands before outcome-2 at %d", finalIdx, outcome2Idx)
	}
	settledIdx := findRecord(recs, harness.KindSubmissionSettled)
	if settledIdx < 0 || settledIdx < finalIdx {
		t.Errorf("settled record at %d, want it after the final assistant text at %d", settledIdx, finalIdx)
	}

	// (5) The resume drive's provider request — the parent's second call —
	// carries both tool results in original call order, with no re-appended
	// input.
	reqs := parent.requests()
	if len(reqs) != 2 {
		t.Fatalf("parent provider calls = %d, want 2 (prompt + resume)", len(reqs))
	}
	resume := reqs[1]
	var results []llm.ToolResultContent
	for _, m := range resume.Messages {
		if tr, ok := m.Content.(llm.ToolResultContent); ok {
			results = append(results, tr)
		}
	}
	if len(results) != 2 {
		t.Fatalf("resume request tool results = %d, want 2", len(results))
	}
	if results[0].CallID != "call-1" || results[0].Content != "research answer" {
		t.Errorf("resume result[0] = %+v, want call-1 with the research answer", results[0])
	}
	if results[1].CallID != "call-2" || results[1].Content != "billing answer" {
		t.Errorf("resume result[1] = %+v, want call-2 with the billing answer", results[1])
	}
	if n := len(userTexts(resume)); n != 1 {
		t.Errorf("resume request user messages = %d, want 1 (no re-appended input)", n)
	}

	// (7) Each child conversation's ParentRef points at the parent
	// conversation and the spawn record for its call.
	spawnRecordID := map[string]string{}
	for _, rec := range recs {
		if rec.Kind != harness.KindTaskSpawned {
			continue
		}
		var p harness.TaskSpawnedPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("DecodePayload(task_spawned): %v", err)
		}
		spawnRecordID[p.CallID] = rec.ID
	}
	for callID, ch := range childByCall {
		childRecs, err := rt.Records(ctx, ch.ConversationID, "")
		if err != nil {
			t.Fatalf("Records(child %s): %v", callID, err)
		}
		createdIdx := findRecord(childRecs, harness.KindConversationCreated)
		if createdIdx < 0 {
			t.Fatalf("child %s conversation has no conversation_created record", callID)
		}
		var created harness.ConversationCreatedPayload
		if err := childRecs[createdIdx].DecodePayload(&created); err != nil {
			t.Fatalf("DecodePayload(conversation_created): %v", err)
		}
		if created.ParentRef == nil {
			t.Fatalf("child %s conversation_created has no ParentRef", callID)
		}
		if created.ParentRef.ConversationID != res.ConversationID {
			t.Errorf("child %s ParentRef.ConversationID = %q, want %q", callID, created.ParentRef.ConversationID, res.ConversationID)
		}
		if created.ParentRef.SpawnRecordID != spawnRecordID[callID] {
			t.Errorf("child %s ParentRef.SpawnRecordID = %q, want spawn record id %q", callID, created.ParentRef.SpawnRecordID, spawnRecordID[callID])
		}
	}

	// (8) The observer stream saw the parent's lifecycle in causal order:
	// admitted → claimed → spawned×2 → waiting → resumed → settled
	// (succeeded).
	var parentEvents []harness.HarnessEvent
	spawned := map[string]string{} // callID → child submission id
	for _, ev := range obs.snapshot() {
		var corr harness.Correlation
		switch e := ev.(type) {
		case harness.SubmissionSpawnedEvent:
			corr = e.Correlation
			spawned[e.CallID] = e.ChildSubmissionID
		case harness.SubmissionAdmittedEvent:
			corr = e.Correlation
		case harness.SubmissionClaimedEvent:
			corr = e.Correlation
		case harness.SubmissionWaitingEvent:
			corr = e.Correlation
		case harness.SubmissionResumedEvent:
			corr = e.Correlation
		case harness.SubmissionSettledEvent:
			corr = e.Correlation
		default:
			continue
		}
		if corr.SubmissionID == parentSub.ID {
			parentEvents = append(parentEvents, ev)
		}
	}
	assertOrderedSubsequence(t, observedKinds(parentEvents),
		[]string{"admitted", "claimed", "spawned", "spawned", "waiting", "resumed", "settled"})
	if len(spawned) != 2 {
		t.Fatalf("spawned events for the parent = %d, want 2", len(spawned))
	}
	for callID, ch := range childByCall {
		if spawned[callID] != ch.ID {
			t.Errorf("spawned event for %s names child %q, want %q", callID, spawned[callID], ch.ID)
		}
	}
	last, ok := parentEvents[len(parentEvents)-1].(harness.SubmissionSettledEvent)
	if !ok {
		t.Fatalf("last parent event = %T, want SubmissionSettledEvent", parentEvents[len(parentEvents)-1])
	}
	if last.Payload.Status != harness.SettledSucceeded {
		t.Errorf("parent settled event = %+v, want succeeded", last.Payload)
	}
}
