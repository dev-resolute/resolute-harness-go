package harness_test

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// scriptProvider plays a fixed script of event batches — one per Stream
// call, in order — and records each request. Unlike mock.MockProvider it
// scripts call ids explicitly, which the subagent tests need for parallel
// and replayed task calls in a single turn.
type scriptProvider struct {
	name   string
	script [][]llm.LLMEvent
	// gates optionally holds the Stream for step i until gates[i] closes
	// (nil = ungated) — the test holds a run mid-flight while asserting
	// parked state, keeping wake/park ordering deterministic (HARNESS-15).
	gates []<-chan struct{}

	mu   sync.Mutex
	reqs []llm.LLMRequest
}

func (p *scriptProvider) Name() string { return p.name }

func (p *scriptProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true, ToolCalling: true, ParallelToolCalls: true}
}

func (p *scriptProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	p.mu.Lock()
	step := len(p.reqs)
	p.reqs = append(p.reqs, req)
	p.mu.Unlock()

	events := make(chan llm.LLMEvent, 16)
	done := make(chan llm.StreamResult, 1)
	go func() {
		defer close(events)
		if step < len(p.gates) && p.gates[step] != nil {
			select {
			case <-p.gates[step]:
			case <-ctx.Done():
				done <- llm.StreamResult{Err: context.Cause(ctx)}
				return
			}
		}
		if step >= len(p.script) {
			done <- llm.StreamResult{Err: fmt.Errorf("scriptProvider %q: unexpected call #%d: %w", p.name, step+1, llm.ErrProviderFatal)}
			return
		}
		for _, ev := range p.script[step] {
			select {
			case events <- ev:
			case <-ctx.Done():
				done <- llm.StreamResult{Err: context.Cause(ctx)}
				return
			}
		}
		done <- llm.StreamResult{}
	}()
	return llm.NewEventStream(events, done)
}

func (p *scriptProvider) requests() []llm.LLMRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.reqs)
}

// taskArgs builds the task tool's argument JSON.
func taskArgs(agent, prompt string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"agent":%q,"prompt":%q}`, agent, prompt))
}

// safeCallID mirrors the harness's sanitizeCallID so tests can pin the
// derived identifiers exactly: sanitize to [A-Za-z0-9_-], truncate to 48
// bytes, mix in 8 hex chars of fnv32(raw).
func safeCallID(raw string) string {
	b := []byte(raw)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-' {
			continue
		}
		b[i] = '-'
	}
	if len(b) > 48 {
		b = b[:48]
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(raw))
	return fmt.Sprintf("%s-%08x", string(b), h.Sum32())
}

// taskCallTurn scripts a turn ending in a single task tool call.
func taskCallTurn(callID, agent, prompt string) []llm.LLMEvent {
	return []llm.LLMEvent{
		llm.ToolCallEndEvent{CallID: callID, ToolName: "task", Args: taskArgs(agent, prompt)},
		llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
	}
}

// textTurn scripts a plain text answer turn.
func textTurn(text string) []llm.LLMEvent {
	return []llm.LLMEvent{
		llm.TextDeltaEvent{Delta: text},
		llm.MessageEndEvent{StopReason: llm.StopReasonStop},
	}
}

// subagentConfig builds a Runtime config with a "triage" parent definition
// (policy: may spawn "reviewer") and a "reviewer" child definition, each
// backed by its own scripted provider.
func subagentConfig(store harness.Store, parentProv, childProv llm.LLMProvider, limits harness.SubagentLimits) harness.Config {
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
	return harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"triage":   define("routes work to specialists", parentProv),
			"reviewer": define("reviews code for defects", childProv),
		},
		Store:          store,
		ClaimInterval:  20 * time.Millisecond,
		LeaseDuration:  300 * time.Millisecond,
		Subagents:      harness.SubagentPolicy{"triage": {"reviewer"}},
		SubagentLimits: limits,
	}
}

// waitForStatus polls until the submission reaches the given status.
func waitForStatus(t *testing.T, store harness.Store, submissionID string, status harness.SubmissionStatus) harness.Submission {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		sub, err := store.GetSubmission(context.Background(), submissionID)
		if err == nil && sub.Status == status {
			return sub
		}
		select {
		case <-deadline:
			t.Fatalf("submission %s never reached %s", submissionID, status)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// findRecord returns the index of the first record of the given kind, or -1.
func findRecord(recs []harness.Record, kind harness.RecordKind) int {
	for i, rec := range recs {
		if rec.Kind == kind {
			return i
		}
	}
	return -1
}

// assertCallSpawnAdjacency pins the call→spawn ordering: every
// task_spawned lands AFTER an assistant_tool_call carrying its CallID and
// before any tool_outcome or second task_spawned for the same call.
// Strict adjacency is NOT required — agent-core emits MessageEndEvent
// before tool execution, so a text+task turn legitimately lands an
// assistant_message_completed between the call and the spawn.
func assertCallSpawnAdjacency(t *testing.T, recs []harness.Record) {
	t.Helper()
	firstCall := map[string]int{}
	spawned := map[string]int{}
	outcome := map[string]int{}
	for i, rec := range recs {
		switch rec.Kind {
		case harness.KindAssistantToolCall:
			var p harness.AssistantToolCallPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(assistant_tool_call): %v", err)
			}
			if _, ok := firstCall[p.CallID]; !ok {
				firstCall[p.CallID] = i
			}
		case harness.KindToolOutcome:
			var p harness.ToolOutcomePayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(tool_outcome): %v", err)
			}
			if _, ok := outcome[p.CallID]; !ok {
				outcome[p.CallID] = i
			}
		case harness.KindTaskSpawned:
			var p harness.TaskSpawnedPayload
			if err := rec.DecodePayload(&p); err != nil {
				t.Fatalf("DecodePayload(task_spawned): %v", err)
			}
			if _, ok := firstCall[p.CallID]; !ok {
				t.Fatalf("task_spawned at %d has no preceding assistant_tool_call for %q", i, p.CallID)
			}
			if prev, dup := spawned[p.CallID]; dup {
				t.Fatalf("second task_spawned for %q at %d (first at %d)", p.CallID, i, prev)
			}
			spawned[p.CallID] = i
			if oi, ok := outcome[p.CallID]; ok {
				t.Fatalf("task_spawned for %q at %d lands after its tool_outcome at %d", p.CallID, i, oi)
			}
		}
	}
}

// A task call on an allowed target admits exactly one durable child, lands
// the task_spawned record immediately after the call, and parks the parent
// in waiting — no tool outcome, no settlement.
func TestTaskToolAdmitsChildAndSuspends(t *testing.T) {
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

	// The parent suspends: waiting, never settled. The child is gated
	// mid-run, so it cannot settle (and wake the parent) before the parked
	// state is asserted.
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)

	// The injected tool reached the model with its routing metadata.
	reqs := parent.requests()
	if len(reqs) == 0 {
		t.Fatal("parent provider saw no requests")
	}
	taskDefIdx := -1
	for i, td := range reqs[0].Tools {
		if td.Name == "task" {
			taskDefIdx = i
		}
	}
	if taskDefIdx < 0 {
		t.Fatal("parent request missing the injected task tool")
	}
	if desc := reqs[0].Tools[taskDefIdx].Description; !strings.Contains(desc, "reviewer — reviews code for defects") {
		t.Errorf("task tool description = %q, want it to enumerate targets with descriptions", desc)
	}

	// Exactly one child, linked back to the call that spawned it.
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want 1", len(children))
	}
	childSub := children[0]
	if childSub.ParentSubmissionID != parentSub.ID || childSub.ParentCallID != "call-1" {
		t.Errorf("child parent link = (%q, %q), want (%q, call-1)", childSub.ParentSubmissionID, childSub.ParentCallID, parentSub.ID)
	}
	if childSub.Depth != 1 {
		t.Errorf("child depth = %d, want 1", childSub.Depth)
	}
	if want := parentSub.ID + ":" + safeCallID("call-1"); childSub.ID != want {
		t.Errorf("child dispatch id = %q, want %q", childSub.ID, want)
	}

	// Parent log: assistant_tool_call immediately followed by task_spawned;
	// no tool_outcome — the pending call is the suspension point. (The
	// reconciler cannot synthesize one here: the waiting parent heads its
	// session queue, so no drive scans this conversation while the child is
	// unsettled. Task 10 pins the exempt-by-design scan.)
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	callIdx := findRecord(recs, harness.KindAssistantToolCall)
	spawnIdx := findRecord(recs, harness.KindTaskSpawned)
	if callIdx < 0 || spawnIdx != callIdx+1 {
		t.Fatalf("record order: call at %d, spawn at %d, want spawn immediately after call", callIdx, spawnIdx)
	}
	assertCallSpawnAdjacency(t, recs)
	if n := countKind(recs, harness.KindToolOutcome); n != 0 {
		t.Fatalf("tool_outcome records = %d, want 0 while the child is unsettled", n)
	}

	var spawn harness.TaskSpawnedPayload
	if err := recs[spawnIdx].DecodePayload(&spawn); err != nil {
		t.Fatalf("DecodePayload(task_spawned): %v", err)
	}
	wantSpawn := harness.TaskSpawnedPayload{
		CallID:              "call-1",
		Agent:               "reviewer",
		ChildInstance:       "acme:" + safeCallID("call-1"),
		ChildConversationID: childSub.ConversationID,
		ChildSubmissionID:   childSub.ID,
		Prompt:              "check it",
	}
	if spawn != wantSpawn {
		t.Errorf("task_spawned payload = %+v, want %+v", spawn, wantSpawn)
	}

	// The child conversation's ParentRef names the spawn record.
	childRecs, err := rt.Records(ctx, childSub.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(child): %v", err)
	}
	createdIdx := findRecord(childRecs, harness.KindConversationCreated)
	if createdIdx < 0 {
		t.Fatal("child conversation has no conversation_created record")
	}
	var created harness.ConversationCreatedPayload
	if err := childRecs[createdIdx].DecodePayload(&created); err != nil {
		t.Fatalf("DecodePayload(conversation_created): %v", err)
	}
	if created.ParentRef == nil {
		t.Fatal("child conversation_created has no ParentRef")
	}
	if created.ParentRef.ConversationID != res.ConversationID {
		t.Errorf("ParentRef.ConversationID = %q, want %q", created.ParentRef.ConversationID, res.ConversationID)
	}
	if created.ParentRef.SpawnRecordID != recs[spawnIdx].ID {
		t.Errorf("ParentRef.SpawnRecordID = %q, want spawn record id %q", created.ParentRef.SpawnRecordID, recs[spawnIdx].ID)
	}

	// Release the child: it settles, the wake lands the parent's outcome and
	// requeues the parent, whose resume drive settles it (wake_test.go pins
	// the wake itself).
	close(childGate)
	settled, err := rt.Wait(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("child settled = %+v, want succeeded", settled)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A re-executed callID (the replay shape of a re-driven parent) admits no
// second child and lands no second spawn record: the DispatchID replay and
// the spawn-record dedupe make the spawn idempotent.
func TestTaskToolAdmissionIdempotent(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	args := taskArgs("reviewer", "check it")
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			// Two identical task calls in one turn: same callID, same
			// arguments — the in-turn shape of an admission replay.
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
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
		t.Fatalf("children = %d, want exactly 1 after a replayed callID", len(children))
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 1 {
		t.Fatalf("task_spawned records = %d, want exactly 1", n)
	}
	assertCallSpawnAdjacency(t, recs)
	if n := countKind(recs, harness.KindToolOutcome); n != 0 {
		t.Fatalf("tool_outcome records = %d, want 0 (both calls suspended)", n)
	}

	// Release the child: the wake requeues the parent, whose resume drive
	// settles it.
	close(childGate)
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A replayed callID whose child is still live must NOT count against the
// fan-out bound: two identical in-turn calls under MaxChildrenPerRun=1
// admit one child, land one spawn record, and produce no error outcome.
func TestTaskToolReplaySkipsFanOutBound(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	args := taskArgs("reviewer", "check it")
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{MaxChildrenPerRun: 1}))
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
		t.Fatalf("children = %d, want exactly 1 after a replayed callID under MaxChildrenPerRun=1", len(children))
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 1 {
		t.Fatalf("task_spawned records = %d, want exactly 1", n)
	}
	assertCallSpawnAdjacency(t, recs)
	if n := countKind(recs, harness.KindToolOutcome); n != 0 {
		t.Fatalf("tool_outcome records = %d, want 0 — the replay must not error against the bound", n)
	}

	// Release the child: the wake requeues the parent, whose resume drive
	// settles it.
	close(childGate)
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A provider-controlled callID is sanitized before it flows into the child
// Instance (a URL path segment), Session, and DispatchID; the raw callID
// still names the call in the spawn record and the parent link.
func TestTaskToolSanitizesCallID(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call/1?x y", "reviewer", "check it"),
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
	if want := "acme:" + safeCallID("call/1?x y"); string(childSub.SessionKey.Instance) != want {
		t.Errorf("child instance = %q, want %q (sanitized callID)", childSub.SessionKey.Instance, want)
	}
	if want := "task-" + safeCallID("call/1?x y"); childSub.SessionKey.Session != want {
		t.Errorf("child session = %q, want %q (sanitized callID)", childSub.SessionKey.Session, want)
	}
	if want := parentSub.ID + ":" + safeCallID("call/1?x y"); childSub.ID != want {
		t.Errorf("child dispatch id = %q, want %q (sanitized callID)", childSub.ID, want)
	}
	if childSub.ParentCallID != "call/1?x y" {
		t.Errorf("child parent link CallID = %q, want the raw callID", childSub.ParentCallID)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	spawnIdx := findRecord(recs, harness.KindTaskSpawned)
	if spawnIdx < 0 {
		t.Fatal("no task_spawned record")
	}
	assertCallSpawnAdjacency(t, recs)
	var spawn harness.TaskSpawnedPayload
	if err := recs[spawnIdx].DecodePayload(&spawn); err != nil {
		t.Fatalf("DecodePayload(task_spawned): %v", err)
	}
	if spawn.CallID != "call/1?x y" {
		t.Errorf("spawn CallID = %q, want the raw callID (wake correlates on it)", spawn.CallID)
	}
	if want := "acme:" + safeCallID("call/1?x y"); spawn.ChildInstance != want {
		t.Errorf("spawn ChildInstance = %q, want %q (sanitized callID)", spawn.ChildInstance, want)
	}

	// Release the child: the wake correlates on the RAW callID (the child
	// row's ParentCallID), lands the outcome, and requeues the parent.
	close(childGate)
	if _, err := rt.Wait(ctx, childSub.ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A Suspend result WITHOUT spawn data (a non-task tool) is not a
// suspension: the outcome is authored as if Suspend were unset and the
// submission settles normally — it must never park in waiting.
func TestBareSuspendWithoutSpawnDataSettlesNormally(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "park", Args: json.RawMessage(`{}`)},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
	}}
	park := pi.NewTool(pi.Tool[struct{}]{
		Name: "park", Description: "returns a bare suspend result",
		Execute: func(ctx context.Context, p struct{}) (pi.ToolResult, error) {
			return pi.ToolResult{Suspend: true, Content: "parked"}, nil
		},
	})
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"lonely": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						Tools:         []pi.RegisteredTool{park},
					}, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "lonely", Instance: "acme", Message: harness.UserMessage("work alone"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded — a bare Suspend must not park the submission", settled)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 0 {
		t.Fatalf("task_spawned records = %d, want 0 for a non-task tool", n)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome — a bare Suspend must be authored as if Suspend were unset")
	}
	var outcome harness.ToolOutcomePayload
	if err := recs[outcomeIdx].DecodePayload(&outcome); err != nil {
		t.Fatalf("DecodePayload(tool_outcome): %v", err)
	}
	if outcome.CallID != "call-1" || outcome.Content != "parked" {
		t.Errorf("outcome = %+v, want the bare-suspend result authored normally", outcome)
	}
}

// A call naming an agent outside the policy is a plain error result naming
// the allowed targets — no child, no spawn record, no suspension.
func TestTaskToolRejectsStrangers(t *testing.T) {
	t.Parallel()
	store := memory.New()
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "scout", "spy on it"),
		textTurn("cannot delegate"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("parent settled = %+v, want succeeded (error result, not a suspension)", settled)
	}

	children, err := store.ListChildSubmissions(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 0 {
		t.Fatalf("children = %d, want 0 for a policy violation", len(children))
	}
	if n := child.requests(); len(n) != 0 {
		t.Fatalf("child provider called %d times, want 0", len(n))
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 0 {
		t.Fatalf("task_spawned records = %d, want 0", n)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome for the rejected call")
	}
	var outcome harness.ToolOutcomePayload
	if err := recs[outcomeIdx].DecodePayload(&outcome); err != nil {
		t.Fatalf("DecodePayload(tool_outcome): %v", err)
	}
	if !outcome.IsError {
		t.Error("rejected call outcome IsError = false, want true")
	}
	if !strings.Contains(outcome.Content, "scout") || !strings.Contains(outcome.Content, "reviewer") {
		t.Errorf("rejection content = %q, want it to name the rejected agent and the allowed targets", outcome.Content)
	}
}

// With MaxChildrenPerRun=1, a second parallel spawn overflows: it is a
// plain error result while the first admits and suspends.
func TestTaskToolRejectsOverflow(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "first")},
			llm.ToolCallEndEvent{CallID: "call-2", ToolName: "task", Args: taskArgs("reviewer", "second")},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}
	rt := startRuntime(t, subagentConfig(store, parent, child, harness.SubagentLimits{MaxChildrenPerRun: 1}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)

	// Exactly one of the two parallel calls admitted a child; the other
	// overflowed. (Which one wins the tool's mutex is nondeterministic.)
	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want exactly 1 under MaxChildrenPerRun=1", len(children))
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 1 {
		t.Fatalf("task_spawned records = %d, want 1", n)
	}
	outcomeIdx := findRecord(recs, harness.KindToolOutcome)
	if outcomeIdx < 0 {
		t.Fatal("no tool_outcome for the overflowed call")
	}
	var outcome harness.ToolOutcomePayload
	if err := recs[outcomeIdx].DecodePayload(&outcome); err != nil {
		t.Fatalf("DecodePayload(tool_outcome): %v", err)
	}
	if !outcome.IsError {
		t.Error("overflow outcome IsError = false, want true")
	}
	if !strings.Contains(outcome.Content, "child limit reached") {
		t.Errorf("overflow content = %q, want it to explain the limit", outcome.Content)
	}
	if outcome.CallID == children[0].ParentCallID {
		t.Errorf("overflow outcome CallID = %q, the call that admitted the child; want the other call", outcome.CallID)
	}

	// Release the child: the wake lands the admitted call's outcome (the
	// overflow outcome already landed in-turn) and requeues the parent.
	close(childGate)
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A definition with no policy entry never sees a task tool.
func TestNoPolicyNoTaskTool(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("done alone"),
	}}
	echo := pi.NewTool(pi.Tool[struct {
		Value string `json:"value"`
	}]{
		Name: "echo", Description: "echo the value",
		Execute: func(ctx context.Context, p struct {
			Value string `json:"value"`
		}) (pi.ToolResult, error) {
			return pi.ToolResult{Content: p.Value}, nil
		},
	})
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"lonely": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						Tools:         []pi.RegisteredTool{echo},
					}, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "lonely", Instance: "acme", Message: harness.UserMessage("work alone"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if _, err := rt.Wait(ctx, res.SubmissionID); err != nil {
		t.Fatalf("Wait: %v", err)
	}

	reqs := provider.requests()
	if len(reqs) == 0 {
		t.Fatal("provider saw no requests")
	}
	sawEcho := false
	for _, td := range reqs[0].Tools {
		if td.Name == "task" {
			t.Fatal("task tool injected for an agent with no policy entry")
		}
		if td.Name == "echo" {
			sawEcho = true
		}
	}
	if !sawEcho {
		t.Error("request is missing the configured echo tool — tool wiring broken, assertion is vacuous")
	}
}

// Two raw callIDs that sanitize to the same string ("call/1", "call?1")
// still derive distinct dispatch IDs: the fnv mix-in keeps them from
// aliasing into a replay, so both admit a child.
func TestTaskToolSanitizesCallIDCollisions(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call/1", ToolName: "task", Args: taskArgs("reviewer", "first")},
			llm.ToolCallEndEvent{CallID: "call?1", ToolName: "task", Args: taskArgs("reviewer", "second")},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn("both back"),
	}}
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
		t.Fatalf("children = %d, want 2 — colliding sanitized callIDs must not alias into a replay", len(children))
	}
	ids := map[string]bool{}
	for _, ch := range children {
		ids[ch.ID] = true
	}
	for _, raw := range []string{"call/1", "call?1"} {
		if want := parentSub.ID + ":" + safeCallID(raw); !ids[want] {
			t.Errorf("no child with dispatch id %q (raw callID %q)", want, raw)
		}
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 2 {
		t.Fatalf("task_spawned records = %d, want 2", n)
	}
	assertCallSpawnAdjacency(t, recs)

	// Release the gated child: the first settle does not requeue (a sibling
	// is still in flight); the second does, and the parent settles.
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
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}

// A ToolCallEndEvent lost to agent-core's lossy event channel (dropped
// after 100ms of backpressure) never reaches the consumer, so no
// task_spawned is authored even though Dispatch admitted the child.
// Park-time reconciliation repairs the record from durable child state —
// recovering the SpawnRecordID from the child conversation's ParentRef —
// and the parent still parks instead of settling SUCCESS with an orphan.
func TestTaskToolDroppedSpawnEventReconciledAtParkTime(t *testing.T) {
	t.Parallel()
	store := memory.New()
	childGate := make(chan struct{})
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "dropped-call", ToolName: "task", Args: json.RawMessage(`{}`)},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
		textTurn("triage done"),
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}, gates: []<-chan struct{}{childGate}}

	// The rig simulates the drop: the tool admits the child synchronously
	// (Dispatch writes the parent link and the child conversation's
	// ParentRef with the pre-minted spawn record ID), then returns a bare
	// Suspend — the spawn payload never reaches the consumer, as if the
	// ToolCallEndEvent had been dropped.
	type dropRig struct {
		parent        chan harness.DispatchResult
		spawnRecordID string
		rt            *harness.Runtime
		child         harness.DispatchResult
	}
	rig := &dropRig{
		parent:        make(chan harness.DispatchResult, 1),
		spawnRecordID: "01J0000DROPPEDSPAWNRECORD000",
	}
	drop := pi.NewTool(pi.Tool[struct{}]{
		Name: "task", Description: "simulates a dropped spawn event",
		Execute: func(ctx context.Context, _ struct{}) (pi.ToolResult, error) {
			parentRes := <-rig.parent
			childRes, err := rig.rt.Dispatch(ctx, harness.Dispatch{
				Agent:      "reviewer",
				Instance:   "acme:dropped",
				Session:    "task-dropped",
				DispatchID: parentRes.SubmissionID + ":dropped-call",
				Message:    harness.UserMessage("check it"),
				Parent: &harness.SpawnParent{
					SubmissionID:   parentRes.SubmissionID,
					CallID:         "dropped-call",
					ConversationID: parentRes.ConversationID,
					SpawnRecordID:  rig.spawnRecordID,
				},
			})
			if err != nil {
				return pi.ToolResult{}, err
			}
			rig.child = childRes
			// No Data: the spawn payload never arrived.
			return pi.ToolResult{Suspend: true}, nil
		},
	})

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
	parentDef := define("routes work to specialists", parent)
	parentDefTools := parentDef.Initialize
	parentDef.Initialize = func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
		cfg, err := parentDefTools(ctx, id, env)
		if err != nil {
			return cfg, err
		}
		cfg.Tools = []pi.RegisteredTool{drop}
		return cfg, nil
	}
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"triage":   parentDef,
			"reviewer": define("reviews code for defects", child),
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
		// No Subagents policy: the rigged tool stands in for the injected
		// task tool so the spawn payload can be "lost".
	})
	rig.rt = rt
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "triage", Instance: "acme", Message: harness.UserMessage("triage this"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	rig.parent <- res

	// The parent parks despite the lost spawn payload: children-existence,
	// not the event channel, gates the park.
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)

	// The reconciled spawn record recovers every field from durable child
	// state, including the SpawnRecordID from the child conversation's
	// conversation_created ParentRef.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	spawnIdx := findRecord(recs, harness.KindTaskSpawned)
	if spawnIdx < 0 {
		t.Fatal("park-time reconciliation did not author the missing task_spawned")
	}
	if recs[spawnIdx].ID != rig.spawnRecordID {
		t.Errorf("spawn record ID = %q, want %q recovered from the child conversation's ParentRef", recs[spawnIdx].ID, rig.spawnRecordID)
	}
	var spawn harness.TaskSpawnedPayload
	if err := recs[spawnIdx].DecodePayload(&spawn); err != nil {
		t.Fatalf("DecodePayload(task_spawned): %v", err)
	}
	wantSpawn := harness.TaskSpawnedPayload{
		CallID:              "dropped-call",
		Agent:               "reviewer",
		ChildInstance:       "acme:dropped",
		ChildConversationID: rig.child.ConversationID,
		ChildSubmissionID:   rig.child.SubmissionID,
		Prompt:              "check it",
	}
	if spawn != wantSpawn {
		t.Errorf("reconciled task_spawned payload = %+v, want %+v", spawn, wantSpawn)
	}

	// Release the child: it settles, the wake lands the parent's outcome
	// (correlated via the durable child row — the reconciled spawn record
	// is not the correlation source) and requeues the parent, whose resume
	// drive settles it.
	close(childGate)
	settled, err := rt.Wait(ctx, rig.child.SubmissionID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("child settled = %+v, want succeeded", settled)
	}
	parentSettled, err := rt.Wait(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("Wait(parent): %v", err)
	}
	if parentSettled.Status != harness.SettledSucceeded {
		t.Errorf("parent settled = %+v, want succeeded after the wake re-drive", parentSettled)
	}
}
