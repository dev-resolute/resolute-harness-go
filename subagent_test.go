package harness_test

import (
	"context"
	"encoding/json"
	"fmt"
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

// A task call on an allowed target admits exactly one durable child, lands
// the task_spawned record immediately after the call, and parks the parent
// in waiting — no tool outcome, no settlement.
func TestTaskToolAdmitsChildAndSuspends(t *testing.T) {
	t.Parallel()
	store := memory.New()
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		taskCallTurn("call-1", "reviewer", "check it"),
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

	// The parent suspends: waiting, never settled.
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
	if want := parentSub.ID + ":call-1"; childSub.ID != want {
		t.Errorf("child dispatch id = %q, want %q", childSub.ID, want)
	}

	// The child runs as an ordinary submission and settles.
	settled, err := rt.Wait(ctx, childSub.ID)
	if err != nil {
		t.Fatalf("Wait(child): %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("child settled = %+v, want succeeded", settled)
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
		ChildInstance:       "acme:call-1",
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

	// The wake is Task 9: with the child settled the parent stays parked.
	again, err := store.GetSubmission(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("GetSubmission(parent): %v", err)
	}
	if again.Status != harness.StatusWaiting {
		t.Errorf("parent status after child settle = %s, want waiting", again.Status)
	}
}

// A re-executed callID (the replay shape of a re-driven parent) admits no
// second child and lands no second spawn record: the DispatchID replay and
// the spawn-record dedupe make the spawn idempotent.
func TestTaskToolAdmissionIdempotent(t *testing.T) {
	t.Parallel()
	store := memory.New()
	args := taskArgs("reviewer", "check it")
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			// Two identical task calls in one turn: same callID, same
			// arguments — the in-turn shape of an admission replay.
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: args},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
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
	parentSub := waitForStatus(t, store, res.SubmissionID, harness.StatusWaiting)

	children, err := store.ListChildSubmissions(ctx, parentSub.ID)
	if err != nil {
		t.Fatalf("ListChildSubmissions: %v", err)
	}
	if len(children) != 1 {
		t.Fatalf("children = %d, want exactly 1 after a replayed callID", len(children))
	}
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
	}

	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records(parent): %v", err)
	}
	if n := countKind(recs, harness.KindTaskSpawned); n != 1 {
		t.Fatalf("task_spawned records = %d, want exactly 1", n)
	}
	if n := countKind(recs, harness.KindToolOutcome); n != 0 {
		t.Fatalf("tool_outcome records = %d, want 0 (both calls suspended)", n)
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
	parent := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		{
			llm.ToolCallEndEvent{CallID: "call-1", ToolName: "task", Args: taskArgs("reviewer", "first")},
			llm.ToolCallEndEvent{CallID: "call-2", ToolName: "task", Args: taskArgs("reviewer", "second")},
			llm.MessageEndEvent{StopReason: llm.StopReasonToolUse},
		},
	}}
	child := &scriptProvider{name: "mock", script: [][]llm.LLMEvent{
		textTurn("looks good"),
	}}
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
	if _, err := rt.Wait(ctx, children[0].ID); err != nil {
		t.Fatalf("Wait(child): %v", err)
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
