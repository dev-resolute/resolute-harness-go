package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// subagentTargets returns the sorted set of agent definitions the named
// agent may spawn from the given depth, or nil when the task tool must not
// be injected: the agent has no policy entry, or the depth budget is
// exhausted (HARNESS-15).
func (rt *Runtime) subagentTargets(agent string, depth int) []string {
	targets := rt.subagents[agent]
	if len(targets) == 0 {
		return nil
	}
	if depth >= rt.limits.MaxDepth {
		return nil
	}
	return slices.Sorted(slices.Values(targets))
}

// newTaskTool builds the per-drive task tool: the durable-spawn operation
// injected only for agents with a policy entry and depth budget remaining.
func newTaskTool(rt *Runtime, run *submissionRun, targets []string) pi.RegisteredTool {
	t := &taskTool{rt: rt, run: run, targets: targets, limits: rt.limits}
	return pi.NewDynamicTool("task", t.description(), t.schema(), t.execute)
}

// taskTool is the harness-injected durable-subagent tool (HARNESS-15). One
// instance per drive, closing over the run's correlation and the
// policy-resolved target set. A successful execute admits a child
// submission and returns Suspend: the child's settlement becomes this
// call's tool outcome (the wake, Task 9).
type taskTool struct {
	rt      *Runtime
	run     *submissionRun
	targets []string // policy-resolved, sorted for a stable schema
	limits  SubagentLimits

	// mu serializes same-turn parallel task calls so the replay check and
	// the fan-out bound see each prior call's durable effects.
	mu sync.Mutex
}

// spawnData rides ToolResult.Data from the task tool's Execute to the
// consumer goroutine, which authors the task_spawned record (HARNESS-15).
// Authoring on the consumer serializes the spawn after the
// assistant_tool_call record by construction.
type spawnData struct {
	CallID              string `json:"callId"`
	Agent               string `json:"agent"`
	ChildInstance       string `json:"childInstance"`
	ChildConversationID string `json:"childConversationId"`
	ChildSubmissionID   string `json:"childSubmissionId"`
	Prompt              string `json:"prompt"`
	SpawnRecordID       string `json:"spawnRecordId"`
}

// sanitizeCallID maps a provider-controlled call id into a safe charset for
// the identifiers derived from it — the child Instance (a URL path
// segment), Session, and DispatchID. Any byte outside [A-Za-z0-9_-]
// becomes '-', and the result is bounded to 64 bytes.
func sanitizeCallID(callID string) string {
	const maxLen = 64
	b := []byte(callID)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-' {
			continue
		}
		b[i] = '-'
	}
	if len(b) > maxLen {
		b = b[:maxLen]
	}
	return string(b)
}

// taskParams is the task tool's argument shape.
type taskParams struct {
	Agent  string `json:"agent"`
	Prompt string `json:"prompt"`
}

// description enumerates the allowed targets with their definitions'
// descriptions so the model can route.
func (t *taskTool) description() string {
	var b strings.Builder
	b.WriteString("Spawn a durable child run. Targets: ")
	for i, target := range t.targets {
		if i > 0 {
			b.WriteString("; ")
		}
		fmt.Fprintf(&b, "%s — %s", target, t.rt.agents[target].Description)
	}
	return b.String()
}

func (t *taskTool) schema() json.RawMessage {
	schema, err := json.Marshal(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"agent": map[string]interface{}{
				"type":        "string",
				"enum":        t.targets,
				"description": "The agent to spawn as a durable child run",
			},
			"prompt": map[string]interface{}{
				"type":        "string",
				"description": "The self-contained task for the child",
			},
		},
		"required": []string{"agent", "prompt"},
	})
	if err != nil {
		panic(fmt.Sprintf("marshal task tool schema: %v", err))
	}
	return schema
}

// execute admits one durable child for the call and suspends. Policy
// violations and fan-out overflow are plain error results — never
// suspensions.
//
// Ordering invariant: admission happens here; the spawn record does NOT.
// Execute returns the spawn payload in ToolResult.Data and the consumer
// goroutine authors task_spawned when the call ends, so the record lands
// after the assistant_tool_call record by construction. A crash between
// admission and the record leaves a genuinely dangling call the reconciler
// can error-synthesize — the safe direction. Never suspend without the
// spawn data.
func (t *taskTool) execute(ctx context.Context, callID string, args json.RawMessage) (pi.ToolResult, error) {
	var p taskParams
	if err := json.Unmarshal(args, &p); err != nil {
		return pi.ToolResult{IsError: true, Content: fmt.Sprintf("task: invalid arguments: %v", err)}, nil
	}
	// Belt past the schema enum: providers are not bound by it.
	if !slices.Contains(t.targets, p.Agent) {
		return pi.ToolResult{
			IsError: true,
			Content: fmt.Sprintf("task: %q is not a spawnable agent; allowed targets: %s", p.Agent, strings.Join(t.targets, ", ")),
		}, nil
	}

	// callID is provider-controlled and flows into derived identifiers —
	// the child Instance (a URL path segment), Session, and DispatchID —
	// so sanitize it once and use the safe form for all of them. The raw
	// callID still names the call in records and the parent link.
	safeCallID := sanitizeCallID(callID)
	dispatchID := t.run.sub.ID + ":" + safeCallID

	t.mu.Lock()
	defer t.mu.Unlock()

	// Replay short-circuit FIRST: a callID that already dispatched (a
	// re-executed call from the same batch, or a re-driven attempt) admits
	// nothing new, so it must not count against the fan-out bound.
	replay := true
	if _, err := t.rt.store.GetSubmission(ctx, dispatchID); err != nil {
		if !errors.Is(err, ErrSubmissionNotFound) {
			return pi.ToolResult{}, fmt.Errorf("check dispatch replay for task call %s: %w", callID, err)
		}
		replay = false
	}

	if !replay {
		children, err := t.rt.store.ListChildSubmissions(ctx, t.run.sub.ID)
		if err != nil {
			return pi.ToolResult{}, fmt.Errorf("list child submissions: %w", err)
		}
		live := 0
		for _, ch := range children {
			if ch.Status != StatusSettled {
				live++
			}
		}
		if live >= t.limits.MaxChildrenPerRun {
			return pi.ToolResult{
				IsError: true,
				Content: fmt.Sprintf("task: child limit reached (%d live, max %d); wait for a running child to settle or serialize the work", live, t.limits.MaxChildrenPerRun),
			}, nil
		}
	}

	// The spawn record ID is minted before admission so the child
	// conversation's ParentRef and the spawn record the consumer authors
	// from the result Data name the same ID.
	spawnRecordID := newULID()
	childInstance := string(t.run.sub.SessionKey.Instance) + ":" + safeCallID
	res, err := t.rt.Dispatch(ctx, Dispatch{
		Agent:      p.Agent,
		Instance:   InstanceID(childInstance),
		Session:    "task-" + safeCallID,
		DispatchID: dispatchID,
		Message:    DispatchMessage{Kind: InboundUser, Body: p.Prompt},
		Parent: &SpawnParent{
			SubmissionID:   t.run.sub.ID,
			CallID:         callID,
			ConversationID: t.run.conv.ID,
			SpawnRecordID:  spawnRecordID,
			Depth:          t.run.sub.Depth,
		},
	})
	if err != nil {
		return pi.ToolResult{}, fmt.Errorf("admit child for task call %s: %w", callID, err)
	}

	if replay {
		// The original admission already named a spawn record in the child
		// conversation's ParentRef; reuse that ID so the record the
		// consumer authors keeps the link valid.
		id, err := t.childSpawnRecordID(ctx, res.ConversationID)
		if err != nil {
			return pi.ToolResult{}, err
		}
		if id != "" {
			spawnRecordID = id
		}
	}

	data, err := json.Marshal(spawnData{
		CallID:              callID,
		Agent:               p.Agent,
		ChildInstance:       childInstance,
		ChildConversationID: res.ConversationID,
		ChildSubmissionID:   res.SubmissionID,
		Prompt:              p.Prompt,
		SpawnRecordID:       spawnRecordID,
	})
	if err != nil {
		return pi.ToolResult{}, fmt.Errorf("marshal spawn data for task call %s: %w", callID, err)
	}
	// Task 12: emit SubmissionSpawnedEvent
	return pi.ToolResult{Suspend: true, Data: data}, nil
}

// childSpawnRecordID recovers the spawn record ID the child conversation
// was created with, so a replayed admission reuses it. Empty when the
// child conversation has no ParentRef (partial-honor, or a crash window);
// the caller keeps its fresh ID and consumer-side dedupe still bounds the
// records to one.
func (t *taskTool) childSpawnRecordID(ctx context.Context, childConversationID string) (string, error) {
	recs, err := t.rt.store.ReadRecords(ctx, childConversationID, "")
	if err != nil {
		return "", fmt.Errorf("read child records for spawn record id: %w", err)
	}
	for _, rec := range recs {
		if rec.Kind != KindConversationCreated {
			continue
		}
		var p ConversationCreatedPayload
		if err := rec.DecodePayload(&p); err != nil {
			return "", fmt.Errorf("decode conversation_created: %w", err)
		}
		if p.ParentRef != nil {
			return p.ParentRef.SpawnRecordID, nil
		}
		return "", nil
	}
	return "", nil
}

// spawnRecordExists reports whether the conversation already holds a
// task_spawned record for callID. Called from the consumer goroutine — the
// single author of spawn records — so a miss reliably means "author one".
func (r *submissionRun) spawnRecordExists(ctx context.Context, callID string) (bool, error) {
	recs, err := r.rt.store.ReadRecords(ctx, r.conv.ID, "")
	if err != nil {
		return false, fmt.Errorf("read records for spawn dedupe: %w", err)
	}
	for _, rec := range recs {
		if rec.Kind != KindTaskSpawned {
			continue
		}
		var p TaskSpawnedPayload
		if err := rec.DecodePayload(&p); err != nil {
			return false, fmt.Errorf("decode task_spawned for dedupe: %w", err)
		}
		if p.CallID == callID {
			return true, nil
		}
	}
	return false, nil
}

// waitDeadline bounds a suspension when SubagentLimits.MaxWait is set; a
// zero WaitUntil means an unbounded wait.
func (rt *Runtime) waitDeadline() time.Time {
	if rt.limits.MaxWait <= 0 {
		return time.Time{}
	}
	return time.Now().Add(rt.limits.MaxWait)
}
