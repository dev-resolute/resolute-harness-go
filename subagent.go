package harness

import (
	"context"
	"encoding/json"
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

	// mu serializes same-turn parallel task calls so the fan-out bound and
	// the spawn-record dedupe see each prior call's durable effects.
	mu sync.Mutex
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
// Ordering invariant: admission first, spawn record second; Suspend only
// after the record landed. A crash between admission and the record leaves
// a genuinely dangling call the reconciler can error-synthesize — the safe
// direction. Never suspend without the record.
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

	t.mu.Lock()
	defer t.mu.Unlock()

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

	// The spawn record ID is minted before admission so the child
	// conversation's ParentRef and the task_spawned record appended below
	// name the same ID.
	spawnRecordID := newULID()
	childInstance := string(t.run.sub.SessionKey.Instance) + ":" + callID
	res, err := t.rt.Dispatch(ctx, Dispatch{
		Agent:      p.Agent,
		Instance:   InstanceID(childInstance),
		Session:    "task-" + callID,
		DispatchID: t.run.sub.ID + ":" + callID,
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

	// Replay (a re-executed callID): the admission replayed by DispatchID;
	// the spawn record must not duplicate either.
	spawned, err := t.spawnRecordExists(ctx, callID)
	if err != nil {
		return pi.ToolResult{}, err
	}
	if !spawned {
		rec := t.run.record(KindTaskSpawned, &TaskSpawnedPayload{
			CallID:              callID,
			Agent:               p.Agent,
			ChildInstance:       childInstance,
			ChildConversationID: res.ConversationID,
			ChildSubmissionID:   res.SubmissionID,
			Prompt:              p.Prompt,
		})
		rec.ID = spawnRecordID
		if err := t.run.append(ctx, rec); err != nil {
			return pi.ToolResult{}, err
		}
	}
	// Task 12: emit SubmissionSpawnedEvent
	return pi.ToolResult{Suspend: true}, nil
}

// spawnRecordExists reports whether the parent conversation already holds a
// task_spawned record for callID.
func (t *taskTool) spawnRecordExists(ctx context.Context, callID string) (bool, error) {
	recs, err := t.rt.store.ReadRecords(ctx, t.run.conv.ID, "")
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
