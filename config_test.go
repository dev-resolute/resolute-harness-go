package harness

import (
	"context"
	"strings"
	"testing"
	"time"
)

// nopStore satisfies Store for NewRuntime validation tests; NewRuntime never
// calls a store method.
type nopStore struct{ Store }

// policyTestAgents returns two definitions so policy entries can reference
// real parents and targets.
func policyTestAgents() map[string]AgentDefinition {
	init := func(ctx context.Context, id InstanceID, env Env) (AgentRuntimeConfig, error) {
		return AgentRuntimeConfig{}, nil
	}
	return map[string]AgentDefinition{
		"parent":     {Initialize: init},
		"researcher": {Initialize: init},
	}
}

// TestResolveSubagentLimitsDefaults pins the documented defaults (HARNESS-15):
// a zero SubagentLimits resolves to (8, 1, 0, CancelChildren).
func TestResolveSubagentLimitsDefaults(t *testing.T) {
	t.Parallel()
	got := Config{}.resolveLimits()
	want := SubagentLimits{MaxChildrenPerRun: 8, MaxDepth: 1, MaxWait: 0, OnParentTerminal: CancelChildren}
	if got != want {
		t.Fatalf("resolveLimits() = %+v, want %+v", got, want)
	}
}

// TestResolveSubagentLimitsExplicitSurvive pins that explicit limit values
// pass through resolution untouched.
func TestResolveSubagentLimitsExplicitSurvive(t *testing.T) {
	t.Parallel()
	in := SubagentLimits{MaxChildrenPerRun: 3, MaxDepth: 2, MaxWait: 30 * time.Second, OnParentTerminal: CancelChildren}
	if got := (Config{SubagentLimits: in}).resolveLimits(); got != in {
		t.Fatalf("resolveLimits() = %+v, want %+v", got, in)
	}
}

// TestNewRuntimeSubagentPolicyUnknownParent pins that a policy entry keyed by
// an unregistered agent name fails NewRuntime and names the parent.
func TestNewRuntimeSubagentPolicyUnknownParent(t *testing.T) {
	t.Parallel()
	_, err := NewRuntime(Config{
		Agents:    policyTestAgents(),
		Store:     nopStore{},
		Subagents: SubagentPolicy{"ghost": {"researcher"}},
	})
	if err == nil {
		t.Fatal("NewRuntime: expected error for unknown parent, got nil")
	}
	if !strings.Contains(err.Error(), `"ghost"`) {
		t.Fatalf("NewRuntime error %q does not name unknown parent %q", err, "ghost")
	}
}

// TestNewRuntimeSubagentPolicyUnknownTarget pins that a policy entry
// targeting an unregistered agent name fails NewRuntime and names both the
// parent and the target.
func TestNewRuntimeSubagentPolicyUnknownTarget(t *testing.T) {
	t.Parallel()
	_, err := NewRuntime(Config{
		Agents:    policyTestAgents(),
		Store:     nopStore{},
		Subagents: SubagentPolicy{"parent": {"ghost"}},
	})
	if err == nil {
		t.Fatal("NewRuntime: expected error for unknown target, got nil")
	}
	if !strings.Contains(err.Error(), `"parent"`) || !strings.Contains(err.Error(), `"ghost"`) {
		t.Fatalf("NewRuntime error %q does not name both parent %q and target %q", err, "parent", "ghost")
	}
}

// TestNewRuntimeSubagentPolicyValid pins that a valid policy passes
// validation and lands on the Runtime alongside the resolved limits.
func TestNewRuntimeSubagentPolicyValid(t *testing.T) {
	t.Parallel()
	policy := SubagentPolicy{"parent": {"researcher"}}
	rt, err := NewRuntime(Config{
		Agents:    policyTestAgents(),
		Store:     nopStore{},
		Subagents: policy,
	})
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if len(rt.subagents) != 1 || len(rt.subagents["parent"]) != 1 || rt.subagents["parent"][0] != "researcher" {
		t.Fatalf("rt.subagents = %v, want %v", rt.subagents, policy)
	}
	wantLimits := SubagentLimits{MaxChildrenPerRun: 8, MaxDepth: 1, MaxWait: 0, OnParentTerminal: CancelChildren}
	if rt.limits != wantLimits {
		t.Fatalf("rt.limits = %+v, want %+v", rt.limits, wantLimits)
	}
}
