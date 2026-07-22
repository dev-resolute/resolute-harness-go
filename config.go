package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	llm "github.com/dev-resolute/resolute-llm-go"
)

// InstanceID identifies one durable agent instance, addressed
// /agents/{name}/{id}. Instances are materialized on first dispatch and
// exist only in the stores.
type InstanceID string

// Env is the injection seam for secrets and config lookup inside
// AgentDefinition initializers, keeping definitions unit-testable.
type Env interface {
	// Secret returns the named secret, or "" when absent.
	Secret(name string) string
}

type osEnv struct{}

func (osEnv) Secret(name string) string { return os.Getenv(name) }

// OSEnv returns an Env backed by the process environment. It is the default
// when Config.Env is nil.
func OSEnv() Env { return osEnv{} }

// AgentRuntimeConfig is the result of an AgentDefinition initializer: the
// complete, catalog-free declaration of how one agent instance runs
// (ADR-0007). Model is a "provider/model" ref resolved against Providers by
// name; ContextWindow is required and drives compaction thresholds.
type AgentRuntimeConfig struct {
	Model         string
	ContextWindow int
	MaxTokens     int
	Providers     []llm.LLMProvider
	SystemPrompt  string
	Tools         []pi.RegisteredTool
	Skills        []pi.Skill
	// ReserveTokens and KeepRecentTokens tune agent-core's compaction cut
	// point; zero values use agent-core's defaults.
	ReserveTokens    int
	KeepRecentTokens int
	// SummarizationRetry configures agent-core's bounded retry of transient
	// summarization failures during Compact (agent-core v0.7.0). The zero
	// value disables retries; enabling it keeps a transient 429/5xx from
	// failing a compaction outright. Retry lifecycle is surfaced to
	// Observers as RecoveryEvents.
	SummarizationRetry pi.SummarizationRetryPolicy
	// MaxAttempts is the durability budget on execution tries, recomputed
	// from durable history on every claim; 0 means DefaultMaxAttempts.
	MaxAttempts int
	// SubmissionTimeout bounds a submission's total lifetime from admission;
	// 0 means DefaultSubmissionTimeout.
	SubmissionTimeout time.Duration
}

// Durability budget defaults (architecture.md §4.2 invariant 7).
const (
	DefaultMaxAttempts       = 10
	DefaultSubmissionTimeout = time.Hour
)

func (c AgentRuntimeConfig) validate() error {
	if c.Model == "" {
		return errors.New("agent runtime config: Model is required")
	}
	if c.ContextWindow <= 0 {
		return errors.New("agent runtime config: ContextWindow is required and must be positive")
	}
	if len(c.Providers) == 0 {
		return errors.New("agent runtime config: at least one provider is required")
	}
	return nil
}

// AgentDefinition is a named initializer registered in Runtime config
// (ADR-0009). Initialize runs on every claim, so per-instance dynamic setup
// — tenant prompts, per-user tools — is first-class.
type AgentDefinition struct {
	Initialize func(ctx context.Context, id InstanceID, env Env) (AgentRuntimeConfig, error)
}

// Config carries everything NewRuntime needs: the named agent definitions,
// the store, and optional environment, logging, and engine-timing seams.
type Config struct {
	Agents map[string]AgentDefinition
	Store  Store
	// Env is passed to every AgentDefinition initializer; nil means OSEnv().
	Env Env
	// Logger receives engine diagnostics; nil means slog.Default().
	Logger *slog.Logger
	// ClaimInterval is the coordinator's poll cadence between wake nudges;
	// 0 means 250ms.
	ClaimInterval time.Duration
	// LeaseDuration bounds attempt ownership; heartbeats renew at a third of
	// it. 0 means 30s.
	LeaseDuration time.Duration
	// DeltaFlushBytes flushes a pending delta batch once it reaches this
	// size; 0 means 1024. Message boundaries always flush regardless.
	DeltaFlushBytes int
	// DeltaFlushInterval flushes a pending delta batch once its oldest
	// fragment is this stale; 0 means 200ms.
	DeltaFlushInterval time.Duration
	// Observers receive ephemeral HarnessEvents synchronously (ADR-0008).
	Observers []Observer
	// Interceptors wrap every operation boundary in registration order
	// (first is outermost).
	Interceptors []Interceptor
}

func (c Config) validate() error {
	if len(c.Agents) == 0 {
		return errors.New("runtime config: at least one agent definition is required")
	}
	for name, def := range c.Agents {
		if name == "" {
			return errors.New("runtime config: agent name must be non-empty")
		}
		if def.Initialize == nil {
			return fmt.Errorf("runtime config: agent %q has no Initialize function", name)
		}
	}
	if c.Store == nil {
		return errors.New("runtime config: Store is required")
	}
	return nil
}
