package harness_test

import (
	"context"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
)

// thinkingProvider is recordingTextProvider advertising thinking support;
// agent-core downgrades the level to ThinkingOff for a provider that does not.
type thinkingProvider struct{ recordingTextProvider }

func (p *thinkingProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Thinking: true}
}

// TestThinkingLevelReachesProvider pins the AgentRuntimeConfig.Thinking
// pass-through: the configured level must arrive on every LLMRequest the
// prompt engine issues (agent-core's DefaultThinking).
func TestThinkingLevelReachesProvider(t *testing.T) {
	t.Parallel()
	provider := &thinkingProvider{}
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {Initialize: func(context.Context, harness.InstanceID, harness.Env) (harness.AgentRuntimeConfig, error) {
				return harness.AgentRuntimeConfig{
					Model:         "mock/test-model",
					ContextWindow: 8_000,
					Providers:     []llm.LLMProvider{provider},
					Thinking:      llm.ThinkingHigh,
				}, nil
			}},
		},
		Store:         memory.New(),
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 2 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("think hard"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if settled, err := rt.Wait(ctx, res.SubmissionID); err != nil || settled.Status != harness.SettledSucceeded {
		t.Fatalf("Wait: settled=%+v err=%v", settled, err)
	}

	reqs := provider.requests()
	if len(reqs) == 0 {
		t.Fatal("provider saw no requests")
	}
	for i, req := range reqs {
		if req.Thinking != llm.ThinkingHigh {
			t.Errorf("request %d Thinking = %v, want ThinkingHigh", i, req.Thinking)
		}
	}
}
