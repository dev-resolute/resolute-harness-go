// Command multitenant is the concurrency-model example for resolute-harness-go:
// one agent definition serving many tenants, where the instance id picks the
// tenant (per-instance system prompts via Initialize) and sessions inside an
// instance give independent, durably ordered conversations.
//
// The scheduling contract on display:
//
//   - one session = one queue: submissions to the same session run strictly
//     in admission order (head-of-line);
//   - different sessions — same tenant or different tenants — run
//     concurrently.
//
// Run it keyless (the local provider takes a fixed ~700ms per prompt, which
// makes the ordering visible on a stopwatch):
//
//	go run ./examples/multitenant
//
// Then fire the demo load (four prompts: two queued into acme/support, one
// into acme/billing, one into globex — watch the queued one settle a beat
// later than everything else):
//
//	./examples/multitenant/load.sh
//
// or by hand:
//
//	curl -s 'localhost:8489/agents/concierge/acme?wait=true' -d '{"kind":"user","body":"first in support","session":"support"}'
//	curl -N 'localhost:8489/agents/concierge/acme?session=support'   # watch a session's records
//
// Set GEMINI_API_KEY (and optionally MODEL) to serve every tenant with the
// real model; the scheduling contract is the harness's and does not change.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/gemini"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/sqlite"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run() error {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	dataDir := envOr("DATA_DIR", "./multitenant-data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return fmt.Errorf("create data dir: %w", err)
	}
	store, err := sqlite.Open(dataDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer store.Close()

	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"concierge": {Initialize: initializeConcierge},
		},
		Store:  store,
		Logger: logger,
		// Narrate claims and settlements with session keys, so the demo load's
		// interleaving is visible in this process's log too.
		Observers: []harness.Observer{func(ev harness.HarnessEvent) {
			switch e := ev.(type) {
			case harness.SubmissionClaimedEvent:
				logger.Info("claimed", "instance", e.SessionKey.Instance, "session", e.SessionKey.Session, "submission", e.SubmissionID)
			case harness.SubmissionSettledEvent:
				logger.Info("settled", "instance", e.SessionKey.Instance, "session", e.SessionKey.Session, "submission", e.SubmissionID, "status", e.Payload.Status)
			}
		}},
	})
	if err != nil {
		return fmt.Errorf("build runtime: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := rt.Start(ctx); err != nil {
		return fmt.Errorf("start runtime: %w", err)
	}
	defer rt.Close()

	addr := envOr("ADDR", ":8489")
	server := &http.Server{Addr: addr, Handler: rt.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("multitenant up", "addr", addr, "dataDir", dataDir, "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// initializeConcierge runs on every claim, so each tenant (instance) gets its
// own system prompt from one shared definition — the multi-tenant seam.
func initializeConcierge(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
	cfg := harness.AgentRuntimeConfig{
		SystemPrompt: fmt.Sprintf("You are the dedicated concierge for tenant %q. Answer briefly and mention the tenant by name.", id),
	}
	if key := env.Secret("GEMINI_API_KEY"); key != "" {
		provider, err := gemini.New(gemini.Config{APIKey: key})
		if err != nil {
			return harness.AgentRuntimeConfig{}, fmt.Errorf("gemini provider: %w", err)
		}
		cfg.Providers = []llm.LLMProvider{provider}
		cfg.Model = envOr("MODEL", "gemini/gemini-3.1-pro-preview")
		cfg.ContextWindow = 1_000_000
		return cfg, nil
	}
	cfg.Providers = []llm.LLMProvider{&tenantProvider{tenant: string(id), thinkTime: 700 * time.Millisecond}}
	cfg.Model = "local/concierge-1"
	cfg.ContextWindow = 100_000
	return cfg, nil
}

// tenantProvider is the keyless stand-in model. Unlike the llm mock, it is
// safe for concurrent streams (each call's state is local to the call) —
// concurrent sessions are the whole point of this example. The fixed
// thinkTime makes queueing visible: a session's second prompt settles one
// thinkTime after its first, while other sessions settle in parallel.
type tenantProvider struct {
	tenant    string
	thinkTime time.Duration
}

func (p *tenantProvider) Name() string { return "local" }

func (p *tenantProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true}
}

func (p *tenantProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		select {
		case <-time.After(p.thinkTime):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		lastUser := ""
		for _, m := range req.Messages {
			if tc, ok := m.Content.(llm.TextContent); ok && m.Role == "user" {
				lastUser = tc.Text
			}
		}
		text := fmt.Sprintf("[%s concierge] handled: %s", p.tenant, lastUser)
		if err := emit(llm.TextDeltaEvent{Delta: text}); err != nil {
			return nil, err
		}
		if err := emit(llm.MessageEndEvent{}); err != nil {
			return nil, err
		}
		return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: text}}), nil
	})
}

func providerMode() string {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return "gemini (env-gated)"
	}
	return "local concierge, ~700ms per prompt (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
