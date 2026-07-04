// Command triage is the structured-results example for resolute-harness-go:
// a bug-triage endpoint where the dispatch carries a resultSchema, the
// harness validates the run's final answer against it, and the corrective
// retry loop is visible in the record stream.
//
// Run it keyless (a deterministic local provider stands in for the model —
// it deliberately answers in prose first and only conforms after the
// harness's schema feedback, so every dispatch demonstrates the loop):
//
//	go run ./examples/triage
//
// Then, in another terminal:
//
//	# triage a bug and block for the validated JSON result:
//	curl -s 'localhost:8486/agents/triage/demo?wait=true' -d '{
//	  "kind": "user",
//	  "body": "clicking login crashes the app on v2.1",
//	  "resultSchema": {
//	    "type": "object",
//	    "properties": {
//	      "severity":  {"type": "string", "enum": ["low", "medium", "high"]},
//	      "component": {"type": "string"}
//	    },
//	    "required": ["severity", "component"],
//	    "additionalProperties": false
//	  }
//	}'
//
//	# the corrective turn is durable: the replay shows TWO user_message
//	# records (the prompt, then the harness's schema feedback) before the
//	# conforming answer:
//	curl -N localhost:8486/agents/triage/demo
//
//	# exhaust the retry budget: this provider never conforms when the bug
//	# report contains "prose only", so a budget of 1 settles
//	# failed/result_schema_invalid:
//	curl -s 'localhost:8486/agents/triage/demo?wait=true' -d '{
//	  "kind": "user", "body": "prose only please", "resultRetries": 1,
//	  "resultSchema": {"type":"object","properties":{"severity":{"type":"string"}},"required":["severity"],"additionalProperties":false}
//	}'
//
// Set GEMINI_API_KEY (and optionally MODEL) to triage with the real model
// instead; real models usually conform on the first try, so the corrective
// turn only appears when they slip.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
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

	dataDir := envOr("DATA_DIR", "./triage-data")
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
			"triage": {Initialize: initializeTriage},
		},
		Store:  store,
		Logger: logger,
		// Narrate settlements: schema failures surface here as
		// failed/result_schema_invalid. The corrective turns themselves are
		// canonical user_message records, visible in the SSE stream.
		Observers: []harness.Observer{func(ev harness.HarnessEvent) {
			if e, ok := ev.(harness.SubmissionSettledEvent); ok {
				logger.Info("settled", "submission", e.SubmissionID,
					"status", e.Payload.Status, "errorCode", e.Payload.ErrorCode)
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

	addr := envOr("ADDR", ":8486")
	server := &http.Server{Addr: addr, Handler: rt.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("triage up", "addr", addr, "dataDir", dataDir, "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func initializeTriage(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
	cfg := harness.AgentRuntimeConfig{
		SystemPrompt: "You are a bug-triage agent. Classify incoming bug reports. When asked for a structured result, answer with the JSON only.",
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
	cfg.Providers = []llm.LLMProvider{&stubbornProvider{}}
	cfg.Model = "local/triage-prose-1"
	cfg.ContextWindow = 100_000
	return cfg, nil
}

// stubbornProvider is the keyless stand-in model, scripted to make the
// validation loop visible: it answers the original prompt in prose, and only
// conforms once the harness's corrective turn (which names the schema) is in
// the transcript. Reports containing "prose only" never conform, which
// demonstrates the ResultRetries budget settling failed/result_schema_invalid.
type stubbornProvider struct{}

func (*stubbornProvider) Name() string { return "local" }

func (*stubbornProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true}
}

func (*stubbornProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		report, corrected := "", false
		for _, m := range req.Messages {
			tc, ok := m.Content.(llm.TextContent)
			if !ok || m.Role != "user" {
				continue
			}
			if strings.Contains(tc.Text, "schema") {
				corrected = true // the harness's validation feedback turn
			} else {
				report = tc.Text // the actual bug report
			}
		}

		if !corrected || strings.Contains(strings.ToLower(report), "prose only") {
			return respondText(req, emit, "Sounds rough! I'd say that is a pretty serious bug somewhere around the login flow.")
		}
		answer, err := json.Marshal(classify(report))
		if err != nil {
			return nil, fmt.Errorf("marshal classification: %w", err)
		}
		return respondText(req, emit, string(answer))
	})
}

// classification is the shape the demo schema asks for.
type classification struct {
	Severity  string `json:"severity"`
	Component string `json:"component"`
}

// classify derives a deterministic classification from the report text so
// the demo output visibly depends on the input.
func classify(report string) classification {
	lower := strings.ToLower(report)
	c := classification{Severity: "low", Component: "backend"}
	switch {
	case strings.Contains(lower, "crash"), strings.Contains(lower, "breaks"), strings.Contains(lower, "data loss"):
		c.Severity = "high"
	case strings.Contains(lower, "slow"), strings.Contains(lower, "flaky"):
		c.Severity = "medium"
	}
	switch {
	case strings.Contains(lower, "login"), strings.Contains(lower, "auth"):
		c.Component = "auth"
	case strings.Contains(lower, "button"), strings.Contains(lower, "render"), strings.Contains(lower, "ui"):
		c.Component = "ui"
	}
	return c
}

func respondText(req llm.LLMRequest, emit func(llm.LLMEvent) error, text string) ([]llm.Message, error) {
	if err := emit(llm.TextDeltaEvent{Delta: text}); err != nil {
		return nil, err
	}
	if err := emit(llm.MessageEndEvent{}); err != nil {
		return nil, err
	}
	return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: text}}), nil
}

func providerMode() string {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return "gemini (env-gated)"
	}
	return "local stubborn prose (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
