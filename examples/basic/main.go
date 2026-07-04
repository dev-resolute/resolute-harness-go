// Command basic is the runnable end-to-end example for resolute-harness-go:
// one agent definition with per-instance setup, a SQLite store, a logging
// Observer, a timing Interceptor, and one real tool, served over HTTP.
//
// Run it keyless (a deterministic local provider stands in for the model):
//
//	go run ./examples/basic
//
// or against Gemini by setting GEMINI_API_KEY (and optionally MODEL, e.g.
// "gemini/gemini-3.1-pro-preview").
//
// Then, in another terminal:
//
//	# dispatch asynchronously (202 + ids), or block on the result:
//	curl -s localhost:8484/agents/assistant/demo -d '{"kind":"user","body":"what time is it?"}'
//	curl -s 'localhost:8484/agents/assistant/demo?wait=true' -d '{"kind":"user","body":"hello there"}'
//
//	# watch the conversation live (replay + tail; reconnect with Last-Event-ID):
//	curl -N localhost:8484/agents/assistant/demo
//
// Durability walkthrough: dispatch a prompt, `kill -9` this process before
// it settles, start it again — the submission is reclaimed on the same
// SQLite store, re-attempted, and settles; the SSE replay shows records
// from both attempts (distinguished by attemptId).
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
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

	dataDir := envOr("DATA_DIR", "./harness-data")
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
			"assistant": {Initialize: initializeAssistant},
		},
		Store:        store,
		Logger:       logger,
		Observers:    []harness.Observer{loggingObserver(logger)},
		Interceptors: []harness.Interceptor{timingInterceptor(logger)},
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

	addr := envOr("ADDR", ":8484")
	server := &http.Server{
		Addr:              addr,
		Handler:           rt.Handler(), // auth etc. would wrap here — plain net/http middleware
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("harness up", "addr", addr, "dataDir", dataDir, "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// initializeAssistant is the AgentDefinition initializer: it runs on every
// claim, so per-instance dynamic setup (here: an instance-specific system
// prompt and an env-gated provider choice) is first-class.
func initializeAssistant(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
	cfg := harness.AgentRuntimeConfig{
		SystemPrompt: fmt.Sprintf("You are the assistant for workspace %q. Be brief. Use the clock tool when asked about time.", id),
		Tools:        []pi.RegisteredTool{clockTool()},
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
	// Keyless default: a deterministic local provider.
	cfg.Providers = []llm.LLMProvider{&localProvider{}}
	cfg.Model = "local/echo-1"
	cfg.ContextWindow = 100_000
	return cfg, nil
}

// clockTool is the example's one real tool.
func clockTool() pi.RegisteredTool {
	return pi.NewTool(pi.Tool[struct{}]{
		Name:        "clock",
		Description: "Returns the current wall-clock time in RFC3339.",
		Execute: func(ctx context.Context, _ struct{}) (pi.ToolResult, error) {
			return pi.ToolResult{Content: time.Now().Format(time.RFC3339)}, nil
		},
	})
}

// loggingObserver narrates engine decisions with the same correlation ids
// as the canonical records.
func loggingObserver(logger *slog.Logger) harness.Observer {
	return func(ev harness.HarnessEvent) {
		switch e := ev.(type) {
		case harness.SubmissionAdmittedEvent:
			logger.Info("admitted", "submission", e.SubmissionID, "session", e.SessionKey)
		case harness.SubmissionClaimedEvent:
			logger.Info("claimed", "submission", e.SubmissionID, "attempt", e.AttemptID, "attemptCount", e.AttemptCount)
		case harness.ToolCallEndedEvent:
			logger.Info("tool finished", "tool", e.ToolName, "call", e.CallID, "isError", e.IsError)
		case harness.RecoveryEvent:
			logger.Warn("recovery", "decision", e.Decision, "submission", e.SubmissionID, "detail", e.Detail)
		case harness.SubmissionSettledEvent:
			logger.Info("settled", "submission", e.SubmissionID, "status", e.Payload.Status, "error", e.Payload.Error)
		}
	}
}

// timingInterceptor times every operation boundary; a real tracing adapter
// (OTel) would start spans here and propagate them via ctx.
func timingInterceptor(logger *slog.Logger) harness.Interceptor {
	return func(ctx context.Context, op harness.OpInfo, next func(context.Context) error) error {
		start := time.Now()
		err := next(ctx)
		logger.Debug("op", "kind", op.Kind, "operation", op.Operation, "tool", op.ToolName,
			"submission", op.SubmissionID, "took", time.Since(start))
		return err
	}
}

// localProvider is the keyless stand-in model: it echoes the last user
// message and calls the clock tool when asked about the time. It exists so
// `go run ./examples/basic` works with zero setup; set GEMINI_API_KEY for
// the real thing.
type localProvider struct{}

func (*localProvider) Name() string { return "local" }

func (*localProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true, ToolCalling: true}
}

func (*localProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		lastUser, lastClock := "", ""
		for _, m := range req.Messages {
			switch c := m.Content.(type) {
			case llm.TextContent:
				if m.Role == "user" {
					lastUser = c.Text
				}
			case llm.ToolResultContent:
				if c.ToolName == "clock" {
					lastClock = c.Content
				}
			}
		}

		if lastClock != "" {
			return respondText(req, emit, "The clock tool says it is "+lastClock+".")
		}
		if strings.Contains(strings.ToLower(lastUser), "time") {
			tc := llm.ToolCallContent{CallID: "call-clock-1", ToolName: "clock", Args: []byte(`{}`)}
			if err := emit(llm.ToolCallStartEvent{CallID: tc.CallID, ToolName: tc.ToolName, Args: tc.Args}); err != nil {
				return nil, err
			}
			if err := emit(llm.ToolCallEndEvent{CallID: tc.CallID}); err != nil {
				return nil, err
			}
			if err := emit(llm.MessageEndEvent{}); err != nil {
				return nil, err
			}
			return append(req.Messages, llm.Message{Role: "assistant", Content: tc}), nil
		}
		return respondText(req, emit, "You said: "+lastUser)
	})
}

func respondText(req llm.LLMRequest, emit func(llm.LLMEvent) error, text string) ([]llm.Message, error) {
	for _, chunk := range splitChunks(text, 12) {
		if err := emit(llm.TextDeltaEvent{Delta: chunk}); err != nil {
			return nil, err
		}
	}
	if err := emit(llm.MessageEndEvent{}); err != nil {
		return nil, err
	}
	return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: text}}), nil
}

func splitChunks(s string, size int) []string {
	var out []string
	for len(s) > size {
		out = append(out, s[:size])
		s = s[size:]
	}
	return append(out, s)
}

// providerMode names the active provider for the startup log line.
func providerMode() string {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return "gemini (env-gated)"
	}
	return "local echo (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
