// Command coder is the coding-assistant example for resolute-harness-go: one
// agent wired to the four built-in execution tools (read, write, edit, bash)
// from resolute-agent-core-go's tools package, rooted at a workspace
// directory, with a stdout observer that narrates tool activity — including
// a running bash command's partial output — as it happens.
//
// Run it keyless (a deterministic local provider stands in for the model):
//
//	go run ./examples/coder
//
// or against Gemini by setting GEMINI_API_KEY (and optionally MODEL, e.g.
// "gemini/gemini-3.1-pro-preview") — see README.md for the three live-gate
// scenarios (write+read+edit, image read, streamed bash) this example is
// built to demonstrate.
//
// Then, in another terminal:
//
//	# dispatch asynchronously (202 + ids), or block on the result:
//	curl -s localhost:8490/agents/coder/demo -d '{"kind":"user","body":"list the files in your workspace"}'
//	curl -s 'localhost:8490/agents/coder/demo?wait=true' -d '{"kind":"user","body":"hello there"}'
//
//	# watch the conversation live (replay + tail; reconnect with Last-Event-ID):
//	curl -N localhost:8490/agents/coder/demo
//
// The workspace tools operate under WORKSPACE (default DATA_DIR/workspace),
// created on first claim.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
	"github.com/dev-resolute/resolute-agent-core-go/tools"
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
			"coder": {Initialize: initializeCoder(dataDir)},
		},
		Store:     store,
		Logger:    logger,
		Observers: []harness.Observer{toolActivityObserver()},
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

	addr := envOr("ADDR", ":8490")
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

// initializeCoder builds the AgentDefinition initializer for the "coder"
// agent: a coding assistant with the built-in execution tools rooted in a
// workspace directory (WORKSPACE, default DATA_DIR/workspace). dataDir is
// closed over rather than read from a package global, so the default
// workspace location stays explicit config rather than mutable state
// (golang.md CFG-2).
func initializeCoder(dataDir string) func(context.Context, harness.InstanceID, harness.Env) (harness.AgentRuntimeConfig, error) {
	return func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
		workspace := envOr("WORKSPACE", filepath.Join(dataDir, "workspace"))
		if err := os.MkdirAll(workspace, 0o755); err != nil {
			return harness.AgentRuntimeConfig{}, fmt.Errorf("create workspace: %w", err)
		}
		execEnv, err := tools.NewOSEnv(workspace)
		if err != nil {
			return harness.AgentRuntimeConfig{}, fmt.Errorf("execution env: %w", err)
		}

		cfg := harness.AgentRuntimeConfig{
			SystemPrompt: fmt.Sprintf("You are a coding assistant working in %s. Use the read, write, edit, and bash tools to complete tasks. Prefer edit over write for existing files.", workspace),
			Tools: []pi.RegisteredTool{
				tools.NewReadTool(tools.ReadToolOptions{Env: execEnv}),
				tools.NewWriteTool(tools.WriteToolOptions{Env: execEnv}),
				tools.NewEditTool(tools.EditToolOptions{Env: execEnv}),
				tools.NewBashTool(tools.BashToolOptions{Env: execEnv}),
			},
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
}

// toolActivityObserver prints tool call lifecycle events to stdout,
// including ToolCallUpdatedEvent's partial results — so a long-running bash
// command's progress (e.g. a multi-second loop) is visible line-by-line
// while it runs, not just once at the end.
func toolActivityObserver() harness.Observer {
	return func(ev harness.HarnessEvent) {
		switch e := ev.(type) {
		case harness.ToolCallStartedEvent:
			fmt.Printf("[tool %s] start\n", e.ToolName)
		case harness.ToolCallUpdatedEvent:
			fmt.Printf("[tool %s] %s\n", e.ToolName, lastLine(e.Result.Content))
		case harness.ToolCallEndedEvent:
			status := "ok"
			if e.IsError {
				status = "error"
			}
			fmt.Printf("[tool %s] done (%s)\n", e.ToolName, status)
		}
	}
}

// lastLine returns the trailing non-empty line of s: trailing newlines are
// trimmed first, then the text after the last remaining newline (or all of
// s, if it has none) is returned. Used to print a streaming tool's growing
// output buffer as a single line of progress instead of the whole buffer.
func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if idx := strings.LastIndexByte(s, '\n'); idx >= 0 {
		return s[idx+1:]
	}
	return s
}

// localProvider is the keyless stand-in model: it echoes the last user
// message. It exists so `go run ./examples/coder` works with zero setup;
// set GEMINI_API_KEY for the real thing, which is what exercises the
// read/write/edit/bash tools (see README.md).
type localProvider struct{}

func (*localProvider) Name() string { return "local" }

func (*localProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true, ToolCalling: true}
}

func (*localProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		lastUser := ""
		for _, m := range req.Messages {
			if tc, ok := m.Content.(llm.TextContent); ok && m.Role == "user" {
				lastUser = tc.Text
			}
		}
		text := "You said: " + lastUser
		for _, chunk := range splitChunks(text, 12) {
			if err := emit(llm.TextDeltaEvent{Delta: chunk}); err != nil {
				return nil, err
			}
		}
		if err := emit(llm.MessageEndEvent{}); err != nil {
			return nil, err
		}
		return append(req.Messages, llm.Message{Role: "assistant", Content: llm.TextContent{Text: text}}), nil
	})
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
