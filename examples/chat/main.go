// Command chat is the browser example for resolute-harness-go: a single Go
// binary serving an embedded HTML chat page over the harness's own HTTP
// surface — no npm, no build step, no external assets.
//
// The page demonstrates the full read/write surface:
//
//   - dispatches are plain POSTs; the reply streams in live over the
//     conversation's SSE feed (deltas, tool calls, settlement);
//   - the session box switches between durable conversations of the same
//     instance — switching back replays the full history from the store;
//   - the Steer box injects guidance into a run mid-flight: ask the agent to
//     "research something" (the research tool takes ~4s) and steer while the
//     tool runs; the model's final answer visibly incorporates the steer.
//
// Run it keyless (a deterministic local provider stands in for the model):
//
//	go run ./examples/chat
//
// then open http://localhost:8485 — or set GEMINI_API_KEY (and optionally
// MODEL) first to chat with the real thing.
//
// Durability walkthrough: send a message, kill the process, restart it, and
// reload the page — the conversation replays from SQLite, and a submission
// killed mid-run is reclaimed and settles after the restart.
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

	dataDir := envOr("DATA_DIR", "./chat-data")
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
			"chat": {Initialize: initializeChat},
		},
		Store:  store,
		Logger: logger,
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

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(page))
	})
	mux.Handle("/", rt.Handler())

	addr := envOr("ADDR", ":8485")
	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("chat up", "url", "http://localhost"+addr, "dataDir", dataDir, "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func initializeChat(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
	cfg := harness.AgentRuntimeConfig{
		SystemPrompt: "You are a friendly chat assistant. Use the research tool when asked to research a topic. When guidance arrives mid-run, incorporate it into your answer and say so.",
		Tools:        []pi.RegisteredTool{researchTool(4 * time.Second)},
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
	cfg.Providers = []llm.LLMProvider{&chatProvider{}}
	cfg.Model = "local/chat-1"
	cfg.ContextWindow = 100_000
	return cfg, nil
}

// researchTool is deliberately slow: it holds the turn open long enough to
// steer the run from the browser while the tool executes.
func researchTool(d time.Duration) pi.RegisteredTool {
	type args struct {
		Topic string `json:"topic"`
	}
	return pi.NewTool(pi.Tool[args]{
		Name:        "research",
		Description: "Research a topic (takes a few seconds). Args: topic.",
		Execute: func(ctx context.Context, a args) (pi.ToolResult, error) {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return pi.ToolResult{}, ctx.Err()
			}
			return pi.ToolResult{Content: fmt.Sprintf("three sources agree %q is promising; one dissents on cost", a.Topic)}, nil
		},
	})
}

// chatProvider is the keyless stand-in model. "research <topic>" triggers the
// slow tool and the final answer names any guidance steered in while the tool
// ran, so the browser demo shows steering changing a live run deterministically.
type chatProvider struct{}

func (*chatProvider) Name() string { return "local" }

func (*chatProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true, ToolCalling: true}
}

func (*chatProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		turn := readTurn(req.Messages)

		if err := emit(llm.ThinkingDeltaEvent{Delta: "reading the conversation…"}); err != nil {
			return nil, err
		}

		switch {
		case turn.researchTopic != "" && !turn.researched:
			args := fmt.Sprintf(`{"topic":%q}`, turn.researchTopic)
			tc := llm.ToolCallContent{
				CallID:   fmt.Sprintf("call-research-%d", len(req.Messages)),
				ToolName: "research",
				Args:     []byte(args),
			}
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

		case turn.researched:
			text := fmt.Sprintf("Research on %q finished: %s.", turn.researchTopic, turn.findings)
			if turn.steer != "" {
				text += fmt.Sprintf(" You steered me mid-run (%q) — factored in.", turn.steer)
			}
			return respondText(req, emit, text)

		default:
			return respondText(req, emit,
				fmt.Sprintf("You said: %q. Try \"research quantum kettles\" and press Steer while the tool runs.", turn.lastUser))
		}
	})
}

// turnState is what the scripted model extracts from the transcript: the
// latest research request, whether its tool already ran, and any guidance
// steered in after it.
type turnState struct {
	lastUser      string
	researchTopic string // topic of the latest "research …" request
	researched    bool   // its tool result is already in the transcript
	findings      string
	steer         string // last user text injected after the research request
}

// readTurn scans the transcript positionally: only tool results and steers
// after the LATEST research request belong to the current turn — earlier
// research exchanges are settled history.
func readTurn(msgs []llm.Message) turnState {
	var t turnState
	reqIdx := -1
	for i, m := range msgs {
		tc, ok := m.Content.(llm.TextContent)
		if !ok || m.Role != "user" {
			continue
		}
		t.lastUser = tc.Text
		if topic, found := strings.CutPrefix(strings.ToLower(tc.Text), "research "); found {
			t.researchTopic, reqIdx = strings.TrimSpace(topic), i
			t.researched, t.steer = false, ""
		} else if reqIdx >= 0 && i > reqIdx {
			t.steer = tc.Text
		}
	}
	if reqIdx < 0 {
		return t
	}
	for _, m := range msgs[reqIdx:] {
		if tr, ok := m.Content.(llm.ToolResultContent); ok && tr.ToolName == "research" {
			t.researched, t.findings = true, tr.Content
		}
	}
	return t
}

func respondText(req llm.LLMRequest, emit func(llm.LLMEvent) error, text string) ([]llm.Message, error) {
	for _, chunk := range splitChunks(text, 16) {
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

func providerMode() string {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return "gemini (env-gated)"
	}
	return "local chat (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
