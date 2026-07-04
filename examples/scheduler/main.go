// Command scheduler is the time-driven example for resolute-harness-go: an
// in-process ticker fires scheduled signal dispatches into a durable
// conversation, and deterministic per-window dispatch ids make the schedule
// idempotent — a killed-and-restarted process re-fires the current window's
// dispatch and the store deduplicates it, so nothing runs twice.
//
// Run it keyless with a fast tick to watch it work:
//
//	TICK=10s go run ./examples/scheduler
//
// Then, in another terminal:
//
//	# watch digests accumulate, one signal + one digest per window:
//	curl -N localhost:8488/agents/reporter/daily
//
// Durability walkthrough: note the current window in the logs, `kill -9` the
// process, and restart it within the same window. The scheduler immediately
// re-dispatches that window's tick — the log shows "replayed (idempotent)"
// instead of "admitted", and the record stream grows no duplicate signal.
// The same property holds for real deployments: N replicas can all run this
// scheduler against one store and each window still fires exactly once.
//
// Set GEMINI_API_KEY (and optionally MODEL) to have a real model write the
// digests; the tick cadence is unchanged.
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

	interval, err := time.ParseDuration(envOr("TICK", "30s"))
	if err != nil || interval < time.Second {
		return fmt.Errorf("TICK must be a duration of at least 1s, got %q", os.Getenv("TICK"))
	}

	dataDir := envOr("DATA_DIR", "./scheduler-data")
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
			"reporter": {Initialize: initializeReporter},
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

	go runSchedule(ctx, scheduleParams{
		rt:       rt,
		store:    store,
		interval: interval,
		logger:   logger,
	})

	addr := envOr("ADDR", ":8488")
	server := &http.Server{Addr: addr, Handler: rt.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("scheduler up", "addr", addr, "dataDir", dataDir, "tick", interval, "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// scheduleParams carries the schedule loop's dependencies.
type scheduleParams struct {
	rt       *harness.Runtime
	store    harness.Store
	interval time.Duration
	logger   *slog.Logger
}

// runSchedule fires one dispatch per wall-clock window until ctx ends. It
// dispatches immediately for the window it starts in — safe on restart,
// because the window's deterministic dispatch id makes the re-fire an
// idempotent replay.
func runSchedule(ctx context.Context, p scheduleParams) {
	for {
		window := time.Now().Truncate(p.interval)
		dispatchTick(ctx, p, window)
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Until(window.Add(p.interval))):
		}
	}
}

// dispatchTick admits the signal for one schedule window. The dispatch id is
// derived from the window's timestamp: every process that fires this window
// produces the same id, so exactly one submission exists per window.
func dispatchTick(ctx context.Context, p scheduleParams, window time.Time) {
	dispatchID := fmt.Sprintf("standup-%d", window.Unix())
	// The dispatch id doubles as the submission id, so the store says whether
	// this window already fired (e.g. before a restart).
	outcome := "admitted"
	if _, err := p.store.GetSubmission(ctx, dispatchID); err == nil {
		outcome = "replayed (idempotent)"
	} else if !errors.Is(err, harness.ErrSubmissionNotFound) {
		p.logger.Error("probe tick", "window", window, "error", err)
		return
	}

	res, err := p.rt.Dispatch(ctx, harness.Dispatch{
		Agent:      "reporter",
		Instance:   "daily",
		DispatchID: dispatchID,
		Message: harness.SignalMessage(
			fmt.Sprintf("Compile the standup digest for the window starting %s.", window.Format(time.RFC3339)),
			harness.SignalMeta{
				Type:   "scheduled_tick",
				Sender: map[string]string{"source": "scheduler", "schedule": "standup"},
				Tag:    dispatchID,
			},
		),
	})
	if err != nil {
		if !errors.Is(err, context.Canceled) {
			p.logger.Error("dispatch tick", "window", window, "error", err)
		}
		return
	}
	p.logger.Info("tick", "window", window.Format(time.RFC3339), "dispatch", dispatchID,
		"submission", res.SubmissionID, "outcome", outcome)
}

func initializeReporter(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
	cfg := harness.AgentRuntimeConfig{
		SystemPrompt: "You are a standup reporter. Each scheduled tick, write a two-sentence digest for the named window.",
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
	cfg.Providers = []llm.LLMProvider{&digestProvider{}}
	cfg.Model = "local/digest-1"
	cfg.ContextWindow = 100_000
	return cfg, nil
}

// digestProvider is the keyless stand-in model: it answers each tick with a
// canned digest that echoes the window it was asked about, so the record
// stream shows which tick produced which digest.
type digestProvider struct{}

func (*digestProvider) Name() string { return "local" }

func (*digestProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true}
}

func (*digestProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		window := "an unknown window"
		for _, m := range req.Messages {
			tc, ok := m.Content.(llm.TextContent)
			if !ok {
				continue
			}
			if _, after, found := strings.Cut(tc.Text, "window starting "); found {
				window = strings.Trim(strings.Fields(after)[0], `".,\`)
			}
		}
		text := fmt.Sprintf("Digest for %s: all queues drained, no incidents; next check in one window.", window)
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
	return "local digest (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
