// Command github-bot is the channel example for resolute-harness-go: verified
// GitHub webhook ingress translated into signal dispatches, with delivery-id
// idempotency and one narrow application-owned tool that posts the agent's
// reply back to the issue.
//
// Run it keyless (a deterministic local provider stands in for the model, and
// replies are logged instead of posted):
//
//	go run ./examples/github-bot
//
// Then simulate GitHub deliveries in another terminal:
//
//	# a new issue comment (the X-GitHub-Delivery header is the idempotency key):
//	curl -si localhost:8487/webhooks/github \
//	  -H 'X-GitHub-Event: issue_comment' -H 'X-GitHub-Delivery: d-1001' \
//	  -d '{"action":"created","repository":{"full_name":"acme/website"},"issue":{"number":42},"comment":{"body":"Login breaks on v2.1","user":{"login":"alice"}}}'
//
//	# GitHub redelivery (same delivery id, same bytes) → 202 with the SAME submission:
//	#   re-run the exact curl above
//	# a forged/mutated payload under the same delivery id → 409 conflict:
//	#   re-run it with a different comment body
//
//	# watch the triage conversation live (instance is derived from repo+issue):
//	curl -N localhost:8487/agents/triager/acme-website-42
//
// Environment (all optional):
//
//	GITHUB_WEBHOOK_SECRET  verify X-Hub-Signature-256 over the exact request
//	                       bytes; without it, ingress is unverified demo mode
//	GITHUB_TOKEN           post replies to the real GitHub API instead of the log
//	GEMINI_API_KEY, MODEL  use Gemini instead of the keyless local provider
//
// Design notes, mirroring flue's github-channel example: the handler
// completes dispatch admission before returning (GitHub wants a 2xx within
// ten seconds and does not auto-retry), the delivery id deduplicates
// redeliveries durably, and the reply tool is deliberately narrow application
// policy — it can only comment on the one issue its instance is bound to.
package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
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

	dataDir := envOr("DATA_DIR", "./github-bot-data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return fmt.Errorf("create data dir: %w", err)
	}
	store, err := sqlite.Open(dataDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer store.Close()

	issues := &issueRegistry{}
	rt, err := harness.NewRuntime(harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"triager": {Initialize: triagerInitializer(issues, logger)},
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
	mux.Handle("POST /webhooks/github", &webhookHandler{
		rt:     rt,
		issues: issues,
		secret: os.Getenv("GITHUB_WEBHOOK_SECRET"),
		logger: logger,
	})
	mux.Handle("/", rt.Handler())

	addr := envOr("ADDR", ":8487")
	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	logger.Info("github-bot up", "addr", addr, "dataDir", dataDir,
		"ingress", ingressMode(), "replies", replyMode(), "provider", providerMode())
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// issueRef names the one issue an instance is bound to.
type issueRef struct {
	FullName string // owner/repo
	Number   int
}

// issueRegistry maps instance ids to the issue they were derived from, so the
// per-instance reply tool knows its destination. It is in-memory: keyless
// mode never needs it, and a real deployment would derive the destination
// from durable state instead.
type issueRegistry struct{ m sync.Map }

func (r *issueRegistry) put(id harness.InstanceID, ref issueRef) { r.m.Store(id, ref) }

func (r *issueRegistry) get(id harness.InstanceID) (issueRef, bool) {
	v, ok := r.m.Load(id)
	if !ok {
		return issueRef{}, false
	}
	return v.(issueRef), true
}

// instanceFor derives the canonical instance id for an issue: one instance
// (and thus one durable conversation) per repo+issue, like flue's
// github-channel. The same payload always lands in the same conversation.
func instanceFor(ref issueRef) harness.InstanceID {
	repo := strings.ReplaceAll(ref.FullName, "/", "-")
	return harness.InstanceID(fmt.Sprintf("%s-%d", repo, ref.Number))
}

// issueCommentEvent is the subset of GitHub's issue_comment payload the bot
// consumes.
type issueCommentEvent struct {
	Action     string `json:"action"`
	Repository struct {
		FullName string `json:"full_name"`
	} `json:"repository"`
	Issue struct {
		Number int `json:"number"`
	} `json:"issue"`
	Comment struct {
		Body string `json:"body"`
		User struct {
			Login string `json:"login"`
		} `json:"user"`
	} `json:"comment"`
}

// webhookHandler is the ingress seam: it verifies the delivery, derives the
// canonical instance, and completes dispatch admission before responding.
type webhookHandler struct {
	rt     *harness.Runtime
	issues *issueRegistry
	secret string
	logger *slog.Logger
}

func (h *webhookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Signatures cover the exact bytes GitHub sent, so read the raw body
	// before any decoding.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	if h.secret != "" && !verifySignature(h.secret, body, r.Header.Get("X-Hub-Signature-256")) {
		http.Error(w, "signature mismatch", http.StatusUnauthorized)
		return
	}

	if event := r.Header.Get("X-GitHub-Event"); event != "issue_comment" {
		w.WriteHeader(http.StatusNoContent) // ping and friends: verified, ignored
		return
	}
	var ev issueCommentEvent
	if err := json.Unmarshal(body, &ev); err != nil {
		http.Error(w, "malformed payload", http.StatusBadRequest)
		return
	}
	if ev.Action != "created" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if ev.Repository.FullName == "" || ev.Issue.Number == 0 || ev.Comment.Body == "" {
		http.Error(w, "payload missing repository, issue, or comment", http.StatusBadRequest)
		return
	}

	ref := issueRef{FullName: ev.Repository.FullName, Number: ev.Issue.Number}
	instance := instanceFor(ref)
	h.issues.put(instance, ref)

	// The delivery id is the idempotency key: GitHub redeliveries replay the
	// original admission; a mutated payload under the same id is a conflict.
	res, err := h.rt.Dispatch(r.Context(), harness.Dispatch{
		Agent:      "triager",
		Instance:   instance,
		DispatchID: r.Header.Get("X-GitHub-Delivery"),
		Message: harness.SignalMessage(ev.Comment.Body, harness.SignalMeta{
			Type:   "github_issue_comment",
			Sender: map[string]string{"handle": ev.Comment.User.Login, "platform": "github"},
			Tag:    fmt.Sprintf("issue-%d", ev.Issue.Number),
		}),
	})
	if err != nil {
		status := http.StatusInternalServerError
		switch {
		case errors.Is(err, harness.ErrDispatchConflict):
			status = http.StatusConflict
		case errors.Is(err, harness.ErrInvalidDispatch):
			status = http.StatusBadRequest
		}
		http.Error(w, err.Error(), status)
		return
	}
	h.logger.Info("delivery admitted", "delivery", r.Header.Get("X-GitHub-Delivery"),
		"instance", instance, "submission", res.SubmissionID)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(struct {
		harness.DispatchResult
		Watch string `json:"watch"`
	}{res, fmt.Sprintf("/agents/triager/%s", instance)})
}

// verifySignature checks GitHub's X-Hub-Signature-256 ("sha256=<hex>") over
// the exact request bytes.
func verifySignature(secret string, body []byte, header string) bool {
	sig, ok := strings.CutPrefix(header, "sha256=")
	if !ok {
		return false
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	want := hex.EncodeToString(mac.Sum(nil))
	return hmac.Equal([]byte(sig), []byte(want))
}

// triagerInitializer builds the per-instance agent config: a triage prompt
// naming the bound issue and the one narrow reply tool.
func triagerInitializer(issues *issueRegistry, logger *slog.Logger) func(context.Context, harness.InstanceID, harness.Env) (harness.AgentRuntimeConfig, error) {
	return func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
		cfg := harness.AgentRuntimeConfig{
			SystemPrompt: fmt.Sprintf(
				"You are the triage bot for GitHub issue instance %q. When a new comment arrives, acknowledge it with the post_issue_comment tool: thank the commenter by handle and state one concrete next step. Then summarize what you did in one sentence.", id),
			Tools: []pi.RegisteredTool{postCommentTool(id, issues, env, logger)},
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
		cfg.Providers = []llm.LLMProvider{&triageProvider{}}
		cfg.Model = "local/triage-1"
		cfg.ContextWindow = 100_000
		return cfg, nil
	}
}

// postCommentTool is the application-owned reply tool, bound to the one issue
// its instance was derived from. Keyless mode logs the reply; with
// GITHUB_TOKEN set it posts to the real GitHub API.
func postCommentTool(id harness.InstanceID, issues *issueRegistry, env harness.Env, logger *slog.Logger) pi.RegisteredTool {
	type args struct {
		Body string `json:"body"`
	}
	return pi.NewTool(pi.Tool[args]{
		Name:        "post_issue_comment",
		Description: "Post a comment on the GitHub issue this conversation is bound to. Args: body (markdown).",
		Execute: func(ctx context.Context, a args) (pi.ToolResult, error) {
			ref, ok := issues.get(id)
			if !ok {
				return pi.ToolResult{Content: "error: no issue bound to this instance (restart lost the in-memory registry; redeliver the webhook)", IsError: true}, nil
			}
			token := env.Secret("GITHUB_TOKEN")
			if token == "" {
				logger.Info("reply (dry-run, set GITHUB_TOKEN to post for real)",
					"repo", ref.FullName, "issue", ref.Number, "body", a.Body)
				return pi.ToolResult{Content: fmt.Sprintf("dry-run: logged reply to %s#%d", ref.FullName, ref.Number)}, nil
			}
			if err := postGitHubComment(ctx, token, ref, a.Body); err != nil {
				return pi.ToolResult{Content: "github api: " + err.Error(), IsError: true}, nil
			}
			return pi.ToolResult{Content: fmt.Sprintf("posted comment on %s#%d", ref.FullName, ref.Number)}, nil
		},
	})
}

// postGitHubComment POSTs the reply through GitHub's REST API with plain
// net/http — the narrow policy needs one endpoint, not a client library.
func postGitHubComment(ctx context.Context, token string, ref issueRef, body string) error {
	payload, err := json.Marshal(map[string]string{"body": body})
	if err != nil {
		return fmt.Errorf("marshal comment: %w", err)
	}
	url := fmt.Sprintf("https://api.github.com/repos/%s/issues/%d/comments", ref.FullName, ref.Number)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/vnd.github+json")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("post comment: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("post comment: status %d: %s", resp.StatusCode, detail)
	}
	return nil
}

// triageProvider is the keyless stand-in model: on a fresh comment it calls
// the reply tool with a canned acknowledgement, and once the tool result is
// back it closes the turn with a one-line summary.
type triageProvider struct{}

func (*triageProvider) Name() string { return "local" }

func (*triageProvider) Capabilities(string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{Streaming: true, ToolCalling: true}
}

func (*triageProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, func(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, _ map[string]string, _ func(int, map[string]string)) ([]llm.Message, error) {
		var toolOutcome string
		lastText := ""
		for _, m := range req.Messages {
			switch c := m.Content.(type) {
			case llm.TextContent:
				lastText = c.Text
			case llm.ToolResultContent:
				if c.ToolName == "post_issue_comment" {
					toolOutcome = c.Content
				}
			}
		}

		if toolOutcome != "" {
			return respondText(req, emit, "Acknowledged the comment ("+toolOutcome+").")
		}
		handle := senderHandle(lastText)
		reply := fmt.Sprintf("Thanks @%s — triaging this now; next step: reproduce against the latest release and label severity.", handle)
		args, err := json.Marshal(map[string]string{"body": reply})
		if err != nil {
			return nil, fmt.Errorf("marshal tool args: %w", err)
		}
		tc := llm.ToolCallContent{CallID: "call-reply-1", ToolName: "post_issue_comment", Args: args}
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
	})
}

// senderHandle pulls the commenter's handle out of the rendered signal text.
// Signals reach the model as their JSON payload (signal.go), so the sender
// attributes decode directly.
func senderHandle(rendered string) string {
	start, end := strings.Index(rendered, "{"), strings.LastIndex(rendered, "}")
	if start < 0 || end <= start {
		return "there"
	}
	var sig struct {
		Sender map[string]string `json:"sender"`
	}
	if err := json.Unmarshal([]byte(rendered[start:end+1]), &sig); err != nil || sig.Sender["handle"] == "" {
		return "there"
	}
	return sig.Sender["handle"]
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

func ingressMode() string {
	if os.Getenv("GITHUB_WEBHOOK_SECRET") != "" {
		return "verified (X-Hub-Signature-256)"
	}
	return "unverified demo mode (set GITHUB_WEBHOOK_SECRET)"
}

func replyMode() string {
	if os.Getenv("GITHUB_TOKEN") != "" {
		return "live GitHub API"
	}
	return "dry-run log (set GITHUB_TOKEN)"
}

func providerMode() string {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return "gemini (env-gated)"
	}
	return "local triage (keyless default)"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
