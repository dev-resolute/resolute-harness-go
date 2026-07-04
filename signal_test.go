package harness_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
)

func TestSignalDispatchLandsSignalRecord(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// The rendered signal must reach the model with the sender identity
	// distinguishable from the principal.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			tc, ok := m.Content.(llm.TextContent)
			if !ok {
				continue
			}
			if strings.Contains(tc.Text, "github_issue_comment") && strings.Contains(tc.Text, "alice") {
				return true
			}
		}
		return false
	})).RespondText("Replying to alice's comment.").Add()

	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	body := `{"kind":"signal","body":"This breaks on v2.1","signal":{"type":"github_issue_comment","sender":{"handle":"alice","platform":"github"},"tag":"issue-42"},"dispatchId":"gh-comment-1"}`
	var res struct {
		harness.DispatchResult
		harness.SettledPayload
	}
	postDispatch(t, server, "/agents/support/acme?wait=true", body, http.StatusOK, &res)
	if res.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded (mock only matches when the rendered signal reached it)", res.SettledPayload)
	}

	// Idempotent replay, like user messages.
	var replay harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", body, http.StatusAccepted, &replay)
	if replay.SubmissionID != res.SubmissionID {
		t.Fatalf("replayed signal dispatch = %+v, want original submission %s", replay, res.SubmissionID)
	}

	recs, err := rt.Records(context.Background(), res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	var sig harness.SignalPayload
	found := false
	for _, rec := range recs {
		if rec.Kind == harness.KindSignal {
			if err := rec.DecodePayload(&sig); err != nil {
				t.Fatalf("decode signal payload: %v", err)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("no signal record in the stream")
	}
	if sig.Type != "github_issue_comment" || sig.Sender["handle"] != "alice" || sig.Tag != "issue-42" || sig.Body != "This breaks on v2.1" {
		t.Fatalf("signal payload = %+v, want type/sender/tag/body preserved", sig)
	}
}

func TestUserAndSignalInterleaveInOneConversation(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("noted the signal").Add()
	// The follow-up user prompt must still see the earlier signal in context.
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		for _, m := range msgs {
			if tc, ok := m.Content.(llm.TextContent); ok && strings.Contains(tc.Text, "slack_thread_reply") {
				return true
			}
		}
		return false
	})).RespondText("summarizing the thread").Add()

	rt := newTestRuntime(t, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sigRes, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message: harness.SignalMessage("what's the status here?", harness.SignalMeta{
			Type:   "slack_thread_reply",
			Sender: map[string]string{"handle": "bob"},
		}),
	})
	if err != nil {
		t.Fatalf("Dispatch signal: %v", err)
	}
	if _, err := rt.Wait(ctx, sigRes.SubmissionID); err != nil {
		t.Fatalf("Wait signal: %v", err)
	}

	userRes, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message:  harness.UserMessage("summarize the thread for me"),
	})
	if err != nil {
		t.Fatalf("Dispatch user: %v", err)
	}
	settled, err := rt.Wait(ctx, userRes.SubmissionID)
	if err != nil {
		t.Fatalf("Wait user: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("user prompt settled = %+v, want succeeded (projection must serve the signal)", settled)
	}
	if userRes.ConversationID != sigRes.ConversationID {
		t.Fatalf("user and signal landed in different conversations")
	}

	recs, err := rt.Records(ctx, sigRes.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	assertKindSubsequence(t, recs, []harness.RecordKind{
		harness.KindSignal,
		harness.KindAssistantMessageCompleted,
		harness.KindUserMessage,
		harness.KindAssistantMessageCompleted,
	})
}

func TestMalformedInboundRejectedAtAdmission(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	tests := []struct {
		name string
		body string
	}{
		{name: "unknown kind", body: `{"kind":"smoke-signal","body":"hi"}`},
		{name: "signal missing type", body: `{"kind":"signal","body":"hi","signal":{"sender":{"handle":"x"}}}`},
		{name: "signal missing signal object", body: `{"kind":"signal","body":"hi"}`},
		{name: "user with signal fields", body: `{"kind":"user","body":"hi","signal":{"type":"x"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			postDispatch(t, server, "/agents/support/acme", tt.body, http.StatusBadRequest, nil)
		})
	}

	// Nothing entered the store: the conversation was never created.
	resp, err := http.Get(server.URL + "/agents/support/acme")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("stream status = %d, want 404 (no conversation should exist)", resp.StatusCode)
	}
}
