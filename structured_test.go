package harness_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
)

const severitySchema = `{
	"type": "object",
	"properties": {
		"severity": {"type": "string", "enum": ["low", "medium", "high"]},
		"component": {"type": "string"}
	},
	"required": ["severity", "component"],
	"additionalProperties": false
}`

func TestStructuredResultConforming(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText(`{"severity":"high","component":"auth"}`).Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	body := `{"kind":"user","body":"triage this bug","resultSchema":` + severitySchema + `}`
	var res struct {
		harness.DispatchResult
		harness.SettledPayload
	}
	postDispatch(t, server, "/agents/support/acme?wait=true", body, http.StatusOK, &res)
	if res.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", res.SettledPayload)
	}
	var result struct {
		Severity  string `json:"severity"`
		Component string `json:"component"`
	}
	if err := json.Unmarshal(res.Result, &result); err != nil {
		t.Fatalf("unmarshal result %s: %v", res.Result, err)
	}
	if result.Severity != "high" || result.Component != "auth" {
		t.Fatalf("result = %+v, want severity high, component auth", result)
	}

	// The settled record itself carries the validated result.
	recs, err := rt.Records(context.Background(), res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	for _, rec := range recs {
		if rec.Kind != harness.KindSubmissionSettled {
			continue
		}
		var p harness.SettledPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("decode settled: %v", err)
		}
		if len(p.Result) == 0 {
			t.Fatal("settled record carries no result")
		}
	}
}

func TestStructuredResultRetryWithFeedback(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	// First answer is prose; the corrective turn must carry the validation
	// feedback, and the retry conforms.
	provider.OnAny().RespondText("I think it's a high severity auth bug.").Add()
	provider.OnPrompt(mock.Predicate(func(msgs []llm.Message) bool {
		// The corrective turn must reach the model with the schema feedback.
		var lastUser string
		for _, m := range msgs {
			if m.Role != "user" {
				continue
			}
			if tc, ok := m.Content.(llm.TextContent); ok {
				lastUser = tc.Text
			}
		}
		return strings.Contains(lastUser, "schema")
	})).RespondText(`{"severity":"high","component":"auth"}`).Add()

	rt := newTestRuntime(t, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message: harness.DispatchMessage{
			Kind:         harness.InboundUser,
			Body:         "triage this bug",
			ResultSchema: json.RawMessage(severitySchema),
		},
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded within the feedback budget", settled)
	}

	// The corrective turn is visible in the record stream: a second
	// user_message carrying schema feedback.
	recs, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	var userBodies []string
	for _, rec := range recs {
		if rec.Kind != harness.KindUserMessage {
			continue
		}
		var p harness.UserMessagePayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("decode user message: %v", err)
		}
		userBodies = append(userBodies, p.Body)
	}
	if len(userBodies) != 2 {
		t.Fatalf("user_message records = %d, want 2 (original + corrective)", len(userBodies))
	}
	if !strings.Contains(userBodies[1], "schema") {
		t.Fatalf("corrective turn body %q does not mention the schema", userBodies[1])
	}
}

func TestStructuredResultBudgetExhausted(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("still prose").Add()
	provider.OnAny().RespondText("more prose, sorry").Add()

	rt := newTestRuntime(t, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent:    "support",
		Instance: "acme",
		Message: harness.DispatchMessage{
			Kind:          harness.InboundUser,
			Body:          "triage this bug",
			ResultSchema:  json.RawMessage(severitySchema),
			ResultRetries: 1,
		},
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrResultInvalid {
		t.Fatalf("settled = %+v, want failed/result_schema_invalid", settled)
	}
	if !strings.Contains(settled.Error, "severity") && !strings.Contains(settled.Error, "schema") {
		t.Fatalf("error %q does not name the schema violation", settled.Error)
	}
	// Exactly one corrective turn: original + 1 retry consumed both steps.
	if provider.Called() != 2 {
		t.Fatalf("provider called %d times, want 2 (budget 1 retry)", provider.Called())
	}
}

func TestPromptWithoutSchemaUntouched(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("plain prose answer").Add()
	rt := newTestRuntime(t, provider)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("just chat"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded || len(settled.Result) != 0 {
		t.Fatalf("settled = %+v, want succeeded with no result field", settled)
	}
	if provider.Called() != 1 {
		t.Fatalf("provider called %d times, want 1 (no validation turns)", provider.Called())
	}
}
