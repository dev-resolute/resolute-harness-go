package harness_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// A deterministic provider-fatal error (e.g. a Gemini 4xx classified by
// LLM-11) settles failed/run_failed on the FIRST attempt — no backoff ladder,
// no attempt-budget burn (HARNESS-12).
func TestProviderFatalErrorSettlesFirstAttempt(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().Error(fmt.Errorf("%w: Error 400, Message: Function call is missing a thought_signature, Status: INVALID_ARGUMENT", llm.ErrProviderFatal)).Add()

	rt, store := recoveryRuntime(t, provider, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("doomed"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrRunFailed {
		t.Fatalf("settled = %+v, want failed/run_failed", settled)
	}
	if !strings.Contains(settled.Error, "Error 400") {
		t.Errorf("settled.Error = %q, want it to name the underlying 400", settled.Error)
	}
	sub, err := store.GetSubmission(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if sub.AttemptCount != 1 {
		t.Errorf("attempt count = %d, want 1 (fatal errors must not retry)", sub.AttemptCount)
	}
	if provider.Called() != 1 {
		t.Errorf("provider called %d times, want 1", provider.Called())
	}
}

// When the attempt budget is exhausted by repeated transient failures, the
// settled record must name the last underlying error, not just the budget
// arithmetic (HARNESS-12: settled-error DX half).
func TestAttemptBudgetSettlementCarriesLastError(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().Error(fmt.Errorf("upstream 503: first flake")).Add()
	provider.OnAny().Error(fmt.Errorf("upstream 503: second flake")).Add()

	rt, store := recoveryRuntime(t, provider, 2)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	res, err := rt.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("doomed"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	settled, err := rt.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrAttemptBudget {
		t.Fatalf("settled = %+v, want failed/attempt_budget_exhausted", settled)
	}
	if !strings.Contains(settled.Error, "second flake") {
		t.Errorf("settled.Error = %q, want it to carry the last underlying error", settled.Error)
	}
	sub, err := store.GetSubmission(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if sub.LastError == "" || !strings.Contains(sub.LastError, "second flake") {
		t.Errorf("submission LastError = %q, want the last run error", sub.LastError)
	}
}
