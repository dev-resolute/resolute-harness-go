package harness_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "github.com/dev-resolute/resolute-llm-go"
	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
	"github.com/dev-resolute/resolute-harness-go/sqlite"
)

// engineConfig builds a Runtime config with tightened engine timings so
// crash and lease tests need no long sleeps. cfgFn may adjust the agent
// runtime config (budgets etc.).
func engineConfig(provider *mock.MockProvider, store harness.Store, cfgFn func(*harness.AgentRuntimeConfig)) harness.Config {
	return harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					cfg := harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
						SystemPrompt:  "You are a support agent.",
					}
					if cfgFn != nil {
						cfgFn(&cfg)
					}
					return cfg, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
	}
}

func startRuntime(t *testing.T, cfg harness.Config) *harness.Runtime {
	t.Helper()
	rt, err := harness.NewRuntime(cfg)
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	if err := rt.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return rt
}

// seededSubmission seeds a conversation plus an admitted submission directly
// through the public store API — the crash-state seeding mechanism.
func seededSubmission(t *testing.T, store harness.Store, body string) (harness.Conversation, harness.Submission) {
	t.Helper()
	ctx := context.Background()
	key := harness.SessionKey{Agent: "support", Instance: "acme", Session: "default"}
	conv, _, err := store.EnsureConversation(ctx, harness.Conversation{
		ID:        "01A0SEEDCONV00000000000000",
		Key:       key,
		CreatedAt: time.Now(),
	})
	if err != nil {
		t.Fatalf("EnsureConversation: %v", err)
	}
	sub, err := store.AdmitSubmission(ctx, harness.Submission{
		ID:             "01A0SEEDSUB000000000000000",
		SessionKey:     key,
		ConversationID: conv.ID,
		Status:         harness.StatusQueued,
		Input:          harness.UserMessage(body),
		CreatedAt:      time.Now(),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission: %v", err)
	}
	return conv, sub
}

// claimSeeded moves a seeded submission to running as a dead owner with an
// already-expired lease, simulating a crash after the claim.
func claimSeeded(t *testing.T, store harness.Store, sub harness.Submission) harness.Submission {
	t.Helper()
	claimed, err := store.ClaimSubmission(context.Background(), harness.SubmissionClaim{
		SubmissionID:   sub.ID,
		AttemptID:      "01A0DEADATTEMPT00000000000",
		OwnerID:        "dead-owner",
		LeaseExpiresAt: time.Now().Add(-time.Minute),
	})
	if err != nil {
		t.Fatalf("ClaimSubmission: %v", err)
	}
	return claimed
}

func settledRecords(t *testing.T, rt *harness.Runtime, conversationID, submissionID string) []harness.Record {
	t.Helper()
	recs, err := rt.Records(context.Background(), conversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	var out []harness.Record
	for _, rec := range recs {
		if rec.Kind == harness.KindSubmissionSettled && rec.SubmissionID == submissionID {
			out = append(out, rec)
		}
	}
	return out
}

func countKind(recs []harness.Record, kind harness.RecordKind) int {
	n := 0
	for _, rec := range recs {
		if rec.Kind == kind {
			n++
		}
	}
	return n
}

// Crash boundary: post-claim, pre-marker. The dead owner's lease is expired;
// a fresh Runtime over the same store must release, re-claim, and settle
// with correct attempt accounting.
func TestCrashPostClaimPreMarkerResumes(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	provider.OnAny().RespondText("recovered answer").Add()
	_, sub := seededSubmission(t, store, "hello after crash")
	claimSeeded(t, store, sub)

	rt := startRuntime(t, engineConfig(provider, store, nil))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	settled, err := rt.Wait(ctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	got, err := store.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.AttemptCount != 2 {
		t.Fatalf("attempt count = %d, want 2 (dead attempt + recovery attempt)", got.AttemptCount)
	}
	if n := len(settledRecords(t, rt, sub.ConversationID, sub.ID)); n != 1 {
		t.Fatalf("settled records = %d, want exactly 1", n)
	}
}

// Crash boundary: mid-turn. The dead attempt already authored the input
// record; the recovery attempt must not duplicate it.
func TestCrashMidTurnResumes(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	provider.OnAny().RespondText("recovered answer").Add()
	conv, sub := seededSubmission(t, store, "hello after crash")
	claimed := claimSeeded(t, store, sub)
	ctx := context.Background()
	if err := store.StartAttempt(ctx, harness.Attempt{
		ID: claimed.AttemptID, SubmissionID: sub.ID, OwnerID: "dead-owner", StartedAt: time.Now(),
	}); err != nil {
		t.Fatalf("StartAttempt: %v", err)
	}
	inputRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0SEEDREC000000000000001",
			Kind:           harness.KindUserMessage,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub.ID,
			AttemptID:      claimed.AttemptID,
			Time:           time.Now(),
		},
	}
	inputRec.Payload, _ = json.Marshal(map[string]string{"body": "hello after crash"})
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{inputRec}); err != nil {
		t.Fatalf("AppendRecords: %v", err)
	}

	rt := startRuntime(t, engineConfig(provider, store, nil))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded", settled)
	}

	recs, err := rt.Records(ctx, conv.ID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if n := countKind(recs, harness.KindUserMessage); n != 1 {
		t.Fatalf("user_message records = %d, want 1 (input dedupe across attempts)", n)
	}
	// The recovery attempt authored the assistant records under its own id.
	for _, rec := range recs {
		if rec.Kind == harness.KindAssistantMessageCompleted && rec.AttemptID == claimed.AttemptID {
			t.Fatalf("assistant record carries the dead attempt id %s", rec.AttemptID)
		}
	}
}

// Crash boundary: post-reserve, pre-finalize, with the terminal record
// already durable. Reconciliation must finalize without re-running the agent.
func TestCrashPostReservePreFinalizeFinalizes(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock") // no scripted steps: any model call would fail the run
	conv, sub := seededSubmission(t, store, "hello")
	claimed := claimSeeded(t, store, sub)
	ctx := context.Background()
	settledRec := harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             "01A0SEEDREC000000000000002",
			Kind:           harness.KindSubmissionSettled,
			ConversationID: conv.ID,
			Session:        "default",
			SubmissionID:   sub.ID,
			AttemptID:      claimed.AttemptID,
			Time:           time.Now(),
		},
	}
	settledRec.Payload, _ = json.Marshal(harness.SettledPayload{Status: harness.SettledSucceeded})
	if err := store.AppendRecords(ctx, conv.ID, []harness.Record{settledRec}); err != nil {
		t.Fatalf("AppendRecords: %v", err)
	}
	if err := store.ReserveSettlement(ctx, sub.ID, claimed.AttemptID); err != nil {
		t.Fatalf("ReserveSettlement: %v", err)
	}

	rt := startRuntime(t, engineConfig(provider, store, nil))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want the seeded succeeded outcome", settled)
	}
	if n := len(settledRecords(t, rt, conv.ID, sub.ID)); n != 1 {
		t.Fatalf("settled records = %d, want exactly the seeded one", n)
	}
	if provider.Called() != 0 {
		t.Fatalf("provider called %d times during finalize-only recovery, want 0", provider.Called())
	}
}

// Crash boundary: post-reserve with no terminal record. The outcome is
// unknowable; reconciliation settles failed with the indeterminate code.
func TestCrashTerminalizingWithoutRecordSettlesIndeterminate(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	_, sub := seededSubmission(t, store, "hello")
	claimed := claimSeeded(t, store, sub)
	if err := store.ReserveSettlement(context.Background(), sub.ID, claimed.AttemptID); err != nil {
		t.Fatalf("ReserveSettlement: %v", err)
	}

	rt := startRuntime(t, engineConfig(provider, store, nil))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(ctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrIndeterminate {
		t.Fatalf("settled = %+v, want failed/settlement_indeterminate", settled)
	}
}

// Attempt budget: recomputed from durable history, enforced across restarts.
func TestAttemptBudgetExhaustionSettlesFailed(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	_, sub := seededSubmission(t, store, "hello")

	// Burn two attempts through the public store API.
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		claimed, err := store.ClaimSubmission(ctx, harness.SubmissionClaim{
			SubmissionID:   sub.ID,
			AttemptID:      fmt.Sprintf("seed-attempt-%d", i),
			OwnerID:        "dead-owner",
			LeaseExpiresAt: time.Now().Add(time.Minute),
		})
		if err != nil {
			t.Fatalf("ClaimSubmission %d: %v", i, err)
		}
		if err := store.ReleaseSubmission(ctx, sub.ID, claimed.AttemptID); err != nil {
			t.Fatalf("ReleaseSubmission %d: %v", i, err)
		}
	}

	rt := startRuntime(t, engineConfig(provider, store, func(cfg *harness.AgentRuntimeConfig) {
		cfg.MaxAttempts = 2
	}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrAttemptBudget {
		t.Fatalf("settled = %+v, want failed/attempt_budget_exhausted", settled)
	}
	if provider.Called() != 0 {
		t.Fatalf("provider called %d times past the attempt budget, want 0", provider.Called())
	}

	// A restart must not retry a settled submission.
	rt2 := startRuntime(t, engineConfig(provider, store, func(cfg *harness.AgentRuntimeConfig) {
		cfg.MaxAttempts = 2
	}))
	time.Sleep(100 * time.Millisecond) // give the claim loop a few ticks
	if provider.Called() != 0 {
		t.Fatalf("provider called after restart of an exhausted submission")
	}
	got, err := store.GetSubmission(ctx, sub.ID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.Status != harness.StatusSettled {
		t.Fatalf("status after restart = %s, want settled", got.Status)
	}
	_ = rt2
}

// Timeout budget: a submission older than its durability timeout settles
// failed without running.
func TestTimeoutBudgetSettlesFailed(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	ctx := context.Background()
	key := harness.SessionKey{Agent: "support", Instance: "acme", Session: "default"}
	conv, _, err := store.EnsureConversation(ctx, harness.Conversation{ID: "01A0TIMEOUTCONV00000000000", Key: key, CreatedAt: time.Now()})
	if err != nil {
		t.Fatalf("EnsureConversation: %v", err)
	}
	sub, err := store.AdmitSubmission(ctx, harness.Submission{
		ID:             "01A0TIMEOUTSUB000000000000",
		SessionKey:     key,
		ConversationID: conv.ID,
		Status:         harness.StatusQueued,
		Input:          harness.UserMessage("too old"),
		CreatedAt:      time.Now().Add(-2 * time.Hour),
	})
	if err != nil {
		t.Fatalf("AdmitSubmission: %v", err)
	}

	rt := startRuntime(t, engineConfig(provider, store, func(cfg *harness.AgentRuntimeConfig) {
		cfg.SubmissionTimeout = time.Hour
	}))
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(wctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledFailed || settled.ErrorCode != harness.SettledErrTimeout {
		t.Fatalf("settled = %+v, want failed/timeout_exceeded", settled)
	}
	if provider.Called() != 0 {
		t.Fatalf("provider called %d times past the timeout, want 0", provider.Called())
	}
}

// A wedged owner's expired lease is reclaimed by the running coordinator's
// periodic scan (not just boot reconciliation).
func TestLeaseExpiryReclaimedWhileRunning(t *testing.T) {
	t.Parallel()
	store := memory.New()
	provider := mock.New("mock")
	provider.OnAny().RespondText("reclaimed answer").Add()

	rt := startRuntime(t, engineConfig(provider, store, nil))

	// Seed the wedged state after the Runtime is already running, so boot
	// reconciliation cannot have handled it.
	_, sub := seededSubmission(t, store, "wedged work")
	claimSeeded(t, store, sub)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	settled, err := rt.Wait(ctx, sub.ID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded after lease reclaim", settled)
	}
}

// Idempotent admission surfaced over HTTP: a replayed dispatch id returns
// the original 202 body; a mutated payload is a 409.
func TestHTTPIdempotentReplayAndConflict(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("answer").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	body := `{"kind":"user","body":"hello","dispatchId":"delivery-7"}`
	var first, replay harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", body, http.StatusAccepted, &first)
	postDispatch(t, server, "/agents/support/acme", body, http.StatusAccepted, &replay)
	if first != replay {
		t.Fatalf("replayed 202 body = %+v, want original %+v", replay, first)
	}
	mutated := `{"kind":"user","body":"DIFFERENT","dispatchId":"delivery-7"}`
	postDispatch(t, server, "/agents/support/acme", mutated, http.StatusConflict, nil)
}

// routingProvider fans requests out to per-session MockProviders keyed by a
// substring of the last user message. MockProvider consumes its script in
// strict call order and is not safe for concurrent streams, so concurrent
// sessions each get their own serial mock behind one provider name.
type routingProvider struct {
	name   string
	routes map[string]*mock.MockProvider // last-user-text substring → mock
}

func (p *routingProvider) Name() string { return p.name }

func (p *routingProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{}
}

func (p *routingProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	var lastUser string
	for _, m := range req.Messages {
		if m.Role != "user" {
			continue
		}
		if tc, ok := m.Content.(llm.TextContent); ok {
			lastUser = tc.Text
		}
	}
	for substr, inner := range p.routes {
		if strings.Contains(lastUser, substr) {
			return inner.Stream(ctx, req)
		}
	}
	panic(fmt.Sprintf("routingProvider: no route for user text %q", lastUser))
}

// Two sessions of one instance run concurrently; two submissions to one
// session run strictly in admission order.
func TestSessionSerializationAndCrossSessionConcurrency(t *testing.T) {
	t.Parallel()
	sessionA := mock.New("mock")
	sessionA.OnPrompt(mock.LastUser("slow work")).Delay(600 * time.Millisecond).RespondText("slow done").Add()
	sessionA.OnPrompt(mock.LastUser("queued behind slow")).RespondText("second done").Add()
	sessionB := mock.New("mock")
	sessionB.OnPrompt(mock.LastUser("fast work")).RespondText("fast done").Add()
	provider := &routingProvider{name: "mock", routes: map[string]*mock.MockProvider{
		"slow work": sessionA, "queued behind slow": sessionA, "fast work": sessionB,
	}}

	store := memory.New()
	rt := startRuntime(t, harness.Config{
		Agents: map[string]harness.AgentDefinition{
			"support": {
				Initialize: func(ctx context.Context, id harness.InstanceID, env harness.Env) (harness.AgentRuntimeConfig, error) {
					return harness.AgentRuntimeConfig{
						Model:         "mock/test-model",
						ContextWindow: 200_000,
						Providers:     []llm.LLMProvider{provider},
					}, nil
				},
			},
		},
		Store:         store,
		ClaimInterval: 20 * time.Millisecond,
		LeaseDuration: 300 * time.Millisecond,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	dispatch := func(session, body string) harness.DispatchResult {
		res, err := rt.Dispatch(ctx, harness.Dispatch{
			Agent: "support", Instance: "acme", Session: session,
			Message: harness.UserMessage(body),
		})
		if err != nil {
			t.Fatalf("Dispatch(%s): %v", body, err)
		}
		return res
	}

	slowRes := dispatch("session-a", "slow work")
	queuedRes := dispatch("session-a", "queued behind slow")
	fastRes := dispatch("session-b", "fast work")

	// Cross-session concurrency: session B settles while session A's slow
	// head is still in flight.
	if _, err := rt.Wait(ctx, fastRes.SubmissionID); err != nil {
		t.Fatalf("Wait(fast): %v", err)
	}
	slowSub, err := store.GetSubmission(ctx, slowRes.SubmissionID)
	if err != nil {
		t.Fatalf("GetSubmission(slow): %v", err)
	}
	if slowSub.Status == harness.StatusSettled {
		t.Fatal("slow submission already settled when fast finished — sessions did not run concurrently")
	}

	// Per-session serialization: the second submission's records all land
	// after the first submission settles.
	if _, err := rt.Wait(ctx, queuedRes.SubmissionID); err != nil {
		t.Fatalf("Wait(queued): %v", err)
	}
	recs, err := rt.Records(ctx, slowRes.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	slowSettledID := ""
	for _, rec := range recs {
		if rec.Kind == harness.KindSubmissionSettled && rec.SubmissionID == slowRes.SubmissionID {
			slowSettledID = rec.ID
		}
	}
	if slowSettledID == "" {
		t.Fatal("slow submission has no settled record")
	}
	for _, rec := range recs {
		if rec.SubmissionID == queuedRes.SubmissionID && rec.ID <= slowSettledID {
			t.Fatalf("record %s (%s) of the queued submission precedes the slow settle %s", rec.ID, rec.Kind, slowSettledID)
		}
	}
}

// Restart over the same SQLite store: stop the Runtime mid-run, construct a
// fresh one, and watch the submission settle with a record history spanning
// both attempts.
func TestRestartOverSameSQLiteStoreResumes(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "harness.db")
	store, err := sqlite.Open(path)
	if err != nil {
		t.Fatalf("sqlite.Open: %v", err)
	}
	t.Cleanup(func() { store.Close() })

	provider := mock.New("mock")
	// Attempt 1 stalls long enough for Close to interrupt it; attempt 2
	// answers immediately.
	provider.OnAny().Delay(5 * time.Second).RespondText("never delivered").Add()
	provider.OnAny().RespondText("finished after restart").Add()

	rt1, err := harness.NewRuntime(engineConfig(provider, store, nil))
	if err != nil {
		t.Fatalf("NewRuntime 1: %v", err)
	}
	if err := rt1.Start(context.Background()); err != nil {
		t.Fatalf("Start 1: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	res, err := rt1.Dispatch(ctx, harness.Dispatch{
		Agent: "support", Instance: "acme", Message: harness.UserMessage("survive a restart"),
	})
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	// Wait until the first attempt has durably authored the input record
	// (so the history provably spans both attempts), then stop the Runtime
	// mid-run — the model call is stalled on the scripted delay.
	waitForRecordKind(t, store, res.ConversationID, harness.KindUserMessage)
	if err := rt1.Close(); err != nil {
		t.Fatalf("Close 1: %v", err)
	}

	rt2 := startRuntime(t, engineConfig(provider, store, nil))
	settled, err := rt2.Wait(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if settled.Status != harness.SettledSucceeded {
		t.Fatalf("settled = %+v, want succeeded after restart", settled)
	}

	recs, err := rt2.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if n := countKind(recs, harness.KindUserMessage); n != 1 {
		t.Fatalf("user_message records = %d, want 1", n)
	}
	if n := countKind(recs, harness.KindSubmissionSettled); n != 1 {
		t.Fatalf("settled records = %d, want 1", n)
	}
	attempts := map[string]bool{}
	for _, rec := range recs {
		if rec.AttemptID != "" {
			attempts[rec.AttemptID] = true
		}
	}
	if len(attempts) < 2 {
		t.Fatalf("record history names %d attempts, want ≥2 (spanning the restart)", len(attempts))
	}
	got, err := store.GetSubmission(ctx, res.SubmissionID)
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.AttemptCount < 2 {
		t.Fatalf("attempt count = %d, want ≥2", got.AttemptCount)
	}
	// The final answer came from the second script step.
	sawFinal := false
	for _, rec := range recs {
		if rec.Kind != harness.KindAssistantMessageCompleted {
			continue
		}
		var p harness.AssistantMessageCompletedPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("decode assistant message: %v", err)
		}
		if strings.Contains(string(p.Message.Body), "finished after restart") {
			sawFinal = true
		}
	}
	if !sawFinal {
		t.Fatal("final assistant message from the restarted attempt not found in the stream")
	}
}

// waitForRecordKind polls until a record of the given kind is durable in the
// conversation.
func waitForRecordKind(t *testing.T, store harness.Store, conversationID string, kind harness.RecordKind) {
	t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		recs, err := store.ReadRecords(context.Background(), conversationID, "")
		if err == nil && countKind(recs, kind) > 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("conversation %s never grew a %s record", conversationID, kind)
		case <-time.After(10 * time.Millisecond):
		}
	}
}
