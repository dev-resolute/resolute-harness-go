package harness

import (
	"context"
	"fmt"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// Correlation carries the same ids as record envelopes, so observer output
// lines up with the durable record of what happened.
type Correlation struct {
	SessionKey     SessionKey
	ConversationID string
	SubmissionID   string
	AttemptID      string
	TurnID         string
}

// HarnessEvent is the sealed union of ephemeral engine events delivered to
// Observers (ADR-0008). Distinct from canonical records (durable) and from
// agent-core's per-prompt events (which the engine consumes).
type HarnessEvent interface{ isHarnessEvent() }

// SubmissionAdmittedEvent reports a durably admitted submission.
type SubmissionAdmittedEvent struct {
	Correlation
	Input DispatchMessage
}

// SubmissionClaimedEvent reports a claim CAS that took ownership.
type SubmissionClaimedEvent struct {
	Correlation
	OwnerID      string
	AttemptCount int
}

// AttemptStartedEvent reports the durable attempt marker landing.
type AttemptStartedEvent struct{ Correlation }

// SubmissionSettledEvent reports terminal settlement.
type SubmissionSettledEvent struct {
	Correlation
	Payload SettledPayload
}

// OperationStartedEvent and OperationEndedEvent bound one session operation
// ("prompt" or "compact").
type OperationStartedEvent struct {
	Correlation
	Operation string
}

// OperationEndedEvent closes an OperationStartedEvent; Err is "" on success.
type OperationEndedEvent struct {
	Correlation
	Operation string
	Err       string
}

// TurnStartedEvent and TurnEndedEvent bound one pi.Agent LLM round-trip.
type TurnStartedEvent struct {
	Correlation
	Turn int
}

// TurnEndedEvent closes a TurnStartedEvent.
type TurnEndedEvent struct {
	Correlation
	Turn int
}

// DeltaEvent reports one streamed fragment (unbatched — observers see what
// the model emitted, not the flush policy).
type DeltaEvent struct {
	Correlation
	Kind RecordKind // assistant_text_delta or assistant_thinking_delta
	Text string
}

// ToolCallStartedEvent and ToolCallEndedEvent bound one tool execution.
type ToolCallStartedEvent struct {
	Correlation
	CallID   string
	ToolName string
}

// ToolCallEndedEvent closes a ToolCallStartedEvent.
type ToolCallEndedEvent struct {
	Correlation
	CallID   string
	ToolName string
	IsError  bool
}

// CompactionEvent reports a compaction landing (manual or recovery-driven).
type CompactionEvent struct {
	Correlation
	Reason string // "manual" | "overflow"
}

// RecoveryEvent reports an engine recovery decision.
type RecoveryEvent struct {
	Correlation
	// Decision is "overflow_compact_retry", "transient_backoff", or one of
	// the "summarization_retry_*" lifecycle decisions (scheduled /
	// attempt_start / finished) relayed from agent-core's
	// OnSummarizationRetry hook.
	Decision string
	Detail   string
}

// summarizationRetryObserver converts agent-core's OnSummarizationRetry hook
// calls into RecoveryEvents. Compact has no event stream in agent-core, so
// this hook is how operators observe summarization retries (agent-core
// v0.7.0, upstream 0.81.1). The returned hook is safe for the concurrent
// calls split-turn summarization can make, as long as observe is.
func summarizationRetryObserver(observe func(HarnessEvent), corr Correlation) func(context.Context, pi.SummarizationRetryCtx) {
	return func(_ context.Context, c pi.SummarizationRetryCtx) {
		var decision, detail string
		switch c.Phase {
		case pi.SummarizationRetryScheduled:
			decision = "summarization_retry_scheduled"
			detail = fmt.Sprintf("attempt %d/%d in %s: %v", c.Attempt, c.MaxAttempts, c.Delay, c.Err)
		case pi.SummarizationRetryAttemptStart:
			decision = "summarization_retry_attempt_start"
			detail = fmt.Sprintf("attempt %d", c.Attempt)
		case pi.SummarizationRetryFinished:
			decision = "summarization_retry_finished"
			if c.Success {
				detail = fmt.Sprintf("succeeded on attempt %d", c.Attempt)
			} else if c.Err != nil {
				detail = c.Err.Error()
			}
		default:
			return
		}
		observe(RecoveryEvent{Correlation: corr, Decision: decision, Detail: detail})
	}
}

func (SubmissionAdmittedEvent) isHarnessEvent() {}
func (SubmissionClaimedEvent) isHarnessEvent()  {}
func (AttemptStartedEvent) isHarnessEvent()     {}
func (SubmissionSettledEvent) isHarnessEvent()  {}
func (OperationStartedEvent) isHarnessEvent()   {}
func (OperationEndedEvent) isHarnessEvent()     {}
func (TurnStartedEvent) isHarnessEvent()        {}
func (TurnEndedEvent) isHarnessEvent()          {}
func (DeltaEvent) isHarnessEvent()              {}
func (ToolCallStartedEvent) isHarnessEvent()    {}
func (ToolCallEndedEvent) isHarnessEvent()      {}
func (CompactionEvent) isHarnessEvent()         {}
func (RecoveryEvent) isHarnessEvent()           {}

// Observer receives HarnessEvents synchronously. Observers are read-only
// and cheap; panics are logged and never affect execution (ADR-0008).
type Observer func(HarnessEvent)

// OpKind names an interceptor boundary.
type OpKind string

// The four operation boundaries wrapped by interceptors.
const (
	OpAttempt   OpKind = "attempt"
	OpOperation OpKind = "operation"
	OpTurn      OpKind = "turn"
	OpTool      OpKind = "tool"
)

// OpInfo describes the boundary an Interceptor wraps, with the same
// correlation ids as record envelopes.
type OpInfo struct {
	Kind OpKind
	// Operation is "prompt" or "compact" at the OpOperation boundary.
	Operation string
	Correlation
	// ToolName and CallID are set at the OpTool boundary.
	ToolName string
	CallID   string
}

// Interceptor is onion middleware wrapped around every operation boundary —
// submission attempt, session operation, model turn, tool execution — so
// trace context propagates through context.Context natively. Interceptors
// compose in registration order (the first registered is outermost); an
// interceptor that returns without calling next aborts the operation and
// the error is accounted like any attempt failure. This pair of seams is
// the whole observability surface: the future OTel adapter is built on
// Observer + Interceptor with no engine changes (ADR-0008).
type Interceptor func(ctx context.Context, op OpInfo, next func(context.Context) error) error

// observe delivers ev to every observer, recovering panics.
func (rt *Runtime) observe(ev HarnessEvent) {
	for _, ob := range rt.observers {
		func() {
			defer func() {
				if r := recover(); r != nil {
					rt.logger.Error("observer panicked", "event", ev, "panic", r)
				}
			}()
			ob(ev)
		}()
	}
}

// intercept runs fn inside the composed interceptor chain for op.
func (rt *Runtime) intercept(ctx context.Context, op OpInfo, fn func(context.Context) error) error {
	next := fn
	for i := len(rt.interceptors) - 1; i >= 0; i-- {
		ic := rt.interceptors[i]
		inner := next
		next = func(c context.Context) error { return ic(c, op, inner) }
	}
	return next(ctx)
}
