package harness

import (
	"encoding/json"
	"errors"
	"fmt"
)

// InboundKind discriminates the two DispatchMessage kinds (ADR-0005).
type InboundKind string

// The two inbound kinds. User is a direct 1:1 exchange with the agent's
// principal; Signal is one participant's activity in a multi-party
// conversation the agent participates in.
const (
	InboundUser   InboundKind = "user"
	InboundSignal InboundKind = "signal"
)

// SignalMeta carries the signal-specific fields of a DispatchMessage:
// what kind of activity it was, who sent it, and an optional correlation tag.
type SignalMeta struct {
	Type   string            `json:"type"`
	Sender map[string]string `json:"sender,omitempty"`
	Tag    string            `json:"tag,omitempty"`
}

// DefaultResultRetries is the feedback-retry budget when a Prompt requests a
// structured result and leaves ResultRetries at 0.
const DefaultResultRetries = 2

// DispatchMessage is the inbound payload of a dispatch: a discriminated
// user-or-signal union, plus the optional structured-result request. It is
// stored durably on the submission, so re-attempts and idempotency
// comparisons see the schema too.
type DispatchMessage struct {
	Kind        InboundKind     `json:"kind"`
	Body        string          `json:"body"`
	Attachments []AttachmentRef `json:"attachments,omitempty"`
	Signal      *SignalMeta     `json:"signal,omitempty"`
	// ResultSchema, when set, is a JSON Schema the run's final answer must
	// validate against; the validated JSON rides the submission_settled
	// record.
	ResultSchema json.RawMessage `json:"resultSchema,omitempty"`
	// ResultRetries bounds the validate→feedback→retry loop; 0 means
	// DefaultResultRetries.
	ResultRetries int `json:"resultRetries,omitempty"`
}

// UserMessage builds a user-kind DispatchMessage.
func UserMessage(body string) DispatchMessage {
	return DispatchMessage{Kind: InboundUser, Body: body}
}

// SignalMessage builds a signal-kind DispatchMessage.
func SignalMessage(body string, meta SignalMeta) DispatchMessage {
	return DispatchMessage{Kind: InboundSignal, Body: body, Signal: &meta}
}

// ErrInvalidDispatch reports a dispatch rejected at admission; nothing
// entered the store.
var ErrInvalidDispatch = errors.New("invalid dispatch")

// Validate checks the structural rules of the inbound union. It is called at
// admission; a failing message never enters the store.
func (m DispatchMessage) Validate() error {
	switch m.Kind {
	case InboundUser:
		if m.Signal != nil {
			return fmt.Errorf("%w: user message must not carry signal fields", ErrInvalidDispatch)
		}
	case InboundSignal:
		if m.Signal == nil || m.Signal.Type == "" {
			return fmt.Errorf("%w: signal message requires a signal type", ErrInvalidDispatch)
		}
	default:
		return fmt.Errorf("%w: unknown inbound kind %q", ErrInvalidDispatch, m.Kind)
	}
	if m.Body == "" {
		return fmt.Errorf("%w: body is required", ErrInvalidDispatch)
	}
	return nil
}

// Dispatch is an inbound request to run work: admission, not execution.
// DispatchID is the idempotency key; when empty a fresh one is generated,
// which opts the caller out of idempotent replay.
type Dispatch struct {
	Agent      string
	Instance   InstanceID
	Session    string // empty means "default"
	DispatchID string
	Message    DispatchMessage
}

// DispatchResult is the admission receipt: the durable submission created
// (or replayed) for the dispatch, and the conversation it targets.
type DispatchResult struct {
	SubmissionID   string `json:"submissionId"`
	ConversationID string `json:"conversationId"`
}
