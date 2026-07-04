package harness

import (
	"encoding/json"
	"fmt"
	"time"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// RecordKind identifies the type of a canonical record. The SSE wire format
// uses the kind as the event name.
type RecordKind string

// The full v1 record kind set. Later slices author the delta, signal, and
// compaction kinds; declaring them now pins the schema (ADR-0005/0006).
const (
	KindConversationCreated       RecordKind = "conversation_created"
	KindUserMessage               RecordKind = "user_message"
	KindSignal                    RecordKind = "signal"
	KindAssistantMessageStarted   RecordKind = "assistant_message_started"
	KindAssistantTextDelta        RecordKind = "assistant_text_delta"
	KindAssistantThinkingDelta    RecordKind = "assistant_thinking_delta"
	KindAssistantToolCall         RecordKind = "assistant_tool_call"
	KindToolOutcome               RecordKind = "tool_outcome"
	KindAssistantMessageCompleted RecordKind = "assistant_message_completed"
	KindCompaction                RecordKind = "compaction"
	KindSubmissionSettled         RecordKind = "submission_settled"
)

// RecordEnvelope is the correlation header every canonical record carries.
// The record ID is a ULID and doubles as the SSE stream offset.
type RecordEnvelope struct {
	ID             string     `json:"id"`
	Kind           RecordKind `json:"kind"`
	ConversationID string     `json:"conversationId"`
	Session        string     `json:"session"`
	SubmissionID   string     `json:"submissionId,omitempty"`
	TurnID         string     `json:"turnId,omitempty"`
	AttemptID      string     `json:"attemptId,omitempty"`
	Time           time.Time  `json:"time"`
}

// Record is one append-only entry in the durable conversation log. Payload
// holds the kind-specific body as opaque JSON; use the typed payload
// accessors to decode it. The JSON encoding of Record is the SSE wire format.
type Record struct {
	RecordEnvelope
	Payload json.RawMessage `json:"payload,omitempty"`
}

// AttachmentRef points at bytes stored out-of-line in the AttachmentStore.
// It is part of the record schema from day one (ADR-0006) even though v1 has
// no ingestion path.
type AttachmentRef struct {
	Digest    string `json:"digest"`
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
}

// MessagePayload is the harness wire form of an agent-core message. The
// record schema owns its JSON shape; conversion to and from pi.Message
// happens at the projection boundary.
type MessagePayload struct {
	Role string          `json:"role"`
	Type string          `json:"type"`
	Body json.RawMessage `json:"body,omitempty"`
}

func messageFromPi(m pi.Message) MessagePayload {
	return MessagePayload{Role: m.Role, Type: m.Type, Body: m.Body}
}

// ToPi converts the wire form back into an agent-core message.
func (m MessagePayload) ToPi() pi.Message {
	return pi.Message{Role: m.Role, Type: m.Type, Body: m.Body}
}

// ConversationCreatedPayload is the payload of a conversation_created record.
type ConversationCreatedPayload struct {
	Agent    string     `json:"agent"`
	Instance InstanceID `json:"instance"`
	Session  string     `json:"session"`
}

// UserMessagePayload is the payload of a user_message record.
type UserMessagePayload struct {
	Body        string          `json:"body"`
	Attachments []AttachmentRef `json:"attachments,omitempty"`
}

// AssistantToolCallPayload is the payload of an assistant_tool_call record.
type AssistantToolCallPayload struct {
	CallID   string          `json:"callId"`
	ToolName string          `json:"toolName"`
	Args     json.RawMessage `json:"args,omitempty"`
}

// ToolOutcomePayload is the payload of a tool_outcome record. It carries the
// full pi.ToolResult in wire form, correlated to its assistant_tool_call by
// CallID.
type ToolOutcomePayload struct {
	CallID   string          `json:"callId"`
	ToolName string          `json:"toolName"`
	Content  string          `json:"content,omitempty"`
	Data     json.RawMessage `json:"data,omitempty"`
	IsError  bool            `json:"isError,omitempty"`
}

// AssistantMessageCompletedPayload is the payload of an
// assistant_message_completed record: the final message as agent-core
// appended it to the transcript.
type AssistantMessageCompletedPayload struct {
	Message MessagePayload `json:"message"`
}

// SettledStatus is the terminal outcome carried on a submission_settled
// record.
type SettledStatus string

// Terminal outcomes of a submission.
const (
	SettledSucceeded SettledStatus = "succeeded"
	SettledFailed    SettledStatus = "failed"
)

// SettledErrorCode classifies why a submission settled as failed, stable for
// programmatic branching; Error carries the human-readable detail.
type SettledErrorCode string

// Failure classifications carried on submission_settled records.
const (
	// SettledErrRunFailed is a terminal error from the agent run itself.
	SettledErrRunFailed SettledErrorCode = "run_failed"
	// SettledErrAttemptBudget means the max-attempts durability budget was
	// exhausted.
	SettledErrAttemptBudget SettledErrorCode = "attempt_budget_exhausted"
	// SettledErrTimeout means the submission outlived its durability timeout.
	SettledErrTimeout SettledErrorCode = "timeout_exceeded"
	// SettledErrIndeterminate means a crash interrupted settlement before the
	// terminal record landed; the run's outcome is unknown.
	SettledErrIndeterminate SettledErrorCode = "settlement_indeterminate"
	// SettledErrResultInvalid means the structured result never validated
	// against the requested schema within the feedback budget.
	SettledErrResultInvalid SettledErrorCode = "result_schema_invalid"
)

// SettledPayload is the payload of a submission_settled record. Result is
// present only when the prompt requested a structured result.
type SettledPayload struct {
	Status    SettledStatus    `json:"status"`
	Error     string           `json:"error,omitempty"`
	ErrorCode SettledErrorCode `json:"errorCode,omitempty"`
	Result    json.RawMessage  `json:"result,omitempty"`
}

// DecodePayload unmarshals the record payload into dst, which must be a
// pointer to the payload type matching the record kind.
func (r Record) DecodePayload(dst interface{ payloadKind() RecordKind }) error {
	if want := dst.payloadKind(); r.Kind != want {
		return fmt.Errorf("record %s is kind %q, not %q", r.ID, r.Kind, want)
	}
	if err := json.Unmarshal(r.Payload, dst); err != nil {
		return fmt.Errorf("decode %s payload of record %s: %w", r.Kind, r.ID, err)
	}
	return nil
}

func (*ConversationCreatedPayload) payloadKind() RecordKind { return KindConversationCreated }
func (*UserMessagePayload) payloadKind() RecordKind         { return KindUserMessage }
func (*AssistantToolCallPayload) payloadKind() RecordKind   { return KindAssistantToolCall }
func (*ToolOutcomePayload) payloadKind() RecordKind         { return KindToolOutcome }
func (*AssistantMessageCompletedPayload) payloadKind() RecordKind {
	return KindAssistantMessageCompleted
}
func (*SettledPayload) payloadKind() RecordKind { return KindSubmissionSettled }

// mustPayload marshals a payload value, panicking on failure. Payload types
// are plain data structs; a marshal failure is a programmer error.
func mustPayload(v interface{ payloadKind() RecordKind }) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("marshal %s payload: %v", v.payloadKind(), err))
	}
	return b
}
