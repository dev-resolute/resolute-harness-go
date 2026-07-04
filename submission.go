package harness

import (
	"fmt"
	"time"
)

// SessionKey addresses one session of one agent instance:
// agent name / instance id / session name.
type SessionKey struct {
	Agent    string     `json:"agent"`
	Instance InstanceID `json:"instance"`
	Session  string     `json:"session"`
}

// String renders the key as "agent/instance/session".
func (k SessionKey) String() string {
	return fmt.Sprintf("%s/%s/%s", k.Agent, k.Instance, k.Session)
}

// SubmissionStatus is the durable lifecycle state of a submission:
// queued → running → terminalizing → settled.
type SubmissionStatus string

// Submission lifecycle states.
const (
	StatusQueued        SubmissionStatus = "queued"
	StatusRunning       SubmissionStatus = "running"
	StatusTerminalizing SubmissionStatus = "terminalizing"
	StatusSettled       SubmissionStatus = "settled"
)

// Submission is the durable record of one admitted dispatch — the unit of
// leasing, attempts, and settlement. Its ID is the dispatch id and therefore
// the idempotency key.
type Submission struct {
	ID             string           `json:"id"`
	SessionKey     SessionKey       `json:"sessionKey"`
	ConversationID string           `json:"conversationId"`
	Status         SubmissionStatus `json:"status"`
	Input          DispatchMessage  `json:"input"`
	AttemptCount   int              `json:"attemptCount"`
	AttemptID      string           `json:"attemptId,omitempty"`
	OwnerID        string           `json:"ownerId,omitempty"`
	LeaseExpiresAt time.Time        `json:"leaseExpiresAt,omitzero"`
	CreatedAt      time.Time        `json:"createdAt"`
}
