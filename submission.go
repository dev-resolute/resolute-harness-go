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
// queued → running → waiting → running → terminalizing → settled.
type SubmissionStatus string

// Submission lifecycle states.
const (
	StatusQueued        SubmissionStatus = "queued"
	StatusRunning       SubmissionStatus = "running"
	StatusWaiting       SubmissionStatus = "waiting" // suspended on child submissions; no lease held
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
	// LastError is the most recent run error recorded when an attempt was
	// released for retry. It survives re-claims so a budget-exhaustion
	// settlement can name the underlying failure (HARNESS-12).
	LastError string    `json:"lastError,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
	// ParentSubmissionID/ParentCallID link a child submission to the task
	// call that spawned it (HARNESS-15); empty for root dispatches.
	ParentSubmissionID string `json:"parentSubmissionId,omitempty"`
	ParentCallID       string `json:"parentCallId,omitempty"`
	// Depth is the spawn depth (0 for root dispatches; child = parent+1).
	Depth int `json:"depth,omitempty"`
	// PendingResume marks a submission parked in waiting whose next drive
	// must Resume (not Prompt) — set by WaitSubmission, kept by
	// ResumeSubmission, consumed by the claim that re-drives it (the claimed
	// row carries the flag; the stored row clears it).
	PendingResume bool `json:"pendingResume,omitempty"`
	// WaitUntil bounds the wait when SubagentLimits.MaxWait is set; zero = unbounded.
	WaitUntil time.Time `json:"waitUntil,omitzero"`
	// CancelRequested asks the owning coordinator to cancel the attempt at
	// the next turn boundary (orphan cascade).
	CancelRequested bool `json:"cancelRequested,omitempty"`
}
