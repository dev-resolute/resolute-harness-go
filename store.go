package harness

import (
	"context"
	"errors"
	"time"
)

// Store-level sentinel errors. Every implementation returns these (possibly
// wrapped) so callers can branch with errors.Is.
var (
	// ErrDispatchConflict reports a re-admission of an existing dispatch id
	// with a different payload. Identical replays are not an error — they
	// return the original submission.
	ErrDispatchConflict = errors.New("dispatch id already admitted with a different payload")
	// ErrSubmissionNotFound reports an unknown submission id.
	ErrSubmissionNotFound = errors.New("submission not found")
	// ErrConversationNotFound reports an unknown conversation.
	ErrConversationNotFound = errors.New("conversation not found")
	// ErrClaimLost reports a claim CAS that did not apply because the
	// submission was not claimable in the expected state.
	ErrClaimLost = errors.New("submission claim lost")
)

// Conversation is the stored identity of one session's conversation log.
type Conversation struct {
	ID        string     `json:"id"`
	Key       SessionKey `json:"key"`
	CreatedAt time.Time  `json:"createdAt"`
}

// SubmissionClaim carries the parameters of a claim CAS: the submission to
// move queued→running and the attempt taking ownership.
type SubmissionClaim struct {
	SubmissionID   string
	AttemptID      string
	OwnerID        string
	LeaseExpiresAt time.Time
}

// SubmissionStore is the durable submission half of the store contract.
// Implementations must make AdmitSubmission idempotent by submission ID and
// ClaimSubmission an atomic queued→running CAS.
type SubmissionStore interface {
	// AdmitSubmission durably admits sub. Re-admitting the same ID with an
	// identical Input returns the previously stored submission; a different
	// Input returns ErrDispatchConflict.
	AdmitSubmission(ctx context.Context, sub Submission) (Submission, error)
	// GetSubmission returns the submission by id, or ErrSubmissionNotFound.
	GetSubmission(ctx context.Context, id string) (Submission, error)
	// ListRunnable returns, per session key, the oldest unsettled submission
	// — and only when it is claimable (queued). A session whose head is
	// running is busy and contributes nothing.
	ListRunnable(ctx context.Context) ([]Submission, error)
	// ClaimSubmission atomically moves the submission queued→running,
	// recording the attempt id, owner, and lease expiry, and incrementing
	// AttemptCount. It returns ErrClaimLost when the submission is not
	// queued.
	ClaimSubmission(ctx context.Context, claim SubmissionClaim) (Submission, error)
	// SettleSubmission moves the submission to settled. The terminal outcome
	// itself lives on the submission_settled conversation record.
	SettleSubmission(ctx context.Context, id string) error
}

// ConversationStore is the conversation-log half of the store contract:
// an append-only record log per conversation plus the session-key mapping.
type ConversationStore interface {
	// EnsureConversation returns the conversation for key, creating it with
	// the supplied candidate (ID, CreatedAt) when absent. The bool reports
	// whether this call created it.
	EnsureConversation(ctx context.Context, candidate Conversation) (Conversation, bool, error)
	// GetConversation returns the conversation for key, or
	// ErrConversationNotFound.
	GetConversation(ctx context.Context, key SessionKey) (Conversation, error)
	// AppendRecords appends records to the conversation log in order.
	AppendRecords(ctx context.Context, conversationID string, recs []Record) error
	// ReadRecords returns records with IDs strictly greater than afterID in
	// append order; afterID "" reads from the start.
	ReadRecords(ctx context.Context, conversationID string, afterID string) ([]Record, error)
}

// Store is the single narrow persistence contract every backend implements
// (ADR-0006). The attachment half joins in the storage slice (HARNESS-2).
type Store interface {
	SubmissionStore
	ConversationStore
}
