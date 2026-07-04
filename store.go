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
	// ErrClaimLost reports a state CAS (claim, release, reserve, finalize,
	// lease renewal) that did not apply because the submission was not in the
	// expected state or owned by the expected attempt.
	ErrClaimLost = errors.New("submission claim lost")
	// ErrAttachmentNotFound reports an unknown attachment digest.
	ErrAttachmentNotFound = errors.New("attachment not found")
	// ErrUnsupportedSchema reports a store opened over a persisted schema
	// version this build does not support.
	ErrUnsupportedSchema = errors.New("unsupported store schema version")
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

// Attempt is the durable marker of one execution try. The marker is written
// before any work happens, so reconciliation can distinguish "started then
// died" from "never started"; budgets are recomputed from the marker history.
type Attempt struct {
	ID           string    `json:"id"`
	SubmissionID string    `json:"submissionId"`
	OwnerID      string    `json:"ownerId"`
	StartedAt    time.Time `json:"startedAt"`
}

// LeaseRenewal carries the parameters of a heartbeat: the owning attempt and
// its new lease expiry.
type LeaseRenewal struct {
	SubmissionID   string
	AttemptID      string
	LeaseExpiresAt time.Time
}

// SubmissionRelease carries the parameters of a release CAS: the owning
// attempt giving the submission back to the queue, and optionally the run
// error that caused it. A non-empty LastError overwrites the submission's
// stored last error; empty preserves it (a shutdown or lease-reclaim release
// is not a model failure and must not erase the real one).
type SubmissionRelease struct {
	SubmissionID string
	AttemptID    string
	LastError    string
}

// SubmissionStore is the durable submission half of the store contract.
// Implementations must make AdmitSubmission idempotent by submission ID and
// every state transition an atomic CAS on the expected prior state.
type SubmissionStore interface {
	// AdmitSubmission durably admits sub. Re-admitting the same ID with an
	// identical Input returns the previously stored submission; a different
	// Input returns ErrDispatchConflict.
	AdmitSubmission(ctx context.Context, sub Submission) (Submission, error)
	// GetSubmission returns the submission by id, or ErrSubmissionNotFound.
	GetSubmission(ctx context.Context, id string) (Submission, error)
	// ListRunnable returns, per session key, the oldest unsettled submission
	// — and only when it is claimable (queued). A session whose head is
	// running, terminalizing, or reserved is busy and contributes nothing.
	ListRunnable(ctx context.Context) ([]Submission, error)
	// ListByStatus returns every submission in the given status, in
	// admission order. Reconciliation uses it to find interrupted work.
	ListByStatus(ctx context.Context, status SubmissionStatus) ([]Submission, error)
	// ClaimSubmission atomically moves the submission queued→running,
	// recording the attempt id, owner, and lease expiry, and incrementing
	// AttemptCount. It returns ErrClaimLost when the submission is not
	// queued.
	ClaimSubmission(ctx context.Context, claim SubmissionClaim) (Submission, error)
	// StartAttempt durably records the attempt marker. It is written after a
	// successful claim and before any work.
	StartAttempt(ctx context.Context, attempt Attempt) error
	// ListAttempts returns the attempt markers of a submission in start
	// order.
	ListAttempts(ctx context.Context, submissionID string) ([]Attempt, error)
	// RenewLease extends the lease of a running submission. It returns
	// ErrClaimLost when the submission is not running or is owned by a
	// different attempt.
	RenewLease(ctx context.Context, renewal LeaseRenewal) error
	// ListExpiredLeases returns running submissions whose lease expired at or
	// before now.
	ListExpiredLeases(ctx context.Context, now time.Time) ([]Submission, error)
	// ReleaseSubmission moves the submission running→queued so a fresh
	// attempt can claim it, recording release.LastError when non-empty (see
	// SubmissionRelease). It returns ErrClaimLost when the submission is
	// not running or is owned by a different attempt.
	ReleaseSubmission(ctx context.Context, release SubmissionRelease) error
	// ReserveSettlement atomically moves the submission
	// running→terminalizing — phase one of settlement. It returns
	// ErrClaimLost when the submission is not running or is owned by a
	// different attempt.
	ReserveSettlement(ctx context.Context, submissionID, attemptID string) error
	// FinalizeSettlement moves the submission terminalizing→settled — phase
	// two. It is idempotent: finalizing an already-settled submission is a
	// no-op, so a crash between the phases resolves cleanly on retry.
	FinalizeSettlement(ctx context.Context, submissionID string) error
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

// Attachment is one out-of-line blob plus its ref. Records carry only the
// ref; the bytes live in the AttachmentStore keyed by content digest.
type Attachment struct {
	Ref  AttachmentRef
	Data []byte
}

// AttachmentStore is the digest-keyed blob half of the store contract. It is
// in the schema from day one (ADR-0006) so vision later is a feature, not a
// migration; v1 ships no ingestion path.
type AttachmentStore interface {
	// PutAttachment stores data and returns its ref. The digest is
	// "sha256:<hex>" over the raw bytes; putting identical bytes twice is
	// idempotent and returns the same ref.
	PutAttachment(ctx context.Context, mediaType string, data []byte) (AttachmentRef, error)
	// GetAttachment returns the attachment by digest, or
	// ErrAttachmentNotFound.
	GetAttachment(ctx context.Context, digest string) (Attachment, error)
}

// Store is the single narrow persistence contract every backend implements
// (ADR-0006): one tier, no SQL-only extensions. Every implementation must
// pass the exported conformance suite in package storetest.
type Store interface {
	SubmissionStore
	ConversationStore
	AttachmentStore
}
