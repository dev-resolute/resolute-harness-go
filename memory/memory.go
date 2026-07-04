// Package memory provides the in-memory Store used by tests and embedded
// runs. It implements the full store contract and passes the same
// conformance suite as the SQLite backend.
package memory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"sync"
	"time"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// Store is the in-memory implementation of harness.Store.
type Store struct {
	mu sync.Mutex

	convByKey map[string]harness.Conversation
	records   map[string][]harness.Record // conversation id → log

	subs     map[string]harness.Submission
	subOrder []string                     // admission order of submission ids
	attempts map[string][]harness.Attempt // submission id → markers in start order

	attachments map[string]harness.Attachment // digest → blob
}

var _ harness.Store = (*Store)(nil)

// New returns an empty in-memory store.
func New() *Store {
	return &Store{
		convByKey:   make(map[string]harness.Conversation),
		records:     make(map[string][]harness.Record),
		subs:        make(map[string]harness.Submission),
		attempts:    make(map[string][]harness.Attempt),
		attachments: make(map[string]harness.Attachment),
	}
}

// EnsureConversation implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) EnsureConversation(ctx context.Context, candidate harness.Conversation) (harness.Conversation, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := candidate.Key.String()
	if existing, ok := s.convByKey[key]; ok {
		return existing, false, nil
	}
	s.convByKey[key] = candidate
	return candidate, true, nil
}

// GetConversation implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetConversation(ctx context.Context, key harness.SessionKey) (harness.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	conv, ok := s.convByKey[key.String()]
	if !ok {
		return harness.Conversation{}, harness.ErrConversationNotFound
	}
	return conv, nil
}

// AppendRecords implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) AppendRecords(ctx context.Context, conversationID string, recs []harness.Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records[conversationID] = append(s.records[conversationID], recs...)
	return nil
}

// ReadRecords implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReadRecords(ctx context.Context, conversationID string, afterID string) ([]harness.Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	log := s.records[conversationID]
	// Record IDs are ULIDs in append order; binary-search the offset.
	start := 0
	if afterID != "" {
		start = sort.Search(len(log), func(i int) bool { return log[i].ID > afterID })
	}
	out := make([]harness.Record, len(log)-start)
	copy(out, log[start:])
	return out, nil
}

// AdmitSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) AdmitSubmission(ctx context.Context, sub harness.Submission) (harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.subs[sub.ID]; ok {
		if !sameInput(existing.Input, sub.Input) {
			return harness.Submission{}, harness.ErrDispatchConflict
		}
		return existing, nil
	}
	s.subs[sub.ID] = sub
	s.subOrder = append(s.subOrder, sub.ID)
	return sub, nil
}

func sameInput(a, b harness.DispatchMessage) bool {
	aj, err := json.Marshal(a)
	if err != nil {
		return false
	}
	bj, err := json.Marshal(b)
	if err != nil {
		return false
	}
	return bytes.Equal(aj, bj)
}

// GetSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetSubmission(ctx context.Context, id string) (harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[id]
	if !ok {
		return harness.Submission{}, harness.ErrSubmissionNotFound
	}
	return sub, nil
}

// ListRunnable implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListRunnable(ctx context.Context) ([]harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Head-of-line per session: the oldest unsettled submission per session
	// key is the head; it is runnable only while queued.
	seen := make(map[string]bool)
	var out []harness.Submission
	for _, id := range s.subOrder {
		sub := s.subs[id]
		key := sub.SessionKey.String()
		if seen[key] {
			continue
		}
		if sub.Status == harness.StatusSettled {
			continue
		}
		seen[key] = true // this is the session head, runnable or busy
		if sub.Status == harness.StatusQueued {
			out = append(out, sub)
		}
	}
	return out, nil
}

// ClaimSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ClaimSubmission(ctx context.Context, claim harness.SubmissionClaim) (harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[claim.SubmissionID]
	if !ok {
		return harness.Submission{}, harness.ErrSubmissionNotFound
	}
	if sub.Status != harness.StatusQueued {
		return harness.Submission{}, harness.ErrClaimLost
	}
	sub.Status = harness.StatusRunning
	sub.AttemptID = claim.AttemptID
	sub.OwnerID = claim.OwnerID
	sub.LeaseExpiresAt = claim.LeaseExpiresAt
	sub.AttemptCount++
	s.subs[sub.ID] = sub
	return sub, nil
}

// ListByStatus implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListByStatus(ctx context.Context, status harness.SubmissionStatus) ([]harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []harness.Submission
	for _, id := range s.subOrder {
		if sub := s.subs[id]; sub.Status == status {
			out = append(out, sub)
		}
	}
	return out, nil
}

// StartAttempt implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) StartAttempt(ctx context.Context, attempt harness.Attempt) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.subs[attempt.SubmissionID]; !ok {
		return harness.ErrSubmissionNotFound
	}
	s.attempts[attempt.SubmissionID] = append(s.attempts[attempt.SubmissionID], attempt)
	return nil
}

// ListAttempts implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListAttempts(ctx context.Context, submissionID string) ([]harness.Attempt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]harness.Attempt, len(s.attempts[submissionID]))
	copy(out, s.attempts[submissionID])
	return out, nil
}

// RenewLease implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) RenewLease(ctx context.Context, renewal harness.LeaseRenewal) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[renewal.SubmissionID]
	if !ok {
		return harness.ErrSubmissionNotFound
	}
	if sub.Status != harness.StatusRunning || sub.AttemptID != renewal.AttemptID {
		return harness.ErrClaimLost
	}
	sub.LeaseExpiresAt = renewal.LeaseExpiresAt
	s.subs[sub.ID] = sub
	return nil
}

// ListExpiredLeases implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListExpiredLeases(ctx context.Context, now time.Time) ([]harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []harness.Submission
	for _, id := range s.subOrder {
		sub := s.subs[id]
		if sub.Status == harness.StatusRunning && !sub.LeaseExpiresAt.After(now) {
			out = append(out, sub)
		}
	}
	return out, nil
}

// ReleaseSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReleaseSubmission(ctx context.Context, release harness.SubmissionRelease) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[release.SubmissionID]
	if !ok {
		return harness.ErrSubmissionNotFound
	}
	if sub.Status != harness.StatusRunning || sub.AttemptID != release.AttemptID {
		return harness.ErrClaimLost
	}
	sub.Status = harness.StatusQueued
	sub.OwnerID = ""
	sub.AttemptID = ""
	sub.LeaseExpiresAt = time.Time{}
	if release.LastError != "" {
		sub.LastError = release.LastError
	}
	s.subs[sub.ID] = sub
	return nil
}

// ReserveSettlement implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReserveSettlement(ctx context.Context, submissionID, attemptID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[submissionID]
	if !ok {
		return harness.ErrSubmissionNotFound
	}
	if sub.Status != harness.StatusRunning || sub.AttemptID != attemptID {
		return harness.ErrClaimLost
	}
	sub.Status = harness.StatusTerminalizing
	s.subs[sub.ID] = sub
	return nil
}

// FinalizeSettlement implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) FinalizeSettlement(ctx context.Context, submissionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[submissionID]
	if !ok {
		return harness.ErrSubmissionNotFound
	}
	switch sub.Status {
	case harness.StatusSettled:
		return nil // idempotent: reconciliation may finalize again after a crash
	case harness.StatusTerminalizing:
		sub.Status = harness.StatusSettled
		s.subs[sub.ID] = sub
		return nil
	default:
		return harness.ErrClaimLost
	}
}

// PutAttachment implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) PutAttachment(ctx context.Context, mediaType string, data []byte) (harness.AttachmentRef, error) {
	sum := sha256.Sum256(data)
	ref := harness.AttachmentRef{
		Digest:    "sha256:" + hex.EncodeToString(sum[:]),
		MediaType: mediaType,
		Size:      int64(len(data)),
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.attachments[ref.Digest]; !ok {
		stored := make([]byte, len(data))
		copy(stored, data)
		s.attachments[ref.Digest] = harness.Attachment{Ref: ref, Data: stored}
	}
	return ref, nil
}

// GetAttachment implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetAttachment(ctx context.Context, digest string) (harness.Attachment, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	att, ok := s.attachments[digest]
	if !ok {
		return harness.Attachment{}, harness.ErrAttachmentNotFound
	}
	return att, nil
}
