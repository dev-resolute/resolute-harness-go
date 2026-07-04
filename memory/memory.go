// Package memory provides the in-memory Store used by tests and embedded
// runs. It implements the full store contract and passes the same
// conformance suite as the SQLite backend.
package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"sort"
	"sync"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// Store is the in-memory implementation of harness.Store.
type Store struct {
	mu sync.Mutex

	convByKey map[string]harness.Conversation
	records   map[string][]harness.Record // conversation id → log

	subs     map[string]harness.Submission
	subOrder []string // admission order of submission ids
}

var _ harness.Store = (*Store)(nil)

// New returns an empty in-memory store.
func New() *Store {
	return &Store{
		convByKey: make(map[string]harness.Conversation),
		records:   make(map[string][]harness.Record),
		subs:      make(map[string]harness.Submission),
	}
}

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

func (s *Store) GetConversation(ctx context.Context, key harness.SessionKey) (harness.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	conv, ok := s.convByKey[key.String()]
	if !ok {
		return harness.Conversation{}, harness.ErrConversationNotFound
	}
	return conv, nil
}

func (s *Store) AppendRecords(ctx context.Context, conversationID string, recs []harness.Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records[conversationID] = append(s.records[conversationID], recs...)
	return nil
}

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

func (s *Store) GetSubmission(ctx context.Context, id string) (harness.Submission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[id]
	if !ok {
		return harness.Submission{}, harness.ErrSubmissionNotFound
	}
	return sub, nil
}

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

func (s *Store) SettleSubmission(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[id]
	if !ok {
		return harness.ErrSubmissionNotFound
	}
	sub.Status = harness.StatusSettled
	s.subs[id] = sub
	return nil
}
