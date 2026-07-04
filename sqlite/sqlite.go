// Package sqlite provides the batteries-included durable Store
// (modernc.org/sqlite — pure Go, no cgo, per ADR-0006). It passes the same
// exported conformance suite as the memory backend.
package sqlite

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite" // database/sql driver

	harness "github.com/dev-resolute/resolute-harness-go"
)

// schemaVersion is stamped into PRAGMA user_version on initialization and
// checked on every open. Opening a database from a newer build fails with
// harness.ErrUnsupportedSchema instead of corrupting it.
const schemaVersion = 1

const schema = `
CREATE TABLE IF NOT EXISTS conversations (
	key         TEXT PRIMARY KEY,
	id          TEXT NOT NULL UNIQUE,
	agent       TEXT NOT NULL,
	instance    TEXT NOT NULL,
	session     TEXT NOT NULL,
	created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
	seq             INTEGER PRIMARY KEY AUTOINCREMENT,
	id              TEXT NOT NULL UNIQUE,
	conversation_id TEXT NOT NULL,
	kind            TEXT NOT NULL,
	json            BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS records_by_conversation ON records(conversation_id, id);
CREATE TABLE IF NOT EXISTS submissions (
	seq              INTEGER PRIMARY KEY AUTOINCREMENT,
	id               TEXT NOT NULL UNIQUE,
	session_key      TEXT NOT NULL,
	agent            TEXT NOT NULL,
	instance         TEXT NOT NULL,
	session          TEXT NOT NULL,
	conversation_id  TEXT NOT NULL,
	status           TEXT NOT NULL,
	input_json       BLOB NOT NULL,
	attempt_count    INTEGER NOT NULL DEFAULT 0,
	attempt_id       TEXT NOT NULL DEFAULT '',
	owner_id         TEXT NOT NULL DEFAULT '',
	lease_expires_ns INTEGER NOT NULL DEFAULT 0,
	created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS submissions_by_session ON submissions(session_key, seq);
CREATE INDEX IF NOT EXISTS submissions_by_status ON submissions(status);
CREATE TABLE IF NOT EXISTS attempts (
	seq           INTEGER PRIMARY KEY AUTOINCREMENT,
	id            TEXT NOT NULL,
	submission_id TEXT NOT NULL,
	owner_id      TEXT NOT NULL,
	started_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_by_submission ON attempts(submission_id, seq);
CREATE TABLE IF NOT EXISTS attachments (
	digest     TEXT PRIMARY KEY,
	media_type TEXT NOT NULL,
	size       INTEGER NOT NULL,
	data       BLOB NOT NULL
);
`

// Store is the SQLite implementation of harness.Store.
type Store struct {
	db *sql.DB
}

var _ harness.Store = (*Store)(nil)

// Open opens (creating if needed) the database at path and verifies the
// schema version. When path is an existing directory, the database file is
// "harness.db" inside it.
func Open(path string) (*Store, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "harness.db")
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %s: %w", path, err)
	}
	// One writer connection keeps every statement serialized, which is all a
	// single-process v1 needs; WAL keeps readers cheap.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;"); err != nil {
		db.Close()
		return nil, fmt.Errorf("configure sqlite %s: %w", path, err)
	}

	var version int
	if err := db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		db.Close()
		return nil, fmt.Errorf("read schema version of %s: %w", path, err)
	}
	switch version {
	case 0:
		if _, err := db.Exec(schema); err != nil {
			db.Close()
			return nil, fmt.Errorf("initialize schema in %s: %w", path, err)
		}
		if _, err := db.Exec(fmt.Sprintf("PRAGMA user_version = %d", schemaVersion)); err != nil {
			db.Close()
			return nil, fmt.Errorf("stamp schema version in %s: %w", path, err)
		}
	case schemaVersion:
		// Supported.
	default:
		db.Close()
		return nil, fmt.Errorf("%s has schema version %d, this build supports %d: %w",
			path, version, schemaVersion, harness.ErrUnsupportedSchema)
	}
	return &Store{db: db}, nil
}

// Close releases the underlying database.
func (s *Store) Close() error {
	return s.db.Close()
}

// EnsureConversation implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) EnsureConversation(ctx context.Context, candidate harness.Conversation) (harness.Conversation, bool, error) {
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO conversations (key, id, agent, instance, session, created_at)
		VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO NOTHING`,
		candidate.Key.String(), candidate.ID, candidate.Key.Agent,
		string(candidate.Key.Instance), candidate.Key.Session,
		candidate.CreatedAt.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return harness.Conversation{}, false, fmt.Errorf("ensure conversation: %w", err)
	}
	inserted, err := res.RowsAffected()
	if err != nil {
		return harness.Conversation{}, false, fmt.Errorf("ensure conversation rows: %w", err)
	}
	if inserted == 1 {
		return candidate, true, nil
	}
	existing, err := s.GetConversation(ctx, candidate.Key)
	if err != nil {
		return harness.Conversation{}, false, err
	}
	return existing, false, nil
}

// GetConversation implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetConversation(ctx context.Context, key harness.SessionKey) (harness.Conversation, error) {
	var conv harness.Conversation
	var createdAt string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, agent, instance, session, created_at FROM conversations WHERE key = ?`,
		key.String()).Scan(&conv.ID, &conv.Key.Agent, &conv.Key.Instance, &conv.Key.Session, &createdAt)
	if errors.Is(err, sql.ErrNoRows) {
		return harness.Conversation{}, harness.ErrConversationNotFound
	}
	if err != nil {
		return harness.Conversation{}, fmt.Errorf("get conversation %s: %w", key, err)
	}
	conv.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return harness.Conversation{}, fmt.Errorf("parse conversation created_at: %w", err)
	}
	return conv, nil
}

// AppendRecords implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) AppendRecords(ctx context.Context, conversationID string, recs []harness.Record) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin append: %w", err)
	}
	defer tx.Rollback()
	for _, rec := range recs {
		blob, err := json.Marshal(rec)
		if err != nil {
			return fmt.Errorf("marshal record %s: %w", rec.ID, err)
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO records (id, conversation_id, kind, json) VALUES (?, ?, ?, ?)`,
			rec.ID, conversationID, string(rec.Kind), blob); err != nil {
			return fmt.Errorf("append record %s: %w", rec.ID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit append: %w", err)
	}
	return nil
}

// ReadRecords implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReadRecords(ctx context.Context, conversationID string, afterID string) ([]harness.Record, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT json FROM records WHERE conversation_id = ? AND id > ? ORDER BY seq`,
		conversationID, afterID)
	if err != nil {
		return nil, fmt.Errorf("read records of %s: %w", conversationID, err)
	}
	defer rows.Close()
	var out []harness.Record
	for rows.Next() {
		var blob []byte
		if err := rows.Scan(&blob); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		var rec harness.Record
		if err := json.Unmarshal(blob, &rec); err != nil {
			return nil, fmt.Errorf("unmarshal record: %w", err)
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

// AdmitSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) AdmitSubmission(ctx context.Context, sub harness.Submission) (harness.Submission, error) {
	inputJSON, err := json.Marshal(sub.Input)
	if err != nil {
		return harness.Submission{}, fmt.Errorf("marshal input: %w", err)
	}
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO submissions (id, session_key, agent, instance, session, conversation_id,
			status, input_json, attempt_count, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING`,
		sub.ID, sub.SessionKey.String(), sub.SessionKey.Agent, string(sub.SessionKey.Instance),
		sub.SessionKey.Session, sub.ConversationID, string(sub.Status), inputJSON,
		sub.AttemptCount, sub.CreatedAt.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return harness.Submission{}, fmt.Errorf("admit submission %s: %w", sub.ID, err)
	}
	inserted, err := res.RowsAffected()
	if err != nil {
		return harness.Submission{}, fmt.Errorf("admit submission rows: %w", err)
	}
	existing, err := s.GetSubmission(ctx, sub.ID)
	if err != nil {
		return harness.Submission{}, err
	}
	if inserted == 0 {
		existingInput, err := json.Marshal(existing.Input)
		if err != nil {
			return harness.Submission{}, fmt.Errorf("marshal existing input: %w", err)
		}
		if string(existingInput) != string(inputJSON) {
			return harness.Submission{}, harness.ErrDispatchConflict
		}
	}
	return existing, nil
}

const submissionColumns = `id, agent, instance, session, conversation_id, status,
	input_json, attempt_count, attempt_id, owner_id, lease_expires_ns, created_at`

func scanSubmission(row interface{ Scan(...any) error }) (harness.Submission, error) {
	var sub harness.Submission
	var inputJSON []byte
	var leaseNS int64
	var createdAt string
	err := row.Scan(&sub.ID, &sub.SessionKey.Agent, &sub.SessionKey.Instance, &sub.SessionKey.Session,
		&sub.ConversationID, &sub.Status, &inputJSON, &sub.AttemptCount, &sub.AttemptID,
		&sub.OwnerID, &leaseNS, &createdAt)
	if err != nil {
		return harness.Submission{}, err
	}
	if err := json.Unmarshal(inputJSON, &sub.Input); err != nil {
		return harness.Submission{}, fmt.Errorf("unmarshal input of %s: %w", sub.ID, err)
	}
	if leaseNS != 0 {
		sub.LeaseExpiresAt = time.Unix(0, leaseNS)
	}
	sub.CreatedAt, err = time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return harness.Submission{}, fmt.Errorf("parse created_at of %s: %w", sub.ID, err)
	}
	return sub, nil
}

// GetSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetSubmission(ctx context.Context, id string) (harness.Submission, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT `+submissionColumns+` FROM submissions WHERE id = ?`, id)
	sub, err := scanSubmission(row)
	if errors.Is(err, sql.ErrNoRows) {
		return harness.Submission{}, harness.ErrSubmissionNotFound
	}
	if err != nil {
		return harness.Submission{}, fmt.Errorf("get submission %s: %w", id, err)
	}
	return sub, nil
}

func (s *Store) querySubmissions(ctx context.Context, query string, args ...any) ([]harness.Submission, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []harness.Submission
	for rows.Next() {
		sub, err := scanSubmission(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, sub)
	}
	return out, rows.Err()
}

// ListRunnable implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListRunnable(ctx context.Context) ([]harness.Submission, error) {
	subs, err := s.querySubmissions(ctx, `
		SELECT `+submissionColumns+` FROM submissions s
		WHERE s.status = 'queued' AND s.seq = (
			SELECT MIN(s2.seq) FROM submissions s2
			WHERE s2.session_key = s.session_key AND s2.status != 'settled')
		ORDER BY s.seq`)
	if err != nil {
		return nil, fmt.Errorf("list runnable: %w", err)
	}
	return subs, nil
}

// ListByStatus implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListByStatus(ctx context.Context, status harness.SubmissionStatus) ([]harness.Submission, error) {
	subs, err := s.querySubmissions(ctx, `
		SELECT `+submissionColumns+` FROM submissions WHERE status = ? ORDER BY seq`,
		string(status))
	if err != nil {
		return nil, fmt.Errorf("list by status %s: %w", status, err)
	}
	return subs, nil
}

// ClaimSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ClaimSubmission(ctx context.Context, claim harness.SubmissionClaim) (harness.Submission, error) {
	res, err := s.db.ExecContext(ctx, `
		UPDATE submissions SET status = 'running', attempt_id = ?, owner_id = ?,
			lease_expires_ns = ?, attempt_count = attempt_count + 1
		WHERE id = ? AND status = 'queued'`,
		claim.AttemptID, claim.OwnerID, claim.LeaseExpiresAt.UnixNano(), claim.SubmissionID)
	if err != nil {
		return harness.Submission{}, fmt.Errorf("claim %s: %w", claim.SubmissionID, err)
	}
	if err := casApplied(res, s.submissionExists(ctx, claim.SubmissionID)); err != nil {
		return harness.Submission{}, err
	}
	return s.GetSubmission(ctx, claim.SubmissionID)
}

// casApplied maps a zero-row UPDATE to the right sentinel: the submission is
// either missing or in the wrong state.
func casApplied(res sql.Result, exists error) error {
	affected, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("cas rows affected: %w", err)
	}
	if affected == 1 {
		return nil
	}
	if exists != nil {
		return exists
	}
	return harness.ErrClaimLost
}

// submissionExists returns nil when the submission exists, or
// ErrSubmissionNotFound.
func (s *Store) submissionExists(ctx context.Context, id string) error {
	var one int
	err := s.db.QueryRowContext(ctx, `SELECT 1 FROM submissions WHERE id = ?`, id).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return harness.ErrSubmissionNotFound
	}
	return nil
}

// StartAttempt implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) StartAttempt(ctx context.Context, attempt harness.Attempt) error {
	if err := s.submissionExists(ctx, attempt.SubmissionID); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO attempts (id, submission_id, owner_id, started_at) VALUES (?, ?, ?, ?)`,
		attempt.ID, attempt.SubmissionID, attempt.OwnerID,
		attempt.StartedAt.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return fmt.Errorf("start attempt %s: %w", attempt.ID, err)
	}
	return nil
}

// ListAttempts implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListAttempts(ctx context.Context, submissionID string) ([]harness.Attempt, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, submission_id, owner_id, started_at FROM attempts
		WHERE submission_id = ? ORDER BY seq`, submissionID)
	if err != nil {
		return nil, fmt.Errorf("list attempts of %s: %w", submissionID, err)
	}
	defer rows.Close()
	var out []harness.Attempt
	for rows.Next() {
		var att harness.Attempt
		var startedAt string
		if err := rows.Scan(&att.ID, &att.SubmissionID, &att.OwnerID, &startedAt); err != nil {
			return nil, fmt.Errorf("scan attempt: %w", err)
		}
		att.StartedAt, err = time.Parse(time.RFC3339Nano, startedAt)
		if err != nil {
			return nil, fmt.Errorf("parse attempt started_at: %w", err)
		}
		out = append(out, att)
	}
	return out, rows.Err()
}

// RenewLease implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) RenewLease(ctx context.Context, renewal harness.LeaseRenewal) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE submissions SET lease_expires_ns = ?
		WHERE id = ? AND status = 'running' AND attempt_id = ?`,
		renewal.LeaseExpiresAt.UnixNano(), renewal.SubmissionID, renewal.AttemptID)
	if err != nil {
		return fmt.Errorf("renew lease of %s: %w", renewal.SubmissionID, err)
	}
	return casApplied(res, s.submissionExists(ctx, renewal.SubmissionID))
}

// ListExpiredLeases implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ListExpiredLeases(ctx context.Context, now time.Time) ([]harness.Submission, error) {
	subs, err := s.querySubmissions(ctx, `
		SELECT `+submissionColumns+` FROM submissions
		WHERE status = 'running' AND lease_expires_ns <= ? ORDER BY seq`,
		now.UnixNano())
	if err != nil {
		return nil, fmt.Errorf("list expired leases: %w", err)
	}
	return subs, nil
}

// ReleaseSubmission implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReleaseSubmission(ctx context.Context, submissionID, attemptID string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE submissions SET status = 'queued', attempt_id = '', owner_id = '', lease_expires_ns = 0
		WHERE id = ? AND status = 'running' AND attempt_id = ?`,
		submissionID, attemptID)
	if err != nil {
		return fmt.Errorf("release %s: %w", submissionID, err)
	}
	return casApplied(res, s.submissionExists(ctx, submissionID))
}

// ReserveSettlement implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) ReserveSettlement(ctx context.Context, submissionID, attemptID string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE submissions SET status = 'terminalizing'
		WHERE id = ? AND status = 'running' AND attempt_id = ?`,
		submissionID, attemptID)
	if err != nil {
		return fmt.Errorf("reserve settlement of %s: %w", submissionID, err)
	}
	return casApplied(res, s.submissionExists(ctx, submissionID))
}

// FinalizeSettlement implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) FinalizeSettlement(ctx context.Context, submissionID string) error {
	res, err := s.db.ExecContext(ctx, `
		UPDATE submissions SET status = 'settled'
		WHERE id = ? AND status IN ('terminalizing', 'settled')`,
		submissionID)
	if err != nil {
		return fmt.Errorf("finalize settlement of %s: %w", submissionID, err)
	}
	return casApplied(res, s.submissionExists(ctx, submissionID))
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
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO attachments (digest, media_type, size, data) VALUES (?, ?, ?, ?)
		ON CONFLICT(digest) DO NOTHING`,
		ref.Digest, ref.MediaType, ref.Size, data)
	if err != nil {
		return harness.AttachmentRef{}, fmt.Errorf("put attachment: %w", err)
	}
	return ref, nil
}

// GetAttachment implements the corresponding harness.Store method; semantics
// are specified on the contract and pinned by the conformance suite.
func (s *Store) GetAttachment(ctx context.Context, digest string) (harness.Attachment, error) {
	var att harness.Attachment
	err := s.db.QueryRowContext(ctx, `
		SELECT digest, media_type, size, data FROM attachments WHERE digest = ?`, digest).
		Scan(&att.Ref.Digest, &att.Ref.MediaType, &att.Ref.Size, &att.Data)
	if errors.Is(err, sql.ErrNoRows) {
		return harness.Attachment{}, harness.ErrAttachmentNotFound
	}
	if err != nil {
		return harness.Attachment{}, fmt.Errorf("get attachment %s: %w", digest, err)
	}
	return att, nil
}
