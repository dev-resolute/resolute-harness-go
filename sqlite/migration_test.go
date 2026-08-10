package sqlite_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	_ "modernc.org/sqlite"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/sqlite"
)

// schemaV2 mirrors the pre-v3 submissions DDL (git show 970d8b2:sqlite/sqlite.go),
// before the parent-link and waiting-state columns landed (HARNESS-15).
const schemaV2 = `
CREATE TABLE conversations (
	key         TEXT PRIMARY KEY,
	id          TEXT NOT NULL UNIQUE,
	agent       TEXT NOT NULL,
	instance    TEXT NOT NULL,
	session     TEXT NOT NULL,
	created_at  TEXT NOT NULL
);
CREATE TABLE records (
	seq             INTEGER PRIMARY KEY AUTOINCREMENT,
	id              TEXT NOT NULL UNIQUE,
	conversation_id TEXT NOT NULL,
	kind            TEXT NOT NULL,
	json            BLOB NOT NULL
);
CREATE INDEX records_by_conversation ON records(conversation_id, id);
CREATE TABLE submissions (
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
	last_error       TEXT NOT NULL DEFAULT '',
	created_at       TEXT NOT NULL
);
CREATE INDEX submissions_by_session ON submissions(session_key, seq);
CREATE INDEX submissions_by_status ON submissions(status);
CREATE TABLE attempts (
	seq           INTEGER PRIMARY KEY AUTOINCREMENT,
	id            TEXT NOT NULL,
	submission_id TEXT NOT NULL,
	owner_id      TEXT NOT NULL,
	started_at    TEXT NOT NULL
);
CREATE INDEX attempts_by_submission ON attempts(submission_id, seq);
CREATE TABLE attachments (
	digest     TEXT PRIMARY KEY,
	media_type TEXT NOT NULL,
	size       INTEGER NOT NULL,
	data       BLOB NOT NULL
);
`

// A v2 database opens under the current build: the v2->v3 migration adds the
// six new columns, stamps user_version 3, and leaves pre-existing rows
// reading back with zero-valued new fields.
func TestMigrationV2ToV3(t *testing.T) {
	path := filepath.Join(t.TempDir(), "harness.db")
	ctx := context.Background()

	// Build a v2-shaped database by hand, with one legacy submission.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	if _, err := raw.Exec(schemaV2); err != nil {
		t.Fatalf("create v2 schema: %v", err)
	}
	if _, err := raw.Exec("PRAGMA user_version = 2"); err != nil {
		t.Fatalf("stamp user_version 2: %v", err)
	}
	inputJSON, err := json.Marshal(harness.UserMessage("legacy input"))
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	createdAt := time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC)
	if _, err := raw.Exec(`
		INSERT INTO submissions (id, session_key, agent, instance, session, conversation_id,
			status, input_json, attempt_count, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		"legacy-sub", "support/acme/legacy", "support", "acme", "legacy", "conv-legacy",
		"queued", inputJSON, 2, createdAt.Format(time.RFC3339Nano)); err != nil {
		t.Fatalf("insert legacy row: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close v2 db: %v", err)
	}

	// The current Open migrates it.
	store, err := sqlite.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})

	// user_version is stamped 3 and the six new columns exist.
	check, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open (check): %v", err)
	}
	defer check.Close()
	var version int
	if err := check.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		t.Fatalf("read user_version: %v", err)
	}
	if version != 3 {
		t.Fatalf("user_version = %d, want 3", version)
	}
	rows, err := check.Query("PRAGMA table_info(submissions)")
	if err != nil {
		t.Fatalf("table_info: %v", err)
	}
	columns := map[string]bool{}
	for rows.Next() {
		var cid int
		var name, ctype string
		var notnull, pk int
		var dflt sql.NullString
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			t.Fatalf("scan table_info: %v", err)
		}
		columns[name] = true
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("table_info rows: %v", err)
	}
	for _, col := range []string{
		"parent_submission_id", "parent_call_id", "depth",
		"pending_resume", "wait_until_ns", "cancel_requested",
	} {
		if !columns[col] {
			t.Fatalf("submissions missing migrated column %q (have %v)", col, columns)
		}
	}

	// The legacy row reads back with zero-valued new fields and its v2 data
	// intact.
	got, err := store.GetSubmission(ctx, "legacy-sub")
	if err != nil {
		t.Fatalf("GetSubmission: %v", err)
	}
	if got.ParentSubmissionID != "" || got.ParentCallID != "" || got.Depth != 0 ||
		got.PendingResume || !got.WaitUntil.IsZero() || got.CancelRequested {
		t.Fatalf("migrated row has non-zero new fields: %+v", got)
	}
	if got.Status != harness.StatusQueued || got.AttemptCount != 2 ||
		got.SessionKey.Session != "legacy" || !got.CreatedAt.Equal(createdAt) {
		t.Fatalf("migrated row lost v2 data: %+v", got)
	}
	if got.Input.Kind != harness.InboundUser || got.Input.Body != "legacy input" {
		t.Fatalf("migrated row input = %+v, want legacy user message", got.Input)
	}
}
