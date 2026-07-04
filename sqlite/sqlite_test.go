package sqlite_test

import (
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/sqlite"
	"github.com/dev-resolute/resolute-harness-go/storetest"
)

func TestConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) harness.Store {
		s, err := sqlite.Open(filepath.Join(t.TempDir(), "harness.db"))
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		t.Cleanup(func() {
			if err := s.Close(); err != nil {
				t.Errorf("Close: %v", err)
			}
		})
		return s
	})
}

func TestOpenDirectoryUsesDefaultFilename(t *testing.T) {
	dir := t.TempDir()
	s, err := sqlite.Open(dir)
	if err != nil {
		t.Fatalf("Open(dir): %v", err)
	}
	defer s.Close()
	if _, err := os.Stat(filepath.Join(dir, "harness.db")); err != nil {
		t.Fatalf("expected harness.db inside the directory: %v", err)
	}
}

func TestOpenPersistsAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "harness.db")
	s, err := sqlite.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	ctx := t.Context()
	conv := harness.Conversation{
		ID:  "conv-1",
		Key: harness.SessionKey{Agent: "support", Instance: "acme", Session: "default"},
	}
	if _, _, err := s.EnsureConversation(ctx, conv); err != nil {
		t.Fatalf("EnsureConversation: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	reopened, err := sqlite.Open(path)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	defer reopened.Close()
	got, err := reopened.GetConversation(ctx, conv.Key)
	if err != nil {
		t.Fatalf("GetConversation after reopen: %v", err)
	}
	if got.ID != conv.ID {
		t.Fatalf("reopened conversation id = %q, want %q", got.ID, conv.ID)
	}
}

func TestOpenRejectsUnsupportedSchemaVersion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "harness.db")
	s, err := sqlite.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Stamp a future schema version directly, as a newer build would.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw db: %v", err)
	}
	if _, err := db.Exec("PRAGMA user_version = 999"); err != nil {
		t.Fatalf("stamp version: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}
	_, err = sqlite.Open(path)
	if !errors.Is(err, harness.ErrUnsupportedSchema) {
		t.Fatalf("Open(v999) error = %v, want ErrUnsupportedSchema", err)
	}
}
