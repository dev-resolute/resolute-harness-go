package harness_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// recBuilder fabricates deterministic record logs for reducer properties.
type recBuilder struct{ n int }

func (b *recBuilder) next(kind harness.RecordKind, payload any) harness.Record {
	b.n++
	blob, _ := json.Marshal(payload)
	return harness.Record{
		RecordEnvelope: harness.RecordEnvelope{
			ID:             fmt.Sprintf("01A%023d", b.n),
			Kind:           kind,
			ConversationID: "conv-1",
			Session:        "default",
			Time:           time.Unix(int64(b.n), 0),
		},
		Payload: blob,
	}
}

func (b *recBuilder) user(body string) harness.Record {
	return b.next(harness.KindUserMessage, harness.UserMessagePayload{Body: body})
}

func (b *recBuilder) assistant(text string) harness.Record {
	body, _ := json.Marshal(text)
	return b.next(harness.KindAssistantMessageCompleted, harness.AssistantMessageCompletedPayload{
		Message: harness.MessagePayload{Role: "assistant", Type: "text", Body: body},
	})
}

func (b *recBuilder) compaction(summary, firstKeptID string) harness.Record {
	return b.next(harness.KindCompaction, harness.CompactionPayload{
		Summary:          summary,
		FirstKeptEntryID: firstKeptID,
	})
}

// randomLog builds a seeded random record log with interleaved compactions
// pointing at valid kept entries.
func randomLog(seed int64, length int) []harness.Record {
	rng := rand.New(rand.NewSource(seed))
	b := &recBuilder{}
	var log []harness.Record
	for len(log) < length {
		switch rng.Intn(6) {
		case 0:
			log = append(log, b.assistant(fmt.Sprintf("answer %d", len(log))))
		case 1:
			if len(log) > 2 {
				// Point at a random existing entry as the kept boundary.
				kept := log[rng.Intn(len(log))]
				log = append(log, b.compaction(fmt.Sprintf("summary %d", len(log)), kept.ID))
				continue
			}
			log = append(log, b.user(fmt.Sprintf("prompt %d", len(log))))
		default:
			log = append(log, b.user(fmt.Sprintf("prompt %d", len(log))))
		}
	}
	return log
}

// Determinism: the same log always reduces to the same tree.
func TestReduceDeterministic(t *testing.T) {
	t.Parallel()
	for seed := int64(0); seed < 20; seed++ {
		log := randomLog(seed, 40)
		a := harness.Reduce(log)
		b := harness.Reduce(log)
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("seed %d: Reduce is not deterministic", seed)
		}
	}
}

// Prefix consistency: Reduce(log[:n]) holds exactly the first n records in
// order, every parent pointer references an earlier entry (or ""), and a
// parent link only changes when a LATER compaction record re-parents it.
func TestReducePrefixConsistent(t *testing.T) {
	t.Parallel()
	for seed := int64(0); seed < 10; seed++ {
		log := randomLog(seed, 30)
		full := harness.Reduce(log)
		for n := 0; n <= len(log); n++ {
			prefix := harness.Reduce(log[:n])
			if len(prefix.Entries) != n {
				t.Fatalf("seed %d n %d: prefix has %d entries", seed, n, len(prefix.Entries))
			}
			inPrefix := map[string]bool{}
			for _, e := range prefix.Entries {
				inPrefix[e.Record.ID] = true
			}
			for i, e := range prefix.Entries {
				if e.Record.ID != log[i].ID {
					t.Fatalf("seed %d n %d: entry %d is %s, want %s (log order)", seed, n, i, e.Record.ID, log[i].ID)
				}
				// Parents always reference an entry of the same prefix. A
				// kept entry legitimately points FORWARD to the compaction
				// node that re-parented it.
				if e.ParentID != "" && !inPrefix[e.ParentID] {
					t.Fatalf("seed %d n %d: entry %s has unknown parent %s", seed, n, e.Record.ID, e.ParentID)
				}

				// A parent may differ from the full reduction only if some
				// record beyond the prefix is a compaction (re-parenting is
				// the only sanctioned mutation).
				if e.ParentID != full.Entries[i].ParentID {
					reparentedLater := false
					for _, later := range log[n:] {
						if later.Kind == harness.KindCompaction {
							reparentedLater = true
							break
						}
					}
					if !reparentedLater {
						t.Fatalf("seed %d n %d: entry %s parent %q differs from full %q without a later compaction",
							seed, n, e.Record.ID, e.ParentID, full.Entries[i].ParentID)
					}
				}
			}
		}
	}
}

// Re-parent correctness under one and multiple compactions.
func TestReduceCompactionReparents(t *testing.T) {
	t.Parallel()
	b := &recBuilder{}
	u1 := b.user("one")
	a1 := b.assistant("answer one")
	u2 := b.user("two")
	a2 := b.assistant("answer two")
	c1 := b.compaction("summary of one", u2.ID) // keep from u2 onward
	u3 := b.user("three")
	a3 := b.assistant("answer three")
	c2 := b.compaction("summary through two", u3.ID) // keep from u3 onward
	u4 := b.user("four")

	log := []harness.Record{u1, a1, u2, a2, c1, u3, a3, c2, u4}
	tree := harness.Reduce(log)

	if len(tree.Entries) != len(log) {
		t.Fatalf("entries = %d, want %d (append-only: nothing removed)", len(tree.Entries), len(log))
	}

	path := tree.ActiveLeafPath()
	var pathIDs []string
	for _, rec := range path {
		pathIDs = append(pathIDs, rec.ID)
	}
	want := []string{c2.ID, u3.ID, a3.ID, u4.ID}
	if !reflect.DeepEqual(pathIDs, want) {
		t.Fatalf("active leaf path = %v, want %v (latest compaction roots the branch)", pathIDs, want)
	}

	// Single-compaction intermediate state: after c1, the path roots at c1.
	mid := harness.Reduce(log[:6]) // through u3
	var midIDs []string
	for _, rec := range mid.ActiveLeafPath() {
		midIDs = append(midIDs, rec.ID)
	}
	if !reflect.DeepEqual(midIDs, []string{c1.ID, u2.ID, a2.ID, u3.ID}) {
		t.Fatalf("mid path = %v, want rooted at first compaction", midIDs)
	}

	// Summarized entries stay reachable off the abandoned branch.
	byID := map[string]harness.ReducedEntry{}
	for _, e := range tree.Entries {
		byID[e.Record.ID] = e
	}
	if byID[a1.ID].ParentID != u1.ID {
		t.Fatalf("abandoned branch mutated: a1 parent = %q, want %q", byID[a1.ID].ParentID, u1.ID)
	}
}

// A compaction whose kept pointer is missing or unknown falls back to
// chaining as the new leaf.
func TestReduceCompactionWithoutKeptPointer(t *testing.T) {
	t.Parallel()
	b := &recBuilder{}
	u1 := b.user("one")
	c := b.compaction("summary of everything", "")
	tree := harness.Reduce([]harness.Record{u1, c})
	var ids []string
	for _, rec := range tree.ActiveLeafPath() {
		ids = append(ids, rec.ID)
	}
	if !reflect.DeepEqual(ids, []string{u1.ID, c.ID}) {
		t.Fatalf("path = %v, want the summary chained as leaf", ids)
	}
}
