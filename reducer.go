package harness

import "encoding/json"

// ReducedEntry is one node of the reduced conversation tree: a canonical
// record plus its parent link. ParentID is "" for a root entry.
type ReducedEntry struct {
	Record   Record
	ParentID string
}

// ConversationTree is the reduced, parent-linked projection of a conversation
// log. It is derived exclusively by Reduce and never stored.
type ConversationTree struct {
	// Entries holds every reduced entry in log order. Nothing is ever
	// removed: compaction re-parents, it never deletes.
	Entries []ReducedEntry
	// LeafID is the ID of the active leaf entry; "" for an empty tree.
	LeafID string
}

// Reduce is the pure projection from a record log to a conversation tree.
// It is deterministic, and prefix-consistent: Reduce(log[:n]) contains
// exactly the first n records in order, and an entry's parent only ever
// changes when a later compaction record re-parents it.
//
// Non-compaction records chain onto the current leaf. A compaction record
// carrying FirstKeptEntryID becomes a summary node at the root of the active
// branch: the kept entry re-parents onto it, and everything the summary
// covered remains reachable as an abandoned branch — history is never
// rewritten, only re-rooted.
func Reduce(records []Record) ConversationTree {
	tree := ConversationTree{Entries: make([]ReducedEntry, 0, len(records))}
	index := make(map[string]int, len(records))
	leaf := ""
	for _, rec := range records {
		if rec.Kind == KindCompaction {
			if kept, ok := index[compactionFirstKept(rec)]; ok {
				tree.Entries = append(tree.Entries, ReducedEntry{Record: rec, ParentID: ""})
				index[rec.ID] = len(tree.Entries) - 1
				tree.Entries[kept].ParentID = rec.ID
				// The leaf is unchanged: the kept entry's descendants still
				// chain to it.
				continue
			}
			// No kept entry (everything summarized, or a malformed pointer):
			// the summary node becomes the new leaf.
		}
		tree.Entries = append(tree.Entries, ReducedEntry{Record: rec, ParentID: leaf})
		index[rec.ID] = len(tree.Entries) - 1
		leaf = rec.ID
		tree.LeafID = leaf
	}
	tree.LeafID = leaf
	return tree
}

// compactionFirstKept extracts the re-parent pointer from a compaction
// record; "" when absent or undecodable.
func compactionFirstKept(rec Record) string {
	var p CompactionPayload
	if err := json.Unmarshal(rec.Payload, &p); err != nil {
		return ""
	}
	return p.FirstKeptEntryID
}

// ActiveLeafPath returns the records on the path from the active branch's
// root to the leaf, in order. This is what the projection adapter serves to
// agent-core.
func (t ConversationTree) ActiveLeafPath() []Record {
	byID := make(map[string]ReducedEntry, len(t.Entries))
	for _, e := range t.Entries {
		byID[e.Record.ID] = e
	}
	var path []Record
	for id := t.LeafID; id != ""; {
		e, ok := byID[id]
		if !ok {
			break
		}
		path = append(path, e.Record)
		id = e.ParentID
	}
	// Reverse into root→leaf order.
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}
