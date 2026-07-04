package harness

// ReducedEntry is one node of the reduced conversation tree: a canonical
// record plus its parent link. ParentID is "" for the root entry.
type ReducedEntry struct {
	Record   Record
	ParentID string
}

// ConversationTree is the reduced, parent-linked projection of a conversation
// log. It is derived exclusively by Reduce and never stored.
type ConversationTree struct {
	// Entries holds every reduced entry in log order.
	Entries []ReducedEntry
	// LeafID is the ID of the active leaf entry; "" for an empty tree.
	LeafID string
}

// Reduce is the pure projection from a record log to a conversation tree.
// It is deterministic and prefix-consistent: Reduce(log[:n]) agrees with
// Reduce(log) on the first n entries for all n.
//
// v1 walking-skeleton semantics: entries chain linearly, each parented on the
// previous record. Compaction re-parenting joins in the tree slice
// (HARNESS-5).
func Reduce(records []Record) ConversationTree {
	tree := ConversationTree{Entries: make([]ReducedEntry, 0, len(records))}
	parent := ""
	for _, rec := range records {
		tree.Entries = append(tree.Entries, ReducedEntry{Record: rec, ParentID: parent})
		parent = rec.ID
		tree.LeafID = rec.ID
	}
	return tree
}

// ActiveLeafPath returns the records on the path from the root to the active
// leaf, in order. This is what the projection adapter serves to agent-core.
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
