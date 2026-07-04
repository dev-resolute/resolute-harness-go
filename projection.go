package harness

import (
	"context"
	"errors"
	"fmt"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// CompactionPayload is the payload of a compaction record: the summary text
// plus the re-parent point. StartIdx/EndIdx mirror agent-core's
// BranchSummary range so LoadBranchSummaries can round-trip it.
type CompactionPayload struct {
	Summary          string `json:"summary"`
	FirstKeptEntryID string `json:"firstKeptEntryId,omitempty"`
	StartIdx         int    `json:"startIdx"`
	EndIdx           int    `json:"endIdx"`
}

func (*CompactionPayload) payloadKind() RecordKind { return KindCompaction }

// projection is the pi.SessionRepo implementation backed by the reduced
// conversation tree (ADR-0003). It is a read-side view: Load serves the
// active leaf path, Append is a no-op (canonical records are authored from
// the event stream), and AppendBranchSummary writes a compaction record.
//
// One projection serves one run of one submission; the correlation fields
// stamp any record it authors.
type projection struct {
	store        ConversationStore
	conv         Conversation
	systemPrompt string

	submissionID string
	attemptID    string
	turnID       func() string
}

var _ pi.SessionRepo = (*projection)(nil)

func (p *projection) Create(ctx context.Context) (pi.SessionID, error) {
	// The engine always passes an explicit SessionID; a Create call would
	// mean the conversation pre-exists, so hand back its id.
	return pi.SessionID(p.conv.ID), nil
}

// Append is deliberately a no-op: canonical records are authored from the
// per-prompt event stream, not from the repo. agent-core keeps its own
// in-memory transcript in sync during a prompt, and the next Load rebuilds
// from the log.
func (p *projection) Append(ctx context.Context, id pi.SessionID, msgs ...pi.Message) error {
	return nil
}

func (p *projection) Load(ctx context.Context, id pi.SessionID) ([]pi.Message, error) {
	recs, err := p.store.ReadRecords(ctx, string(id), "")
	if err != nil {
		return nil, fmt.Errorf("load conversation %s: %w", id, err)
	}
	path := Reduce(recs).ActiveLeafPath()
	var msgs []pi.Message
	if p.systemPrompt != "" {
		msgs = append(msgs, pi.NewSystem(p.systemPrompt))
	}
	for _, rec := range path {
		m, ok, err := recordToMessage(rec)
		if err != nil {
			return nil, err
		}
		if ok {
			msgs = append(msgs, m)
		}
	}
	return msgs, nil
}

// recordToMessage maps a canonical record to the transcript message it
// represents. The bool is false for record kinds that carry no transcript
// content (lifecycle, deltas, settlement).
func recordToMessage(rec Record) (pi.Message, bool, error) {
	switch rec.Kind {
	case KindUserMessage:
		var p UserMessagePayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return pi.NewText("user", p.Body), true, nil
	case KindSignal:
		var p SignalPayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return signalToMessage(p), true, nil
	case KindAssistantMessageCompleted:
		var p AssistantMessageCompletedPayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return p.Message.ToPi(), true, nil
	case KindAssistantToolCall:
		var p AssistantToolCallPayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return pi.NewToolCallWithSignature("assistant", p.CallID, p.ToolName, p.Args, p.ThoughtSignature), true, nil
	case KindToolOutcome:
		var p ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return pi.NewToolResult("tool", p.CallID, p.ToolName, p.Content, p.Data, p.IsError), true, nil
	case KindCompaction:
		var p CompactionPayload
		if err := rec.DecodePayload(&p); err != nil {
			return pi.Message{}, false, err
		}
		return pi.NewBranchSummaryMessage(p.Summary), true, nil
	default:
		return pi.Message{}, false, nil
	}
}

func (p *projection) List(ctx context.Context) ([]pi.SessionMeta, error) {
	return []pi.SessionMeta{{ID: pi.SessionID(p.conv.ID), CreatedAt: p.conv.CreatedAt}}, nil
}

// AppendBranchSummary lands the compaction record. The summary's EndIdx —
// a message index into what Load returned to the compactor — is translated
// into FirstKeptEntryID, the record-space re-parent point the reducer folds
// into the tree.
func (p *projection) AppendBranchSummary(ctx context.Context, id pi.SessionID, summary pi.BranchSummary) error {
	recs, err := p.store.ReadRecords(ctx, string(id), "")
	if err != nil {
		return fmt.Errorf("read records before compaction: %w", err)
	}
	firstKept, err := p.firstKeptEntryID(recs, summary.EndIdx)
	if err != nil {
		return err
	}
	rec := Record{
		RecordEnvelope: RecordEnvelope{
			ID:             newULID(),
			Kind:           KindCompaction,
			ConversationID: p.conv.ID,
			Session:        p.conv.Key.Session,
			SubmissionID:   p.submissionID,
			AttemptID:      p.attemptID,
			Time:           summary.CreatedAt,
		},
		Payload: mustPayload(&CompactionPayload{
			Summary:          summary.Summary,
			FirstKeptEntryID: firstKept,
			StartIdx:         summary.StartIdx,
			EndIdx:           summary.EndIdx,
		}),
	}
	if p.turnID != nil {
		rec.TurnID = p.turnID()
	}
	if err := p.store.AppendRecords(ctx, p.conv.ID, []Record{rec}); err != nil {
		return fmt.Errorf("append compaction record: %w", err)
	}
	return nil
}

// firstKeptEntryID finds the record whose message index equals endIdx —
// the first message the compaction keeps — walking the active leaf path
// with the same message numbering Load produces (the optional system
// message occupies index 0).
func (p *projection) firstKeptEntryID(recs []Record, endIdx int) (string, error) {
	msgIdx := 0
	if p.systemPrompt != "" {
		msgIdx = 1
	}
	for _, rec := range Reduce(recs).ActiveLeafPath() {
		_, isMessage, err := recordToMessage(rec)
		if err != nil {
			return "", err
		}
		if !isMessage {
			continue
		}
		if msgIdx == endIdx {
			return rec.ID, nil
		}
		msgIdx++
	}
	// Everything was summarized; the reducer chains the summary as the leaf.
	return "", nil
}

// LoadBranchSummaries deliberately returns nothing: compaction summaries
// live INLINE on the re-parented active leaf path as branch_summary
// messages (recordToMessage), so each Compact sees the already-compacted
// view and summaries stay incremental. Serving them here too would double
// count.
func (p *projection) LoadBranchSummaries(ctx context.Context, id pi.SessionID) ([]pi.BranchSummary, error) {
	return nil, nil
}

func (p *projection) Delete(ctx context.Context, id pi.SessionID) error {
	return errors.New("conversation deletion is not supported through the projection")
}
