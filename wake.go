package harness

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// wakeParent performs the settlement hand-off (HARNESS-15): when a spawned
// child settles, append the parent's tool_outcome for the pending call and —
// once it is the last unsettled child — requeue the parent. Idempotent: an
// existing outcome naming the pending CallID wins, so re-wakes (the
// in-reservation run plus the post-finalize replay, or a crash-recovery
// re-run) no-op.
//
// Children are correlated via ListChildSubmissions(parent.ID) — the durable
// parent link written at admission — never via task_spawned records: a
// record can be missing (dropped spawn event, unrecoverable ParentRef), the
// child rows cannot.
//
// The wake is the second sanctioned out-of-band record author after the
// dangling-call reconciler: the outcome for a suspended task call comes
// from the child's settlement, not from the parent's event stream.
func (c *coordinator) wakeParent(ctx context.Context, child Submission, payload SettledPayload) error {
	parent, err := c.rt.store.GetSubmission(ctx, child.ParentSubmissionID)
	if err != nil {
		return fmt.Errorf("load parent %s: %w", child.ParentSubmissionID, err)
	}
	if parent.Status == StatusSettled || parent.Status == StatusTerminalizing {
		// Late wake against a terminal parent: guarded no-op.
		return nil
	}

	// Repair a missing spawn record BEFORE appending the outcome: the child
	// may have settled before the parent's consumer authored task_spawned
	// (a fast settle, or the crash window between admission and the spawn
	// event). The repair shares park-time reconciliation's recovery — the
	// same check-then-append by CallID, the same record ID recovered from
	// the child conversation's ParentRef — so the log order is
	// call→spawn→outcome universally. A missing ParentRef skips the repair;
	// the outcome still lands (the park-time disclosure).
	if err := spawnRecordRepair(ctx, c.rt, parent, child, Correlation{
		SessionKey:     parent.SessionKey,
		ConversationID: parent.ConversationID,
		SubmissionID:   parent.ID,
	}); err != nil {
		return fmt.Errorf("repair spawn record for wake: %w", err)
	}

	content, isError, err := c.wakeContent(ctx, child, payload)
	if err != nil {
		return err
	}

	// Append the parent's outcome for the pending call unless one already
	// names the CallID — check-then-append like appendSettledRecordOnce. An
	// existing outcome wins: an idempotent re-drive's reconciler may have
	// synthesized one, and a re-wake must not double-author. The
	// check-then-append idempotency relies on single-process wake
	// serialization; multi-node duplicate-wake fencing is deferred (see
	// docs/adr/0010-v1-scope-full-engine-semantics.md). The same
	// check-then-append races expireWait's wait-expiry outcomes for the
	// same CallID; as there, the eventual close is a store-level guarded
	// append.
	recs, err := c.rt.store.ReadRecords(ctx, parent.ConversationID, "")
	if err != nil {
		return fmt.Errorf("read parent records for wake: %w", err)
	}
	outcomeExists := false
	for _, rec := range recs {
		if rec.Kind != KindToolOutcome {
			continue
		}
		var p ToolOutcomePayload
		if err := rec.DecodePayload(&p); err != nil {
			return fmt.Errorf("decode tool_outcome for wake: %w", err)
		}
		if p.CallID == child.ParentCallID {
			outcomeExists = true
			break
		}
	}
	if !outcomeExists {
		rec := Record{
			RecordEnvelope: RecordEnvelope{
				ID:             newULID(),
				Kind:           KindToolOutcome,
				ConversationID: parent.ConversationID,
				Session:        parent.SessionKey.Session,
				SubmissionID:   parent.ID,
				Time:           time.Now(),
			},
			Payload: mustPayload(&ToolOutcomePayload{
				CallID:   child.ParentCallID,
				ToolName: "task",
				IsError:  isError,
				Content:  content,
			}),
		}
		if err := c.rt.store.AppendRecords(ctx, parent.ConversationID, []Record{rec}); err != nil {
			return fmt.Errorf("append wake outcome: %w", err)
		}
		c.rt.notifyAppend()
	}

	// Requeue only when every child of the parent is settled. The settling
	// child itself is terminalizing inside the two-phase reservation (its
	// settled record just landed), so it counts as settled.
	children, err := c.rt.store.ListChildSubmissions(ctx, parent.ID)
	if err != nil {
		return fmt.Errorf("list children for wake: %w", err)
	}
	for _, ch := range children {
		if ch.Status != StatusSettled && ch.ID != child.ID {
			return nil // a sibling is still in flight
		}
	}
	if err := c.rt.store.ResumeSubmission(ctx, parent.ID); err != nil {
		if errors.Is(err, ErrClaimLost) {
			return nil // a racing wake already requeued
		}
		return fmt.Errorf("requeue parent %s: %w", parent.ID, err)
	}
	// Nudge the claim loop without blocking (same as Dispatch).
	select {
	case c.rt.wake <- struct{}{}:
	default:
	}
	return nil
}

// wakeContent builds the parent's tool_outcome content from the child's
// settlement: the structured result when present, else the child's final
// assistant text. A failure carries the error (and code) plus any partial
// final text, and marks the outcome IsError.
func (c *coordinator) wakeContent(ctx context.Context, child Submission, payload SettledPayload) (string, bool, error) {
	if payload.Status == SettledSucceeded {
		if len(payload.Result) > 0 {
			return string(payload.Result), false, nil
		}
		text, err := finalAssistantText(ctx, c.rt.store, child.ConversationID)
		if err != nil {
			return "", false, err
		}
		return text, false, nil
	}
	content := payload.Error
	if payload.ErrorCode != "" {
		if content == "" {
			content = fmt.Sprintf("error (code: %s)", payload.ErrorCode)
		} else {
			content = fmt.Sprintf("%s (code: %s)", content, payload.ErrorCode)
		}
	}
	text, err := finalAssistantText(ctx, c.rt.store, child.ConversationID)
	if err != nil {
		return "", false, err
	}
	if text != "" {
		content += "\n\nPartial output before failure:\n" + text
	}
	return content, true, nil
}

// finalAssistantText returns the text of the last completed assistant
// message on the conversation's active leaf path — the run's answer when no
// structured result was requested ("" when the run produced none).
func finalAssistantText(ctx context.Context, store ConversationStore, conversationID string) (string, error) {
	recs, err := store.ReadRecords(ctx, conversationID, "")
	if err != nil {
		return "", fmt.Errorf("read records for final assistant text: %w", err)
	}
	path := Reduce(recs).ActiveLeafPath()
	for i := len(path) - 1; i >= 0; i-- {
		if path[i].Kind != KindAssistantMessageCompleted {
			continue
		}
		var p AssistantMessageCompletedPayload
		if err := path[i].DecodePayload(&p); err != nil {
			return "", fmt.Errorf("decode assistant_message_completed: %w", err)
		}
		return p.Message.ToPi().Text(), nil
	}
	return "", nil
}
