package harness_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// liveSSE is an open SSE connection whose frames arrive on Events as they
// are written by the server.
type liveSSE struct {
	Events <-chan sseEvent
	Done   <-chan error // closed with the terminal read error (nil on clean EOF)
	cancel context.CancelFunc
}

func (l *liveSSE) Close() { l.cancel() }

// openSSE opens a live SSE connection and parses frames incrementally.
func openSSE(t *testing.T, server *httptest.Server, path string, lastEventID string) *liveSSE {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+path, nil)
	if err != nil {
		cancel()
		t.Fatalf("build GET %s: %v", path, err)
	}
	if lastEventID != "" {
		req.Header.Set("Last-Event-ID", lastEventID)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		cancel()
		t.Fatalf("GET %s: %v", path, err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		cancel()
		t.Fatalf("GET %s status = %d, want 200", path, resp.StatusCode)
	}

	events := make(chan sseEvent, 256)
	done := make(chan error, 1)
	go func() {
		defer close(events)
		defer resp.Body.Close()
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		var cur sseEvent
		for scanner.Scan() {
			line := scanner.Text()
			switch {
			case line == "":
				if cur.ID != "" || cur.Data != "" {
					events <- cur
				}
				cur = sseEvent{}
			case strings.HasPrefix(line, "id: "):
				cur.ID = strings.TrimPrefix(line, "id: ")
			case strings.HasPrefix(line, "event: "):
				cur.Event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				cur.Data = strings.TrimPrefix(line, "data: ")
			}
		}
		err := scanner.Err()
		if err == io.ErrUnexpectedEOF || err == context.Canceled {
			err = nil
		}
		done <- err
		close(done)
	}()
	t.Cleanup(cancel)
	return &liveSSE{Events: events, Done: done, cancel: cancel}
}

// collectUntil reads frames until pred is satisfied or the timeout hits.
func collectUntil(t *testing.T, live *liveSSE, timeout time.Duration, pred func([]sseEvent) bool) []sseEvent {
	t.Helper()
	var got []sseEvent
	deadline := time.After(timeout)
	for {
		if pred(got) {
			return got
		}
		select {
		case ev, ok := <-live.Events:
			if !ok {
				return got
			}
			got = append(got, ev)
		case <-deadline:
			t.Fatalf("timed out collecting SSE frames; got %d: %+v", len(got), kindsOf(got))
		}
	}
}

func kindsOf(events []sseEvent) []string {
	out := make([]string, len(events))
	for i, ev := range events {
		out[i] = ev.Event
	}
	return out
}

func hasKind(events []sseEvent, kind harness.RecordKind) func([]sseEvent) bool {
	_ = events
	return func(got []sseEvent) bool {
		for _, ev := range got {
			if ev.Event == string(kind) {
				return true
			}
		}
		return false
	}
}

// A live reader attached before the dispatch sees message-started, batched
// deltas, and settlement while the run is in flight, then the stream closes
// once the conversation is idle.
func TestLiveTailStreamsDeltasWhileInFlight(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().
		RespondThinking("pondering the request").
		RespondText("streamed answer").
		Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	// Dispatch, then immediately attach a live reader from offset "".
	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", `{"kind":"user","body":"stream to me"}`, http.StatusAccepted, &res)
	live := openSSE(t, server, "/agents/support/acme", "")
	defer live.Close()

	events := collectUntil(t, live, 15*time.Second, hasKind(nil, harness.KindSubmissionSettled))

	// Ordered expectations: started before deltas, deltas before completed.
	wantOrder := []string{
		string(harness.KindUserMessage),
		string(harness.KindAssistantMessageStarted),
		string(harness.KindAssistantThinkingDelta),
		string(harness.KindAssistantTextDelta),
		string(harness.KindAssistantMessageCompleted),
		string(harness.KindSubmissionSettled),
	}
	i := 0
	for _, ev := range events {
		if i < len(wantOrder) && ev.Event == wantOrder[i] {
			i++
		}
	}
	if i != len(wantOrder) {
		t.Fatalf("stream kinds %v missing ordered subsequence %v", kindsOf(events), wantOrder)
	}

	// Delta concatenation reproduces the final text.
	var text strings.Builder
	for _, ev := range events {
		if ev.Event != string(harness.KindAssistantTextDelta) {
			continue
		}
		var rec harness.Record
		if err := json.Unmarshal([]byte(ev.Data), &rec); err != nil {
			t.Fatalf("unmarshal delta record: %v", err)
		}
		var p harness.TextDeltaPayload
		if err := rec.DecodePayload(&p); err != nil {
			t.Fatalf("decode delta payload: %v", err)
		}
		text.WriteString(p.Text)
	}
	if text.String() != "streamed answer" {
		t.Fatalf("concatenated text deltas = %q, want %q", text.String(), "streamed answer")
	}

	// Once the conversation is idle the server closes the stream cleanly.
	select {
	case err := <-live.Done:
		if err != nil {
			t.Fatalf("stream ended with error: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("stream did not close after the conversation went idle")
	}
}

// Disconnecting mid-stream and reconnecting with Last-Event-ID yields
// exactly the missed records — no gaps, no duplicates.
func TestReconnectMidStreamNoGapsNoDuplicates(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondToolCall("get_weather", json.RawMessage(`{"city":"Berlin"}`)).Add()
	provider.OnAny().RespondText("It is sunny.").Add()

	rt := newTestRuntimeWithTools(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme?wait=true", `{"kind":"user","body":"What's the weather in Berlin?"}`, http.StatusOK, &res)

	// First reader takes the first three frames, then disconnects.
	first := openSSE(t, server, "/agents/support/acme", "")
	head := collectUntil(t, first, 10*time.Second, func(got []sseEvent) bool { return len(got) >= 3 })
	first.Close()

	// Reconnect from the last seen offset and read to close.
	second := openSSE(t, server, "/agents/support/acme", head[2].ID)
	var tail []sseEvent
	for ev := range second.Events {
		tail = append(tail, ev)
	}

	ctx := context.Background()
	all, err := rt.Records(ctx, res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	var gotIDs []string
	for _, ev := range append(head[:3:3], tail...) {
		gotIDs = append(gotIDs, ev.ID)
	}
	if len(gotIDs) != len(all) {
		t.Fatalf("head+tail = %d frames, want %d records", len(gotIDs), len(all))
	}
	for i, rec := range all {
		if gotIDs[i] != rec.ID {
			t.Fatalf("frame %d id = %s, want %s (gap or duplicate at reconnect)", i, gotIDs[i], rec.ID)
		}
	}
}

// Two concurrent live readers observe identical record sequences.
func TestTwoConcurrentReadersSeeIdenticalStreams(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("same for everyone").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", `{"kind":"user","body":"hello"}`, http.StatusAccepted, &res)

	readerA := openSSE(t, server, "/agents/support/acme", "")
	readerB := openSSE(t, server, "/agents/support/acme", "")
	settled := hasKind(nil, harness.KindSubmissionSettled)
	gotA := collectUntil(t, readerA, 15*time.Second, settled)
	gotB := collectUntil(t, readerB, 15*time.Second, settled)

	if len(gotA) != len(gotB) {
		t.Fatalf("reader A saw %d frames, reader B %d", len(gotA), len(gotB))
	}
	for i := range gotA {
		if gotA[i].ID != gotB[i].ID {
			t.Fatalf("frame %d differs: A=%s B=%s", i, gotA[i].ID, gotB[i].ID)
		}
	}
}

// A reader attaching to an already-settled conversation gets the full replay
// and a cleanly closed stream.
func TestSettledConversationReplaysThenCloses(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("all done").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme?wait=true", `{"kind":"user","body":"hello"}`, http.StatusOK, &res)

	live := openSSE(t, server, "/agents/support/acme", "")
	var got []sseEvent
	for ev := range live.Events {
		got = append(got, ev)
	}
	if err := <-live.Done; err != nil {
		t.Fatalf("stream ended with error: %v", err)
	}

	all, err := rt.Records(context.Background(), res.ConversationID, "")
	if err != nil {
		t.Fatalf("Records: %v", err)
	}
	if len(got) != len(all) {
		t.Fatalf("replay = %d frames, want %d records", len(got), len(all))
	}
}
