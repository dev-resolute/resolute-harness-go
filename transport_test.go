package harness_test

import (
	"bufio"

	"encoding/json"

	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/dev-resolute/resolute-llm-go/mock"

	harness "github.com/dev-resolute/resolute-harness-go"
)

// postDispatch POSTs a dispatch body and decodes the JSON response into out
// (any JSON-decodable pointer; nil skips decoding).
func postDispatch(t *testing.T, server *httptest.Server, path string, body string, wantStatus int, out any) {
	t.Helper()
	resp, err := http.Post(server.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != wantStatus {
		t.Fatalf("POST %s status = %d, want %d", path, resp.StatusCode, wantStatus)
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			t.Fatalf("decode POST %s response: %v", path, err)
		}
	}
}

// sseEvent is one parsed SSE frame.
type sseEvent struct {
	ID    string
	Event string
	Data  string
}

// readSSE fetches the conversation stream and parses all frames until EOF.
func readSSE(t *testing.T, server *httptest.Server, path string, lastEventID string) []sseEvent {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, server.URL+path, nil)
	if err != nil {
		t.Fatalf("build GET %s: %v", path, err)
	}
	if lastEventID != "" {
		req.Header.Set("Last-Event-ID", lastEventID)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s status = %d, want 200", path, resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("GET %s content type = %q, want text/event-stream", path, ct)
	}

	var events []sseEvent
	var cur sseEvent
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			if cur.ID != "" || cur.Data != "" {
				events = append(events, cur)
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
	return events
}

func TestHTTPDispatchReturns202AndWorkProceeds(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("async answer").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res harness.DispatchResult
	postDispatch(t, server, "/agents/support/acme", `{"kind":"user","body":"hello"}`, http.StatusAccepted, &res)
	if res.SubmissionID == "" || res.ConversationID == "" {
		t.Fatalf("202 body missing ids: %+v", res)
	}

	// Work proceeds after the response: the settled record eventually shows
	// up in the replayed stream.
	deadline := time.After(10 * time.Second)
	for {
		events := readSSE(t, server, "/agents/support/acme", "")
		if hasEventKind(events, harness.KindSubmissionSettled) {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("submission never settled; stream events: %+v", events)
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func TestHTTPDispatchWaitTrueReturnsSettledResult(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("blocking answer").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	var res struct {
		harness.DispatchResult
		harness.SettledPayload
	}
	postDispatch(t, server, "/agents/support/acme?wait=true", `{"kind":"user","body":"hello"}`, http.StatusOK, &res)
	if res.Status != harness.SettledSucceeded {
		t.Fatalf("wait=true settled status = %q (error %q), want %q", res.Status, res.Error, harness.SettledSucceeded)
	}
	if res.SubmissionID == "" || res.ConversationID == "" {
		t.Fatalf("wait=true body missing ids: %+v", res)
	}
}

func TestHTTPStreamReplaysFromLastEventID(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	provider.OnAny().RespondText("replayed answer").Add()
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	postDispatch(t, server, "/agents/support/acme?wait=true", `{"kind":"user","body":"hello"}`, http.StatusOK, nil)

	full := readSSE(t, server, "/agents/support/acme", "")
	if len(full) < 3 {
		t.Fatalf("full replay has %d events, want at least 3: %+v", len(full), full)
	}
	// Every frame is a well-formed record whose id/event match the envelope.
	for _, ev := range full {
		var rec harness.Record
		if err := json.Unmarshal([]byte(ev.Data), &rec); err != nil {
			t.Fatalf("frame data is not a record: %v (%s)", err, ev.Data)
		}
		if rec.ID != ev.ID || string(rec.Kind) != ev.Event {
			t.Fatalf("frame id/event = %s/%s, record envelope = %s/%s", ev.ID, ev.Event, rec.ID, rec.Kind)
		}
	}

	offset := full[1].ID
	partial := readSSE(t, server, "/agents/support/acme", offset)
	if want := len(full) - 2; len(partial) != want {
		t.Fatalf("replay from offset %s returned %d events, want %d", offset, len(partial), want)
	}
	if partial[0].ID != full[2].ID {
		t.Fatalf("replay from offset starts at %s, want %s", partial[0].ID, full[2].ID)
	}
}

func TestHTTPStreamUnknownConversation404s(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	resp, err := http.Get(server.URL + "/agents/support/nobody-here")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
}

func TestHTTPDispatchValidation(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	tests := []struct {
		name       string
		path       string
		body       string
		wantStatus int
	}{
		{name: "unknown agent", path: "/agents/nope/x", body: `{"kind":"user","body":"hi"}`, wantStatus: http.StatusNotFound},
		{name: "unknown kind", path: "/agents/support/x", body: `{"kind":"carrier-pigeon","body":"hi"}`, wantStatus: http.StatusBadRequest},
		{name: "missing body", path: "/agents/support/x", body: `{"kind":"user"}`, wantStatus: http.StatusBadRequest},
		{name: "malformed json", path: "/agents/support/x", body: `{"kind"`, wantStatus: http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			postDispatch(t, server, tt.path, tt.body, tt.wantStatus, nil)
		})
	}
}

func TestHealthz(t *testing.T) {
	t.Parallel()
	provider := mock.New("mock")
	rt := newTestRuntime(t, provider)
	server := httptest.NewServer(rt.Handler())
	t.Cleanup(server.Close)

	resp, err := http.Get(server.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/healthz status = %d, want 200", resp.StatusCode)
	}
}

func hasEventKind(events []sseEvent, kind harness.RecordKind) bool {
	for _, ev := range events {
		if ev.Event == string(kind) {
			return true
		}
	}
	return false
}
