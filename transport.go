package harness

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

// transport is the HTTP surface of the Runtime (ADR-0004):
//
//	POST /agents/{name}/{id}            → 202 {submissionId, conversationId}
//	POST /agents/{name}/{id}?wait=true  → 200 settled result
//	GET  /agents/{name}/{id}            → SSE canonical records (Last-Event-ID = replay offset)
//	GET  /healthz                       → liveness
type transport struct {
	rt  *Runtime
	mux *http.ServeMux
}

func newTransport(rt *Runtime) *transport {
	t := &transport{rt: rt, mux: http.NewServeMux()}
	t.mux.HandleFunc("POST /agents/{name}/{id}", t.handleDispatch)
	t.mux.HandleFunc("GET /agents/{name}/{id}", t.handleStream)
	t.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	return t
}

func (t *transport) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	t.mux.ServeHTTP(w, r)
}

// dispatchBody is the POST request body: the inbound message plus optional
// session and dispatch id.
type dispatchBody struct {
	Kind        InboundKind     `json:"kind"`
	Body        string          `json:"body"`
	Attachments []AttachmentRef `json:"attachments,omitempty"`
	Signal      *SignalMeta     `json:"signal,omitempty"`
	Session     string          `json:"session,omitempty"`
	DispatchID  string          `json:"dispatchId,omitempty"`
}

func (t *transport) handleDispatch(w http.ResponseWriter, r *http.Request) {
	var body dispatchBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("decode request body: %w", err))
		return
	}
	d := Dispatch{
		Agent:      r.PathValue("name"),
		Instance:   InstanceID(r.PathValue("id")),
		Session:    body.Session,
		DispatchID: body.DispatchID,
		Message: DispatchMessage{
			Kind:        body.Kind,
			Body:        body.Body,
			Attachments: body.Attachments,
			Signal:      body.Signal,
		},
	}
	res, err := t.rt.Dispatch(r.Context(), d)
	if err != nil {
		writeError(w, dispatchStatus(err), err)
		return
	}

	if r.URL.Query().Get("wait") == "true" {
		settled, err := t.rt.Wait(r.Context(), res.SubmissionID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, struct {
			DispatchResult
			SettledPayload
		}{res, settled})
		return
	}
	writeJSON(w, http.StatusAccepted, res)
}

func (t *transport) handleStream(w http.ResponseWriter, r *http.Request) {
	name, id := r.PathValue("name"), r.PathValue("id")
	if _, ok := t.rt.agents[name]; !ok {
		writeError(w, http.StatusNotFound, fmt.Errorf("%w: %q", ErrUnknownAgent, name))
		return
	}
	session := r.URL.Query().Get("session")
	if session == "" {
		session = "default"
	}
	key := SessionKey{Agent: name, Instance: InstanceID(id), Session: session}
	conv, err := t.rt.store.GetConversation(r.Context(), key)
	if err != nil {
		if errors.Is(err, ErrConversationNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	afterID := r.Header.Get("Last-Event-ID")
	recs, err := t.rt.Records(r.Context(), conv.ID, afterID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	for _, rec := range recs {
		if err := writeSSE(w, rec); err != nil {
			return
		}
	}
	if flusher != nil {
		flusher.Flush()
	}
	// Replay-only for the walking skeleton; the live tail joins in the
	// streaming slice (HARNESS-4).
}

// writeSSE frames one record as an SSE event: id = record ID, event = kind,
// data = the record JSON. The wire format is the record schema.
func writeSSE(w http.ResponseWriter, rec Record) error {
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal record %s: %w", rec.ID, err)
	}
	_, err = fmt.Fprintf(w, "id: %s\nevent: %s\ndata: %s\n\n", rec.ID, rec.Kind, data)
	return err
}

func dispatchStatus(err error) int {
	switch {
	case errors.Is(err, ErrUnknownAgent):
		return http.StatusNotFound
	case errors.Is(err, ErrInvalidDispatch):
		return http.StatusBadRequest
	case errors.Is(err, ErrDispatchConflict):
		return http.StatusConflict
	default:
		return http.StatusInternalServerError
	}
}

func writeJSON[T any](w http.ResponseWriter, status int, v T) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
