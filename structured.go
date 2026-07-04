package harness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// resultInvalidError reports a final answer that never validated against the
// requested result schema within the feedback budget.
type resultInvalidError struct {
	reason string
}

func (e *resultInvalidError) Error() string {
	return fmt.Sprintf("structured result did not validate against the requested schema: %s", e.reason)
}

// validateStructuredResult checks the final answer text against the JSON
// Schema. It returns the canonical JSON on success, or a human-readable
// reason (fed back to the agent as the corrective turn) on failure.
func validateStructuredResult(schema json.RawMessage, answer string) (json.RawMessage, string) {
	raw, ok := extractJSON(answer)
	if !ok {
		return nil, "the reply is not a JSON document"
	}
	compiled, err := compileSchema(schema)
	if err != nil {
		return nil, fmt.Sprintf("the result schema itself is invalid: %v", err)
	}
	instance, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Sprintf("the reply is not valid JSON: %v", err)
	}
	if err := compiled.Validate(instance); err != nil {
		return nil, err.Error()
	}
	return raw, ""
}

func compileSchema(schema json.RawMessage) (*jsonschema.Schema, error) {
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(schema))
	if err != nil {
		return nil, fmt.Errorf("parse schema: %w", err)
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("inline://result-schema.json", doc); err != nil {
		return nil, fmt.Errorf("add schema resource: %w", err)
	}
	compiled, err := compiler.Compile("inline://result-schema.json")
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}
	return compiled, nil
}

// extractJSON pulls a JSON document out of the answer: the whole trimmed
// text, or the first fenced code block when the model wrapped it in
// markdown.
func extractJSON(answer string) (json.RawMessage, bool) {
	candidates := []string{strings.TrimSpace(answer)}
	if i := strings.Index(answer, "```"); i >= 0 {
		rest := answer[i+3:]
		rest = strings.TrimPrefix(rest, "json")
		if j := strings.Index(rest, "```"); j >= 0 {
			candidates = append(candidates, strings.TrimSpace(rest[:j]))
		}
	}
	for _, c := range candidates {
		if c == "" {
			continue
		}
		if json.Valid([]byte(c)) {
			return json.RawMessage(c), true
		}
	}
	return nil, false
}

// correctiveMessage is the documented default feedback prompt for a
// non-conforming structured result.
func correctiveMessage(reason string, schema json.RawMessage) string {
	return fmt.Sprintf(
		"Your previous reply did not validate against the required JSON schema.\nValidation error: %s\nRequired schema:\n%s\nReply with ONLY a JSON document that validates against the schema — no prose, no code fences.",
		reason, schema)
}
