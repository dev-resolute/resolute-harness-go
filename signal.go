package harness

import (
	"encoding/json"

	pi "github.com/dev-resolute/resolute-agent-core-go"
)

// signalToMessage renders a signal into the agent transcript as a custom
// "signal"-typed message whose body is the signal's JSON (type, body,
// sender, tag). agent-core's default LLM conversion surfaces custom types
// as text, so the model sees the participant attributes verbatim and the
// sender stays distinguishable from the agent's principal (ADR-0005).
//
// The template is deliberately minimal and PROVISIONAL: it gets revisited
// when the first real channel adapter (Slack) can validate it.
func signalToMessage(p SignalPayload) pi.Message {
	body, err := json.Marshal(p)
	if err != nil {
		// SignalPayload is a plain data struct; a marshal failure is a
		// programmer error.
		panic("marshal signal payload: " + err.Error())
	}
	return pi.Message{Role: "user", Type: "signal", Body: body}
}

// inputToMessage renders a submission's inbound payload into the transcript
// message that starts its prompt.
func inputToMessage(input DispatchMessage) pi.Message {
	if input.Kind == InboundSignal && input.Signal != nil {
		return signalToMessage(SignalPayload{
			Type:   input.Signal.Type,
			Body:   input.Body,
			Sender: input.Signal.Sender,
			Tag:    input.Signal.Tag,
		})
	}
	return pi.NewText("user", input.Body)
}
