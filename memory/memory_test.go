package memory_test

import (
	"testing"

	harness "github.com/dev-resolute/resolute-harness-go"
	"github.com/dev-resolute/resolute-harness-go/memory"
	"github.com/dev-resolute/resolute-harness-go/storetest"
)

func TestConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) harness.Store {
		return memory.New()
	})
}
