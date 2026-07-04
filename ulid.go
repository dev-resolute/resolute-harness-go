package harness

import (
	crand "crypto/rand"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
)

var (
	ulidMu      sync.Mutex
	ulidEntropy = ulid.Monotonic(crand.Reader, 0)
)

// newULID returns a fresh ULID string. ULIDs are lexicographically ordered
// by generation time, which is what lets record IDs double as SSE offsets;
// the shared monotonic source keeps same-millisecond IDs strictly increasing.
func newULID() string {
	ulidMu.Lock()
	defer ulidMu.Unlock()
	return ulid.MustNew(ulid.Timestamp(time.Now()), ulidEntropy).String()
}
