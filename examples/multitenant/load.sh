#!/usr/bin/env bash
# Demo load for examples/multitenant: four ?wait=true prompts fired at once.
#
#   acme/support   gets TWO prompts  -> the second queues behind the first
#   acme/billing   gets one prompt   -> settles in parallel with support
#   globex/support gets one prompt   -> settles in parallel with acme
#
# Expected wall-clock (local provider thinks ~700ms per prompt): three
# prompts settle around 0.7s and "second in support" around 1.4s — same
# session serializes, everything else runs concurrently.
set -euo pipefail

addr="${ADDR:-localhost:8489}"
start=$(date +%s.%N)

fire() { # fire <instance> <session> <body>
  local instance=$1 session=$2 body=$3
  local reply elapsed
  reply=$(curl -s "http://$addr/agents/concierge/$instance?wait=true" \
    -d "{\"kind\":\"user\",\"body\":\"$body\",\"session\":\"$session\"}")
  elapsed=$(echo "$(date +%s.%N) - $start" | bc)
  printf 'settled %-14s %-8s %-20s at %4.1fs  %s\n' "$instance" "$session" "\"$body\"" "$elapsed" \
    "$(echo "$reply" | grep -o '"status":"[a-z]*"')"
}

fire acme   support "first in support"   &
fire acme   billing "billing question"   &
fire globex support "globex question"    &
sleep 0.15 # admit "second" after "first" so the queue order matches the labels
fire acme   support "second in support"  &
wait
