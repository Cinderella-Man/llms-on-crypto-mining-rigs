#!/bin/bash
# Tests of soak_chain.sh's control flow on any Linux machine: the real script runs in a temporary folder against
# stand-ins for climb.py, soak.py and model_check.sh that record how they were called. Nothing is set or run on a card.
#
#   bash bench/mem_offset/test/test_chain.sh
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d /tmp/chain-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
cp "$KIT/soak_chain.sh" "$T/"
cat > "$T/climb.py" <<'EOF'
import os, sys
a = sys.argv[1:]
open(os.environ["CALLS"], "a").write("climb " + " ".join(a) + "\n")
if a[0] == "start-check":
    print("start check: PASS" if os.environ.get("STUB_STARTCHECK", "0") == "0" else "start check: FAIL")
    sys.exit(int(os.environ.get("STUB_STARTCHECK", "0")))
if a[0] == "apply":
    open(os.path.join(os.environ["WORK"], "APPLY_DONE"), "w").write(os.environ.get("STUB_APPLY", "exit 0") + "\n")
EOF
cat > "$T/soak.py" <<'EOF'
import os, sys
open(os.environ["CALLS"], "a").write("soak " + " ".join(sys.argv[1:]) + "\n")
open(os.path.join(os.environ["WORK"], "SOAK_DONE"), "w").write(os.environ.get("STUB_SOAK", "exit 0 rounds 100 differs 0") + "\n")
EOF
cat > "$T/model_check.sh" <<'EOF'
echo "model TAG=$TAG ROUNDS=$ROUNDS MINUTES=${MINUTES:-}" >> "$CALLS"
if [ "$TAG" = stock ]; then exit "${STUB_MODEL_STOCK:-0}"; else exit "${STUB_MODEL_PROFILE:-0}"; fi
EOF
fails=0
run() {   # run NAME [VAR=value ...]: one chain run in a fresh work folder
  name=$1; shift
  export WORK="$T/work-$name" CALLS="$T/calls-$name"
  : > "$CALLS"
  env PROFILE=p1.json CHAIN_NOW=10:00 "$@" bash "$T/soak_chain.sh" > "$T/console-$name" 2>&1
  DONE=$(cat "$WORK/CHAIN_DONE" 2>/dev/null)
}
want() {  # want WHAT GREP-PATTERN FILE-OR-TEXT...
  if ! grep -q -- "$2" <<< "$3"; then echo "FAIL $name: $1 (wanted '$2' in: $3)"; fails=$((fails + 1)); fi
}
wantnot() {
  if grep -q -- "$2" <<< "$3"; then echo "FAIL $name: $1 (did not want '$2' in: $3)"; fails=$((fails + 1)); fi
}

run all-pass MODEL=m.gguf LADDER="p2.json p3.json"
want "ends done" "^done: apply, the soak and the model" "$DONE"
want "the ladder reaches soak.py" "^soak --minutes 30 --keep-going --profile p1.json --ladder p2.json p3.json$" "$(cat "$CALLS")"
want "the stock reference first" "^model TAG=stock ROUNDS=2" "$(sed -n 2p "$CALLS")"
want "the model with the offsets for 30 min" "^model TAG=profile ROUNDS=2 MINUTES=30$" "$(cat "$CALLS")"
wantnot "no reset when everything passes" "reset-all" "$(cat "$CALLS")"

run soak-differs MODEL=m.gguf LADDER="p2.json" STUB_SOAK="exit 1 rounds 90 differs 2 left 7"
want "says the soak did not pass" "the soak did not pass (exit 1 rounds 90 differs 2 left 7)" "$DONE"
want "points at soaked.json" "soak/soaked.json" "$DONE"
want "points at use.json" "soak/use.json, the next fresh boot's profile" "$DONE"
want "every offset to 0" "^climb reset-all$" "$(cat "$CALLS")"
wantnot "no model check after a failed soak" "TAG=profile" "$(cat "$CALLS")"

run model-fails MODEL=m.gguf STUB_MODEL_PROFILE=1
want "says the model failed" "the model with the offsets failed (exit 1): every offset 0" "$DONE"
want "every offset to 0" "^climb reset-all$" "$(cat "$CALLS")"

run too-late MODEL=m.gguf LADDER="p2.json p3.json" CHAIN_NOW=20:20
want "no soak that would end after END_BY" "^no soak: it would end after 21:50" "$DONE"
wantnot "soak.py not started" "^soak " "$(cat "$CALLS")"

run fits-exactly MODEL=m.gguf LADDER="p2.json p3.json" CHAIN_NOW=20:15
want "a 95-minute ladder from 20:15 runs" "^soak " "$(cat "$CALLS")"

run model-too-late MODEL=m.gguf CHAIN_NOW=21:10
want "a 35-minute soak from 21:10 runs" "^soak " "$(cat "$CALLS")"
want "the 45-minute model check does not" "no model check: it would end after 21:50" "$DONE"
wantnot "the model is not loaded with the offsets" "TAG=profile" "$(cat "$CALLS")"

run apply-fails STUB_APPLY="exit 1"
want "says apply failed" "^apply failed (exit 1)" "$DONE"
wantnot "soak.py not started" "^soak " "$(cat "$CALLS")"

run not-fresh STUB_STARTCHECK=1 MODEL=m.gguf
want "stops at the start check" "^start check failed: stop" "$DONE"
wantnot "nothing loaded" "^model " "$(cat "$CALLS")"
wantnot "nothing applied" "^climb apply" "$(cat "$CALLS")"

run no-model
want "a soak without a model check" "^the soak passed; no model reference (skipped)" "$DONE"
want "plain soak arguments" "^soak --minutes 30 --keep-going --profile p1.json$" "$(cat "$CALLS")"

run no-reference MODEL=m.gguf STUB_MODEL_STOCK=1
want "the soak still runs without a reference" "^soak " "$(cat "$CALLS")"
want "and says so" "no model reference (1)" "$DONE"

# R201: an env file that exports the rung's MINUTES=30 made the stock reference run 30 minutes, not two rounds
run stock-own-minutes MODEL=m.gguf MINUTES=30 LADDER="p2.json"
want "the stock reference: two rounds, no minutes" "^model TAG=stock ROUNDS=2 MINUTES=0$" "$(cat "$CALLS")"
want "the rungs keep their minutes" "^soak --minutes 30 " "$(cat "$CALLS")"
want "the model with the offsets keeps MODEL_MINUTES" "^model TAG=profile ROUNDS=2 MINUTES=30$" "$(cat "$CALLS")"

run stock-minutes-set MODEL=m.gguf MINUTES=30 STOCK_MINUTES=20
want "a longer stock control is its own setting" "^model TAG=stock ROUNDS=2 MINUTES=20$" "$(cat "$CALLS")"

run stock-too-late MODEL=m.gguf STOCK_MINUTES=60 CHAIN_NOW=20:50
wantnot "no stock control that would end after END_BY" "TAG=stock" "$(cat "$CALLS")"
want "the soak still runs" "^soak " "$(cat "$CALLS")"
want "and no model check without the reference" "no model reference (too late)" "$DONE"

run failures FAILURES=earlier.json LADDER="p2.json"
want "the earlier failures reach soak.py" "^soak --minutes 30 --keep-going --profile p1.json --failures earlier.json --ladder p2.json$" "$(cat "$CALLS")"

if [ "$fails" = 0 ]; then echo "test_chain: 14 runs, all checks pass"; else echo "test_chain: $fails check(s) FAILED"; exit 1; fi
