#!/usr/bin/env bash
# The v2.0 release setup search on agentic speed (2026-10-26): one bench/agent_check/release_check.sh run (the 12
# bug-fix tasks, one attempt each) per candidate server setup, one arm per run, results in OUT/<candidate>/. Run from
# the repository root with nothing else on the rig. Env: ENV_FILE (the machine values, as in
# bench/agent_check/release.env.example: REL_BIN = the release build 37e717ff6, STOCK_BIN = stock d77256999, budgets),
# Q8_MODEL, K2H_MODEL, UD_MODEL, UD5_MODEL (GGUF paths on the rig), CLIENT_ROOT (results folder on the client), CANDS, OUT.
# Candidates (the release engine at -c 262144 unless noted; Q8_0 always with -ts 6,6,5,5,6,5,5,6,6):
#   page       Q8_0, the model page's line: -b 512 -ub 128 (the most that fits at 262k) --pipe-depth 3
#   cache      page + the agent prompt cache: --prefix-cache (shared system and tool prefix kept across sessions)
#              and --checkpoint-gen
#   stock      stock llama.cpp, Q8_0, -c 65536 (its host-side mask does not fit the 4 GB host at 262k), -b 2048 -ub 512
#   noguess    page without --pipe-depth (pipelined speculation's on/off test)
#   d2, d4     page at --pipe-depth 2 and 4
#   k2h        Flint-Pascal-Q5 (recipe K2h-r16), -b 512 -ub 128 --pipe-depth 3, no -ts (layers by count)
#   k2h-ub512  the same with -b 2048 -ub 512 (the server was killed while loading: host RAM)
#   k2h-d4     Flint-Pascal-Q5 at --pipe-depth 4: the release command line
#   q8-d3b, q8-d4b   Q8_0 at depth 3 and 4 again, back to back (depth 4 wins a tie)
#   ud-d4, ud5-d4    Unsloth's UD-Q4_K_XL and UD-Q5_K_XL on the release command line
#   stock-ud, stock-k2h   stock llama.cpp, -c 65536 -b 2048 -ub 512, with UD-Q4_K_XL and with Flint-Pascal-Q5
# The v1 engine's run is agent_v1.sh.
set -uo pipefail
: "${ENV_FILE:?the machine values for release_check.sh}" "${CLIENT_ROOT:?results folder on the client}"
O=${OUT:-agent}
mkdir -p "$O"
set -a; . "$ENV_FILE"; set +a
CANDS=${CANDS:-page cache stock noguess d2 d4 k2h k2h-ub512 k2h-d4 q8-d3b q8-d4b ud-d4 stock-ud stock-k2h ud5-d4}
Q8="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6"
LINE="-b 512 -ub 128"
CACHE_DIR=${CACHE_DIR:-/tmp/agent-pfx}
for c in $CANDS; do
  ARMS=rel; CTX=262144; M=${Q8_MODEL:-}; REL_ARGS=""; STOCK_ARGS=""
  case $c in
    page|q8-d3b) REL_ARGS="$Q8 --pipe-depth 3" ;;
    cache)     REL_ARGS="$Q8 --pipe-depth 3 --prefix-cache $CACHE_DIR-$c --checkpoint-gen" ;;
    stock)     ARMS=stock; CTX=65536; STOCK_ARGS="-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6" ;;
    noguess)   REL_ARGS="$Q8" ;;
    d2)        REL_ARGS="$Q8 --pipe-depth 2" ;;
    d4|q8-d4b) REL_ARGS="$Q8 --pipe-depth 4" ;;
    k2h)       M=${K2H_MODEL:-}; REL_ARGS="$LINE --pipe-depth 3" ;;
    k2h-ub512) M=${K2H_MODEL:-}; REL_ARGS="-b 2048 -ub 512 --pipe-depth 3" ;;
    k2h-d4)    M=${K2H_MODEL:-}; REL_ARGS="$LINE --pipe-depth 4" ;;
    ud-d4)     M=${UD_MODEL:-}; REL_ARGS="$LINE --pipe-depth 4" ;;
    ud5-d4)    M=${UD5_MODEL:-}; REL_ARGS="$LINE --pipe-depth 4" ;;
    stock-ud)  ARMS=stock; CTX=65536; M=${UD_MODEL:-}; STOCK_ARGS="-b 2048 -ub 512" ;;
    stock-k2h) ARMS=stock; CTX=65536; M=${K2H_MODEL:-}; STOCK_ARGS="-b 2048 -ub 512" ;;
    *) echo "unknown candidate $c"; continue ;;
  esac
  [ -n "$M" ] || { echo "=== $c: its model path is not set"; continue; }
  echo "=== $c: MODEL=$M ARMS=$ARMS CTX=$CTX REL_ARGS='$REL_ARGS' STOCK_ARGS='$STOCK_ARGS' $(date +%T)"
  [ $c = cache ] && ssh "$RIG" "rm -rf $CACHE_DIR-$c && mkdir -p $CACHE_DIR-$c"
  MODEL=$M ARMS=$ARMS CTX=$CTX REL_ARGS="$REL_ARGS" STOCK_ARGS="$STOCK_ARGS" \
    CLIENT_OUT=$CLIENT_ROOT/out-$c STEPS='prep run report collect' \
    bench/agent_check/release_check.sh "$O/$c" || echo "=== $c failed ($?)"
  [ $c = cache ] && ssh "$RIG" "rm -rf $CACHE_DIR-$c"
  [ -r "$O/$c/table.md" ] && grep -E "^\| (rel|stock) " "$O/$c/table.md" | sed "s/^/$c /"
  [ -r "$O/$c/compare.md" ] && grep -E "^\| (rel|stock) \| [0-9]+ / " "$O/$c/compare.md" | sed "s/^/$c speeds /"
done
echo "=== agent search done $(date +%T)"
