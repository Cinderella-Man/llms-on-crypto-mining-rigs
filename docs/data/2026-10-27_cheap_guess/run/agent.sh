#!/usr/bin/env bash
# Flint's MTP head with a cheaper guess on the agent check (bench/agent_check/release_check.sh: the pi agent fixing 12
# Exercism bugs, 25-minute budget per arm), Flint Q8_0 on nine cards at 262,144 tokens, the release line of Miner Llama
# v2.0 (-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4) on the cheap-guess build (variants.sh put it in RIG_DIR).
# Run from a machine that reaches the rig and the client over ssh, with nothing else on the rig; results in OUT/<arm>/.
#   base   the release line (--pipe-guess cont)
#   stair  + --pipe-guess staircase --pipe-draft HEAD: the copy lookup, the request's continuations of 2-token or longer
#          contexts, then the head; DRAFT_ENV sets the head's options (LLAMA_DRAFT_*, GGML_CUDA_MMVQ_MAX_COLS)
#   RIG=rig CLIENT=<client ssh alias> CLIENT_WORK=<folder on the client> MODEL=<rig path of Flint Q8_0> \
#   RIG_DIR=<rig folder of the build> NAME=<build name> HEAD=<rig path of the head GGUF> OUT=<local folder> \
#   DRAFT_ENV='LLAMA_DRAFT_HEAD=q8_0 ...' ARMS_LIST="stair base" bash agent.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
LIST=${ARMS_LIST:-stair base}
DRAFT_ENV=${DRAFT_ENV:-}
B=${RIG_DIR:?rig folder of the build}/${NAME:?build name}/bin
HEAD=${HEAD:?rig path of the head GGUF}
O=${OUT:?local results folder}
mkdir -p "$O"
# the harness's machine values (bench/agent_check/release_check.sh)
export RIG=${RIG:-rig} CLIENT=${CLIENT:?ssh alias of the client machine} MODEL=${MODEL:?rig path of Flint Q8_0}
export RIG_PORT=${RIG_PORT:-8091} RELAY_PORT=${RELAY_PORT:-8098} CLIENT_PORT=${CLIENT_PORT:-18391} \
  PROXY_PORT=${PROXY_PORT:-18392}
CW=${CLIENT_WORK:?folder on the client}
export CLIENT_DIR=$CW/bench RIG_LOGDIR=/tmp/agent-check-rel TIME_CAP_MIN=12 BUDGET_MIN_rel=25 CTX=262144
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
ssh "$RIG" "test -x $B/llama-server" || { echo "the build is not on the rig"; exit 1; }
# the head's options reach llama-server through a wrapper next to it (removed at the end)
ssh "$RIG" "printf '#!/bin/sh\nexec env $DRAFT_ENV $B/llama-server \"\$@\"\n' > $B/llama-server-draft && chmod +x $B/llama-server-draft"
for c in $LIST; do
    case $c in
        base)  REL_BIN=$B/llama-server; REL_ARGS="$REL" ;;
        stair) REL_BIN=$B/llama-server-draft; REL_ARGS="$REL --pipe-guess staircase --pipe-draft $HEAD" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_BIN=$REL_BIN REL_ARGS='$REL_ARGS' DRAFT_ENV='$DRAFT_ENV' $(date +%T)"
    REL_BIN=$REL_BIN ARMS=rel REL_ARGS="$REL_ARGS" CLIENT_OUT=$CW/out-cheap-guess-$c \
        STEPS='prep run report collect' "$REPO/bench/agent_check/release_check.sh" "$O/$c" || echo "=== $c failed ($?)"
    [ -r "$O/$c/table.md" ] && grep -E "^\| rel " "$O/$c/table.md" | sed "s/^/$c /"
    # the head's guesses as served (llama-server's per-request timings)
    [ -r "$O/$c/rig/server-rel.log" ] && grep "pipelined = " "$O/$c/rig/server-rel.log" | sed 's/.*pipelined = *//' | awk -v a=$c \
        '{r+=$3; w+=$5; dg+=$12; dr+=$14} END {printf "%s: checked %d, right %d; drafter %d, right %d (%.3f)\n", a, r+w, r, dg, dr, dg?dr/dg:0}'
done
ssh "$RIG" "rm -f $B/llama-server-draft"
echo "=== done $(date +%T)"
