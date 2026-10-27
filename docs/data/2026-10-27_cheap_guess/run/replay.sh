#!/usr/bin/env bash
# The MTP head with a cheaper guess, paired: recorded agent-check traces replayed to the release line with and without
# the head, through bench/agent_check/release_check.sh's replay mode (the same prompts, a fixed seed per request and the
# recorded reply lengths). Pipelined decoding is exact, so both servers generate the same tokens and the ratio of ms
# per generated token is the head's effect alone, with a 95 % interval over traces; llama-server's guess counters give
# the head's acceptance as served. Both arms run the cheap-guess build with the same CUDA setting (COMMON_ENV:
# GGML_CUDA_MMVQ_MAX_COLS=4 also moves Flint's own 5-8-token products to MMQ, so the release-line arm gets it too); the
# head arm adds DRAFT_ENV. Run from the repository root with nothing else on the rig, after variants.sh put the build
# in RIG_DIR.
# Env: ENV_FILE (the machine values, as in bench/agent_check/release.env.example; MODEL = Flint Q8_0 on the rig),
# RIG_DIR, NAME (the build), HEAD (the head's GGUF on the rig), REPLAY_LOG (a recorded arm's proxy.jsonl on the
# client), CLIENT_ROOT (results folder on the client), OUT; optional REPLAY_TRACES (default: the six traces the answer
# store's replay used), ARMS_LIST (default "stair base"), COMMON_ENV, DRAFT_ENV (default LLAMA_DRAFT_HEAD=q8_0).
set -uo pipefail
: "${ENV_FILE:?}" "${RIG_DIR:?}" "${NAME:?}" "${HEAD:?}" "${REPLAY_LOG:?}" "${CLIENT_ROOT:?}"
LIST=${ARMS_LIST:-stair base}
O=${OUT:-replay}
COMMON_ENV=${COMMON_ENV-GGML_CUDA_MMVQ_MAX_COLS=4}
DRAFT_ENV=${DRAFT_ENV-LLAMA_DRAFT_HEAD=q8_0}
B=$RIG_DIR/$NAME/bin
REL="-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"
set -a; . "$ENV_FILE"; set +a
export REPLAY_LOG REPLAY_REF=base
export REPLAY_TRACES=${REPLAY_TRACES:-rel/bugfix-javascript-space-age/a1/t1,rel/bugfix-javascript-phone-number/a1/t1,rel/bugfix-javascript-complex-numbers/a1/t1,rel/bugfix-python-affine-cipher/a1/t1,rel/bugfix-python-list-ops/a1/t1,rel/bugfix-python-scale-generator/a1/t1}
ssh "$RIG" "test -x $B/llama-server" || { echo "the build is not on the rig"; exit 1; }
# the environments reach llama-server through wrappers next to it (removed at the end)
for w in base:"$COMMON_ENV" stair:"$COMMON_ENV $DRAFT_ENV"; do
    ssh "$RIG" "printf '#!/bin/sh\nexec env ${w#*:} $B/llama-server \"\$@\"\n' > $B/llama-server-${w%%:*} && chmod +x $B/llama-server-${w%%:*}"
done
steps='prep run report collect'
for c in $LIST; do
    case $c in
        base)  REL_ARGS="$REL" ;;
        stair) REL_ARGS="$REL --pipe-guess staircase --pipe-draft $HEAD" ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: $B/llama-server-$c REL_ARGS='$REL_ARGS' $(date +%T)"
    REL_BIN=$B/llama-server-$c REPLAY_LABEL=$c ARMS=rel CTX=262144 REL_ARGS="$REL_ARGS" \
        CLIENT_OUT=$CLIENT_ROOT/out-cheap-guess-replay STEPS=$steps bench/agent_check/release_check.sh "$O" \
        || echo "=== $c failed ($?)"
    steps='run report collect'
    # the head's guesses as served (llama-server's per-request counters, in the rig log the step collected)
    [ -r "$O/rig/server-rel.log" ] && grep "pipelined = " "$O/rig/server-rel.log" | sed 's/.*pipelined = *//' | \
        awk '{r+=$3; w+=$5; dg+=$12; dr+=$14} END {printf "so far: checked %d, right %d; drafter %d, right %d\n", r+w, r, dg, dr}'
done
ssh "$RIG" "rm -f $B/llama-server-base $B/llama-server-stair"
echo "=== done $(date +%T)"
