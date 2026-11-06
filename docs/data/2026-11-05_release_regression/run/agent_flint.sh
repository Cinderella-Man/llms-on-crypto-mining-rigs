#!/usr/bin/env bash
# Release test T2 (2026-11-05), step 2: the agent check (bench/agent_check/release_check.sh: a coding agent fixing 12
# small bugs with real tool runs, one attempt each, 12-minute cap) with Flint Q8_0 on nine cards at 262,144 tokens, on
# the v2 release line: -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 (release_check.sh adds -c, -np 1, -bs,
# --ctx-checkpoints 4, --cache-ram 0, Flint's sampling and the rest). Arms, in ARMS_LIST order:
#   v3base  BIN (the v3 build) with no variables: every switch at its default
#   v3add   BIN with v3's measured additions: LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 (Q260, Q263)
#           and GGML_CUDA_KEEPALIVE=decode (C533)
#   v2ref   V2_BIN (the v2.0 build) with no variables, a same-day control
# and the side-by-side, one change each against v3base:
#   ka      GGML_CUDA_KEEPALIVE=decode alone (C533)
#   reread  LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 alone (Q260, Q263)
#   staged  GGML_CUDA_HOP_STAGED=1 alone: the staged hop, exact on F32 hops (C557)
#   store   --pipe-store STORE: the answer store of the pipelined guesser (R2b)
#   neg     -t 1 -tb 1: the "Negligible gains" members that apply to Flint at run time (the scheduler copies are a
#           build option, 4 in this build)
#   stock   STOCK_BIN (stock llama.cpp, the v3 build's upstream base) on the same split and batches, no pipelined
#           speculation (a fork option), at STOCK_CTX (262144); stock64k the same at 65,536 tokens; STOCK_LINE
#           replaces the line (stock_tune.sh's pick)
# Run from the repository root with nothing else on the rig.
# Env: the harness's machine values (RIG, CLIENT, RIG_PORT, RELAY_PORT, CLIENT_PORT, PROXY_PORT, CLIENT_DIR, MODEL,
#   RIG_LOGDIR, as in bench/agent_check/release.env.example), BIN (and V2_BIN for v2ref) on the rig, CLIENT_ROOT (the
#   results folder on the client), OUT (local), STORE (the store file on the rig, for store); optional ARMS_LIST
#   ("v3base v3add"), BUDGET_MIN_rel (40), STEPS_FIRST (the first arm's steps, default "prep run report collect"),
#   REL_LINE (replaces the release line for every arm but stock, e.g. "-b 512 -ub 128 --pipe-depth 4" for the Q5 files,
#   which v2 ran with layers by count)
set -uo pipefail
: "${BIN:?}" "${CLIENT_ROOT:?}" "${OUT:?}"
LIST=${ARMS_LIST:-v3base v3add}
REL=${REL_LINE:-"-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"}
export TIME_CAP_MIN=${TIME_CAP_MIN:-12} BUDGET_MIN_rel=${BUDGET_MIN_rel:-40}
mkdir -p "$OUT"
steps="prep run report collect"
for c in $LIST; do
    b=$BIN; args=$REL; ctx=262144
    case $c in
        v3base) ENVS="" ;;
        v3add)  ENVS="LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 GGML_CUDA_KEEPALIVE=decode" ;;
        v2ref)  ENVS=""; b=${V2_BIN:?} ;;
        ka)     ENVS="GGML_CUDA_KEEPALIVE=decode" ;;
        reread) ENVS="LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096" ;;
        staged) ENVS="GGML_CUDA_HOP_STAGED=1" ;;
        store)  ENVS=""; args="$REL --pipe-store ${STORE:?}" ;;
        neg)    ENVS=""; args="$REL -t 1 -tb 1" ;;
        stock)  ENVS=""; b=${STOCK_BIN:?}; args="${STOCK_LINE:--b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6}"; ctx=${STOCK_CTX:-262144} ;;
        stock64k) ENVS=""; b=${STOCK_BIN:?}; args="${STOCK_LINE:--b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6}"; ctx=65536 ;;
        *) echo "unknown arm $c"; continue ;;
    esac
    echo "=== $c: REL_ENV='$ENVS' $b $(date -Is)"
    ARMS=rel CTX=$ctx REL_BIN="$b" REL_ENV="$ENVS" REL_ARGS="$args" CLIENT_OUT="$CLIENT_ROOT/out-$c" \
        STEPS="${STEPS_FIRST:-$steps}" bench/agent_check/release_check.sh "$OUT/$c" || echo "=== $c failed ($?)"
    steps="run report collect"; STEPS_FIRST=""
    [ -r "$OUT/$c/table.md" ] && grep -E "^\| rel " "$OUT/$c/table.md" | sed "s/^/$c /"
    [ -r "$OUT/$c/compare.md" ] && grep -E "^\| rel \| [0-9]+ / " "$OUT/$c/compare.md" | sed "s/^/$c speeds /"
done
echo "=== done $(date -Is)"
