#!/usr/bin/env bash
# Upstream fixes, the draws reused after a speculative checkpoint restore: the grammar arm of the statistical test of
# 2026-11-04_block_verify (its run/stat.sh, serve_lib.sh, stat_tool.gbnf and stat_check.py, mounted at /r) on upstream
# master and on master with the fix. Sampled decoding (temperature 1, top-k 8) of a short JSON tool call under a grammar
# with 30 outcomes, N_STAT_G requests per arm, each arm its own seeds:
#   gram-nospec   master, plain decoding: the reference
#   gram-rej      master, draft-simple (--spec-draft-n-max 4, probabilistic draft sampling, --spec-draft-p-min 0):
#                 token-by-token rejection with the restore as master has it
#   gram-rejkeep  the fix, the same draft settings
# The target (random-weight qwen4exp, a hybrid with recurrent layers, GPT-2's vocabulary) cannot roll back part of a
# draft longer than its rollback snapshots, so with draft-simple every partly accepted draft restores the checkpoint.
# Usage (inside a container with curl, jq and python3; $THREADS threads per arm):
#   B_MASTER=<llama-server> B_FIX=<llama-server> ST=<target gguf> SD=<draft gguf> [N_STAT_G=1500] [PORT=18291]
#   [STAT_THREADS=2] bash stat.sh; then python3 -I /r/stat_check.py /c/stat
# Output: /c/stat/gram-<arm>.tsv (seed, reply, drafted, accepted) and the servers' logs
set -u
OUT=/c/stat
mkdir -p "$OUT"
T=$ST
DR=$SD
THREADS=${STAT_THREADS:-2}
PORT=${PORT:-18291}
N_STAT_G=${N_STAT_G:-1500}
. /r/serve_lib.sh
P_TOOL="Call one tool as JSON:"

gram_arm() { # <arm> <seed offset> <port> <bin> <server args> [env...]
    local arm=$1 s0=$2 PORT=$3 bin=$4 sargs=$5; shift 5
    serve "gram-$arm" "$bin" "$sargs" "$@" || return 1
    local t0=$SECONDS
    for s in $(seq 1 "$N_STAT_G"); do
        complete "$OUT/r-gram-$arm.json" "$P_TOOL" 32 1.0 $((s0 + s)) /r/stat_tool.gbnf
        jq -r --argjson s "$s" '[$s, (.content | gsub("[\t\n]"; " ")), (.timings.draft_n // 0), (.timings.draft_n_accepted // 0)] | @tsv' "$OUT/r-gram-$arm.json"
    done > "$OUT/gram-$arm.tsv"
    echo "gram-$arm: $(wc -l < "$OUT/gram-$arm.tsv") requests in $((SECONDS - t0)) s"
    # LLAMA_TRACE=1: a line per verified draft, marked when it restored the checkpoint
    echo "gram-$arm: verified drafts $(grep -c "draft tokens" "$OUT/gram-$arm.server.log"), restores $(grep -c "(restore checkpoint)" "$OUT/gram-$arm.server.log")"
    stop "gram-$arm"
}

# the seed offsets of 2026-11-04_block_verify's arms of the same names
gram_arm nospec  4000000  $((PORT + 0)) "$B_MASTER" "" &
gram_arm rej     5000000  $((PORT + 1)) "$B_MASTER" "$DRAFT" LLAMA_TRACE=1 &
gram_arm rejkeep 12000000 $((PORT + 2)) "$B_FIX"    "$DRAFT" LLAMA_TRACE=1 &
wait
