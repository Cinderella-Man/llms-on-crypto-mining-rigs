#!/usr/bin/env bash
# C662 (Q1150): the whole test-backend-ops through the compute 6.1 PTX on one card, in five shards by op name that run
# side by side (one process each), with the fork's CUDA paths off and forced on (the settings of
# docs/data/2026-11-03_upstream_rebase/run/check.sh). One process took more than two hours on a busy machine; the
# shards cover every op once: MUL_MAT, the other MUL_MAT_* ops, FLASH_ATTN*, the convolutions and pools, and the rest.
# An -o entry with a parenthesis is read as a whole test case, so the shards' regexes have none.
#   MODE=support only lists every case's support on the card (no computing): the shards' counts must add up to the
#   count of a run without -o.
# Usage (on the build machine): GPU=<uuid> TREE=<unpacked rig tarball> OUT=<dir> [ARMS='off on'] [MODE=test|support]
#   [OFF_PARAMS=<-p regex for the off arm>] [TIMEOUT=14400] [DOCKER_ARGS=...] bash tbo_shards.sh
# Output: OUT/<arm>-<shard>.log, OUT/summary.txt (one line per arm and shard); done when OUT/done exists
set -uo pipefail
: "${GPU:?}" "${TREE:?}" "${OUT:?}"
MODE=${MODE:-test}
ARMS=${ARMS:-off on}
TIMEOUT=${TIMEOUT:-14400}
# the off arm leaves out the cases with a repeated expert id: with the fork's kernels off the first of them stops the run (Q1222)
OFF_PARAMS=${OFF_PARAMS-'^((?!dup=1).)*$'}
mkdir -p "$OUT/nvcache"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
OFF="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_FATTN_TILE_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0 GGML_CUDA_NORM_Q8=0"
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_NORM_Q8=1"
REST='^[^MFCIP],^M[^U],^MU[^L],^MUL$,^MUL[^_],^MUL_[^M],^MUL_M[^A],^MUL_MA[^T],^F[^L],^FL[^A],^FLA[^S],^C[^O],^CO[^N],^CON[^V],^I[^M],^IM[^2],^P[^O],^PO[^O]'
shard_filter() {
    case $1 in
        mulmat)  echo '^MUL_MAT$' ;;
        mulmatx) echo '^MUL_MAT_' ;;
        fattn)   echo '^FLASH_ATTN' ;;
        conv)    echo '^CONV,^IM2COL,^POOL' ;;
        rest)    echo "$REST" ;;
        all)     echo '' ;;
    esac
}
run() { # <arm> <shard>
    local arm=$1 sh=$2 e f
    e="$OFF"; [ "$arm" = on ] && e="$ON"
    f=$(shard_filter "$sh")
    local args=(-b CUDA0)
    [ "$MODE" = support ] && args=(support -b CUDA0)
    [ -n "$f" ] && args+=(-o "$f")
    [ "$arm" = off ] && [ -n "$OFF_PARAMS" ] && args+=(-p "$OFF_PARAMS")
    local ee=()
    for kv in $e; do ee+=(-e "$kv"); done
    # shellcheck disable=SC2086
    docker run --rm --name "c662-tbo-$MODE-$arm-$sh" --cpu-shares=256 ${DOCKER_ARGS:-} --gpus "\"device=$GPU\"" -v "$TREE:$TREE:ro" -v "$OUT:/k" \
        -e CUDA_CACHE_PATH=/k/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 "${ee[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 \
        timeout "$TIMEOUT" "$TREE/bin/test-backend-ops" "${args[@]}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$arm-$sh.log"
    echo "$arm $sh: exit ${PIPESTATUS[0]}; $(grep -a -m1 'tests passed' "$OUT/$arm-$sh.log" | sed 's/^ *//'); case lines $(grep -a -c '^  [A-Z_0-9]*(' "$OUT/$arm-$sh.log"); OK $(grep -a -c ': OK' "$OUT/$arm-$sh.log"); FAIL lines $(grep -a -c 'FAIL' "$OUT/$arm-$sh.log"); not supported $(grep -a -c 'not supported' "$OUT/$arm-$sh.log"); $(grep -a -m1 'CUDA error' "$OUT/$arm-$sh.log")" >> "$OUT/summary.txt"
}
for arm in $ARMS; do
    for sh in ${SHARDS:-mulmat mulmatx fattn conv rest}; do
        run "$arm" "$sh" &
    done
done
wait
sort -o "$OUT/summary.txt" "$OUT/summary.txt"
date -Is > "$OUT/done"
