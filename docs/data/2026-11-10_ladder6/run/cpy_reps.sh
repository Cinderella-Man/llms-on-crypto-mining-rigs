#!/usr/bin/env bash
# C662 (Q1150): the CPY operator cases through the compute 6.1 PTX on one card, N times per build, with the fork's CUDA paths
# off and forced on (the settings of docs/data/2026-11-03_upstream_rebase/run/check.sh): does a failing copy case fail
# on the release build too, and how often?
# Usage (on the build machine): GPU=<uuid> BINS='<name>=<tree> ...' OUT=<dir> [N=5] [OP=CPY] bash cpy_reps.sh
set -uo pipefail
: "${GPU:?}" "${BINS:?}" "${OUT:?}"
N=${N:-5}; OP=${OP:-CPY}
mkdir -p "$OUT/nvcache"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
OFF="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_FATTN_TILE_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0 GGML_CUDA_NORM_Q8=0"
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_NORM_Q8=1"
m=""; for b in $BINS; do m="$m -v ${b#*=}:${b#*=}:ro"; done
# shellcheck disable=SC2086
docker run --rm --name pq-c662-cpy --cpu-shares=256 ${DOCKER_ARGS:-} --gpus "\"device=$GPU\"" $m -v "$OUT:/k" \
    -e CUDA_CACHE_PATH=/k/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 -e N="$N" -e OP="$OP" -e BINS="$BINS" -e OFF="$OFF" -e ON="$ON" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    for b in $BINS; do
        for arm in off on; do
            e="$OFF"; [ $arm = on ] && e="$ON"
            for i in $(seq 1 $N); do
                env $e timeout 3600 ${b#*=}/bin/test-backend-ops -b CUDA0 -o "$OP" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/${b%%=*}.$arm.$i.log
                echo "${b%%=*} $arm $i: $(grep -a -m1 "tests passed" /k/${b%%=*}.$arm.$i.log | sed "s/^ *//"); FAIL: $(grep -a "FAIL" /k/${b%%=*}.$arm.$i.log | sed -E "s/^.*ERR = ([0-9.]+) > ([0-9.]+) +CPY\(type_src=([a-z0-9_]+),type_dst=([a-z0-9_]+),ne_src=(\[[0-9,]+\]).*/\3>\4 \5 \1/" | tr "\n" ";")" >> /k/summary.txt
            done
        done
    done
    chown -R '"$(id -u):$(id -g)"' /k'
date -Is > "$OUT/done"
