#!/usr/bin/env bash
# C662 (Q1150): a filtered test-backend-ops run through the compute 6.1 PTX on one card, per build, with the fork's CUDA paths
# off or forced on (check.sh's settings), to find and compare a crashing case.
# Usage (on the build machine): GPU=<uuid> BINS='<name>=<tree> ...' OUT=<dir> ARGS='<test-backend-ops arguments>' [ARMS='off on'] [EXTRA_ENV='K=V ...'] bash tbo_case.sh
set -uo pipefail
: "${GPU:?}" "${BINS:?}" "${OUT:?}" "${ARGS:?}"
mkdir -p "$OUT/nvcache"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
OFF="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_FATTN_TILE_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0 GGML_CUDA_NORM_Q8=0"
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_NORM_Q8=1"
m=""; for b in $BINS; do m="$m -v ${b#*=}:${b#*=}:ro"; done
# shellcheck disable=SC2086
docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} --gpus "\"device=$GPU\"" $m -v "$OUT:/k" \
    -e CUDA_CACHE_PATH=/k/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 -e BINS="$BINS" -e OFF="$OFF" -e ON="$ON" -e ARMS="${ARMS:-off on}" \
    -e ARGS="$ARGS" -e EXTRA_ENV="${EXTRA_ENV:-}" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    for b in $BINS; do
        for arm in $ARMS; do
            e="$OFF"; [ $arm = on ] && e="$ON"
            eval "set -- $ARGS"
            env $e $EXTRA_ENV timeout 3600 ${b#*=}/bin/test-backend-ops -b CUDA0 "$@" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/${b%%=*}.$arm.log
            echo "${b%%=*} $arm: exit ${PIPESTATUS[0]}; $(grep -a -m1 "tests passed" /k/${b%%=*}.$arm.log | sed "s/^ *//"); OK lines $(grep -a -c ": OK" /k/${b%%=*}.$arm.log); FAIL lines $(grep -a -c "FAIL" /k/${b%%=*}.$arm.log); last case: $(grep -a ": OK\|: FAIL" /k/${b%%=*}.$arm.log | tail -n 1 | cut -c1-150); $(grep -a -m1 "CUDA error" /k/${b%%=*}.$arm.log)" >> /k/summary.txt
        done
    done
    chown -R '"$(id -u):$(id -g)"' /k'
date -Is > "$OUT/done"
