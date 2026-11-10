#!/usr/bin/env bash
# C662 (Q1150): is one architecture's CPU result the same from run to run? BASE's test-llama-archs writes the logits once, then
# each build of BINS compares against them N times, with and without the settings of FLIP.
# Usage (on the build machine): BASE=<bin/> BINS='<name>=<bin/> ...' ARCH=<name> OUT=<dir> [N=10] [FLIP='K=V ...'] bash repeat_arch.sh
set -uo pipefail
: "${BASE:?}" "${BINS:?}" "${ARCH:?}" "${OUT:?}"
N=${N:-10}
mkdir -p "$OUT"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
m="-v $BASE:$BASE:ro"; for b in $BINS; do m="$m -v ${b#*=}:${b#*=}:ro"; done
# shellcheck disable=SC2086
docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} $m -v "$OUT:/k" -e N="$N" -e ARCH="$ARCH" -e BINS="$BINS" -e FLIP="${FLIP:-}" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$BASE"'/test-llama-archs -b CPU -s 1 -a "^$ARCH\$" > /k/base.log 2>&1
    for b in $BINS; do
        for arm in default flip; do
            [ $arm = flip ] && [ -z "$FLIP" ] && continue
            e="X=1"; [ $arm = flip ] && e="$FLIP"
            for i in $(seq 1 $N); do
                env $e LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 ${b#*=}/test-llama-archs -b CPU -s 1 -a "^$ARCH\$" 2>&1 | sed "s/\x1b\[[0-9;]*m//g" \
                    | grep "logits against" | sed "s/^.*f32: //" >> /k/${b%%=*}.$arm.txt
            done
            echo "${b%%=*} $arm ($e): $(sort /k/${b%%=*}.$arm.txt | uniq -c | tr -s " \n" " ")" >> /k/summary.txt
        done
    done
    rm -f /k/logits-*.f32; chown -R '"$(id -u):$(id -g)"' /k'
date -Is > "$OUT/done"
