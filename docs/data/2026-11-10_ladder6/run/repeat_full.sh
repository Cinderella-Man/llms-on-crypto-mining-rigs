#!/usr/bin/env bash
# C662 (Q1150): the whole test-llama-archs run (every architecture in one process) repeated: BIN writes every model's logits
# once at its defaults, then compares N times at its defaults and N times with FLIP set; per run, the models that differ.
# Usage (on the build machine): BIN=<CPU bin/> OUT=<dir> FLIP='K=V ...' [N=3] bash repeat_full.sh
set -uo pipefail
: "${BIN:?}" "${OUT:?}" "${FLIP:?}"
N=${N:-3}
mkdir -p "$OUT"; rm -f "$OUT/done"; : > "$OUT/summary.txt"
docker run --rm --cpu-shares=256 ${DOCKER_ARGS:-} -v "$BIN:$BIN:ro" -v "$OUT:/k" -e N="$N" -e FLIP="$FLIP" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -c '
    env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$BIN"'/test-llama-archs -b CPU -s 1 > /k/base.log 2>&1
    for arm in default flip; do
        e="X=1"; [ $arm = flip ] && e="$FLIP"
        for i in $(seq 1 $N); do
            env $e LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$BIN"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$arm.$i.log
            echo "$arm $i ($e): $(grep -c "logits against" /k/$arm.$i.log) compared; differ: $(grep "logits against" /k/$arm.$i.log | grep -v ", 0 of " | sed -E "s/^([^:]+): .*nmse ([0-9.e+-]+).*/\1 \2/" | tr "\n" ";")" >> /k/summary.txt
        done
    done
    rm -f /k/logits-*.f32; chown -R '"$(id -u):$(id -g)"' /k'
date -Is > "$OUT/done"
