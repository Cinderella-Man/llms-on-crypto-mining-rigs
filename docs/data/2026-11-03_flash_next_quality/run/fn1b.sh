#!/usr/bin/env bash
# FN1b: the reference's logits on the three texts (the KL divergence bases) on all seven GPUs. First a fit check: one
# 8,192-token chunk of the agent text, GPU memory sampled every 5 s ($KLD/fit.mem); the first tensor split that runs
# is written to $KLD/gpu_args, which kld.sh adds to every later run, so that every file is scored with the same layers
# on the same cards as the bases. Then the floor: the reference itself at -ub 256 (the same weights in another
# summation order) against its own bases, as in ../../2026-11-03_table_bits.
#   (env: the chain's) bash fn1b.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/common.sh"
: "${REF:?}" "${KLD:?}" "${GPUS_ALL:?}" "${PPL:?}" "${AGENT:?}"
export GPUS=$GPUS_ALL OUT=$KLD
mkdir -p "$KLD" "$WORK/tmp"
wait_gpus "$GPUS"
bindir=$(dirname "$(dirname "$PPL")")
if [ ! -f "$KLD/gpu_args" ]; then
    for ts in "" "-ts 9,11,24,24,24,24,24" "-ts 7,9,24,24,24,24,24" "-ts 5,7,24,24,24,24,24"; do
        log "fit check, split ${ts:-by free memory (default)}"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$KLD/fit.mem" &
        smi=$!
        rc=0
        docker run --rm $DOCKER_BASE --name "pq-$TAG-fit" --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            -v "$bindir:$bindir" $MOUNT_ARGS "$IMAGE" "$PPL" -m "$REF" -f "$AGENT" -c 8192 -b 8192 -ub 512 --chunks 1 \
            -ngl 99 -fa on $ts --kl-divergence-base "$WORK/tmp/fit.kld" > "$KLD/fit.log" 2>&1 || rc=$?
        kill $smi 2> /dev/null || true
        rm -f "$WORK/tmp/fit.kld"
        log "peak MiB per GPU: $(awk -F', ' '{if ($2 > m[$1]) m[$1] = $2} END {for (i = 0; i < 7; i++) printf "%s:%s ", i, m[i]}' "$KLD/fit.mem")"
        if [ $rc -eq 0 ] && grep -q "Final estimate" "$KLD/fit.log"; then
            echo "$ts" > "$KLD/gpu_args"
            cp "$KLD/fit.log" "$KLD/fit.ok.log"; cp "$KLD/fit.mem" "$KLD/fit.ok.mem"
            log "fits: $(grep "Final estimate" "$KLD/fit.log")"
            break
        fi
        log "does not fit (exit $rc): $(grep -i -m 3 -E "error|out of memory|failed|abort" "$KLD/fit.log" | tr '\n' ' ')"
        wait_gpus "$GPUS"
    done
    [ -f "$KLD/gpu_args" ] || { log "no split fits the seven GPUs"; exit 1; }
fi
log "split: $(cat "$KLD/gpu_args")"
bash "$HERE/kld.sh" base "$REF"
EXTRA="-ub 256" bash "$HERE/kld.sh" kld "$REF" q8_0-ub256 || log "the -ub 256 control failed; the bases stand"
