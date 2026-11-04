#!/usr/bin/env bash
# smoke_gpus.sh: persistent_pipe's correctness on two newer GPUs (no timings): the Pascal code path through the driver's
# PTX JIT (CUDA_FORCE_PTX_JIT=1), then the native sm_86 code, in a CUDA container. Every run must end with
# "checks: every variant gives the same token ids and final residual" (exit code 0).
#
# Two GPUs cover the middle cards (input and output mailboxes) by listing them more than once in pipeline order
# (--devices 0,1,0,1) with small persistent grids (--grid 8), so several pipeline stages share a GPU; the hops between
# stages still cross between the two GPUs. Never list a card twice on a rig: there each kernel holds the whole card.
#
# Env: BIN    the binary (persistent_pipe)
#      GPUS   the docker --gpus device list, two cards (e.g. "device=0,1" or two UUIDs)
#      OUT    output folder
#      IMAGE  the container (default nvidia/cuda:12.9.2-devel-ubuntu24.04)
#      DOCKER_ARGS  more docker run options
#      STOP_ON_FAIL 1: stop after the first run that fails
set -uo pipefail
: "${BIN:?the persistent_pipe binary}" "${GPUS:?docker --gpus device list}" "${OUT:?output folder}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
bindir=$(cd "$(dirname "$BIN")" && pwd)
log() { echo "$(date -Is) $*" | tee -a "$OUT/smoke.log"; }

runs=(
    "f32-2:--cards 2 --smoke"
    "q8-2:--cards 2 --smoke --wire q8"
    "f32-4:--devices 0,1,0,1 --grid 8 --smoke"
    "q8k4-4:--devices 0,1,0,1 --grid 8 --smoke --wire q8 --k 4"
    "k8-3:--devices 0,1,0 --grid 8 --smoke --k 8 --layers 3 --mail-chunks 16"
    "small-4:--devices 0,1,0,1 --grid 8 --smoke --n 2560 --layers 1 --mail-chunks 3 --wire q8 --k 2 --staged-chunks 3"
    "long-2:--cards 2 --passes 200 --warmup 5 --wire q8"
)
fails=0
for jit in 1 0; do
    for r in "${runs[@]}"; do
        name=${r%%:*}
        opts=${r#*:}
        [ "$jit" = 0 ] && case $name in f32-2|f32-4) ;; *) continue ;; esac
        tag=$name-sm86
        jitenv=()
        [ "$jit" = 1 ] && { tag=$name-ptx; jitenv=(-e CUDA_FORCE_PTX_JIT=1); }
        log "$tag: $opts"
        # shellcheck disable=SC2086
        docker run --rm ${DOCKER_ARGS:-} --gpus "\"$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            "${jitenv[@]}" -v "$bindir:$bindir:ro" -v "$OUT:$OUT" "$IMAGE" \
            "$bindir/persistent_pipe" $opts --csv "$OUT/$tag.csv" > "$OUT/$tag.txt" 2>&1
        rc=$?
        log "$tag: exit $rc, $(grep '^checks:' "$OUT/$tag.txt" || echo 'no verdict')"
        if [ "$rc" != 0 ]; then
            fails=$((fails + 1))
            [ "${STOP_ON_FAIL:-0}" = 1 ] && break 2
        fi
    done
done
log "done, $fails failed"
touch "$OUT/DONE"
