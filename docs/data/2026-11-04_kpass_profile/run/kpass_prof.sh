#!/usr/bin/env bash
# kpass_prof.sh: profiles of whole-model k-token passes (k = 1 and 5 by default) under each GGML_CUDA_PROFILE_MODE of the
# fork's light profiler (Q681), with the Pascal k-column and multi-token kernels on (the defaults) and off. Each run is
# S21's verify_cost.sh `prof` step (llama-bench -p KS -n 0 -r REPS under GGML_CUDA_PROFILE, the host-memory guard and a
# logged quiet check), one model load per arm and mode. Runs ON the rig, detached, from a folder holding verify_cost.sh,
# mem_guard.sh and smoke_request.json in run/. Ends with $OUT/DONE.
# Env: BIN, MODEL (first shard), OUT; optional KS (1,5), REPS (10: sample=8 needs at least 8 replays per k to time every
# group), MODES ("graph layer sample=8 group"), ARMS ("on off"), OFF_ENV (GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0).
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
KS=${KS:-1,5}
REPS=${REPS:-10}
MODES=${MODES:-graph layer sample=8 group}
ARMS=${ARMS:-on off}
OFF_ENV=${OFF_ENV:-GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
for arm in $ARMS; do
    envs=""; [ "$arm" = off ] && envs=$OFF_ENV
    for mode in $MODES; do
        tag=${arm}_${mode//=/}
        mkdir -p "$OUT/$tag"
        echo "$(date -Is) $tag: $envs GGML_CUDA_PROFILE_MODE=$mode" >> "$OUT/status.txt"
        env $envs GGML_CUDA_PROFILE_MODE="$mode" BIN="$BIN" MODEL="$MODEL" DRAFT="$MODEL" OUT="$OUT/$tag/out" STEPS=prof KS="$KS" \
            REPS="$REPS" timeout 1800 bash "$HERE/verify_cost.sh" > "$OUT/$tag/verify_cost.out" 2>&1
        echo "$(date -Is) $tag exit $?" >> "$OUT/status.txt"
        pkill -x llama-bench
    done
done
touch "$OUT/DONE"
