#!/usr/bin/env bash
# run.sh: C1103 (Q1836), the quality of GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 (Q8_0 K scored in int8 with dp4a in the
# fork's compute 6.1 prompt attention kernel) on Ornith-1.5-122B Pascal N2P GGUF with its v5 preset's Q8_0 cache.
# C994's and C1032's form: every arm is scored against the preset's own run on the same tokens, beside two yardsticks
# for what summing in another order does, over C976's 12 stretches of 8,192 tokens of the agent text (-c 8192
# --chunks 12, each chunk scored on its second half), then one window of 49,152 tokens (-c 49152 --chunks 1).
# kld_depth.sh (beside it) is C976's driver, unchanged.
#   s  (12 x 8,192)  q8    the preset as a P106 runs it: writes the reference
#                    kdp   the preset + GGML_CUDA_FATTN_TILE_PASCAL_KDP=1: the arm under test
#                    q8g   the preset with the fork's prompt kernel off (GGML_CUDA_FATTN_TILE_PASCAL=0): the build's
#                          generic tile kernel on F16 copies of K/V, point 26's (C1032's) attention-only yardstick
#                    q8u   the preset at -ub 128: every sum in the model in another order, C994's yardstick
#                    q8f1  the preset + GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1: the prompt kernel's output x 1.05, the
#                          error the check must see (validity)
#   w  (1 x 49,152)  q8 (reference), kdp, q8g, q8u
#   s2 (12 x 8,192)  q8t   the preset with cuBLAS left as ggml sets it on a newer card (no PASCAL_CUBLAS: TF32 on, F16
#                          and quantized sources computed in F16): what C1108's setting changes
#                    q8r   q8 again: the floor
# The preset is read from PRESET_FILE (release/ornith-v5/settings.env): its PRESET_ENV, run as a P106 runs it with
# `$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")` from scripts/pascal_forced.sh (beside this script or in the
# repository): the fork's compute 6.1 paths at 2 with a P106-090's launch shapes, LLAMA_NO_MOE_DOWN_RESCALE=tg and
# cuBLAS in F32 without TF32 (PASCAL_CUBLAS, C1108). Its cache and batch as in PRESET_ARGS (-ctk q8_0 -ctv q8_0 -b 4096
# -ub 512); the context and split are this check's (-c per pass, -ts 1,1,1 over three cards). Each arm's whole variable
# list is its own (kld_depth.sh's PRESET_ENV is left empty), so no arm carries a variable twice. Every arm logs at
# -lv 4 (log level only): the fork's prompt kernel then logs its line when it first takes a pass, and the kdp path a
# second one. llama-perplexity's scored passes are 512 tokens (128 in q8u), so the preset's decode switches
# (GGML_CUDA_FATTN_TILE_PASCAL_MIN=1, GGML_CUDA_FATTN_DEC_PASCAL=4) reach only the load's unscored warm-up call.
# WORK/DONE at the end; WORK/FAILED if a pass failed. Correctness only: no time is read.
# Env: WORK, BIN (the build's bin), MODEL, TEXT, GPUS, MOUNTS, DOCKER_ARGS, CNAME, PRESET_FILE, PREFLIGHT (a CUDA check
#      of the cards, run before each pass; optional), IMAGE, PASSES (default "s w s2"; set and empty: none).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${GPUS:?}" "${PRESET_FILE:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PF=$HERE/pascal_forced.sh
[ -f "$PF" ] || PF=$HERE/../../../../scripts/pascal_forced.sh
# shellcheck source=/dev/null
. "$PF"
PENV=$(set +u; . "$PRESET_FILE" > /dev/null; echo "$PRESET_ENV")
[ -n "$PENV" ] || { echo "no PRESET_ENV in $PRESET_FILE"; exit 1; }
BASE_ENV="$PENV $(pascal_force_preset "$PENV")"
# drop "<NAME=value ...>" NAME...: the list without those names
drop() {
    local kv n out="" keep
    for kv in $1; do
        keep=1
        for n in "${@:2}"; do [ "${kv%%=*}" = "$n" ] && keep=0; done
        [ $keep = 1 ] && out+="$kv "
    done
    echo "${out% }"
}
export BIN MODEL TEXT GPUS MOUNTS DOCKER_ARGS
export CNAME=${CNAME:-kld-depth}
export JIT=$WORK/jit
export ARGS="-ngl 99 -fa on -fit off --no-host -ts 1,1,1"
export PRESET_ENV=""
Q8="-ctk q8_0 -ctv q8_0 -lv 4"
for a in q8 kdp q8g q8u q8f1 q8t q8r; do
    export "ARM_ARGS_$a=$Q8" "ARM_UB_$a=512"
done
export ARM_UB_q8u=128
export ARM_ENV_q8="$BASE_ENV"
export ARM_ENV_kdp="$BASE_ENV GGML_CUDA_FATTN_TILE_PASCAL_KDP=1"
export ARM_ENV_q8g="$(drop "$BASE_ENV" GGML_CUDA_FATTN_TILE_PASCAL) GGML_CUDA_FATTN_TILE_PASCAL=0"
export ARM_ENV_q8u="$BASE_ENV"
export ARM_ENV_q8f1="$BASE_ENV GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1"
export ARM_ENV_q8t="$(drop "$BASE_ENV" NVIDIA_TF32_OVERRIDE GGML_CUDA_CUBLAS_COMPUTE_TYPE)"
export ARM_ENV_q8r="$BASE_ENV"
for a in q8 kdp q8g q8u q8f1 q8t q8r; do v=ARM_ENV_$a; echo "arm $a: -ub $(eval echo \$ARM_UB_$a) ${!v}"; done
PASSES=${PASSES-s w s2}   # set and empty: no pass (prints each arm's settings only)
rm -f "$WORK/DONE" "$WORK/FAILED"
for p in $PASSES; do
    case $p in
        s)  out=s chunks=12 depth=8192  arms="q8 kdp q8g q8u q8f1" ;;
        w)  out=w chunks=1  depth=49152 arms="q8 kdp q8g q8u" ;;
        s2) out=s chunks=12 depth=8192  arms="q8 q8t q8r" ;;   # q8's reference from pass s is kept, so q8 is skipped
        *)  echo "unknown pass $p"; touch "$WORK/FAILED"; continue ;;
    esac
    echo "==== pass $p $(date -Is)"
    if [ -n "${PREFLIGHT:-}" ] && ! "$PREFLIGHT" -g "$GPUS"; then
        echo "preflight failed before pass $p"; touch "$WORK/FAILED"; break
    fi
    OUT=$WORK/$out CHUNKS=$chunks DEPTHS=$depth ARMS="$arms" bash "$HERE/kld_depth.sh"
    [ -f "$WORK/$out/failed" ] && touch "$WORK/FAILED"
done
echo "==== end $(date -Is)"
touch "$WORK/DONE"
