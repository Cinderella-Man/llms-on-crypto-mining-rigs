#!/usr/bin/env bash
# C228 q1: which fork switch makes test-llama-archs' qwen4exp row on the Meta backend stop at
# GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) (ggml-backend-meta.cpp) in ladder2/archs through the compute
# 6.1 PTX. Step "forced": every switch -ladder2 forces on in MODE=ptx set on, then each one off, then each one alone.
# Step "rest": every fork path off as ladder2/archs-off (which passes), then each switch that is on by default back at
# its default. Step "nogdn": every architecture with the forced switches on and the two GDN fusions off, plainly and with
# the card-built attention mask checked, for the architectures ladder2/archs never reached. One card.
#   BIN=<bin/ of the rig tarball> OUT=<dir> [ARCH=qwen4exp] [STEPS="forced rest"|nogdn] bash probe_archs_meta.sh
set -uo pipefail
BIN=${BIN:?}; OUT=${OUT:?}; ARCH=${ARCH:-qwen4exp}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
SW="GGML_CUDA_MMVQ_PASCAL GGML_CUDA_MMVF_PASCAL GGML_CUDA_PERSISTENT_MOE GGML_CUDA_FATTN_PASCAL GGML_CUDA_FATTN_TILE_PASCAL
    GGML_CUDA_ATTN_BLOCK"
arm() { # <tag> <K=V ...>
    local tag=$1
    shift
    env "$@" timeout "${TMO:-900}" "$BIN/test-llama-archs" -a "$ARCH" -s 1 > "$OUT/$tag.log" 2>&1
    local rc=$?
    local meta
    meta=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$tag.log" | grep -a -E "^\| *$ARCH\| *Meta\|" | tr -s ' ' | cut -c1-90 | tr '\n' ';')
    [ "$ARCH" = '.*' ] && meta=""
    meta="$(grep -a -E '^Summary' "$OUT/$tag.log" | tr '\n' ';')$meta"
    printf '%-28s exit %-3s %s %s\n' "$tag" "$rc" "$(grep -a -c 'GGML_ASSERT' "$OUT/$tag.log") asserts;" "$meta" | tee -a "$OUT/console.txt"
}
: > "$OUT/console.txt"
STEPS=${STEPS:-forced rest}
REST="GGML_CUDA_MMVQ_PASCAL_KCOL GGML_CUDA_MMVQ_PASCAL_MTOK GGML_CUDA_NORM_Q8 GGML_CUDA_GDN_FUSE LLAMA_GDN_FUSE"
none=""; for s in $SW; do none="$none $s=0"; done
if [[ " $STEPS " == *" forced "* ]]; then
all=""; for s in $SW; do all="$all $s=2"; done
# shellcheck disable=SC2086
arm all-on $all
# shellcheck disable=SC2086
for s in $SW; do arm "without-${s#GGML_CUDA_}" $all "$s=0"; done
# shellcheck disable=SC2086
arm all-off $none
# shellcheck disable=SC2086
for s in $SW; do arm "only-${s#GGML_CUDA_}" $none "$s=2"; done
fi
if [[ " $STEPS " == *" rest "* ]]; then
    off=$none; for s in $REST; do off="$off $s=0"; done
    # shellcheck disable=SC2086
    arm fork-off $off
    for s in $REST; do
        but=$none; for r in $REST; do [ $r = $s ] || but="$but $r=0"; done
        # shellcheck disable=SC2086
        arm "fork-off-but-${s#GGML_CUDA_}" $but
    done
fi
if [[ " $STEPS " == *" nogdn "* ]]; then
    all=""; for s in $SW; do all="$all $s=2"; done
    ARCH='.*' TMO=3600
    # shellcheck disable=SC2086
    arm nogdn $all LLAMA_GDN_FUSE=0 GGML_CUDA_GDN_FUSE=0
    # shellcheck disable=SC2086
    arm nogdn-mask $all LLAMA_GDN_FUSE=0 GGML_CUDA_GDN_FUSE=0 LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1
fi
date -Is > "$OUT/done"
