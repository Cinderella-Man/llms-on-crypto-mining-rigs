#!/usr/bin/env bash
# C228 q1: the MUL_MAT_ID case with repeated expert ids (dup=1) that faulted in ladder2/iq-off through the compute 6.1
# PTX on an RTX 3090, isolated: under compute-sanitizer memcheck on llama.cpp's stock IQ path (the Pascal kernels
# skipped for IQ2_S and IQ3_XXS, as iq-off runs them), the same case on -ladder3's own kernels plainly and under
# memcheck, and the 5-8-token dup cases of both types on the stock path under memcheck (which token counts fault).
# One card, inside nvidia/cuda:12.9.2-devel-ubuntu24.04 (compute-sanitizer), MODE=ptx settings of ladder2's script.
#   BIN=<bin/ of the rig tarball> OUT=<dir> bash probe_dup_ids.sh
set -uo pipefail
BIN=${BIN:?}; OUT=${OUT:?}
CS=${CS:-/usr/local/cuda/bin/compute-sanitizer}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
ON="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_PASCAL=2
    GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
STOCK="$ON GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs"
CASE='type_a=iq2_s,type_b=f32,n_mats=28,n_used=10,b=0,m=256,n=5,k=2560,.*dup=1'
RANGE='type_a=(iq2_s|iq3_xxs),type_b=f32,n_mats=28,n_used=10,b=[01],m=256,n=[5-8],k=2560,.*dup=1'
T="$BIN/test-backend-ops -b CUDA0 -o MUL_MAT_ID"
MC="$CS --tool memcheck --print-limit 10"
one() { # <tag> <env> <command...>
    local tag=$1 envs=$2
    shift 2
    # shellcheck disable=SC2086
    env $envs timeout 1800 "$@" > "$OUT/$tag.log" 2>&1
    echo "exit $?" >> "$OUT/$tag.log"
    echo "$tag: $(grep -a -E 'tests passed|ERROR SUMMARY' "$OUT/$tag.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//; s/=========//' | tr '\n' ';') $(tail -n 1 "$OUT/$tag.log")" \
        | tee -a "$OUT/console.txt"
}
: > "$OUT/console.txt"
one stock-case-memcheck  "$STOCK" $MC $T -p "$CASE"
one fork-case            "$ON"    $T -p "$CASE"
one fork-case-memcheck   "$ON"    $MC $T -p "$CASE"
one stock-range-memcheck "$STOCK" $MC $T -p "$RANGE"
# the kernels memcheck names and the first bad access of each stock run
for f in "$OUT"/stock-*.log; do
    echo "== $(basename "$f" .log)" >> "$OUT/kernels.txt"
    grep -a -E 'Invalid|at .*in |by thread|Address|is out of bounds|Kernel|__global__|kernel' "$f" | sed 's/^ *//' | sort | uniq -c | sort -rn | head -n 20 >> "$OUT/kernels.txt"
    grep -a -E 'MUL_MAT_ID\(' "$f" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//' >> "$OUT/kernels.txt"
done
date -Is > "$OUT/done"
