#!/usr/bin/env bash
# Ornith-1.5-35B candidate files from the BF16 GGUF of run/convert.sh, with a CPU build of llama-quantize (Docker,
# --cpu-shares 256, nice -n 19, 8 threads; no importance matrix):
#   Q6_K, Q5_K_M, Q4_K_M   llama.cpp's own mixes
#   P-Q5                   routed experts (gate, up, down) Q5_0, every other tensor Q8_0 (as Flint-Pascal-Q5)
#   P-Q4                   routed experts Q4_0, every other tensor Q8_0
#   PK-Q4                  routed experts Q4_K, every other tensor Q8_0
# A recipe name ending in -i (Q4_K_M-i, P-Q4-i, ...) is the same recipe with the importance matrix IMATRIX
# (run/imatrix.sh). Each file is named Ornith-1.5-35B-<recipe>.gguf in OUT; its log is quant-<recipe>.log there.
#   QUANT=<llama-quantize, CPU build> SRC=<BF16 .gguf> OUT=<folder> RECIPE=<name> [IMATRIX=<imatrix.gguf>] bash quantize.sh
set -euo pipefail
: "${QUANT:?}" "${SRC:?}" "${OUT:?}" "${RECIPE:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
THREADS=${THREADS:-8}
EXPERTS() { echo "--tensor-type ffn_gate_exps=$1 --tensor-type ffn_up_exps=$1 --tensor-type ffn_down_exps=$1"; }
IM="" MOUNT_IM=()
if [ "${RECIPE%-i}" != "$RECIPE" ]; then
    : "${IMATRIX:?a -i recipe needs IMATRIX}"
    IM="--imatrix $IMATRIX"; MOUNT_IM=(-v "$(dirname "$IMATRIX"):$(dirname "$IMATRIX"):ro")
fi
case ${RECIPE%-i} in
    Q6_K|Q5_K_M|Q4_K_M) TYPE=${RECIPE%-i}; ARGS="" ;;
    P-Q5) TYPE=Q8_0; ARGS=$(EXPERTS q5_0) ;;
    P-Q4) TYPE=Q8_0; ARGS=$(EXPERTS q4_0) ;;
    PK-Q4) TYPE=Q8_0; ARGS=$(EXPERTS q4_k) ;;
    *) echo "unknown recipe $RECIPE" >&2; exit 1 ;;
esac
NAME=Ornith-1.5-35B-$RECIPE.gguf
bin=$(dirname "$QUANT")
mkdir -p "$OUT"
docker run --rm --label "$LABEL" --cpu-shares 256 -v "$bin:$bin:ro" -v "$(dirname "$SRC"):$(dirname "$SRC"):ro" \
    "${MOUNT_IM[@]}" -v "$OUT:$OUT" "$IMAGE" nice -n 19 "$QUANT" $IM $ARGS "$SRC" "$OUT/$NAME" "$TYPE" "$THREADS" \
    > "$OUT/quant-$RECIPE.log" 2>&1
ls -l "$OUT/$NAME"
