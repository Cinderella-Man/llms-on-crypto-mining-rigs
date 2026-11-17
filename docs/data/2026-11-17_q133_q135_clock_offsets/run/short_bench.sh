#!/usr/bin/env bash
# short_bench.sh: a 7-second streaming-read check of one or more cards with the kit's mmvq_pascal (Q8_0 attn_q only),
# for diagnostics: the same setting at two times, or one card against its neighbours.
#
#   short_bench.sh TAG [CARD ...]     one run per card (default: $GPU); prints the plain and LDG streaming read (GB/s),
#                                     the fastest attn_q variant's time and the card's temperature and power limit
#   short_bench.sh at MEM TAG         the same on card $GPU with its memory offset at MEM (MT/s; negative for an
#                                     underclock), then the offset back to 0 (run with the watchdog on)
#
# Env [defaults]: WORK [/tmp/q133], KIT [$WORK/mem_offset-kit-20261024], OUT [$WORK/out], GPU [0]
set -uo pipefail
WORK=${WORK:-/tmp/q133}
KIT=${KIT:-$WORK/mem_offset-kit-20261024}
OUT=${OUT:-$WORK/out}
GPU=${GPU:-0}
mkdir -p "$OUT"
one() {  # one CARD TAG
    local f="$OUT/mmvq_$2.csv"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$1" "$KIT/bin/mmvq_pascal" --types q8_0 --shapes attn_q \
        --csv "$f" > "$OUT/mmvq_$2.log" 2>&1
    echo "$2 card $1: exit $?, plain $(awk -F, '$3 == "STREAM_READ_PLAIN" { print $15 }' "$f") GB/s, ldg $(awk -F, '$3 == "STREAM_READ_LDG" { print $15 }' "$f") GB/s, fastest attn_q $(awk -F, '$3 == "attn_q" && (m == "" || $12 < m) { m = $12; v = $10 } END { print v, m }' "$f") us, $(nvidia-smi -i "$1" --query-gpu=temperature.gpu,power.limit --format=csv,noheader)"
}
case "${1:-}" in
at)
    mem=${2:?MT/s} tag=${3:?tag}
    python3 "$KIT/nvml_memoc.py" set "$GPU" "$mem" | tail -n 1
    one "$GPU" "$tag"
    python3 "$KIT/nvml_memoc.py" reset "$GPU" | tail -n 1 ;;
"") sed -n '2,11p' "$0"; exit 2 ;;
*)
    tag=$1; shift
    for c in "${@:-$GPU}"; do one "$c" "$tag-card$c"; sleep 2; done ;;
esac
