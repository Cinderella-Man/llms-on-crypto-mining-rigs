#!/usr/bin/env bash
# Runs on the quality machine (started by run.sh score): for each ITEM, composes its file with near_q8.py (CPU,
# Docker, low share), scores it against Flint BF16's logits on 80 × 512 tokens of wikitext-2 with the flags of
# docs/data/2026-10-23_q36_q45/run/kld_gpu.sh (every Flint mix's score), and deletes the composed file. With PREFETCH=1
# the next file is composed while the current one is scored (two composed files at a time). Items whose log holds a
# result are skipped.
#   ITEM: a switch (q0 … dn5, head, embd; near_q8.py specs), NAME=SWITCH,SWITCH,... for a set (NAME= alone is q8r),
#         or NAME@FILE.gguf for an existing file, scored as it is
# Env: WORK (scratch on fast disk: run/, files/, logs/), BASE (Unsloth's Q8_0), DONOR (Flint-Pascal-Q5), REF (folder with
#   wikitext-2-raw/wiki.test.raw and flint-BF16.c512x80.kld), BIN (llama-perplexity), GGUFPY (gguf-py), GPUS,
#   TS (optional -ts split), PREFETCH (0 or 1), MIN_FREE_GB (default 50), EXTRA_MOUNTS.
set -euo pipefail
: "${WORK:?}" "${BASE:?}" "${DONOR:?}" "${REF:?}" "${BIN:?}" "${GGUFPY:?}" "${GPUS:?}"
IMG_PY=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
IMG_CUDA=nvidia/cuda:12.9.2-devel-ubuntu24.04
mkdir -p "$WORK/files" "$WORK/logs"
MOUNTS="-v $WORK:$WORK ${EXTRA_MOUNTS:-}"
for p in "$BASE" "$DONOR" "$REF/x" "$BIN" "$GGUFPY/x"; do
    d=$(dirname "$p"); case " $MOUNTS " in *" -v $d:$d"*) ;; *) MOUNTS="$MOUNTS -v $d:$d:ro" ;; esac
done

name_of() { local n=${1%%=*}; echo "${n%%@*}"; }
switches_of() { case "$1" in *=*) echo "${1#*=}" | tr , ' ' ;; *) echo "$1" ;; esac; }
file_of() { case "$1" in *@*) echo "${1#*@}" ;; *) echo "$WORK/files/flint-near-q8-$(name_of "$1").gguf" ;; esac; }
log_of() { echo "$WORK/logs/kld-$(name_of "$1").log"; }
scored() { grep -q "Mean    KLD" "$(log_of "$1")" 2>/dev/null; }
compose() {
    local out; out=$(file_of "$1")
    case "$1" in *@*) return 0 ;; esac
    [ -s "$out" ] && return 0
    # leave MIN_FREE_GB (default 50) free on WORK's disk after writing a file the size of BASE
    local need avail; need=$(( ${MIN_FREE_GB:-50} + $(stat -L -c %s "$BASE") / 1000000000 + 1 ))
    while avail=$(df -B1000000000 --output=avail "$WORK" | tail -1 | tr -d ' '); [ "$avail" -lt "$need" ]; do
        echo "== $(date +%T) waiting: $avail GB free on $WORK, need $need"; sleep 120
    done
    echo "== $(date +%T) compose $(name_of "$1"): $(switches_of "$1")"
    # shellcheck disable=SC2046
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none $MOUNTS -e PYTHONPATH="$GGUFPY" "$IMG_PY" \
        nice -n 19 python "$WORK/run/near_q8.py" compose "$BASE" "$DONOR" "$out.part" $(switches_of "$1") \
        > "$WORK/logs/compose-$(name_of "$1").log" 2>&1
    mv "$out.part" "$out"
}
score() {
    local f m ts=""; f=$(file_of "$1"); m=""
    case "$1" in *@*) m="-v $(dirname "$f"):$(dirname "$f"):ro" ;; esac
    [ -n "${TS:-}" ] && ts="-ts $TS"
    echo "== $(date +%T) score $(name_of "$1")"
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        $MOUNTS $m "$IMG_CUDA" "$BIN" -m "$f" -f "$REF/wikitext-2-raw/wiki.test.raw" -c 512 --chunks 80 -ngl 99 $ts -fa on \
        --kl-divergence-base "$REF/flint-BF16.c512x80.kld" --kl-divergence > "$(log_of "$1")" 2>&1 \
        || { echo "FAILED, see $(log_of "$1")"; tail -3 "$(log_of "$1")"; return 0; }
    grep -E "^(Mean    KLD|Same top p)" "$(log_of "$1")" || true
}
drop() { case "$1" in *@*) ;; *) rm -f "$(file_of "$1")" ;; esac; }

todo=()
for item in "$@"; do scored "$item" || todo+=("$item"); done
pid=""
for i in "${!todo[@]}"; do
    item=${todo[$i]}
    if [ -n "$pid" ]; then wait "$pid"; pid=""; fi
    compose "$item"
    if [ "${PREFETCH:-0}" = 1 ] && [ $((i + 1)) -lt ${#todo[@]} ]; then compose "${todo[$((i + 1))]}" & pid=$!; fi
    score "$item"
    scored "$item" && drop "$item"
done
[ -n "$pid" ] && wait "$pid"
echo "== $(date +%T) done"
