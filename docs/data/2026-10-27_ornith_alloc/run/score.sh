#!/usr/bin/env bash
# Runs on the quality machine (started by run.sh score): for each ITEM, composes its file from file c with alloc.py
# (CPU, Docker, low share), scores it with llama-perplexity against the Q8_0's KLD bases with the same flags as the
# bases, and deletes the composed file. The next item is composed while the current one is scored, so at most two
# composed files exist at a time. An item whose logs already hold a result is skipped.
#   ITEM: a switch name (gu0 … dense5, see alloc.py specs), or NAME=SWITCH,SWITCH,... for a set (the validation file)
# Env: WORK (scratch on fast disk: run/, files/, logs/), FILE_C, FILE_B, FILE_D, KLD (folder with
#   Ornith-1.5-122B-Q8_0.<text>.kld and the agent and code texts), WIKI (wiki.test.raw), BIN (llama-perplexity that
#   knows N2P/N3P), GGUFPY (gguf-py that knows N2P/N3P), GPUS (default 3,4), TEXTS (default "wiki agent code"),
#   PREFETCH (1, the default, composes the next item while the current one is scored; 0 keeps one composed file),
#   MIN_FREE_GB (default 50), EXTRA_MOUNTS (more "-v a:a" for symlinked folders).
set -euo pipefail
: "${WORK:?}" "${FILE_C:?}" "${FILE_B:?}" "${FILE_D:?}" "${KLD:?}" "${WIKI:?}" "${BIN:?}" "${GGUFPY:?}"
GPUS=${GPUS:-3,4}; TEXTS=${TEXTS:-wiki agent code}
IMG_PY=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
IMG_CUDA=nvidia/cuda:12.9.2-devel-ubuntu24.04
mkdir -p "$WORK/files" "$WORK/logs"
MOUNTS="-v $WORK:$WORK ${EXTRA_MOUNTS:-}"
for p in "$FILE_C" "$FILE_B" "$FILE_D" "$KLD/x" "$WIKI" "$BIN" "$GGUFPY/x"; do
    d=$(dirname "$p"); case " $MOUNTS " in *" -v $d:$d"*) ;; *) MOUNTS="$MOUNTS -v $d:$d:ro" ;; esac
done

name_of() { echo "${1%%=*}"; }
switches_of() { case "$1" in *=*) echo "${1#*=}" | tr , ' ' ;; *) echo "$1" ;; esac; }
file_of() { echo "$WORK/files/Ornith-1.5-122B-alloc-$(name_of "$1").gguf"; }
scored() {
    local t; for t in $TEXTS; do grep -q "Mean    KLD" "$WORK/logs/alloc-$(name_of "$1").$t.kld.log" 2>/dev/null || return 1; done
}
compose() {
    local out; out=$(file_of "$1")
    [ -s "$out" ] && return 0
    # leave MIN_FREE_GB (default 50) free on WORK's disk after writing a file the size of file c plus 1 GB
    local need avail; need=$(( ${MIN_FREE_GB:-50} + $(stat -L -c %s "$FILE_C") / 1000000000 + 2 ))
    while avail=$(df -B1000000000 --output=avail "$WORK" | tail -1 | tr -d ' '); [ "$avail" -lt "$need" ]; do
        echo "== $(date +%T) waiting: $avail GB free on $WORK, need $need"; sleep 120
    done
    echo "== $(date +%T) compose $(name_of "$1"): $(switches_of "$1")"
    # shellcheck disable=SC2046
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none $MOUNTS -e PYTHONPATH="$GGUFPY" "$IMG_PY" \
        nice -n 19 python "$WORK/run/alloc.py" compose "$FILE_C" "$FILE_B" "$FILE_D" "$out.part" $(switches_of "$1") \
        > "$WORK/logs/compose-$(name_of "$1").log" 2>&1
    mv "$out.part" "$out"
}
score() {
    local f t txt c n log; f=$(file_of "$1")
    for t in $TEXTS; do
        case $t in
            wiki)  txt=$WIKI; c=512; n=80 ;;
            agent) txt=$KLD/agent-heldout.txt; c=8192; n=12 ;;
            code)  txt=$KLD/code-exercism.txt; c=512; n=80 ;;
            *) echo "unknown text $t"; continue ;;
        esac
        log=$WORK/logs/alloc-$(name_of "$1").$t.kld.log
        grep -q "Mean    KLD" "$log" 2>/dev/null && continue
        echo "== $(date +%T) score $(name_of "$1") $t"
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
            $MOUNTS "$IMG_CUDA" "$BIN" -m "$f" -f "$txt" -c "$c" -b "$c" -ub 512 --chunks "$n" -ngl 99 -fa on \
            --kl-divergence-base "$KLD/Ornith-1.5-122B-Q8_0.$t.kld" --kl-divergence > "$log" 2>&1 \
            || { echo "FAILED, see $log"; tail -3 "$log"; continue; }
        grep -E "^(Mean    KLD|Same top p)" "$log" || true
    done
}

todo=()
for item in "$@"; do scored "$item" || todo+=("$item"); done
pid=""
for i in "${!todo[@]}"; do
    item=${todo[$i]}
    if [ -n "$pid" ]; then wait "$pid"; pid=""; fi
    compose "$item"
    if [ "${PREFETCH:-1}" = 1 ] && [ $((i + 1)) -lt ${#todo[@]} ]; then compose "${todo[$((i + 1))]}" & pid=$!; fi
    score "$item"
    scored "$item" && rm -f "$(file_of "$item")"
done
[ -n "$pid" ] && wait "$pid"
echo "== $(date +%T) done"
