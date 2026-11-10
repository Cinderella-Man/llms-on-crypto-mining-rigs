#!/usr/bin/env bash
# Reads the next file's shards from a slow disk into the page cache while the current file is scored, beside a
# one-text run of ../../2026-11-04_pascal_mix/run/score_all.sh (TEXTS=<text>, files read where they lie through
# score_direct.sh's link folders), so that llama-perplexity's load of the next file does not wait for the disk. With
# 38 GB files on a hard disk, loading a file takes longer than scoring it on one text. The host's memory must hold two
# files beside the n-gram table, or the pages read ahead are dropped before they are used.
# It stays one file ahead: file i+1 is read once the chain has started file i and file i's own read is over (it was
# read here before, or its weights are on the cards: CARD holds more than MIN_MIB, or it has ended). It only reads
# (cat to /dev/null at the lowest CPU and I/O priority), skips shards outside SLOW (the n-gram table's link to its
# copy on the fast disk), changes no score, and exits when the chain has ended (END exists) or is asked to stop.
#   LOG=<score_all.sh's log> DONE=<its marker folder> LINKS=<score_direct.sh's link folder> SLOW=<path prefix of the
#   slow disks> CARD=<NVML index of one card the files load onto> TEXT=<the text being scored> [END=$DONE/all]
#   [MIN_MIB=5000] LIST="<name> ...
#   ..." setsid nohup bash readahead.sh >> <its own log> 2>&1 < /dev/null &
# (END=$DONE/finished under ../../2026-11-04_n2p_flash/run/kld.sh, which calls score_all.sh once per file.)
set -uo pipefail
: "${LOG:?}" "${DONE:?}" "${LINKS:?}" "${SLOW:?}" "${CARD:?}" "${TEXT:?}" "${LIST:?}"
END=${END:-$DONE/all} MIN_MIB=${MIN_MIB:-5000}
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
over() { [ -e "$END" ] || [ -e "$DONE/STOP" ]; }
# the chain has reached this file
seen() { grep -q " $1 $TEXT: start\| $1: scored before" "$LOG"; }
# this file's own read from the disk is over
loaded() {
    grep -q " $1 $TEXT: Mean\|FAILED $1 $TEXT\| $1: scored before" "$LOG" && return 0
    [ "$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr -d ' ')" -gt "$MIN_MIB" ] 2> /dev/null
}
low="nice -n 19"
command -v ionice > /dev/null && low="$low ionice -c 2 -n 7"
mapfile -t N < <(awk 'NF {print $1}' <<< "$LIST")
mine=""
for ((i = 1; i < ${#N[@]}; i++)); do
    prev=${N[$((i - 1))]} next=${N[$i]}
    seen "$next" && continue
    until seen "$prev"; do
        over && exit 0
        sleep 5
    done
    if [ "$mine" != "$prev" ]; then
        until loaded "$prev"; do
            over && exit 0
            sleep 5
        done
    fi
    over && exit 0
    t=$(date +%s) b=0
    for s in "$LINKS/$next"/*.gguf; do
        f=$(readlink -f "$s")
        case $f in "$SLOW"*) ;; *) continue ;; esac
        $low cat "$f" > /dev/null || log "$next: could not read $f"
        b=$((b + $(stat -c %s "$f")))
    done
    mine=$next
    log "$next: $((b / 1048576)) MiB read ahead in $(($(date +%s) - t)) s"
done
