#!/usr/bin/env bash
# Q1028 (C556) on the rig: does -moeskip's switch-off path (LLAMA_MOE_TAIL unset, the expert kernels still checking every
# id for -1) cost time against -kshape, the build it sits on? T(k) on real agent text at k = 4, 5 and 6 (rig_tk.sh
# STEPS=tk ARMS=off: one llama-server load per run), loads alternating between the two builds in ORDER, so that a steady
# drift over the run falls on both equally (A B B A B A A B: each pair (1,2), (3,4), (5,6), (7,8) holds one load of
# each, and the linear drift cancels in the mean of the pairs). Runs ON the rig, detached, the rig idle; ends with
# $OUT/DONE.
# Usage: BIN_A=<bin/ of the -kshape tarball> BIN_B=<bin/ of the -moeskip tarball> MODEL=<Swift Q2_0 first shard>
#   TEXT=<the agent text file of C271> OUT=<dir> [NAME_A=kshape] [NAME_B=moeskip] [ORDER='A B B A B A A B']
#   [KS='4 5 6'] [rig_tk.sh's other variables] bash c556_alt.sh
# Output: OUT/L<n>-<name>/ (rig_tk.sh's output of load n), OUT/requests.tsv (load, build, then rig_tk.sh's columns),
#   OUT/status.txt (per load the cards' temperature and clock before it starts)
set -uo pipefail
: "${BIN_A:?}" "${BIN_B:?}" "${MODEL:?}" "${TEXT:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
NAME_A=${NAME_A:-kshape}; NAME_B=${NAME_B:-moeskip}
ORDER=${ORDER:-A B B A B A A B}
export KS=${KS:-4 5 6} MODEL TEXT
mkdir -p "$OUT"
rm -f "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }

n=0
for x in $ORDER; do
    n=$((n + 1))
    if [ "$x" = A ]; then name=$NAME_A; bin=$BIN_A; else name=$NAME_B; bin=$BIN_B; fi
    log "load $n: $name ($bin); cards $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr '\n' ';' | tr -d ' ')"
    BIN="$bin" OUT="$OUT/L$n-$name" STEPS=tk ARMS=off bash "$HERE/rig_tk.sh" > "$OUT/L$n-$name.out" 2>&1 < /dev/null
    log "load $n: $name exit $? ($(grep -c '^tk-off' "$OUT/L$n-$name/requests.tsv" 2>/dev/null) requests)"
done

# one table: load, build, then the requests of rig_tk.sh
{
    printf 'load\tbuild\t'; head -n 1 "$(ls "$OUT"/L1-*/requests.tsv)"
    n=0
    for x in $ORDER; do
        n=$((n + 1))
        if [ "$x" = A ]; then name=$NAME_A; else name=$NAME_B; fi
        tail -n +2 "$OUT/L$n-$name/requests.tsv" | sed "s/^/$n\t$name\t/"
    done
} > "$OUT/requests.tsv"
log "done: $(($(wc -l < "$OUT/requests.tsv") - 1)) requests"
touch "$OUT/DONE"
