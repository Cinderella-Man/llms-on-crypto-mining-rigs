#!/usr/bin/env bash
# The single-switch files: the start (pmix-q2-fast) with one switch's tensors taken byte for byte from its donor file
# (scripts/gguf_compose.py compose, then verify), one after another on the CPU. SWITCHES is bench/pascal_mix/galloc.py
# switches' table (name, donor_file and regex per switch); LIST names the switches to build, in order. Each file goes
# to MODELS/fmt_<name>/<PREFIX>-<name>-0000k-of-0000N.gguf in the start's shards: a shard with nothing taken is a
# symbolic link to the start's own (the n-gram table, and shard 1 or 2 when the switch touches only the other), and the
# table shard finally links to TABLE when it is set. md5 of every written shard in <name>.md5. WORK/done/<name> marks a
# built file, WORK/failed/<name> a failed one (the next still runs); built files are skipped on a rerun.
# CPU only, pinned to CPUS at nice 19 and idle I/O priority.
#   REPO=<this repository> SWITCHES=<switches.tsv> BASE=<first shard of the start> DONORS="<donor_file>=<first shard>
#   ..." MODELS=<output folder> WORK=<scratch for the markers and logs> PREFIX=<name prefix> LIST="<name> ..."
#   [TABLE=<n-gram table shard>] [CPUS=10,11,22,23] bash build_switches.sh
set -uo pipefail
: "${REPO:?}" "${SWITCHES:?}" "${BASE:?}" "${DONORS:?}" "${MODELS:?}" "${WORK:?}" "${PREFIX:?}" "${LIST:?}"
CPUS=${CPUS:-10,11,22,23}
mkdir -p "$WORK/done" "$WORK/failed" "$WORK/logs"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
run() { taskset -c "$CPUS" nice -n 19 ionice -c3 "$@"; }
shards() { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf; }
declare -A DONOR
while read -r kv; do [ -n "$kv" ] && DONOR[${kv%%=*}]=${kv#*=}; done <<< "$(tr ' ' '\n' <<< "$DONORS")"
mapfile -t B < <(shards "$BASE")
compose=(python3 -I "$REPO/scripts/gguf_compose.py")
rc=0
for name in $LIST; do
    [ -f "$WORK/done/$name" ] && { log "$name: built before"; continue; }
    rm -f "$WORK/failed/$name"
    row=$(awk -F'\t' -v n="$name" 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
        $c["name"] == n { print $c["donor_file"] "\t" $c["regex"] }' "$SWITCHES")
    [ -n "$row" ] || { log "$name: not in $SWITCHES"; echo "not in switches" > "$WORK/failed/$name"; rc=1; continue; }
    dfile=${row%%$'\t'*} regex=${row#*$'\t'}
    first=${DONOR[$dfile]:-}
    [ -n "$first" ] || { log "$name: no shard given for donor $dfile"; echo "no donor" > "$WORK/failed/$name"; rc=1; continue; }
    mapfile -t D < <(shards "$first")
    out=$MODELS/fmt_$name
    mkdir -p "$out"
    log "$name: from $dfile, re:$regex -> $out"
    if run "${compose[@]}" compose --base "${B[@]}" --donor "${D[@]}" --take "re:$regex" --out-dir "$out" \
            --prefix "$PREFIX-$name" --link-unchanged > "$WORK/logs/$name.compose.log" 2>&1; then
        mapfile -t F < <(shards "$out/$PREFIX-$name-00001-of-0000${#B[@]}.gguf")
        if run "${compose[@]}" verify --base "${B[@]}" --donor "${D[@]}" --take "re:$regex" --new "${F[@]}" \
                > "$WORK/logs/$name.verify.log" 2>&1; then
            if [ -n "${TABLE:-}" ]; then
                for f in "${F[@]}"; do
                    [ -L "$f" ] && [ "$(readlink -f "$f")" = "$(readlink -f "${B[${#B[@]} - 1]}")" ] && ln -sfn "$TABLE" "$f"
                done
            fi
            (cd "$out" && for f in "${F[@]}"; do [ -L "$f" ] || run md5sum "$(basename "$f")"; done) > "$out/$name.md5"
            date -Is > "$WORK/done/$name"
            log "$name: done: $(tr '\n' ' ' < "$out/$name.md5")"
            continue
        fi
        log "$name: verify FAILED ($WORK/logs/$name.verify.log)"
    else
        log "$name: compose FAILED ($WORK/logs/$name.compose.log)"
    fi
    echo "exit at $(date -Is)" > "$WORK/failed/$name"
    rc=1
done
date -Is > "$WORK/done/all"
exit $rc
