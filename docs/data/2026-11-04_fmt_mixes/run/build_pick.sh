#!/usr/bin/env bash
# A chosen set of switches as one file (Q219's allocation, `galloc.py solve`'s pick): the start with every picked
# switch's tensors from its donor, one compose per donor in turn (`galloc.py final` gives the steps), each step checked
# with gguf_compose.py verify against its own inputs. Intermediate steps are written beside the output (the same disk),
# and a final shard that is a link to one of them is moved into place, so the result never links to a removed file;
# shards untouched by every step stay links to the start's. md5 of the written shards in <name>.md5, WORK/done/<name>.
# CPU only, pinned to CPUS at nice 19 and idle I/O priority.
#   REPO=<this repository> SWITCHES=<galloc.py switches table> PICKS=<name,name,...> NAME=<file name>
#   BASE=<first shard of the start> DONORS="<donor_file>=<first shard> ..." MODELS=<output folder> WORK=<scratch>
#   PREFIX=<name prefix> [TABLE=<n-gram table shard>] [CPUS=10,11,22,23] bash build_pick.sh
set -euo pipefail
: "${REPO:?}" "${SWITCHES:?}" "${PICKS:?}" "${NAME:?}" "${BASE:?}" "${DONORS:?}" "${MODELS:?}" "${WORK:?}" "${PREFIX:?}"
CPUS=${CPUS:-10,11,22,23}
log() { printf '%s %s %s\n' "$(date '+%F %T')" "$NAME" "$*"; }
run() { taskset -c "$CPUS" nice -n 19 ionice -c3 "$@"; }
shards() { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf; }
declare -A DONOR
for kv in $DONORS; do DONOR[${kv%%=*}]=${kv#*=}; done
out=$MODELS/fmt_$NAME
[ -f "$WORK/done/$NAME" ] && { log "built before"; exit 0; }
mkdir -p "$out" "$WORK/done" "$WORK/logs"
mapfile -t STEPS < <(python3 -I "$REPO/bench/pascal_mix/galloc.py" final --switches "$SWITCHES" --picks "$PICKS")
mapfile -t CUR < <(shards "$BASE")
n=${#STEPS[@]}
for i in "${!STEPS[@]}"; do
    IFS=$'\t' read -r -a f <<< "${STEPS[$i]}"
    dfile=${f[1]} takes=()
    for p in "${f[@]:2}"; do takes+=(--take "$p"); done
    mapfile -t D < <(shards "${DONOR[$dfile]:?no shard given for donor $dfile}")
    dir=$out
    [ $((i + 1)) -lt "$n" ] && dir=$out/.step$((i + 1))
    log "step $((i + 1)) of $n: $(((${#takes[@]}) / 2)) patterns from $dfile -> $dir"
    mkdir -p "$dir"
    run python3 -I "$REPO/scripts/gguf_compose.py" compose --base "${CUR[@]}" --donor "${D[@]}" "${takes[@]}" \
        --out-dir "$dir" --prefix "$PREFIX-$NAME" --link-unchanged > "$WORK/logs/$NAME.step$((i + 1)).log" 2>&1
    mapfile -t NEW < <(shards "$dir/$PREFIX-$NAME-00001-of-0000${#CUR[@]}.gguf")
    run python3 -I "$REPO/scripts/gguf_compose.py" verify --base "${CUR[@]}" --donor "${D[@]}" "${takes[@]}" \
        --new "${NEW[@]}" | tee -a "$WORK/logs/$NAME.verify.log"
    CUR=("${NEW[@]}")
done
for f in "${CUR[@]}"; do
    if [ -L "$f" ] && [[ $(readlink "$f") == "$out/.step"* ]]; then
        mv -f "$(readlink "$f")" "$f.tmp" && mv -f "$f.tmp" "$f"
    fi
    if [ -n "${TABLE:-}" ] && [ -L "$f" ] && [ "$(readlink -f "$f")" = "$(readlink -f "$(shards "$BASE" | tail -n 1)")" ]; then
        ln -sfn "$TABLE" "$f"
    fi
done
rm -rf "$out"/.step*
(cd "$out" && for f in "${CUR[@]}"; do [ -L "$f" ] || run md5sum "$(basename "$f")"; done) > "$out/$NAME.md5"
date -Is > "$WORK/done/$NAME"
log "done: $PICKS; $(tr '\n' ' ' < "$out/$NAME.md5")"
