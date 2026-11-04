#!/usr/bin/env bash
# Q238's tiered Swift Flash-Next files, CPU only: each layer's experts split by Q232's agent-text ranking into a hot
# tier, the makers' own slabs byte for byte, and a cold tier requantized from the Swift-1.5 Q8_0 with FN4's importance
# matrix (scripts/gguf_tier.py; llama-quantize of a CPU build).
#   stage     a copy of each STAGE file's shards in WORK/stage/<its folder>/ (a link stays a link), read once at idle
#             I/O priority, so the tier and verify steps read their bases there and not on a disk other jobs read
#   split     gguf_tier.py split --hot N for each N in HOTS (hot.tsv, cold.tsv per N)
#   extract   the cold ids of the smallest N (every other split's cold ids are among them) from the Q8_0, with the
#             importance matrix cut the same way
#   quantize  llama-quantize of the extract once per cold recipe in RECIPES, "<name>:<gate/up type>:<down type>"
#             (Flash-Next's down rows hold 640 values, so the down type needs blocks of 32 or 64; "base": the cold
#             down projections stay the base's own, so llama-quantize copies them and tier takes only gate and up from
#             the recipe); the log must show every gate/up weight converted, and every down weight as the recipe says
#   tier      per FILES entry "<folder>|<base first shard>|<prefix>|<N>|<cold>[|<hot first shard>]": gguf_tier.py
#             tier into OUT/<folder>: hot = the base's own slabs, or those of the hot file (another quant of the same
#             model); cold = a recipe's quantized slabs, "base" (the base's own), or a file's first shard (that file's
#             slabs); a recipe whose down type is "base" takes the down projections from COLD_REST (a first shard;
#             default the base); the unchanged table shard a link
#   verify    gguf_tier.py verify of each file, md5 of each written shard (OUT/<folder>/md5.txt)
#   headers   header-only copies of each file for scripts/fit_probe.sh, verified, packed sparse (HEADERS/<folder>.tar.gz)
#   bytes     run/bytes.py over the tiered files and REFS: bytes per token weighted by Q232's routings (bytes.tsv)
# Env: REPO, WORK (scratch: the extract and the quantized cold experts, ~95 GB), OUT, QUANTIZE (llama-quantize),
#      Q8 (the Q8_0's first shard), IMATRIX, EXPERTS (docs/data/2026-11-04_expert_usage/experts.tsv.gz), FILES, REFS
#      ("<name>=<first shard> ..." for the bytes table), STAGE (first shards to copy), COLD_REST, HOTS ("256 320"), RECIPES ("c1:iq1_m:base"),
#      STEPS (all), THREADS (8), CPUS (a cpuset), IMAGE (for llama-quantize), PY_IMAGE (python:3.12-slim-bookworm),
#      PY_MEMORY (8g: the memory, page cache included, the Python steps may hold), MOUNTS (folders the containers
#      mount, at their own paths), HEADERS (WORK/headers)
# Writes WORK/split/h<N>/, WORK/cold/, OUT/<folder>/, HEADERS/, WORK/files.tsv (one verdict line per step and file).
set -uo pipefail
: "${REPO:?}" "${WORK:?}" "${OUT:?}"
HOTS=${HOTS:-256 320}
RECIPES=${RECIPES:-c1:iq1_m:base}
STEPS=${STEPS:-stage split extract quantize tier verify headers bytes}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PY_IMAGE=${PY_IMAGE:-python:3.12-slim-bookworm}
PY_MEMORY=${PY_MEMORY:-8g}
HEADERS=${HEADERS:-$WORK/headers}
TOOL=$REPO/scripts/gguf_tier.py
mounts=""
for d in $(printf '%s\n' "$WORK" "$OUT" "$REPO" ${QUANTIZE:+"$(dirname "$QUANTIZE")"} ${MOUNTS:-} | sort -u); do
    mounts="$mounts -v $d:$d"
done
D="--rm --cpu-shares 256 --label pascal-quants=1 ${CPUS:+--cpuset-cpus $CPUS}"
mkdir -p "$WORK" "$OUT" "$HEADERS"
touch "$WORK/files.tsv"

log()     { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/files.tsv"; }
# shellcheck disable=SC2086
py()      { docker run $D --memory "$PY_MEMORY" $mounts "$PY_IMAGE" nice -n 19 ionice -c3 python3 -I "$@"; }
# every shard of a split file, from its first
shards()  { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf; }
# the split with the fewest hot experts has the most cold ones, among which are every other split's
largest_cold=$(printf '%s\n' $HOTS | sort -n | head -n 1)
C=$WORK/cold

if step stage; then
    for first in ${STAGE:-}; do
        dst=$WORK/stage/$(basename "$(dirname "$first")")
        mkdir -p "$dst"
        for s in $(shards "$first"); do
            t=$dst/$(basename "$s")
            if [ -L "$s" ]; then ln -sfn "$(realpath "$s")" "$t"; continue; fi
            [ -f "$t" ] && [ "$(stat -c %s "$t")" = "$(stat -c %s "$s")" ] && continue
            nice -n 19 ionice -c3 cp "$s" "$t.part" && mv "$t.part" "$t"
        done
        # the copies against the md5 list beside the source (read from the copy's disk only)
        sums=$(ls "$(dirname "$first")"/*.md5 2> /dev/null | head -n 1)
        if [ -n "$sums" ] && (cd "$dst" && nice -n 19 ionice -c3 md5sum -c --quiet "$sums") > "$dst.md5check" 2>&1; then
            verdict "stage $(basename "$dst")" OK "$(grep -c . "$sums") shards copied, md5 as $(basename "$sums")"
        else
            verdict "stage $(basename "$dst")" FAIL "${sums:-no md5 list beside the source}; $dst.md5check"
        fi
    done
fi

if step split; then
    for n in $HOTS; do
        py "$TOOL" split --experts "$EXPERTS" --hot "$n" --out-dir "$WORK/split/h$n" > "$WORK/split_h$n.log" 2>&1
        rc=$?
        verdict "split h$n" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(tail -n 1 "$WORK/split_h$n.log")"
    done
fi

if step extract; then
    : "${Q8:?}" "${IMATRIX:?}"
    log "extract: the cold ids of h$largest_cold from the Q8_0"
    mapfile -t S < <(shards "$Q8")
    rm -rf "$C" && mkdir -p "$C"
    py "$TOOL" extract --ids "$WORK/split/h$largest_cold/cold.tsv" --imatrix "$IMATRIX" --out-dir "$C" "${S[@]}" \
        > "$C/extract.log" 2>&1
    rc=$?
    verdict extract "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(tr '\n' ' ' < "$C/extract.log" | cut -c1-300)"
fi

if step quantize; then
    : "${QUANTIZE:?}"
    for r in $RECIPES; do
        IFS=: read -r name gu dn <<< "$r"
        out=$C/experts-$name.gguf
        [ -f "$out" ] && grep -q "^done" "$C/quantize-$name.status" 2> /dev/null && { log "quantize $name: done before"; continue; }
        log "quantize $name: gate/up $gu, down $dn"
        # the extract is Q8_0: a down type of "base" is copied unconverted (tier takes the base's own down)
        qdn=$dn
        [ "$dn" = base ] && qdn=q8_0
        # shellcheck disable=SC2086
        docker run $D --name "pq-coldtier-quantize-$name" $mounts "$IMAGE" nice -n 19 "$QUANTIZE" --allow-requantize \
            --imatrix "$C/imatrix.gguf" --tensor-type "ffn_(gate|up)_exps=$gu" --tensor-type "ffn_down_exps=$qdn" \
            "$C/experts.gguf" "$out.part" Q8_0 "$THREADS" > "$C/quantize-$name.log" 2>&1
        rc=$?
        # every gate/up weight converted to the recipe's type, every down weight to its type (or copied for "base")
        count() { awk -v want="$1" -v conv="$2" '/^\[ *[0-9]+\/ *[0-9]+\]/ && $0 ~ want {
            n++; if (conv == "" ? $0 !~ /converting to/ : index($0, "converting to " conv " ")) k++ } END { print k + 0, n + 0 }' \
            "$C/quantize-$name.log"; }
        read -r gk gn <<< "$(count 'ffn_(gate|up)_exps.weight' "$gu")"
        read -r dk dnn <<< "$(count 'ffn_down_exps.weight' "$([ "$dn" = base ] || echo "$dn")")"
        if [ $rc -eq 0 ] && [ "$gn" -gt 0 ] && [ "$gk" = "$gn" ] && [ "$dnn" -gt 0 ] && [ "$dk" = "$dnn" ]; then
            mv "$out.part" "$out"
            echo "done $(date -Is)" > "$C/quantize-$name.status"
            verdict "quantize $name" OK "$gk gate/up weights to $gu, $dk down $([ "$dn" = base ] && echo "copied (the base's own in tier)" || echo "to $dn"), $(stat -c %s "$out") bytes"
        else
            verdict "quantize $name" FAIL "exit $rc, gate/up $gk of $gn, down $dk of $dnn as wanted, $C/quantize-$name.log"
        fi
    done
fi

# the options of tier and verify for one FILES entry: <N> <recipe> <hot first shard or empty> <base shards...>
tier_opts() {
    local n=$1 recipe=$2 hot=$3
    shift 3
    printf '%s\n' --hot-ids "$WORK/split/h$n/hot.tsv" --cold-ids "$WORK/split/h$n/cold.tsv"
    if [ "$recipe" = base ]; then printf '%s\n' --cold "$@"
    elif [[ "$recipe" == *.gguf ]]; then printf '%s\n' --cold; shards "$recipe"
    else
        printf '%s\n' --cold "$C/experts-$recipe.gguf" --cold-in "$C/ids.tsv"
        # a recipe whose down type is "base" gives only gate and up; the down projections are COLD_REST's or the base's
        local r
        for r in $RECIPES; do
            if [ "${r%%:*}" = "$recipe" ] && [ "${r##*:}" = base ]; then
                printf '%s\n' --cold-only 'ffn_(gate|up|gate_up)_exps'
                [ -z "${COLD_REST:-}" ] || { echo --cold-rest; shards "$COLD_REST"; }
            fi
        done
    fi
    [ -z "$hot" ] || { echo --hot; shards "$hot"; }
}

for f in ${FILES:-}; do
    IFS='|' read -r folder base prefix n recipe hot <<< "$f"
    mapfile -t B < <(shards "$base")
    mapfile -t T < <(tier_opts "$n" "$recipe" "${hot:-}" "${B[@]}")
    o=$OUT/$folder
    if step tier; then
        log "tier $folder: hot $n of ${hot:+the hot file, not }the base, cold $recipe"
        mkdir -p "$o"
        py "$TOOL" tier "${T[@]}" --out-dir "$o" --prefix "$prefix" --link-unchanged "${B[@]}" > "$o/tier.log" 2>&1
        rc=$?
        verdict "tier $folder" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(tail -n 1 "$o/tier.log")"
    fi
    mapfile -t N < <(shards "$o/$prefix-00001-of-0000${#B[@]}.gguf")
    if step verify; then
        log "verify $folder"
        py "$TOOL" verify "${B[@]}" --new "${N[@]}" "${T[@]}" > "$o/verify.log" 2>&1
        rc=$?
        (cd "$o" && for s in "${N[@]}"; do [ -L "$s" ] || nice -n 19 ionice -c3 md5sum "$(basename "$s")"; done) > "$o/md5.txt"
        verdict "verify $folder" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(wc -l < "$o/verify.log") shards; md5 $(cut -c1-8 "$o/md5.txt" | tr '\n' ' ')"
    fi
    if step headers; then
        h=$HEADERS/$folder
        rm -rf "$h" && mkdir -p "$h"
        py "$TOOL" tier "${T[@]}" --out-dir "$h" --prefix "$prefix" --headers-only "${B[@]}" > "$h.log" 2>&1 &&
            py "$TOOL" verify "${B[@]}" --new $(for s in "${N[@]}"; do echo "$h/$(basename "$s")"; done) "${T[@]}" \
                --headers-only >> "$h.log" 2>&1
        rc=$?
        tar -C "$HEADERS" -cSzf "$h.tar.gz" "$folder"
        verdict "headers $folder" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(basename "$h").tar.gz ($(stat -c %s "$h.tar.gz") bytes)"
    fi
done

if step bytes; then
    args=()
    for f in ${FILES:-}; do
        IFS='|' read -r folder base prefix n recipe _hot <<< "$f"
        mapfile -t B < <(shards "$base")
        args+=("$folder=$OUT/$folder/$prefix-00001-of-0000${#B[@]}.gguf")
    done
    # shellcheck disable=SC2206
    args+=(${REFS:-})
    py "$REPO/docs/data/2026-11-04_cold_tier/run/bytes.py" --experts "$EXPERTS" --out "$WORK/bytes.tsv" "${args[@]}" \
        > "$WORK/bytes.log" 2>&1
    rc=$?
    verdict bytes "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$WORK/bytes.tsv"
fi
