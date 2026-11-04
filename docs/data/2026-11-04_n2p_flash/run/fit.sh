#!/usr/bin/env bash
# The files' fit on nine 6 GB cards from their GGUF headers alone (bench/file_choice/fit.py: every context, cache type,
# micro-batch and draft of its cost table, whole layers and with routed-expert tensors moved by -ot), on any machine
# with Python 3. Each file is fitted once its build marker exists; a missing marker after its build ended skips it.
#   REPO=<this repository> OUT=<dir> FILES='<name>=<first shard>[@<build marker>] ...' bash fit.sh
# Output: OUT/fit.tsv (fit.py's rows for every file), OUT/fit_64k.tsv (per file, cache type and draft at 64k: the
#   largest -ub that leaves at least 150 MiB on every card, on whole layers and with experts moved, its free MiB and -ts)
set -uo pipefail
: "${REPO:?}" "${OUT:?}" "${FILES:?}"
mkdir -p "$OUT"
shards() { ls "${1%-00001-of-*}"-0000?-of-0000?.gguf | paste -sd,; }
for spec in $FILES; do
    name=${spec%%=*} rest=${spec#*=}
    first=${rest%%@*} marker=""
    [ "$rest" != "$first" ] && marker=${rest#*@}
    if [ -n "$marker" ]; then
        until [ -e "$marker" ]; do
            ps -eo args | grep -q "[q]uantize.sh" || { echo "$(date -Is) $name: no build running and no $marker: skipped"; continue 2; }
            sleep 60
        done
    fi
    echo "$(date -Is) $name: fit from $(shards "$first" | tr ',' ' ')"
    python3 -I "$REPO/bench/file_choice/fit.py" "$name=$(shards "$first")" > "$OUT/fit-$name.tsv" || continue
done
{ head -n 1 "$(ls "$OUT"/fit-*.tsv | head -n 1)"; for f in "$OUT"/fit-*.tsv; do tail -n +2 "$f"; done; } > "$OUT/fit.tsv"
awk -F'\t' 'NR > 1 && $2 == 65536 {
        k = $1 "\t" $3 "\t" $5 "\t" $6
        if ($7 >= 150 && (!(k in ub) || $4 + 0 > ub[k] + 0)) { ub[k] = $4; free[k] = $7; ts[k] = $8; mv[k] = $9 }
        if (!(k in best) || $7 + 0 > best[k] + 0) best[k] = $7
        seen[k] = 1 }
    END { print "file\tkv\tdraft\tmode\tlargest_ub\tsmallest_free_mib\tts\tmoved"
          for (k in seen) print k "\t" (k in ub ? ub[k] "\t" free[k] "\t" ts[k] "\t" mv[k] : "no\t" best[k] "\t\t") }' \
    "$OUT/fit.tsv" | sort > "$OUT/fit_64k.tsv"
cat "$OUT/fit_64k.tsv"
