#!/usr/bin/env bash
# C831: the tables of docs/data/2026-11-13_c831_upstream from checks.sh's work folder, with the work folder's path
# written as $M. Usage: M=<checks.sh's work root> OUT=<dir> bash tables.sh
set -uo pipefail
M=$(realpath "${M:?M=<work root>}")
OUT=${OUT:?OUT=<dir>}
mkdir -p "$OUT"
clean() { sed "s#$M#\$M#g"; }

# every verdict line of both chains
clean < "$M/summary.tsv" > "$OUT/summary.tsv"
cp "$M/union/union.tsv" "$OUT/union.tsv"

# per random model: the tip's and the revert build's logits against the base's (nmse, max abs diff, differing logits)
{
    printf 'arch\ttip_nmse\ttip_max_abs\ttip_differ\trevert_nmse\trevert_max_abs\trevert_differ\n'
    LC_ALL=C join -t $'\t' \
        <(awk '{ a = $1; sub(":", "", a); for (i = 1; i <= NF; i++) { if ($i == "nmse") n = $(i + 1); if ($i == "diff") d = $(i + 1); if ($i == "of") c = $(i - 1) }
                 gsub(",", "", n); gsub(",", "", d); print a "\t" n "\t" d "\t" c }' "$M/off/tip.lines.txt" | LC_ALL=C sort) \
        <(awk '{ a = $1; sub(":", "", a); for (i = 1; i <= NF; i++) { if ($i == "nmse") n = $(i + 1); if ($i == "diff") d = $(i + 1); if ($i == "of") c = $(i - 1) }
                 gsub(",", "", n); gsub(",", "", d); print a "\t" n "\t" d "\t" c }' "$M/off/revert.lines.txt" | LC_ALL=C sort)
} > "$OUT/logits-by-model.tsv"

# the fork's test-llama-archs checks: mode, verdict, the summary line
{
    printf 'log\tended\tsummary\n'
    for f in "$M"/modes/*.log; do
        printf '%s\t%s\t%s\n' "$(basename "$f" .log)" "$(grep -a -m 1 '^Summary' "$f" > /dev/null && echo done || echo aborted)" \
            "$(grep -a -m 1 '^Summary' "$f" || grep -a 'GGML_ASSERT' "$f" | head -n 1 | clean)"
    done
} > "$OUT/modes.tsv"

# the server checks' verdicts and the draft step's lines
for v in out rout btip brev; do
    [ -f "$M/server/$v/verdict.txt" ] && clean < "$M/server/$v/verdict.txt" > "$OUT/server-$v.txt"
done
for f in qwen4exp-moe-mtp qwen4exp-moe-mtp-draft; do
    [ -f "$M/server/names-$f.txt" ] && cp "$M/server/names-$f.txt" "$OUT/tensors-$f.txt"
done
grep -h -aE 'max_nodes = |graph( \(pp bs=[0-9]+, tg bs=[0-9]+\))?: nodes = |loaded meta data' "$M"/draftmd/*/server.log 2> /dev/null \
    | sed -E 's/^[0-9.]+ [A-Z] //' | clean | sort | uniq -c > "$OUT/draftmd-graphs.txt"

# the GPU functions of llama-server against the base's, SASS and PTX
for w in sass ptx; do
    [ -f "$M/sass/diff_$w.txt" ] && cp "$M/sass/diff_$w.txt" "$OUT/sass-$w-families.txt"
    [ -f "$M/sass/diff_$w.tsv" ] && cp "$M/sass/diff_$w.tsv" "$OUT/sass-$w-by-function.tsv"
done
# the switch lists
[ -f "$M/logs/names-tip.txt" ] && grep '^    "' "$M/logs/names-tip.txt" | tr -d ' ",' > "$OUT/switch-names-tip.txt"
ls -l "$OUT"
