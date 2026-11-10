#!/usr/bin/env bash
# C662 (Q1150): the published tables of this folder, made on the build machine from the checks' work folders. Paths are
# replaced by the names of the variables that held them.
# Usage: WORK=<checks.sh's WORK> STOCK=<stock_identity.sh's WORK> NOREV=<stock_identity.sh's WORK of the REVERT run>
#   DRAFTER=<drafter_revert.sh's WORK> [PTX_OPS=<dir with one folder per group of the operator checks>] [EXTRA=<dir with
#   the repeat runs>] BASE_BUILD=<release rig tree> BASE_CPU=<release CPU bin/> PREV_CPU=<last upstream line's CPU
#   bin/> OUT=<dir> bash tables.sh
set -uo pipefail
: "${WORK:?}" "${STOCK:?}" "${NOREV:?}" "${DRAFTER:?}" "${BASE_BUILD:?}" "${BASE_CPU:?}" "${PREV_CPU:?}" "${OUT:?}"
mkdir -p "$OUT"
scrub() {
    sed -e "s#$BASE_BUILD#\$BASE_BUILD#g" -e "s#$BASE_CPU#\$BASE_CPU#g" -e "s#$PREV_CPU#\$PREV_CPU#g" -e "s#$WORK#\$WORK#g" \
        -e "s#$STOCK#\$STOCK#g" -e "s#$NOREV#\$NOREV#g" -e "s#$DRAFTER#\$DRAFTER#g" ${PTX_OPS:+-e "s#$PTX_OPS#\$PTX_OPS#g"} \
        ${EXTRA:+-e "s#$EXTRA#\$EXTRA#g"} -e 's/  *========== == CUDA == ==========.*$//' -e 's/\x1b\[[0-9;]*m//g'
}
# every verdict line of the CPU checks
scrub < "$WORK/summary.tsv" > "$OUT/summary.tsv"
for s in static shared; do [ -f "$WORK/c565-$s/summary.tsv" ] && scrub < "$WORK/c565-$s/summary.tsv" > "$OUT/summary-c565-$s.tsv"; done
[ -f "$WORK/c80/summary.tsv" ] && scrub < "$WORK/c80/summary.tsv" > "$OUT/summary-c80.tsv"
[ -f "$WORK/c524/summary.tsv" ] && scrub < "$WORK/c524/summary.tsv" > "$OUT/summary-c524.tsv"
# the union check: per file, the lines of neither upstream nor a commit of the branch
cp "$WORK/checks/union/union.tsv" "$OUT/union.tsv"
# per model, the logits against each base: nmse (0 = the same bits) or "-" where the base has no such model; in
# alphabetical order (stock's run has every model)
nm() { # <merged.log>: model, nmse, max abs diff, differing logits, for every compared model
    grep 'logits against' "$1" | sed -E 's/^([^:]+): .*nmse ([0-9.e+-]+), max abs diff ([0-9.e+-]+), ([0-9]+) of ([0-9]+) differ.*/\1\t\2\t\3\t\4/' | sort -u
}
R=$WORK/checks/release
{
    printf 'model\trelease_rig_nmse\trelease_cpu_nmse\trelease_cpu_max_abs_diff\trelease_cpu_logits_differing\tprev_upstream_line_cpu_nmse\trelease_cpu_nmse_with_31854_reverted\tstock_b12246_cpu_nmse\tstock_logits_differing\n'
    awk -F'\t' -v OFS='\t' '
        FNR == 1 { fi++ }
        { k = $1; if (!(k in seen)) { seen[k] = 1; order[++n] = k } v[fi, k] = $2; d[fi, k] = $3; c[fi, k] = $4 }
        function g(a, i, k) { return ((i, k) in a) ? a[i, k] : "-" }
        END { for (i = 1; i <= n; i++) { k = order[i]; print k, g(v, 2, k), g(v, 3, k), g(d, 3, k), g(c, 3, k), g(v, 4, k), g(v, 5, k), g(v, 1, k), g(c, 1, k) } }' \
        <(nm "$STOCK/stock/cmp/merged.log") <(nm "$R/rig/merged.log") <(nm "$R/cpu/merged.log") <(nm "$R/prev-cpu/merged.log") \
        <(nm "$NOREV/revert/cmp/merged.log")
} > "$OUT/logits-by-model.tsv"
# the kernels whose sm_61 code differs from the release's, by name
S=$WORK/checks/c524/sass
if [ -d "$S" ]; then
    {
        printf 'class\tkernel\tinstances\n'
        for c in differ new gone; do
            sed -E 's/<.*//; s/\(.*//; s/^void //' "$S/$c.demangled.txt" | sort | uniq -c | awk -v c=$c '{ n=$1; $1=""; sub(/^ /, ""); print c "\t" $0 "\t" n }'
        done
    } > "$OUT/sass-by-kernel.tsv"
fi
# test-llama-context's cases on the random models
K=$WORK/checks/context
[ -f "$K/default.log" ] && { printf 'model\tcase\tverdict\n'; tr ' ' '\t' < "$K/default.log"; } > "$OUT/context-cases.tsv"
# stock upstream, the revert, the drafter
{
    echo "stock b12246 with the fork's test file against the branch's CPU build:"
    scrub < "$STOCK/stock/cmp/verdict.txt"
    sed 's/^/    /' "$STOCK/stock/cmp/differ.txt"
    [ -f "$STOCK/flips/summary.txt" ] && { echo "single settings against stock's logits:"; sed 's/^/    /' "$STOCK/flips/summary.txt"; }
    echo "the release's CPU build against the branch with upstream 12ab59798 (#31854) reverted:"
    scrub < "$NOREV/revert/cmp/verdict.txt"
    echo "the staircase drafter's check (llama-pipe-drafter-check, native CPU builds):"
    sed 's/^/    /' "$DRAFTER/verdict.txt"
} > "$OUT/stock-and-revert.txt"
# the operator checks through the compute 6.1 PTX
if [ -n "${PTX_OPS:-}" ]; then
    {
        for g in off on archs archs-nogdn archs-cuda0; do
            [ -f "$PTX_OPS/$g/console.txt" ] && { echo "== group $g"; scrub < "$PTX_OPS/$g/console.txt"; }
            for f in "$PTX_OPS/$g"/logs/tbo-*.log; do
                [ -f "$f" ] && sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -a 'FAIL\|CUDA error\|tests passed' | sed 's/^ */    /' | sort | uniq -c | scrub
            done
        done
        for d in shards shards-support off-nodup case-dup case-iq cpy-reps; do
            [ -f "$PTX_OPS/$d/summary.txt" ] && { echo "== $d"; scrub < "$PTX_OPS/$d/summary.txt"; }
        done
    } > "$OUT/ptx-ops.txt"
fi
[ -f "$WORK/ptx/console.txt" ] && scrub < "$WORK/ptx/console.txt" > "$OUT/ptx.txt"
# the repeated runs
if [ -n "${EXTRA:-}" ]; then
    {
        # whole runs (every model in one process): the build's own logits at its defaults, compared again and again
        for d in "$EXTRA"/repeat-full* "$EXTRA"/repeat-flip*; do
            [ -f "$d/summary.txt" ] || continue
            echo "== $(basename "$d"): $(wc -l < "$d/summary.txt") whole runs ($(grep -c '^default' "$d/summary.txt") at the defaults, $(grep -c '^flip' "$d/summary.txt") with $(grep -m 1 '^flip' "$d/summary.txt" | sed -E 's/^[^(]*\(([^)]*)\).*/\1/')), $(grep -c 'differ: [a-z]' "$d/summary.txt") with a difference"
            grep 'differ: [a-z]' "$d/summary.txt" | sed 's/^/    /'
        done
        [ -f "$EXTRA/repeat-deepseek4/summary.txt" ] && { echo "== deepseek4 alone, 12 runs per line, against stock's logits"; sed 's/^/    /' "$EXTRA/repeat-deepseek4/summary.txt"; }
        for n in l6 stock; do
            ls "$EXTRA"/par-$n-*/$n.default.txt > /dev/null 2>&1 || continue
            echo "== deepseek4 alone, eight runs at a time, $n: default $(cat "$EXTRA"/par-$n-*/$n.default.txt | sort | uniq -c | tr -s ' \n' ' '); LLAMA_KV_HEAD_TAIL=0 $(cat "$EXTRA"/par-$n-*/$n.flip.txt | sort | uniq -c | tr -s ' \n' ' ')"
        done
        for d in "$EXTRA"/gdn-*; do [ -f "$d/summary.txt" ] && { echo "== fused against unfused GDN, $(basename "$d")"; sed 's/^/    /' "$d/summary.txt"; }; done
    } > "$OUT/repeats.txt"
fi
ls -la "$OUT"
