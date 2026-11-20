#!/usr/bin/env bash
# C898 (Q1623): tests.md from build_check.sh's test outputs (WORK/test/{base,new}-*.txt, copied to OUT).
# Usage: OUT=<folder with the test outputs> bash run/tables.sh > tests.md
set -euo pipefail
: "${OUT:?set OUT to the folder holding base-*.txt and new-*.txt}"
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
md() {  # a test-llama-archs table (lines starting with '|') as a Markdown table
    strip "$1" | grep -a '^|' | awk -F'|' '{
        s = "|"; for (i = 2; i <= NF; i++) { f = $i; gsub(/^ +| +$/, "", f); s = s " " f " |" } print s
        if (NR == 1) { s = "|"; for (i = 2; i <= NF; i++) s = s "---|"; print s } }'
}
echo "# test-llama-archs on qwen4exp, seed 1, CPU: base \`602691153\` and new \`c9c3134d4\`"
echo
echo "Each table is the new tree's. \"Base\" says whether the base tree's output of the same test is the same bytes"
echo "after its first line (a timestamp)."
echo
for t in mtp-kv-only kvonly-sw mtp-graph-keep mtp-replan mtp-catchup-overlap mtp-plain-attn; do
    n=$OUT/new-$t.txt; b=$OUT/base-$t.txt
    case $t in
        kvonly-sw) title="--mtp-kv-only with LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1" ;;
        *) title="--$t" ;;
    esac
    if [ -f "$b" ]; then
        if cmp -s <(tail -n +2 "$b") <(tail -n +2 "$n"); then base="the same bytes"; else base="**differs**"; fi
    else
        base="not in the base tree (the new test)"
    fi
    echo "## \`$title\`"
    echo
    echo "$(strip "$n" | grep -a '^Summary'), $(grep -a '^rc=' "$n"). Base: $base."
    echo
    md "$n"
    echo
done
if [ -f "$OUT/new-archs-all.txt" ]; then
    echo "## Every architecture (\`test-llama-archs -s 1\`)"
    echo
    a=$(strip "$OUT/new-archs-all.txt" | grep -a '^Summary'); r=$(grep -a '^rc=' "$OUT/new-archs-all.txt")
    if cmp -s <(tail -n +2 "$OUT/base-archs-all.txt") <(tail -n +2 "$OUT/new-archs-all.txt"); then base="the same bytes"; else base="**differs**"; fi
    echo "New: $a, $r. Base: $(strip "$OUT/base-archs-all.txt" | grep -a '^Summary'); its output $base."
    echo "qwen4exp on the CPU: $(strip "$OUT/new-archs-all.txt" | grep -a '^| *qwen4exp' | grep -av SKIP |
        awk -F'|' '{ gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $6); gsub(/^ +| +$/, "", $7); print "NMSE against the CPU " $5 ", round trip " $6 ", mixed batch " $7 }')."
fi
if [ -f "$OUT/basetest-mtp-plain-attn.txt" ]; then
    f=$OUT/basetest-mtp-plain-attn.txt
    echo
    echo "## \`--mtp-plain-attn\` on the base's code (\`602691153\` with the new test file only)"
    echo
    echo "$(grep -a '^rc=' "$f") (134: aborted). The rows it printed; the third stops on the assertion:"
    echo
    md "$f"
fi
