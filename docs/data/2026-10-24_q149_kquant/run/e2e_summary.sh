#!/usr/bin/env bash
# Q149 e2e summary (2026-10-24): perplexity and KLD statistics of each run, and the greedy texts with the Pascal kernels
# off against forced (identical, or the length of the common prefix). Its output is ../e2e_summary.txt.
# Usage: WORK=<dir> e2e_summary.sh   (after run_all.sh and e2e_ctl.sh)
cd ${WORK:?WORK=<work dir>}/e2e
for s in ppl-off1 ppl-on1 ppl-off4 ppl-on4 ppl-skip1 ppl-on-vs-skip1; do
    f=logs/$s.log
    [ -f $f ] || continue
    echo "== $s ($(awk -v s="== $s " 'index($0, s) == 1 { print; getline; print; exit }' logs/progress.txt | xargs))"
    tr '\r' '\n' < $f | grep -aE "Final estimate|^Mean PPL|^Mean +KLD|^Maximum KLD|^99.9% +KLD|^99.0% +KLD|^Median +KLD|^Same top p|^RMS Δp|^Mean +Δp|^Maximum Δp|ln\(PPL" | sed 's/^/   /'
done
echo "== greedy text, kernels off against forced"
for p in 1 3; do
    a=gen-off-p$p.txt; b=gen-on-p$p.txt
    [ -f $a ] && [ -f $b ] || { echo "   p$p: missing"; continue; }
    if cmp -s $a $b; then
        echo "   p$p: identical ($(wc -c < $a) bytes)"
    else
        n=$(cmp $a $b | grep -oE "byte [0-9]+" | grep -oE "[0-9]+")
        echo "   p$p: differ from byte $n of $(wc -c < $a) / $(wc -c < $b)"
    fi
done
