#!/usr/bin/env bash
# shexp_times.sh: the shared expert's launches in R41's profile of the release line (docs/data/2026-11-10_release_profile),
# per launch group and pass size: calls per pass, GPU time per pass (ms) and per call (us), the profiler's cost per span
# subtracted as R41 does (7.6 us). Reads R41's kept rows (release_parts.py's <tag>-k<k>.csv) and its cut.tsv (kept passes
# per request). Every launch group whose label names a _shexp tensor is listed, also the fused routed + shared launch
# (R41 counts that one under the routed experts) and the gate's multiply (under other operations).
# Usage: RES=<release_parts.py's --out dir> [KS="1 4 5"] [EVENT_US=7.6] bash shexp_times.sh > times.tsv
set -euo pipefail
: "${RES:?RES=<dir with release-k<k>.csv and cut.tsv>}"
KS=${KS:-1 4 5}
EVENT_US=${EVENT_US:-7.6}
echo -e "k\tops\ttensors\tcalls_per_pass\tms_per_pass\tus_per_call\tR41_part"
for k in $KS; do
    kept=$(awk -F'\t' -v k="$k" '$2 == k { print $6 }' "$RES/cut.tsv")
    awk -F, -v k="$k" -v p="$kept" -v ev="$EVENT_US" '
        NR > 1 && $5 ~ /shexp/ {
            s[$5] += $6 - ev; n[$5]++
        }
        END {
            for (l in s) {
                ops = l; sub(/ .*/, "", ops)
                t = l; sub(/^[^ ]+ /, "", t)
                part = l ~ /^MUL_MAT_ID/ ? "routed experts" : l ~ /_shexp:/ ? "shared expert" : "other operations"
                printf "%d\t%s\t%s\t%.2f\t%.3f\t%.2f\t%s\n", k, ops, t, n[l] / p, s[l] / 1000 / p, s[l] / n[l], part
            }
        }' "$RES/release-k$k.csv" | sort -t$'\t' -k5,5gr
done
