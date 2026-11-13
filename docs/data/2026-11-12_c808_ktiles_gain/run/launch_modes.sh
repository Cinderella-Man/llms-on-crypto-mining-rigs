#!/usr/bin/env bash
# launch_modes.sh: R133 (C808, Q1414). The served verify pass's Q4_K ssm_out and attn_output launches (one-op MUL_MAT of
# a 6,144-wide Q4_K weight) on cards 0 to 7, split in two at 158 us: with the tiles off they all run near 148 us, with
# _DMT_KT=1 about half of them run near 171. Times are the profiler's per launch group (2026-11-10_served_verify_parts'
# prof-verify.csv: device, eval, host_ms, group, label, us), with its event cost (about 7.6 us) included. Launch groups of
# 135 to 250 us only (the two modes; the shorter groups and the longer ones, such as prompt chunks, left out).
#   TRACES=<dir with <k>-<arm>/prof/prof-verify.csv> bash launch_modes.sh > launch_modes.tsv
set -euo pipefail
: "${TRACES:?}"
printf 'load\tlaunches_135_158\tmean_us_135_158\tlaunches_158_250\tmean_us_158_250\n'
for f in "$TRACES"/*/prof/prof-verify.csv; do
    d=$(basename "$(dirname "$(dirname "$f")")")
    awk -F, -v d="$d" '$5 ~ /^MUL_MAT (ssm_out|attn_output):q4_K:/ && $1 < 8 && $6 >= 135 && $6 < 250 {
            if ($6 < 158) { a++; sa += $6 } else { b++; sb += $6 } }
        END { printf "%s\t%d\t%.2f\t%d\t%.2f\n", d, a, (a ? sa / a : 0), b, (b ? sb / b : 0) }' "$f"
done
