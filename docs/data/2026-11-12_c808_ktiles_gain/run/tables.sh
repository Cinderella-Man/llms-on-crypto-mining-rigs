#!/usr/bin/env bash
# tables.sh: R133 (C808, Q1414). rig_c808.sh's output as tables, on a build machine's CPU at the lowest priority (no
# container, no GPU). Run from a folder holding this repository's docs/data/ and scripts/ (or its root).
# Env: IN (the rig's out/ copied here), TRACES (each served load's prof/prof-verify.csv under <k>-<arm>/), OUT (tables),
#   NICE (default "nice -n 19"), R120B (R120b's perf/wide.tsv, default this repository's)
# Out: $OUT/perf (perf_c808.py: perf.md, wide, spread, verdict, focus, alone), $OUT/served (each load's verify_parts.py
#   folder, served_table.py's served.md and TSVs, served_stack.py's stack.md and stack.tsv)
set -euo pipefail
: "${IN:?}" "${TRACES:?}" "${OUT:?}"
NICE=${NICE:-nice -n 19}
D=docs/data
R120B=${R120B:-$D/2026-11-12_r120b_kern_speed/perf/wide.tsv}
PY="$NICE python3 -I"
mkdir -p "$OUT/perf" "$OUT/served"
ALONE=()
[ -s "$IN/alone/us_run.tsv" ] && ALONE=(--alone "$IN/alone/us_run.tsv")
R=()
[ -s "$R120B" ] && R=(--r120b "$R120B")
$PY $D/2026-11-12_c808_ktiles_gain/run/perf_c808.py "$IN/perf/us_run.tsv" "${ALONE[@]}" "${R[@]}" --out "$OUT/perf" \
    > "$OUT/perf/perf.md"
for t in "$TRACES"/*/prof/prof-verify.csv; do
    d=$(basename "$(dirname "$(dirname "$t")")")
    mkdir -p "$OUT/served/$d"
    $PY $D/2026-11-10_served_verify_parts/run/verify_parts.py \
        --kpass-parts $D/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
        --event-us 7.6 --out "$OUT/served/$d" "$t" > "$OUT/served/$d/report.md"
done
$PY $D/2026-11-11_r79_c722_p106/run/served_table.py "$IN/served" --parts "$OUT/served" --out "$OUT/served" \
    --types iq4_xs,q4_K,q6_K > "$OUT/served/served.md"
$PY $D/2026-11-12_r120b_kern_speed/run/served_stack.py "$OUT/served" --out "$OUT/served" > "$OUT/served/stack.md"
echo "tables done: $OUT"
