#!/usr/bin/env bash
# rig_prompt_chunks.sh: R131's prompt-chunk table (C811, Q1417, Q1426), on any CPU; nothing here needs a GPU.
# For each load, the second (long) request's prompt evaluations on cards 0-3 and 5-7 (the cards that run no draft
# evaluation), grouped into chunks by each card's evaluation count, with each card's host entry minus card 0's. When
# the host queues a chunk on every card at once, a card enters a few ms after the card before; when it waits for each
# card in turn, about one card's GPU time after it. The prompt's window comes from FN85's timeline.py output for the
# long request (TABLES/timeline/<load>_long.md: T_s, the generation's start, and the prompt time).
#   REPO=<a checkout of this repository> nice -n 19 bash rig_prompt_chunks.sh OUT TABLES   [LOADS='f0 f1 f0-2 f1-2']
set -uo pipefail
OUT=${1:?OUT=<the out/ folder of the rig run>}
TB=${2:?TABLES=<the folder rig_tables.sh wrote>}
: "${REPO:?REPO=<a checkout of this repository>}"
P=$REPO/docs/data/2026-11-10_served_step_split/run
F=$REPO/docs/data/2026-11-12_fn85_server_gap/run
LOADS=${LOADS:-f0 f1 f0-2 f1-2}
mkdir -p "$TB/timeline"
{
echo "| load | chunk | card 0's entry, ms after the prompt starts | cards 1, 2, 3, 5, 6, 7: entry minus card 0's, ms | card 0's GPU time, ms |"
echo "|---|---:|---:|---|---:|"
for l in $LOADS; do
    [ -s "$TB/timeline/${l}_long.md" ] || python3 -I "$F/timeline.py" "$OUT" "$l" long --step-split "$P" > "$TB/timeline/${l}_long.md"
    h=$(head -n 1 "$TB/timeline/${l}_long.md")
    ts=$(sed -E 's/.*T_s ([0-9.]+),.*/\1/' <<< "$h")
    pr=$(sed -E 's/.*prompt ([0-9.]+) ms.*/\1/' <<< "$h")
    awk -F, -v l="$l" -v ts="$ts" -v pr="$pr" '
        NR > 1 && $3 >= ts - pr && $3 < ts && $1 != 4 && $1 != 8 { n = ++k[$1]; t[$1, n] = $3; g[$1, n] = $5 / 1000 }
        END {
            s = ts - pr
            for (i = 1; i <= k[0]; i++) {
                r = ""
                for (d = 1; d <= 7; d++) if (d != 4 && ((d, i) in t)) r = r sprintf("%s%.1f", r == "" ? "" : ", ", t[d, i] - t[0, i])
                printf "| %s | %d | %.1f | %s | %.1f |\n", l, i, t[0, i] - s, r, g[0, i]
            }
        }' "$OUT/$l/host_profile.csv"
done
} > "$TB/prompt_chunks.md"
cat "$TB/prompt_chunks.md"
