#!/usr/bin/env bash
# Share of each card's samples at boost, base and low SM clock during an agent check, from the gpu.csv that
# bench/agent_check/rig_server.sh writes (nvidia-smi every 10 s: timestamp, index, temperature, power, clocks.sm,
# memory.used, utilization.gpu). A sample counts when the model is loaded (memory.used > 1,000 MiB) and some card
# reports utilization above 0 % at that timestamp. Boost ≥ 1,700 MHz (the P106-090's top bins are ~1,830–1,911),
# base 1,300–1,699 (the 1,354 MHz base clock), low below 1,300.
#   bash clock_share.sh <label>=<gpu.csv> ...   > clock_share.tsv
set -euo pipefail
printf 'run\tcard\tsamples\tboost_pct\tbase_pct\tlow_pct\n'
for spec in "$@"; do
    label=${spec%%=*}; f=${spec#*=}
    awk -F', ' -v label="$label" '
    {
        split($5, c, " "); split($6, m, " "); split($7, u, " "); ts = substr($1, 1, 19)
        if (m[1] + 0 > 1000) { clk[ts "," $2] = c[1]; if (u[1] + 0 > 0) act[ts] = 1 }
    }
    END {
        for (k in clk) {
            split(k, p, ","); if (!(p[1] in act)) continue
            v = clk[k]; g = p[2]; tot[g]++
            if (v >= 1700) boost[g]++; else if (v >= 1300) base[g]++; else low[g]++
        }
        for (g = 0; g in tot; g++)
            printf "%s\t%d\t%d\t%.0f\t%.0f\t%.0f\n", label, g, tot[g], 100 * boost[g] / tot[g], 100 * base[g] / tot[g],
                   100 * low[g] / tot[g]
    }' "$f"
done
