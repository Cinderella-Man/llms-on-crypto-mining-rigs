#!/usr/bin/env bash
# r149_diffs.sh: R149's verdict tables from rig_c794.sh's perf/us_run.tsv (run on a build machine, not on the rig).
#   diffs.tsv         per type, token count and pass: us a product of kt - kern, ktm1 - kern and ktm1 - kt, and the spread
#                     of kern's two passes (|pass 1 - pass 2|)
#   verify_diffs.tsv  per token count and pass: est. ms of a verify pass's 2,560 x 6,144 products (COUNTS, as tables.sh)
#                     of kt - kern, kt - off, ktm1 - kern and ktm1 - kt
# Usage: IN=<perf/us_run.tsv> OUT=<dir> [COUNTS="iq4_xs:17 q4_K:15"] bash r149_diffs.sh
set -euo pipefail
: "${IN:?}" "${OUT:?}"
COUNTS=${COUNTS:-iq4_xs:17 q4_K:15}
mkdir -p "$OUT"
awk -F'\t' 'BEGIN { OFS = "\t" }
    { us[$1 FS $2 FS $3 FS $6] = $7; key[$3 FS $6] = 1 }
    END {
        print "type", "tokens", "pass", "kt - kern", "ktm1 - kern", "ktm1 - kt", "kern spread"
        for (k in key) {
            split(k, f, FS); t = f[1]; n = f[2]
            s = us["kern" FS 1 FS t FS n] - us["kern" FS 2 FS t FS n]; if (s < 0) s = -s
            for (p = 1; p <= 2; p++) {
                kt = us["kt" FS p FS t FS n]; km = us["kern" FS p FS t FS n]; m1 = us["ktm1" FS p FS t FS n]
                printf "%s\t%s\t%d\t%+.2f\t%+.2f\t%+.2f\t%.2f\n", t, n, p, kt - km, m1 - km, m1 - kt, s
            }
        }
    }' "$IN" | { read -r h; echo "$h"; sort -k1,1 -k2,2n -k3,3n; } > "$OUT/diffs.tsv"
awk -F'\t' -v counts="$COUNTS" 'BEGIN { OFS = "\t"; n = split(counts, c, " "); for (i = 1; i <= n; i++) { split(c[i], kv, ":"); cnt[kv[1]] = kv[2] } }
    ($3 in cnt) { ms[$1 FS $2 FS $6] += cnt[$3] * $7 / 1000; key[$6 FS $2] = 1 }
    END {
        print "tokens", "pass", "kt - kern", "kt - off", "ktm1 - kern", "ktm1 - kt"
        for (k in key) {
            split(k, f, FS); n = f[1]; p = f[2]
            kt = ms["kt" FS p FS n]; km = ms["kern" FS p FS n]; o = ms["off" FS p FS n]; m1 = ms["ktm1" FS p FS n]
            printf "%s\t%s\t%+.3f\t%+.3f\t%+.3f\t%+.3f\n", n, p, kt - km, kt - o, m1 - km, m1 - kt
        }
    }' "$IN" | { read -r h; echo "$h"; sort -k1,1n -k2,2n; } > "$OUT/verify_diffs.tsv"
cat "$OUT/diffs.tsv" "$OUT/verify_diffs.tsv"
