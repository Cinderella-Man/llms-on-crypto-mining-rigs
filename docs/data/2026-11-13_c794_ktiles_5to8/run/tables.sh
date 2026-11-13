#!/usr/bin/env bash
# tables.sh: R149's tables from rig_c794.sh's perf/us_run.tsv (run on a build machine, not on the rig).
#   perf.tsv     per type and token count: us a product of each arm, pass by pass, and kt against the faster of off and
#                kern, pass by pass
#   verify.tsv   per token count and arm: est. ms of a verify pass's 2,560 x 6,144 IQ4_XS and Q4_K products, pass by pass:
#                COUNTS products a pass of each type times its us (default iq4_xs:17 q4_K:15, est. from R120b's served
#                4-token pass: IQ4_XS ssm_out 1.912 ms at 112.60 / 113.33 us a product, Q4_K ssm_out and attn_output
#                1.552 + 0.567 ms at 142.29 / 141.75)
# Usage: IN=<perf/us_run.tsv> OUT=<dir> [COUNTS="iq4_xs:17 q4_K:15"] bash tables.sh
set -euo pipefail
: "${IN:?}" "${OUT:?}"
COUNTS=${COUNTS:-iq4_xs:17 q4_K:15}
mkdir -p "$OUT"
awk -F'\t' 'BEGIN { OFS = "\t" }
    { us[$3 FS $6 FS $1 FS $2] = $7; ty[$3] = 1; nt[$6] = 1; ar[$1] = 1; ps[$2] = 1 }
    END {
        printf "type\ttokens"
        for (a in ar) for (p in ps) printf "\t%s pass %s", a, p
        for (p in ps) printf "\tkt - min(off, kern) pass %s", p
        print ""
        for (t in ty) for (n in nt) {
            printf "%s\t%s", t, n
            for (a in ar) for (p in ps) printf "\t%s", us[t FS n FS a FS p]
            for (p in ps) {
                o = us[t FS n FS "off" FS p]; k = us[t FS n FS "kern" FS p]; m = (k != "" && k < o) ? k : o
                printf "\t%+.2f", us[t FS n FS "kt" FS p] - m
            }
            print ""
        }
    }' "$IN" | { read -r h; echo "$h"; sort -k1,1 -k2,2n; } > "$OUT/perf.tsv"
awk -F'\t' -v counts="$COUNTS" 'BEGIN { OFS = "\t"; n = split(counts, c, " "); for (i = 1; i <= n; i++) { split(c[i], kv, ":"); cnt[kv[1]] = kv[2] } }
    ($3 in cnt) && $6 >= 4 { ms[$6 FS $1 FS $2] += cnt[$3] * $7 / 1000; key[$6 FS $1 FS $2] = 1 }
    END { print "tokens", "arm", "pass", "est. ms a pass (" counts ")"
          for (k in key) { split(k, f, FS); printf "%s\t%s\t%s\t%.3f\n", f[1], f[2], f[3], ms[k] } }' "$IN" \
    | { read -r h; echo "$h"; sort -k1,1n -k2,2 -k3,3n; } > "$OUT/verify.tsv"
cat "$OUT/perf.tsv" "$OUT/verify.tsv"
