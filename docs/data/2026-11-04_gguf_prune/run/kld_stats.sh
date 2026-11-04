#!/usr/bin/env bash
# Q891: the distribution lines of each file's KL-divergence logs (the mean is in kld.tsv): median, 99th and 99.9th
# percentile and the largest KLD, and the share of tokens whose top choice stays the same, per file and text. A cut that
# moves the mean through a few tokens shows in the tail; one that moves every token shows in the median.
#   OUT=<the kld.sh run's folder> FILES=<its file list> bash kld_stats.sh > kld_stats.tsv
set -uo pipefail
: "${OUT:?}" "${FILES:?}"
printf 'file\ttext\tmean_kld\tmedian_kld\tp99_kld\tp999_kld\tmax_kld\tsame_top_pct\n'
while IFS='=' read -r name _first; do
    [ -z "$name" ] && continue
    for t in agent wiki code; do
        lg=$OUT/$name.$t.log
        [ -f "$lg" ] || continue
        awk -v n="$name" -v t="$t" '
            /^Mean    KLD:/ { m = $3 }
            /^Median  KLD:/ { md = $3 }
            /^99.0%   KLD:/ { p99 = $3 }
            /^99.9%   KLD:/ { p999 = $3 }
            /^Maximum KLD:/ { mx = $3 }
            /^Same top p:/ { s = $4 }
            END { if (m != "") printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", n, t, m, md, p99, p999, mx, s }' "$lg"
    done
done < "$FILES"
