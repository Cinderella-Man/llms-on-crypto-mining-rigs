#!/usr/bin/env bash
# The tables of kld_tail.sh's logs. For each arm and text: the mean KLD and its standard error, the same-top-token
# rate, the KLD's change against the two controls run on the same cards, paired by chunk (OFF, default "off": the
# switch unset; CTRL, default "p1": the switch on at p = 1, the same graph with nothing dropped), and the routed
# experts kept (mean of the 10 per token and layer, the share of the routed experts' reads that remains, the histogram
# by count). The paired change: llama-perplexity prints each chunk's running mean, so chunk i's own mean is
# i * K_i - (i - 1) * K_(i-1) (every chunk scores the same number of tokens); its standard error is the spread of
# the per-chunk differences over sqrt(chunks).
#   OUT=<kld_tail.sh's OUT> [OFF=off] [CTRL=p1] bash table.sh  > kld_by_arm.tsv
#   OUT=<kld_tail.sh's OUT> LAYERS=1 bash table.sh             > kept_by_layer.tsv (arm, text, layer, mean kept)
set -uo pipefail
: "${OUT:?}"
OFF=${OFF:-off}
CTRL=${CTRL:-p1}

chunks() { # "chunk KLD" of a log, one chunk a line (chunk 1 may share its line with the ETA message)
    awk '{ for (j = 1; j + 10 <= NF; j++)
             if ($j ~ /^[0-9]+$/ && $(j + 2) == "±" && $(j + 5) == "±" && $(j + 8) == "±") {
                 i = $j + 0; k = $(j + 7); if (i == last + 1) printf "%d %.6f\n", i, i * k - (i - 1) * prev
                 last = i; prev = k; break } }' "$1"
}

paired() { # paired <control arm> <arm> <text>: "change se chunks", or "- - -"
    local base=$OUT/$1.$3.log lg=$OUT/$2.$3.log
    if [ "$1" = "$2" ] || ! grep -q "^Mean    KLD" "$base" 2> /dev/null; then echo "- - -"; return; fi
    awk 'NR == FNR { b[$1] = $2; next } ($1 in b) { x = $2 - b[$1]; s += x; ss += x * x; n++ }
         END { if (n > 1) { m = s / n; v = (ss - n * m * m) / (n - 1); if (v < 0) v = 0;
               printf "%.5f %.5f %d\n", m, sqrt(v / n), n } else print "- - -" }' <(chunks "$base") <(chunks "$lg")
}

if [ -n "${LAYERS:-}" ]; then
    printf 'arm\ttext\tlayer\texperts_kept_mean\n'
    for lg in "$OUT"/*.log; do
        f=$(basename "$lg" .log) arm=${f%.*} text=${f##*.}
        grep -o 'mean experts kept per MoE layer:.*' "$lg" | tail -n 1 | cut -d: -f2 |
            awk -v a="$arm" -v t="$text" '{ for (i = 1; i <= NF; i++) printf "%s\t%s\t%d\t%s\n", a, t, i - 1, $i }'
    done
    exit 0
fi

printf 'arm\ttext\tmoe_tail\tmean_kld\tkld_se\tsame_top_p_pct\td_kld_vs_off\td_vs_off_se\td_kld_vs_p1\td_vs_p1_se\tchunks\texperts_kept_mean\trouted_reads_pct\tkept_by_count\n'
for lg in "$OUT"/*.log; do
    f=$(basename "$lg" .log) arm=${f%.*} text=${f##*.}
    grep -q "^Mean    KLD" "$lg" || continue
    kld=$(awk '/^Mean    KLD/ {print $3}' "$lg" | tail -n 1)
    se=$(awk '/^Mean    KLD/ {print $5}' "$lg" | tail -n 1)
    top=$(awk '/^Same top p/ {print $4}' "$lg" | tail -n 1)
    env=$(grep -o 'moe tail ([^)]*)' "$lg" | tail -n 1 | sed 's/moe tail (//; s/)//; s/ = /=/g; s/renorm /renorm=/; s/, /,/g')
    line=$(grep -o 'experts kept: mean [0-9.]* of [0-9]* ([0-9.]* %.*' "$lg" | tail -n 1)
    mean=$(echo "$line" | awk '{print $4}')
    pct=$(echo "$line" | awk '{gsub(/\(/, "", $7); print $7}')
    hist=$(echo "$line" | sed -n 's/.*by count: //p' | sed 's/ %/%/g; s/: /:/g; s/ /,/g')
    read -r d dse n < <(paired "$OFF" "$arm" "$text")
    read -r c cse cn < <(paired "$CTRL" "$arm" "$text")
    [ "$n" = - ] && n=$cn
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$text" "${env:--}" "${kld:--}" "${se:--}" \
        "${top:--}" "$d" "$dse" "$c" "$cse" "$n" "${mean:-10}" "${pct:-100}" "${hist:--}"
done
