#!/usr/bin/env bash
# Q891: the KL divergence of usage-pruned files against the Q8_0 reference's logits of ../../2026-11-03_flash_next_quality,
# with that folder's run/fn2.sh and run/kld.sh: the same three texts, flags and fit (wikitext-2 80 x 512, held-out agent
# sessions 12 x 8,192, Exercism code 80 x 512, -ub 512 -fa on, every layer on the GPUs, the bases' gpu_args), one file
# at a time once the GPUs are free, a file outside FAST first copied to HOT. List the uncut file first: scored again on
# the same cards, it is the control the cuts are read against (the published scores ran on other cards, and bases are
# only exact on the cards they came from).
#   (env: fn2.sh's and kld.sh's PPL GPUS MOUNTS WIKI AGENT CODE TAG HOT FAST [IMAGE PREFLIGHT EXTRA], and BASES OUT
#    FILES) bash kld.sh
# BASES: the folder with base.{wiki,agent,code}.kld and gpu_args, linked into OUT; FILES: a file of
# "<name>=<first shard>" lines, scored in order. Logs that are complete are not run again. Writes
# $OUT/<name>.<text>.log and the table $OUT/kld.tsv.
set -uo pipefail
q=$(cd "$(dirname "$0")/../../2026-11-03_flash_next_quality/run" && pwd)
: "${BASES:?}" "${OUT:?}" "${FILES:?}" "${GPUS:?}" "${HOT:?}" "${FAST:?}"
mkdir -p "$OUT"
for t in wiki agent code; do ln -sfn "$BASES/base.$t.kld" "$OUT/base.$t.kld"; done
cp "$BASES/gpu_args" "$OUT/gpu_args"
KLD=$OUT GPUS_ALL=$GPUS FN2_FILES=$(cat "$FILES") bash "$q/fn2.sh"
failed=$?
{
    printf 'file\ttext\tmean_kld\tkld_err\tsame_top_p_pct\ttop_p_err\tppl_ratio\n'
    while IFS='=' read -r name _first; do
        [ -z "$name" ] && continue
        for t in agent wiki code; do
            lg=$OUT/$name.$t.log
            [ -f "$lg" ] || continue
            awk -v n="$name" -v t="$t" '
                /^Mean    KLD:/ { k = $3; ke = $5 }
                /^Same top p:/ { p = $4; pe = $6 }
                /^Mean PPL\(Q\)\/PPL\(base\)/ { r = $4 }
                END { if (k != "") printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", n, t, k, ke, p, pe, r }' "$lg"
        done
    done < "$FILES"
} > "$OUT/kld.tsv"
echo "table: $OUT/kld.tsv"
exit $failed
