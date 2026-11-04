#!/usr/bin/env bash
# C200 (Q750, Q231): what the residual's round trip through F16 or Q8_0 at the rig's eight card boundaries costs in
# KL divergence, on the bases of the Flash-Next quality chain (../../2026-11-03_flash_next_quality: the reference
# Swift-1.5 Flash-Next Q8_0's logits over the three texts, written by llama-perplexity on the GPUs of a build machine)
# and with its kld.sh, unchanged. Each arm runs the reference model itself with the build under test, so the KLD of an
# arm is the cost of its switches plus the floor of that build against the bases' build, which the arm "off" measures:
#   off     the switches off (the floor)
#   f16     LLAMA_HOP_CAST=f16  at the rig's boundaries (LLAMA_HOP_CAST_AT=$AT)
#   bf16    LLAMA_HOP_CAST=bf16 at the same
#   q8_0    LLAMA_HOP_CAST=q8_0 at the same
# The rig's boundaries: Flash-Next on nine cards with -ts 6,5,5,6,5,5,6,5,6 puts layers 0-5, 6-10, 11-15, 16-21,
# 22-26, 27-31, 32-37, 38-42 and 43-47 on cards 0-8, so the residual crosses before layers 6,11,16,22,27,32,38,43. The
# build machine's own split has other boundaries, which stay F32 (LLAMA_HOP_CAST_AT marks the casts' layers instead).
# With MODEL2 (e.g. the rig's Swift Q2_0) the arms off and q8_0 also run on that file as off2 and q8_02 (the cast
# on top of the weights' own error), against the same bases.
# llama-perplexity must be a build of the branch for the build machine's GPUs (cpu_checks.sh STEPS=sm86 makes one);
# each arm gets a wrapper next to it (bin/ppl-hop-<arm>) that sets the arm's variables, so kld.sh needs no change.
#   (env: PPL (the build's llama-perplexity) BASES (the folder with base.<text>.kld and gpu_args) MODEL OUT GPUS
#    MOUNTS WIKI AGENT CODE TAG [ARMS="off f16 q8_0"] [AT=6,11,16,22,27,32,38,43] [MODEL2] [TEXTS] [IMAGE])
#   bash kld_hops.sh
# Output: OUT/hops-<arm>.<text>.log (llama-perplexity's KLD statistics), OUT/kld_hops.tsv (arm, text, mean KLD, its
# error, 99.9 % KLD, same top token), OUT/DONE.
set -uo pipefail
: "${PPL:?}" "${BASES:?}" "${MODEL:?}" "${OUT:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}" "${TAG:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
KLD_SH=${KLD_SH:-$HERE/../../2026-11-03_flash_next_quality/run/kld.sh}
ARMS=${ARMS:-off f16 q8_0}
AT=${AT:-6,11,16,22,27,32,38,43}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
# kld.sh reads the bases and the split from its OUT: link them in, the bases unchanged
for t in ${TEXTS:-wiki agent code}; do
    [ -e "$OUT/base.$t.kld" ] || ln -s "$BASES/base.$t.kld" "$OUT/base.$t.kld"
done
cp "$BASES/gpu_args" "$OUT/gpu_args"

declare -A ENV_OF=([off]="" [f16]="LLAMA_HOP_CAST=f16 LLAMA_HOP_CAST_AT=$AT" [bf16]="LLAMA_HOP_CAST=bf16 LLAMA_HOP_CAST_AT=$AT"
                   [q8_0]="LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=$AT")
bin=$(dirname "$PPL")
runs=""
for a in $ARMS; do runs="$runs $a:$MODEL:hops-$a"; done
if [ -n "${MODEL2:-}" ]; then runs="$runs off:$MODEL2:hops-off2 q8_0:$MODEL2:hops-q8_02"; fi
failed=0
for r in $runs; do
    IFS=: read -r a m name <<< "$r"
    w=$bin/ppl-hop-$a
    printf '#!/usr/bin/env bash\n# kld_hops.sh arm %s\n%s\nexec "$(dirname "$0")/%s" "$@"\n' "$a" \
        "$( [ -n "${ENV_OF[$a]}" ] && echo "export ${ENV_OF[$a]}")" "$(basename "$PPL")" > "$w"
    chmod +x "$w"
    PPL=$w OUT=$OUT GPUS=$GPUS TAG=$TAG MOUNTS=$MOUNTS WIKI=$WIKI AGENT=$AGENT CODE=$CODE TEXTS=${TEXTS:-} \
        bash "$KLD_SH" kld "$m" "$name" || failed=1
    # the boundaries the run cast at (none for off), from llama-perplexity's log
    for t in ${TEXTS:-wiki agent code}; do
        grep -m 1 'card hops in' "$OUT/$name.$t.log" 2>/dev/null | sed "s/^/$name $t: /"
    done
done
printf 'arm\ttext\tkld_mean\tkld_mean_err\tkld_99_9\tsame_top_pct\n' > "$OUT/kld_hops.tsv"
for f in "$OUT"/hops-*.*.log; do
    b=$(basename "$f" .log); arm=${b%.*}; t=${b##*.}
    m=$(awk '/^Mean    KLD:/ { print $3 "\t" $5 }' "$f")
    p=$(awk '/^99.9%   KLD:/ { print $3 }' "$f")
    s=$(awk '/^Same top p:/ { print $4 }' "$f")
    [ -n "$m" ] && printf '%s\t%s\t%s\t%s\t%s\n' "${arm#hops-}" "$t" "$m" "$p" "$s" >> "$OUT/kld_hops.tsv"
done
cat "$OUT/kld_hops.tsv"
date -Is > "$OUT/DONE"
exit $failed
