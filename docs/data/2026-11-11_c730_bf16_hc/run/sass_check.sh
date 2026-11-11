#!/usr/bin/env bash
# sass_check.sh: C730's static read of the sm_61 code of Miner Llama's Pascal BF16 kernel (no GPU). From cuobjdump's
# output of the build's object (ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvf-pascal.cu.o: -res-usage and -sass),
# for every mul_mat_vec_bf16_pascal<G, R, T, has_bias = false>, T = 1 to 8:
#   - registers, stack and local memory (spills);
#   - the main loop (the instructions from the backward branch's target to the branch; the source unrolls it twice, so
#     one pass of it is two 8-weight chunks of each of the lane's R rows): LDG (all of them 128-bit), FFMA, the BF16
#     to FP32 conversions (SHL and LOP32I), and other instructions;
#   - SHFL and STG in the whole kernel.
# A loop LDG count of 2R + 4T means one 16-byte load per row and chunk, whatever the number of tokens, and two per
# token and chunk (its 8 FP32 activations), whatever the number of rows; 16R conversions mean each weight is converted
# once for all tokens.
# Usage: SASS=<dir with mmvf_res.txt and mmvf.sass> bash sass_check.sh > sass_counts.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
# hex to number in any awk (mawk has no strtonum)
HEX='function hex(s, i, n) { n = 0; s = tolower(s); sub(/^0x/, "", s)
    for (i = 1; i <= length(s); ++i) n = n * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1; return n }'
echo -e "G\tR\tT\tREG\tSTACK\tLOCAL\tloop_LDG\tloop_LDG_2R+4T\tloop_FFMA\tloop_FFMA_16RT\tloop_SHL\tloop_LOP32I\tloop_other\tSHFL\tSTG"
awk '/Function _Z23mul_mat_vec_bf16_pascal/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/mmvf_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ ILi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELb0E ]] || continue
    g=${BASH_REMATCH[1]} r=${BASH_REMATCH[2]} t=${BASH_REMATCH[3]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/mmvf.sass" |
        sed -E 's/\/\* 0x[0-9a-f]+ \*\///;s/ +/ /g' | grep -E '^ /\*[0-9a-f]+\*/ ' > "$SASS/.k.sass" || true
    # the main loop: the widest backward branch, from its target to itself
    read -r from to < <(awk "$HEX"'{ a = hex(substr($1, 3, length($1) - 4)) }
        / BRA 0x/ { for (i = 1; i <= NF; ++i) if ($i == "BRA") { b = hex($(i + 1)); if (b < a && a - b > e - s) { s = b; e = a } } }
        END { print s, e }' "$SASS/.k.sass")
    awk -v s="$from" -v e="$to" "$HEX"'{ a = hex(substr($1, 3, length($1) - 4)) } a >= s && a <= e' \
        "$SASS/.k.sass" > "$SASS/.loop.sass"
    cnt() { grep -cE "^ /\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$2" || true; }
    ldg=$(cnt LDG "$SASS/.loop.sass")
    ffma=$(cnt FFMA "$SASS/.loop.sass")
    shl=$(cnt SHL "$SASS/.loop.sass")
    lop=$(cnt LOP32I "$SASS/.loop.sass")
    all=$(grep -cvE ' (NOP|SYNC)? ?;$' "$SASS/.loop.sass" || true)
    other=$((all - ldg - ffma - shl - lop))
    echo -e "$g\t$r\t$t\t$reg\t$stack\t$local_\t$ldg\t$((2 * r + 4 * t))\t$ffma\t$((16 * r * t))\t$shl\t$lop\t$other\t$(cnt SHFL "$SASS/.k.sass")\t$(cnt STG "$SASS/.k.sass")"
done | sort -t$'\t' -k1,1n -k2,2n -k3,3n
rm -f "$SASS/.k.sass" "$SASS/.loop.sass"
