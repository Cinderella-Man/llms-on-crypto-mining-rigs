#!/usr/bin/env bash
# sass_x4.sh: C750's static read of the sm_61 code of the Pascal BF16 kernels at 3-4 tokens (no GPU), from cuobjdump's
# output of the build's ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvf-pascal.cu.o (-res-usage and -sass), for
# GGML_CUDA_MMVF_PASCAL_EXACT's kernels mul_mat_vec_bf16_pascal_x4p<R, T, fault> (=1, the four chains in passes) and
# mul_mat_vec_bf16_pascal_x4<R, T, fault> (=2, interleaved; x4i below) and, beside them,
# mul_mat_vec_bf16_pascal<G, 4, T, false> at G = 32 and 128, T = 3 and 4 (the shapes the rule and RULE=1 give the
# 10,240-wide rows):
#   - registers, stack and local memory (spills);
#   - the main loop (the instructions from the widest backward branch's target to the branch; x4p has four, one a pass,
#     alike): LDG, FFMA, the BF16 to FP32 conversions (SHL and LOP32I) and other instructions. One pass of it is 4 chunks
#     of 8 weights of each of the lane's R rows in x4i (#pragma unroll 1 over 4 chunks) and 2 in the others (#pragma
#     unroll 2), so a kernel loads once per row and chunk if its LDG count is 2R + 4T a 2-chunk pass (x4i: 4R + 8T), and
#     converts each weight once if it has 16R conversions (x4i: 32R);
#   - SHFL, STG and BAR in the whole kernel (x4p's and x4i's butterflies: 5 x 4 x T x R; G = 32's: 5 T R);
#   - the stack's stores and loads (STL, LDL) in the whole kernel and within any loop (from a backward branch's target
#     to the branch): spills outside the loops cost a few accesses a call, inside them a few a chunk.
# Usage: SASS=<dir with mmvf_res.txt and mmvf.sass> bash sass_x4.sh > sass_x4.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
# hex to number in any awk (mawk has no strtonum)
HEX='function hex(s, i, n) { n = 0; s = tolower(s); sub(/^0x/, "", s)
    for (i = 1; i <= length(s); ++i) n = n * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1; return n }'
echo -e "kernel\tG\tR\tT\tfault\tREG\tSTACK\tLOCAL\tchunks_a_pass\tloop_LDG\tloop_LDG_expected\tloop_FFMA\tloop_FFMA_expected\tloop_SHL\tloop_LOP32I\tloop_other\tSHFL\tSTG\tBAR\tSTL_LDL\tSTL_LDL_in_loops"
awk '/Function _Z(23|26|27)mul_mat_vec_bf16_pascal/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/mmvf_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    if [[ $fn =~ _x4pILi([0-9]+)ELi([0-9]+)ELb([01])E ]]; then
        k=x4p g=32 r=${BASH_REMATCH[1]} t=${BASH_REMATCH[2]} fault=${BASH_REMATCH[3]} c=2
    elif [[ $fn =~ _x4ILi([0-9]+)ELi([0-9]+)ELb([01])E ]]; then
        k=x4i g=32 r=${BASH_REMATCH[1]} t=${BASH_REMATCH[2]} fault=${BASH_REMATCH[3]} c=4
    elif [[ $fn =~ pascalILi(32|128)ELi4ELi([34])ELb0E ]]; then
        k=base g=${BASH_REMATCH[1]} r=4 t=${BASH_REMATCH[2]} fault=- c=2
    else
        continue
    fi
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
    # the stack's stores and loads, and those within any backward branch's range
    read -r spill spill_loop < <(awk "$HEX"'{ a = hex(substr($1, 3, length($1) - 4)) }
        / (STL|LDL)[ .]/ { s[++n] = a }
        / BRA 0x/ { for (i = 1; i <= NF; ++i) if ($i == "BRA") { b = hex($(i + 1)); if (b < a) { lo[++m] = b; hi[m] = a } } }
        END { k = 0; for (i = 1; i <= n; ++i) for (j = 1; j <= m; ++j) if (s[i] >= lo[j] && s[i] <= hi[j]) { ++k; break }
              print n + 0, k }' "$SASS/.k.sass")
    echo -e "$k\t$g\t$r\t$t\t$fault\t$reg\t$stack\t$local_\t$c\t$ldg\t$((c * r + 2 * c * t))\t$ffma\t$((8 * c * r * t))\t$shl\t$lop\t$other\t$(cnt SHFL "$SASS/.k.sass")\t$(cnt STG "$SASS/.k.sass")\t$(cnt BAR "$SASS/.k.sass")\t$spill\t$spill_loop"
done | sort -t$'\t' -k1,1 -k2,2n -k3,3n -k4,4n -k5,5
rm -f "$SASS/.k.sass" "$SASS/.loop.sass"
