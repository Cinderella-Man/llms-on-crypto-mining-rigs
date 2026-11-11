#!/usr/bin/env bash
# sass_check.sh: C727's static check of the sm_61 code (no GPU), C726's carried to the Q6_K instantiations of C722's
# kernel. From cuobjdump's output of the build's ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-dmt.cu.o
# (-res-usage and -sass), for every mul_mat_vec_q_pascal_dmt<GGML_TYPE_Q6_K, NT, R, PF>:
#   - registers, stack and local memory (spills);
#   - instruction counts of the whole kernel and of its main loop alone (the backward branch of the largest span, from
#     its target to it: one step, i.e. a warp's staging, its next segments' prefetch, one super-block of the lane group's
#     rows for every token, the reduction and the stores):
#     IDP (dp4a), LDS and STS (shared), LDG, FFMA, FMUL, FADD, I2F, MUFU, SHFL, XMAD (integer multiplies on Pascal), PRMT.
#     A step at R = 1 should hold 64 NT dp4a, 25 NT FFMA (two exact products and one sumf a sub-block and token, d a
#     token), 8 NT + 16 FADD (the halves' sums, the 16 scales), 16 FMUL (-12582912 times each scale), no I2F, no XMAD.
# Usage: SASS=<dir with dmt_res.txt, dmt.sass> bash sass_check.sh > sass_counts.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
OPS="IDP LDS STS LDG FFMA FMUL FADD I2F MUFU SHFL XMAD PRMT"
# one instruction a line: "/*addr*/ OPCODE operands"
clean() { sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///;s/^ //;s/ *; *$//' | grep -E '^/\*[0-9a-f]+\*/ '; }
cnt() { grep -cE "^/\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$2" || true; }
loop() { # <clean sass>: the lines of the main loop (a branch target is a bundle's address, so the loop starts at the
         # first instruction at or after it)
    awk 'function hex(s,   i, c, v) { v = 0; for (i = 1; i <= length(s); i++) { c = index("0123456789abcdef", substr(s, i, 1)) - 1; v = v * 16 + c }; return v }
         { a = $1; gsub(/[^0-9a-f]/, "", a); addr[NR] = hex(a); line[NR] = $0
           if ($0 ~ /BRA 0x[0-9a-f]+$/) { t = $NF; sub(/^0x/, "", t); t = hex(t)
               if (addr[NR] - t > best) { best = addr[NR] - t; to = NR; tgt = t } } }
         END { if (to) { for (i = 1; i <= to; i++) if (addr[i] >= tgt) { from = i; break }
                         for (i = from; i <= to; i++) print line[i] } }' "$1"
}
hdr="kernel\tNT\tR\tPF\tREG\tSTACK\tLOCAL"
for op in $OPS; do hdr="$hdr\t$op"; done
for op in $OPS; do hdr="$hdr\tloop $op"; done
echo -e "$hdr\tloop lines"
awk '/Function _Z.*mul_mat_vec_q_pascal_dmtIL9ggml_type14E/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/dmt_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ dmtIL9ggml_type14ELi([0-9])ELi([0-9])ELb([01])E ]] || continue
    nt=${BASH_REMATCH[1]} r=${BASH_REMATCH[2]} pf=${BASH_REMATCH[3]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/dmt.sass" | clean > "$SASS/.k.sass"
    loop "$SASS/.k.sass" > "$SASS/.loop.sass"
    line="mul_mat_vec_q_pascal_dmt<Q6_K>\t$nt\t$r\t$pf\t$reg\t$stack\t$local_"
    for op in $OPS; do line="$line\t$(cnt "$op" "$SASS/.k.sass")"; done
    for op in $OPS; do line="$line\t$(cnt "$op" "$SASS/.loop.sass")"; done
    echo -e "$line\t$(wc -l < "$SASS/.loop.sass")"
done | sort -t$'\t' -k2,2n -k3,3n -k4,4n
rm -f "$SASS/.k.sass" "$SASS/.loop.sass"
