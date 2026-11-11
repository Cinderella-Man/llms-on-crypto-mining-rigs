#!/usr/bin/env bash
# sass_check.sh: C726's static check of the sm_61 code (no GPU), C722's carried to mmvq-pascal-q3kmt.cu. From cuobjdump's
# output of the build's objects (ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-q3kmt.cu.o: -res-usage and
# -sass; mmvq-pascal-kcol-q3_K.cu.o: -sass), for every mul_mat_vec_q_pascal_q3kmt<NT, R, PF>:
#   - registers, stack and local memory (spills);
#   - instruction counts of the whole kernel and of its main loop alone (the backward branch of the largest span, from
#     its target to it: one step, i.e. a warp's staging, its next segments' prefetch, one super-block of the lane group's
#     rows for every token, the reduction and the stores):
#     IDP (dp4a), LDS and STS (shared), LDG, FFMA, FMUL, FADD, I2F, MUFU, SHFL, XMAD (integer multiplies on Pascal), PRMT;
#   - the activation staging's d8 / f: the three instructions from the I2F.F32.S32 of f on, beside those of the k-column
#     kernel's quantizer (kcol::quantize_records<KQ3_K>): both I2F, MUFU.RCP, FMUL, so both give the same value.
# Usage: SASS=<dir with q3kmt_res.txt, q3kmt.sass, kcol_q3k.sass.gz> bash sass_check.sh > sass_counts.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
OPS="IDP LDS STS LDG FFMA FMUL FADD I2F MUFU SHFL XMAD PRMT"
# one instruction a line: "/*addr*/ OPCODE operands"
clean() { sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///;s/^ //;s/ *; *$//' | grep -E '^/\*[0-9a-f]+\*/ '; }
cnt() { grep -cE "^/\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$2" || true; }
d8f() { # <clean sass>: the I2F.F32.S32 (not .RP) and the next MUFU and FMUL
    local n
    n=$(grep -nE '^/\*[0-9a-f]+\*/ (@!?P[0-9] )?I2F\.F32\.S32 ' "$1" | head -n 1 | cut -d: -f1)
    [ -n "$n" ] || return 0
    sed -n "${n},\$p" "$1" | grep -E '^/\*[0-9a-f]+\*/ (@!?P[0-9] )?(I2F|MUFU|FMUL)' | head -n 3 \
        | sed -E 's/^\/\*[0-9a-f]+\*\/ //;s/ R[0-9]+.*//' | tr '\n' ' ' | sed 's/ $//'
}
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
echo -e "$hdr\tloop lines\td8 / f"
awk '/Function _Z.*mul_mat_vec_q_pascal_q3kmt/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/q3kmt_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ q3kmtILi([0-9])ELi([0-9])ELb([01])E ]] || continue
    nt=${BASH_REMATCH[1]} r=${BASH_REMATCH[2]} pf=${BASH_REMATCH[3]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/q3kmt.sass" | clean > "$SASS/.k.sass"
    loop "$SASS/.k.sass" > "$SASS/.loop.sass"
    line="mul_mat_vec_q_pascal_q3kmt\t$nt\t$r\t$pf\t$reg\t$stack\t$local_"
    for op in $OPS; do line="$line\t$(cnt "$op" "$SASS/.k.sass")"; done
    for op in $OPS; do line="$line\t$(cnt "$op" "$SASS/.loop.sass")"; done
    echo -e "$line\t$(wc -l < "$SASS/.loop.sass")\t$(d8f "$SASS/.k.sass")"
done
zcat "$SASS/kcol_q3k.sass.gz" | awk '/Function : /{ f = ($3 ~ /quantize_records/) } f' | clean > "$SASS/.q.sass"
line="kcol::quantize_records<KQ3_K>\t\t\t\t\t\t"
for op in $OPS $OPS; do line="$line\t"; done
echo -e "$line\t\t$(d8f "$SASS/.q.sass")"
rm -f "$SASS/.k.sass" "$SASS/.loop.sass" "$SASS/.q.sass"
