#!/usr/bin/env bash
# sass_check.sh: C757's static check of the sm_61 code (no GPU). From cuobjdump's output of the build's objects
# (ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-glu.cu.o: -res-usage and -sass; unary.cu.o: -sass):
#   - for every mul_mat_vec_q_pascal_glu<NT, PF>: registers, stack and local memory (spills), and instruction counts of
#     the whole kernel: IDP (dp4a), I2F, LDS and STS (shared), LDG, FFMA, FMUL, FADD, MUFU, SHFL, PRMT, BAR (the pairs'
#     named barriers and the staging's __syncthreads);
#   - the SWIGLU: the instructions from the FMUL by -log2(e) that starts silu's exp to the store, in the kernel's
#     epilogue and in the separate GLU kernel (unary_gated_op_kernel<op_silu, float>), with the registers masked; the
#     same sequence means the same operations on the same two values.
# Usage: SASS=<dir with glu_res.txt, glu.sass, unary.sass> bash sass_check.sh > sass_counts.tsv (writes SASS/swiglu.txt)
set -euo pipefail
: "${SASS:?SASS=<dir>}"
OPS="IDP I2F LDS STS LDG FFMA FMUL FADD MUFU SHFL PRMT BAR"
# one instruction a line: "/*addr*/ OPCODE operands"
clean() { sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///;s/^ //;s/ *; *$//' | grep -E '^/\*[0-9a-f]+\*/ '; }
cnt() { grep -cE "^/\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .0-9]" "$2" || true; }
# the silu's instructions up to the store: from each FMUL32I by -1.44269... to the next STG or STS, floating-point and
# MUFU/RRO only, registers masked
silu() {
    awk '/FMUL32I\.FTZ R[0-9]+, R[0-9]+, -1\.442695/ { on = 1 }
         on && /(MUFU|RRO|FMUL|FADD|FFMA|FSETP|FSEL|FMNMX)/ { s = $0; sub(/^\/\*[0-9a-f]+\*\/ /, "", s); gsub(/R[0-9]+/, "R", s); print s }
         on && /(STG|STS)/ { print "---"; on = 0 }' "$1"
}
hdr="kernel\tNT\tPF\tREG\tSTACK\tLOCAL"
for op in $OPS; do hdr="$hdr\t$op"; done
echo -e "$hdr\tswiglu"
awk '/Function : unary_gated|Function : _Z21unary_gated_op_kernel/ { f = ($0 ~ /op_silu/ && $0 ~ /EfEEfEvPKT0_S3_PS1_/) ; next }
     /Function :/ { f = 0 } f' "$SASS/unary.sass" | clean > "$SASS/.u.sass"
silu "$SASS/.u.sass" | awk '/^---$/ { exit } { print }' > "$SASS/.u.silu"
{
    echo "separate GLU kernel (unary_gated_op_kernel<op_silu, float>):"
    sed 's/^/  /' "$SASS/.u.silu"
} > "$SASS/swiglu.txt"
awk '/Function _Z.*mul_mat_vec_q_pascal_glu/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/glu_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ gluILi([0-9])ELb([01])E ]] || continue
    nt=${BASH_REMATCH[1]} pf=${BASH_REMATCH[2]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/glu.sass" | clean > "$SASS/.k.sass"
    # every silu of the kernel's epilogue (one per unrolled copy) against the separate kernel's
    silu "$SASS/.k.sass" | awk -v ref="$SASS/.u.silu" '
        BEGIN { while ((getline l < ref) > 0) r[++nr] = l }
        /^---$/ { n++; if (k == nr && ok) same++; k = 0; ok = 1; next }
        { k++; if (k == 1) ok = 1; if ($0 != r[k]) ok = 0 }
        END { printf "%d of %d the same\n", same, n }' > "$SASS/.k.res"
    line="mul_mat_vec_q_pascal_glu\t$nt\t$pf\t$reg\t$stack\t$local_"
    for op in $OPS; do line="$line\t$(cnt "$op" "$SASS/.k.sass")"; done
    echo -e "$line\t$(cat "$SASS/.k.res")"
    {
        echo "mul_mat_vec_q_pascal_glu<$nt, $pf>: $(cat "$SASS/.k.res"); the first:"
        silu "$SASS/.k.sass" | awk '/^---$/ { exit } { print "  " $0 }'
    } >> "$SASS/swiglu.txt"
done
rm -f "$SASS/.k.sass" "$SASS/.u.sass" "$SASS/.u.silu" "$SASS/.k.res"
