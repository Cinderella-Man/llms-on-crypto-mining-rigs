#!/usr/bin/env bash
# sass_check.sh: C690's static check of the sm_61 code (no GPU). From cuobjdump's output of the build's objects
# (ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-q5k.cu.o and mmvq-pascal.cu.o: -res-usage and -sass), for
# every mul_mat_vec_q5_K_pascal_mt<NT, R, PF>:
#   - registers, stack and local memory (spills);
#   - instruction counts of one super-block's code: IDP (dp4a), LDS and STS (shared), LDG, FFMA, FMUL, FADD, I2F, SHFL,
#     and XMAD (integer multiplies on Pascal);
#   - the super-block's last two FFMA (acc + d * sumd - dmin * summ), with the one-token kernel's
#     (mul_mat_vec_q_pascal<Q5_K, R = 1, C = 1, staged, plain>) beside them: each "t = FFMA(sumd, d, acc); acc =
#     FFMA(summ, -dmin, t)", so that both round the same way.
# Usage: SASS=<dir with q5k_res.txt, q5k.sass, mmvqp.sass.gz> bash sass_check.sh > sass_counts.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
strip() { sed -E 's/^ *\/\*[0-9a-f]+\*\/ //;s/ *; *$//'; }
pair() { # <sass file>: the first FFMA with a negated operand, and the FFMA before it that writes its addend
    local f=$1 n neg add prev
    n=$(grep -nE 'FFMA[^;]*-R' "$f" | head -n 1 | cut -d: -f1)
    [ -n "$n" ] || return 0
    neg=$(sed -n "${n}p" "$f" | strip)
    add=$(sed -E 's/.*, (R[0-9]+)(\.reuse)?$/\1/' <<< "$neg")
    prev=$(head -n $((n - 1)) "$f" | grep -E "FFMA[^ ]* $add, " | tail -n 1 | strip)
    echo "$prev; $neg"
}
echo -e "kernel\tNT\tR\tPF\tREG\tSTACK\tLOCAL\tIDP\tLDS\tSTS\tLDG\tFFMA\tFMUL\tFADD\tI2F\tSHFL\tXMAD\tlast two FFMA"
awk '/Function _Z.*mul_mat_vec_q5_K_pascal_mt/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/q5k_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ ILi([0-9])ELi([0-9])ELb([01])E ]] || continue
    nt=${BASH_REMATCH[1]} r=${BASH_REMATCH[2]} pf=${BASH_REMATCH[3]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/q5k.sass" | sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///' \
        > "$SASS/.k.sass"
    cnt() { grep -cE "^ /\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$SASS/.k.sass" || true; }
    echo -e "mul_mat_vec_q5_K_pascal_mt\t$nt\t$r\t$pf\t$reg\t$stack\t$local_\t$(cnt IDP)\t$(cnt LDS)\t$(cnt STS)\t$(cnt LDG)\t$(cnt FFMA)\t$(cnt FMUL)\t$(cnt FADD)\t$(cnt I2F)\t$(cnt SHFL)\t$(cnt XMAD)\t$(pair "$SASS/.k.sass")"
done
zcat "$SASS/mmvqp.sass.gz" | awk '/Function : /{ f = ($3 ~ /^_Z.*mul_mat_vec_q_pascalIL9ggml_type13ELi1ELi1ELb1ELi0ELi16ELb0EE/) } f' |
    sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///' > "$SASS/.one.sass"
echo -e "mul_mat_vec_q_pascal (one token)\t1\t1\t-\t\t\t\t\t\t\t\t\t\t\t\t\t\t$(pair "$SASS/.one.sass")"
rm -f "$SASS/.k.sass" "$SASS/.one.sass"
