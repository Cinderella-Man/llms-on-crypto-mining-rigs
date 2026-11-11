#!/usr/bin/env bash
# sass_check.sh: C728's static check of the sm_61 code (no GPU), C722's carried to the K tiles. From cuobjdump's output
# of the build's objects (ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq-pascal-dmt.cu.o and mmvq-pascal.cu.o:
# -res-usage and -sass), for every mul_mat_vec_q_pascal_dmt<type, NT, R, PF> and mul_mat_vec_q_pascal_dmt_kt<type, NT,
# M, PF>:
#   - registers, stack and local memory (spills);
#   - instruction counts of the kernel: IDP (dp4a), LDS and STS (shared), LDG, FFMA, FMUL, FADD, I2F (and apart those of
#     an integer division, I2F.F32.S32.RP: C722's staging divides by the blocks per token), SHFL, XMAD (integer
#     multiplies on Pascal), PRMT, BAR (barriers);
#   - Q4_K: the super-block's last two FFMA (acc + d * sumd - dmin * summ), with the one-token kernel's
#     (mul_mat_vec_q_pascal<Q4_K, R = 1, C = 1, staged, plain>) beside them: each "t = FFMA(sumd, d, acc); acc =
#     FFMA(summ, -dmin, t)", so that all round the same way. IQ4_XS ends a super-block with an explicit
#     fmaf(d, sumf, acc) in every kernel, so its column stays empty.
# Usage: SASS=<dir with dmt_res.txt, dmt.sass, mmvqp.sass.gz> bash sass_check.sh > sass_counts.tsv
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
tname() { case $1 in 12) echo q4_K ;; 23) echo iq4_xs ;; *) echo "type$1" ;; esac; }
cnt() { grep -cE "^ /\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$SASS/.k.sass" || true; }
echo -e "kernel\ttype\tNT\tR or M\tPF\tREG\tSTACK\tLOCAL\tIDP\tLDS\tSTS\tLDG\tFFMA\tFMUL\tFADD\tI2F\tI2F.RP\tSHFL\tXMAD\tPRMT\tBAR\tlast two FFMA"
awk '/Function _Z.*mul_mat_vec_q_pascal_dmt/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/dmt_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ IL9ggml_type([0-9]+)ELi([0-9])ELi([0-9])ELb([01])E ]] || continue
    t=${BASH_REMATCH[1]} nt=${BASH_REMATCH[2]} r=${BASH_REMATCH[3]} pf=${BASH_REMATCH[4]}
    k=mul_mat_vec_q_pascal_dmt
    [[ $fn == *mul_mat_vec_q_pascal_dmt_kt* ]] && k=mul_mat_vec_q_pascal_dmt_kt
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/dmt.sass" | sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///' \
        > "$SASS/.k.sass"
    last=""
    [ "$t" = 12 ] && last=$(pair "$SASS/.k.sass")
    echo -e "$k\t$(tname "$t")\t$nt\t$r\t$pf\t$reg\t$stack\t$local_\t$(cnt IDP)\t$(cnt LDS)\t$(cnt STS)\t$(cnt LDG)\t$(cnt FFMA)\t$(cnt FMUL)\t$(cnt FADD)\t$(cnt I2F)\t$(cnt 'I2F\.F32\.S32\.RP')\t$(cnt SHFL)\t$(cnt XMAD)\t$(cnt PRMT)\t$(cnt BAR)\t$last"
done | sort -t$'\t' -k1,1 -k2,2 -k3,3n -k4,4n -k5,5n
for t in 12 23; do
    zcat "$SASS/mmvqp.sass.gz" | awk -v re="^_Z.*mul_mat_vec_q_pascalIL9ggml_type${t}ELi1ELi1ELb1ELi0ELi16ELb0EE" \
        '/Function : /{ f = ($3 ~ re) } f' | sed -E 's/ +/ /g;s/\/\* 0x[0-9a-f]+ \*\///' > "$SASS/.one.sass"
    last=""
    [ "$t" = 12 ] && last=$(pair "$SASS/.one.sass")
    echo -e "mul_mat_vec_q_pascal (one token)\t$(tname "$t")\t1\t1\t-\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t$last"
done
rm -f "$SASS/.k.sass" "$SASS/.one.sass"
