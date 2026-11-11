#!/usr/bin/env bash
# sass_f32.sh: C754's static read of the sm_61 code of llama.cpp's mul_mat_vec_f for F32 weights (no GPU), from
# cuobjdump's output of the build's ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvf.cu.o (-res-usage and -sass), for
# every instantiation mul_mat_vec_f<float, float, T, bs, fusion, multi_token_id> (T = 1 to 8 tokens, bs = 32 to 256
# threads):
#   - registers, stack and local memory (spills);
#   - in the whole kernel: FFMA, FMUL, FADD, SHFL, and the stack's stores and loads (STL, LDL);
#   - in every loop (from a backward branch's target to the branch): LDG and FFMA. A pair of weights takes one load and
#     each token's pair of activations one more (1 + T loads), and adds two FMAs per token (2T), so a loop that runs only
#     the kernel's sums has FFMA x (1 + T) = 2T x LDG. The fused kernel (fusion = 1, one token) has a loop of each form:
#     without a gate as above, with one a pair takes 1 + 1 + T loads and 4T FMAs.
# The sums are each token's own chain of FMAs if every product is an FFMA (FMUL 0) and the only FADDs are the
# butterflies' (5 per token and warp reduction: 5T at 32 threads, 10T above). The fused kernel's epilogue (bias, GLU)
# adds FMULs and FADDs of its own, so its expected FADD is given as "-".
# Usage: SASS=<dir with mmvf_res.txt and mmvf.sass> bash sass_f32.sh > sass_f32.tsv
set -euo pipefail
: "${SASS:?SASS=<dir>}"
HEX='function hex(s, i, n) { n = 0; s = tolower(s); sub(/^0x/, "", s)
    for (i = 1; i <= length(s); ++i) n = n * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1; return n }'
echo -e "T\tbs\tfusion\tmulti_token_id\tREG\tSTACK\tLOCAL\tFFMA\tFMUL\tFADD\tFADD_expected\tSHFL\tSTL_LDL\tloops\tloop_LDG\tloop_FFMA\tloops_sums_only"
awk '/Function _Z13mul_mat_vec_fIffLi/ { f = $2 }
     f != "" && /REG:/ { print f "\t" $0; f = "" }' "$SASS/mmvf_res.txt" |
while IFS=$'\t' read -r fn res; do
    fn=${fn%:}
    [[ $fn =~ _Z13mul_mat_vec_fIffLi([0-9]+)ELi([0-9]+)ELb([01])ELb([01])E ]] || continue
    t=${BASH_REMATCH[1]} bs=${BASH_REMATCH[2]} fu=${BASH_REMATCH[3]} mt=${BASH_REMATCH[4]}
    reg=$(sed -E 's/.*REG:([0-9]+).*/\1/' <<< "$res")
    stack=$(sed -E 's/.*STACK:([0-9]+).*/\1/' <<< "$res")
    local_=$(sed -E 's/.*LOCAL:([0-9]+).*/\1/' <<< "$res")
    awk -v fn="$fn" '/Function : /{ f = ($3 == fn) } f' "$SASS/mmvf.sass" |
        sed -E 's/\/\* 0x[0-9a-f]+ \*\///;s/ +/ /g' | grep -E '^ /\*[0-9a-f]+\*/ ' > "$SASS/.k.sass" || true
    cnt() { grep -cE "^ /\*[0-9a-f]+\*/ (@!?P[0-9] )?$1[ .]" "$2" || true; }
    red=$(( bs > 32 ? 10 * t : 5 * t ))
    [ "$fu" = 1 ] && red=-
    # every loop: LDG and FFMA, and whether FFMA x (1 + T) = 2T x LDG (in the fused kernel also FFMA x (2 + T) = 4T x LDG)
    read -r nloop ldgs ffmas ok < <(awk -v T="$t" -v FU="$fu" "$HEX"'{ a = hex(substr($1, 3, length($1) - 4)); line[++n] = $0; addr[n] = a }
        / BRA 0x/ { for (i = 1; i <= NF; ++i) if ($i == "BRA") { b = hex($(i + 1)); if (b < a) { lo[++m] = b; hi[m] = a } } }
        END { ok = 0; sl = ""; sf = ""
              for (j = 1; j <= m; ++j) {
                  l = 0; f = 0
                  for (i = 1; i <= n; ++i) if (addr[i] >= lo[j] && addr[i] <= hi[j]) {
                      if (line[i] ~ /\*\/ (@!?P[0-9] )?LDG[ .]/) ++l
                      if (line[i] ~ /\*\/ (@!?P[0-9] )?FFMA[ .]/) ++f
                  }
                  if (f * (1 + T) == 2 * T * l || (FU == 1 && f * (2 + T) == 4 * T * l)) ++ok
                  sl = sl (j > 1 ? "," : "") l; sf = sf (j > 1 ? "," : "") f
              }
              print m + 0, (m ? sl : "-"), (m ? sf : "-"), ok }' "$SASS/.k.sass")
    spill=$(grep -cE ' (STL|LDL)[ .]' "$SASS/.k.sass" || true)
    echo -e "$t\t$bs\t$fu\t$mt\t$reg\t$stack\t$local_\t$(cnt FFMA "$SASS/.k.sass")\t$(cnt FMUL "$SASS/.k.sass")\t$(cnt FADD "$SASS/.k.sass")\t$red\t$(cnt SHFL "$SASS/.k.sass")\t$spill\t$nloop\t$ldgs\t$ffmas\t$ok"
done | sort -t$'\t' -k3,3n -k4,4n -k2,2n -k1,1n
rm -f "$SASS/.k.sass"
