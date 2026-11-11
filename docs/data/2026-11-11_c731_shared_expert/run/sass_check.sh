#!/usr/bin/env bash
# sass_check.sh: C731's static read of the sm_61 code of the kernels that take the shared expert (dump_sass.sh's output,
# no GPU). For every instantiation of the kernels below: registers, stack and local memory (spills), then its main loop,
# the loop (the code from a backward branch's target to the branch) that holds the most dp4a (IDP), the innermost one
# on a tie, with the instruction counts in it:
#   LDG  global loads: the weights (the activations of the k-column and mtok kernels come from shared memory, staged once
#        per block before the loop, and the one-token kernel's per activation load)
#   LDS, STS  shared loads and stores (the k-column kernel: staged weights and every token's activation records)
#   IDP  dp4a, I2F, FMUL, FFMA, FADD  the integer sums and their FP32 arithmetic
# A loop whose LDG count does not grow with the token count reads each weight once per call, whatever the tokens.
#   mul_mat_kcol<T, NC, R>                         k-column kernel (mmvq-pascal-kcol-<type>.cu.o): T the type, NC the
#                                                  tokens (columns), R the rows per lane group
#   mul_mat_vec_q2_0_pascal_mtok<gate, st, NTM>    the Q2_0 expert kernel of 2-8 tokens (mmvq-pascal-mtok.cu.o) with the
#                                                  routed + shared expert launch; it holds one loop per token count up
#                                                  to NTM, so every loop with dp4a is listed (loop = its rank by dp4a)
#   mul_mat_vec_q_pascal<Q2_0/Q4_0/Q5_0, ...>      the one-token kernel (mmvq-pascal.cu.o), for the arithmetic per block
#   mul_mat_vec_f<float, float, 1-4, 256>          llama.cpp's F32 kernel (mmvf.cu.o), the main loop by FFMA
# Usage: SASS=<dump_sass.sh's OUT> bash sass_check.sh > sass_counts.tsv   (needs c++filt)
set -euo pipefail
: "${SASS:?SASS=<dir>}"
echo -e "object\tkernel\tREG\tSTACK\tLOCAL\tloop\tloop_instr\tLDG\tLDS\tSTS\tIDP\tI2F\tFMUL\tFFMA\tFADD"
scan() { # <object> <mangled-name regex> <all loops with dp4a: 1, the main loop only: 0> [the mnemonic that ranks the loops: IDP]
    local obj=$1 re=$2 all=$3 key=${4:-IDP}
    zcat "$SASS/$obj.sass.gz" | awk -v re="$re" -v all="$all" -v obj="$obj" -v key="$key" '
        function hex(s,   i, n) { n = 0; s = tolower(s); sub(/^0x/, "", s)
            for (i = 1; i <= length(s); ++i) n = n * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1; return n }
        function op(line,   t) { t = line; sub(/^ *\/\*[0-9a-f]+\*\/ */, "", t); sub(/^@!?P[0-9T] +/, "", t)
            sub(/[ .;].*/, "", t); return t }
        function flush(   i, j, k, nl, s, e, c, best, bi, inner) {
            if (fn == "") return
            nl = 0
            for (i = 1; i <= n; ++i) if (tgt[i] >= 0 && tgt[i] < addr[i]) { ++nl; ls[nl] = tgt[i]; le[nl] = addr[i] }
            for (j = 1; j <= nl; ++j) {
                delete c
                for (i = 1; i <= n; ++i) if (addr[i] >= ls[j] && addr[i] <= le[j]) { c[ops[i]]++; c["ALL"]++ }
                for (k in cnt) delete cnt[k]
                L_IDP[j] = c[key] + 0; L_ALL[j] = c["ALL"] + 0
                L_txt[j] = (c["LDG"] + 0) "\t" (c["LDS"] + 0) "\t" (c["STS"] + 0) "\t" (c["IDP"] + 0) "\t" (c["I2F"] + 0) "\t" \
                           (c["FMUL"] + 0) "\t" (c["FFMA"] + 0) "\t" (c["FADD"] + 0)
            }
            if (all) {
                # loops with dp4a that hold no other loop with dp4a, ranked by dp4a
                k = 0
                for (j = 1; j <= nl; ++j) {
                    if (L_IDP[j] == 0) continue
                    inner = 1
                    for (i = 1; i <= nl; ++i) if (i != j && L_IDP[i] > 0 && ls[i] >= ls[j] && le[i] <= le[j] && L_ALL[i] < L_ALL[j]) inner = 0
                    if (inner) { ++k; R_IDP[k] = L_IDP[j]; R_j[k] = j }
                }
                for (i = 1; i <= k; ++i) for (j = i + 1; j <= k; ++j) if (R_IDP[j] < R_IDP[i]) {
                    t = R_IDP[i]; R_IDP[i] = R_IDP[j]; R_IDP[j] = t; t = R_j[i]; R_j[i] = R_j[j]; R_j[j] = t }
                for (i = 1; i <= k; ++i) print obj "\t" fn "\t" i "\t" L_ALL[R_j[i]] "\t" L_txt[R_j[i]]
            } else {
                best = -1; bi = 0
                for (j = 1; j <= nl; ++j)
                    if (L_IDP[j] > best || (L_IDP[j] == best && L_ALL[j] < L_ALL[bi])) { best = L_IDP[j]; bi = j }
                if (bi) print obj "\t" fn "\tmain\t" L_ALL[bi] "\t" L_txt[bi]
            }
            fn = ""
        }
        /Function : / { flush(); n = 0; fn = ($3 ~ re) ? $3 : ""; next }
        fn != "" && /^ *\/\*[0-9a-f]+\*\/ / {
            ++n; a = $1; gsub(/[\/*]/, "", a); addr[n] = hex(a); ops[n] = op($0); tgt[n] = -1
            if (match($0, / BRA [`(]*0x[0-9a-f]+/)) { b = substr($0, RSTART, RLENGTH); sub(/.*0x/, "0x", b); tgt[n] = hex(b) }
        }
        END { flush() }' |
    while IFS=$'\t' read -r o f rest; do
        res=$(awk -v f="$f" '$0 ~ "Function " f ":" { getline; print; exit }' "$SASS/$o.res.txt")
        reg=$(sed -nE 's/.*REG:([0-9]+).*/\1/p' <<< "$res")
        stack=$(sed -nE 's/.*STACK:([0-9]+).*/\1/p' <<< "$res")
        local_=$(sed -nE 's/.*LOCAL:([0-9]+).*/\1/p' <<< "$res")
        name=$(c++filt "$f" | sed -E 's/\(anonymous namespace\):://g; s/\(ggml_type\)//g; s/\(.*//; s/^void //')
        echo -e "$o\t$name\t$reg\t$stack\t$local_\t$rest"
    done
}
for t in q2_0 q3_K q4_K q5_K q6_K iq4_xs q4_0 q5_0 iq4_nl q8_0; do
    scan "mmvq-pascal-kcol-$t" 'mul_mat_kcol' 0
done
scan mmvq-pascal-mtok 'mul_mat_vec_q2_0_pascal_mtok' 1
# the one-token kernel <type, R, C, staged, fk, SU, tile> for Q4_0 (ggml type 2), Q5_0 (6) and Q2_0 (42), staged, with
# 16-byte staging units (in an anonymous namespace, so its mangled name starts with _ZN..._GLOBAL__N_...)
scan mmvq-pascal "${ONE_TOKEN_RE:-20mul_mat_vec_q_pascalIL9ggml_type(2|6|42)ELi[0-9]+ELi[0-9]+ELb1ELi[0-9]ELi16E}" 0
# llama.cpp's F32 kernel of 1-3 tokens, mul_mat_vec_f<float, float, ncols_dst, block_size = 256> (the block size that
# K = 2,560 picks), ranked by FFMA: its global loads are the weights' and each token's activations'
scan mmvf "${MMVF_RE:-^_Z13mul_mat_vec_fIffLi[1-4]ELi256ELb0ELb0EEv}" 0 FFMA
