#!/usr/bin/env bash
# The output of run/rig_kernels.sh (stdin) as a table (stdout, TSV): one row per kernel, weight type, product and token
# count, with microseconds per call, the cost relative to the same kernel at one token, and at one token picoseconds per
# weight read.
#   kernel: MT = mul_mat_vec_q_pascal_mt (decodes each weight once for all tokens), MC = the earlier multi-token kernel
#   (GGML_CUDA_MMVQ_PASCAL_MT=0), llama.cpp = llama.cpp's kernel for 2-4 tokens (GGML_CUDA_MMVQ_PASCAL_MC=0); the
#   variant runs add their setting (MT RB=2 ...). All three run the same one-token Pascal kernel at 1 token.
#   Weights read per call at one token: m x k for a dense product, 8 experts x m x k for an expert product,
#   2 x 8 x 3072 x 1024 for the fused gate/up (GGML_CUDA_PROFILE time of its launch group, median over evaluations).
#   bash table.sh < rig_kernels.txt > kernels.tsv
set -euo pipefail
awk -v OFS='\t' '
function val(key,   s) { if (match($0, "[(,]" key "=[0-9]+")) { s = substr($0, RSTART + 1, RLENGTH - 1); sub(/^[a-z_]+=/, "", s); return s } return "" }
function row(t, prod, n, us, w,   key) {
    key = kern SUBSEP t SUBSEP prod
    if (n == 1) { one[key] = us }
    print kern, t, prod, n, us, (key in one ? sprintf("%.2f", us / one[key]) : ""), (n == 1 ? sprintf("%.2f", us * 1e6 / w) : "")
}
BEGIN { print "kernel", "type", "product", "tokens", "us_per_call", "vs_1_token", "ps_per_weight" }
/^-- / {
    fused = ($0 ~ /fused gate\/up/)
    if ($0 ~ /PASCAL_MC=0|^-- llama.cpp/) kern = "llama.cpp"
    else if ($0 ~ /PASCAL_MT=0|^-- MC/) kern = "MC"
    else if (match($0, /PASCAL_MT_[A-Z_]+=[0-9]+/)) { kern = "MT " substr($0, RSTART + 10, RLENGTH - 10) }
    else kern = "MT"
    next
}
fused && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9.]+$/ { row($1, "fused gate/up 3072->1024", $2, $3, 2 * 8 * 3072 * 1024); next }
/us\/run/ {
    match($0, /type_a=[a-zA-Z0-9_]+/); t = substr($0, RSTART + 7, RLENGTH - 7)
    m = val("m"); n = val("n"); k = val("k")
    match($0, /[0-9.]+ us\/run/); us = substr($0, RSTART, RLENGTH - 7) + 0
    if ($0 ~ /^MUL_MAT_ID/) { prod = (m == 1024 ? "expert gate 3072->1024" : "expert down 1024->3072"); w = 8 * m * k }
    else { prod = "dense " k "->" m; w = m * k }
    row(t, prod, n, us, w)
}'
