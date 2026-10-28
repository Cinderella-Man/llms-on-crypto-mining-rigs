#!/usr/bin/env bash
# The output of docs/data/2026-10-27_lowbit/run/rig_kernels.sh (stdin) as a table (stdout, TSV): one row per weight
# type, product, token count and kernel, with microseconds per call and, at one token, picoseconds per weight read.
#   kernel: Pascal = Miner Llama's Pascal matrix-vector kernels (N2P, N3P, Q4_0, Q8_0 in this build); llama.cpp = its
#   own kernel (IQ2_S, IQ3_XXS, Q2_K, Q3_K in this build, and N2P/N3P with GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p).
#   Weights read per call at one token: m x k for a dense product, 8 experts x m x k for an expert product,
#   2 x 8 x 3072 x 1024 for the fused gate/up (GGML_CUDA_PROFILE time of its launch group, median over evaluations).
#   bash table.sh < rig_kernels.txt > kernels.tsv
set -euo pipefail
awk -v OFS='\t' '
function val(key,   s) { if (match($0, "[(,]" key "=[0-9]+")) { s = substr($0, RSTART + 1, RLENGTH - 1); sub(/^[a-z_]+=/, "", s); return s } return "" }
function kernel(t) { return (skip || t !~ /^(n2p|n3p|q4_0|q8_0)$/) ? "llama.cpp" : "Pascal" }
function row(t, prod, n, us, w) { print t, prod, n, kernel(t), us, (n == 1 ? sprintf("%.2f", us * 1e6 / w) : "") }
BEGIN { print "type", "product", "tokens", "kernel", "us_per_call", "ps_per_weight" }
/^-- / { skip = ($0 ~ /GGML_CUDA_MMVQ_PASCAL_SKIP|llama.cpp.s kernel for n2p/); fused = ($0 ~ /fused gate\/up/); next }
fused && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9.]+$/ { row($1, "fused gate/up 3072->1024", $2, $3, 2 * 8 * 3072 * 1024); next }
/us\/run/ {
    match($0, /type_a=[a-zA-Z0-9_]+/); t = substr($0, RSTART + 7, RLENGTH - 7)
    m = val("m"); n = val("n"); k = val("k")
    match($0, /[0-9.]+ us\/run/); us = substr($0, RSTART, RLENGTH - 7) + 0
    if ($0 ~ /^MUL_MAT_ID/) { prod = (m == 1024 ? "expert gate 3072->1024" : "expert down 1024->3072"); w = 8 * m * k }
    else { prod = "dense " k "->" m; w = m * k }
    row(t, prod, n, us, w)
}'
