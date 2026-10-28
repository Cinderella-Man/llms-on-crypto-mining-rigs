#!/usr/bin/env bash
# Header census of Flash-Next GGUF files: one gguf_census.py report per model folder, the tensor table of each, and one
# summary row per model (sizes on the GPU and in the n-gram table, bytes read per generated token).
#
#   GGUF_PY=<llama.cpp>/gguf-py OUT=<dir> bash census_all.sh MODEL_DIR [MODEL_DIR ...]
#
# Each MODEL_DIR holds the shards of one model; files named mtp-* are reported on their own. In OUT:
#   <dir name>.census.txt  the full report (metadata, tensors by class and format, summary)
#   <dir name>.tensors.tsv the tensor table only: class, format, count, shape, GB
#   summary.tsv            one row per report: file, experts, total, ngram_table, on_gpu, routed_experts, dense,
#                          read_per_token (GB)
set -euo pipefail
: "${GGUF_PY:?}" "${OUT:?}"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export GGUF_PY

report() {  # name, files...
    local name=$1; shift
    nice -n 19 python3 -I "$here/gguf_census.py" "$@" > "$OUT/$name.census.txt"
    awk '/^class\t/{p=1} p&&/^$/{exit} p' "$OUT/$name.census.txt" > "$OUT/$name.tensors.tsv"
    awk -F'\t' -v name="$name" '
        /^summary\t/ {p=1; next}
        p && /^experts\t/ {split($2, e, " "); experts=e[1]}
        p && /^total\t/ {total=$2}
        p && /^ngram_table\t/ {ngram=$2}
        p && /^on_gpu/ {gpu=$2}
        p && /^routed_experts\t/ {exps=$2}
        p && /^dense/ {dense=$2}
        p && /^read_per_token/ {print name, experts, total, ngram, gpu, exps, dense, $2}
    ' OFS='\t' "$OUT/$name.census.txt" >> "$OUT/summary.tsv"
}

printf 'file\texperts\ttotal_GB\tngram_table_GB\ton_gpu_GB\trouted_experts_GB\tdense_GB\tread_per_token_GB\n' \
    > "$OUT/summary.tsv"
for dir in "$@"; do
    name=$(basename "$dir")
    mapfile -t main < <(ls "$dir"/*.gguf | grep -v '/mtp-' || true)
    mapfile -t mtp < <(ls "$dir"/mtp-*.gguf 2>/dev/null || true)
    if [ ${#main[@]} -gt 0 ]; then report "$name" "${main[@]}"; fi
    if [ ${#mtp[@]} -gt 0 ]; then report "${name}_mtp" "${mtp[@]}"; fi
done
