#!/usr/bin/env bash
# Q232: Flash-Next's expert usage on agent text, from FN4's importance matrix, and what a usage-pruned file would fit
# on nine 6 GB cards. CPU only, standard-library Python at low priority; reads headers, the matrix's per-expert tensors
# and a few MB of weights.
#   IMATRIX=<imatrix.gguf> Q8=<Q8_0 shard list> ORIG=<ISTA IQ3_S shards> PRUNED=<ISTA Coder shards> \
#   FILES=<prune-fit list> OUT=<folder> bash run.sh
# Shard lists are comma-separated; FILES has one NAME=SHARD[,SHARD...] line per file to plan.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${IMATRIX:?}" "${Q8:?}" "${ORIG:?}" "${PRUNED:?}" "${FILES:?}" "${OUT:?}"
KEEP=${KEEP:-512,496,480,464,448,416,384,352,320,288,256}
py() { nice -n 19 python3 -I "$@"; }
mkdir -p "$OUT"
py "$here/expert_usage.py" "$IMATRIX" "$OUT"
IFS=, read -r -a q8 <<< "$Q8"
py "$here/layer_pairs.py" "$OUT/experts.tsv" "$OUT" "${q8[@]}"
IFS=, read -r -a orig <<< "$ORIG"
IFS=, read -r -a pruned <<< "$PRUNED"
py "$here/coder_kept.py" "${orig[@]}" --pruned "${pruned[@]}" --experts "$OUT/experts.tsv" \
  --usage-model "${q8[@]}" --out "$OUT"
mapfile -t specs < <(grep -v '^$' "$FILES")
py "$here/prune_fit.py" --keep "$KEEP" --kv f16 --drafts none,dflash2 --bytes "$OUT/prune_bytes.tsv" \
  --md "$OUT/prune_fit.md" "${specs[@]}" > "$OUT/prune_fit.tsv"
echo "done: $OUT"
