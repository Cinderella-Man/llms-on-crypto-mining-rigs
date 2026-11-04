#!/usr/bin/env bash
# The file-choice table end to end: header dumps, the fit grid and the speed and ranking tables, from GGUF headers and
# llama-perplexity KLD logs only (no GPU; under a minute of one CPU core at low priority).
#   FILES=<list> KLD=<folder> OUT=<folder> bash run.sh
# FILES: one line per file, NAME=SHARD[,SHARD...] (every shard of the split file, in order); NAME is also the stem of
# its KLD logs, KLD/NAME.<agent|wiki|code>.log, and the reference's floor logs are KLD/q8_0-ub256.*.log (model.py
# --floor). Extra model.py options go in MODEL_ARGS (e.g. "--base swift-q2_0 --margin 150").
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${FILES:?}" "${KLD:?}" "${OUT:?}"
mkdir -p "$OUT/dumps"
specs=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  specs+=("$line")
  name=${line%%=*}
  IFS=, read -r -a shards <<< "${line#*=}"
  nice -n 19 python3 -I "$here/dump.py" "$name" "${shards[@]}" > "$OUT/dumps/$name.json"
done < "$FILES"
nice -n 19 python3 -I "$here/fit.py" "${specs[@]}" > "$OUT/fit.tsv"
nice -n 19 python3 -I "$here/model.py" --dumps "$OUT/dumps" --fit "$OUT/fit.tsv" --kld "$KLD" --out "$OUT" ${MODEL_ARGS:-}
echo "done: $OUT"
