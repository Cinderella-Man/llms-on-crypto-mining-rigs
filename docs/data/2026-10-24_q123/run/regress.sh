#!/usr/bin/env bash
# Q123 (2026-10-24): regression check of analyze.py's refactor. TOOLS, DUMPS, MODEL, OUT as in run.sh; TOOLS also holds
# analyze_head.py, the version before the refactor (git show 5143f10^:bench/long_context_kld/analyze.py). It printed
# REGRESS_SAME: both versions give byte-identical tables on the full and the --only-chunks path.
# analyze.py after the refactor (load_dumps, check_same_tokens, piece_decoder) against the committed version: same output
set -euo pipefail
: "${TOOLS:?}" "${DUMPS:?}" "${MODEL:?}" "${OUT:?}"
cd "$TOOLS"
W=$DUMPS; M=$MODEL
for v in analyze_head analyze; do
  python $v.py --ref ud --chunks $W/agent/chunks.jsonl --model $M ud=$W/dump-ud-agent.tsv g3=$W/dump-g3-agent.tsv k2h=$W/dump-k2h-agent.tsv > "$OUT/regress-full-$v.md"
  python $v.py --ref ud --chunks $W/agent12/chunks.jsonl --model $M --only-chunks 0,1,3,6,9,10,12,13,15,18,21,22 ud=$W/dump-ud-agent.tsv g3=$W/dump-g3-agent.tsv > "$OUT/regress-sub-$v.md"
done
cd "$OUT"
cmp regress-full-analyze_head.md regress-full-analyze.md && cmp regress-sub-analyze_head.md regress-sub-analyze.md && echo REGRESS_SAME
