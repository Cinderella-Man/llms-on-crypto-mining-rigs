#!/usr/bin/env bash
# r123a_logs.sh: R123a's (Q1401) per-load facts from rig_taper.sh's load folders, for the README's tables.
#   bash r123a_logs.sh base=r123a_base graphs=r123a_graphs ... OUT=<dir>
# Writes $OUT/loads.tsv (the load's start, server up, replay end and done times from run.log, the server's count of
# switches set, checkpoint and taper lines, the replay's own per-pass line), $OUT/cards.tsv (each card's memory after
# the load and after the replay, from memory_loaded.txt and memory_after.txt) and $OUT/switch_lines.txt (the lines the
# ladder's switches print at load and at exit, colour codes removed).
# Since C804 (Q1408, 2026-11-12) also $OUT/realloc_marks.tsv: the target graph's reallocations per load from each
# server log (scripts/realloc_counts.py --marks ladder: every rung against the rung before, a repeat such as base2 in
# its rung), and a load marked `reallocates` where the rung before has none. Needs python3 and this repository's
# scripts/ (REPO, default the checkout this script is in); without them the file is not written.
set -euo pipefail
: "${OUT:?}"
mkdir -p "$OUT"
printf 'load\tstart\tup_s\treplay_end\tdone\tswitches_set\tcheckpoint_lines\ttaper_lines\treplay\n' > "$OUT/loads.tsv"
printf 'load\twhen\tcard\tmemory_used_mib\n' > "$OUT/cards.tsv"
: > "$OUT/switch_lines.txt"
for spec in "$@"; do
  n=${spec%%=*}; d=${spec#*=}
  r=$d/run.log
  start=$(grep -m1 'load A: --ctx' "$r" | cut -d' ' -f1)
  up=$(grep -m1 'load A: up after' "$r" | sed -E 's/.*up after ([0-9]+) s.*/\1/')
  rend=$(grep -m1 'replay exit' "$r" | cut -d' ' -f1)
  done_t=$(grep -m1 ' done$' "$r" | cut -d' ' -f1)
  rep=$(grep -m1 'replay exit' "$r" | sed -E 's/.*replay exit [0-9]+; //; s/ greedy output.*//')
  ck=$(wc -l < "$d/load-A/checkpoints.txt"); tp=$(wc -l < "$d/load-A/taper_lines.txt")
  log=$(sed 's/\x1b\[[0-9;]*m//g' "$d/load-A/server.log")
  sw=$(grep -m1 -o 'switches: [0-9]* set' <<< "$log" | cut -d' ' -f2)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$start" "$up" "$rend" "$done_t" "$sw" "$ck" "$tp" "$rep" >> "$OUT/loads.tsv"
  for w in loaded after; do
    grep -E '^[0-9]+, [0-9]+ MiB' "$d/load-A/memory_$w.txt" | sed -E "s/^([0-9]+), ([0-9]+) MiB.*/$n\t$w\t\1\t\2/" >> "$OUT/cards.tsv"
  done
  { echo "== $n"; grep -m1 -o 'switches: [0-9]* set: .*' <<< "$log" | sed 's/ (this build.*//';
    grep -E 'GGML_CUDA_GRAPHS_FORCE:|LLAMA_MTP_KV_ONLY:|LLAMA_MTP_CATCHUP_OVERLAP:|LLAMA_NGRAM_PREFETCH:' <<< "$log" \
      | sed -E 's/^[0-9.]+ [A-Z] //' || true; } >> "$OUT/switch_lines.txt"
done
REPO=${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
if [ -f "$REPO/scripts/realloc_counts.py" ]; then
  logs=()
  for spec in "$@"; do logs+=("${spec%%=*}=${spec#*=}/load-A/server.log"); done
  python3 -I "$REPO/scripts/realloc_counts.py" --marks ladder "${logs[@]}" > "$OUT/realloc_marks.tsv"
else
  echo "no $REPO/scripts/realloc_counts.py: realloc_marks.tsv not written" >&2
fi
