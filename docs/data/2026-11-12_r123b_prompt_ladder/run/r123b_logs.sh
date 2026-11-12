#!/usr/bin/env bash
# r123b_logs.sh: R123b's (Q1401 (b)) per-load facts from rig_taper.sh's load folders, for the README's tables.
# R123a's r123a_logs.sh (../../2026-11-12_r123a_prompt_ladder/run/) with this ladder's lines, plus the host's memory.
#   OUT=<dir> bash r123b_logs.sh base=data/r123b_base embd=data/r123b_embd ...
# Each folder may have its launcher's file beside it (<folder>.launch: the run script's md5, its line and the
# LLAMA_, GGML_ and CUDA_ variables the load started with).
# Writes $OUT/loads.tsv (the load's start, server up, replay end and done times from run.log, the server's count of
# switches set, checkpoint and taper lines, the replay's own per-pass line), $OUT/cards.tsv (each card's memory after
# the load and after the replay), $OUT/host.tsv (the host's used, buff/cache and available MiB from free -m after the
# load and after the replay) and $OUT/switch_lines.txt (per load: the switch list, the CUDA_ variables, the -otd rule
# in the line, n_batch, where the draft's token table went, and the lines the ladder's switches print; colour codes
# removed).
# Since C804 (Q1408, 2026-11-12) also $OUT/realloc_marks.tsv: the target graph's reallocations per load from each
# server log (scripts/realloc_counts.py --marks ladder: every rung against the rung before, a repeat such as base2 in
# its rung), and a load marked `reallocates` where the rung before has none. Needs python3 and this repository's
# scripts/ (REPO, default the checkout this script is in); without them the file is not written.
set -euo pipefail
: "${OUT:?}"
mkdir -p "$OUT"
printf 'load\tstart\tup_s\treplay_end\tdone\tswitches_set\tcheckpoint_lines\ttaper_lines\treplay\n' > "$OUT/loads.tsv"
printf 'load\twhen\tcard\tmemory_used_mib\n' > "$OUT/cards.tsv"
printf 'load\twhen\ttotal_mib\tused_mib\tfree_mib\tshared_mib\tbuff_cache_mib\tavailable_mib\n' > "$OUT/host.tsv"
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
    awk -v n="$n" -v w="$w" '/^Mem:/ { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", n, w, $2, $3, $4, $5, $6, $7 }' \
      "$d/load-A/memory_$w.txt" >> "$OUT/host.tsv"
  done
  { echo "== $n"; grep -m1 -o 'switches: [0-9]* set: .*' <<< "$log" | sed 's/ (this build.*//';
    [ -f "$d.launch" ] && grep -E '^CUDA_' "$d.launch" | sed 's/^/env: /' || true
    grep -m1 '^ARGS=' "$d/env.txt" | grep -o -- '-otd [^ ]*' | sed 's/^/line: /' || echo "line: no -otd rule"
    grep -o -E 'n_u?batch *= *[0-9]+' <<< "$log" | sed -E 's/ +/ /g' | sort | uniq -c | sed -E 's/^ +//; s/^/contexts: /' || true
    grep -E 'token_embd\.weight.*(overridden to|in CUDA|using CUDA)|LLAMA_DRAFT_EMBD_DEV|FATTN_SPARSE_PASCAL_PROMPT|GGML_CUDA_GRAPHS_FORCE:|LLAMA_MTP_KV_ONLY:|LLAMA_MTP_CATCHUP_OVERLAP:' <<< "$log" \
      | sed -E 's/^[0-9.]+ [A-Z] //' | sort | uniq -c | sed -E 's/^ +//' || true; } >> "$OUT/switch_lines.txt"
done
REPO=${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
if [ -f "$REPO/scripts/realloc_counts.py" ]; then
  logs=()
  for spec in "$@"; do logs+=("${spec%%=*}=${spec#*=}/load-A/server.log"); done
  python3 -I "$REPO/scripts/realloc_counts.py" --marks ladder "${logs[@]}" > "$OUT/realloc_marks.tsv"
else
  echo "no $REPO/scripts/realloc_counts.py: realloc_marks.tsv not written" >&2
fi
