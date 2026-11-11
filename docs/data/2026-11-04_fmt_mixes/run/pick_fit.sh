#!/usr/bin/env bash
# A pick's fit on nine 6 GB cards at 64k from GGUF headers alone, beside the start's: scripts/split_plan.py scores both
# files on the two nine-card lines that fewer_hops' plan.sh calibrated against llama.cpp's projections on the rig
# (f16 cache, -b 2048 -ub 512: no draft at -ts 6,5,5,6,5,5,6,5,6; the MTP draft's body on card 8 at
# -ts 6,6,6,6,5,6,5,6,3), MiB per card; then bench/file_choice/fit.py's grid (every context, cache, micro-batch and
# draft) for the pick, and header copies of both for scripts/fit_probe.sh on a rig (fewer_hops' header_copies.sh).
#   REPO=<this repository> OUT=<folder> FILES="pmix-q2-fast=<shard1>,<shard2>,<shard3> <pick>=<shard1>,<shard2>,<shard3>" \
#   bash pick_fit.sh
# Costs as plan.sh's defaults: FREE 6023 MiB a card, KV 161 MiB an attention layer, COMPUTE 344 on the first card and
# 419 on the others, STATE 3 (12.3 with the draft), DRAFT_BODY 2136. Writes OUT/fit_64k_cards.txt, OUT/fit-<name>.tsv,
# OUT/headers/<name>.tar.gz. Runs on any CPU (reads headers only).
set -uo pipefail
: "${REPO:?}" "${OUT:?}" "${FILES:?}"
FREE=${FREE:-6023}; KV=${KV:-161}; COMPUTE=${COMPUTE:-344,419}; STATE=${STATE:-3}; STATE_DRAFT=${STATE_DRAFT:-12.3}
DRAFT_BODY=${DRAFT_BODY:-2136}
PLAN=$REPO/scripts/split_plan.py
mkdir -p "$OUT"
base() { python3 -I "$PLAN" --cards 9 --free "$FREE" --kv-mib "$KV" --compute "$COMPUTE" "$@"; }
{
  for spec in $FILES; do
    name=${spec%%=*}; IFS=, read -ra shards <<< "${spec#*=}"
    echo "### $name: no draft, -ts 6,5,5,6,5,5,6,5,6"
    base --state-mib "$STATE" --ts 6,5,5,6,5,5,6,5,6 "${shards[@]}"
    echo "### $name: the MTP draft's body ($DRAFT_BODY MiB) on card 8, -ts 6,6,6,6,5,6,5,6,3"
    base --state-mib "$STATE_DRAFT" --last-extra "$DRAFT_BODY" --ts 6,6,6,6,5,6,5,6,3 "${shards[@]}"
  done
} | tee "$OUT/fit_64k_cards.txt"
for spec in $FILES; do
  name=${spec%%=*}
  python3 -I "$REPO/bench/file_choice/fit.py" "$spec" > "$OUT/fit-$name.tsv"
done
FILES="$FILES" OUT="$OUT/headers" SCRIPTS="$REPO/scripts" bash "$REPO/docs/data/2026-11-04_fewer_hops/run/header_copies.sh"
