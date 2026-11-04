#!/usr/bin/env bash
# plan.sh: the fewest cards for each Flash-Next file at 64k (f16 cache, -b 2048 -ub 512), from GGUF headers only, with
# scripts/split_plan.py --fewest (Q964). Per file, three lines: no draft; the MTP draft on a card of its own (its whole
# pass, -devd and -otd token_embd onto the free card); the MTP draft on the last card of the split with its token
# embeddings on the card that leaves the most free (S35's placement on fewer cards). First the calibration: S34's and
# S35's nine-card splits of pmix-q2-fast scored by the planner, beside llama.cpp's projections on the rig.
#   FILES="pmix-q2-fast=<shard1>,<shard2>,<shard3> ..." OUT=<folder> bash plan.sh
# Writes OUT/plans-<name><TAG>.txt per file and rebuilds OUT/plans.tsv (every card count of every line of every
# plans-*.txt in OUT). Costs (defaults: llama.cpp's projection on the rig for -kshape, S35's fit probe): FREE (6023 MiB
# per card), KV (161 per attention layer at 64k f16), STATE (3 per Gated DeltaNet layer; STATE_DRAFT 12.3 with a
# draft), COMPUTE (344 on the first card, 419 on the others at -ub 512), DRAFT_BODY (2136: the draft's layer and head,
# cache and compute), DRAFT_EMBD (645: its token embeddings), MARGIN (300), CARDS (9). TAG names a run with other
# costs (e.g. -ub256est with COMPUTE=176,214), LINES picks lines (default "nodraft draft-own draft-last"). PLAN:
# split_plan.py. Runs on any CPU, reads headers only.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PLAN=${PLAN:-$HERE/../../../../scripts/split_plan.py}
: "${FILES:?NAME=shard,shard ...}" "${OUT:?}"
FREE=${FREE:-6023}; KV=${KV:-161}; STATE=${STATE:-3}; STATE_DRAFT=${STATE_DRAFT:-12.3}; COMPUTE=${COMPUTE:-344,419}
DRAFT_BODY=${DRAFT_BODY:-2136}; DRAFT_EMBD=${DRAFT_EMBD:-645}; MARGIN=${MARGIN:-300}; CARDS=${CARDS:-9}
TAG=${TAG:-}; LINES=${LINES:-nodraft draft-own draft-last}
OWN=$(python3 -c "print(round($DRAFT_BODY + $DRAFT_EMBD))")
mkdir -p "$OUT"
base() { python3 -I "$PLAN" --cards "$CARDS" --free "$FREE" --kv-mib "$KV" --compute "$COMPUTE" "$@"; }
for spec in $FILES; do
  name=${spec%%=*}; IFS=, read -ra shards <<< "${spec#*=}"
  {
    echo "### $name$TAG (compute $COMPUTE MiB, KV $KV MiB an attention layer, free $FREE MiB a card, margin $MARGIN)"
    if [ "${CALIBRATE:-1}" = 1 ] && [ "$name" = pmix-q2-fast ]; then
      echo "## calibration: S34's no-draft split (rig 2's projection, -ladder3: 4944 4243 4426 5002 4271 4431 5010 4276 4867 MiB)"
      base --state-mib "$STATE" --ts 6,5,5,6,5,5,6,5,6 "${shards[@]}"
      echo "## calibration: S35's MTP split, the draft's body on card 8 (rig 1's projection, -kshape: main 4992 5178 5047"
      echo "## 5200 3683+210+419 5058 4461 5064 2496, the draft's 2136 on card 8 and its 645 MiB of token embeddings on card 4)"
      base --state-mib "$STATE_DRAFT" --last-extra "$DRAFT_BODY" --ts 6,6,6,6,5,6,5,6,3 "${shards[@]}"
    fi
    for line in $LINES; do
      case $line in
        nodraft) echo "## nodraft"
          base --state-mib "$STATE" --fewest --margin "$MARGIN" --split-experts "${shards[@]}" ;;
        draft-own) echo "## draft-own: the MTP draft's whole pass ($OWN MiB) on the first free card"
          base --state-mib "$STATE_DRAFT" --fewest --margin "$MARGIN" --split-experts --own-card "$OWN" "${shards[@]}" ;;
        draft-last) echo "## draft-last: the draft's body ($DRAFT_BODY MiB) on the last card, its token embeddings ($DRAFT_EMBD MiB) on the best card"
          base --state-mib "$STATE_DRAFT" --fewest --margin "$MARGIN" --split-experts --last-extra "$DRAFT_BODY" \
            --extra-any "$DRAFT_EMBD" "${shards[@]}" ;;
      esac
    done
  } > "$OUT/plans-$name$TAG.txt" 2>&1
  echo "$name$TAG: $(grep -c '^fewest:' "$OUT/plans-$name$TAG.txt") fewest lines, $OUT/plans-$name$TAG.txt"
done
{
  printf 'file\tline\tcards\tfree_cards\town_free_mib\twhole_free_mib\twhole_ts\tot_free_mib\tot_moved\n'
  awk -v OFS='\t' '/^### / { file = $2 } /^## (nodraft|draft-own|draft-last)/ { line = $2; sub(":", "", line) }
    /^[0-9]+\t/ { print file, line, $0 }' "$OUT"/plans-*.txt
} > "$OUT/plans.tsv"
echo "$OUT/plans.tsv: $(($(wc -l < "$OUT/plans.tsv") - 1)) rows"
