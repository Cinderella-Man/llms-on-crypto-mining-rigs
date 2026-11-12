#!/usr/bin/env bash
# R115 (Q1382), part 1: llama-server's own fit projection for gmix-q2 and pmix-q2-fast at 65,536 tokens, on header-only
# copies of both files' three shards (../../2026-11-04_fmt_mixes/run/pick_fit.sh made them; unpacked on the rig with
# tar -xSzf), through scripts/fit_probe.sh. Seconds per line, no weights load. Three lines per file:
#   nodraft  the plan's first line (../../2026-11-04_fmt_mixes/pick_fit_64k_cards.txt): no draft, -ts 6,5,5,6,5,5,6,5,6
#   preset   the Flash-Next v3 preset's line as shipped (release/flash-next-v3/settings.env): the MTP draft,
#            -ts 6,6,6,6,5,6,5,6,3, the draft's token embeddings on card 4, the plan's second line
#   v4       the preset with R115's line (line.env): R90's gains line, the taper short=600:64, no top-k rows switch
# Run from the repository root on a machine that only runs ssh, with nothing else on the rig:
#   RIG=<ssh alias> HEADERS=<rig folder holding gmix-q2/ and pmix-q2-fast/> DRAFT=<rig path of the MTP draft> \
#   BUILDS=<rig folder holding the build> BUILD=<build folder name> OUT=fit_probe.md bash r115_fit.sh
# Then the projection per card beside the plan: ../../2026-11-10_deep_lane/run/card_fit.py --probe OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RIG:?}" "${HEADERS:?}" "${DRAFT:?}" "${BUILDS:?}" "${BUILD:?}" "${OUT:?}"
# shellcheck disable=SC1091
. release/flash-next-v3/settings.env  # PRESET_ENV, PRESET_ARGS
# shellcheck disable=SC1091
. "$HERE/line.env"                    # LINE_ENV, LINE_ARGS
# the preset's arguments without those fit_probe.sh sets itself (-ngl, -fa, -c, -np, --ctx-checkpoints, --cache-ram)
# and without -fit off (fit_probe.sh's -fit on is the probe)
read -ra W <<< "$PRESET_ARGS"
ARGS=()
for ((i = 0; i < ${#W[@]}; i++)); do
  case ${W[i]} in
    -ngl | -fa | -c | -np | --ctx-checkpoints | --cache-ram | -fit) i=$((i + 1)) ;;
    *) ARGS+=("${W[i]}") ;;
  esac
done
PRESET_B="${ARGS[*]} -md $DRAFT"
NODRAFT_B=$(echo "${ARGS[*]}" | sed -E 's/ -ts [0-9,]+/ -ts 6,5,5,6,5,5,6,5,6/; s/ --spec-type [^ ]+| --spec-draft-n-max [^ ]+| -otd [^ ]+| --spec-draft-n-ubatch [^ ]+//g')
V4_B="$(echo "$PRESET_B" | sed 's/ --spec-draft-n-ubatch 128//') $LINE_ARGS"
[ "${DRY:-0}" = 1 ] && { printf 'nodraft: %s\npreset: %s\nv4: %s\nv4 env: %s\n' "$NODRAFT_B" "$PRESET_B" "$V4_B" "$PRESET_ENV $LINE_ENV"; exit 0; }
for f in pmix-q2-fast gmix-q2; do
  case $f in
    pmix-q2-fast) shard=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf ;;
    gmix-q2) shard=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf ;;
  esac
  C="nodraft-$f|$BUILD|65536|$NODRAFT_B|$PRESET_ENV"
  C="$C;preset-$f|$BUILD|65536|$PRESET_B|$PRESET_ENV"
  C="$C;v4-$f|$BUILD|65536|$V4_B|$PRESET_ENV $LINE_ENV"
  RIG="$RIG" MODEL="$HEADERS/$f/$shard" BUILDS="$BUILDS" OUT="$OUT" KEEP=300 CONFIGS="$C" \
    bash "$HERE/../../../../scripts/fit_probe.sh" > /dev/null
done
echo R115_FIT_DONE
