#!/usr/bin/env bash
# The best -ts per configuration for one Flash-Next file on nine P106 cards, from scripts/split_plan.py: the file's own
# block sizes (GGUF headers) plus the per-layer costs that llama.cpp projected for Swift Q2_0 with its default split
# (fit_parts.csv, the q2-def rows): KV cache per attention layer (KV), recurrent state per Gated DeltaNet layer
# (3 MiB, or STATE), and the compute buffer of card 0 (C0) and of every other card (C). The same costs hold for any
# file of this architecture: they depend on context, micro-batch and cache type, not on the weights (the IQ3_XXS
# probes match them to the MiB). With MTP=1 the draft's weights, cache and compute (D) go on the last card, and each
# recurrent layer keeps 12.3 MiB: the main model holds a copy of the state per drafted token to roll back.
#   SHARDS="<first shard> <second shard>" TAG=q2 [MTP=1] [PLAN=<split_plan.py>] [TOP=1] \
#   [PLAN_ARGS="--split-experts --target 300"] bash plan_splits.sh > plans.txt
# Run it where the shards are (header reads only). Each plan's line "#1: -ts ... [-ot ...]" is what fit_grid.sh then
# probes. PLAN_ARGS passes options to split_plan.py, e.g. moving expert tensors across cards with the fewest -ot moves.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PLAN=${PLAN:-$here/../../../../scripts/split_plan.py}
FREE=${FREE:-6001,6003,6003,6003,6003,6003,6001,6007,6007}
: "${SHARDS:?}" "${TAG:?}"
if [ "${MTP:-0}" = 1 ]; then STATE=${STATE:-12.3}; else STATE=${STATE:-3}; fi
# kv ub ctx KV C0 C D (MiB), from the default-split probes of Swift Q2_0 on Miner Llama v2.0. D, the draft on the
# last card, is measured at -ub 128-512 and 64k-262k (the q2-def-*-mtp probes); elsewhere it is 1917 + KV + C - 15
# (C - 30 at -ub 256), the rule those measurements follow within 15 MiB at -ub 128 and 45 MiB at -ub 512.
while read -r kv ub ctx k c0 c d; do
  [ -z "$kv" ] && continue
  extra=0
  [ "${MTP:-0}" = 1 ] && extra=$d
  echo "== $TAG-$kv-ub$ub-$ctx$([ "${MTP:-0}" = 1 ] && echo -mtp) draft_extra=$extra"
  python3 -I "$PLAN" --cards 9 --free "$FREE" --kv-mib "$k" --state-mib "$STATE" --compute "$c0,$c" \
    --last-extra "$extra" --top "${TOP:-1}" ${PLAN_ARGS:-} $SHARDS
done <<'EOF'
f16 64 32768 81 60 70 2053
f16 64 49152 121 81 91 2114
f16 64 65536 161 102 112 2175
f16 64 131072 322 186 195 2419
f16 64 262144 644 353 363 2909
f16 128 32768 81 110 129 2112
f16 128 49152 121 131 150 2173
f16 128 65536 161 161 180 2243
f16 128 131072 322 285 304 2528
f16 128 262144 644 532 551 3097
f16 256 32768 81 220 257 2225
f16 256 49152 121 260 298 2306
f16 256 65536 161 307 344 2392
f16 256 131072 322 527 564 2773
f16 256 262144 644 966 1004 3535
f16 512 65536 161 600 674 2692
f16 512 131072 322 1030 1105 3284
f16 512 262144 644 1886 1961 4462
q8_0 64 32768 43 115 125 2070
q8_0 64 49152 65 160 170 2137
q8_0 64 65536 86 205 215 2203
q8_0 64 131072 172 385 394 2468
q8_0 64 262144 344 744 754 3000
q8_0 128 32768 43 168 187 2132
q8_0 128 49152 65 233 252 2219
q8_0 128 65536 86 294 313 2301
q8_0 128 131072 172 538 557 2631
q8_0 128 262144 344 1025 1044 3290
q8_0 256 32768 43 270 308 2238
q8_0 256 49152 65 355 393 2345
q8_0 256 65536 86 440 477 2450
q8_0 256 131072 172 796 833 2892
q8_0 256 262144 344 1491 1529 3760
q8_0 512 65536 86 748 823 2766
q8_0 512 131072 172 1295 1370 3399
q8_0 512 262144 344 2423 2498 4699
EOF
