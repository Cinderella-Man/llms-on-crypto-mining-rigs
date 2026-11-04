#!/usr/bin/env bash
# This folder's three mixes, planned (bench/pascal_mix/pmix.py) and built (build.sh) one after another on the CPU:
# the speed file on Swift Q2_0's experts, the quality file on Swift IQ3_XXS's, then the Q6_K spine on Q2_0's. A mix
# that fails is logged in WORK/failed/<mix> and the next one still runs; WORK/done/<mix> marks a built one.
#   REPO=<this repository> DUMPS=<bench/file_choice/dump.py's JSONs of swift-q2_0 and swift-iq3_xxs>
#   BASE_Q2=<first shard of Swift Q2_0> BASE_IQ3=<first shard of Swift IQ3_XXS> MODELS=<where the mixes go>
#   (and build.sh's DONOR IMATRIX QUANTIZE WORK [TABLE CPUS THREADS IMAGE MOUNTS]) bash chain.sh
set -uo pipefail
: "${REPO:?}" "${DUMPS:?}" "${BASE_Q2:?}" "${BASE_IQ3:?}" "${MODELS:?}" "${WORK:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
P=Swift-Qwen3.8-Flash-Next-GSQ-RCO
mkdir -p "$WORK/plan" "$WORK/done" "$WORK/failed"
while read -r mix recipe dump base name folder; do
    [ -f "$WORK/done/$mix" ] && continue
    rm -f "$WORK/failed/$mix"
    echo "$(date '+%F %T') $mix: plan ($recipe on $dump)"
    if python3 -I "$REPO/bench/pascal_mix/pmix.py" --dump "$DUMPS/$dump.json" --recipe "$recipe" --name "$mix" \
            --out "$WORK/plan/$mix" > "$WORK/plan/$mix.txt" &&
        PLAN=$WORK/plan/$mix NAME=$name BASE=$base OUT=$MODELS/$folder bash "$HERE/build.sh"; then
        date -Is > "$WORK/done/$mix"
    else
        echo "exit $? at $(date -Is)" > "$WORK/failed/$mix"
        echo "$(date '+%F %T') $mix: FAILED"
    fi
done <<EOF
pmix-q2-fast fast swift-q2_0 $BASE_Q2 $P-Q2_0-pmix-fast swift_q2_0_pmix_fast
pmix-iq3-fast fast swift-iq3_xxs $BASE_IQ3 $P-IQ3_XXS-pmix-fast swift_iq3_xxs_pmix_fast
pmix-q2-spine6 spine6 swift-q2_0 $BASE_Q2 $P-Q2_0-pmix-spine6 swift_q2_0_pmix_spine6
EOF
date -Is > "$WORK/done/chain"
