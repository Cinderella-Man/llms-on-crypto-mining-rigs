#!/usr/bin/env bash
# This folder's builds, one after another on the CPU (build_switches.sh): first C326's spine arms (pmix-q2-fast with
# one dense class from pmix-q2-spine6, or the output layer from the Swift-1.5 Q8_0), then Q219's group switches
# (pmix-q2-fast with one 8-layer group's experts, or its other tensors, from pmix-iq3-fast or Swift IQ2_XS). The switch
# tables are bench/pascal_mix/galloc.py's (`switches`, from the files' header dumps). WORK/done/chain marks the end.
#   REPO=<this repository> DUMPS=<dump.py JSONs: pmix-q2-fast, pmix-q2-spine6, pmix-iq3-fast, swift-iq2_xs,
#   swift15-q8_0> BASE=<first shard of pmix-q2-fast> SPINE6=<first shard of pmix-q2-spine6> IQ3=<first shard of
#   pmix-iq3-fast> IQ2=<first shard of Swift IQ2_XS> Q8=<first shard of the Swift-1.5 Q8_0> MODELS=<output folder>
#   WORK=<scratch> [TABLE=<n-gram table shard>] [CPUS=10,11,22,23] [SPINE_LIST="..."] [GROUP_LIST="..."] bash chain.sh
set -uo pipefail
: "${REPO:?}" "${DUMPS:?}" "${BASE:?}" "${SPINE6:?}" "${IQ3:?}" "${IQ2:?}" "${Q8:?}" "${MODELS:?}" "${WORK:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
P=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast
G=(python3 -I "$REPO/bench/pascal_mix/galloc.py")
mkdir -p "$WORK"
SPINE_LIST=${SPINE_LIST:-"s6gdn_qkv s6gdn_gate s6gdn_out s6fa_qkv s6shexp s6fa_out s6ple q8output"}
# Q219: the switches that change types (galloc.py switches' type_changes > 0); the down projections of groups 1 and 2
# and every IQ2_XS down projection have the start's types (Q2_0) and are left out
GROUP_LIST=${GROUP_LIST:-"guiq35 guiq34 guiq33 guiq32 guiq31 guiq30 dniq35 dniq34 dniq33 dniq30
denseiq35 denseiq34 denseiq33 denseiq32 denseiq31 denseiq30 guiq25 guiq24 guiq23 guiq22 guiq21 guiq20"}
"${G[@]}" switches --dumps "$DUMPS" --spec "$HERE/spine_arms.tsv" --donor s6=pmix-q2-spine6 --donor q8=swift15-q8_0 \
    > "$WORK/spine_switches.tsv" || exit 1
"${G[@]}" switches --dumps "$DUMPS" > "$WORK/switches.tsv" || exit 1
rc=0
SWITCHES=$WORK/spine_switches.tsv DONORS="pmix-q2-spine6=$SPINE6 swift15-q8_0=$Q8" PREFIX=$P LIST=$SPINE_LIST \
    bash "$HERE/build_switches.sh" || rc=1
mv "$WORK/done/all" "$WORK/done/spine" 2> /dev/null
SWITCHES=$WORK/switches.tsv DONORS="pmix-iq3-fast=$IQ3 swift-iq2_xs=$IQ2" PREFIX=$P LIST=$GROUP_LIST \
    bash "$HERE/build_switches.sh" || rc=1
mv "$WORK/done/all" "$WORK/done/groups" 2> /dev/null
date -Is > "$WORK/done/chain"
exit $rc
