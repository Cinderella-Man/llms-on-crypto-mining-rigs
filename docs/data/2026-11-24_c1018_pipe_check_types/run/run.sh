#!/usr/bin/env bash
# C1018 (Q1751): bench/pipespec/gpu_pipe_check.sh on the release files' quant types and the Flash-Next preset's kernel
# switches. Six rounds of the GPU pipe check (all but old by default):
#   defaults  the script's default arms, TIP as base and head, as a landing runs it: the time a build takes (the
#             first two calls ran f32-stock q8-p106 mix-fn-c4, the third the final f32-stock q8-p106 mix-fn-c4@one)
#   old       the earlier default arms (f32-stock q8-p106), the same way: what the new arm adds to that time (run it
#             with defaults in a WORK of its own when no other job shares the machine: ROUNDS="old defaults")
#   reach     every kind of model (q8, mix, q2_0, q4_k, q5_k, q6_k, iq4_xs) under stock, p106, fn, p106-c4 and fn-c4,
#             and f32-stock, on one card, TIP alone: the plain hashes per arm show which settings computed other bits
#   split     the default arms and every kind's fn-c4 arm split over the cards, TIP alone
#   prof      reach's arms on one card with CHECKS=profile alone: every kernel the host check's runs launch, by name
#   fault     mix-fn-c4 and each type's fn-c4 arm with FAULT as base (C984's tarball, which has C932's Flash-Next pipe
#             fault) and TIP as head: the new arms must flag qwen4exp on the base and pass on the head
# Run it on the machine with the cards, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env: WORK, PIPESPEC (a copy of bench/pipespec/ with the changed gpu_pipe_check.sh), TIP and FAULT (TESTS=1 rig
#      tarballs' bin/), QUANT (a llama-quantize for the copies), NSYS (an nsys binary, for prof), GPUS (0,1,2), MOUNTS
#      (folders the containers read), DOCKER_ARGS (labels, --cpu-shares), PREFLIGHT (the CUDA check), ROUNDS (default
#      "defaults reach split prof fault"; a later call with other ROUNDS adds its rows to rounds.tsv)
# Output: WORK/<round>/ (gpu_pipe_check.sh's folders and summary.tsv), WORK/<round>.log, WORK/rounds.tsv, WORK/DONE.
set -uo pipefail
: "${WORK:?}" "${PIPESPEC:?}" "${TIP:?}" "${FAULT:?}" "${QUANT:?}"
GPUS=${GPUS:-0,1,2}
ROUNDS=${ROUNDS:-defaults reach split prof fault}
TYPES="q2_0 q4_k q5_k q6_k iq4_xs"
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
[ -f "$WORK/rounds.tsv" ] || printf 'round\tbuilds\tarms\tlayouts\tchecks\trc\tseconds\tverdicts\n' > "$WORK/rounds.tsv"

all=f32-stock
for k in q8 mix $TYPES; do all="$all $k-stock $k-p106 $k-fn $k-p106-c4 $k-fn-c4"; done
fnarms=mix-fn-c4
for k in $TYPES; do fnarms="$fnarms $k-fn-c4"; done

for r in $ROUNDS; do
    case $r in
    defaults) e=(BUILDS="base=$TIP head=$TIP") ;;
    old) e=(BUILDS="base=$TIP head=$TIP" ARMS="f32-stock q8-p106") ;;
    reach) e=(BUILDS="tip=$TIP" ARMS="$all" LAYOUTS=one) ;;
    split) e=(BUILDS="tip=$TIP" ARMS="f32-stock q8-p106 q8-fn-c4 $fnarms" LAYOUTS=split) ;;
    prof) e=(BUILDS="tip=$TIP" ARMS="$all" LAYOUTS=one CHECKS=profile NSYS="${NSYS:?NSYS for the prof round}") ;;
    fault) e=(BUILDS="base=$FAULT head=$TIP" ARMS="$fnarms") ;;
    *) say "unknown round $r"; continue ;;
    esac
    o=$WORK/$r
    rm -rf "$o"
    t0=$(date +%s)
    env "${e[@]}" WORK="$o" JITCACHE="$WORK/jitcache" GPUS="$GPUS" QUANT="$QUANT" MOUNTS="${MOUNTS:-}" \
        DOCKER_ARGS="${DOCKER_ARGS:-}" PREFLIGHT="${PREFLIGHT:-}" NAME="gpupipe-$r" \
        bash "$PIPESPEC/gpu_pipe_check.sh" > "$WORK/$r.log" 2>&1
    rc=$?
    s=$(($(date +%s) - t0))
    first=$(head -n 1 "$WORK/$r.log" | sed -E 's/^[0-9:]+ gpu pipe check: //')
    vd=$(tail -n +2 "$o/summary.tsv" 2> /dev/null | cut -f6 | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')
    # the builds by their tags only (the bin folders are the caller's paths)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$(echo "${e[0]#BUILDS=}" | sed -E 's/=[^ ]*//g')" \
        "$(echo "$first" | grep -o "arms '[^']*'" | cut -d"'" -f2)" "$(echo "$first" | grep -o "layouts '[^']*'" | cut -d"'" -f2)" \
        "$(echo "$first" | grep -o "checks '[^']*'" | cut -d"'" -f2)" "$rc" "$s" "$vd" >> "$WORK/rounds.tsv"
    say "$r: rc=$rc in $s s; $vd"
done
say "our containers left: $(docker ps -q --filter 'name=^gpupipe-' | wc -l)"
date -Is > "$WORK/DONE"
