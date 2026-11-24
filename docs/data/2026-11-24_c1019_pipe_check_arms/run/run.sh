#!/usr/bin/env bash
# C1019 (Q1752): bench/pipespec/gpu_pipe_check.sh with cpu_tests.sh's pipe step arms on the cards: seeded sampling
# (check sampled), a race of two samples whose winner goes on alone (check race), the floor in every identity check, and
# 256 tokens a run instead of 64. Rounds (ROUNDS, in order; default "old serial jobs3 jobsall defaults fault types"):
#   old       the script before this change (OLDPIPE, C1018's) with its defaults, TIP alone: the cost before
#   serial    the new script's defaults with JOBS=1 (one container at a time), TIP alone: each container's time alone
#   jobs3     the same with JOBS=3
#   jobsall   the same with JOBS=0 (every container of a build at once)
#   defaults  the new script's defaults, TIP as base and head, as a landing runs it
#   fault     the defaults with FAULT as base (C984's tarball, which has C932's Flash-Next pipe fault) and TIP as head:
#             the sampled and race checks must flag qwen4exp on the base, in its tokens, and pass on the head
#   types     each release type's own arm (q2_0-fn-c4 ... iq4_xs-fn-c4) with the new checks, TIP alone: the heavier arms
#             on request
# Run it on the machine with the cards, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env: WORK, PIPESPEC (a copy of bench/pipespec/ with the changed gpu_pipe_check.sh), OLDPIPE (the earlier
#      gpu_pipe_check.sh, beside check_identity.sh and prompts/rewrite.txt; for old), TIP and FAULT (TESTS=1 rig
#      tarballs' bin/), QUANT (a llama-quantize that writes Q2_0), GPUS (0,1,2), MOUNTS (folders the containers read),
#      DOCKER_ARGS (labels, --cpu-shares), PREFLIGHT (the CUDA check), ROUNDS, LABEL (the label our containers carry,
#      to tell other jobs' containers apart in rounds.tsv; default c1019)
# Output: WORK/<round>/ (gpu_pipe_check.sh's folders, summary.tsv, times.tsv, jobs.tsv), WORK/<round>.log,
#         WORK/rounds.tsv (one line a round: its settings, rc, seconds, verdicts, other jobs' containers at its start),
#         WORK/DONE.
set -uo pipefail
: "${WORK:?}" "${PIPESPEC:?}" "${TIP:?}" "${FAULT:?}" "${QUANT:?}"
GPUS=${GPUS:-0,1,2}
ROUNDS=${ROUNDS:-old serial jobs3 jobsall defaults fault types}
LABEL=${LABEL:-c1019}
TYPES="q2_0 q4_k q5_k q6_k iq4_xs"
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
[ -f "$WORK/rounds.tsv" ] ||
    printf 'round\tscript\tbuilds\tsettings\trc\tseconds\tverdicts\tother containers at start\n' > "$WORK/rounds.tsv"
typearms=""
for k in $TYPES; do typearms="$typearms $k-fn-c4"; done

for r in $ROUNDS; do
    s=$PIPESPEC/gpu_pipe_check.sh
    case $r in
    old) s=${OLDPIPE:?OLDPIPE for the old round}; e=(BUILDS="tip=$TIP") ;;
    serial) e=(BUILDS="tip=$TIP" JOBS=1) ;;
    jobs3) e=(BUILDS="tip=$TIP" JOBS=3) ;;
    jobsall) e=(BUILDS="tip=$TIP" JOBS=0) ;;
    defaults) e=(BUILDS="base=$TIP head=$TIP") ;;
    fault) e=(BUILDS="base=$FAULT head=$TIP") ;;
    types) e=(BUILDS="tip=$TIP" ARMS="${typearms# }") ;;
    *) say "unknown round $r"; continue ;;
    esac
    o=$WORK/$r
    rm -rf "$o"
    # other jobs' containers on the machine when the round starts (names only; they share the cards and the processor)
    others=$(docker ps --format '{{.Names}} {{.Labels}}' | grep -v "$LABEL=1" | awk '{print $1}' | paste -sd' ')
    t0=$(date +%s)
    env "${e[@]}" WORK="$o" JITCACHE="$WORK/jitcache" GPUS="$GPUS" QUANT="$QUANT" MOUNTS="${MOUNTS:-}" \
        DOCKER_ARGS="${DOCKER_ARGS:-}" PREFLIGHT="${PREFLIGHT:-}" NAME="gpupipe-$r" bash "$s" > "$WORK/$r.log" 2>&1
    rc=$?
    sec=$(($(date +%s) - t0))
    vd=$(tail -n +2 "$o/summary.tsv" 2> /dev/null | cut -f6 | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')
    # the builds by their tags only (the bin folders are the caller's paths), the other settings as given
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$([ "$r" = old ] && echo old || echo new)" \
        "$(echo "${e[0]#BUILDS=}" | sed -E 's/=[^ ]*//g')" "$(printf '%s ' "${e[@]:1}" ${THREADS:+"THREADS=$THREADS"} | sed 's/ $//')" "$rc" "$sec" "$vd" \
        "${others:-none}" >> "$WORK/rounds.tsv"
    say "$r: rc=$rc in $sec s; $vd"
done
say "our containers left: $(docker ps -q --filter 'name=^gpupipe-' | wc -l)"
date -Is > "$WORK/DONE"
