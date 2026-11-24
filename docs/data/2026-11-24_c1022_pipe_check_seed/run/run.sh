#!/usr/bin/env bash
# C1022 (Q1755): bench/pipespec/gpu_pipe_check.sh takes its random models' seed from MODEL_SEED, and the older name SEED
# only when it is a whole number. Four rounds of the GPU check as a landing runs it, the same build as both trees:
#   plain     neither SEED nor MODEL_SEED set: seed 1
#   seedpath  SEED=/some/path set by the caller, as the landing's driver did: the path is left alone, seed 1, so the
#             same model files as plain
#   model7    MODEL_SEED=7: other model files, every verdict OK
#   badseed   MODEL_SEED=/some/path: the script stops at once with exit status 2 and its message, before it writes or
#             runs anything
#   seednum   SEED=7 (a driver that still passes the older name with a number): the seed is 7, model7's files
# Then rounds.tsv: per round and tree, the exit status, seconds, the seed line, the md5 of the model files' md5 list and
# the verdict counts; and whether plain's files are REF's (the landing's own run of the same build with seed 1).
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env: WORK, PIPESPEC (a copy of bench/pipespec/ with the changed gpu_pipe_check.sh), BIN (a TESTS=1 rig tarball's bin/),
#      QUANT (a llama-quantize for the Q8_0 copies), GPUS (0,1,2), MOUNTS (folders the containers read), DOCKER_ARGS
#      (labels, --cpu-shares), PREFLIGHT (the CUDA check), REF (optional: an earlier WORK/<tree>/models.md5 of the same
#      build with seed 1), ROUNDS (default "plain seedpath model7 badseed seednum"; a later call
#      with other ROUNDS adds its rows to rounds.tsv)
# Output: WORK/<round>/ (gpu_pipe_check.sh's folders, summary.tsv, check.log), WORK/rounds.tsv, WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${PIPESPEC:?}" "${BIN:?}" "${QUANT:?}"
GPUS=${GPUS:-0,1,2}
ROUNDS=${ROUNDS:-plain seedpath model7 badseed seednum}
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
[ -f "$WORK/rounds.tsv" ] ||
    printf 'round\tcaller env\ttree\trc\tseconds\tseed line\tmodels (md5 of the md5 list)\tfiles\tverdicts\n' > "$WORK/rounds.tsv"
mlist() { grep -a '\.gguf$' "$1" 2> /dev/null | sort -k2; }

for r in $ROUNDS; do
    case $r in
    plain) e=() ;;
    seedpath) e=(SEED=/some/path) ;;
    model7) e=(MODEL_SEED=7) ;;
    badseed) e=(MODEL_SEED=/some/path) ;;
    seednum) e=(SEED=7) ;;
    *) say "unknown round $r"; continue ;;
    esac
    o=$WORK/$r
    rm -rf "$o"
    t0=$(date +%s)
    env -u SEED -u MODEL_SEED "${e[@]}" WORK="$o" BUILDS="base=$BIN head=$BIN" GPUS="$GPUS" MOUNTS="${MOUNTS:-}" \
        DOCKER_ARGS="${DOCKER_ARGS:-}" PREFLIGHT="${PREFLIGHT:-}" QUANT="$QUANT" JITCACHE="$WORK/jitcache" NAME="c1022-$r" \
        bash "$PIPESPEC/gpu_pipe_check.sh" > "$WORK/$r.log" 2>&1
    rc=$? s=$(( $(date +%s) - t0 ))
    seedline=$(grep -ao 'model seed .*\|MODEL_SEED=.*nothing was run' "$WORK/$r.log" | head -n 1 | sed 's/\t/ /g')
    say "$r (${e[*]:-no seed variable}): rc=$rc in $s s; ${seedline:-no seed line}; folder made: $([ -d "$o" ] && echo yes || echo no)"
    if [ ! -d "$o" ]; then
        printf '%s\t%s\t-\t%s\t%s\t%s\t-\t-\t-\n' "$r" "${e[*]:--}" "$rc" "$s" "${seedline:--}" >> "$WORK/rounds.tsv"
        continue
    fi
    for t in base head; do
        mh=$(mlist "$o/$t/models.md5" | md5sum | cut -c1-16)
        nf=$(mlist "$o/$t/models.md5" | wc -l)
        vc=$(awk -F'\t' -v t="$t" '$1 == t {print $6}' "$o/summary.tsv" | sort | uniq -c | awk '{printf "%s %s, ", $1, $2}' | sed 's/, $//')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "${e[*]:--}" "$t" "$rc" "$s" "${seedline:--}" "$mh" "$nf" "${vc:--}" >> "$WORK/rounds.tsv"
    done
done

# the model files: plain and seedpath the same (and REF's), model7 other, seednum model7's
same() { [ -n "$(mlist "$1")" ] && [ "$(mlist "$1")" = "$(mlist "$2")" ] && echo same || echo other; }
{
    echo "plain/head vs seedpath/head: $(same "$WORK/plain/head/models.md5" "$WORK/seedpath/head/models.md5")"
    echo "plain/head vs plain/base: $(same "$WORK/plain/head/models.md5" "$WORK/plain/base/models.md5")"
    echo "plain/head vs model7/head: $(same "$WORK/plain/head/models.md5" "$WORK/model7/head/models.md5")"
    echo "model7/head vs seednum/head: $(same "$WORK/model7/head/models.md5" "$WORK/seednum/head/models.md5")"
    [ -n "${REF:-}" ] && echo "plain/head vs REF: $(same "$WORK/plain/head/models.md5" "$REF")"
    for f in "$WORK"/model7/head/models.md5; do
        [ -f "$f" ] && paste <(mlist "$WORK/plain/head/models.md5") <(mlist "$f") | awk '$1 == $3 {n++} END {print "model7 files equal to plain'"'"'s: " n + 0}'
    done
} > "$WORK/models.txt"
cat "$WORK/models.txt"
column -t -s $'\t' "$WORK/rounds.tsv" | cut -c1-240
say "our containers left: $(docker ps -q --filter 'name=^c1022-' | wc -l)"
date -Is > "$WORK/DONE"
