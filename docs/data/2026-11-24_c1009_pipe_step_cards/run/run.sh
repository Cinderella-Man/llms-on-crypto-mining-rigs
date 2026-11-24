#!/usr/bin/env bash
# C1009 (Q1742): bench/pipespec/gpu_pipe_check.sh, the pipe step on the cards, shown on two trees of pascal-p106-v5 on a
# build machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed
# but the step's own duration.
#   tip  pascal-p106-v5 as it is (C932's fault not fixed there): its TESTS=1 tarball TIP_TGZ
#   fix  the same commit with C932's fix (fork commit 80d83fa1a, branch pascal-p106-v5-pipe-logits) as a patch, built here
# Steps (STEPS, default "build unpack step reach virt"):
#   build   the fix tree: scripts/build_llamacpp_docker.sh TESTS=1 with PATCHES=FIX_PATCH (relative to WORK/build) on TIP,
#           in WORK/build (src/ and ccache/ reused), under LOCK
#   unpack  TIP_TGZ and the fix's tarball into WORK/bin/
#   step    gpu_pipe_check.sh as a GPU check runs it, with its defaults (arms f32-stock q8-p106, layouts one and split
#           over the three cards), tip then fix: WORK/out/step/
#   reach   the other arms on both trees, one card and split: f32-p106 (the forced kernels on F32 weights, which none of
#           them takes), q8-stock (llama.cpp's own kernels on the Q8_0 copies) and q8-p106all (also the fork's other
#           compute-6.1-only paths): WORK/out/reach/; with step's runs, which kernels each setting reached
#   virt    the default arms on three virtual devices on the first card (GGML_CUDA_DEVICES=3): WORK/out/virt/
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env: WORK, REPO (a copy of this repository: scripts/, bench/pipespec/), TIP (the commit), TIP_TGZ, FIX_PATCH, LOCK (a
#      flock file other builds share), QUANT (a llama-quantize for the Q8_0 copies: the rig tarballs hold none), DOCKER_ARGS (labels, --cpu-shares), PREFLIGHT (the CUDA check), GPUS (0,1,2),
#      JOBS (8), UI (the web page release, b12483, what the tip's earlier builds embedded), STEPS
# Output: WORK/out/<round>/ (gpu_pipe_check.sh's folders and summary.tsv, round.log), WORK/logs/, WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${REPO:?}" "${TIP:?}" "${TIP_TGZ:?}" "${FIX_PATCH:?}" "${LOCK:?}" "${QUANT:?}"
STEPS=${STEPS:-build unpack step reach virt}
GPUS=${GPUS:-0,1,2}
JOBS=${JOBS:-8}
UI=${UI:-b12483}
DOCKER_ARGS=${DOCKER_ARGS:-}
PREFLIGHT=${PREFLIGHT:-}
PS=$REPO/bench/pipespec
step() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$WORK/logs" "$WORK/bin" "$WORK/out" "$WORK/jitcache"
rm -f "$WORK/DONE"
say "start: steps '$STEPS', tip $TIP"

if step build; then
    t0=$(date +%s)
    flock "$LOCK" env WORK="$WORK/build" TESTS=1 JOBS="$JOBS" HF_UI_VERSION="$UI" PATCHES="$FIX_PATCH" DOCKER_ARGS="$DOCKER_ARGS" \
        timeout 3600 bash "$REPO/scripts/build_llamacpp_docker.sh" "$TIP" > "$WORK/logs/build-fix.log" 2>&1
    say "build fix: rc=$? seconds=$(( $(date +%s) - t0 )); $(ls "$WORK/build"/llama.cpp-*.tar.gz 2> /dev/null | tr '\n' ' ')"
fi

if step unpack; then
    fix_tgz=$(ls -t "$WORK/build"/llama.cpp-*.tar.gz 2> /dev/null | head -n 1)
    for t in "$TIP_TGZ" "$fix_tgz"; do
        [ -f "$t" ] || { say "unpack: no tarball '$t'"; continue; }
        tar xzf "$t" -C "$WORK/bin"
        say "unpack: $(basename "$t") md5 $(md5sum < "$t" | cut -c1-32), $(stat -c %s "$t") bytes"
    done
    ls "$WORK/bin"
fi

TIPBIN=$(ls -d "$WORK/bin/$(basename "$TIP_TGZ" .tar.gz)/bin" 2> /dev/null)
FIXBIN=$(ls -d "$WORK/bin"/llama.cpp-*-"$(basename "$FIX_PATCH" .diff)"/bin 2> /dev/null | head -n 1)
round() {  # <name> [VAR=value ...]: one gpu_pipe_check.sh run over both trees
    local r=$1 t0 rc
    shift
    t0=$(date +%s)
    env "$@" WORK="$WORK/out/$r" BUILDS="tip=$TIPBIN fix=$FIXBIN" GPUS="$GPUS" MOUNTS="$WORK/bin" JITCACHE="$WORK/jitcache" QUANT="$QUANT" \
        DOCKER_ARGS="$DOCKER_ARGS" PREFLIGHT="$PREFLIGHT" NAME="c1009-$r" bash "$PS/gpu_pipe_check.sh" > "$WORK/out/$r.log" 2>&1
    rc=$?
    say "$r: rc=$rc seconds=$(( $(date +%s) - t0 )); verdicts: $(tail -n +2 "$WORK/out/$r/summary.tsv" | cut -f6 | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
}
if step step || step reach || step virt; then
    [ -n "$TIPBIN" ] && [ -n "$FIXBIN" ] || { say "no tip or fix build in $WORK/bin"; date -Is > "$WORK/DONE"; exit 1; }
    md5sum "$TIPBIN"/llama-pipespec "$FIXBIN"/llama-pipespec "$TIPBIN"/llama-server "$FIXBIN"/llama-server "$PS"/*.sh > "$WORK/out/md5.txt"
fi
step step && round step
step reach && round reach ARMS="f32-p106 q8-stock q8-p106all" LAYOUTS="one split"
step virt && round virt LAYOUTS=virt

say "done; containers left: $(docker ps -q --filter label=c1009=1 | wc -l)"
date -Is > "$WORK/DONE"
