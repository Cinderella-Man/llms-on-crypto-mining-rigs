#!/usr/bin/env bash
# C1108 (Q1841): test-llama-archs -b CUDA0 -s 1 of one rig build (built for sm_61; on an Ampere card the driver
# compiles its compute 6.1 PTX) under five settings, each run twice. Correctness only: nothing here is timed.
#   stock   nothing set: llama.cpp's own kernels, cuBLAS as the newer card runs it (TF32, F16 sources in F16)
#   notf32  NVIDIA_TF32_OVERRIDE=0 alone: llama.cpp's own kernels, TF32 off
#   before  PASCAL_FORCED as it was before C1108 (without PASCAL_CUBLAS): the fork's P106 paths, cuBLAS as the card runs it
#   after   before and NVIDIA_TF32_OVERRIDE=0: TF32 off alone
#   f32     PASCAL_FORCED as it is now: before and PASCAL_CUBLAS, so cuBLAS also computes F16 and quantized sources in
#           F32, as on a P106 (a newer card has fast F16 and computes them in F16)
# Every run writes or compares the random models' CPU logits (LLAMA_ARCHS_LOGITS): stock and notf32 against the first
# stock run's, before, after and f32 against the first before run's. The two pairs keep their own files because
# PASCAL_FORCED's LLAMA_NO_MOE_DOWN_RESCALE=tg changes the MoE models' graphs, on the CPU too; inside a pair only
# card settings differ, so each pair's CPU side must be the same bits, and the runs say whether it is.
#
#   WORK=<out folder> BIN=<the build's bin folder> GPU=<NVML index of one card> PF=<scripts/pascal_forced.sh>
#   [JITCACHE=<PTX JIT cache, default WORK/jitcache>] [JIT_SEED=<a cache of the same build to copy into an empty one>]
#   [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [PREFLIGHT=<cuda_preflight.sh>] [LABEL=c1108=1, this run's own label]
#   [RUNS='stock-1 notf32-1 before-1 after-1 f32-1 stock-2 notf32-2 before-2 after-2 f32-2'] [MOUNTS='<dir> ...']
#   bash archs.sh
#
# Output in WORK: logs/<run>.log (ANSI colours stripped), rc.txt (one line per run), md5.txt, settings.txt (each
# setting's variables), DONE (the exit status). The tables: python3 -I tables.py (see the README).
set -u
W=${WORK:?WORK=<out folder>}
BIN=${BIN:?BIN=<the build bin folder>}
GPU=${GPU:?GPU=<NVML index>}
PF=${PF:?PF=<scripts/pascal_forced.sh>}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
RUNS=${RUNS:-stock-1 notf32-1 before-1 after-1 f32-1 stock-2 notf32-2 before-2 after-2 f32-2}
LABEL=${LABEL:-c1108=1}
JIT=${JITCACHE:-$W/jitcache}
LBL="--label pascal-quants=1 --label $LABEL"

# shellcheck source=../../../../scripts/pascal_forced.sh
. "$PF" || { echo "no $PF"; exit 1; }
[ -n "${PASCAL_CUBLAS:-}" ] || { echo "$PF has no PASCAL_CUBLAS (C1108)"; exit 1; }
without_cublas() { # <list>: the list without PASCAL_CUBLAS's entries
    local kv out=""
    for kv in $1; do [[ " $PASCAL_CUBLAS " == *" $kv "* ]] || out+="$kv "; done
    echo "${out% }"
}
S_stock=""
S_notf32="NVIDIA_TF32_OVERRIDE=0"
S_before=$(without_cublas "$PASCAL_FORCED")
S_after="NVIDIA_TF32_OVERRIDE=0 $S_before"
S_f32=$PASCAL_FORCED

mkdir -p "$W/logs" "$W/logits" "$JIT"
rm -f "$W/DONE" "$W/rc.txt"
say() { echo "$(date +%H:%M:%S) $*"; }
finish() { docker ps -q --filter "label=$LABEL" | xargs -r docker stop > /dev/null 2>&1  # only this run's containers
           say "end rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
[ -n "${JIT_SEED:-}" ] && [ -d "$JIT_SEED" ] && [ -z "$(ls -A "$JIT")" ] && cp -a "$JIT_SEED/." "$JIT/"
printf 'stock=%s\nnotf32=%s\nbefore=%s\nafter=%s\nf32=%s\n' "$S_stock" "$S_notf32" "$S_before" "$S_after" "$S_f32" \
    > "$W/settings.txt"
(cd "$BIN" && md5sum test-llama-archs) > "$W/md5.txt"
md5sum "$PF" >> "$W/md5.txt"
cat "$W/md5.txt" "$W/settings.txt"

if [ -n "${PREFLIGHT:-}" ]; then
    PREFLIGHT_DOCKER_ARGS="$LBL --cpu-shares=256" IMAGE="$IMAGE" "$PREFLIGHT" -g "$GPU" || finish 3
fi

VOL=(-v "$W:$W" -v "$BIN:$BIN:ro")
[[ "$JIT" == "$W"/* ]] || VOL+=(-v "$JIT:$JIT")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done

one() { # <run name>: <setting>-<repeat>
    local name=$1 set=${1%-*} envs prefix
    case $set in
        stock)  envs=$S_stock;  prefix=plain- ;;
        notf32) envs=$S_notf32; prefix=plain- ;;
        before) envs=$S_before; prefix=forced- ;;
        after)  envs=$S_after;  prefix=forced- ;;
        f32)    envs=$S_f32;    prefix=forced- ;;
        *) say "unknown run $name"; echo "$name rc=2" >> "$W/rc.txt"; return 2 ;;
    esac
    say "$name start (env:${envs:- none})"
    docker run --rm $LBL --cpu-shares=256 --memory=16g --memory-swap=16g --network none --gpus "device=$GPU" \
        "${VOL[@]}" -e "CUDA_CACHE_PATH=$JIT" -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" bash -c "
        env $envs LLAMA_ARCHS_LOGITS=$W/logits/$prefix nice -n 19 timeout 3600 $BIN/test-llama-archs -b CUDA0 -s 1 2>&1 \
            | sed 's/\x1b\[[0-9;]*m//g' > $W/logs/$name.log
        rc=\${PIPESTATUS[0]}
        chown -R $(id -u):$(id -g) $W/logs $W/logits $JIT
        exit \$rc"
    local rc=$?
    echo "$name rc=$rc" >> "$W/rc.txt"
    say "$name rc=$rc: $(grep -a -m 1 '^Summary' "$W/logs/$name.log" 2>/dev/null)"
}

say "start: GPU $GPU, runs: $RUNS"
for r in $RUNS; do one "$r"; done
rm -f "$W/logits/"*.f32
finish 0
