#!/usr/bin/env bash
# C1039 (Q1772): why the random cohere2 model's card-against-CPU nmse moved from Miner Llama's pascal-p106-v5 build
# (af00bb13b) to the upstream-b12674 candidate (eee615cdb), when its CPU logits did not.
#
# test-llama-archs -b CUDA0 -s 1 of both rig builds (built for sm_61; on an Ampere card the driver compiles their
# compute-6.1 PTX), each with the CUDA backend's fusions on (the default) and off (GGML_CUDA_DISABLE_FUSION=1). The
# first run writes every random model's CPU logits (LLAMA_ARCHS_LOGITS); every later run compares its own CPU logits
# with them, so the CPU side of each comparison is checked too. The two fusion-on runs are repeated once, to show the
# run-to-run spread of the card's result. Correctness only: nothing here is timed.
#
# EXTRA_ENV is added to every run. On an Ampere card cuBLAS rounds F32 products to TF32 (ggml-cuda sets
# CUBLAS_TF32_TENSOR_OP_MATH on every handle; a P106 has no TF32), and that rounding, about 1e-7 in nmse, hides
# differences a P106 sees at 1e-14. C1039 ran three sets (after `. scripts/pascal_forced.sh`):
#   plain   EXTRA_ENV unset
#   f32     EXTRA_ENV="GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $PASCAL_FORCED"   (a P106's kernels, cuBLAS compute type F32)
#   notf32  EXTRA_ENV="NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $PASCAL_FORCED"   (and TF32 off: NVIDIA's
#           switch that overrides a library's own math-mode setting)
#
#   WORK=<out folder> CAND=<candidate build folder> REF=<reference build folder> GPU=<NVML index of one card>
#   [JIT_CAND=<a PTX JIT cache of CAND to copy>] [JIT_REF=<the same for REF>] [MOUNTS='<dir> ...' read-only extras]
#   [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [PREFLIGHT=<cuda_preflight.sh>] [LABEL=c1039=1, this run's own label]
#   [RUNS='ref-on cand-on ref-off cand-off ref-on2 cand-on2'] [EXTRA_ENV='NAME=value ...'] bash run.sh
#
# Output in WORK: logs/<run>.log (ANSI colours stripped), rc.txt (one line per run), md5.txt, DONE (the exit status).
# The tables: python3 -I tables.py <WORK> <data folder>.
set -u
W=${WORK:?WORK=<out folder>}
CAND=${CAND:?CAND=<candidate build folder>}
REF=${REF:?REF=<reference build folder>}
GPU=${GPU:?GPU=<NVML index>}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
RUNS=${RUNS:-ref-on cand-on ref-off cand-off ref-on2 cand-on2}
LABEL=${LABEL:-c1039=1}
EXTRA_ENV=${EXTRA_ENV:-}
LBL="--label pascal-quants=1 --label $LABEL"

mkdir -p "$W/logs" "$W/logits" "$W/jit-cand" "$W/jit-ref"
rm -f "$W/DONE" "$W/rc.txt"
say() { echo "$(date +%H:%M:%S) $*"; }
finish() { docker ps -q --filter "label=$LABEL" | xargs -r docker stop > /dev/null 2>&1  # only this run's containers
           say "end rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }

# a copy of each build's PTX JIT cache, so the first run does not compile every kernel again (the copies stay ours)
seed() { [ -n "$2" ] && [ -d "$2" ] && [ -z "$(ls -A "$1")" ] && cp -a "$2/." "$1/"; return 0; }
seed "$W/jit-cand" "${JIT_CAND:-}"
seed "$W/jit-ref" "${JIT_REF:-}"

(cd "$CAND/bin" && md5sum test-llama-archs | sed "s|\$| (CAND $(basename "$CAND"))|") >  "$W/md5.txt"
(cd "$REF/bin"  && md5sum test-llama-archs | sed "s|\$| (REF $(basename "$REF"))|")   >> "$W/md5.txt"
cat "$W/md5.txt"

if [ -n "${PREFLIGHT:-}" ]; then
    PREFLIGHT_DOCKER_ARGS="$LBL --cpu-shares=256" IMAGE="$IMAGE" "$PREFLIGHT" -g "$GPU" || finish 3
fi

VOL=(-v "$W:$W" -v "$CAND:$CAND:ro" -v "$REF:$REF:ro")
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done

one() { # <run name>: ref-* or cand-*, *-on or *-off (a trailing digit marks a repeat)
    local name=$1 bin jit envs
    case $name in
        ref-*)  bin=$REF/bin;  jit=$W/jit-ref ;;
        cand-*) bin=$CAND/bin; jit=$W/jit-cand ;;
        *) say "unknown run $name"; return 2 ;;
    esac
    case $name in
        *-on*)  envs="" ;;
        *-off*) envs="GGML_CUDA_DISABLE_FUSION=1" ;;
        *) say "unknown run $name"; return 2 ;;
    esac
    envs="$envs $EXTRA_ENV"
    say "$name start ($bin, env:${envs:- none})"
    docker run --rm $LBL --cpu-shares=256 --memory=16g --memory-swap=16g --network none --gpus "device=$GPU" \
        "${VOL[@]}" -e "CUDA_CACHE_PATH=$jit" -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" bash -c "
        env $envs LLAMA_ARCHS_LOGITS=$W/logits/ref- nice -n 19 timeout 3600 $bin/test-llama-archs -b CUDA0 -s 1 2>&1 \
            | sed 's/\x1b\[[0-9;]*m//g' > $W/logs/$name.log
        rc=\${PIPESTATUS[0]}
        chown -R $(id -u):$(id -g) $W/logs $W/logits $jit
        exit \$rc"
    local rc=$?
    echo "$name rc=$rc" >> "$W/rc.txt"
    say "$name rc=$rc: $(grep -a -m 1 '^Summary' "$W/logs/$name.log" 2>/dev/null)"
}

say "start: GPU $GPU, runs: $RUNS"
for r in $RUNS; do one "$r"; done
rm -f "$W/logits/"*.f32
finish 0
