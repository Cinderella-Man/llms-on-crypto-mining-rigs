#!/usr/bin/env bash
# FN93 (Q1487), part 1: which of Ornith's stacked candidates changes the greedy reply at depth. A correctness run, no
# speed: R162's deep lane (docs/data/2026-11-14_r162_ornith_release_deep/) with Ornith-1.5-122B Pascal N2P GGUF on
# v4-rc2's rig build, on newer NVIDIA cards through the build's compute 6.1 PTX, the server in a container and the
# driver (bench/deep_lane/deep_lane.sh) on the host, quiet checks off, every reply kept whole (KEEP_REPLY=1).
# The fork's Pascal kernels run on the newer cards with the P106-090's launch shapes (P106 below: each switch's mode 2
# is "any NVIDIA GPU", and the shapes are chosen for 5 SMs of 96 KB shared memory), so the products take the rig's
# code paths, not llama.cpp's generic ones.
# The arms, each R162's v4 line (release/ornith-v3/settings.env plus the three v4 switches = release/ornith-v4) with:
#   v4    nothing more (the control)
#   v4s   the whole stack, R162's: the staged hop, forced prompt graphs, four checkpoints on the cards and inside a
#         prompt's call, the micro-batch taper and -b 4096 in place of -b 2048
#   hs    GGML_CUDA_HOP_STAGED=1            gf    GGML_CUDA_GRAPHS_FORCE=1
#   ck    --ctx-checkpoints-vram 4          cki   --ctx-checkpoints-vram 4 --ctx-checkpoints-incall (the server turns
#                                                 in-call checkpoints off without checkpoints on the cards)
#   tp    --ubatch-taper short=600:64       b4    -b 4096
# STEP=1 (does this machine write the rig's replies?): v4 and v4s twice each, alternated, at R162's depths. STEP=2: one
# load of v4 and of each candidate arm, at 0, 4,096 and 8,192 tokens (the path R162's cache took to 8,192). STEP=run
# with ARMS and LOADS from the environment (e.g. pairs). STEP=wait returns when the run in OUT has ended. DRY=1 prints
# each arm's server line and starts nothing. The requests are R162's: per depth the prime, two greedy passes and the
# sampler pass, with the same recorded text, tag and tail.
# Run from anywhere on the machine with the cards; env:
#   BIN     v4-rc2's llama-server (the rig tarball's bin/llama-server: sm_61 code and compute 6.1 PTX)
#   MODEL   Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
#   TEXT    the release check's recorded agent text (proxy.jsonl.bodies.gz, proxy.jsonl beside it)
#   OUT     the results folder of this step (one per step)
#   MOUNTS  the folders the container reads or writes, each mounted at its own path: BIN's, MODEL's, OUT's and JIT's
#   GPUS (0,1,2) the cards, as docker's device list; TS (one 1 per card) the split; IMAGE (nvidia/cuda:12.9.2-devel-
#   ubuntu24.04); JIT (OUT/../jit) the compiled-PTX cache, kept between loads and steps; PORT (8197); NAME (fn93-lane)
#   DEPTHS replaces the step's depths
set -uo pipefail
: "${BIN:?BIN=<v4-rc2 llama-server>}" "${MODEL:?MODEL=<the Ornith file>}" "${TEXT:?TEXT=<proxy.jsonl.bodies.gz>}"
: "${OUT:?OUT=<results folder>}"
R=$(cd "$(dirname "$0")/../../../.." && pwd)
STEP=${STEP:-1}
if [ "$STEP" = wait ]; then
  until [ -e "$OUT/done" ] || [ -e "$OUT/failed" ]; do sleep 20; done
  tail -n 3 "$OUT/console.txt"
  [ -e "$OUT/done" ]
  exit
fi
: "${MOUNTS:?MOUNTS=<folders mounted at the same path in the container>}"
GPUS=${GPUS:-0,1,2}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
JIT=${JIT:-$(dirname "$OUT")/jit}
NAME=${NAME:-fn93-lane}
V4_ENV="GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1"
STACK_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1"
STACK_ARGS="--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 -b 4096"
P106="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2
  GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
  GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVF_PASCAL_NSM=5"
export ARM_ENV_v4="$V4_ENV" ARM_ENV_v4s="$V4_ENV $STACK_ENV" ARM_ARGS_v4s="$STACK_ARGS"
export ARM_ENV_hs="$V4_ENV GGML_CUDA_HOP_STAGED=1" ARM_ENV_gf="$V4_ENV GGML_CUDA_GRAPHS_FORCE=1"
export ARM_ENV_ck="$V4_ENV" ARM_ARGS_ck="--ctx-checkpoints-vram 4"
export ARM_ENV_cki="$V4_ENV" ARM_ARGS_cki="--ctx-checkpoints-vram 4 --ctx-checkpoints-incall"
export ARM_ENV_tp="$V4_ENV" ARM_ARGS_tp="--ubatch-taper short=600:64"
export ARM_ENV_b4="$V4_ENV" ARM_ARGS_b4="-b 4096"
case $STEP in
  1) ARMS="v4 v4s" LOADS="v41:v4 s1:v4s v42:v4 s2:v4s" DEPTHS=${DEPTHS:-0 4096 8192 16384 32768 48640} ;;
  2) ARMS="v4 hs gf ck cki tp b4" LOADS="v4:v4 hs:hs gf:gf ck:ck cki:cki tp:tp b4:b4" DEPTHS=${DEPTHS:-0 4096 8192} ;;
  run) : "${ARMS:?}" "${LOADS:?}" "${DEPTHS:?}" ;;
  *) echo "STEP=1|2|run|wait" >&2; exit 2 ;;
esac
n=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
mounts=''
for m in $MOUNTS; do mounts="$mounts -v $m:$m"; done
mkdir -p "$OUT" "$JIT"
# shellcheck disable=SC2086
export ARMS LOADS DEPTHS BIN MODEL TEXT OUT PRESET=$R/release/ornith-v3/settings.env TAG='^t1swift/' SAMPLER_PASS=1 \
  N_PREDICT=256 WARM=0 QUIET=0 KEEP_REPLY=1 LOAD_TIMEOUT_S=3600 PORT=${PORT:-8197} TS=${TS:-$(printf '1%.0s,' $(seq "$n") | sed 's/,$//')} \
  ENV_ADD="CUDA_CACHE_PATH=$JIT CUDA_CACHE_MAXSIZE=4294967296 $(echo $P106)" \
  RUNNER="docker run --rm --init --name $NAME --label pascal-quants=1 --cpu-shares=256 --network host --gpus \"device=$GPUS\"$mounts $IMAGE" \
  STOP_CMD="docker rm -f $NAME"
if [ -n "${DRY:-}" ]; then
  DRY=1 bash "$R/bench/deep_lane/deep_lane.sh"
  exit
fi
setsid -f nohup bash "$R/bench/deep_lane/deep_lane.sh" > "$OUT/console.txt" 2>&1 < /dev/null
echo "started: $OUT/console.txt"
