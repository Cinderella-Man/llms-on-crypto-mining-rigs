#!/usr/bin/env bash
# FN63 (Q1245): the sparse Pascal attention kernel (GGML_CUDA_FATTN_SPARSE_PASCAL) on prompt calls of 129, 256 and 512
# query tokens, through the rig build's compute_61 PTX on one newer NVIDIA card. Correctness only. The check set is
# C705's qsa-prompt (Miner Llama pascal-p106-v4-b12246-c705); its run qp-m2 sends every case the kernel can take to it,
# up to 512 query tokens (the kernel's 1,024 columns at 2 KV heads), and qp-stock keeps every case on the dense kernels.
# On a build machine with Docker and one free card:
#   1. cuda_preflight.sh on that card (PREFLIGHT, optional): CUDA works on it, or the run stops here;
#   2. the suite's tar unpacked beside SET, its md5 checked against SET_MD5, unless SET already holds manifest.tsv;
#   3. one container with that card, running this script again with INSIDE=1:
#      a. the GPU code in test-backend-ops listed (cuobjdump): sm_61 machine code and compute_61 PTX only, so a newer
#         card can run nothing but the PTX, JIT-compiled by its driver;
#      b. scripts/replay_checks.sh with RUNS_FILE=SET/runs-ptx.tsv, RUNS (default qp-stock and qp-m2) and the fork's
#         PTX-mode settings in ENV_ALL (the fork's Pascal kernel paths forced on with a P106's 5 SMs).
# Then run/cases.py on OUT/replay: per case, the verdict and the errors in each run.
# Env: SET (the unpacked suite folder, qsa-prompt), SET_TAR, SET_MD5, BIN (bin/ of the branch's rig tarball), OUT, REPO
#   (this repository), GPU (docker's request for one card, e.g. "--device nvidia.com/gpu=0"), MOUNTS (folders mounted
#   in the container at the same path: every one the paths above lie on), IMAGE (default
#   nvidia/cuda:12.9.2-devel-ubuntu24.04), NAME (container, default fn63-ptx), DOCKER_ARGS (labels, --cpu-shares),
#   PREFLIGHT (path of cuda_preflight.sh, optional), RUNS (default "qp-stock qp-m2"), CPU_CHECK (replay_checks.sh's,
#   default 200), NVCACHE (the driver's JIT cache folder, default OUT/nvcache)
# Output: OUT/steps.txt (one line per step), OUT/archs.txt, OUT/replay/ (replay_checks.sh's console.txt, results.tsv,
#   <run>.log, <run>.tsv), OUT/done
set -uo pipefail
OUT=${OUT:?}
mkdir -p "$OUT"
S=$OUT/steps.txt
step() { echo "$(date -Is) $*" | tee -a "$S"; }
RUNS=${RUNS:-qp-stock qp-m2}

if [ "${INSIDE:-0}" = 1 ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH=${NVCACHE:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
    step "inside: $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"
    {
        f=$BIN/test-backend-ops
        echo "== $(basename "$f") md5 $(md5sum < "$f" | cut -d' ' -f1)"
        echo "machine code: $(cuobjdump --list-elf "$f" | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        echo "PTX: $(cuobjdump --list-ptx "$f" | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    } > "$OUT/archs.txt" 2>&1
    step "archs: $(tr '\n' ' ' < "$OUT/archs.txt")"
    PTX="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
         GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
         GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
    step "replay: RUNS_FILE=$SET/runs-ptx.tsv RUNS='$RUNS' ENV_ALL=$(echo $PTX)"
    SET=$SET BIN=$BIN OUT=$OUT/replay DEV=0 CPU_CHECK=${CPU_CHECK:-200} RUNS_FILE=$SET/runs-ptx.tsv RUNS="$RUNS" \
        ENV_ALL="$(echo $PTX)" bash "$REPO/scripts/replay_checks.sh" > "$OUT/replay.out" 2>&1
    step "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt")"
    exit 0
fi

SET=${SET:?}; BIN=${BIN:?}; REPO=${REPO:?}; GPU=${GPU:?}; MOUNTS=${MOUNTS:?}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-fn63-ptx}
rm -f "$OUT/done"
read -r -a GPU_ARGS <<< "$GPU"
read -r -a EXTRA <<< "${DOCKER_ARGS:-}"
step "== ptx_check SET=$SET BIN=$BIN RUNS='$RUNS' image $IMAGE"
if [ -n "${PREFLIGHT:-}" ]; then
    pf=$(IMAGE=$IMAGE PREFLIGHT_DOCKER_ARGS="${DOCKER_ARGS:-}" "$PREFLIGHT" -q -- "${GPU_ARGS[@]}" 2>&1 | tail -n 1)
    step "preflight: $pf"
    [[ $pf == "PREFLIGHT OK"* ]] || { step "stopped: preflight"; date -Is > "$OUT/done"; exit 1; }
fi
if [ ! -f "$SET/manifest.tsv" ]; then
    step "md5 of ${SET_TAR:?}"
    m=$(md5sum < "$SET_TAR" | cut -d' ' -f1)
    [ -z "${SET_MD5:-}" ] || [ "$m" = "$SET_MD5" ] || { step "stopped: md5 $m, expected $SET_MD5"; date -Is > "$OUT/done"; exit 1; }
    mkdir -p "$(dirname "$SET")"
    tar xf "$SET_TAR" -C "$(dirname "$SET")" || { step "stopped: tar"; date -Is > "$OUT/done"; exit 1; }
    step "unpacked: md5 $m, $(du -sb "$SET" | cut -f1) B, $(ls "$SET/cases" | wc -l) entries in cases/"
fi
V=()
for m in $MOUNTS; do V+=(-v "$m:$m"); done
SELF=$(realpath "$0")
step "container $NAME"
docker run --rm --name "$NAME" "${EXTRA[@]}" "${GPU_ARGS[@]}" "${V[@]}" -e INSIDE=1 -e OUT="$OUT" -e SET="$SET" \
    -e BIN="$BIN" -e REPO="$REPO" -e RUNS="$RUNS" -e CPU_CHECK="${CPU_CHECK:-200}" -e NVCACHE="${NVCACHE:-$OUT/nvcache}" \
    "$IMAGE" bash "$SELF" > "$OUT/container.out" 2>&1
step "container exit $?"
date -Is > "$OUT/done"
