#!/usr/bin/env bash
# FN74 (Q1316): the union kernel of the sparse Pascal attention (GGML_CUDA_FATTN_SPARSE_PASCAL_UNION, Miner Llama
# pascal-p106-v4-b12246-c103) through the rig build's compute_61 PTX on one newer NVIDIA card. Correctness only. On a
# build machine with Docker and one free card:
#   1. cuda_preflight.sh on that card (PREFLIGHT, optional): CUDA works on it, or the run stops here;
#   2. the suite's tar unpacked beside SET, its md5 checked against SET_MD5, unless SET already holds manifest.tsv;
#   3. one container with that card, running this script again with INSIDE=1:
#      a. the GPU code in test-backend-ops and fasp_check listed (cuobjdump): sm_61 machine code and compute_61 PTX
#         only, so a newer card can run nothing but the PTX, JIT-compiled by its driver;
#      b. scripts/replay_checks.sh with RUNS_FILE=SET/runs-ptx.tsv (qu-stock and the eight mode-2 runs) and the
#         fork's PTX-mode settings in ENV_ALL (the fork's Pascal kernel paths forced on with a P106's 5 SMs);
#      c. CUDA_FORCE_PTX_JIT=1 fasp_check --gpu --nsm 5: every case of the CPU check, the per-token and the union
#         kernel launched on the card.
# Env: SET (the unpacked suite folder, qsa-union), SET_TAR, SET_MD5, BIN (bin/ of the branch's rig tarball), FASP
#   (bench/fattn_sparse's fasp_check), OUT, REPO (this repository), GPU (docker's request for one card, e.g.
#   "--device nvidia.com/gpu=0"), MOUNTS (folders mounted in the container at the same path: every one the paths above
#   lie on), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), NAME (container, default fn74-ptx), DOCKER_ARGS
#   (labels, --cpu-shares), PREFLIGHT (path of cuda_preflight.sh, optional), CPU_CHECK (replay_checks.sh's, default
#   200), NVCACHE (the driver's JIT cache folder, default OUT/nvcache)
# Output: OUT/steps.txt (one line per step), OUT/archs.txt, OUT/replay/ (replay_checks.sh's console.txt, results.tsv,
#   <run>.log, <run>.tsv), OUT/fasp_gpu.txt, OUT/done
set -uo pipefail
OUT=${OUT:?}
mkdir -p "$OUT"
S=$OUT/steps.txt
step() { echo "$(date -Is) $*" | tee -a "$S"; }

if [ "${INSIDE:-0}" = 1 ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH=${NVCACHE:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
    step "inside: $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"
    {
        for f in "$BIN/test-backend-ops" "$FASP"; do
            echo "== $(basename "$f")"
            echo "machine code: $(cuobjdump --list-elf "$f" | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
            echo "PTX: $(cuobjdump --list-ptx "$f" | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        done
    } > "$OUT/archs.txt" 2>&1
    step "archs: $(tr '\n' ' ' < "$OUT/archs.txt")"
    PTX="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
         GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
         GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
    step "replay: RUNS_FILE=$SET/runs-ptx.tsv ENV_ALL=$(echo $PTX)"
    SET=$SET BIN=$BIN OUT=$OUT/replay DEV=0 CPU_CHECK=${CPU_CHECK:-200} RUNS_FILE=$SET/runs-ptx.tsv ENV_ALL="$(echo $PTX)" \
        bash "$REPO/scripts/replay_checks.sh" > "$OUT/replay.out" 2>&1
    step "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt")"
    step "fasp_check --gpu --nsm 5 (CUDA_FORCE_PTX_JIT=1)"
    CUDA_VISIBLE_DEVICES=0 CUDA_FORCE_PTX_JIT=1 "$FASP" --gpu --nsm 5 > "$OUT/fasp_gpu.txt" 2>&1
    step "fasp_check exit $?: $(grep -a -i -E 'pass|fail' "$OUT/fasp_gpu.txt" | tail -n 2 | tr '\n' ' ')"
    exit 0
fi

SET=${SET:?}; BIN=${BIN:?}; FASP=${FASP:?}; REPO=${REPO:?}; GPU=${GPU:?}; MOUNTS=${MOUNTS:?}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-fn74-ptx}
rm -f "$OUT/done"
read -r -a GPU_ARGS <<< "$GPU"
read -r -a EXTRA <<< "${DOCKER_ARGS:-}"
step "== ptx_check SET=$SET BIN=$BIN FASP=$FASP image $IMAGE"
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
    -e BIN="$BIN" -e FASP="$FASP" -e REPO="$REPO" -e CPU_CHECK="${CPU_CHECK:-200}" -e NVCACHE="${NVCACHE:-$OUT/nvcache}" \
    "$IMAGE" bash "$SELF" > "$OUT/container.out" 2>&1
step "container exit $?"
date -Is > "$OUT/done"
