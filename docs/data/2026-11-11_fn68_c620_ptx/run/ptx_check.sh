#!/usr/bin/env bash
# FN68 (Q1280): C620's KV-only MTP catch-up (LLAMA_MTP_KV_ONLY, Miner Llama pascal-p106-v4-b12246-c620) checked with
# the rig build's own test-llama-archs in a container that holds one newer NVIDIA card. Correctness only. The rig build
# carries sm_61 machine code and compute_61 PTX only, so whatever the test runs on that card is the PTX, JIT-compiled by
# the card's driver.
# On a build machine with Docker and one free card:
#   1. cuda_preflight.sh on that card (PREFLIGHT, optional): CUDA works on it, or the run stops here;
#   2. one container with that card, running this script again with INSIDE=1:
#      a. the card, and the GPU code in test-llama-archs (cuobjdump);
#      b. test-llama-archs --mtp-kv-only -s <seed> for each seed of SEEDS, with the fork's PTX-mode settings (the
#         fork's Pascal kernel paths forced on with a P106's 5 SMs). For each run: its exit code and summary, the
#         card's largest used memory while it ran (nvidia-smi every 0.5 s; a CUDA context takes some hundred MiB), and
#         the files in the driver's JIT cache after it (a kernel of the build that ran on the card leaves its JIT
#         output there);
#      c. the same at seed 1 with -v 4: the log's buffer and compute-buffer lines say on which device the model and
#         the passes ran;
#      d. the control (CONTROL=1, default): test-llama-archs -a qwen4exp -b CUDA0 -s 1, which loads the same
#         architecture's fixture (its main model, without the MTP block) on the card and compares its logits with the
#         CPU's; the same memory and JIT-cache readings. It shows that the container's card runs the build's PTX
#         when a test puts a model on it.
#   Then OUT/kvonly.tsv: one row per seed and configuration of the --mtp-kv-only table.
# Env: BIN (bin/ of the branch's rig tarball, test-llama-archs in it), OUT, GPU (docker's request for one card, e.g.
#   "--device nvidia.com/gpu=0"), MOUNTS (folders mounted in the container at the same path: every one the paths above
#   lie on), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), NAME (container, default fn68-ptx), DOCKER_ARGS
#   (labels, --cpu-shares), PREFLIGHT (path of cuda_preflight.sh, optional), SEEDS (default "1 2 3"), CONTROL
#   (default 1), NVCACHE (the driver's JIT cache folder, default OUT/nvcache; emptied first)
# Output: OUT/steps.txt (one line per step), OUT/archs.txt, OUT/kvonly-s<seed>.log, OUT/kvonly-s1-v4.log,
#   OUT/control.log, OUT/mem-<run>.txt, OUT/kvonly.tsv, OUT/done
set -uo pipefail
OUT=${OUT:?}
mkdir -p "$OUT"
S=$OUT/steps.txt
step() { echo "$(date -Is) $*" | tee -a "$S"; }
SEEDS=${SEEDS:-1 2 3}
CONTROL=${CONTROL:-1}

if [ "${INSIDE:-0}" = 1 ]; then
    NVC=${NVCACHE:-$OUT/nvcache}
    rm -rf "$NVC"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH=$NVC CUDA_CACHE_MAXSIZE=4294967296
    PTX="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
         GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
         GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
    for kv in $PTX; do export "${kv?}"; done
    step "inside: $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used --format=csv,noheader 2>&1 | tr '\n' ';')"
    step "PTX-mode settings: $(echo $PTX)"
    T=$BIN/test-llama-archs
    {
        echo "== test-llama-archs md5 $(md5sum < "$T" | cut -d' ' -f1)"
        echo "machine code: $(cuobjdump --list-elf "$T" | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        echo "PTX: $(cuobjdump --list-ptx "$T" | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    } > "$OUT/archs.txt" 2>&1
    step "archs: $(tr '\n' ' ' < "$OUT/archs.txt")"
    jit() { echo "$(find "$NVC" -type f 2>/dev/null | wc -l) files, $(du -sb "$NVC" 2>/dev/null | cut -f1) B"; }
    # run <name> <args...>: the test, the card's used memory every 0.5 s while it runs, its largest value after
    run() {
        local name=$1; shift
        ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/mem-$name.txt" 2>&1 &
        local mon=$!
        nice -n 19 "$T" "$@" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/$name.log"
        local rc=${PIPESTATUS[0]}
        sleep 1
        kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
        local mx
        mx=$(grep -E '^[0-9]+$' "$OUT/mem-$name.txt" | sort -n | tail -n 1)
        step "$name ($*): rc=$rc; $(grep -m 1 '^Summary' "$OUT/$name.log"); card memory max ${mx:-?} MiB over $(grep -c -E '^[0-9]+$' "$OUT/mem-$name.txt") samples; JIT cache $(jit)"
    }
    for s in $SEEDS; do
        run "kvonly-s$s" --mtp-kv-only -s "$s"
    done
    run kvonly-s1-v4 --mtp-kv-only -s 1 -v 4
    step "kvonly-s1-v4 device lines: $(grep -c -E 'CUDA[0-9]' "$OUT/kvonly-s1-v4.log") naming CUDA; buffers: $(grep -E 'model buffer size|compute buffer size|KV buffer size|buffer size' "$OUT/kvonly-s1-v4.log" | sed -E 's/^[^:]*: *//; s/ += .*//; s/ size.*//' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    if [ "$CONTROL" = 1 ]; then
        run control -a qwen4exp -b CUDA0 -s 1
    fi
    { printf 'seed\tconfig\tkv_prompt\tdrafts\tkv_end\tnodes_off_on\tstatus\n'
      for s in $SEEDS; do
          grep '^| *qwen4exp|' "$OUT/kvonly-s$s.log" | awk -F'|' -v s="$s" 'BEGIN { OFS = "\t" }
              { for (i = 2; i <= 8; i++) gsub(/^ +| +$/, "", $i); print s, $3, $4, $5, $6, $7, $8 }'
      done; } > "$OUT/kvonly.tsv"
    step "table: $(($(wc -l < "$OUT/kvonly.tsv") - 1)) rows, $(tail -n +2 "$OUT/kvonly.tsv" | cut -f7 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    exit 0
fi

BIN=${BIN:?}; GPU=${GPU:?}; MOUNTS=${MOUNTS:?}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-fn68-ptx}
rm -f "$OUT/done"
read -r -a GPU_ARGS <<< "$GPU"
read -r -a EXTRA <<< "${DOCKER_ARGS:-}"
step "== ptx_check BIN=$BIN SEEDS='$SEEDS' CONTROL=$CONTROL image $IMAGE"
if [ -n "${PREFLIGHT:-}" ]; then
    pf=$(IMAGE=$IMAGE PREFLIGHT_DOCKER_ARGS="${DOCKER_ARGS:-}" "$PREFLIGHT" -q -- "${GPU_ARGS[@]}" 2>&1 | tail -n 1)
    step "preflight: $pf"
    [[ $pf == "PREFLIGHT OK"* ]] || { step "stopped: preflight"; date -Is > "$OUT/done"; exit 1; }
fi
V=()
for m in $MOUNTS; do V+=(-v "$m:$m"); done
SELF=$(realpath "$0")
step "container $NAME"
docker run --rm --name "$NAME" "${EXTRA[@]}" "${GPU_ARGS[@]}" "${V[@]}" -e INSIDE=1 -e OUT="$OUT" -e BIN="$BIN" \
    -e SEEDS="$SEEDS" -e CONTROL="$CONTROL" -e NVCACHE="${NVCACHE:-$OUT/nvcache}" \
    "$IMAGE" bash "$SELF" > "$OUT/container.out" 2>&1
step "container exit $?"
date -Is > "$OUT/done"
