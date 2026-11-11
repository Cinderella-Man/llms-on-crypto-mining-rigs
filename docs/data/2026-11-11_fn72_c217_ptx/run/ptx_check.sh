#!/usr/bin/env bash
# FN72 (Q1301): C217's store_at cases (llama_state_seq_store_at: a state stored into a device store inside a decode
# call, Miner Llama pascal-p106-v4-b12246-c217) checked with the rig build's own test-llama-context in a container that
# holds one newer NVIDIA card. Correctness only. The rig build carries sm_61 machine code and compute_61 PTX only, so
# whatever the test runs on that card is the PTX, JIT-compiled by the card's driver. FN68's script shape.
# On a build machine with Docker and one free card:
#   1. cuda_preflight.sh on that card (PREFLIGHT, optional): CUDA works on it, or the run stops here;
#   2. one container with that card, running this script again with INSIDE=1:
#      a. the card, and the GPU code in test-llama-context (cuobjdump);
#      b. with the fork's PTX-mode settings (the fork's Pascal kernel paths forced on with a P106's 5 SMs), three runs
#         of test-llama-context over the random models in MODELS. For each: its exit code and summary, the card's
#         largest used memory while it ran (nvidia-smi every 0.5 s; a CUDA context takes some hundred MiB), and the
#         files in the driver's JIT cache after it (a kernel of the build that ran on the card leaves its JIT output
#         there):
#         - row:  -ngl 99 --models MODELS --test '^store_at'  (FN72's command; the filter is a whole-name match, so
#                 only store_at);
#         - both: -ngl 99 -lv 4 --test 'store_at(_p)?', one model at a time (-m): store_at and store_at_p. The models
#                 table mode (--models) sets the log threshold to 0, so only one model at a time with -lv 4 keeps the
#                 load lines in its log, which name the device of every buffer and of the state stores;
#         - cpu:  -dev none -lv 4 --test 'store_at(_p)?', one model at a time: the same binary with no device, the
#                 controls' CPU report for both cases, and the card's readings without a model on it;
#      c. each run's per-model logs: the cases' control reports (the "n_rs_seq" lines) and the device lines.
# Env: BIN (bin/ of the branch's rig tarball, test-llama-context in it), MODELS (the random .gguf models, from
#   test-llama-archs -o), OUT, GPU (docker's request for one card, e.g. "--device nvidia.com/gpu=0"), MOUNTS (folders
#   mounted in the container at the same path: every one the paths above lie on), IMAGE (default
#   nvidia/cuda:12.9.2-devel-ubuntu24.04), NAME (container, default fn72-ptx), DOCKER_ARGS (labels, --cpu-shares),
#   PREFLIGHT (path of cuda_preflight.sh, optional), NVCACHE (the driver's JIT cache folder, default OUT/nvcache;
#   emptied first)
# Output: OUT/steps.txt (one line per step), OUT/archs.txt, OUT/<run>.log (the verdicts), OUT/logs-<run>/ (per-model
#   logs), OUT/mem-<run>.txt, OUT/reports.tsv (run, model, case, fill, report), OUT/devices.tsv (run, model, what,
#   lines), OUT/done
set -uo pipefail
OUT=${OUT:?}
mkdir -p "$OUT"
S=$OUT/steps.txt
step() { echo "$(date -Is) $*" | tee -a "$S"; }

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
    T=$BIN/test-llama-context
    {
        echo "== test-llama-context md5 $(md5sum < "$T" | cut -d' ' -f1)"
        echo "machine code: $(cuobjdump --list-elf "$T" | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        echo "PTX: $(cuobjdump --list-ptx "$T" | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    } > "$OUT/archs.txt" 2>&1
    step "archs: $(tr '\n' ' ' < "$OUT/archs.txt")"
    jit() { echo "$(find "$NVC" -type f 2>/dev/null | wc -l) files, $(du -sb "$NVC" 2>/dev/null | cut -f1) B"; }
    nocol() { sed 's/\x1b\[[0-9;]*m//g'; }
    # run <name> <table|each> <args...>: the test over MODELS, as one table (--models) or one model at a time (-m), its
    # per-model logs in OUT/logs-<name> (one model at a time: in OUT/logs-<name>/<model>, since --logs must make a new
    # folder); the card's used memory every 0.5 s while it runs, its largest value after
    run() {
        local name=$1 mode=$2; shift 2
        local rc=0 r m mx
        rm -rf "$OUT/logs-$name"
        ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/mem-$name.txt" 2>&1 &
        local mon=$!
        if [ "$mode" = table ]; then
            ( cd "$OUT" && nice -n 19 "$T" "$@" --models "$MODELS" --logs "$OUT/logs-$name" ) 2>&1 | nocol > "$OUT/$name.log"
            rc=${PIPESTATUS[0]}
        else
            : > "$OUT/$name.log"
            for m in "$MODELS"/*.gguf; do
                echo "== $(basename "$m")" >> "$OUT/$name.log"
                ( cd "$OUT" && nice -n 19 "$T" "$@" -m "$m" --logs "$OUT/logs-$name/$(basename "$m" .gguf)" ) 2>&1 | nocol >> "$OUT/$name.log"
                r=${PIPESTATUS[0]}
                [ "$r" = 0 ] || rc=$r
            done
        fi
        sleep 1
        kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
        mx=$(grep -E '^[0-9]+$' "$OUT/mem-$name.txt" | sort -n | tail -n 1)
        step "$name ($mode: $*): rc=$rc; card memory max ${mx:-?} MiB over $(grep -c -E '^[0-9]+$' "$OUT/mem-$name.txt") samples; JIT cache $(jit)"
        if [ "$mode" = table ]; then
            step "$name verdicts: $(grep -m 1 -o 'models: .*' "$OUT/$name.log"); $(grep -E '^(Model|[a-z0-9-]+\.gguf) ' "$OUT/$name.log" | tr -s ' ' | tr '\n' ';')"
        else
            step "$name verdicts: $(grep -E '^== |^store_at(_p)? +[A-Z]+|^(passed|failed|skipped) +:' "$OUT/$name.log" | tr -s ' ' | tr '\n' ';')"
        fi
    }
    run row  table -ngl 99 --test '^store_at'
    run both each  -ngl 99 -lv 4 --test 'store_at(_p)?'
    run cpu  each  -dev none -lv 4 --test 'store_at(_p)?'
    # the control reports: one line per run, model, case and fill
    { printf 'run\tmodel\tcase\tfill\treport\n'
      for name in row both cpu; do
          for f in $(find "$OUT/logs-$name" -name "*.log" | sort); do
              m=$(basename "$f" .log); m=${m#test-llama-context.}; m=${m%.gguf}
              nocol < "$f" | awk -v r="$name" -v m="$m" 'BEGIN { OFS = "\t" }
                  /^I \(fill 0x[0-9a-f]+\)/ { fill = $3; sub(/\)$/, "", fill) }
                  /^I store_at(_p)?: n_rs_seq/ { c = $2; sub(/:$/, "", c); rep = $0; sub(/^I store_at(_p)?: /, "", rep); print r, m, c, fill, rep }'
          done
      done; } > "$OUT/reports.tsv"
    step "reports: $(($(wc -l < "$OUT/reports.tsv") - 1)) lines; $(tail -n +2 "$OUT/reports.tsv" | cut -f1,3 | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    # where the models and the stores went: the device lines of each per-model log, counted
    { printf 'run\tmodel\twhat\tlines\n'
      for name in both cpu; do
          for f in $(find "$OUT/logs-$name" -name "*.log" | sort); do
              m=$(basename "$f" .log); m=${m#test-llama-context.}; m=${m%.gguf}
              nocol < "$f" | grep -o -E "offloaded [0-9]+/[0-9]+ layers to GPU|[A-Za-z0-9_]+ +(model|KV|RS|compute|output|rotation) buffer size|a state store in '[A-Za-z0-9_]+'" \
                  | sed -E 's/ +/ /g; s/ buffer size$/ buffer/' | sort | uniq -c \
                  | awk -v r="$name" -v m="$m" 'BEGIN { OFS = "\t" } { n = $1; sub(/^ *[0-9]+ /, ""); print r, m, $0, n }'
          done
      done; } > "$OUT/devices.tsv"
    for name in both cpu; do
        step "$name device lines: $(awk -F'\t' -v r="$name" '$1 == r { s[$3] += $4 } END { for (k in s) printf "%s x%d; ", k, s[k] }' "$OUT/devices.tsv")"
    done
    exit 0
fi

BIN=${BIN:?}; MODELS=${MODELS:?}; GPU=${GPU:?}; MOUNTS=${MOUNTS:?}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-fn72-ptx}
rm -f "$OUT/done"
read -r -a GPU_ARGS <<< "$GPU"
read -r -a EXTRA <<< "${DOCKER_ARGS:-}"
step "== ptx_check BIN=$BIN MODELS=$MODELS image $IMAGE"
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
    -e MODELS="$MODELS" -e NVCACHE="${NVCACHE:-$OUT/nvcache}" \
    "$IMAGE" bash "$SELF" > "$OUT/container.out" 2>&1
step "container exit $?"
date -Is > "$OUT/done"
