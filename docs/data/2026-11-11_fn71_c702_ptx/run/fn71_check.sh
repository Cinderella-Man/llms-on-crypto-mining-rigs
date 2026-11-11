#!/usr/bin/env bash
# FN71 (Q1298): C702's captured prompt graphs (GGML_CUDA_GRAPHS_FORCE=1, Miner Llama pascal-p106-v4-b12246-c702)
# against eager launches, with the rig build's own llama-perplexity in a container that holds one newer NVIDIA card.
# Correctness only. The rig build carries sm_61 machine code and compute_61 PTX only, so every ggml kernel it runs on
# that card is the PTX, JIT-compiled by the card's driver (cuBLAS is the image's own library). FN72's script shape.
# On a build machine with Docker and one free card:
#   1. cuda_preflight.sh on that card (PREFLIGHT, optional): CUDA works on it, or the run stops here;
#   2. one container with that card, running this script again with INSIDE=1:
#      a. the card, and the GPU code in llama-perplexity (cuobjdump);
#      b. with the fork's PTX-mode settings (the fork's Pascal kernel paths forced on with a P106's 5 SMs), one
#         llama-perplexity run per name in RUNS, each writing a KL-divergence base file (the log-probabilities of every
#         scored token over the whole vocabulary):
#         - off: the switch off (the warm-up's eager launches, as the release line);
#         - on:  GGML_CUDA_GRAPHS_FORCE=1, with GGML_CUDA_PROFILE=OUT/prof-on.csv GGML_CUDA_PROFILE_MODE=graph (one
#                line per graph evaluation: eager, capture or replay).
#         Each: -m MODEL -f TEXT -c 2048 -b 2048 -ub UB -fa on --chunks CHUNKS -ngl NGL --fit off -lv 4 PPL_ARGS (the load
#         lines that name each buffer's device are logged from -lv 4 on), its exit code,
#         the card's largest used memory while it ran (nvidia-smi every 0.5 s), the files in the driver's JIT cache
#         after it (a kernel that ran on the card leaves its JIT output there; emptied first), the load lines that
#         name each buffer's device, and every chunk's perplexity;
#      c. cmp of the two base files and their md5, the perplexities compared in every digit, and the profile's counts
#         per graph and per kind of evaluation.
# Env: BIN (bin/ of the branch's rig tarball), MODEL, TEXT (wikitext-2 test, wiki.test.raw), OUT, KLD (folder for the
#   base files, default OUT; they are 0.5 GB a chunk with a 248k vocabulary), NGL (layers on the card, default 99),
#   CHUNKS (default 8), UB (the micro-batch, default 512), PPL_ARGS (more llama-perplexity arguments, default none),
#   RUNS (default "off on"), KEEP_KLD (1 keeps the base files, default: deleted after the cmp),
#   GPU (docker's request for one card, e.g. "--device nvidia.com/gpu=0"), MOUNTS (folders mounted at the same path),
#   MOUNTS_RO (the same, read-only: the model's folder), IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04), NAME
#   (container, default fn71-ptx), DOCKER_ARGS (labels, --cpu-shares), PREFLIGHT (path of cuda_preflight.sh,
#   optional), NVCACHE (the driver's JIT cache folder, default OUT/nvcache; emptied first)
# Output: OUT/steps.txt (one line per step), OUT/archs.txt, OUT/<run>.log (the run's console), OUT/mem-<run>.txt,
#   OUT/devices.tsv (run, what, line), OUT/ppl.tsv (run, chunk, perplexity), OUT/kld.txt (sizes, md5, cmp),
#   OUT/prof-on.csv and OUT/prof-graphs.tsv (graph, eager, capture, replay, not ready, the sequence), OUT/done
set -uo pipefail
OUT=${OUT:?}
mkdir -p "$OUT"
S=$OUT/steps.txt
step() { echo "$(date -Is) $*" | tee -a "$S"; }

if [ "${INSIDE:-0}" = 1 ]; then
    NVC=${NVCACHE:-$OUT/nvcache}
    K=${KLD:-$OUT}
    mkdir -p "$K"
    rm -rf "$NVC"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_CACHE_PATH=$NVC CUDA_CACHE_MAXSIZE=4294967296
    PTX="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
         GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
         GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
    for kv in $PTX; do export "${kv?}"; done
    step "inside: $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv,noheader 2>&1 | tr '\n' ';')"
    step "PTX-mode settings: $(echo $PTX)"
    T=$BIN/llama-perplexity
    {
        echo "== llama-perplexity md5 $(md5sum < "$T" | cut -d' ' -f1)"
        echo "machine code: $(cuobjdump --list-elf "$T" | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        echo "PTX: $(cuobjdump --list-ptx "$T" | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    } > "$OUT/archs.txt" 2>&1
    step "archs: $(tr '\n' ' ' < "$OUT/archs.txt")"
    step "model $(basename "$MODEL"): $(stat -L -c %s "$MODEL") B; text $(basename "$TEXT"): $(stat -L -c %s "$TEXT") B"
    jit() { echo "$(find "$NVC" -type f 2>/dev/null | wc -l) files, $(du -sb "$NVC" 2>/dev/null | cut -f1) B"; }
    nocol() { sed 's/\x1b\[[0-9;]*m//g'; }
    : > "$OUT/devices.tsv.tmp"
    : > "$OUT/ppl.tsv.tmp"
    for name in ${RUNS:-off on}; do
        envs=()
        case $name in
            on) rm -f "$OUT/prof-on.csv"
                envs=(GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_PROFILE="$OUT/prof-on.csv" GGML_CUDA_PROFILE_MODE=graph) ;;
        esac
        rm -f "$K/$name.kld"
        ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/mem-$name.txt" 2>&1 &
        mon=$!
        t0=$(date +%s)
        step "$name: start ${envs[*]:-(no switch)}"
        ( cd "$OUT" && env "${envs[@]}" nice -n 19 "$T" -m "$MODEL" -f "$TEXT" -c 2048 -b 2048 -ub "${UB:-512}" -fa on \
              --chunks "${CHUNKS:-8}" -ngl "${NGL:-99}" --fit off -lv 4 ${PPL_ARGS:-} --kl-divergence-base "$K/$name.kld" ) 2>&1 | nocol > "$OUT/$name.log"
        rc=${PIPESTATUS[0]}
        t1=$(date +%s)
        sleep 1
        kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
        mx=$(grep -E '^[0-9]+$' "$OUT/mem-$name.txt" | sort -n | tail -n 1)
        step "$name: rc=$rc after $((t1 - t0)) s; card memory max ${mx:-?} MiB over $(grep -c -E '^[0-9]+$' "$OUT/mem-$name.txt") samples; JIT cache $(jit)"
        step "$name: $(grep -m 1 -o 'Final estimate: .*' "$OUT/$name.log" || echo 'no final estimate')"
        # where the model ran: the load and context lines that name a device
        grep -E "GGML_CUDA_GRAPHS_FORCE|GGML_CUDA_PROFILE_MODE|Device [0-9]+:|offloading|offloaded [0-9]+/[0-9]+ layers|(model|KV|RS|compute|output) buffer size|graph splits|n_layer|failed|error|ERROR|WARN" "$OUT/$name.log" \
            | sed -E 's/^[0-9.]+ [A-Z] +//; s/^[a-z_]+: +//; s/ +/ /g' | awk -v r="$name" 'BEGIN { OFS = "\t" } { print r, $0 }' >> "$OUT/devices.tsv.tmp"
        # every chunk's perplexity, as printed
        grep -o -E '\[[0-9]+\][0-9]+\.[0-9]+' "$OUT/$name.log" | sed -E 's/^\[([0-9]+)\]/\1\t/' \
            | awk -v r="$name" 'BEGIN { OFS = "\t" } { print r, $0 }' >> "$OUT/ppl.tsv.tmp"
    done
    { printf 'run\tline\n'; cat "$OUT/devices.tsv.tmp"; } > "$OUT/devices.tsv"
    { printf 'run\tchunk\tppl\n'; cat "$OUT/ppl.tsv.tmp"; } > "$OUT/ppl.tsv"
    rm -f "$OUT/devices.tsv.tmp" "$OUT/ppl.tsv.tmp"
    set -- ${RUNS:-off on}
    if [ $# -ge 2 ]; then
        a=$1 b=$2
        {
            for n in "$a" "$b"; do echo "$n.kld $(stat -c %s "$K/$n.kld" 2>/dev/null || echo missing) B md5 $(md5sum < "$K/$n.kld" 2>/dev/null | cut -d' ' -f1)"; done
            if cmp "$K/$a.kld" "$K/$b.kld" > "$OUT/cmp.out" 2>&1; then echo "cmp $a.kld $b.kld: identical"; else echo "cmp $a.kld $b.kld: DIFFER: $(head -n 1 "$OUT/cmp.out")"; fi
        } > "$OUT/kld.txt"
        step "kld: $(tr '\n' ';' < "$OUT/kld.txt")"
        pa=$(awk -F'\t' -v r="$a" '$1 == r { printf "%s:%s ", $2, $3 }' "$OUT/ppl.tsv")
        pb=$(awk -F'\t' -v r="$b" '$1 == r { printf "%s:%s ", $2, $3 }' "$OUT/ppl.tsv")
        fa=$(grep -m 1 -o 'Final estimate: .*' "$OUT/$a.log"); fb=$(grep -m 1 -o 'Final estimate: .*' "$OUT/$b.log")
        if [ -n "$pa" ] && [ "$pa" = "$pb" ] && [ -n "$fa" ] && [ "$fa" = "$fb" ]; then
            step "perplexity: $a = $b in every digit (chunks $pa; $fa)"
        else
            step "perplexity: $a != $b ($a: $pa $fa; $b: $pb $fb)"
        fi
        [ "${KEEP_KLD:-0}" = 1 ] || { rm -f "$K/$a.kld" "$K/$b.kld"; step "base files deleted"; }
    fi
    # the graph profile: per graph number, the evaluations by kind and their sequence (e eager, c capture, r replay;
    # upper case a line written as not ready or unmatched)
    if [ -s "$OUT/prof-on.csv" ]; then
        awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
            { g = $c["graph"]; h = $c["how"]; s = substr(h, 1, 1); if ($c["group"] == -1) { s = toupper(s); nr[g]++ } else n[g, h]++
              seq[g] = seq[g] s; seen[g] = 1 }
            END { OFS = "\t"; print "graph", "eager", "capture", "replay", "not_ready_or_unmatched", "sequence"
                  for (g in seen) print g, n[g, "eager"] + 0, n[g, "capture"] + 0, n[g, "replay"] + 0, nr[g] + 0, seq[g] }' \
            "$OUT/prof-on.csv" | sort -t$'\t' -k1,1n > "$OUT/prof-graphs.tsv"
        step "profile: $(($(wc -l < "$OUT/prof-on.csv") - 1)) lines; by kind: $(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next } { k = $c["how"] ($c["group"] == -1 ? " (" $c["label"] ")" : ""); n[k]++ } END { for (k in n) printf "%s %d; ", k, n[k] }' "$OUT/prof-on.csv")graphs: $(($(wc -l < "$OUT/prof-graphs.tsv") - 1))"
    fi
    exit 0
fi

BIN=${BIN:?}; MODEL=${MODEL:?}; TEXT=${TEXT:?}; GPU=${GPU:?}; MOUNTS=${MOUNTS:?}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
NAME=${NAME:-fn71-ptx}
rm -f "$OUT/done"
read -r -a GPU_ARGS <<< "$GPU"
read -r -a EXTRA <<< "${DOCKER_ARGS:-}"
step "== fn71_check BIN=$BIN MODEL=$MODEL NGL=${NGL:-99} CHUNKS=${CHUNKS:-8} UB=${UB:-512} PPL_ARGS=${PPL_ARGS:-} RUNS=${RUNS:-off on} image $IMAGE"
if [ -n "${PREFLIGHT:-}" ]; then
    pf=$(IMAGE=$IMAGE PREFLIGHT_DOCKER_ARGS="${DOCKER_ARGS:-}" "$PREFLIGHT" -q -- "${GPU_ARGS[@]}" 2>&1 | tail -n 1)
    step "preflight: $pf"
    [[ $pf == "PREFLIGHT OK"* ]] || { step "stopped: preflight"; date -Is > "$OUT/done"; exit 1; }
fi
V=()
for m in $MOUNTS; do V+=(-v "$m:$m"); done
for m in ${MOUNTS_RO:-}; do V+=(-v "$m:$m:ro"); done
SELF=$(realpath "$0")
step "container $NAME"
docker run --rm --name "$NAME" "${EXTRA[@]}" "${GPU_ARGS[@]}" "${V[@]}" -e INSIDE=1 -e OUT="$OUT" -e BIN="$BIN" \
    -e MODEL="$MODEL" -e TEXT="$TEXT" -e KLD="${KLD:-$OUT}" -e NGL="${NGL:-99}" -e CHUNKS="${CHUNKS:-8}" -e UB="${UB:-512}" -e PPL_ARGS="${PPL_ARGS:-}" \
    -e RUNS="${RUNS:-off on}" -e KEEP_KLD="${KEEP_KLD:-0}" -e NVCACHE="${NVCACHE:-$OUT/nvcache}" \
    "$IMAGE" bash "$SELF" > "$OUT/container.out" 2>&1
step "container exit $?"
date -Is > "$OUT/done"
