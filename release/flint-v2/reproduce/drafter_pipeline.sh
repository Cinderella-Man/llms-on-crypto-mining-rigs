#!/usr/bin/env bash
# drafter_pipeline.sh: the staircase drafter from scratch on one machine, step after step (release/flint-v2/reproduce/
# drafter.md). The project ran the same commands with several copies in parallel on a 7-GPU machine; this script is the
# plain sequential form, so every step and its arguments are in one public place.
#
#   DATA=<data dir> LLAMA=<llama.cpp tree with llama-server and llama-staircase-dump built> \
#   Q8=<flint-Q8_0.gguf> HF=<Hugging Face cache> STD=<python stdlib dir, e.g. /usr/lib/python3.12> \
#   release/flint-v2/reproduce/drafter_pipeline.sh
#
# Steps (STEPS, default all, in this order):
#   prompts   prompts.jsonl, prompts-ext.jsonl, prompts-ext2.jsonl, prompts-tools.jsonl (76,431 prompts)
#   answers   Flint's answers from llama-server (the makers' sampling, seed 20000 + prompt number, <= 2,048 tokens)
#   testset   Q51, the 38 held-out transcripts: shipped in data/q51/ (copied; TESTSET_REBUILD=1 regenerates them)
#   dump      hidden states (26-layer grid) and top-20 targets from Flint Q8_0, for the answers and for Q51
#   vocab     the 40,000-id draft vocabulary
#   train     U from scratch, R1 from U (six-card windows), then the depth-general head from R1
#   score     the general head on Q51, per card count
#   export    the GGUF side file for the engine
#
# Env [defaults]: DATA, LLAMA, Q8, HF, STD, PYTHON [python3], GPUS [0] (CUDA_VISIBLE_DEVICES for every GPU step),
#   SLOTS [24] (llama-server slots for the answers), PORT [8001], PROMPT_FILES [all four], U_MIN [300], R1_MIN [480],
#   GEN_MIN [134], CARDS (e.g. "6,9" for a specialist head instead of the 3-12-card head), FP16 [0: 1 on V100s],
#   STEPS [all]. Data sizes: ~58 GB per million answer tokens for the dump; plan disk accordingly.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
SD=$repo/bench/staircase_drafter
: "${DATA:?DATA: the data directory}" "${LLAMA:?LLAMA: a llama.cpp tree with the tools built}"
PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-0}
SLOTS=${SLOTS:-24}
PORT=${PORT:-8001}
STEPS=${STEPS:-prompts answers testset dump vocab train score export}
PROMPT_FILES=${PROMPT_FILES:-prompts.jsonl prompts-ext.jsonl prompts-ext2.jsonl prompts-tools.jsonl}
BIN=$LLAMA/build/bin
export PYTHONPATH="$LLAMA/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DATA"/{claims,feats,heads,scores,q51}
cd "$SD"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
log() { echo "$(date '+%F %T') $*"; }
gpu() { CUDA_VISIBLE_DEVICES=$GPUS "$@"; }
fp16=(); [ "${FP16:-0}" = 1 ] && fp16=(--fp16)

if has prompts; then
    : "${HF:?HF}" "${STD:?STD}"
    log "prompts"
    $PYTHON prompts.py "$DATA/prompts.jsonl" "$HF" "$STD" "$LLAMA" --seed 20 --scale 1
    $PYTHON prompts.py "$DATA/prompts-ext.jsonl" "$HF" "$STD" "$LLAMA" --scale 3 --seed 21 --extend "$DATA/prompts.jsonl"
    cat "$DATA/prompts.jsonl" "$DATA/prompts-ext.jsonl" > "$DATA/prompts-all12.jsonl"
    $PYTHON prompts.py "$DATA/prompts-ext2.jsonl" "$HF" "$STD" "$LLAMA" --scale 6 --seed 22 --extend "$DATA/prompts-all12.jsonl"
    cat "$DATA/prompts.jsonl" "$DATA/prompts-ext.jsonl" "$DATA/prompts-ext2.jsonl" > "$DATA/used.jsonl"
    $PYTHON tool_prompts.py "$DATA/prompts-tools.jsonl" "$HF" --extend "$DATA/used.jsonl"
fi

serve() {  # serve MODEL: llama-server for many slots on GPUS, until stop_server
    gpu "$BIN/llama-server" -m "$1" -ngl 99 -fa on --jinja -np "$SLOTS" -c $((SLOTS * 6144)) -no-kvu -bs -b 2048 -ub 512 \
        --cache-ram 0 --host 127.0.0.1 --port "$PORT" > "$DATA/server.log" 2>&1 &
    SRV=$!
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 "$SRV" 2>/dev/null || { echo "llama-server exited, see $DATA/server.log"; exit 1; }
        sleep 5
    done
}
stop_server() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true; }

if has answers; then
    : "${Q8:?Q8: flint-Q8_0.gguf}"
    log "answers: $PROMPT_FILES with ${WRITER:-$Q8}"
    serve "${WRITER:-$Q8}"
    i=0
    for p in $PROMPT_FILES; do
        i=$((i + 1))
        $PYTHON generate.py "http://127.0.0.1:$PORT" "$DATA/$p" "$DATA/gen-$i.jsonl" "$DATA/claims" --conc "$SLOTS" \
            --n-predict 2048 --max-prompt 4096
    done
    stop_server
fi

if has testset; then
    if [ "${TESTSET_REBUILD:-0}" = 1 ]; then
        log "testset: regenerating the 38 Q51 transcripts (bench/expert_overlap, Unsloth Q8_0)"
        serve "$Q8"
        $PYTHON "$repo/bench/expert_overlap/generate.py" "http://127.0.0.1:$PORT" "$here/data/q51/prompts.jsonl" \
            "$DATA/q51/gen.jsonl" "$DATA/q51/seqs.txt" 8 3072
        stop_server
    else
        log "testset: the shipped Q51 transcripts"
        cp "$here/data/q51/gen.jsonl" "$here/data/q51/seqs.txt" "$DATA/q51/"
    fi
fi

GRID=4,6,8,9,10,12,14,16,17,18,20,22,24,25,27,29,30,33,34,36,37,39,41,42,44,47
dump() {  # dump SEQS OUT [PACK]: the grid file, plus the six-card pack (layers 9, 17, 25, 33, 41) when PACK is given
    local pack=(); [ -n "${3:-}" ] && pack=(DUMP_OUT2="$3" DUMP_LAYERS2=9,17,25,33,41)
    env DUMP_IN="$1" DUMP_OUT="$2" DUMP_LAYERS=$GRID DUMP_ALLPOS=4,6,9 DUMP_THREADS=8 "${pack[@]}" \
        CUDA_VISIBLE_DEVICES="$GPUS" "$BIN/llama-staircase-dump" -m "$Q8" -ngl 99 -fa on -c 8192 -b 2048 -ub 512
}
if has dump; then
    log "dump: answers (grid and six-card pack) and Q51 (grid)"
    mkdir -p "$DATA/pack"
    $PYTHON feats.py seqs "$DATA/seqs.txt" "$DATA/map.jsonl" "$DATA"/gen-*.jsonl
    dump "$DATA/seqs.txt" "$DATA/feats/train-1.bin" "$DATA/pack/train-1.bin"
    dump "$DATA/q51/seqs.txt" "$DATA/q51-grid.bin"
    $PYTHON flint_weights.py "$Q8" "$DATA/flint/"
    $PYTHON check_dump.py "$DATA/feats/train-1.bin" "$DATA/flint/"
fi

if has vocab; then
    log "vocab"
    $PYTHON vocab.py "$DATA/vocab.npy" "$DATA"/gen-*.jsonl --size 40000 --test "$DATA/q51/gen.jsonl"
fi

common=(--flint "$DATA/flint/" --vocab "$DATA/vocab.npy")
if has train; then
    log "train: U (scratch), R1 (six-card windows, fused), then the depth-general head"
    gpu $PYTHON train.py --head U --feats "$DATA"/pack/train-*.bin "${common[@]}" --out "$DATA/heads/U" \
        --minutes "${U_MIN:-300}" --lr 2e-4 --warmup 100 "${fp16[@]}"
    gpu $PYTHON train.py --head A --init-from "$DATA/heads/U/head.pt" --fuse --window-sample 0.6,0.25,0.15 \
        --feats "$DATA"/pack/train-*.bin "${common[@]}" --out "$DATA/heads/R1" --lr 1e-4 --warmup 100 \
        --minutes "${R1_MIN:-480}" --val-seqs 300 "${fp16[@]}"
    cards=(); [ -n "${CARDS:-}" ] && cards=(--cards "$CARDS")
    gpu $PYTHON train.py --head A --general "${cards[@]}" --init-from "$DATA/heads/R1/head.pt" \
        --feats "$DATA"/feats/train-1.bin "${common[@]}" --out "$DATA/heads/R2" --lr 1e-4 --warmup 100 --eval-every 20 \
        --val-seqs 300 --minutes "${GEN_MIN:-134}" "${fp16[@]}"
fi

if has score; then
    log "score: R2 on Q51"
    gpu $PYTHON score.py --heads "$DATA/heads/R2" --feats "$DATA/q51-grid.bin" --gen "$DATA/q51/gen.jsonl" "${common[@]}" \
        --out "$DATA/scores/R2" --alpha 3 --window-group 2
    $PYTHON report.py "$DATA/scores/R2/summary.json"
fi

if has export; then
    log "export: $DATA/drafter.gguf"
    $PYTHON export_gguf.py --head "$DATA/heads/R2" --vocab "$DATA/vocab.npy" --out "$DATA/drafter.gguf" --alpha 3
fi
log "done"
