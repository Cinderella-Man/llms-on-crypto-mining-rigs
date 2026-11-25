#!/usr/bin/env bash
# The card checks of IQ2_XXS's fast multi-token kernel (C1138) on one card of a machine with newer GPUs, with the
# fork's Pascal kernels forced (scripts/pascal_forced.sh). Steps (STEPS, default "bops model"):
#   bops   test-backend-ops test -b CUDA0 -o MUL_MAT_ID (llama.cpp's repeated-id cases left out, C315) with
#          GGML_CUDA_MTOK_IQ_FAST at 0 and at each value of MODES, and at the first value of MODES with the test fault
#          GGML_CUDA_MTOK_IQ_FAST_FAULT=1 (the fast kernel takes up and gate the wrong way round: the gate/up cases it
#          takes must then fail, which shows which cases reach it)
#   model  Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's gmix-q2 file (MODEL, its first shard) on the card: the IQ2_XXS
#          layers' routed gate/up experts on the card, every other layer's experts in host memory (-ot), with the MTP
#          draft (DRAFT) so that replies are made of 2-4-token verify passes, and prompts read in micro-batches of 8
#          tokens; the Flash-Next v5 preset's switches (release/flash-next-v5/settings.env, forced as a P106 runs them)
#          with GGML_CUDA_MTOK_IQ_FAST at 0 and at each value of MODES; two greedy replies of NPRED tokens with the 8 most
#          likely tokens' probabilities at every step; the replies, their token ids and those probabilities compared
#          with the 0 arm's
# Env: WORK (output folder), BIN (the build's bin/), REPO (a folder with pascal_forced.sh and settings.env of
# release/flash-next-v5), GPU (docker's device index), MODES (default "1"), MODEL, DRAFT, NPRED (default 192), MOUNTS
# (mounted at the same paths), JITCACHE, DOCKER_ARGS, IMAGE, PORT (default 8338), LOAD_TIMEOUT (seconds, default 1800)
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${REPO:?}" "${GPU:?}"
STEPS=${STEPS:-bops model}
MODES=${MODES:-1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PORT=${PORT:-8338}
NPRED=${NPRED:-192}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
# shellcheck disable=SC1091
. "$REPO/pascal_forced.sh"
mkdir -p "$WORK"
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m"); done
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" ${JITCACHE:+-e CUDA_CACHE_PATH=$JITCACHE} -e CUDA_CACHE_MAXSIZE=4294967296)
say() { echo "$(date +%H:%M:%S) $*"; }

if [[ " $STEPS " == *" bops "* ]]; then
    o=$WORK/bops; mkdir -p "$o"; : > "$o/rc.txt"
    md5sum "$BIN/test-backend-ops" > "$o/md5.txt"
    bops() {  # <tag> <environment>
        "${DR[@]}" --gpus "device=$GPU" "$IMAGE" bash -c "env $PASCAL_FORCED $2 timeout 10800 $BIN/test-backend-ops test -b CUDA0 \
            -o MUL_MAT_ID -p '^(?!.*dup=1)' 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $o/$1.log; echo $1 rc=\${PIPESTATUS[0]} >> $o/rc.txt" \
            > /dev/null 2>&1
        say "bops $1: $(grep "^$1 " "$o/rc.txt"), $(grep -aoE '[0-9]+/[0-9]+ tests passed' "$o/$1.log" | tail -n 1), FAIL lines $(grep -a -c ': FAIL' "$o/$1.log")"
    }
    bops off GGML_CUDA_MTOK_IQ_FAST=0
    for m in $MODES; do bops "fast$m" GGML_CUDA_MTOK_IQ_FAST="$m"; done
    set -- $MODES
    bops fault "GGML_CUDA_MTOK_IQ_FAST=$1 GGML_CUDA_MTOK_IQ_FAST_FAULT=1"
    # the cases and their verdicts, arm against arm
    for t in off $(for m in $MODES; do echo "fast$m"; done) fault; do
        grep -aE '^ +MUL_MAT_ID' "$o/$t.log" | sed -E 's/^ +//' | grep -aE ': (OK|FAIL)' | sed -E 's/ +[0-9.]+ ?(ms|us).*//' > "$o/$t.cases"
    done
    for t in $(for m in $MODES; do echo "fast$m"; done) fault; do
        say "bops $t against off: $(wc -l < "$o/$t.cases") cases, $(diff "$o/off.cases" "$o/$t.cases" | grep -c '^>') with another verdict"
    done
    grep -a ': FAIL' "$o/fault.log" | grep -a MUL_MAT_ID | sed -E 's/^ +//' | cut -c1-200 > "$o/fault-fails.txt"
    say "bops fault: failing cases $(wc -l < "$o/fault-fails.txt"), of them iq2_xxs $(grep -c 'type_a=iq2_xxs' "$o/fault-fails.txt")"
fi

if [[ " $STEPS " == *" model "* ]]; then
    : "${MODEL:?}" "${DRAFT:?}"
    # shellcheck disable=SC1091
    . "$REPO/settings.env"
    M_ENV="$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")"
    IQL='1|3|6|7|9|10|11|14|27|29|30'  # gmix-q2's IQ2_XXS layers
    M_ARGS="-ngl 99 -ngld 99 -fa on -c 8192 -np 1 -b 512 -ub 8 --spec-type draft-mtp --spec-draft-n-max 3"
    M_ARGS="$M_ARGS -ot 'blk\\.($IQL)\\.ffn_(gate|up)_exps=CUDA0' -ot 'ffn_.*_exps=CPU'"
    P1='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
    P2='You are a coding agent. The test tests/test_cache.py::test_eviction fails with KeyError: 3 after cache.put(3, "c") on a full LRU cache of capacity 2. Explain the likely bug in the eviction code and give the fixed put method.'
    load() {  # <tag> <environment>
        local o=$WORK/model/$1 c=c1138-model-$1 up=0 t0
        mkdir -p "$o"
        docker rm -f "$c" > /dev/null 2>&1
        printf 'env: %s %s\nargs: -m %s -md %s %s\n' "$M_ENV" "$2" "$MODEL" "$DRAFT" "$M_ARGS" > "$o/command.txt"
        md5sum "$BIN/llama-server" >> "$o/command.txt"
        "${DR[@]}" -d --name "$c" --gpus "device=$GPU" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" \
            bash -c "env $M_ENV $2 $BIN/llama-server -m $MODEL -md $DRAFT $M_ARGS --host 0.0.0.0 --port $PORT --no-webui \
                > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
        t0=$(date +%s)
        while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
            if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
            docker ps -q --filter "name=^$c\$" | grep -q . || break
            sleep 5
        done
        echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
        if [ $up = 1 ]; then
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" > "$o/mem.txt"
            local i=1
            for p in "$P1" "$P2"; do
                python3 -I -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False, "return_tokens": True, "n_probs": 8}))' "$p" "$NPRED" |
                    curl -s -m 1800 -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/completion" > "$o/reply$i.json" 2> "$o/reply$i.err"
                i=$((i + 1))
            done
        fi
        docker stop -t 60 "$c" > /dev/null 2>&1
        docker rm -f "$c" > /dev/null 2>&1
        say "model $1: $(cat "$o/load.txt"), card MiB $(cat "$o/mem.txt" 2>/dev/null), replies $(wc -c < "$o/reply1.json" 2>/dev/null) and $(wc -c < "$o/reply2.json" 2>/dev/null) B"
    }
    load off GGML_CUDA_MTOK_IQ_FAST=0
    for m in $MODES; do load "fast$m" GGML_CUDA_MTOK_IQ_FAST="$m"; done
    for m in $MODES; do
        for i in 1 2; do
            python3 -I - "$WORK/model/off/reply$i.json" "$WORK/model/fast$m/reply$i.json" "fast$m reply $i" <<'PY' | tee -a "$WORK/model/compare.txt"
import json, sys
def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"error": str(e)}
a, b = load(sys.argv[1]), load(sys.argv[2])
ta, tb = a.get("tokens"), b.get("tokens")
pa, pb = a.get("completion_probabilities"), b.get("completion_probabilities")
t = a.get("timings", {})
print(f"{sys.argv[3]}: tokens {len(ta or [])} against {len(tb or [])}, {'the same token ids' if ta == tb and ta else 'TOKEN IDS DIFFER'}; "
      f"text {'the same' if a.get('content') == b.get('content') and ta else 'DIFFERS'}; "
      f"top-8 probabilities at {len(pa or [])} steps: {'the same to the last digit' if pa == pb and pa else 'DIFFER'}; "
      f"draft tokens {t.get('draft_n')}, accepted {t.get('draft_n_accepted')} (the 0 arm)")
PY
        done
    done
fi
say "done; containers left: $(docker ps -q --filter label=c1138=1 | wc -l)"
