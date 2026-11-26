#!/usr/bin/env bash
# C1080 (Q1813): when the tool-call grammar starts on each build. Swift Qwen3.8-Flash-Next with Swift's own MTP head as
# the draft, on a build machine's newer NVIDIA cards running the rig build's compute 6.1 PTX. Correctness only, nothing
# timed. For each build in BUILDS, one llama-server with Flash-Next's v5 preset (the layer split changed to the cards
# here, -ts $TS) and verbose logging (-v: the slot's parameters with the grammar, every sampled token, every draft's
# acceptance, the grammar's own trigger lines); then, for each recorded agent-check request in REQS, two greedy arms
# (temperature 0, top-k 1, seed 1, no prompt cache, at most N_PREDICT tokens), in the order of ARMS:
#   chat   the request as the agent sent it (/v1/chat/completions with its four tools: the tool-call grammar on)
#   chat2  the same request again (whether the engine gives the same reply twice)
#   none   the same request with "tool_choice": "none": the same chat path, template and prompt (the tools are still
#          rendered), the same prompt split at the user messages, but no tool-call grammar
#   plain  the prompt as the chat template renders it (/apply-template), sent to /completion with no grammar: the
#          prompt is read in other calls (the chat path breaks them at user messages), so the numbers can differ
# Greedy decoding makes the arms comparable token for token: where the grammar never rejects the top token, chat and
# none are the same reply.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into the containers)
#   BUILDS    "label=<bin folder> ..." in the order they run
#   REQS      the request bodies (JSON files of /v1/chat/completions bodies), each named <name>.json
#   REPO      a copy of this repository (release/flash-next-v5/settings.env)
#   FN        Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's maker file (the first of three shards); DRAFT its MTP head
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the models)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before each load as "$PREFLIGHT -g $GPUS" (optional)
#   JIT_SEED     "label=<folder> ...": a PTX JIT cache to start each build's own from (optional)
#   GPUS (0,1), TS (1,1), PORT (18980), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), N_PREDICT (4096),
#   LOAD_TIMEOUT (1800 s), REQ_TIMEOUT (1800 s), NAME (c1080: the containers' name prefix), ARMS ("chat chat2 none plain")
set -uo pipefail
: "${WORK:?}" "${BUILDS:?}" "${REQS:?}" "${REPO:?}" "${FN:?}" "${DRAFT:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1}
TS=${TS:-1,1}
PORT=${PORT:-18980}
N_PREDICT=${N_PREDICT:-4096}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
REQ_TIMEOUT=${REQ_TIMEOUT:-1800}
NAME=${NAME:-c1080}
ARMS=${ARMS:-chat chat2 none plain}
PREFLIGHT=${PREFLIGHT:-}
URL=http://127.0.0.1:$PORT

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
rm -f "$WORK/DONE"
mkdir -p "$WORK"

# shellcheck disable=SC1091
. "$REPO/release/flash-next-v5/settings.env"
M_ENV=$PRESET_ENV
M_ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
[[ "$M_ARGS" == *"-ts $TS"* ]] || { say "no -ts in the preset to change"; exit 1; }

# the request bodies of each arm (python3 -I: no module from the working folder)
for r in $REQS; do
    n=$(basename "$r" .json)
    mkdir -p "$WORK/req"
    python3 -I - "$r" "$WORK/req/$n" "$N_PREDICT" <<'PY'
import json, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.load(open(src))
greedy = {"temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0, "seed": 1}
chat = {k: v for k, v in body.items() if k not in ("stream", "stream_options", "max_tokens")}
chat.update(greedy, stream=False, max_tokens=n, cache_prompt=False, return_tokens=True)
json.dump(chat, open(out + ".chat.json", "w"))
none = dict(chat, tool_choice="none")
json.dump(none, open(out + ".none.json", "w"))
tmpl = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
json.dump(tmpl, open(out + ".tmpl.json", "w"))
PY
done
md5sum $REQS "$WORK"/req/*.json > "$WORK/req/md5.txt"

post() {  # <path> <body file> <out file>
    curl -s -m "$REQ_TIMEOUT" -H 'Content-Type: application/json' -d @"$2" "$URL$1" > "$3" 2> "$3.err"
}

for b in $BUILDS; do
    label=${b%%=*}; bin=${b#*=}
    o=$WORK/$label; c=$NAME-$label
    mkdir -p "$o"
    md5sum "$bin/llama-server" > "$o/md5.txt"
    jit=$WORK/jit-$label
    if [ ! -d "$jit" ]; then
        seed=""
        for s in ${JIT_SEED:-}; do [ "${s%%=*}" = "$label" ] && seed=${s#*=}; done
        if [ -n "$seed" ] && [ -d "$seed" ]; then cp -a "$seed" "$jit"; else mkdir -p "$jit"; fi
    fi
    [ -n "$PREFLIGHT" ] && $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    printf 'env: %s\nargs: -m %s -md %s %s -v\n' "$M_ENV" "$FN" "$DRAFT" "$M_ARGS" > "$o/command.txt"
    docker rm -f "$c" > /dev/null 2>&1
    say "load $label: $bin"
    # shellcheck disable=SC2086
    docker run --rm -d ${DOCKER_ARGS:-} "${VOL[@]}" -e "CUDA_CACHE_PATH=$jit" -e CUDA_CACHE_MAXSIZE=4294967296 \
        --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $M_ENV $bin/llama-server -m $FN -md $DRAFT $M_ARGS -v --host 0.0.0.0 --port $PORT --no-webui \
            > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    up=0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "$URL/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
    say "load $label: $(cat "$o/load.txt")"
    if [ $up = 1 ]; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
        for r in $REQS; do
            n=$(basename "$r" .json)
            for a in $ARMS; do
                case $a in
                    chat|chat2) post /v1/chat/completions "$WORK/req/$n.chat.json" "$o/$n.$a.json" ;;
                    none) post /v1/chat/completions "$WORK/req/$n.none.json" "$o/$n.$a.json" ;;
                    plain)
                        post /apply-template "$WORK/req/$n.tmpl.json" "$o/$n.tmpl.out.json"
                        python3 -I - "$o/$n.tmpl.out.json" "$o/$n.plain.req.json" "$N_PREDICT" <<'PY'
import json, sys
prompt = json.load(open(sys.argv[1]))["prompt"]
json.dump({"prompt": prompt, "n_predict": int(sys.argv[3]), "temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0,
           "seed": 1, "cache_prompt": False, "return_tokens": True}, open(sys.argv[2], "w"))
PY
                        post /completion "$o/$n.plain.req.json" "$o/$n.$a.json" ;;
                    *) say "unknown arm $a"; continue ;;
                esac
                echo "$n $a" >> "$o/order.txt"
                say "  $label $n $a: $(wc -c < "$o/$n.$a.json") B"
            done
        done
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem-after.csv"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "done $label"
done
say "all done"
touch "$WORK/DONE"
