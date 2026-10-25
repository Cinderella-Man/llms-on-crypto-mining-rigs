#!/usr/bin/env bash
# The "regenerate" pattern on the rig: one pipelined llama-server (--pipe-depth 1 --pipe-guess none, -bs, -np 1, context
# checkpoints and prompt caching at their defaults) gets the chat prompt three times (T 0, then T 1 with seeds 1 and 2)
# and the 750-token rewrite prompt three times the same way, as a chat UI's "regenerate" sends them. The third rewrite
# request re-reads one prompt token and supersedes a context checkpoint. The server runs under gdb, so a crash leaves a
# backtrace in the log. One run per entry of RUNS, each a fresh server:
#   base        the build's defaults
#   tile0       GGML_CUDA_FATTN_TILE_PASCAL=0 (compute 6.1's prompt attention kernel off)
#   hostmask    LLAMA_GPU_KQ_MASK=0 (the attention mask built on the host)
#   reserve     LLAMA_SAMPLER_RESERVE=1 (a scheduler reserve at every sampler attach and detach)
#   plain       no --pipe-depth
# Run it on the rig, from this repository (~6 min per run on nine cards):
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> [GPUS=0,...,8] [RUNS='base tile0 hostmask reserve']
#       [ARGS='-ngl 99 -fa on -c 8192 -ub 512 -lv 4'] [PIPE='--pipe-depth 3 --pipe-guess none'] bash /tmp/pipespec/rig_regenerate.sh"
# Without gdb on the machine the server runs as is: a crash then shows as a failed request and the server's exit status
# (139 for a segfault), and the kernel's log line (dmesg) gives the faulting function's offset.
# Output in /tmp/pipespec/out/<TAG>/: <run>.log (server log, gdb's backtrace at the end on a crash), <run>.txt (client).
# Correctness only: the requests' speeds are not kept.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
RUNS=${RUNS:-base}
ARGS=${ARGS:--ngl 99 -fa on -c 8192 -ub 512 -lv 4}
PIPE=${PIPE:---pipe-depth 3 --pipe-guess none}
PORT=${PORT:-8097}
GDB=()
if command -v gdb > /dev/null; then
    # after a crash: the backtrace, and for the crash seen on rc3 (ggml_gallocr_alloc_graph reading a leaf's stale
    # allocation, the leaf in rbp) the leaf's name (ggml_tensor.name at +0x100) and its index (rbx)
    GDB=(gdb -q -batch -ex 'handle SIGPIPE nostop noprint' -ex 'handle SIGTERM nostop noprint pass' -ex run
         -ex 'bt 50' -ex 'x/s $rbp+0x100' -ex 'p $rbx' -ex 'thread apply all bt 8' --args)
fi
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

for run in $RUNS; do
    envs=""; pipe=$PIPE
    case $run in
        base)     ;;
        tile0)    envs="GGML_CUDA_FATTN_TILE_PASCAL=0" ;;
        hostmask) envs="LLAMA_GPU_KQ_MASK=0" ;;
        reserve)  envs="LLAMA_SAMPLER_RESERVE=1" ;;
        plain)    pipe="" ;;
        *) echo "unknown run $run"; continue ;;
    esac
    # shellcheck disable=SC2086
    env $envs CUDA_VISIBLE_DEVICES="$GPUS" "${GDB[@]}" "$BIN/llama-server" -m "$MODEL" $ARGS -np 1 -bs --cache-ram 0 \
        --no-webui --host 127.0.0.1 --port "$PORT" $pipe > "$OUT/$run.log" 2>&1 < /dev/null &
    srv=$!
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 $srv 2> /dev/null || { echo "$run: the server exited during load"; break; }
        sleep 2
    done
    python3 -u - "$PORT" "$HERE/prompts" <<'EOF' > "$OUT/$run.txt" 2>&1
import json, sys, urllib.request
port, pdir = sys.argv[1], sys.argv[2]
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "samplers": ["top_k", "top_p", "min_p", "temperature"]}
i = 0
for p in ("chat", "rewrite"):
    msgs = [{"role": "user", "content": open(f"{pdir}/{p}.txt").read()}]
    for temp, seed in ((0.0, 1), (1.0, 1), (1.0, 2)):
        i += 1
        body = dict(messages=msgs, temperature=temp, seed=seed, max_tokens=256, **SAMPLING)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            print(f"request {i} {p} T{temp:g} seed {seed}: ok, {r['timings']['predicted_n']} tokens")
        except Exception as e:
            print(f"request {i} {p} T{temp:g} seed {seed}: FAILED ({type(e).__name__})")
            sys.exit(1)
EOF
    sleep 3
    # stop the server itself (under gdb its child; gdb passes SIGTERM on), so that a backtrace in the log means a crash
    if [ ${#GDB[@]} -gt 0 ]; then
        for p in $(pgrep -P $srv); do kill -TERM $p 2> /dev/null; done
        sleep 5; kill $srv 2> /dev/null; wait $srv 2> /dev/null
        status=-
    else
        kill -0 $srv 2> /dev/null && { kill -TERM $srv; sleep 5; }
        wait $srv 2> /dev/null; status=$?
    fi
    sed -i -E '/ time =|tokens per second|t\/s/d' "$OUT/$run.log"
    echo "== $run: $(grep -c ': ok' "$OUT/$run.txt") of 6 requests ok; $(grep -m1 'FAILED' "$OUT/$run.txt") (server exit $status; 139 = segfault, 143 = stopped)"
    grep -m1 -E 'received signal|SIGSEGV|SIGABRT|GGML_ASSERT|CUDA error' "$OUT/$run.log"
    grep -E '^#[0-9]+ |^0x[0-9a-f]+:|^\$1 = ' "$OUT/$run.log" | head -25
    command -v gdb > /dev/null || dmesg 2> /dev/null | grep -E 'llama-server.*segfault' | tail -1
done
