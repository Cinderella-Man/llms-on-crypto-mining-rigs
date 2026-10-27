#!/usr/bin/env bash
# serve_prefill.sh: prompt reading as a coding agent sees it, measured on llama-server itself. Per configuration one
# server on the rig and three requests, the server's own timings:
#   1. fresh: a ~13.4k-token prompt, 16 tokens out: prompt tokens per second and the request's wall time
#   2. next turn: the same prompt plus ~2k new tokens, so the cached prefix is reused as in an agent's next turn: the
#      new prompt tokens, their speed and the server's prompt time (the wait for the first token)
#   3. decode: the turn-2 prompt again, 256 tokens out at temperature 0: generation speed at ~15k tokens of context
# The prompt is the source of Python's standard library on the rig (the *.py files of STDLIB, PSF licence), so nothing is
# downloaded; another Python version gives other token counts. llama-batched-bench, which reads one long prompt with no
# cache, reported 2.3x less prompt speed than this on the rig (docs/data/2026-10-26_release_setup).
#
#   RIG=rig BIN=/opt/llama-builds/<build>/bin MODEL=<gguf on the rig> TS=6,6,5,5,6,5,5,6,6 scripts/serve_prefill.sh [OUT]
#
# Env: RIG (ssh alias, default rig), BIN (llama-server's folder on the rig), MODEL, TS (layer split for -ts, empty for
# none), CONFIGS ('label|context|batch flags', ';'-separated; default the four of 2026-10-26 below), SERVER_ARGS (flags
# for every configuration; default '-np 1 -bs --pipe-depth 3 --ctx-checkpoints 4 --cache-ram 0', drop --pipe-depth on
# a build without it), GPUS (default all nine), PORT (on the rig's loopback, default 8095), STDLIB (default
# /usr/lib/python3.12, Ubuntu 24.04's). OUT, if given, also receives the output.
# It stops any llama-server running on the rig first, and stops its own after each configuration; it leaves nothing on
# the rig. Run it with nothing else on the rig.
set -uo pipefail
RIG=${RIG:-rig}
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
TS=${TS:-}
CONFIGS=${CONFIGS:-page 262k|262144|-b 512 -ub 128;b2048 262k|262144|-b 2048 -ub 128;ub512 131k|131072|-b 2048 -ub 512;ub512 64k|65536|-b 2048 -ub 512}
SERVER_ARGS=${SERVER_ARGS:--np 1 -bs --pipe-depth 3 --ctx-checkpoints 4 --cache-ram 0}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
PORT=${PORT:-8095}
STDLIB=${STDLIB:-/usr/lib/python3.12}
OUT=${1:-/dev/null}

{
echo "# Server prompt reading, $(basename "$MODEL"), ${TS:+-ts $TS, }$(basename "$(dirname "$BIN")"), $(date -Is)"
IFS=';' read -ra CFG <<< "$CONFIGS"
for cfg in "${CFG[@]}"; do
  IFS='|' read -r label ctx bargs <<< "$cfg"
  echo "== $label: -c $ctx $bargs $(date +%T)"
  ssh "$RIG" "PORT=$PORT B='$BIN' M='$MODEL' CTX=$ctx BARGS='$bargs' TS='$TS' SARGS='$SERVER_ARGS' GPUS=$GPUS STDLIB='$STDLIB' bash -s" << 'REMOTE' 2>&1 | sed "s/^/$label | /"
pkill -x llama-server; sleep 1
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup $B/llama-server -m $M -ngl 99 -fa on -c $CTX $BARGS ${TS:+-ts $TS} $SARGS \
  --no-webui --host 127.0.0.1 --port $PORT > /tmp/serve_prefill.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  pgrep -x llama-server > /dev/null || break
  sleep 5
done
if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
python3 -I - "$PORT" "$STDLIB" << 'EOF'
import glob, json, sys, time, urllib.request
port, stdlib = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(stdlib + '/*.py'))
def text(chars, skip=0):
    out, n = [], 0
    for f in files[skip:]:
        t = open(f, errors='replace').read()
        out.append('# file ' + f.split('/')[-1] + '\n' + t)
        n += len(t)
        if n >= chars:
            break
    return '\n'.join(out)[:chars]
base = text(56000)
more = text(7000, skip=150)
def ask(content, n):
    body = dict(messages=[{'role': 'user', 'content': content}], max_tokens=n, temperature=0, cache_prompt=True)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r['timings'], time.time() - t0
q = '\n\nIn one sentence: which of these files is the longest?'
t, w = ask(base + q, 16)
print(f"fresh   : {t['prompt_n']:6d} new prompt tokens at {t['prompt_per_second']:7.1f} tok/s, wall {w:6.1f} s")
t, w = ask(base + '\n' + more + q, 16)
print(f"next    : {t['prompt_n']:6d} new prompt tokens at {t['prompt_per_second']:7.1f} tok/s, "
      f"first token after {t['prompt_ms'] / 1000:5.1f} s (cached {t.get('cache_n', '?')})")
t, w = ask(base + '\n' + more + q, 256)
print(f"decode  : {t['predicted_n']:6d} tokens at {t['predicted_per_second']:6.2f} tok/s "
      f"after {t['prompt_n'] + t.get('cache_n', 0)} tokens of context")
EOF
else
  echo 'server did not start:'; grep -E 'out of memory|failed|error' /tmp/serve_prefill.log | head -5
fi
pkill -INT -x llama-server; sleep 5; pkill -9 -x llama-server
rm -f /tmp/serve_prefill.log
REMOTE
done
echo "== done $(date +%T)"
} | tee -a "$OUT"
