#!/usr/bin/env bash
# Attention's share of a decoded token at long context (registry row Q166's R4): one llama-server under
# GGML_CUDA_PROFILE reads a long fresh prompt (CPython standard-library source, CHARS characters, ~40k tokens at the
# default), then decodes 128 tokens after it at temperature 0. The profile's decode-sized evaluations are then the
# steps at that depth; scripts/prof_summary.py gives FLASH_ATTN_EXT's time per token, to compare with a short-context
# profile of the same build and file. Run from a machine that reaches the rig over ssh, with nothing else on the rig:
#   RIG=rig MODEL=<rig path of the GGUF> BUILDS=<rig builds folder> BUILD=<build folder> TS=<split> OUT=<local folder> \
#   bash run/attn_depth.sh
# Output: OUT/attn_depth.log (load, prompt and decode timings) and OUT/attn_depth_prof.csv (the profiler's trace; then
# python3 -I scripts/prof_summary.py OUT/attn_depth_prof.csv).
set -uo pipefail
RIG=${RIG:-rig}
M=${MODEL:?the GGUF on the rig}
BUILDS=${BUILDS:?builds folder on the rig}
BUILD=${BUILD:?build folder}
TS=${TS:?-ts split}
OUT=${OUT:?local output folder}
CTX=${CTX:-49152}
CHARS=${CHARS:-150000}
PORT=8095
mkdir -p "$OUT"
{
echo "# attn_depth $(basename "$M") $BUILD -c $CTX -ts $TS, $CHARS characters, $(date -Is)"
ssh "$RIG" "pkill -x llama-server; sleep 1; rm -f /tmp/attn_depth.log /tmp/attn-prof.csv
  GGML_CUDA_PROFILE=/tmp/attn-prof.csv setsid nohup $BUILDS/$BUILD/bin/llama-server -m $M -ngl 99 -fa on -c $CTX \
    -b 2048 -ub 512 -ts $TS -bs -fit off -np 1 --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
    --no-webui --host 127.0.0.1 --port $PORT > /tmp/attn_depth.log 2>&1 < /dev/null &
  srv=\$!
  t0=\$(date +%s)
  for i in \$(seq 1 200); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 \$srv 2> /dev/null || break; sleep 3; done
  echo \"load \$((\$(date +%s) - t0)) s\"
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo 'server did not start'; tail -5 /tmp/attn_depth.log; exit 0; }
  python3 -I - $PORT $CHARS << 'EOF'
import glob, json, sys, time, urllib.request
port, chars = sys.argv[1], int(sys.argv[2])
out, n = [], 0
for f in sorted(glob.glob('/usr/lib/python3.12/*.py')):
    t = open(f, errors='replace').read()
    out.append('# file ' + f.split('/')[-1] + '\n' + t)
    n += len(t)
    if n >= chars: break
p = '\n'.join(out)[:chars] + '\n\nIn one paragraph: what do the first and the last of these files have in common?'
def ask(n_out):
    body = dict(messages=[{'role': 'user', 'content': p}], max_tokens=n_out, temperature=0, cache_prompt=True)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    return r['timings'], time.time() - t0
t, w = ask(1)
print(f\"prompt : {t['prompt_n']:6d} tokens at {t['prompt_per_second']:6.1f} tok/s, wall {w:6.1f} s\")
t, w = ask(128)
print(f\"decode : {t['predicted_n']:6d} tokens at {t['predicted_per_second']:6.2f} tok/s after {t['prompt_n'] + t.get('cache_n', 0)} tokens of context\")
EOF
  pkill -INT -x llama-server; sleep 5; pkill -9 -x llama-server
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo" 2>&1
ssh "$RIG" 'cat /tmp/attn-prof.csv' > "$OUT/attn_depth_prof.csv" && ssh "$RIG" 'rm -f /tmp/attn-prof.csv /tmp/attn_depth.log'
echo "== done $(date +%T)"
} | tee -a "$OUT/attn_depth.log"
