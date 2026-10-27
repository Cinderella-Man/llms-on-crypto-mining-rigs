#!/usr/bin/env bash
# Diagnostics before an agent check (used for Ornith-1.5-122B, docs/data/2026-10-27_ornith/): per config one
# llama-server on the rig (-np 1), the memory it takes per card and on the host, the load log's buffer sizes and
# feature lines, then three requests as an agent sees them:
#   fresh  a ~13k-token prompt (Python standard library source, PSF), 16 tokens out: prompt tok/s
#   next   the same plus ~2k new tokens (the cached prefix reused, an agent's next turn): first token time
#   decode the next prompt again, 256 tokens out at the makers' sampling (T 0.6, top-p 0.95, top-k 20, seed 1)
# and two identity prompts at T 0 whose output hash compares builds and switches. Run from a machine that reaches the
# rig over ssh, with nothing else on the rig:
#   RIG=rig MODEL=<rig path of the GGUF> BUILDS=<rig folder holding <build>/bin/llama-server> OUT=serve_probe.md \
#   CONFIGS='label|build|ctx|args|env;...' bash scripts/serve_probe.sh
# ONLY_LOAD=1 skips the requests (memory fit only); QUICK=1 runs only the two identity prompts (decode, e.g. under
# the kernel profiler). Output appended to OUT; each server's log goes to LOGS (default
# logs/ beside OUT). An env field with GGML_CUDA_PROFILE=/tmp/ornith-prof.csv also fetches that kernel profile and
# summarizes it with scripts/prof_summary.py (prof-<label>.txt in LOGS).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RIG=${RIG:-rig}
M=${MODEL:?the GGUF on the rig}
BUILDS=${BUILDS:?folder on the rig holding the builds}
OUT=${OUT:?output file}
LOGS=${LOGS:-$(dirname "$OUT")/logs}
mkdir -p "$LOGS"
PORT=8095
{
echo "# serve_probe $(basename "$M") $(date -Is)"
IFS=';' read -ra CFG <<< "${CONFIGS:?}"
for cfg in "${CFG[@]}"; do
  IFS='|' read -r label build ctx bargs envs <<< "$cfg"
  echo "== $label: $build -c $ctx $bargs ${envs:+[env $envs]} $(date +%T)"
  ssh "$RIG" "pkill -x llama-server; sleep 1; rm -f /tmp/serve_probe.log
    env $envs setsid nohup $BUILDS/$build/bin/llama-server -m $M ${NGL--ngl 99} -fa on -c $ctx $bargs -np 1 \
      --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui --host 127.0.0.1 --port $PORT \
      > /tmp/serve_probe.log 2>&1 < /dev/null &
    srv=\$!  # the launched process itself (env, setsid and nohup exec into llama-server): pgrep could race the execs
    t0=\$(date +%s)
    for i in \$(seq 1 200); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 \$srv 2> /dev/null || break; sleep 3; done
    echo \"load \$((\$(date +%s) - t0)) s\"
    grep -E 'model buffer size|KV buffer size|RS buffer size|compute buffer size|n_layer_nextn|n_ctx *=|n_rs_seq|persistent|MoE layers|compute-6.1|Pascal|pascal|rescale|attention kernel|pipe|MTP|mtp|draft|speculat|graph splits|CUDA graph|out of memory|failed|error|warn' /tmp/serve_probe.log | grep -v '^srv  log_server_r' | grep -v 'unused tensor' | cut -c1-240 | head -120
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
    free -m | sed -n 2p
    curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo 'server did not start'; tail -5 /tmp/serve_probe.log; exit 0; }
    [ -n '${ONLY_LOAD:-}' ] && exit 0
    python3 -I - $PORT '${QUICK:-}' << 'EOF'
import glob, json, sys, time, urllib.request
port = sys.argv[1]
files = sorted(glob.glob('/usr/lib/python3.12/*.py'))
def text(chars, skip=0):
    out, n = [], 0
    for f in files[skip:]:
        t = open(f, errors='replace').read()
        out.append('# file ' + f.split('/')[-1] + '\n' + t)
        n += len(t)
        if n >= chars: break
    return '\n'.join(out)[:chars]
base = text(52000)
more = text(7000, skip=150)
def ask(content, n, **kw):
    body = dict(messages=[{'role': 'user', 'content': content}], max_tokens=n, cache_prompt=True, **kw)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r['timings'], time.time() - t0
q = '\n\nIn one sentence: which of these files is the longest?'
if sys.argv[2:] != ['1']:  # QUICK=1: only the identity prompts
    t, w = ask(base + q, 16, temperature=0)
    print(f\"fresh   : {t['prompt_n']:6d} new prompt tokens at {t['prompt_per_second']:7.1f} tok/s, wall {w:6.1f} s\")
    t, w = ask(base + '\n' + more + q, 16, temperature=0)
    print(f\"next    : {t['prompt_n']:6d} new prompt tokens at {t['prompt_per_second']:7.1f} tok/s, first token after {t['prompt_ms']/1000:5.1f} s (cached {t.get('cache_n', '?')})\")
    t, w = ask(base + '\n' + more + q, 256, temperature=0.6, top_p=0.95, top_k=20, min_p=0, seed=1)
    extra = ''.join(f', {k} {t[k]}' for k in ('draft_n', 'draft_n_accepted') if k in t)
    print(f\"decode  : {t['predicted_n']:6d} tokens at {t['predicted_per_second']:6.2f} tok/s after {t['prompt_n'] + t.get('cache_n', 0)} tokens of context{extra}\")
# identity: two fresh prompts at T 0, 192 tokens each; the hash of reasoning + answer compares builds and switches
import hashlib
src = open(files[3], errors='replace').read()[:3000]
for name, p in (('chat', 'Explain in a short paragraph how a hash map handles collisions, then give a tiny Python example.'),
                ('rewrite', 'Rewrite this code with every function name in camelCase, changing nothing else:\n\n' + src)):
    body = dict(messages=[{'role': 'user', 'content': p}], max_tokens=192, temperature=0, cache_prompt=True)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    m = r['choices'][0]['message']
    s = (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or '')
    t = r['timings']
    extra = ''.join(f', {k} {t[k]}' for k in ('draft_n', 'draft_n_accepted') if k in t)
    print(f\"ident {name:7s}: {hashlib.md5(s.encode()).hexdigest()[:12]} {t['predicted_n']} tokens at {t['predicted_per_second']:6.2f} tok/s{extra} | {s[:70]!r}\")
EOF
    grep -E 'draft acceptance|accept|statistics' /tmp/serve_probe.log | tail -3 | cut -c1-200
    free -m | sed -n 2p" 2>&1 | sed "s/^/$label | /"
  ssh "$RIG" "pkill -INT -x llama-server; sleep 5; pkill -9 -x llama-server; cat /tmp/serve_probe.log" > "$LOGS/probe-$label.log" 2>&1
  if [[ "$envs" == *GGML_CUDA_PROFILE=/tmp/ornith-prof.csv* ]]; then
    # the kernel profile comes back to this machine and leaves the rig
    ssh "$RIG" 'cat /tmp/ornith-prof.csv' > "$LOGS/prof-$label.csv" && ssh "$RIG" 'rm -f /tmp/ornith-prof.csv'
    nice -n 19 python3 -I "$HERE/prof_summary.py" "$LOGS/prof-$label.csv" > "$LOGS/prof-$label.txt" 2>&1
    head -12 "$LOGS/prof-$label.txt" | sed "s/^/$label prof | /"
  fi
done
ssh "$RIG" 'rm -f /tmp/serve_probe.log'
echo "== done $(date +%T)"
} | tee -a "$OUT"
