#!/usr/bin/env bash
# selftest.sh WORK: the deep lane end to end against bench/agent_check/mock_server.py: no model, no GPU, under a
# minute of one CPU core. It runs deep_lane.sh on the real release preset (release/flash-next-v3/settings.env) with the
# mock as the server, on recorded text it writes itself, and checks the results. nvidia-smi and ps are stand-ins in
# WORK/bin, so the quiet checks see an idle machine (or a busy one, FAKE_BUSY=1) whatever runs on this one.
# Runs, each in WORK/<name> with its console in WORK/<name>.console:
#   rig     five arms (the preset as it is, the same line again, a faster line, the draft off, a 131,072-token context)
#           × 2 loads after a warm-up, the sampler pass, recorded requests filtered by tag, quiet checks on
#   resume  the same command again: every load is skipped and the tables are rebuilt
#   other   the same folder with another setting: refused
#   busy    a llama process is running: refused before any load
#   ptx     quiet checks off, a RUNNER and a STOP_CMD, three equal cards, -otd dropped, the keep-alive variable dropped
#           and a CUDA_CACHE_PATH added
#   fresh   CACHE=0 on plain text with a TAIL_TEXT
#   short   a depth the recorded text cannot fill: stops in the warm-up load
#   ctx     a depth the context cannot hold: refused before any load
#   table   deep_table.py alone on a hand-made requests.csv with a gain, a negligible and a loss
# Uses port DEEP_TEST_PORT (default 18097) on 127.0.0.1. Prints one line per check and exits non-zero if one fails.
set -uo pipefail
WORK=$(realpath -m "${1:?usage: selftest.sh WORK}")
HERE=$(cd "$(dirname "$0")" && pwd)
MOCK=$HERE/../agent_check/mock_server.py
PORT=${DEEP_TEST_PORT:-18097}
rm -rf "$WORK"
mkdir -p "$WORK/bin" "$WORK/text" "$WORK/models"
. "$HERE/../../release/flash-next-v3/settings.env"
touch "$WORK/models/$PRESET_MODEL" "$WORK/models/$PRESET_DRAFT"

cat > "$WORK/bin/nvidia-smi" << 'EOF'
#!/usr/bin/env bash
# three idle cards for any query; a looping query prints once and stays
for i in 0 1 2; do echo "$i, 4, 31, 1911"; done
case " $* " in *" -l "*) exec sleep 3600 ;; esac
EOF
cat > "$WORK/bin/ps" << 'EOF'
#!/usr/bin/env bash
echo bash
[ "${FAKE_BUSY:-0}" = 1 ] && echo llama-server
exit 0
EOF
chmod +x "$WORK/bin/nvidia-smi" "$WORK/bin/ps"

# Recorded text: 12 sessions of two arms as the agent check's proxy writes them (4 requests each, the history
# growing), and a plain text of agent turns
python3 -I - "$WORK/text" << 'EOF'
import gzip, json, random, sys
out = sys.argv[1]
rng = random.Random(7)
words = ('read write test file list error assert function return value module import class line patch run fix bug '
         'expected actual output input parse token cache depth draft step').split()
def text(n):
    return ' '.join(rng.choice(words) for _ in range(n))
tools = [{'type': 'function', 'function': {'name': n, 'parameters': {'type': 'object'}}} for n in ('bash', 'read', 'write')]
rows, bodies, k = [], [], 0
for arm in ('armA', 'armB'):
    for task in range(6):
        msgs = [{'role': 'system', 'content': 'You are a coding agent. ' + text(40)},
                {'role': 'user', 'content': f'Task {arm} {task}: ' + text(60)}]
        for turn in range(1, 5):
            k += 1
            reply = {'reasoning_content': text(20 + 10 * turn + task), 'content': '',
                     'tool_calls': [{'name': 'bash', 'arguments': json.dumps({'command': text(6)})}]}
            rows.append({'id': f'r{k:04d}', 'tag': f'{arm}/task-{task}/a1/t1/{turn}'})
            bodies.append({'id': f'r{k:04d}', 'body': {'messages': list(msgs), 'tools': tools, 'stream': True}, 'reply': reply})
            msgs.append({'role': 'assistant', 'content': None, 'reasoning_content': reply['reasoning_content'],
                         'tool_calls': [{'id': f'c{k}', 'type': 'function', 'function': reply['tool_calls'][0]}]})
            msgs.append({'role': 'tool', 'tool_call_id': f'c{k}', 'content': text(150)})
with open(out + '/proxy.jsonl', 'w') as f:
    f.writelines(json.dumps(r) + '\n' for r in rows)
with gzip.open(out + '/proxy.jsonl.bodies.gz', 'wt') as f:
    f.writelines(json.dumps(b) + '\n' for b in bodies)
with open(out + '/agent.txt', 'w') as f:
    for s in range(30):
        f.write('### user\n' + text(80) + '\n### assistant\n' + text(120) + '\n### tool\n' + text(200) + '\n\n')
EOF

run() {  # run NAME [VAR=value ...]: deep_lane.sh with the mock as the server; exit code in WORK/NAME.rc
  local name=$1
  shift
  env PATH="$WORK/bin:$PATH" BIN="$MOCK" MODELS="$WORK/models" OUT="$WORK/$name" PORT="$PORT" WATCHDOG=off \
    TEXT="$WORK/text/proxy.jsonl.bodies.gz $WORK/text/agent.txt" TAG='^armA/' DEPTHS="0 512 1024 2048" N_PREDICT=64 \
    ARGS_SET="--prefill-ms 0.002 --decode-ms 0.2 --depth-ms 0.1" QUIET_WAIT_S=0 "$@" \
    bash "$HERE/deep_lane.sh" > "$WORK/$name.console" 2>&1
  echo $? > "$WORK/$name.rc"
}
RIG_ARMS=(ARMS="base same fast nodraft big" SAMPLER_PASS=1 ARM_ENV_same=GGML_CUDA_FATTN_SPARSE_PASCAL=0
  ARM_ENV_fast=GGML_CUDA_FATTN_SPARSE_PASCAL=1 "ARM_ARGS_fast=--depth-ms 0.0" ARM_DRAFT_nodraft=off
  ARM_CTX_big=131072 ARM_UB_big=256)
run rig "${RIG_ARMS[@]}"
cp "$WORK/rig/requests.csv" "$WORK/rig-first.csv"
run resume OUT="$WORK/rig" "${RIG_ARMS[@]}"
run other OUT="$WORK/rig" "${RIG_ARMS[@]}" N_PREDICT=32
run busy FAKE_BUSY=1
run ptx FAKE_BUSY=1 QUIET=0 RUNNER="nice -n 19" STOP_CMD="touch $WORK/ptx-stopped" TS=1,1,1 ARGS_DROP="-otd" \
  ENV_DROP=GGML_CUDA_KEEPALIVE ENV_ADD=CUDA_CACHE_PATH=$WORK/jit ARMS="dense sparse" REPS=1 \
  ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0 ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1
run fresh TEXT="$WORK/text/agent.txt" TAG= TAIL_TEXT=$'\n### assistant\n' CACHE=0 REPS=1 WARM=0
run short DEPTHS="0 60000"
run ctx DEPTHS="0 65536"

# deep_table.py alone: control c (20.0 and 20.2 tok/s at depth 0, 15.0 and 15.1 at 8,192, 12.0 and 12.1 at 16,384)
# against x: 20.1 and 20.3 (inside the spread), 16.0 and 16.2 (above it), 11.0 and 11.3 (below it)
mkdir -p "$WORK/table"
python3 -I - "$WORK/table/requests.csv" << 'EOF'
import csv, sys
cols = ['arm', 'load', 'kind', 'pass', 'depth', 'prompt_tokens', 'cache_n', 'prompt_n', 'prompt_ms', 'gen_n', 'gen_ms',
        'text_md5', 'ids_md5', 'draft_n', 'draft_accepted', 'tokens_per_step', 'prompt_md5', 'cache_src', 'valid',
        'wall_s', 'time']
speeds = {'c1': (20.0, 15.0, 12.0), 'c2': (20.2, 15.1, 12.1), 'x1': (20.1, 16.0, 11.0), 'x2': (20.3, 16.2, 11.3)}
w = csv.DictWriter(open(sys.argv[1], 'w', newline=''), cols)
w.writeheader()
for load in ('c1', 'x1', 'c2', 'x2'):
    for depth, s in zip((0, 8192, 16384), speeds[load]):
        for p in (1, 2):
            w.writerow(dict.fromkeys(cols, '') | {
                'arm': load[0], 'load': load, 'kind': 'decode', 'pass': p, 'depth': depth, 'prompt_tokens': max(depth, 64),
                'cache_n': 0, 'prompt_n': max(depth, 64), 'prompt_ms': max(depth, 64) * 10, 'gen_n': 256,
                'gen_ms': 256000 / s, 'text_md5': f't{depth}', 'draft_n': 0, 'draft_accepted': 0, 'tokens_per_step': '1.000',
                'prompt_md5': f'p{depth}', 'valid': 1})
EOF
python3 -I "$HERE/deep_table.py" "$WORK/table" --out "$WORK/table/table.md" --tsv "$WORK/table/classes.tsv" > /dev/null
echo $? > "$WORK/table.rc"

python3 -I - "$WORK" << 'EOF'
import csv, json, sys
from pathlib import Path
w = Path(sys.argv[1])
fails = []
def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what)
    if not ok:
        fails.append(what)
def text(name):
    p = w / name
    return p.read_text() if p.exists() else ''
def table(name, tab=False):
    p = w / name
    return list(csv.DictReader(p.open(newline=''), delimiter='\t' if tab else ',')) if p.exists() else []
def classes(name, metric='generation'):
    return {(r['arm'], r['depth']): r['class'] for r in table(name, True) if r['metric'] == metric}
def line(run, load):  # the server line deep_lane.sh logged, and what the mock printed
    return text(f'{run}/server-{load}.log')

# rig
rows = table('rig/requests.csv')
timed = [r for r in rows if r['load'] != 'warm']
dec = [r for r in timed if r['kind'] == 'decode']
arms = ('base', 'same', 'fast', 'nodraft', 'big')
check(text('rig.rc').strip() == '0' and 'DEEP_DONE' in text('rig.console') and (w / 'rig/done').exists(), 'rig: the run ends with DEEP_DONE')
check(len(rows) == 10 * 16 + 4, f'rig: 16 requests in each of 10 timed loads and 4 in the warm-up load ({len(rows)})')
check([r['load'] for r in table('rig/loads.tsv', True)]
      == ['warm'] + [a + str(i) for i in (1, 2) for a in arms], 'rig: a warm-up load, then the arms alternated twice')
check(all(int(r['prompt_tokens']) == max(int(r['depth']), 64) for r in dec), 'rig: every prompt is exactly its depth (depth 0: the 64-id tail)')
check(all(r['gen_n'] == '64' for r in dec) and all(r['valid'] == '1' for r in rows), 'rig: 64 tokens out on every timed request, times valid')
check(len({r['prompt_md5'] for r in dec if r['depth'] == '2048'}) == 1, 'rig: the same prompt ids on every load')
meta = json.loads(text('rig/base1.jsonl').splitlines()[0])
check(meta['sessions'] == 6 and meta['tail_ids'] == 64, f'rig: the tag keeps one arm\'s 6 sessions ({meta["sessions"]})')
check('-ts 6,6,6,6,5,6,5,6,3' in line('rig', 'base1') and ' -md ' in line('rig', 'base1')
      and '--spec-type draft-mtp' in line('rig', 'base1') and '-c 65536' in line('rig', 'base1'), 'rig: the preset\'s line as it is')
nd = line('rig', 'nodraft1').splitlines()[0]
check(' -md ' not in nd and '--spec-' not in nd and '-otd' not in nd and '-ub 512' in nd, 'rig: the draft-off arm has no -md, --spec-* or -otd')
check('-c 131072' in line('rig', 'big1') and '-ub 256' in line('rig', 'big1'), 'rig: the context arm has -c 131072 -ub 256')
env = [l for l in line('rig', 'fast1').splitlines() if l.startswith('environment:')]
check(bool(env) and 'GGML_CUDA_FATTN_SPARSE_PASCAL=1' in env[0] and 'GGML_CUDA_KEEPALIVE=decode' in env[0], 'rig: the arm\'s variable and the preset\'s reach the server')
c = classes('rig/classes.tsv')
depths = ('0', '512', '1024', '2048')
check(all(c.get(('same', d)) == 'negligible' and c.get(('big', d)) == 'negligible' for d in depths), 'rig: an arm as fast as the control is negligible at every depth')
check(all(c.get(('fast', d)) == 'gain' for d in depths), 'rig: the faster arm is a gain at every depth')
check(all(c.get(('nodraft', d)) == 'loss' for d in depths), 'rig: the arm without the draft is a loss at every depth')
check(all(int(r['draft_n']) > 0 and float(r['tokens_per_step']) > 1 for r in dec if r['arm'] == 'base')
      and all(r['draft_n'] == '0' and r['tokens_per_step'] == '1.000' for r in dec if r['arm'] == 'nodraft'), 'rig: draft counters and tokens a step with the draft, none without')
check(text('rig/table.md').count('the same on every pass and load of every arm.') == 4, 'rig: the greedy reply is the same in every arm at every depth')
pr = {(r['load'], r['depth']): r for r in timed if r['kind'] == 'prime'}
p1 = {(r['load'], r['depth']): r for r in dec if r['pass'] == '1'}
p2 = {(r['load'], r['depth']): r for r in dec if r['pass'] == '2'}
check(all((pr[(l, '1024')]['cache_n'], pr[(l, '1024')]['prompt_n'], pr[(l, '1024')]['cache_src']) == ('444', '516', 'checkpoint')
          for l in {r['load'] for r in timed}), 'rig: a prime reads only the chunk past the previous depth\'s checkpoint (516 on 444)')
check(all(p1[k]['cache_n'] == str(int(k[1]) - 68) and p2[k]['cache_n'] == str(int(k[1]) - 4) for k in p1 if k[1] != '0'),
      'rig: pass 1 reads the tail on the prime\'s checkpoint, pass 2 the last 4 tokens')
sm = [r for r in timed if r['kind'] == 'sample']
check(len(sm) == 40 and all(r['ids_md5'] != p1[(r['load'], r['depth'])]['ids_md5'] for r in sm), 'rig: the sampler pass writes another reply than the greedy passes')
check(text('rig/quiet_base1.txt').count('no llama process') == 1 and (w / 'rig/fit_base1.txt').exists(), 'rig: the quiet check and the loaded cards are written per load')
# resume, other settings
check(text('resume.console').count('skipped') == 10 and 'LOAD warm' not in text('resume.console')
      and 'DEEP_DONE' in text('resume.console') and text('rig-first.csv') == text('rig/requests.csv'), 'resume: every load skipped, the same requests.csv')
check(text('other.rc').strip() == '1' and 'other settings' in text('other.console') and (w / 'rig/done').exists(), 'other: a folder of another run is refused and left as it is')
# busy
check(text('busy.rc').strip() == '1' and 'DEEP_FAILED: the machine is not idle' in text('busy.console')
      and not list((w / 'busy').glob('server-*')), 'busy: refused before any load')
# ptx
rows = table('ptx/requests.csv')
check(text('ptx.rc').strip() == '0' and len(rows) == 2 * 12 + 3 and all(r['valid'] == '0' for r in rows), 'ptx: done, every request marked valid=0')
t = text('ptx/table.md')
check('not measurements' in t and 'tok/s' not in t and 'ms a step' not in t and 'a step' in t and table('ptx/classes.tsv', True) == [],
      'ptx: no speed in the tables, the draft counters kept')
l = line('ptx', 'sparse1')
check('-ts 1,1,1' in l and '-otd' not in l and 'nice -n 19 env' in l and (w / 'ptx-stopped').exists(), 'ptx: three equal cards, -otd dropped, the RUNNER and the STOP_CMD used')
env = [x for x in l.splitlines() if x.startswith('environment:')]
check(bool(env) and 'CUDA_CACHE_PATH=' in env[0] and 'KEEPALIVE' not in env[0] and 'GGML_CUDA_FATTN_SPARSE_PASCAL=1' in env[0],
      'ptx: a CUDA_CACHE_PATH added, the keep-alive variable dropped')
# fresh
rows = [r for r in table('fresh/requests.csv') if r['kind'] != 'warmup']
check(text('fresh.rc').strip() == '0' and len(rows) == 8 and all(r['kind'] == 'decode' and r['cache_n'] == '0'
      and r['prompt_n'] == r['prompt_tokens'] for r in rows), 'fresh: no prime, every request reads its whole prompt')
check([r['prompt_tokens'] for r in rows if r['pass'] == '1'] == ['4', '512', '1024', '2048'], 'fresh: plain text with a tail text, depth 0 the tail alone')
# short, ctx
check(text('short.rc').strip() == '1' and 'need 59936 for depth 60000' in text('short.console')
      and not (w / 'short/requests-base1.csv').exists(), 'short: too little text stops the run in the warm-up load')
check(text('ctx.rc').strip() == '1' and 'does not fit its context of 65536' in text('ctx.console') and not (w / 'ctx/loads.tsv').exists(),
      'ctx: a depth beyond the context is refused before any load')
# table
c = classes('table/classes.tsv')
check(text('table.rc').strip() == '0' and (c.get(('x', '0')), c.get(('x', '8192')), c.get(('x', '16384'))) == ('negligible', 'gain', 'loss'),
      f'table: inside the spread negligible, above it gain, below it loss ({c})')
r = [x for x in table('table/classes.tsv', True) if x['metric'] == 'generation' and x['depth'] == '8192'][0]
check((r['spread'], r['delta'], r['percent']) == ('0.20', '1.05', '7.0'), f'table: spread 0.20, +1.05 tok/s, +7.0 % at 8,192 ({r["spread"]}, {r["delta"]}, {r["percent"]})')
print(f'{len(fails)} of the checks failed' if fails else 'selftest passed')
sys.exit(1 if fails else 0)
EOF
