#!/usr/bin/env python3
# depth_client.py: decode speed at fixed context depths, as llama-server reports it. Runs on the rig next to the server
# (run/depth_sweep.sh pipes it over ssh), standard library only:
#   python3 -I depth_client.py <port> <n_predict> <stdlib folder> <depths, comma-separated> <load name>
# Each prompt is one chat turn in the model's own template (the server's /apply-template): the user message is source
# code followed by QUESTION, and the model answers in its reasoning, as an agent's turn does. The code is the source of
# Python's standard library (the sorted *.py files of the folder, PSF licence), tokenized once by the server, cut so
# that the whole prompt is exactly <depth> tokens (the template and the question alone, ~30 tokens, for depth 0): the
# template's head, the first ids of the code, the template's tail. Every load of every build reads the same ids (their
# md5 per depth is in the meta line). Per load: one untimed warm-up (64 ids in, 32 out), then per depth the same request
# twice (pass 1 and 2), each read fresh (cache_prompt false) with <n_predict> greedy tokens out, end of text ignored.
# One JSON line per request on stdout (the server's timings, the request's wall time, the md5 of the generated ids),
# one readable line per request on stderr.
import glob, hashlib, json, sys, time, urllib.request

port, n_predict, stdlib, depths, load = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
depths = [int(d) for d in depths.split(',')]
QUESTION = 'Explain what the code above does, file by file.'
MARK = '@@CODE@@'


def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def tok(text, special):
    return post('/tokenize', {'content': text, 'add_special': special})['tokens']


def emit(rec):
    print(json.dumps(rec), flush=True)


need = max(depths)
files = sorted(glob.glob(stdlib + '/*.py'))
parts, used, code = [], 0, []
for f in files:
    parts.append('# file ' + f.split('/')[-1] + '\n' + open(f, errors='replace').read())
    used += 1
    if sum(map(len, parts)) < need * 4:
        continue
    code = tok('\n'.join(parts), False)
    if len(code) >= need:
        break
if len(code) < need:
    sys.exit(f'only {len(code)} tokens from {stdlib}, need {need}')
text = post('/apply-template', {'messages': [{'role': 'user', 'content': MARK + '\n\n' + QUESTION}]})['prompt']
head, tail = text.split(MARK)
head, tail = tok(head, True), tok(tail, False)
prompts = {d: head + code[:max(d - len(head) - len(tail), 0)] + tail for d in depths}
emit({'kind': 'meta', 'load': load, 'files': used, 'question': QUESTION, 'template': text, 'head_ids': len(head),
      'tail_ids': len(tail), 'prompt_ids': {d: len(p) for d, p in prompts.items()},
      'ids_md5': {d: hashlib.md5(json.dumps(p).encode()).hexdigest() for d, p in prompts.items()},
      'n_predict': n_predict, 'depths': depths, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})

GREEDY = {'temperature': 0, 'top_k': 1, 'seed': 1, 'ignore_eos': True, 'return_tokens': True, 'stream': False}


def ask(kind, depth, pas, prompt, n):
    t0 = time.time()
    r = post('/completion', {'prompt': prompt, 'n_predict': n, 'cache_prompt': False, **GREEDY})
    wall = time.time() - t0
    t = r.get('timings', {})
    out = r.get('tokens') or []
    emit({'kind': kind, 'load': load, 'depth': depth, 'pass': pas, 'prompt_tokens': len(prompt), 'wall_s': round(wall, 3),
          'gen_md5': hashlib.md5(json.dumps(out).encode()).hexdigest(), 'n_out': len(out),
          'stop_type': r.get('stop_type'), 'content_head': r.get('content', '')[:160],
          'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'timings': t})
    print(f"{load} {kind} depth {depth:6d} pass {pas}: prompt {t.get('prompt_n', 0):6d} at "
          f"{t.get('prompt_per_second', 0):7.1f} tok/s | {t.get('predicted_n', 0)} out at "
          f"{t.get('predicted_per_second', 0):6.2f} tok/s | wall {wall:6.1f} s", file=sys.stderr, flush=True)


ask('warmup', 64, 0, (head + code)[:64], 32)
for d in depths:
    for pas in (1, 2):
        ask('decode', d, pas, prompts[d], n_predict)
