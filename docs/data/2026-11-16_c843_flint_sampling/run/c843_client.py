#!/usr/bin/env python3
"""c843_client.py: C843's client. Sends recorded agent requests (requests.jsonl from c843_pick.py) to one llama-server,
each once per seed, at the recorded request's own sampling (Flint's: temperature 1, top-p 0.95, top-k 20, min-p 0) with
a fixed seed, and writes one JSON line per request: the generated token ids, the reply (reasoning, text, tool calls), how
it ended and its counts. Every request reads its prompt from an empty cache (cache_prompt false), so each one starts from
the same state in every arm and in a repeat. Changes to the recorded body: stream off, max_tokens N, the seed,
cache_prompt false, return_tokens and verbose on (the token ids and the end reason in the reply; neither changes what is
sampled). Requests in order: every request, every seed; then the repeats (LABEL:SEED), sent again.
  python3 -I c843_client.py --port 8297 --arm v3 --requests requests.jsonl --seeds "11 12 13" --repeat "A1:11" \
    --n-predict 2048 --out v3.jsonl
"""
import argparse, hashlib, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8297)
ap.add_argument('--arm', required=True)
ap.add_argument('--requests', required=True)
ap.add_argument('--seeds', default='11 12 13')
ap.add_argument('--repeat', default='')
ap.add_argument('--n-predict', type=int, default=2048)
ap.add_argument('--timeout', type=int, default=900)
ap.add_argument('--out', required=True)
A = ap.parse_args()


def md5(x):
    if not isinstance(x, (str, bytes)):
        x = json.dumps(x, sort_keys=True, ensure_ascii=False)
    if isinstance(x, str):
        x = x.encode()
    return hashlib.md5(x).hexdigest()


reqs = [json.loads(l) for l in open(A.requests) if l.strip()]
by_label = {r['label']: r for r in reqs}
seeds = [int(s) for s in A.seeds.split()]
plan = [(r['label'], s, 0) for r in reqs for s in seeds]
for rp in A.repeat.split():
    lab, s = rp.split(':')
    plan.append((lab, int(s), 1))

url = f'http://127.0.0.1:{A.port}/v1/chat/completions'
with open(A.out, 'a') as out:
    for i, (lab, seed, rep) in enumerate(plan):
        body = dict(by_label[lab]['body'])
        body.pop('stream_options', None)
        body.update({'stream': False, 'max_tokens': A.n_predict, 'seed': seed, 'cache_prompt': False,
                     'return_tokens': True, 'verbose': True})
        data = json.dumps(body).encode()
        t0 = time.time()
        rq = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(rq, timeout=A.timeout) as f:
            res = json.loads(f.read())
        wall = time.time() - t0
        ch = res['choices'][0]
        msg = ch.get('message', {})
        vb = res.get('__verbose', {})
        gs = vb.get('generation_settings', {})
        ids = vb.get('tokens') or []
        prompt = vb.get('prompt', '')
        rec = {
            'arm': A.arm, 'i': i, 'label': lab, 'seed': seed, 'rep': rep, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'wall_s': round(wall, 3), 'finish_reason': ch.get('finish_reason'), 'stop_type': vb.get('stop_type'),
            'stopping_word': vb.get('stopping_word'), 'truncated': vb.get('truncated'),
            'n_ids': len(ids), 'ids_md5': md5(','.join(map(str, ids))) if ids else '', 'ids': ids,
            'tokens_predicted': vb.get('tokens_predicted'), 'tokens_evaluated': vb.get('tokens_evaluated'),
            'tokens_cached': vb.get('tokens_cached'),
            'usage': res.get('usage'), 'timings': res.get('timings'),
            'prompt_md5': md5(prompt) if prompt else '', 'prompt_chars': len(prompt),
            'reasoning': msg.get('reasoning_content') or '', 'content': msg.get('content') or '',
            'tool_calls': msg.get('tool_calls') or [],
            'seed_echo': gs.get('seed'), 'temperature': gs.get('temperature'), 'top_k': gs.get('top_k'),
            'top_p': gs.get('top_p'), 'min_p': gs.get('min_p'), 'samplers': gs.get('samplers'),
            'chat_format': gs.get('chat_format'), 'grammar_lazy': gs.get('grammar_lazy'),
            'backend_sampling': gs.get('backend_sampling'), 'reasoning_format': gs.get('reasoning_format'),
            'grammar_md5': md5(gs.get('grammar') or ''), 'body_md5': md5(data),
        }
        rec['reasoning_md5'] = md5(rec['reasoning'])
        rec['content_md5'] = md5(rec['content'])
        rec['tool_calls_md5'] = md5([{'name': t.get('function', {}).get('name'),
                                      'arguments': t.get('function', {}).get('arguments')} for t in rec['tool_calls']])
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out.flush()
        print(f"{rec['time']} {A.arm} {lab} seed {seed}{' repeat' if rep else ''}: {rec['n_ids']} tokens, "
              f"{rec['finish_reason']}/{rec['stop_type']}, prompt {rec['tokens_evaluated']}, {wall:.1f} s", flush=True)
