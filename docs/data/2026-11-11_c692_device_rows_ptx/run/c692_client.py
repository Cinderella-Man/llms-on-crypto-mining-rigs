#!/usr/bin/env python3
"""c692_client.py: one load's requests for C692 (registry Q1225), on the loopback.

  python3 -I c692_client.py --port 18792 --label day-pre --out <work folder> --long-file <script> \
      [--n-predict 512] [--repeat 2]

Three prompts, each as one user message in the server's chat template:
  code   R57's coding request (an ISO 8601 parser and three pytest tests)
  prose  R57's prose request (a hash map's collisions, one Python example)
  long   a review of --long-file (a Python script of this repository): its prompt is longer than one call of 2,048
         tokens, so the target's rows for the draft's catch-up come from several micro-batches and two calls
The first load renders them with /apply-template and keeps the text in OUT/prompts_rendered.json; every later load,
of either build, sends that same text. Each load: a warm-up, then each prompt --repeat times, greedy (temperature 0,
top-k 1), the end of text ignored, without the prompt cache, so every request is computed from its first token.
Writes OUT/requests_<label>.jsonl, one line per request: the prompt's token ids (md5, from /tokenize), the reply's ids,
its text and the draft counters. No time is recorded.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--long-file', required=True)
ap.add_argument('--n-predict', type=int, default=512)
ap.add_argument('--repeat', type=int, default=2)
A = ap.parse_args()

TEXTS = {
    'code': 'Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest '
            'tests for it.',
    'prose': 'Explain in one paragraph how a hash map handles collisions, then give one example in Python.',
}
LONG = ('Review this Python script. List its bugs and fragile spots, the most serious first, each with the line it is '
        'on and a fix.\n\n```python\n{}\n```')
GREEDY = {'temperature': 0, 'top_k': 1, 'seed': 1}


def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def md5(obj):
    return hashlib.md5((obj if isinstance(obj, str) else json.dumps(obj)).encode()).hexdigest()


def main():
    os.makedirs(A.out, exist_ok=True)
    rendered_path = os.path.join(A.out, 'prompts_rendered.json')
    if os.path.exists(rendered_path):
        with open(rendered_path) as f:
            rendered = json.load(f)
    else:
        with open(A.long_file) as f:
            texts = dict(TEXTS, long=LONG.format(f.read()))
        rendered = {'rendered_by': A.label, 'prompts': {}}
        for name, text in texts.items():
            r = post('/apply-template', {'messages': [{'role': 'user', 'content': text}]})
            rendered['prompts'][name] = {'text': text, 'prompt': r['prompt']}
        with open(rendered_path + '.tmp', 'w') as f:
            json.dump(rendered, f, indent=1)
        os.replace(rendered_path + '.tmp', rendered_path)
    prompts = rendered['prompts']

    log = open(os.path.join(A.out, f'requests_{A.label}.jsonl'), 'w')

    def ask(kind, name, rep, prompt, n):
        tok = post('/tokenize', {'content': prompt, 'add_special': True}).get('tokens') or []
        r = post('/completion', {'prompt': prompt, 'n_predict': n, 'cache_prompt': False, 'ignore_eos': True,
                                 'return_tokens': True, 'stream': False, **GREEDY})
        t = r.get('timings') or {}
        ids = r.get('tokens') or []
        row = {'label': A.label, 'kind': kind, 'prompt': name, 'rep': rep, 'prompt_tokens': len(tok),
               'prompt_md5': md5(tok), 'tokens_evaluated': r.get('tokens_evaluated'), 'n_gen': len(ids),
               'ids_md5': md5(ids) if ids else '', 'text_md5': md5(r.get('content') or ''),
               'draft_n': t.get('draft_n', 0), 'draft_accepted': t.get('draft_n_accepted', 0),
               'ids': ids, 'content': r.get('content') or ''}
        log.write(json.dumps(row) + '\n')
        log.flush()
        return row

    w = ask('warmup', 'warmup', 0, 'Hello. Count from one to ten.', 32)
    print(f"{A.label} warmup: {w['n_gen']} tokens, drafted {w['draft_n']}, accepted {w['draft_accepted']}",
          file=sys.stderr, flush=True)
    for name in ('code', 'prose', 'long'):
        for rep in range(1, A.repeat + 1):
            g = ask('gen', name, rep, prompts[name]['prompt'], A.n_predict)
            print(f"{A.label} {name} {rep}: prompt {g['prompt_tokens']} tokens ({g['prompt_md5'][:8]}), reply "
                  f"{g['n_gen']} tokens, ids {g['ids_md5']}, drafted {g['draft_n']}, accepted {g['draft_accepted']}",
                  file=sys.stderr, flush=True)


try:
    main()
except urllib.error.HTTPError as e:
    sys.exit(f'the server answered {e.code}: {e.read().decode(errors="replace")[:400]}')
except urllib.error.URLError as e:
    sys.exit(f'the server on port {A.port} does not answer: {e.reason}')
