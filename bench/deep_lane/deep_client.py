#!/usr/bin/env python3
"""deep_client.py: one load's requests of the deep lane, against a llama-server on this machine's loopback.

    python3 -I deep_client.py --port 8097 --arm dense --load dense1 --depths 0,8192,16384 --text RECORDED [RECORDED …]
        --csv OUT/requests-dense1.csv [--jsonl OUT/dense1.jsonl] [--prompts OUT/prompts.json] [--cache 0|1]
        [--sampler-pass 1] [--valid 0|1] [--n-predict 256] [--need-depth N] [--tag REGEX] [--tail-ids 64]
        [--tail-rank 1] [--tail-text TEXT] [--ignore-eos 0|1] [--seed 1] [--pause S]

deep_lane.sh runs it once per load, beside the server, so no prompt crosses a network. Standard library only.

The prompts. --text names recorded agent text, each file one of:
- recorded chat requests: the agent check's proxy.jsonl.bodies.gz (one JSON line per request: id, body, reply; the
  proxy.jsonl beside it gives each request's tag, which --tag filters on), JSON lines of bare request bodies, or one
  JSON document holding a body or a list of them. The requests of one trace (the proxy's tag without its turn
  number; untagged: requests whose messages continue an earlier request's) are one session, and each session counts
  once, as its last request (the longest history), rendered by the server's own chat template (/apply-template,
  with the tools);
- plain text, read as it is.
The sessions and texts, in the order given, are tokenized by the server (/tokenize) into one stream of token ids. Every
prompt ends in the same tail, so that the model writes the same turn at every depth and only the context in front of
it grows: the last --tail-ids ids of one recorded request's rendered prompt (the end of its last message and the
template's start of the assistant's turn). That request is the one whose recorded reply was the longest (--tail-rank K:
the K-th longest), and its session is left out of the stream; with no recorded reply it is the first session's last
request. Plain text alone has no tail unless --tail-text gives one. A prompt of depth D is exactly D ids: what the
tokenizer puts in front of a prompt, the stream's first ids, the tail. Depth 0 is the tail alone (plain text: the
stream's first --tail-ids ids). --prompts keeps the ids in a file, so every load of a run sends the same prompts
without reading the recorded text again; the file is rebuilt when the text, the tail settings or the depth change.

The requests, depths ascending, all to /completion with token ids as the prompt:
- one untimed warm-up (64 ids in, 32 out);
- per depth, with --cache 1 (default): a `prime` request of the prompt without its tail and no output asked (the
  server reads what the cache lacks, a chunk on top of the previous depth's, and its prompt timings are the prompt
  reading at that depth); then the whole prompt twice (`decode`, pass 1 and 2), --n-predict tokens at temperature 0
  with end of text ignored; then with --sampler-pass 1 once more with the server line's own sampler and a fixed seed
  (`sample`, pass s). With --cache 0 no prime, and every request reads its whole prompt fresh (cache_prompt false).
Pass 1 is the first request at a new depth, as every turn of an agent is. --pause S waits S seconds before every
request, so that a profiler's trace (GGML_CUDA_PROFILE) falls into one piece per request at its idle gaps.

One CSV line per request (--csv): arm, load, kind, pass, depth, prompt_tokens (the ids sent), cache_n (reused from
the cache), prompt_n and prompt_ms (read), gen_n and gen_ms (generated), text_md5 (the reply's text), ids_md5 (its
token ids, when the server returns them), draft_n and draft_accepted (the server's draft counters, 0 without a draft),
tokens_per_step, prompt_md5 (the ids sent), cache_src, valid (--valid: 1 when the machine was checked quiet, so the
times count), wall_s, time. A decode step yields one token of its own and the draft tokens it accepts, so
steps = gen_n - draft_accepted and tokens_per_step = gen_n / steps. --jsonl also keeps each request's whole `timings`
and the first 160 characters of the reply.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

COLS = ['arm', 'load', 'kind', 'pass', 'depth', 'prompt_tokens', 'cache_n', 'prompt_n', 'prompt_ms', 'gen_n', 'gen_ms',
        'text_md5', 'ids_md5', 'draft_n', 'draft_accepted', 'tokens_per_step', 'prompt_md5', 'cache_src', 'valid',
        'wall_s', 'time']
TEMPLATE_KEYS = ('messages', 'tools', 'tool_choice', 'parallel_tool_calls', 'chat_template_kwargs')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--load', required=True)
ap.add_argument('--depths', required=True)
ap.add_argument('--need-depth', type=int, default=0)
ap.add_argument('--n-predict', type=int, default=256)
ap.add_argument('--text', nargs='+', required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--tail-ids', type=int, default=64)
ap.add_argument('--tail-rank', type=int, default=1)
ap.add_argument('--tail-text', default='')
ap.add_argument('--cache', type=int, default=1)
ap.add_argument('--sampler-pass', type=int, default=0)
ap.add_argument('--ignore-eos', type=int, default=1)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--valid', type=int, default=1)
ap.add_argument('--prompts', default='')
ap.add_argument('--csv', required=True)
ap.add_argument('--jsonl', default='')
ap.add_argument('--pause', type=float, default=0)
A = ap.parse_args()
DEPTHS = sorted({int(d) for d in A.depths.split(',')})


def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def tok(text, special=False):
    return post('/tokenize', {'content': text, 'add_special': special, 'parse_special': True})['tokens']


def md5(obj):
    return hashlib.md5((obj if isinstance(obj, str) else json.dumps(obj)).encode()).hexdigest()


def opener(path):
    return (gzip.open if path.endswith('.gz') else open)(path, 'rt', errors='replace')


def recorded(path):
    """'lines' or 'json' for a file of recorded chat requests, None for plain text."""
    first = ''
    with opener(path) as f:
        for first in f:
            if first.strip():
                break
    if first.lstrip()[:1] not in ('{', '['):
        return None
    try:
        json.loads(first)
        return 'lines'
    except ValueError:
        pass
    try:
        with opener(path) as f:
            json.load(f)
        return 'json'
    except ValueError:
        return None


def records(path, mode):
    """(body, reply, tag) of each chat request in a recorded file, in file order, one line in memory at a time."""
    tags = {}
    log = path[:-len('.bodies.gz')] if path.endswith('.bodies.gz') else ''
    if log and os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tags[r.get('id')] = r.get('tag') or ''
    if A.tag and not tags:
        sys.exit(f'--tag needs the proxy log beside {path} (the same name without .bodies.gz)')
    with opener(path) as f:
        docs = (json.loads(line) for line in f if line.strip()) if mode == 'lines' else [json.load(f)]
        for doc in docs:
            for d in doc if isinstance(doc, list) else [doc]:
                body = d.get('body', d) if isinstance(d, dict) else None
                if isinstance(body, dict) and body.get('messages'):
                    yield body, d.get('reply') or {}, tags.get(d.get('id'), '')


def reply_len(reply):
    return (len(reply.get('reasoning_content') or '') + len(reply.get('content') or '')
            + sum(len(c.get('arguments') or '') for c in reply.get('tool_calls') or []))


def render(body):
    return post('/apply-template', {k: body[k] for k in TEMPLATE_KEYS if k in body})['prompt']


def chunks(text, size=200_000):
    """The text in pieces of at most `size` characters, each cut after a line end when it has one."""
    while text:
        cut = len(text) if len(text) <= size else text.rfind('\n', 0, size) + 1 or size
        yield text[:cut]
        text = text[cut:]


def build_pack(key, need_depth):
    tag_re = re.compile(A.tag) if A.tag else None
    sessions, traces, order, top, seq = [], {}, [], [], 0
    for path in A.text:
        mode = recorded(path)
        if not mode:
            order.append(('text', path))
            continue
        for body, reply, tag in records(path, mode):
            if tag_re and not tag_re.search(tag):
                continue
            msgs = body['messages']
            if tag:  # one session per trace: the proxy's tag without its turn number
                i = traces.setdefault((path, re.sub(r'/\d+$', '', tag)), len(sessions))
            else:  # untagged: a request whose messages continue an earlier request's is the same session
                i = next((j for j in range(len(sessions) - 1, -1, -1)
                          if msgs[:len(sessions[j]['messages'])] == sessions[j]['messages']), len(sessions))
            if i == len(sessions):
                sessions.append(body)
                order.append(('session', i))
            sessions[i] = body
            seq += 1
            # the --tail-rank longest recorded replies so far, each with its request and session
            top = sorted(top + [(reply_len(reply), -seq, i, body)], key=lambda t: t[:2], reverse=True)[:A.tail_rank]
    tail_session, tail_body, tail_reply = None, None, 0
    if top and top[-1][0] > 0:
        tail_reply, _, tail_session, tail_body = top[-1]
    elif sessions:
        tail_session, tail_body = 0, sessions[0]
    bos = tok('', True)
    if tail_body is not None:
        tail = tok(render(tail_body))[-A.tail_ids:]
    else:
        tail = tok(A.tail_text) if A.tail_text else []
    need = max(need_depth - len(bos) - len(tail), A.tail_ids)
    stream, used = [], 0
    for kind, ref in order:
        if len(stream) >= need:
            break
        if kind == 'session' and ref == tail_session:
            continue
        if kind == 'text':
            with opener(ref) as f:
                text = f.read()
        else:
            text = render(sessions[ref])
        used += 1
        for piece in chunks(text):
            stream += tok(piece)
            if len(stream) >= need:
                break
    n_text = sum(1 for kind, _ in order if kind == 'text')
    if len(stream) < need:
        sys.exit(f'only {len(stream)} tokens in front of the {len(tail)}-token tail from {len(sessions)} recorded '
                 f'sessions and {n_text} text files, need {need} for depth {need_depth}: pass more recorded text')
    meta = {'sessions': len(sessions), 'text_files': n_text, 'pieces_used': used, 'bos_ids': len(bos),
            'tail_ids': len(tail), 'tail_session': tail_session, 'tail_reply_chars': tail_reply, 'need_depth': need_depth}
    return {'key': key, 'meta': meta, 'bos': bos, 'tail': tail, 'stream': stream[:need]}


def prompts():
    """The token ids of the run: read from --prompts when it was built from the same text for at least this depth."""
    need_depth = max(A.need_depth, DEPTHS[-1])
    key = {'text': [[os.path.basename(p), os.path.getsize(p)] for p in A.text], 'tag': A.tag, 'tail_ids': A.tail_ids,
           'tail_rank': A.tail_rank, 'tail_text': A.tail_text}
    if A.prompts and os.path.exists(A.prompts):
        with open(A.prompts) as f:
            pack = json.load(f)
        if pack['key'] == key and pack['meta']['need_depth'] >= need_depth:
            return pack
    pack = build_pack(key, need_depth)
    if A.prompts:
        with open(A.prompts + '.tmp', 'w') as f:
            json.dump(pack, f)
        os.replace(A.prompts + '.tmp', A.prompts)
    return pack


def main():
    pack = prompts()
    bos, tail, stream = pack['bos'], pack['tail'], pack['stream']

    def cut(depth):  # (the prompt of that depth, the prompt without its tail)
        base = bos + stream[:max(depth - len(bos) - len(tail), 0)]
        return (base + tail) or stream[:A.tail_ids], base

    out = open(A.csv, 'w', newline='')
    table = csv.DictWriter(out, COLS)
    table.writeheader()
    log = open(A.jsonl, 'w') if A.jsonl else None
    if log:
        log.write(json.dumps({'kind': 'meta', 'arm': A.arm, 'load': A.load, 'depths': DEPTHS, 'n_predict': A.n_predict,
                              'cache': A.cache, 'valid': A.valid, 'text': [os.path.basename(p) for p in A.text],
                              'tag': A.tag, **pack['meta'], 'prompt_ids': {d: len(cut(d)[0]) for d in DEPTHS},
                              'prompt_md5': {d: md5(cut(d)[0]) for d in DEPTHS},
                              'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}) + '\n')

    def ask(kind, pas, depth, prompt, n, sampling):
        if A.pause > 0:
            time.sleep(A.pause)  # an idle gap before every request, so that a profiler's trace cuts into requests
        t0 = time.time()
        r = post('/completion', {'prompt': prompt, 'n_predict': n, 'cache_prompt': bool(A.cache) and kind != 'warmup',
                                 'ignore_eos': bool(A.ignore_eos), 'return_tokens': True, 'stream': False, **sampling})
        wall = time.time() - t0
        t = r.get('timings') or {}
        ids, text = r.get('tokens') or [], r.get('content') or ''
        gen_n, accepted = t.get('predicted_n', 0), t.get('draft_n_accepted', 0)
        timed = kind in ('decode', 'sample')
        row = {'arm': A.arm, 'load': A.load, 'kind': kind, 'pass': pas, 'depth': depth, 'prompt_tokens': len(prompt),
               'cache_n': t.get('cache_n', ''), 'prompt_n': t.get('prompt_n', ''),
               'prompt_ms': round(t.get('prompt_ms', 0), 3), 'gen_n': gen_n, 'gen_ms': round(t.get('predicted_ms', 0), 3),
               'text_md5': md5(text), 'ids_md5': md5(ids) if ids else '', 'draft_n': t.get('draft_n', 0),
               'draft_accepted': accepted,
               'tokens_per_step': f'{gen_n / (gen_n - accepted):.3f}' if timed and gen_n > accepted else '',
               'prompt_md5': md5(prompt), 'cache_src': t.get('cache_src', ''), 'valid': A.valid,
               'wall_s': round(wall, 3), 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        table.writerow(row)
        out.flush()
        if log:
            log.write(json.dumps({**row, 'stop_type': r.get('stop_type'), 'content_head': text[:160], 'timings': t})
                      + '\n')
            log.flush()
        line = (f"{A.load} {kind:6s} depth {depth:6d} pass {pas}: {len(prompt):6d} ids, {row['cache_n'] or 0} cached, "
                f"{row['prompt_n'] or 0} read | {gen_n} out")
        if A.valid and t.get('prompt_ms') and t.get('predicted_ms'):
            line += (f" | read {t['prompt_n'] * 1000 / t['prompt_ms']:.1f} tok/s, "
                     f"out {gen_n * 1000 / t['predicted_ms']:.2f} tok/s")
        if timed and t.get('draft_n'):
            line += f" | draft {accepted} of {t['draft_n']} accepted, {row['tokens_per_step']} tokens a step"
        print(line, file=sys.stderr, flush=True)

    greedy = {'temperature': 0, 'top_k': 1, 'seed': A.seed}
    ask('warmup', 0, 0, stream[:64], 32, greedy)
    for i, d in enumerate(DEPTHS):
        full, base = cut(d)
        if A.cache and len(base) > len(bos):
            ask('prime', 0, d, base, 0, greedy)
        for pas in (1, 2):
            ask('decode', pas, d, full, A.n_predict, greedy)
        if A.sampler_pass:
            ask('sample', 's', d, full, A.n_predict, {'seed': A.seed + 1000 + i})


try:
    main()
except urllib.error.HTTPError as e:
    sys.exit(f'the server answered {e.code}: {e.read().decode(errors="replace")[:400]}')
except urllib.error.URLError as e:
    sys.exit(f'the server on port {A.port} does not answer: {e.reason}')
