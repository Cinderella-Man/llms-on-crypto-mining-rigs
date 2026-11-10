#!/usr/bin/env python3
"""exact_client.py: one load's requests for the draft line's exactness check (registry Q1175), on the loopback.

  python3 exact_client.py --port 8097 --prompts prompts.json --arm nodraft --out <folder> \
      [--ref-csv requests.csv] [--score replies_draft.json] [--stop-file <path>]

--prompts is the prompts.json bench/deep_lane/deep_client.py wrote (`bos`, `stream`, `tail`): a prompt of depth D is
bos + stream[:D - len(bos) - len(tail)] + tail, depth 0 the tail alone.

Phase 1, every arm (the replies), in deep_client.py's request order so that a reply can be checked against that tool's
hashes (--ref-csv: its requests.csv, the ids_md5 of this arm's first decode pass per depth): a warm-up, then per depth
  prime  the prompt without its tail, no output asked
  gen    the whole prompt, --n-predict tokens at temperature 0, end of text ignored
  genp   the same once more with the --n-probs most probable tokens of every position (the line's own view of each
         choice while it generates)
  probe  (with --score) where this arm's reply and the other arm's part: the prompt plus the common tokens, one token
         asked, its most probable tokens
Writes replies_<arm>.json after each depth.

Phase 2, with --score <replies_<other arm>.json>, meant for the line without the draft: teacher forcing, per depth
  prime  as above, before each reply
  tf     for every token i of a reply: prompt + reply[:i] as the prompt with the prompt cache on, one token asked: the
         line's first choice at that position and the lead the reply's own token loses by. This arm's reply from token
         0; the other arm's from the first differing token (before it the prompts are the same).
One JSON line per request in requests_<arm>.jsonl (token ids, the most probable tokens with their log-probabilities,
the server's counters); scored_<arm>.json has the teacher-forced positions. A file at --stop-file ends phase 2 after
the request in flight. Token ids and reply text are derived from the recorded text: keep these files with it.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--prompts', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--depths', default='0,8192,16384,32768,49152')
ap.add_argument('--n-predict', type=int, default=256)
ap.add_argument('--n-probs', type=int, default=10)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--ref-csv', default='')
ap.add_argument('--ref-arm', default='')
ap.add_argument('--score', default='')
ap.add_argument('--stop-file', default='')
A = ap.parse_args()
DEPTHS = sorted({int(d) for d in A.depths.split(',')})
GREEDY = {'temperature': 0, 'top_k': 1, 'seed': A.seed}


def post(path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def md5(obj):
    return hashlib.md5((obj if isinstance(obj, str) else json.dumps(obj)).encode()).hexdigest()


def logp(e):
    if e.get('logprob') is not None:
        return e['logprob']
    p = e.get('prob')
    return math.log(p) if p else None


def compact(probs):
    """completion_probabilities as [{id, lp, top: [[id, logprob, piece], ...]}, ...], most probable first."""
    out = []
    for e in probs or []:
        top = e.get('top_logprobs') or e.get('top_probs') or []
        out.append({'id': e.get('id'), 'lp': logp(e), 'top': [[t.get('id'), logp(t), t.get('token')] for t in top]})
    return out


def lead_lost(pos, want):
    """(first choice id, the lead `want` loses by in log-probability, whether `want` is among the listed tokens)."""
    top = pos['top']
    first = top[0]
    for tid, lp, _ in top:
        if tid == want:
            return first[0], first[1] - lp, True
    return first[0], first[1] - top[-1][1], False  # not listed: it loses by more than the last listed token does


def main():
    with open(A.prompts) as f:
        pack = json.load(f)
    bos, tail, stream = pack['bos'], pack['tail'], pack['stream']

    def cut(depth):  # (the prompt of that depth, the prompt without its tail), as deep_client.py cuts them
        base = bos + stream[:max(depth - len(bos) - len(tail), 0)]
        return (base + tail) or stream[:len(tail)], base

    ref = {}
    if A.ref_csv:
        with open(A.ref_csv) as f:
            for row in csv.DictReader(f):
                if row['arm'] == (A.ref_arm or A.arm) and row['kind'] == 'decode' and row['pass'] == '1':
                    ref[int(row['depth'])] = row['ids_md5']
    other, other_arm = {}, 'other'
    if A.score:
        with open(A.score) as f:
            pack_other = json.load(f)
        other_arm = pack_other['arm']
        other = {int(d): r for d, r in pack_other['replies'].items()}

    os.makedirs(A.out, exist_ok=True)
    log = open(os.path.join(A.out, f'requests_{A.arm}.jsonl'), 'w')

    def ask(kind, depth, prompt, n, extra=None, cache=True, note=None):
        r = post('/completion', {'prompt': prompt, 'n_predict': n, 'cache_prompt': cache, 'ignore_eos': True,
                                 'return_tokens': True, 'stream': False, **GREEDY, **(extra or {})})
        t = r.get('timings') or {}
        ids = r.get('tokens') or []
        row = {'arm': A.arm, 'kind': kind, 'depth': depth, **(note or {}), 'prompt_tokens': len(prompt),
               'cache_n': t.get('cache_n'), 'prompt_n': t.get('prompt_n'), 'gen_n': t.get('predicted_n'),
               'draft_n': t.get('draft_n', 0), 'draft_accepted': t.get('draft_n_accepted', 0),
               'ids_md5': md5(ids) if ids else '', 'ids': ids, 'probs': compact(r.get('completion_probabilities')),
               'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        if kind in ('gen', 'genp'):
            row['content'] = r.get('content') or ''
        log.write(json.dumps(row) + '\n')
        log.flush()
        return row

    def save(name, obj):
        path = os.path.join(A.out, name)
        with open(path + '.tmp', 'w') as f:
            json.dump(obj, f)
        os.replace(path + '.tmp', path)

    # Phase 1: the replies
    replies = {}
    ask('warmup', 0, stream[:64], 32, cache=False)
    for d in DEPTHS:
        full, base = cut(d)
        if len(base) > len(bos):
            ask('prime', d, base, 0)
        g = ask('gen', d, full, A.n_predict)
        p = ask('genp', d, full, A.n_predict, {'n_probs': A.n_probs})
        rep = {'ids': g['ids'], 'ids_md5': g['ids_md5'], 'ref_md5': ref.get(d, ''),
               'genp_same_ids': p['ids'] == g['ids'], 'genp_ids': p['ids'], 'probs': p['probs'],
               'draft_n': g['draft_n'], 'draft_accepted': g['draft_accepted'],
               'genp_draft_n': p['draft_n'], 'genp_draft_accepted': p['draft_accepted']}
        line = (f"{A.arm} depth {d:6d}: {len(g['ids'])} tokens, ids {g['ids_md5']}"
                f"{' = ref' if ref.get(d) == g['ids_md5'] else ' != ref ' + ref[d] if d in ref else ''}"
                f", with probabilities {'the same' if rep['genp_same_ids'] else 'OTHER TOKENS'}")
        if d in other:
            o = other[d]['ids']
            c = next((i for i, (x, y) in enumerate(zip(g['ids'], o)) if x != y), min(len(g['ids']), len(o)))
            rep['first_diff'] = c
            if c < min(len(g['ids']), len(o)):
                pr = ask('probe', d, full + g['ids'][:c], 1, {'n_probs': A.n_probs}, note={'i': c})
                rep['probe'] = pr['probs'][0] if pr['probs'] else None
                rep['probe_prompt_n'] = pr['prompt_n']
                line += f", parts from the other arm at token {c}"
            else:
                line += ", the same tokens as the other arm"
        replies[d] = rep
        save(f'replies_{A.arm}.json', {'arm': A.arm, 'n_predict': A.n_predict, 'replies': replies})
        print(line, file=sys.stderr, flush=True)
    if not A.score:
        return

    # Phase 2: teacher forcing on this line, this arm's reply and the other arm's
    scored = {}
    stopped = False
    for d in DEPTHS:
        if d not in other or stopped:
            continue
        full, base = cut(d)
        mine, theirs = replies[d]['ids'], other[d]['ids']
        c = replies[d]['first_diff']
        scored[d] = {}
        for name, reply, start in ((A.arm, mine, 0), (other_arm, theirs, c)):
            rows = []
            if start < len(reply) and len(base) > len(bos):
                ask('prime', d, base, 0, note={'reply': name})
            for i in range(start, len(reply)):
                if A.stop_file and os.path.exists(A.stop_file):
                    stopped = True
                    break
                r = ask('tf', d, full + reply[:i], 1, {'n_probs': A.n_probs}, note={'reply': name, 'i': i})
                if not r['probs']:
                    # the server lists no probabilities for a token whose text ends inside a character: the token it
                    # wrote is still its first choice
                    first = r['ids'][0] if r['ids'] else None
                    rows.append({'i': i, 'want': reply[i], 'first': first, 'is_first': first == reply[i],
                                 'lead_lost': 0.0 if first == reply[i] else None, 'listed': False, 'want_lp': None,
                                 'first_lp': None, 'prompt_n': r['prompt_n'], 'cache_n': r['cache_n']})
                    continue
                pos = r['probs'][0]
                first, lead, listed = lead_lost(pos, reply[i])
                rows.append({'i': i, 'want': reply[i], 'first': first, 'is_first': first == reply[i],
                             'lead_lost': 0.0 if first == reply[i] else lead, 'listed': listed,
                             'want_lp': next((lp for tid, lp, _ in pos['top'] if tid == reply[i]), None),
                             'first_lp': pos['top'][0][1], 'prompt_n': r['prompt_n'], 'cache_n': r['cache_n']})
            scored[d][name] = {'start': start, 'rows': rows}
            off = [x for x in rows if not x['is_first']]
            worst = max((x for x in off if x['lead_lost'] is not None), key=lambda x: x['lead_lost'], default=None)
            print(f"{A.arm} line, depth {d:6d}, reply of {name}: tokens {start} to {start + len(rows) - 1}, "
                  f"{len(off)} not the first choice of the listed tokens (end of text among them)"
                  + (f", largest lead lost {worst['lead_lost']:.4f} at token {worst['i']}" if worst else ''),
                  file=sys.stderr, flush=True)
            save(f'scored_{A.arm}.json', {'line': A.arm, 'stopped': stopped, 'scored': scored})
            if stopped:
                break
    if stopped:
        print('stopped by the stop file', file=sys.stderr, flush=True)


try:
    main()
except urllib.error.HTTPError as e:
    sys.exit(f'the server answered {e.code}: {e.read().decode(errors="replace")[:400]}')
except urllib.error.URLError as e:
    sys.exit(f'the server on port {A.port} does not answer: {e.reason}')
