#!/usr/bin/env python3
"""c843_pick.py: picks C843's requests out of the release agent check's recorded request bodies (proxy.jsonl.bodies.gz,
one {"id", "body"} line per request) and writes requests.jsonl: one {"label", "source", "body"} line per pick, the body
as the agent sent it. A pick is LABEL=ID (the proxy's request id); a BODIES file is searched for every id.
  python3 -I c843_pick.py --out requests.jsonl --bodies <run>/proxy.jsonl.bodies.gz ... \
    --pick A1=<id> --pick B1=<id> ...
It prints per pick its label, the number of messages and tools and the body's size; no text.
"""
import argparse, gzip, json

ap = argparse.ArgumentParser()
ap.add_argument('--bodies', action='append', required=True)
ap.add_argument('--pick', action='append', required=True)
ap.add_argument('--out', required=True)
A = ap.parse_args()

want = dict(p.split('=', 1)[::-1] for p in A.pick)  # id -> label
found = {}
for f in A.bodies:
    with gzip.open(f, 'rt') as g:
        for line in g:
            r = json.loads(line)
            if r['id'] in want and want[r['id']] not in found:
                found[want[r['id']]] = r
missing = [l for l in want.values() if l not in found]
if missing:
    raise SystemExit(f'not found: {missing}')
with open(A.out, 'w') as out:
    for p in A.pick:
        lab = p.split('=', 1)[0]
        r = found[lab]
        out.write(json.dumps({'label': lab, 'source': r['id'], 'body': r['body']}, ensure_ascii=False) + '\n')
        b = r['body']
        print(lab, len(b['messages']), 'messages', len(b.get('tools', [])), 'tools',
              len(json.dumps(b, ensure_ascii=False)), 'bytes')
