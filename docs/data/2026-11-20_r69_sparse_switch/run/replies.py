#!/usr/bin/env python3
"""replies.py: where each reply of a deep-lane run first differs from the control's first load.

    python3 -I replies.py OUT [--control-load c10a] [--loads c10a,r075a,...] > replies.tsv

OUT is a deep lane's results folder (bench/deep_lane) run with KEEP_REPLY=1, so that each load's <load>.jsonl holds
every reply's token ids and each id's text. For every prime, decode and sample request (depth, kind, pass) it prints
one line per load: the reply's token-id md5 and text md5 (the first 8 characters), the first token where the ids differ
from the control load's reply at the same request (empty when they are the same), with that token's text in both, and
the draft's accepted of drafted tokens. A line starting with # after each request groups the loads by token-id md5.
Standard library only.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out')
ap.add_argument('--control-load', default='c10a')
ap.add_argument('--loads', default='c10a,r075a,r0625a,r0625b,r075b,c10b')
A = ap.parse_args()
loads = A.loads.split(',')
R = {}
for ld in loads:
    with open(os.path.join(A.out, ld + '.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if r.get('kind') in ('decode', 'sample', 'prime'):
                R[(ld, r['kind'], str(r['pass']), r['depth'])] = r
keys = sorted({k[1:] for k in R}, key=lambda k: (k[2], k[0], k[1]))
print('depth\tkind\tpass\tload\tids_md5\ttext_md5\tfirst_diff_vs_control\tdraft')
for kind, ps, d in keys:
    ref = R[(A.control_load, kind, ps, d)]
    groups = {}
    for ld in loads:
        r = R[(ld, kind, ps, d)]
        a, b = ref.get('ids') or [], r.get('ids') or []
        fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if fd is None and len(a) != len(b):
            fd = min(len(a), len(b))
        groups.setdefault(r['ids_md5'], []).append(ld)
        piece = ''
        if fd is not None and fd < min(len(a), len(b)) and r.get('pieces') and ref.get('pieces'):
            piece = f" {A.control_load} {ref['pieces'][fd]!r} -> {r['pieces'][fd]!r}"
        print(f"{d}\t{kind}\t{ps}\t{ld}\t{r['ids_md5'][:8]}\t{r['text_md5'][:8]}\t"
              f"{'' if fd is None else fd}{piece}\t{r['draft_accepted']}/{r['draft_n']}")
    print(f"# {d} {kind} {ps}: groups " + '; '.join(f"{m[:8]}: {' '.join(v)}" for m, v in groups.items()))
