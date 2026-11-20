#!/usr/bin/env python3
"""calls.py SERVER_LOG CLIENT_DIR: the prompt reads of R183's Ornith arm, step by step, from llama-server's log.

The server logs a progress line after each decode call of a prompt ("prompt processing, n_tokens = N, ... t = S s").
For every request this prints one row a call: the tokens read before and after it, the call's tokens and seconds and
its speed, then the request's totals with the server's checkpoint counters from the client's reply files
(<name>-1.json: timings ckpt_created_n, ckpt_incall_n, ckpt_save_ms). The last part of a prompt (from the last progress
line to its end) has no progress line and is left out of the rows; the totals hold it.
"""
import glob
import json
import os
import re
import sys

log, client = sys.argv[1], sys.argv[2]
prog = re.compile(r'task (\d+) \| prompt processing, n_tokens = +(\d+), progress = [0-9.]+, t = +([0-9.]+) s')
launch = re.compile(r'task (\d+) \| processing task')
calls = {}
order = []
for line in open(log, encoding='utf-8', errors='replace'):
    m = launch.search(line)
    if m:
        order.append(m.group(1))
        calls[m.group(1)] = [(0, 0.0)]
        continue
    m = prog.search(line)
    if m and m.group(1) in calls:
        calls[m.group(1)].append((int(m.group(2)), float(m.group(3))))

replies = []
for f in glob.glob(os.path.join(client, '*-1.json')):
    d = json.load(open(f))
    replies.append(d)
replies.sort(key=lambda d: int(d['row']['prompt_tokens']))

print('\t'.join(['prompt', 'call', 'from_tokens', 'to_tokens', 'tokens', 'seconds', 'tok_s']))
totals = []
for task, d in zip(order, replies):
    name = d['row']['name']
    pts = calls[task]
    for i in range(1, len(pts)):
        (a, ta), (b, tb) = pts[i - 1], pts[i]
        print(f'{name}\t{i}\t{a}\t{b}\t{b - a}\t{tb - ta:.2f}\t{(b - a) / (tb - ta):.1f}')
    t = d['reply'].get('timings', {})
    totals.append((name, t.get('prompt_n'), round(t.get('prompt_ms', 0) / 1000, 2), t.get('ckpt_created_n'),
                   t.get('ckpt_incall_n'), round(t.get('ckpt_save_ms', 0) / 1000, 2)))
print()
print('\t'.join(['prompt', 'prompt_tokens', 'prompt_s', 'ckpt_created_n', 'ckpt_incall_n', 'ckpt_save_s']))
for r in totals:
    print('\t'.join(str(x) for x in r))
