#!/usr/bin/env python3
"""prompt_buckets.py: an agent check's prompt reading split by the size of each turn's new prompt.

    python3 -I prompt_buckets.py TURNS_CSV ARM... > prompt_buckets.tsv

TURNS_CSV is an agent check's res/turns.csv (bench/agent_check: per request the arm, prompt_n new prompt tokens,
cache_n cached ones, prompt_ms, ttft_s). Per arm and bucket of new prompt tokens: the turns, their new tokens, the
server's prompt seconds, tok/s over the bucket, and the median time to first token as the client saw it.
Standard library only.
"""
import csv
import statistics
import sys

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9))
path, arms = sys.argv[1], sys.argv[2:]
rows = [r for r in csv.DictReader(open(path)) if r['arm'] in arms and r['prompt_n'] and r['prompt_ms']]
print('arm\tnew_tokens\tturns\ttokens\tprompt_s\tprompt_tok_s\tttft_median_s')
for arm in arms:
    for lo, hi in BUCKETS + ((0, 10 ** 9),):
        b = [r for r in rows if r['arm'] == arm and lo <= int(r['prompt_n']) < hi]
        if not b:
            continue
        n = sum(int(r['prompt_n']) for r in b)
        s = sum(float(r['prompt_ms']) for r in b) / 1000
        ttft = statistics.median(float(r['ttft_s']) for r in b if r['ttft_s'])
        name = 'all' if (lo, hi) == (0, 10 ** 9) else (f'{lo}+' if hi == 10 ** 9 else f'{lo}-{hi - 1}')
        print(f'{arm}\t{name}\t{len(b)}\t{n}\t{s:.1f}\t{n / s:.1f}\t{ttft:.2f}')
