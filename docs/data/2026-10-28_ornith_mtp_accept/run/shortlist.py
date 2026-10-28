#!/usr/bin/env python3
"""Draft-head shortlists (FR-Spec style) from Ornith's own generated text: its replies in the agent check's request logs.

  python3 -I shortlist.py text BODIES.jsonl.gz ... > replies.txt
      every assistant message (reasoning, text, tool calls) of the logged requests, each distinct one once
  python3 -I shortlist.py ids TOKENIZER.json TEXT ... > IDS.txt
      the texts' token ids with the model's own tokenizer (needs the `tokenizers` package; the rest is standard library)
  python3 -I shortlist.py lists IDS.txt OUT_PREFIX 8192 16384 32768 65536
      writes OUT_PREFIX-<K>.txt, the K most frequent token ids, one per line, most frequent first (ids the text never
      uses, lowest first, fill a list past its distinct ids), and prints each list's coverage of IDS.txt's own tokens;
      further IDS files after a lone "--" are only measured: their coverage per list

The logs are the agent check's proxy bodies (one JSON object per request, the request body under "body"). A request
carries the conversation so far, so every reply except each task's last appears in a later request.
"""
import collections
import gzip
import hashlib
import json
import re
import sys


def text(paths):
    seen = set()
    for path in paths:
        for line in gzip.open(path, 'rt', errors='replace'):
            try:
                body = json.loads(line).get('body') or {}
            except json.JSONDecodeError:
                continue
            for m in body.get('messages') or []:
                if m.get('role') != 'assistant':
                    continue
                parts = [m.get('reasoning_content') or '', m.get('content') or '']
                if isinstance(parts[1], list):
                    parts[1] = ''.join(c.get('text', '') for c in parts[1] if isinstance(c, dict))
                for c in m.get('tool_calls') or []:
                    f = c.get('function') or {}
                    parts.append(f'{f.get("name", "")}({f.get("arguments", "")})')
                t = '\n'.join(p for p in parts if p)
                h = hashlib.md5(t.encode('utf-8', 'replace')).hexdigest()
                if t and h not in seen:
                    seen.add(h)
                    sys.stdout.write(t + '\n\n')
    print(f'{len(seen)} distinct replies', file=sys.stderr)


def ids(tokenizer, paths):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    n = 0
    for path in paths:
        t = open(path, errors='replace').read()
        for i in range(0, len(t), 1 << 20):
            e = tok.encode(t[i:i + (1 << 20)], add_special_tokens=False).ids
            sys.stdout.write(' '.join(map(str, e)) + '\n')
            n += len(e)
    print(f'{n} tokens', file=sys.stderr)


def read_ids(path):
    return [int(x) for x in re.findall(r'-?\d+', open(path).read())]


def lists(ids_path, prefix, sizes, extra):
    all_ids = read_ids(ids_path)
    count = collections.Counter(all_ids)
    ranked = [t for t, _ in count.most_common()]
    print(f'{len(all_ids)} tokens, {len(count)} distinct ids', file=sys.stderr)
    # ids the text never uses fill a list past its distinct ids, lowest id first (a BPE vocabulary's earlier merges)
    seen = set(ranked)
    ranked += [t for t in range(max(sizes)) if t not in seen][:max(0, max(sizes) - len(ranked))]
    others = [(p, collections.Counter(read_ids(p))) for p in extra]
    print('k\tids\tcoverage' + ''.join(f'\t{p}' for p, _ in others))
    for k in sizes:
        top = ranked[:k]
        with open(f'{prefix}-{k}.txt', 'w') as f:
            f.write('\n'.join(map(str, top)) + '\n')
        cover = sum(count[t] for t in top) / len(all_ids)
        row = f'{k}\t{len(top)}\t{cover:.4f}'
        for _, c in others:
            row += f'\t{sum(c[t] for t in top) / max(1, sum(c.values())):.4f}'
        print(row)


if __name__ == '__main__':
    if sys.argv[1] == 'text':
        text(sys.argv[2:])
    elif sys.argv[1] == 'ids':
        ids(sys.argv[2], sys.argv[3:])
    else:
        args = sys.argv[4:]
        extra = args[args.index('--') + 1:] if '--' in args else []
        sizes = [int(x) for x in (args[:args.index('--')] if '--' in args else args)]
        lists(sys.argv[2], sys.argv[3], sizes, extra)
