#!/usr/bin/env python3
# Long-context KLD analysis: per-token dumps of llama-perplexity --kl-divergence (LLAMA_KLD_DUMP, see README.md) for
# several files against one reference, compared chunk by chunk with a reference file (UD-Q4_K_XL in Q106).
#
#   analyze.py --ref NAME [--chunks CHUNKS.jsonl --model MODEL.gguf] [--only-chunks 0,1,3,...] NAME=DUMP.tsv [...]
#
# For every file: mean KLD and same-top rate over the scored tokens; the paired comparison with --ref by chunk (mean of
# the per-chunk ratios is not used: the ratio of means, with a bootstrap interval over chunks, and the mean and
# standard error of the per-chunk differences). With --chunks (agent text from build_eval.py) and --model (for the
# vocabulary), the same split by token class: tool schemas, user turns, tool output, reasoning, calls, answers, chat
# markup, the first token after each </think> (call or answer), and long-range copies (the token ends an 8-token run
# that already appeared ending more than 512 tokens earlier). Every dump's token ids are checked against the chunks.
# --only-chunks keeps those chunks of dumps made on a full text and renumbers them in that order, to compare with dumps
# made on a subset_eval.py subset (whose chunks.jsonl then goes to --chunks).
# Needs numpy (and gguf-py with --model).
import argparse, json, sys
from collections import defaultdict
import numpy as np

MARKERS = [b'<|system|>', b'<|user|>', b'<|assistant|>', b'<|observation|>', b'<|developer|>', b'<think>', b'</think>',
           b'<tool_call>', b'</tool_call>', b'<tool_response>', b'</tool_response>']
CLASSES = ['tools/system', 'user', 'tool output', 'reasoning', 'call', 'answer', 'markup', 'after </think>',
           'long-range copy']
# what the model reads (prefilled, never sampled in an agent loop) and what it writes (sampled)
GROUPS = {'read (tools, user, tool output)': ['tools/system', 'user', 'tool output'],
          'written (reasoning, calls, answers)': ['reasoning', 'call', 'answer']}


def load_dump(path):
    d = np.genfromtxt(path, delimiter='\t', names=True, dtype=None, encoding=None)
    return d


def load_dumps(specs, only_chunks=None):
    """NAME=DUMP.tsv specs as an ordered dict; only_chunks keeps those chunks, renumbered in that order"""
    dumps = {}
    for s in specs:
        name, path = s.split('=', 1)
        d = load_dump(path)
        if only_chunks:
            keep = [int(x) for x in only_chunks.split(',')]
            order = {c: i for i, c in enumerate(keep)}
            d = d[np.isin(d['chunk'], keep)]
            d['chunk'] = np.array([order[int(c)] for c in d['chunk']])
            d = d[np.lexsort((d['pos'], d['chunk']))]
        dumps[name] = d
    return dumps


def check_same_tokens(dumps, ref_name):
    ref = dumps[ref_name]
    for n, d in dumps.items():
        if len(d) != len(ref) or not (np.array_equal(d['chunk'], ref['chunk']) and np.array_equal(d['pos'], ref['pos'])
                                      and np.array_equal(d['token'], ref['token'])):
            sys.exit(f'{n}: its scored tokens differ from {ref_name}\'s')


def piece_decoder(model):
    """token id -> its bytes, from the GGUF's vocabulary (needs gguf-py)"""
    import gguf
    from build_eval import bytes_decoder
    r = gguf.GGUFReader(model)
    tok_strs = r.get_field('tokenizer.ggml.tokens').contents()
    tok_types = r.get_field('tokenizer.ggml.token_type').contents()
    dec = bytes_decoder()
    cache = {}

    def piece(t):
        if t not in cache:
            s = tok_strs[t]
            cache[t] = s.encode('utf-8') if tok_types[t] in (3, 4) else bytes(dec[c] for c in s)
        return cache[t]
    return piece


def labels_for(tokens, detok_piece):
    """class of every token of one chunk, by the chat markup around it"""
    pieces = [detok_piece(t) for t in tokens]
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    text = b''.join(pieces)
    events = []
    for m in MARKERS:
        i = text.find(m)
        while i >= 0:
            events.append((i, i + len(m), m))
            i = text.find(m, i + 1)
    events.sort()
    lab = ['tools/system'] * len(tokens)
    role, think, call = 'system', False, False
    ei = 0
    after_think = set()
    for k, s in enumerate(starts):
        # apply every marker that starts before this token's end
        end = s + len(pieces[k])
        in_marker = False
        while ei < len(events) and events[ei][0] < end:
            a, b, m = events[ei]
            in_marker = True
            if m in (b'<|system|>', b'<|developer|>'):
                role, think, call = 'system', False, False
            elif m == b'<|user|>':
                role, think, call = 'user', False, False
            elif m == b'<|assistant|>':
                role, think, call = 'assistant', False, False
            elif m == b'<|observation|>':
                role, think, call = 'observation', False, False
            elif m == b'<think>':
                think = True
            elif m == b'</think>':
                think = False
                after_think.add(b)  # the next token starting at or after this offset
            elif m == b'<tool_call>':
                call = True
            elif m == b'</tool_call>':
                call = False
            ei += 1
        if in_marker:
            lab[k] = 'markup'
        elif role == 'system':
            lab[k] = 'tools/system'
        elif role == 'user':
            lab[k] = 'user'
        elif role == 'observation':
            lab[k] = 'tool output'
        else:
            lab[k] = 'reasoning' if think else ('call' if call else 'answer')
    first_after = set()
    for off in after_think:
        k = int(np.searchsorted(starts, off, side='left'))
        if k < len(tokens):
            first_after.add(k)
    return lab, first_after


def long_range_copies(tokens, n=8, dist=512):
    first_end = {}
    out = set()
    for p in range(n - 1, len(tokens)):
        g = tuple(tokens[p - n + 1:p + 1])
        q = first_end.get(g)
        if q is None:
            first_end[g] = p
        elif q < p - dist:
            out.add(p)
    return out


def paired(kld_by_chunk, ref_by_chunk, rng, n_boot=2000):
    a, b = np.array(kld_by_chunk), np.array(ref_by_chunk)
    ratio = a.mean() / b.mean()
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = a[idx].mean(1) / b[idx].mean(1)
    d = a - b
    return ratio, np.percentile(boots, [2.5, 97.5]), d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--chunks'); ap.add_argument('--model'); ap.add_argument('--only-chunks')
    ap.add_argument('dumps', nargs='+')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    dumps = load_dumps(a.dumps, a.only_chunks)
    names = list(dumps)
    if a.ref not in dumps:
        sys.exit(f'--ref {a.ref} has no dump')
    ref = dumps[a.ref]
    check_same_tokens(dumps, a.ref)
    chunks = sorted(set(ref['chunk'].tolist()))

    cls = None
    if a.chunks:
        piece = piece_decoder(a.model)
        meta = [json.loads(l) for l in open(a.chunks)]
        cls = defaultdict(lambda: np.zeros(len(ref), bool))
        row_of = {(c, p): i for i, (c, p) in enumerate(zip(ref['chunk'], ref['pos']))}
        for ci, m in enumerate(meta):
            toks = m['tokens']
            lab, after = labels_for(toks, piece)
            copies = long_range_copies(toks)
            for p in range(len(toks)):
                i = row_of.get((ci, p))
                if i is None:
                    continue
                if toks[p] != ref['token'][i]:
                    sys.exit(f'chunk {ci} position {p}: dump token {ref["token"][i]} but the chunk has {toks[p]}')
                cls[lab[p]][i] = True
                if p in after:
                    cls['after </think>'][i] = True
                if p in copies:
                    cls['long-range copy'][i] = True

    print(f'{len(chunks)} chunks, {len(ref)} scored tokens, reference for ratios: {a.ref}\n')
    print('| File | KLD | Same top | Ratio to ' + a.ref + ' (95 % bootstrap over chunks) | Mean paired difference ± SE |')
    print('|---|---|---|---|---|')
    per_chunk = {n: [d['kld'][d['chunk'] == c].mean() for c in chunks] for n, d in dumps.items()}
    for n in names:
        d = dumps[n]
        ratio, ci, dm, se = paired(per_chunk[n], per_chunk[a.ref], rng)
        print(f'| {n} | {d["kld"].mean():.4f} | {100 * d["same_top"].mean():.2f} % | {ratio:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) '
              f'| {dm:+.4f} ± {se:.4f} |')
    print('\nBy position in the chunk (KLD):\n')
    bins = [(4096, 5120), (5120, 6144), (6144, 7168), (7168, 8192)] if ref['pos'].max() > 4096 else None
    if bins:
        print('| File | ' + ' | '.join(f'{lo}-{hi - 1}' for lo, hi in bins) + ' |')
        print('|---|' + '---|' * len(bins))
        for n in names:
            d = dumps[n]
            print(f'| {n} | ' + ' | '.join(f'{d["kld"][(d["pos"] >= lo) & (d["pos"] < hi)].mean():.4f}' for lo, hi in bins) + ' |')
    if cls is not None:
        print('\nBy token class (KLD; share of scored tokens; ratio to ' + a.ref + '):\n')
        print('| Class | Tokens | ' + ' | '.join(names) + ' | ' + ' | '.join(f'{n}/{a.ref}' for n in names if n != a.ref) + ' |')
        print('|---|---|' + '---|' * (2 * len(names) - 1))
        for c in CLASSES:
            m = cls[c]
            if not m.any():
                continue
            vals = {n: dumps[n]['kld'][m].mean() for n in names}
            print(f'| {c} | {m.sum()} ({100 * m.mean():.1f} %) | ' + ' | '.join(f'{vals[n]:.4f}' for n in names) + ' | '
                  + ' | '.join(f'{vals[n] / vals[a.ref]:.3f}' for n in names if n != a.ref) + ' |')
        print('\nRead against written, paired by chunk (chunks with tokens of the group):\n')
        print('| Group | Tokens | ' + ' | '.join(f'{n} KLD' for n in names) + ' | '
              + ' | '.join(f'{n}/{a.ref} (95 % bootstrap)' for n in names if n != a.ref) + ' |')
        print('|---|---|' + '---|' * (2 * len(names) - 1))
        for g, members in GROUPS.items():
            m = np.zeros(len(ref), bool)
            for c in members:
                m |= cls[c]
            use = [c for c in chunks if (m & (ref['chunk'] == c)).any()]
            pc = {n: [dumps[n]['kld'][m & (dumps[n]['chunk'] == c)].mean() for c in use] for n in names}
            cells = []
            for n in names:
                if n == a.ref:
                    continue
                ratio, ci, _, _ = paired(pc[n], pc[a.ref], rng)
                cells.append(f'{ratio:.3f} ({ci[0]:.3f}-{ci[1]:.3f})')
            print(f'| {g} | {m.sum()} ({100 * m.mean():.1f} %) | ' + ' | '.join(f'{dumps[n]["kld"][m].mean():.4f}' for n in names)
                  + ' | ' + ' | '.join(cells) + ' |')
        print('\nSame top token by class:\n')
        print('| Class | ' + ' | '.join(names) + ' |')
        print('|---|' + '---|' * len(names))
        for c in CLASSES:
            m = cls[c]
            if m.any():
                print(f'| {c} | ' + ' | '.join(f'{100 * dumps[n]["same_top"][m].mean():.2f} %' for n in names) + ' |')
    nan = {n: int((~np.isfinite(d['kld'])).sum()) for n, d in dumps.items()}
    print(f'\nNon-finite KLD values: {nan}')


if __name__ == '__main__':
    main()
