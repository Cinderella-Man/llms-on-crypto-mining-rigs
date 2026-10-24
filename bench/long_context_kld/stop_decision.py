#!/usr/bin/env python3
# Does a quantized file move where Flint stops reasoning? From the same per-token dumps as analyze.py (nll = -log p of
# the actual token under the file, nll_base under the dump's reference model), the log-probability every file gives
# the actual token at two kinds of positions in the agent conversations:
#   - stops: the token is </think> and closes a non-empty reasoning block, where the producing model chose to stop;
#   - markers: a reasoning word that opens a hesitation or re-check ("wait", "but", "maybe", ...; the word list of
#     bench/answer_time/lengths.py), counted at the word's first token.
# A file that gives the actual </think> less probability than --ref is less willing to stop there; more probability
# on markers means more hesitation.
#
#   stop_decision.py --ref NAME --chunks CHUNKS.jsonl --model MODEL.gguf [--base-name Q8_0] [--only-chunks 0,1,3,...]
#     NAME=DUMP.tsv [...]
#
# For every file: mean log p and mean p at each kind of position; the mean difference log p(file) - log p(--ref) with a
# 95 % bootstrap interval that resamples chunks (positions pooled within the resampled chunks), overall and by the
# model that produced the conversation (build_eval.py's "file"); and the same difference against the dumps' reference
# model (--base-name). Empty reasoning blocks (</think> right after <think>) are counted and left out.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, json, os, re, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'answer_time'))
from analyze import check_same_tokens, labels_for, load_dumps, piece_decoder
from lengths import MARKERS

KILL = 0.05  # nats: a smaller shift with an interval covering 0, for every file, closes quant choice as a length lever


def positions(meta, ref, piece):
    """row masks of the scored tokens: stops, empty blocks, markers; and the producing model of every row"""
    row_of = {(c, p): i for i, (c, p) in enumerate(zip(ref['chunk'], ref['pos']))}
    n = len(ref)
    stop, empty, marker = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    producer = np.empty(n, object)
    for ci, m in enumerate(meta):
        toks = m['tokens']
        pieces = [piece(t) for t in toks]
        lab, _ = labels_for(toks, piece)
        last_open = None
        for p, t in enumerate(toks):
            i = row_of.get((ci, p))
            if i is not None:
                if toks[p] != ref['token'][i]:
                    sys.exit(f'chunk {ci} position {p}: dump token {ref["token"][i]} but the chunk has {toks[p]}')
                producer[i] = m['file']
            if pieces[p] == b'<think>':
                last_open = p
                continue
            if pieces[p] == b'</think>':
                if i is not None:
                    inner = b''.join(pieces[last_open + 1:p]) if last_open is not None else b'x'
                    (stop if inner.strip() else empty)[i] = True
                continue
            if i is None or lab[p] != 'reasoning':
                continue
            s = pieces[p].decode('utf-8', 'ignore')
            if not s.strip():
                continue
            prev = pieces[p - 1].decode('utf-8', 'ignore') if p else ' '
            word_start = s[0].isspace() or not prev or not (prev[-1].isalnum() or prev[-1] in "'-")
            if word_start and MARKERS.match(b''.join(pieces[p:p + 8]).decode('utf-8', 'ignore').lstrip()):
                marker[i] = True
    return stop, empty, marker, producer


def shift(delta, chunk, mask, rng, n_boot=4000):
    """mean of delta over the masked rows, with a bootstrap over chunks"""
    cs = np.unique(chunk)
    sums = np.array([delta[mask & (chunk == c)].sum() for c in cs])
    cnts = np.array([(mask & (chunk == c)).sum() for c in cs])
    idx = rng.integers(0, len(cs), size=(n_boot, len(cs)))
    with np.errstate(invalid='ignore'):
        boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return sums.sum() / cnts.sum(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True); ap.add_argument('--chunks', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--only-chunks')
    ap.add_argument('--base-name', default='base')
    ap.add_argument('dumps', nargs='+')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    dumps = load_dumps(a.dumps, a.only_chunks)
    if a.ref not in dumps:
        sys.exit(f'--ref {a.ref} has no dump')
    check_same_tokens(dumps, a.ref)
    ref = dumps[a.ref]
    names = list(dumps)
    base_spread = max(float(np.abs(d['nll_base'] - ref['nll_base']).max()) for d in dumps.values())
    piece = piece_decoder(a.model)
    meta = [json.loads(l) for l in open(a.chunks)]
    stop, empty, marker, producer = positions(meta, ref, piece)
    chunk = ref['chunk']
    producers = sorted(set(producer[stop | marker]))

    print(f'{len(meta)} chunks, {len(ref)} scored tokens; reference for differences: {a.ref}; dump reference: '
          f'{a.base_name} (its log p agrees across the dumps to {base_spread:.1e} nats)\n')
    print('| Positions | All | ' + ' | '.join(f'produced by {p}' for p in producers) + ' |')
    print('|---|---|' + '---|' * len(producers))
    for label, m in (('stops (</think> closing a non-empty block)', stop), ('empty blocks (left out)', empty),
                     ('hesitation markers in reasoning', marker)):
        print(f'| {label} | {m.sum()} in {len(np.unique(chunk[m]))} chunks | '
              + ' | '.join(str((m & (producer == p)).sum()) for p in producers) + ' |')

    verdict = {}
    for label, m in (('Stops', stop), ('Hesitation markers', marker)):
        print(f'\n{label}: log p of the actual token\n')
        print(f'| File | mean log p | mean p | log p - {a.ref} (95 % bootstrap over chunks) | '
              + ' | '.join(f'produced by {p}' for p in producers) + f' | log p - {a.base_name} |')
        print('|---|---|---|---|' + '---|' * len(producers) + '---|')
        for n in names:
            lp = -dumps[n]['nll'][m]
            d_ref = ref['nll'] - dumps[n]['nll']
            d_base = dumps[n]['nll_base'] - dumps[n]['nll']
            cells = []
            if n == a.ref:
                cells.append('')
                cells += [''] * len(producers)
            else:
                v, lo, hi = shift(d_ref, chunk, m, rng)
                cells.append(f'{v:+.4f} ({lo:+.4f} to {hi:+.4f})')
                verdict.setdefault(n, []).append((label, v, lo, hi))
                for p in producers:
                    v, lo, hi = shift(d_ref, chunk, m & (producer == p), rng)
                    cells.append(f'{v:+.4f} ({lo:+.4f} to {hi:+.4f})')
            v, lo, hi = shift(d_base, chunk, m, rng)
            cells.append(f'{v:+.4f} ({lo:+.4f} to {hi:+.4f})')
            print(f'| {n} | {lp.mean():.4f} | {np.exp(lp).mean():.4f} | ' + ' | '.join(cells) + ' |')

    print(f'\nKill rule: |log p - {a.ref}| < {KILL} nats with an interval covering 0, at stops and markers, for every file\n')
    closed = True
    for n, rows in verdict.items():
        ok = all(abs(v) < KILL and lo <= 0 <= hi for _, v, lo, hi in rows)
        closed &= ok
        print(f'- {n}: ' + '; '.join(f'{l.lower()} {v:+.4f} ({lo:+.4f} to {hi:+.4f})' for l, v, lo, hi in rows)
              + (' -> within' if ok else ' -> outside'))
    print(f'\nQuant choice as a reasoning-length lever: {"closed" if closed else "open"}')


if __name__ == '__main__':
    main()
