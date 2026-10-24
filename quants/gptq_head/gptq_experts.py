#!/usr/bin/env python3
# Offline check of GPTQ on Flint's routed experts (Q114, the first step of GPTQ for the body's E2): for chosen MoE
# layers, the H-weighted output error of each expert's gate, up and down projection in Q5_0 as the mix holds it, as
# round-to-nearest with the importance matrix (llama-quantize's rule; "= mix" counts rows whose bytes equal the mix's),
# and as GPTQ (gptq_head.py's rounding: searched block scales, column order), with each expert's own Hessian from
# act_hessian's expert mode.
#
#   gptq_experts.py LAYERS BF16.gguf MIX.gguf IMATRIX.gguf HESS_DIR [--sample 64] [--damp 0.01] [--workers 8]
#
# LAYERS = 9,25,41,47. HESS_DIR holds experts-L-gate_up.e64 and experts-L-down.e64, and ffn_norm-L.h64 (the all-token
# Hessian of the same input, which fills in for experts with fewer rows than inputs). --sample takes that many experts
# per layer spread evenly over the ranking by routed rows (rarest to busiest; 0 = all).
# Damping: --damp of the mean diagonal, 10x for an expert with fewer than 2n rows; an expert's gate/up Hessian with
# fewer than n rows gets (n - rows) rows' worth of the all-token Hessian added before GPTQ. Errors are always measured
# with the expert's own Hessian, as sums over its rows, so the per-layer ratios weight experts by how often they run.
# The kill rule for E2: GPTQ / RTN error above 0.85.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq_head as g
from gptq_body import read_h64, upper_factor

KILL = 0.85


def e64_header(path):
    with open(path, 'rb') as f:
        n, ne = (int(x) for x in np.fromfile(f, np.int64, 2))
        rows = np.fromfile(f, np.int64, ne)
    return n, ne, rows, 8 * (2 + ne)


def e64_expert(path, e):
    """expert e's Hessian (sum over its rows, float64, full n x n) and its row count"""
    n, ne, rows, off = e64_header(path)
    m = n * (n + 1) // 2
    tri = np.fromfile(path, np.float64, m, offset=off + 8 * m * e)
    H = np.zeros((n, n))
    H[np.triu_indices(n)] = tri
    return H + np.triu(H, 1).T, int(rows[e])


_A = {}


def _init(bf16, mix, imatrix, hess_dir, damp):
    import gguf
    _A['bt'] = {t.name: t for t in gguf.GGUFReader(bf16).tensors}
    _A['mt'] = {t.name: t for t in gguf.GGUFReader(mix).tensors}
    _A['it'] = {t.name: t for t in gguf.GGUFReader(imatrix).tensors}
    _A['hess_dir'], _A['damp'] = hess_dir, damp


def solve(task):
    """(layer, kind, expert) -> errors (sums over the expert's rows) of mix, RTN, GPTQ, the energy, = mix share"""
    l, kind, e = task
    name = f'blk.{l}.ffn_{kind}_exps.weight'
    path = os.path.join(_A['hess_dir'], f'experts-{l}-{"down" if kind == "down" else "gate_up"}.e64')
    H, rows_e = e64_expert(path, e)
    t, mt = _A['bt'][name], _A['mt'][name]
    W = g.head_f32(t, rows=e)
    W = W.reshape(int(t.shape[1]), int(t.shape[0]))  # rows (outputs) x n (inputs)
    r, n = W.shape
    Wmix = g.head_f32(mt, rows=e).reshape(r, n)
    it = _A['it']
    s2 = np.asarray(it[name + '.in_sum2'].data).reshape(-1, n)[e]
    cnt = float(np.asarray(it[name + '.counts'].data).reshape(-1)[e])
    c, d = g.imat_ggml(W, g.imatrix_weights(s2, cnt))
    rtn_raw = g.pack(c, d)
    Wrtn = g.unpack(rtn_raw, n)
    same = None
    if mt.tensor_type.name == 'Q5_0':
        same = float((np.asarray(mt.data[e]).reshape(r, -1) == rtn_raw.reshape(r, -1)).all(axis=1).mean())
    Hs = H.copy()
    damp = _A['damp'] * (10 if rows_e < 2 * n else 1)
    if kind != 'down' and rows_e < n:
        Hall, _ = read_h64(os.path.join(_A['hess_dir'], f'ffn_norm-{l}.h64'))  # mean over all tokens
        Hs += (n - rows_e) * Hall
    if np.diag(Hs).max() == 0:
        Hs = np.eye(n)
    Hs /= max(rows_e, 1)
    c, d = g.gptq_rows(np.ascontiguousarray(W.T), upper_factor(Hs, damp), np.diag(Hs).astype(np.float32), 'search')
    Wg = g.unpack(g.pack(c, d), n)
    H32 = H.astype(np.float32)
    err = [float((((x - W) @ H32) * (x - W)).sum(dtype=np.float64)) for x in (Wmix, Wrtn, Wg)]
    energy = float(((W @ H32) * W).sum(dtype=np.float64))
    return l, kind, e, rows_e, err, energy, same, mt.tensor_type.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('layers'); ap.add_argument('bf16'); ap.add_argument('mix')
    ap.add_argument('imatrix'); ap.add_argument('hess_dir')
    ap.add_argument('--sample', type=int, default=64)
    ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(',')]
    tasks, picked = [], {}
    for l in layers:
        n, ne, rows, _ = e64_header(os.path.join(a.hess_dir, f'experts-{l}-gate_up.e64'))
        order = np.argsort(rows, kind='stable')
        idx = order if a.sample <= 0 or a.sample >= ne else order[np.unique(np.linspace(0, ne - 1, a.sample).round().astype(int))]
        picked[l] = (rows, idx)
        tasks += [(l, k, int(e)) for e in idx for k in ('gate', 'up', 'down')]

    import multiprocessing as mp
    initargs = (a.bf16, a.mix, a.imatrix, a.hess_dir, a.damp)
    res = {}
    with mp.get_context('fork').Pool(a.workers, _init, initargs) as pool:
        for out in pool.imap_unordered(solve, tasks):
            res[out[:3]] = out[3:]

    print(f'Experts per layer: {a.sample if a.sample > 0 else "all"}, spread over the ranking by routed rows; errors are '
          f'sums over each expert\'s rows (the layer ratios weight experts by use)\n')
    print('| Layer | Tensor | mix type | experts (rows: min / median / max) | mix / RTN | RTN = mix (rows) | GPTQ / RTN '
          '| per expert: median, 10th-90th pct | relative RTN error (median) |')
    print('|---|---|---|---|---|---|---|---|---|')
    overall = np.zeros(2)
    verdict = []
    for l in layers:
        rows, idx = picked[l]
        for k in ('gate', 'up', 'down'):
            v = [res[(l, k, int(e))] for e in idx]
            rws = np.array([x[0] for x in v])
            err = np.array([x[1] for x in v])          # [experts, 3]: mix, RTN, GPTQ
            en = np.array([x[2] for x in v])
            same = [x[3] for x in v if x[3] is not None]
            used = err[:, 1] > 0
            per = err[used, 2] / err[used, 1]
            ratio = err[:, 2].sum() / err[:, 1].sum()
            overall += np.array([err[:, 1].sum(), err[:, 2].sum()])
            verdict.append((l, k, ratio))
            print(f'| {l} | {k} | {v[0][4]} | {len(idx)} ({rws.min()} / {int(np.median(rws))} / {rws.max()}) | '
                  f'{err[:, 0].sum() / err[:, 1].sum():.3f} | {100 * np.mean(same) if same else float("nan"):.1f} % | '
                  f'**{ratio:.3f}** | {np.median(per):.3f}, {np.percentile(per, 10):.3f}-{np.percentile(per, 90):.3f} | '
                  f'{np.median(err[used, 1] / en[used]):.2e} |')
    print(f'\nAll sampled experts: GPTQ / RTN error {overall[1] / overall[0]:.3f} (kill above {KILL})')
    print('Layers/tensors above the kill line: ' + (', '.join(f'{l} {k} ({r:.3f})' for l, k, r in verdict if r > KILL) or 'none'))


if __name__ == '__main__':
    main()
