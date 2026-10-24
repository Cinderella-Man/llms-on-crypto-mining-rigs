#!/usr/bin/env python3
# GPTQ for groups of Flint's always-read body tensors (Q112): the same Q5_0 rounding as gptq_head.py (searched block
# scales, column order, 1 % damping), with input Hessians captured by act_hessian/ from the mix's own forward pass.
#
#   gptq_body.py GROUP BF16.gguf MIX.gguf IMATRIX.gguf HESS_DIR DONOR_OUT.gguf [--damp 0.01] [--workers 1]
#
# GROUP is a named group or a regex over tensor names (every always-read matrix of MIX it matches):
#   o08 (attention output, layers 0-8), sh916 (shared expert gate/up/down, 9-16), kv916 (attention k/v, 9-16): E1 on G3;
#   k2h_q (attention q 0-8, 25-48), k2h_kvo (attention k/v/output 33-48), k2h_sh (shared expert 33-45), k2h_q5 (all
#   three): K2h's Q5_0 always-read tensors; k2h_q8 (every always-read matrix K2h keeps in Q8_0, the head excluded).
# For every tensor of the group it prints the H-weighted relative output error tr(dW H dW^T) / tr(W H W^T) of the
# tensor as the mix holds it, of round-to-nearest Q5_0 with the importance matrix (what llama-quantize writes; "= mix"
# is the share of rows whose bytes equal the mix's when the mix holds Q5_0), and of GPTQ Q5_0, and writes the GPTQ
# tensors to DONOR_OUT.gguf for split_compose.py.
# HESS_DIR holds act_hessian's <graph tensor>.h64 files: attn_norm-L feeds attn_q/k/v, kqv_out-L attn_output, ffn_norm-L
# the shared expert's (and layers 0-1's dense) gate/up, ffn_swiglu-L their down projection.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq_head as g

GROUPS = {
    'o08':   [f'blk.{l}.attn_output.weight' for l in range(0, 9)],
    'sh916': [f'blk.{l}.ffn_{k}_shexp.weight' for l in range(9, 17) for k in ('gate', 'up', 'down')],
    'kv916': [f'blk.{l}.attn_{k}.weight' for l in range(9, 17) for k in ('k', 'v')],
    'k2h_q':   r'^blk\.([0-8]|2[5-9]|3\d|4[0-8])\.attn_q\.weight$',
    'k2h_kvo': r'^blk\.(3[3-9]|4[0-8])\.attn_(k|v|output)\.weight$',
    'k2h_sh':  r'^blk\.(3[3-9]|4[0-5])\.ffn_(gate|up|down)_shexp\.weight$',
    'k2h_q5':  r'^blk\.([0-8]|2[5-9]|3\d|4[0-8])\.attn_q\.weight$|^blk\.(3[3-9]|4[0-8])\.attn_(k|v|output)\.weight$|'
               r'^blk\.(3[3-9]|4[0-5])\.ffn_(gate|up|down)_shexp\.weight$',
    'k2h_q8':  'Q8_0',
}
ALWAYS_READ = re.compile(r'^blk\.(\d+)\.(attn_[qkv]|attn_output|ffn_(gate|up|down)(_shexp)?)\.weight$')


def hess_name(name):
    """the act_hessian graph tensor whose rows feed this weight"""
    m = ALWAYS_READ.match(name)
    if not m:
        sys.exit(f'{name}: not an always-read matrix with a known input')
    l, k = m.group(1), m.group(2)
    if k in ('attn_q', 'attn_k', 'attn_v'):
        return f'attn_norm-{l}'
    if k == 'attn_output':
        return f'kqv_out-{l}'
    return f'ffn_swiglu-{l}' if k.startswith('ffn_down') else f'ffn_norm-{l}'


def group_tensors(group, mix_tensors):
    spec = GROUPS.get(group, group)
    if isinstance(spec, list):
        return spec
    if spec == 'Q8_0':
        return [n for n, t in mix_tensors.items() if ALWAYS_READ.match(n) and t.tensor_type.name == 'Q8_0']
    pat = re.compile(spec)
    out = [n for n in mix_tensors if pat.search(n) and ALWAYS_READ.match(n)]
    if not out:
        sys.exit(f'{group}: no always-read matrix matches')
    return out


def read_h64(path):
    with open(path, 'rb') as f:
        n, rows = np.fromfile(f, np.int64, 2)
        h = np.fromfile(f, np.float64, n * n).reshape(n, n)
    return h / rows, int(rows)


def upper_factor(H, damp):
    """U with (H + damp mean(diag) I)^-1 = U^T U, as gptq_head's workers compute it"""
    H = H.copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))
    Linv = np.linalg.inv(np.linalg.cholesky(H))
    return np.linalg.cholesky(Linv.T @ Linv).T.astype(np.float32)


def importance(it, name):
    return g.imatrix_weights(np.asarray(it[name + '.in_sum2'].data).reshape(-1),
                             float(np.asarray(it[name + '.counts'].data).reshape(-1)[0]))


_A = {}


def _init(bf16, mix, imatrix, hess_dir, damp):
    import gguf
    _A['bt'] = {t.name: t for t in gguf.GGUFReader(bf16).tensors}
    _A['mt'] = {t.name: t for t in gguf.GGUFReader(mix).tensors}
    _A['it'] = {t.name: t for t in gguf.GGUFReader(imatrix).tensors}
    _A['hess_dir'], _A['damp'] = hess_dir, damp


def solve(name):
    """one tensor -> (name, table row values, GPTQ raw blocks)"""
    hname = hess_name(name)
    H, hrows = read_h64(os.path.join(_A['hess_dir'], hname + '.h64'))
    t = _A['bt'][name]
    W = g.head_f32(t).reshape(int(t.shape[1]), int(t.shape[0]))  # rows (outputs) x n (inputs)
    rows, n = W.shape
    if H.shape != (n, n):
        sys.exit(f'{name}: Hessian {hname} is {H.shape}, the tensor reads {n} values')
    H32 = H.astype(np.float32)
    mt = _A['mt'][name]
    Wmix = g.head_f32(mt).reshape(rows, n)
    c, d = g.imat_ggml(W, importance(_A['it'], name))
    rtn_raw = g.pack(c, d)
    Wrtn = g.unpack(rtn_raw, n)
    same = None
    if mt.tensor_type.name == 'Q5_0':
        same = float((np.asarray(mt.data).reshape(rows, -1) == rtn_raw.reshape(rows, -1)).all(axis=1).mean())
    c, d = g.gptq_rows(np.ascontiguousarray(W.T), upper_factor(H, _A['damp']), np.diag(H).astype(np.float32), 'search')
    raw = g.pack(c, d)
    Wg = g.unpack(raw, n)
    e = [g.objective(W, x, H32) for x in (Wmix, Wrtn, Wg)]
    energy = float(((W @ H32) * W).sum(dtype=np.float64))
    return name, dict(hname=hname, hrows=hrows, rows=rows, n=n, e=e, energy=energy, mixtype=mt.tensor_type.name,
                      same=same), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group'); ap.add_argument('bf16'); ap.add_argument('mix')
    ap.add_argument('imatrix'); ap.add_argument('hess_dir'); ap.add_argument('out')
    ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    import gguf
    rb = gguf.GGUFReader(a.bf16)
    names = group_tensors(a.group, {t.name: t for t in gguf.GGUFReader(a.mix).tensors})

    print(f'| Tensor | rows x n | Hessian (rows) | as in the mix | RTN Q5_0 (imatrix) | = mix | GPTQ Q5_0 | GPTQ / RTN |')
    print('|---|---|---|---|---|---|---|---|')
    tot = np.zeros(3)
    blocks = {}
    initargs = (a.bf16, a.mix, a.imatrix, a.hess_dir, a.damp)
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(a.workers, _init, initargs)
        results = pool.imap(solve, names)
    else:
        _init(*initargs)
        pool, results = None, map(solve, names)
    for name, r, raw in results:
        e = r['e']
        tot += np.array(e) * r['energy']
        blocks[name] = raw
        same = '' if r['same'] is None else f'{100 * r["same"]:.1f} %'
        print(f'| {name} | {r["rows"]} x {r["n"]} | {r["hname"]} ({r["hrows"]}) | {e[0]:.2e} ({r["mixtype"]}) | '
              f'{e[1]:.2e} | {same} | {e[2]:.2e} | {e[2] / e[1]:.3f} |', flush=True)
    if pool:
        pool.close()
    print(f'\ngroup {a.group}: GPTQ / RTN error, output-weighted over the tensors: {tot[2] / tot[1]:.3f}')

    w = gguf.GGUFWriter(a.out, arch=rb.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    for name, raw in blocks.items():
        w.add_tensor_info(name, raw.shape, np.dtype(np.uint8), raw.nbytes, gguf.GGMLQuantizationType.Q5_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for raw in blocks.values():
        w.write_tensor_data(raw)
    w.close()
    print(f'donor: {a.out} ({len(blocks)} tensors, Q5_0)')


if __name__ == '__main__':
    main()
