#!/usr/bin/env python3
# GPTQ for groups of Flint's always-read body tensors (Q112): the same Q5_0 rounding as gptq_head.py (searched block
# scales, column order, 1 % damping), with input Hessians captured by act_hessian/ from the mix's own forward pass.
#
#   gptq_body.py GROUP BF16.gguf MIX.gguf IMATRIX.gguf HESS_DIR DONOR_OUT.gguf [--damp 0.01]
#
# GROUP: o08 (attention output, layers 0-8), sh916 (shared expert gate/up/down, 9-16), kv916 (attention k/v, 9-16).
# For every tensor of the group it prints the H-weighted relative output error tr(dW H dW^T) / tr(W H W^T) of the
# tensor as the mix holds it, of round-to-nearest Q5_0 with the importance matrix (what llama-quantize writes), and of
# GPTQ Q5_0, and writes the GPTQ tensors to DONOR_OUT.gguf for compose_gguf.py.
# HESS_DIR holds act_hessian's <graph tensor>.h64 files: kqv_out-L feeds attn_output, attn_norm-L attn_k/attn_v,
# ffn_norm-L the shared expert's gate/up, ffn_swiglu-L its down projection.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq_head as g

GROUPS = {
    'o08':   [(f'blk.{l}.attn_output.weight', f'kqv_out-{l}') for l in range(0, 9)],
    'sh916': [(f'blk.{l}.ffn_{k}_shexp.weight', f'ffn_norm-{l}' if k != 'down' else f'ffn_swiglu-{l}')
              for l in range(9, 17) for k in ('gate', 'up', 'down')],
    'kv916': [(f'blk.{l}.attn_{k}.weight', f'attn_norm-{l}') for l in range(9, 17) for k in ('k', 'v')],
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group', choices=list(GROUPS)); ap.add_argument('bf16'); ap.add_argument('mix')
    ap.add_argument('imatrix'); ap.add_argument('hess_dir'); ap.add_argument('out')
    ap.add_argument('--damp', type=float, default=0.01)
    a = ap.parse_args()
    import gguf
    rb = gguf.GGUFReader(a.bf16)
    bt = {t.name: t for t in rb.tensors}
    mt = {t.name: t for t in gguf.GGUFReader(a.mix).tensors}
    it = {t.name: t for t in gguf.GGUFReader(a.imatrix).tensors}

    blocks = {}
    print(f'| Tensor | rows x n | Hessian (rows) | as in the mix | RTN Q5_0 (imatrix) | GPTQ Q5_0 | GPTQ / RTN |')
    print('|---|---|---|---|---|---|---|')
    tot = np.zeros(3)
    for name, hname in GROUPS[a.group]:
        H, hrows = read_h64(os.path.join(a.hess_dir, hname + '.h64'))
        t = bt[name]
        W = g.head_f32(t).reshape(int(t.shape[1]), int(t.shape[0]))  # rows (outputs) x n (inputs)
        rows, n = W.shape
        if H.shape != (n, n):
            sys.exit(f'{name}: Hessian {hname} is {H.shape}, the tensor reads {n} values')
        H32 = H.astype(np.float32)
        Wmix = g.head_f32(mt[name]).reshape(rows, n)
        qw = (np.asarray(it[name + '.in_sum2'].data, np.float64).reshape(-1) /
              max(1.0, float(np.asarray(it[name + '.counts'].data).reshape(-1)[0]))).astype(np.float32)
        c, d = g.imat(W, qw)
        Wrtn = g.unpack(g.pack(c, d), n)
        c, d = g.gptq_rows(np.ascontiguousarray(W.T), upper_factor(H, a.damp), np.diag(H).astype(np.float32), 'search')
        raw = g.pack(c, d)
        Wg = g.unpack(raw, n)
        e = [g.objective(W, x, H32) for x in (Wmix, Wrtn, Wg)]
        tot += np.array(e) * float(((W @ H32) * W).sum(dtype=np.float64))
        blocks[name] = raw
        print(f'| {name} | {rows} x {n} | {hname} ({hrows}) | {e[0]:.2e} ({mt[name].tensor_type.name}) | {e[1]:.2e} | '
              f'{e[2]:.2e} | {e[2] / e[1]:.3f} |', flush=True)
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
