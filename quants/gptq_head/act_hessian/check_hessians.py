#!/usr/bin/env python3
# Check llama-act-hessian's output against Hessians recomputed in numpy from the rows it dumped on a short run
# (HESS_DUMP), for one MoE layer L:
#   - ffn_norm-L.h64 = X^T X over the dumped ffn_norm rows;
#   - experts-L-gate_up.e64: per expert, the same over the rows routed to it (dumped ffn_moe_topk ids), row counts exact;
#   - experts-L-down.e64: per expert, over the ffn_moe_swiglu rows it produced;
#   - the sum of the experts' gate/up Hessians = n_used x the all-token Hessian.
#
#   check_hessians.py HESS_DIR L
#   check_hessians.py --compare DIR_A DIR_B      every .h64 / .e64 in both: same shapes and row counts, largest
#                                                difference relative to the largest entry (two builds, same run)
#
# Needs a run with HESS_TENSORS=ffn_norm-L HESS_EXPERTS=L HESS_DUMP=ffn_norm-L,ffn_moe_topk-L,ffn_moe_swiglu-L.
import os, sys
import numpy as np


def e64(path):
    """n, row counts, and the packed upper triangles [n_expert, n(n+1)/2] (memory-mapped)"""
    with open(path, 'rb') as f:
        n, ne = (int(x) for x in np.fromfile(f, np.int64, 2))
        rows = np.fromfile(f, np.int64, ne)
    tri = np.memmap(path, np.float64, 'r', offset=8 * (2 + ne), shape=(ne, n * (n + 1) // 2))
    return n, rows, tri


def full(tri, n):
    H = np.zeros((n, n))
    H[np.triu_indices(n)] = tri
    return H + np.triu(H, 1).T


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def compare(da, db, tol=1e-5):
    names = sorted(set(f for f in os.listdir(da) if f.endswith(('.h64', '.e64'))) &
                   set(f for f in os.listdir(db) if f.endswith(('.h64', '.e64'))))
    worst = 0.0
    for f in names:
        pa, pb = os.path.join(da, f), os.path.join(db, f)
        if f.endswith('.h64'):
            ha, hb = (np.fromfile(p, np.int64, 2) for p in (pa, pb))
            if not np.array_equal(ha, hb):
                sys.exit(f'{f}: headers differ {ha} {hb}')
            a, b = (np.memmap(p, np.float64, 'r', offset=16) for p in (pa, pb))
        else:
            na, ra, a = e64(pa)
            nb, rb, b = e64(pb)
            if na != nb or not np.array_equal(ra, rb):
                sys.exit(f'{f}: sizes or row counts differ')
        e = rel(np.asarray(a), np.asarray(b))
        worst = max(worst, e)
        print(f'{f}: {e:.2e}')
    print(f'{len(names)} files, worst {worst:.2e} (tolerance {tol:.0e}): ' + ('ALL OK' if worst <= tol and names else 'CHECK FAILED'))
    sys.exit(0 if worst <= tol and names else 1)


def main():
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    d, L = sys.argv[1], int(sys.argv[2])
    with open(os.path.join(d, f'ffn_norm-{L}.h64'), 'rb') as f:
        n, rows = (int(x) for x in np.fromfile(f, np.int64, 2))
        Hd = np.fromfile(f, np.float64).reshape(n, n)
    X = np.fromfile(os.path.join(d, f'ffn_norm-{L}.rows.f32'), np.float32).reshape(-1, n).astype(np.float64)
    ids = np.fromfile(os.path.join(d, f'ffn_moe_topk-{L}.ids.i32'), np.int32)
    T = len(X)
    used = len(ids) // T
    ids = ids.reshape(T, used)
    ng, rg, Tg = e64(os.path.join(d, f'experts-{L}-gate_up.e64'))
    nf, rd, Td = e64(os.path.join(d, f'experts-{L}-down.e64'))
    ne = len(rg)
    S = np.fromfile(os.path.join(d, f'ffn_moe_swiglu-{L}.rows.f32'), np.float32).reshape(T, used, nf).astype(np.float64)
    ok = True

    def report(label, err, tol):
        nonlocal ok
        ok &= err <= tol
        print(f'{label}: {err:.2e} (tolerance {tol:.0e}) {"ok" if err <= tol else "FAIL"}')

    print(f'layer {L}: {T} rows, {used} experts per token, {ne} experts; dense rows in the file {rows}')
    report('rows in ffn_norm .h64 vs dump', abs(rows - T), 0)
    report('ffn_norm H vs X^T X (max abs / max)', rel(Hd, X.T @ X), 1e-5)
    cnt = np.bincount(ids.reshape(-1), minlength=ne)
    report('gate/up row counts vs ids', int(np.abs(cnt - rg).max()), 0)
    report('down row counts vs ids', int(np.abs(cnt - rd).max()), 0)
    report('distinct experts per token', int((np.sort(ids, 1)[:, 1:] == np.sort(ids, 1)[:, :-1]).sum()), 0)
    eg = ed = 0.0
    Hsum = np.zeros((ng, ng))
    for e in range(ne):
        t, s = np.nonzero(ids == e)
        He, Hde = full(Tg[e], ng), full(Td[e], nf)
        Hsum += He
        eg = max(eg, rel(He, X[t].T @ X[t]) if len(t) else float(np.abs(He).max()))
        Y = S[t, s]
        ed = max(ed, rel(Hde, Y.T @ Y) if len(t) else float(np.abs(Hde).max()))
    report('per-expert gate/up H vs recomputed (worst expert)', eg, 1e-5)
    report('per-expert down H vs recomputed (worst expert)', ed, 1e-5)
    report('sum of expert gate/up H vs n_used x all-token H', rel(Hsum, used * Hd), 1e-5)
    print('ALL OK' if ok else 'CHECK FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
