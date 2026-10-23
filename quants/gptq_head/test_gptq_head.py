#!/usr/bin/env python3
# Checks for gptq_head.py.
#
#   test_gptq_head.py                          synthetic: Q5_0 packing, and GPTQ against round-to-nearest under a known H
#   test_gptq_head.py <bf16.gguf> <q.gguf>     real files as well: round-to-nearest of bf16's token_embd.weight must be
#                                              byte-identical to q's (a llama-quantize Q5_0 file whose token embedding
#                                              had no importance matrix), and GPTQ with H = I must equal it in error
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import os, sys, tempfile, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq_head as g

fails = []


def check(ok, what):
    print(('ok    ' if ok else 'FAIL  ') + what)
    if not ok:
        fails.append(what)


def synthetic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 256)).astype(np.float32) * 0.02
    c, d = g.rtn_ref(x)
    y = g.unpack(g.pack(c, d), 256)
    ref = ((c.astype(np.float32) - 16).reshape(64, 8, 32) * d.astype(np.float32)[:, :, None]).reshape(64, 256)
    check(np.array_equal(y, ref), 'pack/unpack round trip')
    check(np.linalg.norm(y - x) / np.linalg.norm(x) < 0.05, 'round-to-nearest relative error < 5 %')
    # the extreme value of every block is represented exactly (ggml's rule: largest magnitude -> -16 * d)
    gx, gy = x.reshape(64, 8, 32), y.reshape(64, 8, 32)
    i = np.abs(gx).argmax(axis=2)
    check(np.allclose(np.take_along_axis(gy, i[:, :, None], 2), np.take_along_axis(gx, i[:, :, None], 2), rtol=1e-3),
          'block extremes kept (up to fp16 scale rounding)')

    n, r, T = 256, 512, 20000
    A = rng.standard_normal((n, n)) * 0.3 + np.eye(n)
    sc = np.ones(n); sc[:4] = 20                      # a few outlier input dimensions, as real hidden states have
    X = (rng.standard_normal((T, n)) @ A) * sc
    H = X.T @ X / T
    W = (rng.standard_normal((r, n)) * 0.02).astype(np.float32)
    H32 = H.astype(np.float32)
    obj = lambda c, d: g.objective(W, g.unpack(g.pack(c, d), n), H32)
    o_rtn = obj(*g.rtn_ref(W))
    o_imat = obj(*g.imat(W, np.diag(H)))
    print(f'      objective (relative output error under H): rtn {o_rtn:.3g}, imat {o_imat:.3g}')
    check(o_imat < o_rtn, 'imat beats round-to-nearest')
    hdiag = np.diag(H).astype(np.float32)
    res = {}
    for scale in ['max', 'search']:
        for ao in [False, True]:
            Hd = H + 0.01 * np.diag(H).mean() * np.eye(n)
            perm = np.argsort(-np.diag(H)) if ao else None
            if perm is not None:
                Hd = Hd[perm][:, perm]
            L = np.linalg.cholesky(Hd); Li = np.linalg.inv(L)
            U = np.linalg.cholesky(Li.T @ Li).T.astype(np.float32)
            c, d = g.gptq_rows(np.ascontiguousarray(W.T), U, hdiag, scale, perm)
            res[scale, ao] = obj(c, d)
            print(f'      gptq scale={scale} act_order={ao}: {res[scale, ao]:.3g}, codes {c.min()}..{c.max()}')
            check(res[scale, ao] < o_imat / 2, f'gptq {scale} act_order={ao} at least 2x better than imat')
            if not ao:
                d2 = g.refit_scales(W, c, H32)
                o2 = obj(c, d2.astype(np.float16))
                print(f'      + scale refit: {o2:.3g}')
                check(o2 <= res[scale, ao] * 1.001, f'scale refit does not hurt ({scale})')
    # GPTQ with H = I has nothing to compensate: it is round-to-nearest with the stored fp16 scale (ggml rounds with
    # the unrounded fp32 scale, so ~0.1 % of codes differ from llama-quantize's by one step)
    c, d = g.gptq_rows(np.ascontiguousarray(W.T), np.eye(n, dtype=np.float32), np.ones(n, np.float32), 'max')
    c0, d0 = g.rtn_ref(W)
    d16 = np.repeat(d.astype(np.float32), 32, axis=1)
    nearest = np.clip(np.floor(W / d16 + np.float32(16.5)), 0, 31).astype(np.uint8)
    check(np.array_equal(d, d0) and np.array_equal(c, nearest), 'gptq with H = I is round-to-nearest with the fp16 scale')
    check(np.abs(c.astype(int) - c0).max() <= 1, f'... and within one step of llama-quantize ({100 * (c != c0).mean():.2f} % differ)')


def real(bf16, qfile):
    import gguf
    t = g.head_tensor(qfile, 'token_embd.weight')[1]
    check(t.tensor_type == gguf.GGMLQuantizationType.Q5_0, f'{qfile}: token_embd.weight is Q5_0')
    tb = g.head_tensor(bf16, 'token_embd.weight')[1]
    rows = slice(0, 8192)
    c, d = g.rtn_ref(g.head_f32(tb, rows))
    mine = g.pack(c, d).reshape(-1)
    theirs = np.asarray(t.data[rows]).reshape(-1)
    check(np.array_equal(mine, theirs), f'round-to-nearest byte-identical to llama-quantize (token_embd rows 0-8191, '
          f'{(mine != theirs).sum()} bytes differ)')
    with tempfile.TemporaryDirectory() as tmp:
        np.save(f'{tmp}/I.npy', np.eye(int(tb.shape[0])))
        me = os.path.abspath(g.__file__)
        for mode, extra in [('rtn', []), ('gptq', ['--hessian', f'{tmp}/I.npy', '--damp', '0'])]:
            subprocess.run([sys.executable, me, 'quant', bf16, f'{tmp}/{mode}.q5_0', '--mode', mode,
                            '--tensor', 'token_embd.weight', '--chunk', '4096'] + extra, check=True,
                           stderr=subprocess.DEVNULL)
        a = np.fromfile(f'{tmp}/rtn.q5_0', np.uint8)
        check(np.array_equal(a, np.asarray(t.data).reshape(-1)), 'quant --mode rtn: whole tensor byte-identical')
        n = int(tb.shape[0])
        ra = g.unpack(a.reshape(-1, n // 32 * 22), n)
        rb = g.unpack(np.fromfile(f'{tmp}/gptq.q5_0', np.uint8).reshape(-1, n // 32 * 22), n)
        W = g.head_f32(tb)
        ea, eb = np.linalg.norm(ra - W), np.linalg.norm(rb - W)
        check(abs(ea - eb) / ea < 1e-4 and (ra != rb).mean() < 1e-3,
              f'quant --mode gptq with H = I equals rtn (error {eb / np.linalg.norm(W):.5f} vs {ea / np.linalg.norm(W):.5f})')


synthetic()
if len(sys.argv) == 3:
    real(sys.argv[1], sys.argv[2])
print('all passed' if not fails else f'{len(fails)} FAILED')
sys.exit(1 if fails else 0)
