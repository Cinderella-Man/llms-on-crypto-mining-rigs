#!/usr/bin/env python3
# GPTQ rounding of a GGUF output head to Q5_0 (or round-to-nearest for comparison), checked offline on held-out
# activations and written as a two-file split GGUF whose first file is shared between candidates.
#
#   gptq_head.py hessian <out H.npy> <x.f32> [...]             H = sum of x x^T over the dumped rows (float64)
#   gptq_head.py quant <src.gguf> <out.q5_0> --mode rtn|imat|gptq [--hessian H.npy] [--damp 0.01] [--scale max|search]
#                [--act-order] [--refit N] [--workers 8] [--tensor output.weight]   a tensor of src in Q5_0, raw blocks
#   gptq_head.py proxy <ref.gguf> <x.f32> <a.q5_0|q8_0:file.gguf|bf16> [...] [--rows 10240]
#                                                              KLD / same top token of the head alone, base = ref's head
#   gptq_head.py checkdump <model.gguf> <x.f32>                 dumped rows x the model's head vs the model's own logits
#   gptq_head.py base <file.gguf> <out-base.gguf>              the file without output.weight, as split 1 of 2
#   gptq_head.py shard <base.gguf> <head.q5_0> <dir> <name>    <dir>/<name>-0000{1,2}-of-00002.gguf (1 = link to base)
#   gptq_head.py full <file.gguf> <head.q5_0> <out.gguf>       the file with output.weight replaced, one file
#
# x.f32 files come from llama-head-dump (head_dump/): float32 rows of n_embd values plus an .idx file of
# {sequence, position, token}; rows at position 0 are left out of H.
# Modes: rtn = llama-quantize's Q5_0 without an importance matrix (quantize_row_q5_0_ref, byte-identical);
# imat = llama-quantize's Q5_0 with an importance matrix holding diag(H) (quantize_row_q5_0_impl); gptq = GPTQ
# (Frantar et al. 2022) in column order with lazy 32-column block updates, one group = one Q5_0 block. A group's scale
# is chosen when GPTQ reaches the group, from its error-updated weights: max = ggml's rule (largest magnitude -> -16),
# search = the imat rule weighted by diag(H). --act-order processes columns by decreasing diag(H) with scales fixed
# beforehand from the original weights. --refit N then sets each block's scale by least squares under H for the codes
# found, N times, with a GPTQ pass for the new scales between two refits.
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, os, sys
import numpy as np

QK = 32
BLOCK = np.dtype([('d', '<f2'), ('qh', '<u4'), ('qs', 'u1', 16)])  # block_q5_0, 22 bytes
assert BLOCK.itemsize == 22


# ---- Q5_0 blocks ----

def pack(codes, d):
    """codes uint8 [rows, n] in 0..31, d float16 [rows, n/32] -> block_q5_0 bytes [rows, n/32*22]"""
    r, n = codes.shape
    c = codes.reshape(r, n // QK, QK).astype(np.uint32)
    out = np.empty((r, n // QK), BLOCK)
    out['d'] = d
    out['qs'] = (c[:, :, :16] & 15) | ((c[:, :, 16:] & 15) << 4)
    out['qh'] = ((c >> 4) << np.arange(QK, dtype=np.uint32)).sum(axis=2, dtype=np.uint32)
    return out.view(np.uint8).reshape(r, -1)


def unpack(raw, n):
    """block_q5_0 bytes [rows, n/32*22] -> float32 [rows, n]"""
    b = np.ascontiguousarray(raw).view(BLOCK).reshape(raw.shape[0], n // QK)
    lo = np.concatenate([b['qs'] & 15, b['qs'] >> 4], axis=2).astype(np.int32)
    hi = ((b['qh'][:, :, None] >> np.arange(QK, dtype=np.uint32)) & 1).astype(np.int32)
    q = (lo | (hi << 4)) - 16
    return (q * b['d'].astype(np.float32)[:, :, None]).reshape(raw.shape[0], n)


def rtn_ref(x):
    """quantize_row_q5_0_ref: x float32 [rows, n] -> codes, d (float16)"""
    r, n = x.shape
    g = x.reshape(r, n // QK, QK)
    i = np.abs(g).argmax(axis=2)
    mx = np.take_along_axis(g, i[:, :, None], axis=2)[:, :, 0]
    d = (mx / np.float32(-16)).astype(np.float32)
    with np.errstate(divide='ignore'):
        idd = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = (g * idd[:, :, None] + np.float32(16.5)).astype(np.int8)
    q = np.minimum(q, 31).astype(np.uint8)
    return q.reshape(r, n), d.astype(np.float16)


def make_qx_quants(x, w, nmax=16):
    """ggml make_qx_quants(n, nmax, x, L, rmse_type=1, qw) on rows of one block: x, w float32 [rows, 32] ->
    codes (L + nmax) uint8 [rows, 32], scale float32 [rows]"""
    i = np.abs(x).argmax(axis=1)
    mx = x[np.arange(len(x)), i]
    zero = np.abs(mx) < 1e-30
    mx = np.where(zero, np.float32(1), mx)

    def fit(iscale):
        L = np.clip(np.rint(iscale[:, None] * x), -nmax, nmax - 1)
        sumlx = (w * x * L).sum(axis=1)
        suml2 = (w * L * L).sum(axis=1)
        return L, sumlx, suml2

    iscale = -nmax / mx
    L, sumlx, suml2 = fit(iscale)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1), 0)
    best = scale * sumlx
    bestL = L.copy()
    for s in range(-9, 10):
        if s == 0:
            continue
        L, sumlx, suml2 = fit(-(nmax + 0.1 * s) / mx)
        ok = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        sc = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1), 0)
        bestL[ok] = L[ok]
        scale = np.where(ok, sc, scale)
        best = np.where(ok, sc * sumlx, best)
    scale = np.where(zero, 0, scale).astype(np.float32)
    bestL = np.where(zero[:, None], 0, bestL)
    return (bestL + nmax).astype(np.uint8), scale


def imat(x, qw):
    """quantize_row_q5_0_impl with importance weights qw [n]: x float32 [rows, n] -> codes, d (float16)"""
    r, n = x.shape
    sigma2 = (x.astype(np.float64) ** 2).sum(axis=1) / n
    codes = np.empty((r, n), np.uint8)
    d = np.empty((r, n // QK), np.float16)
    for b in range(n // QK):
        xb = x[:, b * QK:(b + 1) * QK]
        w = (qw[b * QK:(b + 1) * QK][None, :] * np.sqrt(sigma2[:, None] + xb.astype(np.float64) ** 2)).astype(np.float32)
        codes[:, b * QK:(b + 1) * QK], sc = make_qx_quants(xb, w)
        d[:, b] = sc.astype(np.float16)
    return codes, d


# ---- GPTQ ----

def group_scale(g, w, mode):
    """g float32 [32, rows] (one group, transposed), w = diag(H) for its columns -> d float32 [rows] (fp16-exact)"""
    if mode == 'max':
        i = np.abs(g).argmax(axis=0)
        d = g[i, np.arange(g.shape[1])] / np.float32(-16)
    else:
        _, d = make_qx_quants(np.ascontiguousarray(g.T), np.broadcast_to(w[None, :], (g.shape[1], QK)).astype(np.float32))
    return d.astype(np.float16).astype(np.float32)


def quant_col(w, d):
    """nearest Q5_0 code for each row's weight w with that row's scale d -> codes uint8, dequantized float32"""
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.floor(w / d + np.float32(16.5))
    q = np.where(d != 0, np.clip(q, 0, 31), 16).astype(np.uint8)
    return q, (q.astype(np.float32) - 16) * d


def gptq_rows(Wt, U, hdiag, scale_mode, perm=None, d_fixed=None):
    """GPTQ on Wt float32 [n, rows] (transposed head rows, modified in place), U = upper Cholesky factor of H^-1 in
    processing order, perm = processing order (None = column order), d_fixed = scales [rows, n/32] to use instead of
    choosing them -> codes uint8 [rows, n], d float16 [rows, n/32]"""
    n, r = Wt.shape
    codes = np.empty((n, r), np.uint8)
    d = np.empty((n // QK, r), np.float32)
    if d_fixed is not None:
        d[:] = d_fixed.T
    elif perm is not None:
        # act-order: scales fixed from the original weights, in the original column order
        for b in range(n // QK):
            d[b] = group_scale(Wt[b * QK:(b + 1) * QK], hdiag[b * QK:(b + 1) * QK], scale_mode)
    if perm is not None:
        Wt = Wt[perm]
        dcol = d[perm // QK]
    for i1 in range(0, n, QK):
        i2 = i1 + QK
        W1 = Wt[i1:i2]
        if perm is None and d_fixed is None:
            d[i1 // QK] = group_scale(W1, hdiag[i1:i2], scale_mode)
        E = np.empty((QK, r), np.float32)
        for i in range(QK):
            dc = d[i1 // QK] if perm is None else dcol[i1 + i]
            q, deq = quant_col(W1[i], dc)
            codes[i1 + i] = q
            e = (W1[i] - deq) / U[i1 + i, i1 + i]
            W1[i + 1:] -= U[i1 + i, i1 + i + 1:i2][:, None] * e[None, :]
            E[i] = e
        if i2 < n:
            Wt[i2:] -= U[i1:i2, i2:].T @ E
    if perm is not None:
        out = np.empty_like(codes)
        out[perm] = codes
        codes = out
    return np.ascontiguousarray(codes.T), np.ascontiguousarray(d.T).astype(np.float16)


def refit_scales(W, codes, H, sub=256):
    """least-squares block scales for fixed codes: per row, d minimizing (w - Q d)^T H (w - Q d), where column g of Q
    holds block g's codes - 16 -> d float32 [rows, n/32], rounded to float16"""
    r, n = W.shape
    G = n // QK
    out = np.empty((r, G), np.float32)
    Hg = [np.ascontiguousarray(H[:, g * QK:(g + 1) * QK].T) for g in range(G)]   # [32, n] per block
    for a in range(0, r, sub):
        Q = codes[a:a + sub].astype(np.float32) - 16                                # [s, n]
        s = len(Q)
        T = np.empty((s, n, G), np.float32)                                         # T[r, i, h] = sum_{j in h} H_ij Q_rj
        for h in range(G):
            T[:, :, h] = Q[:, h * QK:(h + 1) * QK] @ Hg[h]
        A = np.einsum('rgi,rgih->rgh', Q.reshape(s, G, QK), T.reshape(s, G, QK, G))
        b = (Q * (W[a:a + sub] @ H)).reshape(s, G, QK).sum(axis=2)
        A += (1e-9 * np.trace(A, axis1=1, axis2=2) / G + 1e-30)[:, None, None] * np.eye(G, dtype=np.float32)
        out[a:a + sub] = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return out.astype(np.float16).astype(np.float32)


def objective(W, Wq, H):
    """(w - wq)^T H (w - wq) summed over rows, relative to w^T H w"""
    D = Wq - W
    return float(((D @ H) * D).sum(dtype=np.float64) / ((W @ H) * W).sum(dtype=np.float64))


# ---- files ----

def head_tensor(path, name='output.weight'):
    import gguf
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.name == name)
    return r, t


def head_f32(t, rows=None):
    """rows of a BF16/F16/F32/Q8_0 tensor as float32"""
    import gguf
    data = t.data if rows is None else t.data[rows]
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        u = np.ascontiguousarray(data).view(np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32)
    from gguf.quants import dequantize
    return dequantize(np.ascontiguousarray(data), t.tensor_type)


def load_x(path):
    x = np.memmap(path, np.float32, 'r')
    idx = np.fromfile(path + '.idx', np.int32).reshape(-1, 3)
    n_embd = x.size // len(idx)
    return x.reshape(len(idx), n_embd), idx


# ---- workers for quant ----

_W = {}


def _init(src, tensor, hpath, damp, mode, scale_mode, act_order, refit):
    _W['t'] = head_tensor(src, tensor)[1]
    _W['mode'], _W['scale'], _W['refit'] = mode, scale_mode, refit
    if hpath:
        H = np.load(hpath)
        n = H.shape[0]
        dead = np.diag(H) == 0
        H[dead, dead] = 1
        _W['H'] = H.astype(np.float32)
        hdiag = np.diag(H).copy()
        H += damp * np.mean(np.diag(H)) * np.eye(n)
        perm = np.argsort(-hdiag) if act_order else None
        if perm is not None:
            H = H[perm][:, perm]
        L = np.linalg.cholesky(H)
        Linv = np.linalg.inv(L)
        Hinv = Linv.T @ Linv
        U = np.linalg.cholesky(Hinv).T          # upper factor, Hinv = U^T U
        _W['U'], _W['hdiag'], _W['perm'] = U.astype(np.float32), hdiag.astype(np.float32), perm


def _work(rng):
    a, b = rng
    W = head_f32(_W['t'], slice(a, b))
    mode = _W['mode']
    if mode == 'rtn':
        codes, d = rtn_ref(W)
    elif mode == 'imat':
        codes, d = imat(W, _W['hdiag'])
    else:
        codes, d = gptq_rows(np.ascontiguousarray(W.T), _W['U'], _W['hdiag'], _W['scale'], _W['perm'])
        for k in range(_W['refit']):
            # least-squares scales for these codes; between refits, GPTQ codes for the new scales
            d = refit_scales(W, codes, _W['H'])
            if k + 1 < _W['refit']:
                codes, d = gptq_rows(np.ascontiguousarray(W.T), _W['U'], _W['hdiag'], _W['scale'], _W['perm'], d)
            d = d.astype(np.float16)
    return a, pack(codes, d)


def cmd_quant(a):
    import multiprocessing as mp
    t = head_tensor(a.src, a.tensor)[1]
    rows, n = int(t.shape[1]), int(t.shape[0])
    chunks = [(i, min(i + a.chunk, rows)) for i in range(0, rows, a.chunk)]
    out = np.memmap(a.out, np.uint8, 'w+', shape=(rows, n // QK * BLOCK.itemsize))
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, _init, (a.src, a.tensor, a.hessian, a.damp, a.mode, a.scale, a.act_order, a.refit)) as pool:
        for k, (i, blocks) in enumerate(pool.imap_unordered(_work, chunks)):
            out[i:i + len(blocks)] = blocks
            if k % 8 == 0:
                print(f'{k + 1}/{len(chunks)} chunks', file=sys.stderr, flush=True)
    out.flush()
    print(f'{a.out}: {rows} rows, mode {a.mode}')


def cmd_hessian(a):
    H, cnt = None, 0
    for p in a.x:
        x, idx = load_x(p)
        keep = np.flatnonzero(idx[:, 1] > 0)
        for i in range(0, len(keep), 16384):
            xc = np.asarray(x[keep[i:i + 16384]], np.float64)
            H = xc.T @ xc if H is None else H + xc.T @ xc
        cnt += len(keep)
        print(f'{p}: {len(keep)} rows', file=sys.stderr)
    np.save(a.out, H / cnt)
    dg = np.diag(H / cnt)
    print(f'{a.out}: {cnt} rows; diag(H) mean {dg.mean():.4g}, max {dg.max():.4g} (dim {dg.argmax()}), '
          f'min {dg.min():.4g}; top 8 dims {np.argsort(-dg)[:8].tolist()}')


def cmd_proxy(a):
    """KLD of the head alone: logits from the reference head vs each candidate head, on held-out head inputs"""
    r, t = head_tensor(a.ref)
    rows, n = int(t.shape[1]), int(t.shape[0])
    Wref = head_f32(t)
    x, idx = load_x(a.x)
    sel = np.flatnonzero(idx[:, 1] >= a.min_pos)[:a.rows]
    X = np.asarray(x[sel], np.float32)
    print(f'{len(sel)} held-out rows (positions >= {a.min_pos})')
    cands = []
    for c in a.cand:
        if c == 'bf16':
            Wc = Wref
        elif ':' in c:
            Wc = head_f32(head_tensor(c.split(':', 1)[1])[1])
        else:
            Wc = unpack(np.fromfile(c, np.uint8).reshape(rows, -1), n)
        err = np.sqrt(((Wc - Wref).astype(np.float64) ** 2).sum() / (Wref.astype(np.float64) ** 2).sum())
        cands.append(dict(name=os.path.basename(c), W=Wc, err=err, kl=0.0, top=0, se=0.0))

    def logp(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    ss = 0.0
    for i in range(0, len(X), 512):
        zr = X[i:i + 512] @ Wref.T
        ss += (zr.astype(np.float64) ** 2).sum()
        lr = logp(zr)
        pr = np.exp(lr)
        tr = zr.argmax(axis=1)
        for c in cands:
            zc = X[i:i + 512] @ c['W'].T
            c['se'] += ((zc - zr).astype(np.float64) ** 2).sum()
            c['kl'] += (pr * (lr - logp(zc))).sum(dtype=np.float64)
            c['top'] += (tr == zc.argmax(axis=1)).sum()
    for c in cands:
        print(f"{c['name']}\tkld={c['kl'] / len(X):.5f}\tsame_top={100 * c['top'] / len(X):.2f}%\t"
              f"rel_rms_logits={np.sqrt(c['se'] / ss):.5f}\trel_rms_w={c['err']:.5f}", flush=True)


def cmd_checkdump(a):
    """the dump is right if the dumped rows times the model's own head give the logits the model computed"""
    t = head_tensor(a.model)[1]
    W = head_f32(t)
    x, idx = load_x(a.x)
    rows = np.fromfile(a.x + '.logits.idx', np.int64)
    L = np.fromfile(a.x + '.logits', np.float32).reshape(len(rows), -1)
    Z = np.asarray(x[rows], np.float32) @ W.T
    def logp(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    kl = (np.exp(logp(L)) * (logp(L) - logp(Z))).sum(axis=1)
    rel = np.sqrt(((Z - L) ** 2).sum(axis=1) / (L ** 2).sum(axis=1))
    print(f'{len(rows)} rows ({t.tensor_type.name} head): rel rms diff median {np.median(rel):.2e} max {rel.max():.2e}; '
          f'max abs diff {np.abs(Z - L).max():.3g}; same top token {100 * (Z.argmax(1) == L.argmax(1)).mean():.1f} %; '
          f'KLD mean {kl.mean():.2e} max {kl.max():.2e}')
    # a dump shifted by one row would match its neighbour instead: show that it does not
    sh = np.asarray(x[np.clip(rows - 1, 0, None)], np.float32) @ W.T
    rel_sh = np.sqrt(((sh - L) ** 2).sum(axis=1) / (L ** 2).sum(axis=1))
    print(f'control, rows shifted by one: rel rms diff median {np.median(rel_sh):.2e}')
    # the model computes its logits from 8-bit-quantized activations, so a correct dump agrees to ~1 %, not exactly
    ok = np.median(rel) < 0.03 and (Z.argmax(1) == L.argmax(1)).mean() >= 0.97 and np.median(rel_sh) > 10 * np.median(rel)
    print('dump OK' if ok else 'DUMP DOES NOT MATCH THE MODEL')
    sys.exit(0 if ok else 1)


def _writer(r, path):
    import gguf
    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    w = gguf.GGUFWriter(path, arch=arch, endianess=r.endianess)
    return w, gguf


def _copy_kv(r, w, gguf):
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name.startswith('split.'):
            continue
        vt = field.types[0]
        w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)


def _split_kv(w, no, count, n_tensors):
    w.add_uint16('split.no', no)
    w.add_uint16('split.count', count)
    w.add_int32('split.tensors.count', n_tensors)


def _head_info(w, gguf, raw):
    w.add_tensor_info('output.weight', raw.shape, np.dtype(np.uint8), raw.nbytes, gguf.GGMLQuantizationType.Q5_0)


def cmd_base(a):
    import gguf
    r = gguf.GGUFReader(a.src)
    w, gguf = _writer(r, a.out)
    _copy_kv(r, w, gguf)
    _split_kv(w, 0, 2, len(r.tensors))
    plan = [t for t in r.tensors if t.name != 'output.weight']
    for t in plan:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in plan:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f'{a.out}: {len(plan)} tensors')


def cmd_shard(a):
    import gguf
    r = gguf.GGUFReader(a.base)
    raw = np.fromfile(a.head, np.uint8)
    raw = raw.reshape(raw.size // (a.n_embd // QK * BLOCK.itemsize), -1)
    os.makedirs(a.dir, exist_ok=True)
    p1 = os.path.join(a.dir, f'{a.name}-00001-of-00002.gguf')
    p2 = os.path.join(a.dir, f'{a.name}-00002-of-00002.gguf')
    if os.path.lexists(p1):
        os.remove(p1)
    os.symlink(os.path.relpath(a.base, a.dir), p1)
    w, gguf = _writer(r, p2)
    _split_kv(w, 1, 2, r.get_field('split.tensors.count').contents())
    _head_info(w, gguf, raw)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    w.write_tensor_data(raw)
    w.close()
    print(p1, '->', a.base)
    print(p2)


def cmd_full(a):
    import gguf
    r = gguf.GGUFReader(a.src)
    w, gguf = _writer(r, a.out)
    _copy_kv(r, w, gguf)
    t0 = next(t for t in r.tensors if t.name == 'output.weight')
    raw = np.fromfile(a.head, np.uint8).reshape(int(t0.shape[1]), -1)
    for t in r.tensors:
        if t.name == 'output.weight':
            _head_info(w, gguf, raw)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(raw if t.name == 'output.weight' else t.data, tensor_endianess=r.endianess)
    w.close()
    print(f'{a.out}: output.weight -> Q5_0 from {a.head}')


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('hessian'); q.add_argument('out'); q.add_argument('x', nargs='+')
    q = s.add_parser('quant'); q.add_argument('src'); q.add_argument('out')
    q.add_argument('--mode', choices=['rtn', 'imat', 'gptq'], required=True); q.add_argument('--hessian')
    q.add_argument('--damp', type=float, default=0.01); q.add_argument('--scale', choices=['max', 'search'], default='max')
    q.add_argument('--act-order', action='store_true'); q.add_argument('--refit', type=int, default=0)
    q.add_argument('--workers', type=int, default=8)
    q.add_argument('--chunk', type=int, default=2048); q.add_argument('--tensor', default='output.weight')
    q = s.add_parser('proxy'); q.add_argument('ref'); q.add_argument('x'); q.add_argument('cand', nargs='+')
    q.add_argument('--rows', type=int, default=10240); q.add_argument('--min-pos', type=int, default=256)
    q = s.add_parser('checkdump'); q.add_argument('model'); q.add_argument('x')
    q = s.add_parser('base'); q.add_argument('src'); q.add_argument('out')
    q = s.add_parser('shard'); q.add_argument('base'); q.add_argument('head'); q.add_argument('dir'); q.add_argument('name')
    q.add_argument('--n-embd', type=int, default=2048)
    q = s.add_parser('full'); q.add_argument('src'); q.add_argument('head'); q.add_argument('out')
    a = p.parse_args()
    if a.cmd == 'quant' and a.mode != 'rtn' and not a.hessian:
        sys.exit('--hessian is needed for imat and gptq')
    globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
