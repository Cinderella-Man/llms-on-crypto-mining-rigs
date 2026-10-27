#!/usr/bin/env python3
# A CPU-only quality proxy for 2- and 3-bit weight formats: it quantizes a sample of a model's routed-expert matrices
# with llama.cpp's own quantizers (through libggml-base, importance matrix included) and with candidate formats written
# here in numpy, and reports for every format the importance-weighted relative reconstruction error
#
#   E = sum over rows and columns of imp_j * (w_ij - q_ij)^2  /  sum of imp_j * w_ij^2
#
# with imp_j the importance matrix's mean squared activation of input column j for that expert. If the inputs of a
# matrix were uncorrelated, E would be the expected squared error of its output relative to the output's energy. It
# ranks formats on the same weights; it does not replace a KL-divergence test of a whole quantized model.
#
# Every quantizer, llama.cpp's and the candidates, weighs column j of a row by imp_j * sqrt(sigma2 + w_j^2), sigma2 =
# the row's mean w^2, as llama.cpp's quantizers do, and the error is measured with imp_j alone.
#
# Candidate formats (names as in --formats):
#   u<b>s_<B>       b-bit uniform levels without zero (odd multiples of d/2), one FP16 scale per B weights
#   u<b>a_<B>       b-bit uniform levels with zero (-2^(b-1) .. 2^(b-1)-1 times d), one FP16 scale per B weights
#   u2m_32          2-bit uniform levels d*i + m with an FP16 scale and an FP16 offset per 32 weights (Q4_1-like)
#   nl<b>_<B>       b-bit non-uniform levels from one int8 table (fitted here, see --fit-layer), FP16 scale per B
#   <base>_s<S>x<G> the same levels in 256-weight super-blocks: one FP16 scale per super-block and an S-bit integer
#                   scale per G weights (signed for tables with zero or non-uniform levels, unsigned otherwise)
#   ik<b>_...       ik_llama.cpp's IQ2_K / IQ3_K level tables (iq2nl_values, iq3nl_values) scaled to int8
#   nl<b>x_..., ik<b>x_...  one extra bit per sub-block selects the table shifted up by a constant (ik_llama.cpp's
#                   "extra" bit: +5 of 31 for 2 bits, +4 of 63 for 3 bits); costs 1 bit per sub-block
#   <fmt>+r<R>      the format applied after a random-sign Hadamard rotation of each row in blocks of R columns; the
#                   error is measured after rotating back (a run-time rotation of the matrix's input undoes it)
# Reference formats: any llama.cpp type name (Q4_0, Q2_K, Q3_K, IQ2_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ3_S, IQ4_XS, Q2_0).
#
#   lowbit_proxy.py --model <safetensors dir> --imatrix <imatrix.gguf> --lib <libggml-base.so> --out <results.tsv>
#                   [--layers 2,10,18,26,34,42,47] [--experts 4] [--fit-layer 6] [--formats ...] [--jobs 4]
# Needs numpy only. The safetensors reader handles fused expert tensors [experts, rows, columns] in BF16
# (Qwen3.5-MoE style: experts.gate_up_proj and experts.down_proj) and reads only the sampled experts' bytes.
import argparse, ctypes, json, math, os, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REF_TYPES = {'Q4_0': 2, 'Q4_1': 3, 'Q5_0': 6, 'Q8_0': 8, 'Q2_K': 10, 'Q3_K': 11, 'Q4_K': 12, 'IQ2_XXS': 16, 'IQ2_XS': 17,
             'IQ3_XXS': 18, 'IQ3_S': 21, 'IQ2_S': 22, 'IQ4_XS': 23, 'Q2_0': 42}

DEFAULT_FORMATS = (
    'Q4_0,IQ4_XS,Q3_K,IQ3_S,IQ3_XXS,Q2_K,IQ2_S,IQ2_XS,IQ2_XXS,Q2_0,u4a_32,'
    'u3s_32,u3a_32,u3s_s6x32,u3a_s6x32,u3a_s4x16,u3a_s6x16,'
    'nl3_32,nl3_s6x32,nl3_s4x32,nl3_s4x16,nl3_s6x16,'
    'u2s_s4x16,u2a_s4x16,u2a_s6x16,u2s_s4x8,u2a_s4x8,'
    'nl2_s4x16,nl2_s6x16,nl2_s4x8,nl2_s6x8,'
    'nl3_s6x32+r256')


# ---------------------------------------------------------------- inputs

def st_tensor_index(model_dir):
    """tensor name -> (shard path, dtype, shape, absolute byte offset)"""
    wm = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))['weight_map']
    out, headers = {}, {}
    for name, shard in wm.items():
        if shard not in headers:
            p = os.path.join(model_dir, shard)
            with open(p, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                headers[shard] = (p, 8 + n, json.loads(f.read(n)))
        p, base, h = headers[shard]
        t = h[name]
        out[name] = (p, t['dtype'], t['shape'], base + t['data_offsets'][0])
    return out


def st_read_expert(entry, e, rows=None):
    """Expert e of a fused [experts, rows, cols] BF16 tensor as float32 [rows, cols]; rows=(r0, r1) reads a slice."""
    p, dtype, shape, off = entry
    assert dtype == 'BF16' and len(shape) == 3
    E, R, C = shape
    r0, r1 = rows if rows else (0, R)
    start = off + (e * R + r0) * C * 2
    u = np.fromfile(p, dtype=np.uint16, count=(r1 - r0) * C, offset=start)
    return (u.astype(np.uint32) << 16).view(np.float32).reshape(r1 - r0, C)


def gguf_tensors(path):
    """name -> numpy array for every F32 tensor of a GGUF file (enough for llama.cpp's imatrix.gguf)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, 'rb') as f:
        magic, ver, nt, nkv = struct.unpack('<4sIQQ', f.read(24))
        assert magic == b'GGUF'
        rs = lambda: f.read(struct.unpack('<Q', f.read(8))[0]).decode()
        align = 32
        for _ in range(nkv):
            k = rs()
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                rs()
            elif t == 9:
                at, n = struct.unpack('<IQ', f.read(12))
                for _ in range(n):
                    rs() if at == 8 else f.read(sizes[at])
            else:
                v = f.read(sizes[t])
                if k == 'general.alignment':
                    align = struct.unpack('<I', v)[0]
        infos = []
        for _ in range(nt):
            n = rs()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            t, o = struct.unpack('<IQ', f.read(12))
            infos.append((n, dims, t, o))
        data = (f.tell() + align - 1) // align * align
    out = {}
    for n, dims, t, o in infos:
        assert t == 0, (n, t)
        cnt = int(np.prod(dims))
        out[n] = np.fromfile(path, dtype=np.float32, count=cnt, offset=data + o).reshape(tuple(reversed(dims)))
    return out


def expert_importance(im, gguf_name, e):
    s = im[gguf_name + '.in_sum2']          # [experts, columns]
    c = im[gguf_name + '.counts'].reshape(-1)
    if c[e] > 0:
        return s[e] / c[e], float(c[e])
    tot = c.sum()
    return s.sum(0) / max(tot, 1.0), 0.0


# ---------------------------------------------------------------- llama.cpp reference quantizers

class TypeTraits(ctypes.Structure):
    _fields_ = [('type_name', ctypes.c_char_p), ('blck_size', ctypes.c_int64), ('blck_size_interleave', ctypes.c_int64),
                ('type_size', ctypes.c_size_t), ('is_quantized', ctypes.c_bool), ('to_float', ctypes.c_void_p),
                ('from_float_ref', ctypes.c_void_p)]


class InitParams(ctypes.Structure):
    _fields_ = [('mem_size', ctypes.c_size_t), ('mem_buffer', ctypes.c_void_p), ('no_alloc', ctypes.c_bool)]


_LIB = None


def ggml_lib(path):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(path)
        lib.ggml_init.argtypes = [InitParams]
        lib.ggml_init.restype = ctypes.c_void_p
        lib.ggml_init(InitParams(1 << 20, None, True))
        fp = ctypes.POINTER(ctypes.c_float)
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, fp, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                            ctypes.c_int64, fp]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
        _LIB = lib
    return _LIB


def ref_quantize(lib_path, tname, W, imp):
    lib = ggml_lib(lib_path)
    t = REF_TYPES[tname]
    R, C = W.shape
    src = np.ascontiguousarray(W, dtype=np.float32)
    rs = lib.ggml_row_size(t, C)
    dst = np.zeros(rs * R, dtype=np.uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    iw = np.ascontiguousarray(imp, dtype=np.float32)
    n = lib.ggml_quantize_chunk(t, src.ctypes.data_as(fp), dst.ctypes.data_as(ctypes.c_void_p), 0, R, C,
                                iw.ctypes.data_as(fp))
    assert n == rs * R
    tr = lib.ggml_get_type_traits(t).contents
    to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, fp, ctypes.c_int64)(tr.to_float)
    out = np.zeros(R * C, dtype=np.float32)
    to_float(dst.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(fp), R * C)
    return out.reshape(R, C), 8.0 * rs / C


# ---------------------------------------------------------------- candidate formats

IK_TABLES = {2: [-31, -13, 1, 17], 3: [-63, -40, -23, -10, 1, 13, 28, 47]}   # ik_llama.cpp iq2nl/iq3nl_values
IK_SHIFT = {2: 5, 3: 4}


def parse_format(name):
    f = {'name': name, 'rot': 0, 'xbit': False}
    if '+r' in name:
        name, r = name.split('+r')
        f['rot'] = int(r)
    base, *rest = name.split('_')
    if base.endswith('x'):
        f['xbit'] = True
        base = base[:-1]
    kind = base[:2] if base[:2] in ('nl', 'ik') else base[0] + base[-1]   # 'nl', 'ik', 'us', 'ua', 'um'
    f['bits'] = int(base[2]) if kind in ('nl', 'ik') else int(base[1])
    f['kind'] = {'nl': 'nl', 'ik': 'ik', 'us': 'sym', 'ua': 'asym', 'um': 'min'}[kind]
    spec = rest[0]
    if spec.startswith('s'):
        sb, g = spec[1:].split('x')
        f.update(block=int(g), super=256, sbits=int(sb))
    else:
        f.update(block=int(spec), super=0, sbits=0)
    return f


def format_bpw(f):
    b = f['bits']
    x = 1 if f['xbit'] else 0
    if f['super']:
        return (16 + (f['super'] // f['block']) * (f['sbits'] + x) + f['super'] * b) / f['super']
    return (16 * (2 if f['kind'] == 'min' else 1) + f['block'] * b + x) / f['block']


def levels_of(f, tables):
    K = 1 << f['bits']
    if f['kind'] == 'sym':
        return np.arange(-(K - 1), K, 2, dtype=np.float32)
    if f['kind'] == 'asym':
        return np.arange(-K // 2, K // 2, dtype=np.float32)
    if f['kind'] == 'nl':
        return np.array(tables['nl%d' % f['bits']], dtype=np.float32)
    if f['kind'] == 'ik':
        return np.array(IK_TABLES[f['bits']], dtype=np.float32) * (4 if f['bits'] == 2 else 2)
    return np.arange(K, dtype=np.float32)


def shifted_levels(f, L):
    """The second table of an extra-bit format: ik_llama.cpp's shift, in the table's own int8 units."""
    unit = (4 if f['bits'] == 2 else 2) if f['kind'] == 'ik' else (np.abs(L).max() / (31 if f['bits'] == 2 else 63))
    return L + np.float32(np.rint(IK_SHIFT[f['bits']] * unit))


def nearest(t, L):
    step = L[1] - L[0]
    if np.all(np.diff(L) == step):   # uniform levels: arithmetic instead of a search
        return np.clip(np.rint((t - L[0]) / step), 0, L.size - 1).astype(np.int8)
    mid = (L[1:] + L[:-1]) * 0.5
    return np.searchsorted(mid, t).astype(np.int8)


def fp16(x):
    return x.astype(np.float16).astype(np.float32)


def fit_scales(X, Wt, L, signed, grid=np.linspace(0.55, 1.2, 14)):
    """Best scale per block (rows of X) for levels L: grid over max|x| / max|L| * g (and its negative for asymmetric
    levels), each refined by weighted least squares on its own assignment. Returns float scales [nb]."""
    amax = np.abs(X).max(1)
    amax[amax == 0] = 1e-30
    lm = np.abs(L).max()
    best_s = np.zeros(X.shape[0], np.float32)
    best_e = np.full(X.shape[0], np.inf, np.float32)
    signs = (1.0, -1.0) if signed else (1.0,)
    for sg in signs:
        for g in grid:
            s = sg * amax / lm * g
            q = L[nearest(X / s[:, None], L)]
            num = (Wt * X * q).sum(1)
            den = (Wt * q * q).sum(1)
            so = np.where(den > 0, num / np.maximum(den, 1e-30), s)
            e = (Wt * (X - so[:, None] * q) ** 2).sum(1)
            m = e < best_e
            best_e[m], best_s[m] = e[m], so[m]
    q = L[nearest(X / best_s[:, None], L)]
    num, den = (Wt * X * q).sum(1), (Wt * q * q).sum(1)
    so = np.where(den > 0, num / np.maximum(den, 1e-30), best_s)
    e = (Wt * (X - so[:, None] * q) ** 2).sum(1)
    m = e < best_e
    best_s[m] = so[m]
    return best_s


def assign(X, S, L):
    """Quantized values for fixed scales S [nb] (zero scale -> zero)."""
    Sx = np.where(S == 0, 1.0, S)
    q = L[nearest(X / Sx[:, None], L)] * S[:, None]
    return q


def quant_minmax(X, Wt, K):
    """d*i + m with i in 0..K-1 per block, FP16 d and m (Q4_1-like), weighted least squares over a grid of starts."""
    xmin, xmax = X.min(1), X.max(1)
    rng = np.maximum(xmax - xmin, 1e-30)
    best_e = np.full(X.shape[0], np.inf, np.float32)
    best_d, best_m = np.zeros_like(xmin), xmin.copy()
    for g in np.linspace(0.7, 1.15, 10):
        d = rng / (K - 1) * g
        for it in range(3):
            i = np.clip(np.rint((X - xmin[:, None] if it == 0 else X - m[:, None]) / d[:, None]), 0, K - 1)
            sw, swi, swii = Wt.sum(1), (Wt * i).sum(1), (Wt * i * i).sum(1)
            swx, swix = (Wt * X).sum(1), (Wt * i * X).sum(1)
            det = sw * swii - swi * swi
            ok = det > 0
            dn = np.where(ok, (sw * swix - swi * swx) / np.where(ok, det, 1), d)
            mn = np.where(ok, (swii * swx - swi * swix) / np.where(ok, det, 1), xmin)
            d, m = np.where(dn > 0, dn, d), mn
        e = (Wt * (X - (d[:, None] * i + m[:, None])) ** 2).sum(1)
        b = e < best_e
        best_e[b], best_d[b], best_m[b] = e[b], d[b], m[b]
    d, m = fp16(best_d), fp16(best_m)
    i = np.clip(np.rint((X - m[:, None]) / np.where(d == 0, 1, d)[:, None]), 0, K - 1)
    return d[:, None] * i + m[:, None]


def quant_candidate(f, W, Wt, tables):
    R, C = W.shape
    B = f['block']
    X = W.reshape(-1, B)
    T = Wt.reshape(-1, B)
    if f['kind'] == 'min':
        return quant_minmax(X, T, 1 << f['bits']).reshape(R, C)
    L = levels_of(f, tables)
    signed = f['kind'] != 'sym'
    s = fit_scales(X, T, L, signed)
    if f['xbit']:
        # per block, the table (plain or shifted) with the lower error at its best scale
        L1 = shifted_levels(f, L)
        s1 = fit_scales(X, T, L1, signed)
        e0 = (T * (X - assign(X, s, L)) ** 2).sum(1)
        e1 = (T * (X - assign(X, s1, L1)) ** 2).sum(1)
        ch = e1 < e0
        s = np.where(ch, s1, s)
        assign_b = lambda Xb, Sb: np.where(ch[:, None], assign(Xb, Sb, L1), assign(Xb, Sb, L))
    else:
        assign_b = lambda Xb, Sb: assign(Xb, Sb, L)
    if not f['super']:
        return assign_b(X, fp16(s)).reshape(R, C)
    # super-blocks: an FP16 scale D per 256 weights and integer sub-scales q_b, scale of block b = D * q_b
    nsub = f['super'] // B
    S = s.reshape(-1, nsub)
    Xs, Ts = X.reshape(-1, nsub, B), T.reshape(-1, nsub, B)
    if signed:
        qlo, qhi = -(1 << (f['sbits'] - 1)), (1 << (f['sbits'] - 1)) - 1
    else:
        S = np.abs(S)
        qlo, qhi = 0, (1 << f['sbits']) - 1
    smax = np.abs(S).max(1)
    smax[smax == 0] = 1e-30
    best_e = np.full(S.shape[0], np.inf, np.float32)
    best_q = np.zeros(Xs.shape, np.float32)
    cands = []
    for g in np.linspace(0.85, 1.15, 7):
        cands.append(smax / qhi * g)
        if signed:   # the largest scale at the negative end, which has one more step
            ext = S[np.arange(S.shape[0]), np.abs(S).argmax(1)]
            cands.append(ext / qlo * g)
    for D in cands:
        D = fp16(D)
        Dn = np.where(D == 0, 1, D)
        qb = np.clip(np.rint(S / Dn[:, None]), qlo, qhi)
        Sb = (D[:, None] * qb).reshape(-1)
        q = assign_b(X, Sb).reshape(Xs.shape)
        e = (Ts * (Xs - q) ** 2).sum((1, 2))
        m = e < best_e
        best_e[m] = e[m]
        best_q[m] = q[m]
    return best_q.reshape(R, C)


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def rotate(W, R, seed=1234):
    """W -> W * (D H) per block of R columns (D random signs): an orthogonal map that the inverse map undoes exactly."""
    C = W.shape[1]
    sign = np.random.default_rng(seed).choice([-1.0, 1.0], C).astype(np.float32)
    H = hadamard(R)
    Wr = (W * sign).reshape(W.shape[0], C // R, R) @ H
    return Wr.reshape(W.shape), sign, H


def unrotate(Q, sign, H):
    R = H.shape[0]
    Qb = Q.reshape(Q.shape[0], -1, R) @ H.T
    return Qb.reshape(Q.shape) * sign


def quant_weights(W, imp):
    sigma2 = (W * W).mean(1, keepdims=True)
    return imp[None, :] * np.sqrt(sigma2 + W * W)


def run_format(fname, W, imp, tables, lib):
    if fname in REF_TYPES:
        Q, bpw = ref_quantize(lib, fname, W, imp)
        return Q, bpw
    f = parse_format(fname)
    if f['rot']:
        Wr, sign, H = rotate(W, f['rot'])
        # after the rotation each column's importance is the mean importance of its block
        impr = np.repeat(imp.reshape(-1, f['rot']).mean(1), f['rot'])
        Qr = quant_candidate(f, Wr, quant_weights(Wr, impr), tables)
        return unrotate(Qr, sign, H), format_bpw(f)
    return quant_candidate(f, W, quant_weights(W, imp), tables), format_bpw(f)


def rel_err(W, Q, imp):
    e = (W - Q) ** 2
    return float((e * imp).sum() / ((W * W) * imp).sum()), float(e.sum() / (W * W).sum())


# ---------------------------------------------------------------- non-uniform tables

def fit_table(mats, bits, iters=8):
    """One int8 table of 2^bits levels for all matrices: alternately fit per-32 scales and move each level to the
    weighted mean of the values assigned to it (Lloyd), starting from uniform levels."""
    K = 1 << bits
    L = np.arange(-(K - 1), K, 2, dtype=np.float32)
    for _ in range(iters):
        num = np.zeros(K)
        den = np.zeros(K)
        for W, imp in mats:
            X = W.reshape(-1, 32)
            T = quant_weights(W, imp).reshape(-1, 32)
            s = fit_scales(X, T, L, True)
            s[s == 0] = 1e-30
            t = X / s[:, None]
            w = T * (s * s)[:, None]
            idx = nearest(t, L)
            for k in range(K):
                m = idx == k
                num[k] += (w[m] * t[m]).sum()
                den[k] += w[m].sum()
        L = np.where(den > 0, num / np.maximum(den, 1e-30), L).astype(np.float32)
        L = np.sort(L)
        L = np.rint(L / np.abs(L).max() * 127).astype(np.float32)
    return [int(v) for v in L]


# ---------------------------------------------------------------- driver

def load_weights(st, layer, e, kind):
    pre = 'model.language_model.layers.%d.mlp.experts.' % layer
    if kind in ('gate', 'up'):
        ent = st[pre + 'gate_up_proj']
        R = ent[2][1] // 2
        return st_read_expert(ent, e, (0, R) if kind == 'gate' else (R, 2 * R))
    return st_read_expert(st[pre + 'down_proj'], e)


def load_importance(im, layer, e, kind):
    imp, cnt = expert_importance(im, 'blk.%d.ffn_%s_exps.weight' % (layer, kind), e)
    return imp.astype(np.float32), cnt


_G = {}


def worker_init(model, lib, tables):
    _G['st'] = st_tensor_index(model)
    _G['lib'] = lib
    _G['tables'] = tables


def work(item):
    layer, e, kind, imp, cnt, formats = item
    W = load_weights(_G['st'], layer, e, kind)
    rows = []
    for fname in formats:
        t0 = time.time()
        Q, bpw = run_format(fname, W, imp, _G['tables'], _G['lib'])
        ew, eu = rel_err(W, Q, imp)
        rows.append((layer, e, kind, int(cnt), fname, bpw, ew, eu, time.time() - t0))
    return rows


def pick_experts(im, layer, n, seed):
    c = im['blk.%d.ffn_gate_exps.weight.counts' % layer].reshape(-1)
    order = np.argsort(-c)
    top = list(order[: n // 2])
    rest = [int(x) for x in order[n // 2:] if c[x] > 0]
    rng = np.random.default_rng(seed + layer)
    return [int(x) for x in top] + [int(x) for x in rng.choice(rest, n - len(top), replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--imatrix', required=True)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='2,10,18,26,34,42,47')
    ap.add_argument('--experts', type=int, default=4)
    ap.add_argument('--kinds', default='gate,up,down')
    ap.add_argument('--fit-layer', type=int, default=6)
    ap.add_argument('--fit-experts', type=int, default=2)
    ap.add_argument('--formats', default=DEFAULT_FORMATS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    st = st_tensor_index(a.model)
    im = gguf_tensors(a.imatrix)
    formats = a.formats.split(',')
    kinds = a.kinds.split(',')

    tables_path = os.path.splitext(a.out)[0] + '.tables.json'
    if os.path.exists(tables_path):
        tables = json.load(open(tables_path))
    else:
        fit = []
        for e in pick_experts(im, a.fit_layer, a.fit_experts, a.seed):
            for kind in kinds:
                W = load_weights(st, a.fit_layer, e, kind)
                imp, _ = load_importance(im, a.fit_layer, e, kind)
                fit.append((W[: 262144 // W.shape[1] * 3], imp))   # a row sample: 768 rows of 1024, 256 of 3072
        t0 = time.time()
        tables = {'nl3': fit_table(fit, 3), 'nl2': fit_table(fit, 2), 'fit_layer': a.fit_layer}
        json.dump(tables, open(tables_path, 'w'))
        print('tables (%.0f s): %s' % (time.time() - t0, tables), flush=True)

    items = []
    for layer in [int(x) for x in a.layers.split(',')]:
        for e in pick_experts(im, layer, a.experts, a.seed):
            for kind in kinds:
                imp, cnt = load_importance(im, layer, e, kind)
                items.append((layer, e, kind, imp, cnt, formats))
    del im
    with open(a.out, 'w') as out, ProcessPoolExecutor(a.jobs, initializer=worker_init,
                                                       initargs=(a.model, a.lib, tables)) as ex:
        out.write('layer\texpert\tkind\tcount\tformat\tbpw\terr_imp\terr_plain\tseconds\n')
        for rows in ex.map(work, items):
            for r in rows:
                out.write('%d\t%d\t%s\t%d\t%s\t%.4f\t%.6g\t%.6g\t%.1f\n' % r)
            out.flush()
            print('%s L%d e%d %s done' % (time.strftime('%H:%M:%S'), rows[0][0], rows[0][1], rows[0][2]), flush=True)


if __name__ == '__main__':
    main()
