#!/usr/bin/env python3
# The Pascal trellis proxy (registry Q929): which trellis code, built from operations a Pascal GPU does cheaply, holds
# the most quality per bit on Swift-1.5's routed-expert weights. It extends Q164's proxy (scripts/lowbit_proxy.py,
# imported from --lowbit): the same metric, the importance-weighted relative reconstruction error
#
#   E = sum over rows and columns of imp_j * (w_ij - q_ij)^2  /  sum of imp_j * w_ij^2,
#
# the same quantizer weights imp_j * sqrt(sigma2 + w_ij^2), and the same llama.cpp reference quantizers through a
# CPU-only libggml-base. New here: trellis codes encoded by a Viterbi search in C (run/ptq.c), codebooks trained on two
# held-out layers, and three files' own bytes read back (Swift-1.5's Q2_0, the reference = 1.0; Gyro-M's TQ2_T; the
# ISTA and Swift IQ3_S pair that prices the checkpoint difference behind Gyro's bytes).
#
# Trellis codes (128-weight blocks, one FP16 scale; a tail-biting bitshift trellis, L-bit state, kV new bits a step):
#   tq2t                Gyro's TQ2_T: L 16, 8 bits a step, 4 weights from two hashed 11-bit indices of a fixed 2,048-point
#                       2-D codebook (agentionai's FP16 table); tq2t.i8: their int8 copy of it (the CUDA kernels' table)
#   lut4.L<L>           4 weights a step, 8 bits a step, the L-bit state indexes a 2^L-entry 4-D codebook directly
#   hash4.n<N>          L 16, 8 bits a step, one 16-bit multiply hashes the state into an N-entry 4-D codebook
#   lut2.L<L>           2 weights a step, 4 bits a step, the state indexes a 2^L-entry 2-D codebook
#   hp4.n<n>            L 16, 8 bits a step, 4 weights: one 16-bit multiply (XMAD) of the state, folded, gives four 3-bit
#                       fields that pick each weight from n = 8 int8 levels held in two registers (a byte permute); n 16:
#                       4-bit fields, two permutes; hp4m: Gyro's 32-bit multiply instead (top 16 bits)
#   tcq8.L<L>           1 weight a step, 2 bits a step: 8 levels, the 2 new bits pick one of 4 and a parity of older state
#                       bits picks which 4 (trellis-coded quantization); 4 chains of 32 weights (stride 4) per block
#   tcq8u.L<L>          the same with 8 uniform levels (arithmetic decode, no table)
#   tcq16.L<L>          16 levels, two parity bits
#   .k25                2.5 bits a weight (10 bits a step for 4 weights, 5 for 2)
# Suffixes: .i8 int8 codebook entries (one global scale); +r128 a random-sign Hadamard rotation of each row in blocks of
# 128 columns before the code, the error measured after rotating back (the Gyro files do this at run time).
#
#   ptq_proxy.py --model <Swift-1.5 BF16 dir> --imatrix <FN4 imatrix.gguf> --lowbit <lowbit_proxy.py> --ptq <libptq.so>
#                --lib <libggml-base.so> [--lib-n <libggml-base.so with N2P/N3P>] --gyro-src <agentionai llama.cpp tree>
#                [--q2-0 <Swift Q2_0 shard 1>] [--gyro-m <Gyro-M gguf>] [--iq3s-ista <shard 1>] [--iq3s-swift <shard 1>]
#                --out <results.tsv> [--codes ...] [--refs ...] [--jobs 8]
import argparse, ctypes, glob, importlib.util, json, math, os, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

QB = 128   # weights per block, every trellis code

# ---------------------------------------------------------------- GGUF tensors (quantized, any shard)

GG_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_index(first_shard):
    """name -> (path, type, dims, absolute offset) over every shard of a split GGUF (or a single file)."""
    m = re.match(r'(.*-)(\d{5})-of-(\d{5})\.gguf$', first_shard)
    paths = [first_shard] if not m else ['%s%05d-of-%s.gguf' % (m.group(1), i, m.group(3))
                                         for i in range(1, int(m.group(3)) + 1)]
    out = {}
    for p in paths:
        with open(p, 'rb') as f:
            rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            rs = lambda: f.read(rd('<Q')).decode('utf-8', 'replace')
            assert f.read(4) == b'GGUF'
            rd('<I'); nt = rd('<Q'); nkv = rd('<Q')
            align = 32

            def skip(t):
                if t == 8:
                    rs()
                elif t == 9:
                    at, n = rd('<I'), rd('<Q')
                    for _ in range(n):
                        skip(at)
                else:
                    return f.read(GG_SIZES[t])
            for _ in range(nkv):
                k = rs(); t = rd('<I'); v = skip(t)
                if k == 'general.alignment':
                    align = struct.unpack('<I', v)[0]
            infos = []
            for _ in range(nt):
                n = rs(); nd = rd('<I'); dims = struct.unpack('<%dQ' % nd, f.read(8 * nd)); t, o = struct.unpack('<IQ', f.read(12))
                infos.append((n, t, dims, o))
            base = (f.tell() + align - 1) // align * align
        for n, t, dims, o in infos:
            out[n] = (p, t, dims, base + o)
    return out


def gguf_keys(path, prefix):
    """Header keys starting with prefix (arrays as lists)."""
    out = {}
    with open(path, 'rb') as f:
        rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        rs = lambda: f.read(rd('<Q')).decode('utf-8', 'replace')
        sc = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                at, n = rd('<I'), rd('<Q')
                return [val(at) for _ in range(n)]
            return rd(sc[t])
        assert f.read(4) == b'GGUF'
        rd('<I'); rd('<Q'); nkv = rd('<Q')
        for _ in range(nkv):
            k = rs(); v = val(rd('<I'))
            if k.startswith(prefix):
                out[k] = v
    return out


def gguf_expert_rows(entry, e, rows, row_bytes):
    """Raw bytes of rows (sorted) of expert e of a [cols, rows, experts] tensor, as uint8 [len(rows), row_bytes]."""
    p, t, dims, off = entry
    R = dims[1]
    raw = np.fromfile(p, dtype=np.uint8, count=R * row_bytes, offset=off + e * R * row_bytes).reshape(R, row_bytes)
    return np.ascontiguousarray(raw[rows])


# ---------------------------------------------------------------- Gyro's TQ2_T: codebooks, decoder, state table

def tq2t_tables(src):
    """agentionai's FP16 codebook (2,048 2-D points) from ggml-common.h and the CUDA kernels' int8 copy."""
    txt = open(os.path.join(src, 'ggml/src/ggml-common.h')).read()
    i = txt.index('tq2t_lut_f16, 4096)')
    vals = re.findall(r'0x[0-9a-fA-F]{4}', txt[i:i + 80000])[:4096]
    f16 = np.array([int(v, 16) for v in vals], np.uint16).view(np.float16).astype(np.float32)
    txt = open(os.path.join(src, 'ggml/src/ggml-cuda/tq-lut-i8.cuh')).read()
    sc = re.search(r'#define TQ_LUT_I8_SCALE \(([0-9.]+)f/([0-9.]+)f\)', txt)
    scale = float(sc.group(1)) / float(sc.group(2))
    i = txt.index('tq_lut_i8[2048]')
    vals = np.array([int(v, 16) for v in re.findall(r'0x[0-9a-fA-F]{4}', txt[i:])[:2048]], np.uint16)
    i8 = np.stack([(vals & 0xFF).astype(np.uint8).view(np.int8), (vals >> 8).astype(np.uint8).view(np.int8)], 1)
    return f16, (i8.astype(np.float32) * scale).reshape(-1)


def tq2t_index(s):
    x = ((s & 0x7FFF).astype(np.uint64) * 0x9E3779B1) & 0xFFFFFFFF
    return (x >> 21).astype(np.int64), ((x >> 10) & 2047).astype(np.int64)


def tq2t_dequant(raw, lut):
    """TQ2_T rows (uint8 [rows, nb*34]) -> float32 [rows, nb*128] (the format definition of ggml-common.h)."""
    R = raw.shape[0]
    b = raw.reshape(R, -1, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = b[:, :, 2:].astype(np.uint32)
    s = (np.roll(qs, 1, axis=2) << 8) | qs                      # s_t = qs[(t + 31) % 32] << 8 | qs[t]
    i0, i1 = tq2t_index(s)
    w = np.stack([lut[2 * i0], lut[2 * i0 + 1], lut[2 * i1], lut[2 * i1 + 1]], -1)   # [R, nb, 32, 4]
    return (w * d[:, :, None, None]).reshape(R, -1)


# ---------------------------------------------------------------- the trellis codes

def gauss_ppf(q):
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q else (lo, mid)
    return (lo + hi) / 2


def parity(x):
    x = x.copy()
    p = np.zeros_like(x)
    while np.any(x):
        p ^= x & 1
        x >>= 1
    return p


class Code:
    """A trellis code: tab[v][s] = P[idx[v][s]], entries rounded to int8 when ent == 'i8'."""

    def __init__(self, name, L, kV, V, T, Q, idx, P, train, bpw, perm=None, ent='fp', fixed=None):
        self.name, self.L, self.kV, self.V, self.T, self.Q = name, L, kV, V, T, Q
        self.idx, self.P, self.train, self.bpw, self.ent, self.fixed = idx, P, train, bpw, ent, fixed
        self.perm = np.arange(QB) if perm is None else perm
        self.g0 = 1.0

    def entries(self, P=None):
        P = self.P if P is None else P
        if self.ent == 'i8':
            a = np.abs(P).max()
            return (np.rint(P / a * 127) * (a / 127)).astype(np.float32)
        return P.astype(np.float32)

    def tab(self):
        return np.ascontiguousarray(self.entries()[self.idx], dtype=np.float32)   # [V, 2^L]


def make_code(base, tq_f16=None, tq_i8=None, tcq_masks=None):
    """base: a code name without the .i8 / +r128 suffixes."""
    rng = np.random.default_rng(sum(map(ord, base)))
    k25 = base.endswith('.k25')
    b = base[:-4] if k25 else base
    if b == 'tq2t':
        s = np.arange(1 << 16, dtype=np.int64)
        i0, i1 = tq2t_index(s)
        idx = np.stack([2 * i0, 2 * i0 + 1, 2 * i1, 2 * i1 + 1])
        return Code(base, 16, 8, 4, 32, 1, idx, tq_f16, False, 2.125)
    fam, par = b.split('.')
    if fam == 'lut4':
        L = int(par[1:]); kV = 10 if k25 else 8
        s = np.arange(1 << L)
        idx = np.stack([s * 4 + v for v in range(4)])
        return Code(base, L, kV, 4, 32, 1, idx, rng.standard_normal(4 << L).astype(np.float32), True, kV / 4 + 0.125)
    if fam == 'hash4':
        m = int(round(math.log2(int(par[1:])))); kV = 10 if k25 else 8
        s = np.arange(1 << 16)
        h = ((s * 0x9E37) & 0xFFFF) >> (16 - m)
        idx = np.stack([h * 4 + v for v in range(4)])
        return Code(base, 16, kV, 4, 32, 1, idx, rng.standard_normal(4 << m).astype(np.float32), True, kV / 4 + 0.125)
    if fam in ('hp4', 'hp4m', 'hp4s'):
        # a computed 4-D codebook in registers: the hashed state's four 3-bit (4-bit) fields select int8 levels with
        # one byte permute (two, for 16 levels); hp4: one 16 x 16-bit multiply (XMAD) and a fold, hp4m: Gyro's
        # 32-bit multiply, top half; hp4s: hp4's 3-bit fields, and the hash's top bit switches between two tables
        n = int(par[1:]); kV = 10 if k25 else 8
        s = np.arange(1 << 16, dtype=np.uint64)
        if fam == 'hp4m':
            h = ((s * 0x9E3779B1) & 0xFFFFFFFF) >> 16
        else:
            p = (s * 0x9E37) & 0xFFFFFFFF
            h = (p ^ (p >> 16)) & 0xFFFF
        idx = np.stack([((h >> (4 * v)) & (n - 1)).astype(np.int64) for v in range(4)])
        P = np.array([gauss_ppf((k + 0.5) / n) for k in range(n)], np.float32)
        if fam == 'hp4s':
            idx = idx | (((h >> 15) & 1).astype(np.int64) << 3)[None, :]
            P = np.concatenate([P * 0.9, P * 1.1])
        return Code(base, 16, kV, 4, 32, 1, idx, P, True, kV / 4 + 0.125)
    if fam == 'lut2':
        L = int(par[1:]); kV = 5 if k25 else 4
        s = np.arange(1 << L)
        idx = np.stack([s * 2 + v for v in range(2)])
        return Code(base, L, kV, 2, 64, 1, idx, rng.standard_normal(2 << L).astype(np.float32), True, kV / 2 + 0.125)
    if fam in ('tcq8', 'tcq8u', 'tcq16'):
        L = int(par[1:])
        s = np.arange(1 << L)
        masks = tcq_masks[(fam, L)]
        k = s & 3
        for j, mk in enumerate(masks):
            k = k | (parity(s & mk) << (2 + j))
        perm = np.array([q + 4 * t for q in range(4) for t in range(32)])
        if fam == 'tcq8u':
            c, p = np.arange(8) & 3, np.arange(8) >> 2
            P = (2 * c + p - 3.5).astype(np.float32) / 2.0
            return Code(base, L, 2, 1, 32, 4, k[None, :], P, False, 2.125, perm)
        n = 8 if fam == 'tcq8' else 16
        c, p = np.arange(n) & 3, np.arange(n) >> 2   # TCQ start: supersets interleaved by a fraction of a step
        P = ((2 * c - 3) + (p - (n // 4 - 1) / 2) * (2.0 / (n // 4))).astype(np.float32) / 2.0
        return Code(base, L, 2, 1, 32, 4, k[None, :], P, True, 2.125, perm)
    raise ValueError(base)


def split_name(name):
    rot = name.endswith('+r128')
    n = name[:-5] if rot else name
    i8 = n.endswith('.i8')
    return (n[:-3] if i8 else n), i8, rot


# ---------------------------------------------------------------- encoding through libptq

_PTQ = None


def ptq_lib(path):
    global _PTQ
    if _PTQ is None:
        lib = ctypes.CDLL(path)
        fp, up = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
        lib.ptq_encode.argtypes = [ctypes.c_int] * 5 + [fp, ctypes.c_int, fp, fp, fp, ctypes.c_int, ctypes.c_int, fp, up, fp]
        lib.ptq_encode.restype = ctypes.c_int
        _PTQ = lib
    return _PTQ


def encode(lib_path, code, tab, X, Wt, grid, refine=1):
    """X, Wt: [nb, 128] in block order. Returns (d [nb], states [nb, Q*T], values [nb, 128] in block order, unit scale)."""
    lib = ptq_lib(lib_path)
    nb = X.shape[0]
    xs = np.ascontiguousarray(X[:, code.perm], dtype=np.float32)
    ws = np.ascontiguousarray(Wt[:, code.perm], dtype=np.float32)
    g = np.ascontiguousarray(np.asarray(grid, np.float32) * code.g0)
    d = np.zeros(nb, np.float32)
    st = np.zeros((nb, code.Q * code.T), np.uint32)
    cv = np.zeros((nb, QB), np.float32)
    fp, up = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
    r = lib.ptq_encode(code.L, code.kV, code.V, code.T, code.Q, tab.ctypes.data_as(fp), nb, xs.ctypes.data_as(fp),
                       ws.ctypes.data_as(fp), g.ctypes.data_as(fp), len(g), refine, d.ctypes.data_as(fp),
                       st.ctypes.data_as(up), cv.ctypes.data_as(fp))
    assert r == 0
    out = np.empty_like(cv)
    out[:, code.perm] = cv
    return d, st, out


def fp16(x):
    return x.astype(np.float16).astype(np.float32)


def code_quantize(lib_path, code, tab, W, imp, lp, grid):
    """W [R, C], C a multiple of 128 -> the code's reconstruction."""
    R, C = W.shape
    T = lp.quant_weights(W, imp)
    d, st, cv = encode(lib_path, code, tab, W.reshape(-1, QB), T.reshape(-1, QB), grid)
    return (fp16(d)[:, None] * cv).reshape(R, C)


# ---------------------------------------------------------------- rotation (Q164's, 128-wide)

def rot_apply(lp, W, imp):
    Wr, sign, H = lp.rotate(W, QB)
    return Wr, np.repeat(imp.reshape(-1, QB).mean(1), QB).astype(np.float32), sign, H


# ---------------------------------------------------------------- training

def train_blocks(lp, mats, rot):
    X, Wt = [], []
    for W, imp in mats:
        if rot:
            W, imp, _, _ = rot_apply(lp, W, imp)
        X.append(W.reshape(-1, QB)); Wt.append(lp.quant_weights(W, imp).reshape(-1, QB))
    return np.concatenate(X), np.concatenate(Wt)


def _acc_job(args):
    lib_path, code, tab, X, Wt, grid = args
    d, st, cv = encode(lib_path, code, tab, X, Wt, grid)
    xs, ws = X[:, code.perm], Wt[:, code.perm]
    nb = X.shape[0]
    V = code.V
    sv = np.repeat(st, V, axis=1)                                            # [nb, 128] state of each value
    vv = np.tile(np.arange(V), code.Q * code.T)[None, :].repeat(nb, 0)       # component of each value
    K = code.idx[vv, sv]
    num = np.bincount(K.ravel(), weights=(ws * d[:, None] * xs).ravel(), minlength=code.P.size)
    den = np.bincount(K.ravel(), weights=(ws * (d * d)[:, None]).ravel(), minlength=code.P.size)
    rms = np.sqrt((X * X).mean(1))
    ok = rms > 0
    e = (Wt * (X - fp16(d)[:, None] * cv) ** 2).sum()
    return num, den, np.median(d[ok] / rms[ok]) if ok.any() else 1.0, e, (Wt * X * X).sum()


def train_code(ex, jobs, lib_path, code, X, Wt, iters, grid, log):
    chunks = np.array_split(np.arange(X.shape[0]), jobs * 2)
    for it in range(iters):
        tab = code.tab()
        res = list(ex.map(_acc_job, [(lib_path, code, tab, X[c], Wt[c], grid) for c in chunks]))
        num = sum(r[0] for r in res); den = sum(r[1] for r in res)
        rel = sum(r[3] for r in res) / sum(r[4] for r in res)
        g = float(np.median([r[2] for r in res]))
        if code.train:
            code.P = np.where(den > 0, num / np.maximum(den, 1e-30), code.P).astype(np.float32)
        code.g0 = g
        log('  %s it %d: train err %.5f, g0 %.3f' % (code.name, it, rel, g))
        if not code.train:
            break


def tcq_search(ex, jobs, lib_path, fam, L, n, log):
    """Parity masks of a TCQ code: the best on Gaussian data (uniform union levels), 2- and 3-term XORs of state bits
    2..L-1 that include bit L-1. Returns the mask tuple."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((512, QB)).astype(np.float32)
    Wt = np.ones_like(X)
    bits = list(range(2, L - 1))
    cands = [(1 << (L - 1)) | (1 << i) for i in bits] + \
            [(1 << (L - 1)) | (1 << i) | (1 << j) for i in bits for j in bits if i < j]
    best = None
    sets = [(m,) for m in cands] if fam != 'tcq16' else [(a, b) for a in cands[:8] for b in cands[:8] if a < b]
    for ms in sets:
        code = make_code('%s.L%d' % ('tcq8u' if fam == 'tcq8' else fam, L), tcq_masks={('tcq8u' if fam == 'tcq8' else fam, L): ms})
        tab = code.tab()
        res = list(ex.map(_acc_job, [(lib_path, code, tab, X[c], Wt[c], [0.8, 1.0, 1.2]) for c in np.array_split(np.arange(512), jobs)]))
        e = sum(r[3] for r in res) / sum(r[4] for r in res)
        if best is None or e < best[0]:
            best = (e, ms)
    log('  tcq search %s L%d: masks %s, Gaussian err %.4f' % (fam, L, [hex(m) for m in best[1]], best[0]))
    return best[1]


# ---------------------------------------------------------------- evaluation

_G = {}


def worker_init(cfg):
    _G.update(cfg)
    if cfg.get('model'):
        lp = load_lowbit(cfg['lowbit'])
        _G['lp'] = lp
        _G['st'] = lp.st_tensor_index(cfg['model'])
        _G['gg'] = {k: gguf_index(v) for k, v in cfg['ggufs'].items() if v}


def load_lowbit(path):
    spec = importlib.util.spec_from_file_location('lowbit_proxy', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def use_lib(lp, path):
    """lowbit_proxy keeps one libggml-base; switch it between llama.cpp's and the fork's (N2P/N3P)."""
    libs = _G.setdefault('libs', {})
    if path not in libs:
        lp._LIB = None
        libs[path] = lp.ggml_lib(path)
    lp._LIB = libs[path]
    return libs[path]


def gguf_ref(lp, which, layer, e, kind, rows, C):
    """A file's own bytes for the sampled rows, dequantized (Q2_0, IQ3_S files through libggml; TQ2_T here)."""
    ent = _G['gg'][which]['blk.%d.ffn_%s_exps.weight' % (layer, kind)]
    t = ent[1]
    if t == 144:
        raw = gguf_expert_rows(ent, e, rows, C // QB * 34)
        return tq2t_dequant(raw, _G['tq_f16'])
    lib = use_lib(lp, _G['lib'])
    rb = lib.ggml_row_size(t, C)
    raw = gguf_expert_rows(ent, e, rows, rb)
    tr = lib.ggml_get_type_traits(t).contents
    fptr = ctypes.POINTER(ctypes.c_float)
    to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, fptr, ctypes.c_int64)(tr.to_float)
    out = np.zeros(len(rows) * C, np.float32)
    to_float(raw.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(fptr), len(rows) * C)
    return out.reshape(len(rows), C)


def gyro_unrotate(Wq, C):
    """Gyro stores W' with W' x' = W x for x' = the 128-wide normalized Hadamard of (signs * x); W = (W' H) * signs."""
    sv = _G['gyro_signs'][C]
    H = _G['H128']
    return (Wq.reshape(Wq.shape[0], -1, QB) @ H).reshape(Wq.shape) * sv


def work(item):
    layer, e, kind, cnt, imp, rows = item
    lp = _G['lp']
    W = lp.load_weights(_G['st'], layer, e, kind)[rows]
    C = W.shape[1]
    out = []

    def emit(fmt, bpw, Q, t0):
        ew, eu = lp.rel_err(W, Q, imp)
        out.append((layer, e, kind, int(cnt), fmt, bpw, ew, eu, time.time() - t0))

    for fmt in _G['refs']:
        t0 = time.time()
        if fmt.startswith('file:'):
            which = fmt[5:]
            if not _G['ggufs'].get(which):
                continue
            Q = gguf_ref(lp, which, layer, e, kind, rows, C)
            if which == 'gyro_m':
                # the other order (signs after the transform) as a check of the convention: it must score far worse
                sv = _G['gyro_signs'][C]
                Qa = ((Q * sv).reshape(Q.shape[0], -1, QB) @ _G['H128']).reshape(Q.shape)
                emit(fmt + '.alt', 2.125, Qa, t0)
                Q = gyro_unrotate(Q, C)
            bpw = {'q2_0_swift': 2.25, 'gyro_m': 2.125}.get(which, float('nan'))
            emit(fmt, bpw, Q, t0)
        elif fmt in lp.REF_TYPES:
            if C % 256 and fmt not in ('Q2_0', 'Q4_0', 'Q8_0'):
                continue
            lib = _G['lib_n'] if fmt in ('N2P', 'N3P') else _G['lib']
            if fmt in ('N2P', 'N3P') and not lib:
                continue
            use_lib(lp, lib)
            Q, bpw = lp.ref_quantize(lib, fmt, W, imp)
            emit(fmt, bpw, Q, t0)
        elif fmt.endswith('@128'):   # Q164's numpy formats with 128-weight super-blocks (N2P/N3P-128)
            f = lp.parse_format(fmt[:-4])
            f['super'] = 128
            Q = lp.quant_candidate(f, W, lp.quant_weights(W, imp), _G['tables'])
            emit(fmt, lp.format_bpw(f), Q, t0)
    for name, code in _G['codes'].items():
        t0 = time.time()
        _, _, rot = split_name(name)
        if rot:
            Wr, impr, sign, H = rot_apply(lp, W, imp)
            Qr = code_quantize(_G['ptq'], code, _G['tabs'][name], Wr, impr, lp, _G['grid'])
            Q = lp.unrotate(Qr, sign, H)
        else:
            Q = code_quantize(_G['ptq'], code, _G['tabs'][name], W, imp, lp, _G['grid'])
        emit(name, code.bpw, Q, t0)
    return out


def sample_rows(R, n, seed):
    return np.sort(np.random.default_rng(seed).choice(R, min(n, R), replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--imatrix', required=True)
    ap.add_argument('--lowbit', required=True)
    ap.add_argument('--ptq', required=True)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--lib-n', default='')
    ap.add_argument('--gyro-src', required=True)
    ap.add_argument('--q2-0', default='')
    ap.add_argument('--gyro-m', default='')
    ap.add_argument('--iq3s-ista', default='')
    ap.add_argument('--iq3s-swift', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='2,10,18,26,34,42,47')
    ap.add_argument('--experts', type=int, default=4)
    ap.add_argument('--kinds', default='gate,up,down')
    ap.add_argument('--rows-wide', type=int, default=64, help='rows sampled from gate/up (2,560 wide)')
    ap.add_argument('--rows-narrow', type=int, default=256, help='rows sampled from down (640 wide)')
    ap.add_argument('--fit-layers', default='6,30')
    ap.add_argument('--fit-experts', type=int, default=2)
    ap.add_argument('--fit-rows-wide', type=int, default=48)
    ap.add_argument('--fit-rows-narrow', type=int, default=192)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--grid', default='0.9,1.1')
    ap.add_argument('--refs', default='file:q2_0_swift,Q2_0,IQ2_S,IQ3_XXS,N2P,N3P,ik2x_s4x8@128,ik3x_s5x32@128,'
                                      'file:gyro_m,file:iq3s_ista,file:iq3s_swift')
    ap.add_argument('--codes', default='tq2t,tq2t.i8,tq2t+r128,tq2t.i8+r128')
    ap.add_argument('--tables', default='', help="Q164's tables.json (ik formats need none)")
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    t_start = time.time()
    log = lambda s: print('%s %s' % (time.strftime('%H:%M:%S'), s), flush=True)
    lp = load_lowbit(a.lowbit)
    st = lp.st_tensor_index(a.model)
    im = lp.gguf_tensors(a.imatrix)
    kinds = a.kinds.split(',')
    tq_f16, tq_i8 = tq2t_tables(a.gyro_src)
    grid = [float(x) for x in a.grid.split(',')]
    ggufs = {'q2_0_swift': a.q2_0, 'gyro_m': a.gyro_m, 'iq3s_ista': a.iq3s_ista, 'iq3s_swift': a.iq3s_swift}
    cfg = {'lowbit': a.lowbit, 'model': a.model, 'lib': a.lib, 'lib_n': a.lib_n, 'ptq': a.ptq, 'ggufs': ggufs,
           'refs': [r for r in a.refs.split(',') if r], 'grid': grid, 'tq_f16': tq_f16,
           'tables': json.load(open(a.tables)) if a.tables else {}, 'H128': lp.hadamard(QB)}
    if a.gyro_m:
        k = gguf_keys(a.gyro_m, 'prism.hadamard.')
        assert k['prism.hadamard.block_size'] == QB
        widths, vals = k['prism.hadamard.sign_widths'], np.array(k['prism.hadamard.sign_values'], np.float32)
        offs = np.cumsum([0] + widths)
        cfg['gyro_signs'] = {w: vals[offs[i]:offs[i + 1]] for i, w in enumerate(widths)}
        log('gyro signs: widths %s, keys %s' % (widths, sorted(k)))

    # the codes and their codebooks, trained on the fit layers
    names = sorted([c for c in a.codes.split(',') if c], key=lambda n: (split_name(n)[0], split_name(n)[2], split_name(n)[1]))
    codes = {}
    bases = sorted({split_name(n)[0] for n in names})
    with ProcessPoolExecutor(a.jobs) as ex:
        tcq_masks = {}
        for b in bases:
            fam, par = b.split('.')[:2] if '.' in b else (b, '')
            if fam.startswith('tcq'):
                L = int(par[1:])
                key = ('tcq8' if fam == 'tcq8u' else fam, L)
                if key not in tcq_masks:
                    tcq_masks[key] = tcq_search(ex, a.jobs, a.ptq, key[0], L, 0, log)
                tcq_masks[(fam, L)] = tcq_masks[key]
        fit = {}
        for rot in sorted({split_name(n)[2] for n in names}):
            mats = []
            for layer in [int(x) for x in a.fit_layers.split(',')]:
                for e in lp.pick_experts(im, layer, a.fit_experts * 2, a.seed)[:a.fit_experts]:
                    for kind in kinds:
                        W = lp.load_weights(st, layer, e, kind)
                        imp, _ = lp.load_importance(im, layer, e, kind)
                        rows = sample_rows(W.shape[0], a.fit_rows_wide if W.shape[1] > 1000 else a.fit_rows_narrow,
                                           a.seed + 1000 * layer + e)
                        mats.append((W[rows], imp))
            fit[rot] = train_blocks(lp, mats, rot)
            log('training blocks (rot %d): %d' % (rot, fit[rot][0].shape[0]))
        cb = {}
        for n in names:
            b, i8, rot = split_name(n)
            code = make_code(b, tq_f16, tq_i8, tcq_masks)
            code.name = n
            if b == 'tq2t' and i8:
                code.P, code.ent = tq_i8, 'fp'        # Gyro's own int8 table as it is
            elif i8:
                code.ent = 'i8'
            key = (b, rot)
            if key in cb and code.train:
                code.P, code.g0 = cb[key].P.copy(), cb[key].g0   # start the int8 variant from the trained FP codebook
                train_code(ex, a.jobs, a.ptq, code, fit[rot][0], fit[rot][1], max(2, a.iters // 3), grid, log)
            else:
                train_code(ex, a.jobs, a.ptq, code, fit[rot][0], fit[rot][1], a.iters, grid, log)
                cb[key] = code
            codes[n] = code
            log('code %s: L %d, %d bits a step, V %d, %d params, g0 %.3f, bpw %.4f (%.0f s so far)'
                % (n, code.L, code.kV, code.V, code.P.size, code.g0, code.bpw, time.time() - t_start))
    np.savez_compressed(os.path.splitext(a.out)[0] + '.codebooks.npz',
                        **{n: c.entries() for n, c in codes.items()}, **{n + '.g0': np.float32(c.g0) for n, c in codes.items()})
    json.dump({'tcq_masks': {'%s.L%d' % k: [hex(m) for m in v] for k, v in tcq_masks.items()}},
              open(os.path.splitext(a.out)[0] + '.codes.json', 'w'), indent=1)
    cfg['codes'] = codes
    cfg['tabs'] = {n: c.tab() for n, c in codes.items()}

    items = []
    for layer in [int(x) for x in a.layers.split(',')]:
        for e in lp.pick_experts(im, layer, a.experts, a.seed):
            for kind in kinds:
                imp, cnt = lp.load_importance(im, layer, e, kind)
                R = 640 if kind in ('gate', 'up') else 2560
                rows = sample_rows(R, a.rows_wide if kind in ('gate', 'up') else a.rows_narrow, a.seed + 1000 * layer + e)
                items.append((layer, e, kind, cnt, imp, rows))
    del im
    with open(a.out, 'w') as out, ProcessPoolExecutor(a.jobs, initializer=worker_init, initargs=(cfg,)) as ex:
        out.write('layer\texpert\tkind\tcount\tformat\tbpw\terr_imp\terr_plain\tseconds\n')
        for rows in ex.map(work, items):
            for r in rows:
                out.write('%d\t%d\t%s\t%d\t%s\t%.4f\t%.6g\t%.6g\t%.1f\n' % r)
            out.flush()
            log('L%d e%d %s done (%d formats)' % (rows[0][0], rows[0][1], rows[0][2], len(rows)))
    log('all done in %.0f s' % (time.time() - t_start))


if __name__ == '__main__':
    main()
