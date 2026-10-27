#!/usr/bin/env python3
# Consistency check of Miner Llama's N2P/N3P CPU code (fork branch pascal-p106-v3-lowbit) through ctypes:
#   1. vec_dot(quantized x, Q8_K y) against the float64 dot of dequantize(x) and dequantize(y): must agree to ~1e-6;
#   2. quantize -> dequantize round trip error on Gaussian data, with and without an importance vector.
# Run on the quant machine: python3 -I check_n2p_n3p.py <dir with libggml-base.so and libggml-cpu.so>
import ctypes, sys
import numpy as np

libdir = sys.argv[1]
base = ctypes.CDLL(libdir + '/libggml-base.so', mode=ctypes.RTLD_GLOBAL)
cpu = ctypes.CDLL(libdir + '/libggml-cpu.so', mode=ctypes.RTLD_GLOBAL)
fp = ctypes.POINTER(ctypes.c_float)
vp = ctypes.c_void_p


class TT(ctypes.Structure):
    _fields_ = [('type_name', ctypes.c_char_p), ('blck_size', ctypes.c_int64), ('blck_size_interleave', ctypes.c_int64),
                ('type_size', ctypes.c_size_t), ('is_quantized', ctypes.c_bool), ('to_float', vp), ('from_float_ref', vp)]


class TTC(ctypes.Structure):
    _fields_ = [('from_float', vp), ('vec_dot', vp), ('vec_dot_type', ctypes.c_int), ('nrows', ctypes.c_int64)]


base.ggml_get_type_traits.restype = ctypes.POINTER(TT)
base.ggml_get_type_traits.argtypes = [ctypes.c_int]
cpu.ggml_get_type_traits_cpu.restype = ctypes.POINTER(TTC)
cpu.ggml_get_type_traits_cpu.argtypes = [ctypes.c_int]
cpu.ggml_cpu_init()
base.ggml_row_size.restype = ctypes.c_size_t
base.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
base.ggml_quantize_chunk.restype = ctypes.c_size_t
base.ggml_quantize_chunk.argtypes = [ctypes.c_int, fp, vp, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, fp]

TO_FLOAT = ctypes.CFUNCTYPE(None, vp, fp, ctypes.c_int64)
FROM_FLOAT = ctypes.CFUNCTYPE(None, fp, vp, ctypes.c_int64)
VEC_DOT = ctypes.CFUNCTYPE(None, ctypes.c_int, fp, ctypes.c_size_t, vp, ctypes.c_size_t, vp, ctypes.c_size_t, ctypes.c_int)
Q8_K = 15


def quantize(t, x, imp=None):
    n = x.size
    buf = np.zeros(base.ggml_row_size(t, n), np.uint8)
    base.ggml_quantize_chunk(t, x.ctypes.data_as(fp), buf.ctypes.data_as(vp), 0, 1, n,
                             imp.ctypes.data_as(fp) if imp is not None else None)
    return buf


def dequantize(t, buf, n):
    if t == Q8_K:   # no to_float: float d, int8 qs[256], int16 bsums[16] per block
        b = buf.reshape(-1, 4 + 256 + 32)
        d = b[:, :4].copy().view(np.float32)
        return (d * b[:, 4:260].view(np.int8).astype(np.float32)).reshape(-1)
    out = np.zeros(n, np.float32)
    TO_FLOAT(base.ggml_get_type_traits(t).contents.to_float)(buf.ctypes.data_as(vp), out.ctypes.data_as(fp), n)
    return out


rng = np.random.default_rng(1)
n = 4096
for name, t in (('n2p', 500), ('n3p', 501)):
    tc = cpu.ggml_get_type_traits_cpu(t).contents
    assert tc.vec_dot_type == Q8_K, tc.vec_dot_type
    worst = 0.0
    for trial in range(20):
        x = rng.standard_normal(n).astype(np.float32) * (1 + 3 * (trial % 3))
        y = rng.standard_normal(n).astype(np.float32)
        xq = quantize(t, x)
        yq = np.zeros(base.ggml_row_size(Q8_K, n), np.uint8)
        FROM_FLOAT(cpu.ggml_get_type_traits_cpu(Q8_K).contents.from_float)(y.ctypes.data_as(fp), yq.ctypes.data_as(vp), n)
        s = ctypes.c_float()
        VEC_DOT(tc.vec_dot)(n, ctypes.byref(s), 0, xq.ctypes.data_as(vp), 0, yq.ctypes.data_as(vp), 0, 1)
        ref = float(np.dot(dequantize(t, xq, n).astype(np.float64), dequantize(Q8_K, yq, n).astype(np.float64)))
        worst = max(worst, abs(s.value - ref) / max(1e-9, abs(ref)))
    x = rng.standard_normal(n).astype(np.float32)
    imp = (rng.random(n).astype(np.float32) ** 4) * 10 + 0.01
    e_plain = float(np.sum((x - dequantize(t, quantize(t, x), n)) ** 2) / np.sum(x * x))
    xd = dequantize(t, quantize(t, x, imp), n)
    e_imp = float(np.sum(imp * (x - xd) ** 2) / np.sum(imp * x * x))
    xd0 = dequantize(t, quantize(t, x), n)
    e_imp0 = float(np.sum(imp * (x - xd0) ** 2) / np.sum(imp * x * x))
    print('%s: vec_dot vs dequantized dot, worst relative difference %.2e; Gaussian relative error %.4f; '
          'importance-weighted error with the importance vector %.4f, without %.4f' % (name, worst, e_plain, e_imp, e_imp0))
