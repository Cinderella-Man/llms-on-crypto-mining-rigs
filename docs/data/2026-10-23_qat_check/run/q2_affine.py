# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Q2 follow-up: could any per-group affine grid (s', b') reproduce the INT4 codes as round-to-nearest of the BF16
weights? Per group, find the smallest perturbation eps (in steps, L-inf) that makes the codes RTN-consistent under the
best (s', b'), by a ternary search over s' (the feasibility margin is concave in s'). eps = 0 means some scale and bias
explain the codes exactly (e.g. codes rounded before a scale/bias refit); eps > 0 means the weights the codes were made
from differ from the released BF16 weights. Also: RTN-residual histograms for stored, min-max and MSE-searched grids.

usage: python -I q2_affine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import load_st, mlx_int4  # noqa: E402

G = 64


def margin(wg, qg, sp):
    """F(s') = min_{q>=1}(w - s'(q-.5)) - max_{q<=14}(w - s'(q+.5)), per group; sp: [N,1]."""
    up = np.where(qg >= 1, wg - sp * (qg - 0.5), np.inf).min(1)
    lo = np.where(qg <= 14, wg - sp * (qg + 0.5), -np.inf).max(1)
    return up - lo


def eps_min(w, q, s):
    wg = w.reshape(-1, G).astype(np.float64)
    qg = q.reshape(-1, G).astype(np.float64)
    s0 = s.reshape(-1, 1).astype(np.float64)
    a, b = 0.8 * s0, 1.25 * s0
    for _ in range(90):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        f1, f2 = margin(wg, qg, m1), margin(wg, qg, m2)
        left = (f1 < f2)[:, None]
        a = np.where(left, m1, a)
        b = np.where(left, b, m2)
    best = margin(wg, qg, (a + b) / 2)
    return np.maximum(0.0, -best / 2) / np.abs(s0[:, 0])


def eps_stored(w, q, s, b):
    sE, bE = np.repeat(s, G, -1), np.repeat(b, G, -1)
    x = (w - bE) / sE
    v = np.where((q > 0) & (q < 15), np.abs(x - q) - 0.5,
                 np.where(q == 0, x - 0.5, 14.5 - x))
    return np.maximum(0, v.reshape(-1, G).max(1))


def hist_resid(w, s, b):
    sE, bE = np.repeat(s, G, -1), np.repeat(b, G, -1)
    x = (w - bE) / sE
    qr = np.rint(x)
    inner = (qr > 0) & (qr < 15)
    h, _ = np.histogram((x - qr)[inner], bins=10, range=(-0.5, 0.5))
    return np.round(100 * h / h.sum(), 2)


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def grids(w):
    """(s, b) per group for plain min-max and for a 31-point range-shrink MSE search."""
    wg = w.reshape(-1, G)
    lo, hi = wg.min(1, keepdims=True), wg.max(1, keepdims=True)
    mm = (bf16_round((hi - lo) / 15), bf16_round(lo))
    mid = (lo + hi) / 2
    best = None
    for f in np.linspace(0.7, 1.0, 31):
        l2, h2 = mid + (lo - mid) * f, mid + (hi - mid) * f
        s, b = bf16_round((h2 - l2) / 15), bf16_round(l2)
        e = ((s * np.clip(np.rint((wg - b) / s), 0, 15) + b - wg) ** 2).sum(1, keepdims=True)
        if best is None:
            best = [e, s, b]
        else:
            m = e < best[0]
            best = [np.where(m, e, best[0]), np.where(m, s, best[1]), np.where(m, b, best[2])]
    shp = w.shape[:-1] + (w.shape[-1] // G,)
    return (mm[0].reshape(shp), mm[1].reshape(shp)), (best[1].reshape(shp), best[2].reshape(shp))


cases = []
for L, e in ((3, 0), (24, 0), (24, 255), (45, 3)):
    for kind, rows in (("gate_proj", 512), ("down_proj", 2048)):
        i0 = 252 if e >= 252 else 0
        q, s, b = mlx_int4(f"model.layers.{L}.mlp.switch_mlp.{kind}", i0, i0 + 4)
        cases.append((f"L{L} {kind} e{e}", load_st("bf16", f"model.layers.{L}.mlp.experts.{e}.{kind}.weight@0-{rows}"),
                      q[e - i0], s[e - i0], b[e - i0]))
for L in (3, 24, 45):
    q, s, b = mlx_int4(f"model.layers.{L}.self_attn.o_proj", 0, 2048)
    cases.append((f"L{L} o_proj", load_st("bf16", f"model.layers.{L}.self_attn.o_proj.weight@0-2048"), q, s, b))
q, s, b = mlx_int4("model.layers.24.self_attn.v_proj", 0, 512)
cases.append(("L24 v_proj", load_st("bf16", "model.layers.24.self_attn.v_proj.weight@0-512"), q, s, b))
q, s, b = mlx_int4("lm_head", 20000, 22048)
cases.append(("lm_head 20000+2048", load_st("bf16", "lm_head.weight@20000-22048"), q, s, b))

print(f"{'tensor':20s} {'groups':>7s} | stored s,b: {'exact':>7s} {'eps p99':>7s} | best s',b': {'exact':>7s} "
      f"{'eps>0.01':>8s} {'eps p50|>0':>10s} {'eps p99':>7s} {'eps max':>7s}")
for label, w, q, s, b in cases:
    q = q.astype(np.float64)
    es = eps_stored(w, q, s, b)
    em = eps_min(w, q, s)
    pos = em[em > 1e-9]
    print(f"{label:20s} {em.size:7d} | {'':11s} {np.mean(es <= 1e-9):7.2%} {np.percentile(es, 99):7.4f} | {'':11s} "
          f"{np.mean(em <= 1e-9):7.2%} {np.mean(em > 0.01):8.2%} {np.median(pos) if pos.size else 0:10.4f} "
          f"{np.percentile(em, 99):7.4f} {em.max():7.4f}")

print("\nRTN residual histograms (x - rint(x), 10 bins over -0.5..0.5, % of non-clipped weights)")
for label, w, q, s, b in cases[:1] + cases[-3:-2] + cases[-1:]:
    (sm, bm), (sx, bx) = grids(w)
    print(f"{label:20s} stored grid : {hist_resid(w, s, b)}")
    print(f"{'':20s} min-max grid: {hist_resid(w, sm, bm)}")
    print(f"{'':20s} MSE search  : {hist_resid(w, sx, bx)}")
