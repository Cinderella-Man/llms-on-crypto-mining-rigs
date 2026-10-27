# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Look closer at the RTN mismatches: where they sit, inversion examples, fractional-part histogram.

usage: python -I probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import load_st, mlx_int4  # noqa: E402

G = 64


def get(label):
    if label == "L45 down e3":
        q, s, b = mlx_int4("model.layers.45.mlp.switch_mlp.down_proj", 0, 4)
        return load_st("bf16", "model.layers.45.mlp.experts.3.down_proj.weight@0-2048"), q[3], s[3], b[3]
    if label == "L24 o_proj":
        q, s, b = mlx_int4("model.layers.24.self_attn.o_proj", 0, 2048)
        return load_st("bf16", "model.layers.24.self_attn.o_proj.weight@0-2048"), q, s, b
    if label == "lm_head 20000":
        q, s, b = mlx_int4("lm_head", 20000, 22048)
        return load_st("bf16", "lm_head.weight@20000-22048"), q, s, b


for label in ("L45 down e3", "L24 o_proj", "lm_head 20000"):
    w, q, s, b = get(label)
    q = q.astype(np.float32)
    sE, bE = np.repeat(s, G, -1), np.repeat(b, G, -1)
    x = (w - bE) / sE  # position on the code axis
    qr = np.clip(np.rint(x), 0, 15)
    mism = q != qr
    inner = (q > 0) & (q < 15) & (qr > 0) & (qr < 15)
    print(f"== {label}: mismatch {mism.mean():.3%}")
    # excess over the midpoint for mismatched weights: |x - q| - 0.5, in steps
    ex = np.abs(x - q)[mism] - 0.5
    print("  excess past midpoint (steps) quantiles 50/90/99/max:", np.round(np.percentile(ex, [50, 90, 99, 100]), 4))
    # fractional part histogram of x - qr (RTN residual), 10 bins over [-0.5, 0.5], for non-clipped weights
    fr = (x - qr)[inner]
    h, _ = np.histogram(fr, bins=10, range=(-0.5, 0.5))
    print("  RTN residual histogram (10 bins -0.5..0.5, % of weights):", np.round(100 * h / h.sum(), 2))
    # same for the INT4 code's residual x - q
    fr2 = (x - q)[inner]
    h2, _ = np.histogram(fr2, bins=[-1.5, -1, -0.6, -0.55, -0.52, -0.5, -0.4, 0.4, 0.5, 0.52, 0.55, 0.6, 1, 1.5])
    print("  INT4-code residual x-q counts, bins [-1.5,-1,-.6,-.55,-.52,-.5,-.4,.4,.5,.52,.55,.6,1,1.5]:", h2)
    # mismatch direction: does the code go toward the group mean (shrink) or away?
    mid = 7.5
    toward = np.sign(q - qr) == np.sign(mid - qr)
    print(f"  mismatches moving toward the middle code: {np.mean(toward[mism]):.1%}")
    # mismatch rate by |w| in steps from 0 (relative-rounding hypothesis) and by code value
    aw = np.abs(w / sE)
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 100)):
        m = (aw >= lo) & (aw < hi) & inner
        print(f"    |w|/s in [{lo},{hi}): share {m.mean():.2%}, mismatch {mism[m].mean():.3%}")
    rate_by_q = [mism[(qr == k)].mean() for k in range(16)]
    print("  mismatch by RTN code 0..15 (%):", np.round(100 * np.array(rate_by_q), 2))
    # inversion examples
    wg, qg, xg = w.reshape(-1, G), q.reshape(-1, G), x.reshape(-1, G)
    found = 0
    for gi in range(wg.shape[0]):
        dw = np.sign(wg[gi][:, None] - wg[gi][None, :])
        dq = np.sign(qg[gi][:, None] - qg[gi][None, :])
        bad = np.argwhere((dw != 0) & (dq != 0) & (dw != dq))
        if len(bad):
            i, j = bad[0]
            print(f"  inversion in group {gi}: w_i={wg[gi, i]:.6g} (x={xg[gi, i]:.4f}, q={qg[gi, i]:.0f})  "
                  f"w_j={wg[gi, j]:.6g} (x={xg[gi, j]:.4f}, q={qg[gi, j]:.0f})  dx={xg[gi, j] - xg[gi, i]:+.4f} steps")
            found += 1
            if found >= 4:
                break
    # row / column concentration of mismatches (rows = outputs, cols = inputs)
    rm, cm = mism.mean(1), mism.mean(0)
    print(f"  per-row mismatch: min {rm.min():.2%} med {np.median(rm):.2%} max {rm.max():.2%} (binomial sd "
          f"{np.sqrt(mism.mean() * (1 - mism.mean()) / mism.shape[1]):.2%}); per-col: min {cm.min():.2%} med "
          f"{np.median(cm):.2%} max {cm.max():.2%} (binomial sd {np.sqrt(mism.mean() * (1 - mism.mean()) / mism.shape[0]):.2%})")
    top = np.argsort(cm)[-5:]
    print("  top columns:", top, np.round(100 * cm[top], 2))

a = load_st("int4", "model.norm.weight@0-2048")
bb = load_st("bf16", "model.norm.weight@0-2048")
i = np.nonzero(a != bb)[0]
print("model.norm differing elements:", i, "int4:", a[i], "bf16:", bb[i])
