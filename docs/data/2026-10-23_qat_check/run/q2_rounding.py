# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Q2: are flint-int4's codes round-to-nearest of the BF16 weights at the export's own scale and bias?

usage: python -I q2_rounding.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import gguf_q4_1, load_st, mlx_int4  # noqa: E402

G = 64
RNG = np.random.default_rng(0)


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def minmax_rtn(w, bits=4):
    """Plain min-max affine RTN per group of 64 (scale and bias rounded to BF16): the no-search baseline."""
    wg = w.reshape(-1, G)
    lo, hi = wg.min(1, keepdims=True), wg.max(1, keepdims=True)
    s = bf16_round(np.maximum((hi - lo) / (2 ** bits - 1), 1e-8))
    b = bf16_round(lo)
    q = np.clip(np.rint((wg - b) / s), 0, 2 ** bits - 1)
    return (s * q + b).reshape(w.shape)


def mse_rtn(w, bits=4, grid=np.linspace(0.70, 1.0, 31)):
    """Min-max RTN with a shrink search on the range per group (lowest squared error): a stand-in for scale_rule mse."""
    wg = w.reshape(-1, G)
    lo, hi = wg.min(1, keepdims=True), wg.max(1, keepdims=True)
    mid = (lo + hi) / 2
    best, best_err = None, None
    for f in grid:
        l2, h2 = mid + (lo - mid) * f, mid + (hi - mid) * f
        s = bf16_round(np.maximum((h2 - l2) / (2 ** bits - 1), 1e-8))
        b = bf16_round(l2)
        dq = s * np.clip(np.rint((wg - b) / s), 0, 2 ** bits - 1) + b
        e = ((dq - wg) ** 2).sum(1, keepdims=True)
        if best is None:
            best, best_err = dq, e
        else:
            m = e < best_err
            best = np.where(m, dq, best)
            best_err = np.where(m, e, best_err)
    return best.reshape(w.shape)


def rel(a, w):
    return np.linalg.norm(a - w) / np.linalg.norm(w)


def analyse(label, w, q, s, b):
    """w: BF16 weights [.., out, in]; q codes; s, b [.., out, in/64]."""
    w = w.reshape(-1, w.shape[-1])
    q = q.reshape(w.shape).astype(np.float32)
    s = s.reshape(w.shape[0], -1)
    b = b.reshape(w.shape[0], -1)
    sE, bE = np.repeat(s, G, -1), np.repeat(b, G, -1)
    step = np.abs(sE)
    deq = sE * q + bE
    q_rtn = np.clip(np.rint((w - bE) / sE), 0, 15)
    mism = q != q_rtn
    err = (w - deq) / step  # signed, in steps
    aerr = np.abs(err)
    # weights outside the grid's range by more than half a step (clipping can't be "fixed" by rounding)
    lo = np.minimum(bE, bE + 15 * sE)
    hi = np.maximum(bE, bE + 15 * sE)
    outside = (w < lo - 0.5 * step) | (w > hi + 0.5 * step)
    dq_rtn = sE * q_rtn + bE
    r = {
        "label": label, "n": w.size,
        "neg_scale": float(np.mean(s < 0)),
        "mismatch": float(mism.mean()),
        "mism_by1": float(np.mean(np.abs(q - q_rtn) == 1)),
        "mism_by2+": float(np.mean(np.abs(q - q_rtn) >= 2)),
        "outside": float(outside.mean()),
        "abs_err_p50": float(np.percentile(aerr, 50)), "abs_err_p90": float(np.percentile(aerr, 90)),
        "abs_err_p99": float(np.percentile(aerr, 99)), "abs_err_max": float(aerr.max()),
        "gt0.5": float(np.mean(aerr > 0.5 + 1e-3)), "gt1": float(np.mean(aerr > 1)), "gt2": float(np.mean(aerr > 2)),
        "rel_int4": rel(deq, w), "rel_rtn_stored_sb": rel(dq_rtn, w), "rel_minmax": rel(minmax_rtn(w), w),
        "rel_mse_search": rel(mse_rtn(w), w),
        "w_rms_in_steps": float(np.sqrt(np.mean((w / step) ** 2))),
    }
    # systematic drift: mean signed error per group, in steps (RTN noise alone gives ~0.29/sqrt(64) = 0.036)
    gm = err.reshape(-1, G).mean(1)
    r["group_mean_err_std"] = float(gm.std())
    # per input column mismatch (GPTQ leaves the first-processed column exactly RTN)
    colm = mism.mean(0)
    r["col_mism_min"], r["col_mism_med"], r["col_mism_max"] = float(colm.min()), float(np.median(colm)), float(colm.max())
    r["col_mism_first8"] = float(colm[:8].mean())
    # rank consistency within groups: pairs of weights ordered one way by BF16 value and the other way by code
    ng = w.size // G
    idx = RNG.choice(ng, size=min(ng, 4000), replace=False)
    wg, qg = w.reshape(-1, G)[idx], q.reshape(-1, G)[idx] * np.sign(s.reshape(-1))[idx, None]
    dw = np.sign(wg[:, :, None] - wg[:, None, :])
    dq = np.sign(qg[:, :, None] - qg[:, None, :])
    both = (dw != 0) & (dq != 0)
    r["discordant_pairs"] = float(np.sum(both & (dw != dq)) / np.sum(both))
    r["groups_with_inversion"] = float(np.mean(np.any(both & (dw != dq), axis=(1, 2))))
    # least-squares scale/bias for the stored codes vs stored scale/bias (an MSE rule refit would make them agree)
    A = q.reshape(-1, G)
    Wg = w.reshape(-1, G)
    qm, wm = A.mean(1), Wg.mean(1)
    var = ((A - qm[:, None]) ** 2).sum(1)
    ok = var > 0
    s_ls = ((A - qm[:, None]) * (Wg - wm[:, None])).sum(1)[ok] / var[ok]
    r["ls_scale_ratio_med"] = float(np.median(s_ls / s.reshape(-1)[ok]))
    r["ls_scale_ratio_p10_p90"] = [float(x) for x in np.percentile(s_ls / s.reshape(-1)[ok], [10, 90])]
    # correlation of the INT4 - BF16 difference with the weight itself (shrinkage toward the group mean?)
    d = (deq - w).reshape(-1, G)
    wc = Wg - wm[:, None]
    r["slope_diff_vs_w"] = float((d * wc).sum() / (wc * wc).sum())
    return r


def main():
    rows = []
    for L, ranges in ((3, [(0, 4)]), (24, [(0, 4), (252, 256)]), (45, [(0, 4)])):
        for kind in ("gate_proj", "down_proj"):
            for i0, i1 in ranges:
                q, s, b = mlx_int4(f"model.layers.{L}.mlp.switch_mlp.{kind}", i0, i1)
                for e in range(i0, i1):
                    w = load_st("bf16", f"model.layers.{L}.mlp.experts.{e}.{kind}.weight@0-{'2048' if kind == 'down_proj' else '512'}")
                    rows.append(analyse(f"L{L} {kind} e{e}", w, q[e - i0], s[e - i0], b[e - i0]))
    for L in (3, 24, 45):
        q, s, b = mlx_int4(f"model.layers.{L}.self_attn.o_proj", 0, 2048)
        rows.append(analyse(f"L{L} o_proj", load_st("bf16", f"model.layers.{L}.self_attn.o_proj.weight@0-2048"), q, s, b))
    q, s, b = mlx_int4("model.layers.24.self_attn.v_proj", 0, 512)
    rows.append(analyse("L24 v_proj", load_st("bf16", "model.layers.24.self_attn.v_proj.weight@0-512"), q, s, b))
    for r0 in (20000, 100000):
        q, s, b = mlx_int4("lm_head", r0, r0 + 2048)
        rows.append(analyse(f"lm_head {r0}+2048", load_st("bf16", f"lm_head.weight@{r0}-{r0 + 2048}"), q, s, b))

    # Embeddings (INT8, group 64): the same RTN test at 8 bits
    k = "@20000-20064"
    qe = load_st("int4", "model.embed_tokens.weight" + k).astype(np.float32)
    se = load_st("int4", "model.embed_tokens.scales" + k)
    be = load_st("int4", "model.embed_tokens.biases" + k)
    we = load_st("bf16", "model.embed_tokens.weight" + k)
    sE, bE = np.repeat(se, G, -1), np.repeat(be, G, -1)
    qr = np.clip(np.rint((we - bE) / sE), 0, 255)
    ae = np.abs(we - (sE * qe + bE)) / np.abs(sE)
    emb = (f"embed_tokens rows 20000-20063 (INT8): mismatch {np.mean(qr != qe):.4%}, |err| p99 {np.percentile(ae, 99):.3f} "
           f"max {ae.max():.3f} steps, rel err {rel(sE * qe + bE, we):.4%}")

    cols = [("label", 20, "s"), ("mismatch", 9, ".2%"), ("mism_by1", 9, ".2%"), ("mism_by2+", 9, ".3%"),
            ("gt0.5", 8, ".2%"), ("gt1", 8, ".3%"), ("gt2", 8, ".4%"), ("abs_err_p50", 6, ".3f"), ("abs_err_p99", 6, ".3f"),
            ("abs_err_max", 6, ".2f"), ("outside", 8, ".3%"), ("rel_int4", 7, ".4f"), ("rel_rtn_stored_sb", 7, ".4f"),
            ("rel_minmax", 7, ".4f"), ("rel_mse_search", 7, ".4f"), ("group_mean_err_std", 7, ".4f"),
            ("discordant_pairs", 8, ".3%"), ("groups_with_inversion", 8, ".2%"), ("col_mism_min", 7, ".2%"),
            ("col_mism_first8", 7, ".2%"), ("col_mism_max", 7, ".2%"), ("ls_scale_ratio_med", 7, ".4f"),
            ("slope_diff_vs_w", 8, ".4f"), ("neg_scale", 6, ".2%"), ("w_rms_in_steps", 6, ".2f")]
    print(" ".join(f"{c:>{w}s}" if i else f"{c:{w}s}" for i, (c, w, _) in enumerate(cols)))
    for r in rows:
        print(" ".join(f"{format(r[c], f):>{w}s}" if i else f"{r[c]:{w}s}" for i, (c, w, f) in enumerate(cols)))
    print()
    print(emb)
    import json
    rows = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()} for r in rows]
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "q2_rows.json"), "w"), indent=1)

    # negative control for Q1: GGUF expert 0 vs MLX expert 1 must differ
    gq, _, _ = gguf_q4_1("blk.24.ffn_gate_exps.weight", 0, 2)
    q, _, _ = mlx_int4("model.layers.24.mlp.switch_mlp.gate_proj", 0, 4)
    print(f"negative control: GGUF L24 gate expert 0 vs MLX expert 1 code agreement {np.mean(gq[0] == q[1]):.4f}; "
          f"expert 0 vs 0 {np.mean(gq[0] == q[0]):.4f}")


if __name__ == "__main__":
    main()
