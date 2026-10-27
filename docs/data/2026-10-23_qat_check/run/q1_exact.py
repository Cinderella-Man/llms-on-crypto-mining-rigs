# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Q1: does the makers' Q4_1 GGUF hold exactly flint-int4's codes, scales and biases?

usage: python -I q1_exact.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import gguf_f32, gguf_q4_1, load_st, mlx_int4  # noqa: E402

# (label, mlx name, gguf name, i0, i1): i0..i1 is the outermost axis (expert, or output row for 2-D tensors)
PAIRS = []
for L, ranges in ((3, [(0, 4)]), (24, [(0, 4), (252, 256)]), (45, [(0, 4)])):
    for mk, gk in (("gate_proj", "ffn_gate_exps"), ("down_proj", "ffn_down_exps")):
        for i0, i1 in ranges:
            PAIRS.append((f"L{L} {gk} e{i0}-{i1 - 1}", f"model.layers.{L}.mlp.switch_mlp.{mk}", f"blk.{L}.{gk}.weight", i0, i1))
for L in (3, 24, 45):
    PAIRS.append((f"L{L} attn_output", f"model.layers.{L}.self_attn.o_proj", f"blk.{L}.attn_output.weight", 0, 2048))
PAIRS.append(("L24 attn_v", "model.layers.24.self_attn.v_proj", "blk.24.attn_v.weight", 0, 512))
for r0 in (20000, 100000):
    PAIRS.append((f"output rows {r0}-{r0 + 2047}", "lm_head", "output.weight", r0, r0 + 2048))


def f16(x):
    return x.astype(np.float16).astype(np.float32)


def main():
    print(f"{'tensor':32s} {'weights':>10s} {'codes =':>9s} {'d=f16(s)':>9s} {'m=f16(b)':>9s} "
          f"{'s in f16 range':>14s} {'pair d same':>11s} {'max|d/s-1|':>10s} {'max|m-b|/s':>10s}")
    for label, mname, gname, i0, i1 in PAIRS:
        q, s, b = mlx_int4(mname, i0, i1)
        gq, gd, gm = gguf_q4_1(gname, i0, i1)
        assert gq.shape == q.shape, (label, gq.shape, q.shape)
        s2 = np.repeat(s, 2, axis=-1)  # one group of 64 = two Q4_1 blocks of 32
        b2 = np.repeat(b, 2, axis=-1)
        same_codes = np.mean(gq == q)
        d_exact = np.mean(gd == f16(s2))
        m_exact = np.mean(gm == f16(b2))
        rng_ok = np.mean((np.abs(s2) >= 6.1e-5) & (np.abs(s2) < 65504))
        pair_same = np.mean((gd[..., 0::2] == gd[..., 1::2]) & (gm[..., 0::2] == gm[..., 1::2]))
        rel_d = np.max(np.abs(gd / s2 - 1))
        rel_m = np.max(np.abs(gm - b2) / np.abs(s2))
        print(f"{label:32s} {q.size:10d} {same_codes:9.6f} {d_exact:9.6f} {m_exact:9.6f} {rng_ok:14.6f} "
              f"{pair_same:11.6f} {rel_d:10.2e} {rel_m:10.2e}")
        if same_codes < 1:
            diff = gq.astype(int) - q.astype(int)
            print("   code diffs:", np.unique(diff[diff != 0], return_counts=True))
        if d_exact < 1:
            bad = gd != f16(s2)
            print("   non-matching d: n", bad.sum(), "example s", s2[bad][:5], "d", gd[bad][:5], "f16(s)", f16(s2)[bad][:5])
        if m_exact < 1:
            bad = gm != f16(b2)
            print("   non-matching m: n", bad.sum(), "example b", b2[bad][:5], "m", gm[bad][:5], "f16(b)", f16(b2)[bad][:5])
        # how far the GGUF's dequantized weights are from the MLX ones (FP16 vs BF16 scale/bias)
        w_mlx = s2.repeat(32, -1) * q + b2.repeat(32, -1)
        w_gg = gd.repeat(32, -1) * gq + gm.repeat(32, -1)
        print(f"   dequant: max|w_gguf - w_mlx| = {np.max(np.abs(w_gg - w_mlx)):.3e}, "
              f"rel Frobenius = {np.linalg.norm(w_gg - w_mlx) / np.linalg.norm(w_mlx):.3e}")

    # Embeddings: INT8 group-64 in MLX, F32 in the GGUF
    r0, r1 = 20000, 20064
    k = f"@{r0}-{r1}"
    qe = load_st("int4", "model.embed_tokens.weight" + k).astype(np.float32)
    se = load_st("int4", "model.embed_tokens.scales" + k).repeat(64, -1)
    be = load_st("int4", "model.embed_tokens.biases" + k).repeat(64, -1)
    deq = se * qe + be
    ge = gguf_f32("token_embd.weight", r0, r1)
    print(f"\ntoken_embd rows {r0}-{r1 - 1}: GGUF F32 vs MLX INT8 dequant (f32 math): exact share {np.mean(ge == deq):.6f}, "
          f"max abs diff {np.max(np.abs(ge - deq)):.3e}; vs bf16(dequant): exact share "
          f"{np.mean(ge == (deq.astype(np.float32).view(np.uint32) + 0x8000 & 0xFFFF0000).view(np.float32)):.6f}")


if __name__ == "__main__":
    main()
