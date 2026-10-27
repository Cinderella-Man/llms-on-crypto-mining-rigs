"""Checks of the agent-session training path on synthetic data (CPU, no dump needed):
- attention limited to the last ctx committed positions (--ctx): the blockwise path equals a dense masked reference,
  and a ctx longer than the sequence equals full attention;
- a record scored in pieces (--chunk-targets) with --ctx gives the same logits as the whole record;
- --spans keeps exactly the targets whose next token is generated;
- Record.read_rows(which, lo, hi) equals the matching slice of the whole read;
- feats.py seqs keeps an agent session's spans in its map, and vocab.py counts only the spans.

usage: python check_ctx.py [--dir DIR]   (exits 1 on a failed check)
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drafter  # noqa: E402
import feats  # noqa: E402
import train  # noqa: E402
import vocab  # noqa: E402

V, E, K = 200, 64, 20
FP8_OK = np.array([c for c in range(256) if (c & 0x7F) not in (0x7F,)], np.uint8)   # no NaN codes


def write_feat(path, recs, rng):
    """A feature file in llama-staircase-dump's v3 layout: recs = [(seq_id, n_prompt, n_tok, layers, allpos)]."""
    with open(path, "wb") as f:
        for seq_id, n_prompt, n_tok, layers, allpos in recs:
            a0 = max(0, n_prompt - 8)
            starts = [0 if L in allpos else a0 for L in layers]
            f.write(np.array([seq_id, n_prompt, n_tok, a0, len(layers), K, E, 0x51320003], np.int32).tobytes())
            f.write(np.array(layers, np.int32).tobytes())
            f.write(np.array(starts, np.int32).tobytes())
            f.write(rng.integers(0, V, n_tok).astype(np.int32).tobytes())
            for b in [0] + starts:
                f.write(rng.uniform(0.005, 0.02, n_tok - b).astype(np.float16).tobytes())
                f.write(rng.choice(FP8_OK, (n_tok - b) * E).tobytes())
            n_gen = n_tok - n_prompt
            ids = np.stack([rng.choice(V, K, replace=False) for _ in range(n_gen)]).astype(np.int32)
            lp = np.sort(rng.normal(0, 2, (n_gen, K)), 1)[:, ::-1]
            lp = (lp - np.log(np.exp(lp).sum(1, keepdims=True)) - 0.1).astype(np.float32)
            f.write(ids.tobytes())
            f.write(lp.tobytes())


def dense_committed(self, x, cos, sin, ctx=0):
    """Reference for Block.committed: one dense attention with the band mask."""
    q, k, v = self.qkv(x, cos, sin)
    rep, N = self.n_head // self.n_kv, x.shape[0]
    qp, kp = torch.arange(N)[:, None], torch.arange(N)[None, :]
    mask = (kp <= qp) & ((kp > qp - ctx) if ctx else torch.ones_like(kp, dtype=torch.bool))
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.repeat_interleave(rep, 1).transpose(0, 1)[None],
                                       v.repeat_interleave(rep, 1).transpose(0, 1)[None], attn_mask=mask)
    return self.mlp(x + self.drop(self.wo(o[0].transpose(0, 1).reshape(N, -1)))), k, v


def run(model, b, prof, ctx):
    """Logits per window, keyed by absolute target position."""
    d = train.to_device(b, "cpu", torch.float32)
    committed, ca, win, cextra = train.build_inputs(d, prof, prof["windows"], E, torch.float32)
    with torch.no_grad():
        out = model(d["tokens"], committed, ca, win, committed_extra=cextra, ctx=ctx)
    pos = (b.tpos + b.p0).tolist()
    return {name: dict(zip(pos, logits)) for name, logits in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    a = ap.parse_args()
    tmp = a.dir or tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    bad = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
        if not ok:
            bad.append(name)

    path = os.path.join(tmp, "synthetic.bin")
    write_feat(path, [(400007, 40, 420, [9, 17, 25, 33, 41], {9}), (3, 30, 90, [9, 17, 25, 33, 41], {9})], rng)
    ff = feats.FeatFile(path)
    rec = ff[0]
    for which in [9, 17, 41, "full"]:
        s0, q0, c0 = rec.read_rows(which)
        s1, q1, c1 = rec.read_rows(which, 100, 250)
        r0, r1 = 100 - s0, 250 - s0
        check(f"read_rows({which}, 100, 250)", s1 == 100 and np.array_equal(q1, q0[r0:r1])
              and np.array_equal(c1, c0[r0:r1]), f"first {s1}, rows {len(q1)}")

    emb = torch.randn(V, E)
    model = drafter.Drafter(emb, torch.randn(V, E) * 0.1, torch.ones(E), n_layer=2, n_head=4, n_kv=2, head_dim=16,
                            ffn=128)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.08)
    model.eval()
    prof = drafter.PROFILES["A"]
    t2d = train.t2d_table(np.arange(V), V)

    whole = train.Batch(rec, prof, t2d, min_target=0)
    full = run(model, whole, prof, 0)
    longer = run(model, whole, prof, 10 ** 6)
    diff = max((full[w][p] - longer[w][p]).abs().max().item() for w in full for p in full[w])
    check("ctx longer than the sequence = full attention", diff < 1e-5, f"max diff {diff:.2e}")

    ctx = 32
    banded = run(model, whole, prof, ctx)
    orig, slice_keys = drafter.Block.committed, drafter.SLICE_KEYS
    drafter.Block.committed, drafter.SLICE_KEYS = dense_committed, False
    try:
        ref = run(model, whole, prof, ctx)
    finally:
        drafter.Block.committed, drafter.SLICE_KEYS = orig, slice_keys
    diff = max((banded[w][p] - ref[w][p]).abs().max().item() for w in ref for p in ref[w])
    check(f"blockwise ctx {ctx} = dense band reference", diff < 1e-4, f"max diff {diff:.2e}")
    moved = max((banded[w][p] - full[w][p]).abs().max().item() for w in full for p in full[w])
    check(f"ctx {ctx} changes the logits (the band is in effect)", moved > 1e-3, f"max change {moved:.2e}")

    spans = [[60, 140], [170, 171], [200, 420]]
    gen = np.zeros(rec.n_tok + 1, bool)
    for s, e in spans:
        gen[s:e] = True
    sb = train.Batch(rec, prof, t2d, min_target=0, spans=spans, ctx=ctx)
    want = [t for t in whole.tpos if gen[t + 1]]
    check("spans keep exactly the targets of generated tokens", list(sb.tpos) == want, f"{len(sb.tpos)} targets")
    pieces, n = {w: {} for w in prof["windows"]}, 0
    for part in train.target_parts(rec, 50):
        pb = train.Batch(rec, prof, t2d, min_target=0, spans=spans, part=part, ctx=ctx, n_layer=2)
        if not len(pb.tpos):
            continue
        n += 1
        check(f"piece {part} starts after position 0 when it can", part[0] < 2 * ctx + 8 or pb.p0 > 0, f"p0 {pb.p0}")
        for w, rows in run(model, pb, prof, ctx).items():
            pieces[w].update(rows)
    whole_sp = run(model, sb, prof, ctx)
    same = all(set(pieces[w]) == set(whole_sp[w]) for w in whole_sp)
    diff = max((pieces[w][p] - whole_sp[w][p]).abs().max().item() for w in whole_sp for p in whole_sp[w])
    check(f"{n} pieces of 50 = the whole record (ctx {ctx}, spans)", same and diff < 1e-4, f"max diff {diff:.2e}")

    row = {"id": "agentic_400007", "category": "agentic", "prompt_tokens": [1, 2, 3], "gen_tokens": [4, 5, 6, 7, 8, 9],
           "gen_ranges": [[3, 5], [7, 9]], "session_id": "s1-x-wt-a0"}
    gp, sp, mp = (os.path.join(tmp, x) for x in ("seq.jsonl", "seqs.txt", "map.jsonl"))
    with open(gp, "w") as f:
        f.write(json.dumps(row) + "\n")
    feats.cmd_seqs(argparse.Namespace(exclude=None, gen=[gp], max_tok=100, take=0, seed=1, out_seqs=sp, out_map=mp,
                                      only=None))
    check("seqs map keeps gen_ranges", feats.load_spans([mp]) == {400007: [[3, 5], [7, 9]]})
    check("vocab counts only the spans", vocab.generated(row) == [4, 5, 8, 9], str(vocab.generated(row)))
    print("PASS" if not bad else f"FAILED: {', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
