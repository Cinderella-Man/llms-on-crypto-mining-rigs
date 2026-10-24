"""Offline acceptance of trained staircase-guesser heads on held-out transcripts (Q20/D1, pipeline_drafter.md section 4,
"Measuring acceptance offline").

For every generated token after the first, each head guesses from the inputs a given window profile provides (the
staircase of its placement, the after-miss inputs, ...), assuming the earlier tokens are right. Per position this gives
the Gumbel-coupled match probability with Flint's sampler (pipeline_sim.coupled_match, vectorized) and the probability
that the head's top token is the sample. Positions are split into fresh and copied text (copies of >= 8 tokens from the
prompt or the output so far, bench/ngram_replay/replay.py), reasoning / answer / tool-call segments and prompt
classes; 95 % intervals come from resampling whole transcripts.

usage: python score.py --heads DIR [DIR ...] --feats TEST.bin --gen TEST_GEN.jsonl --vocab vocab.npy --flint DIR
                       --out OUT_DIR [--min-p 0] [--device cuda]
OUT_DIR gets scores-<head>.npz (per-position arrays per window) and summary.json; a table is printed.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ngram_replay"))
import drafter  # noqa: E402
import feats  # noqa: E402
import replay  # noqa: E402
import train  # noqa: E402


def position_labels(gen_rows, ff):
    """Per test sequence (by seq_id = row index): category, per generated token in_copy and segment."""
    out = {}
    for sid, r in enumerate(gen_rows):
        gen = r["gen_tokens"]
        in_copy = np.zeros(len(gen), bool)
        for start, ln, _ in replay.copy_factors(r["prompt_tokens"], gen, 8):
            in_copy[start:start + ln] = True
        out[sid] = (r["category"], in_copy, np.array(r["segs"]))
    return out


def cluster_ci(values, groups, n_boot=2000, seed=0):
    """Mean of per-position values and a 95 % interval from resampling whole groups (transcripts)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    ug = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in ug])
    cnts = np.array([(groups == g).sum() for g in ug])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--flint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="NAME=ANCHOR,ANCHOR,...: one more window to score, newest first (e.g. oracle=full)")
    ap.add_argument("--init-only", action="store_true",
                    help="score the untrained initialization (a check: window oracle=full must nearly reproduce Flint)")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="sharpen the guesser's distribution (q^alpha) before coupling; choose it on validation data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    vocab = np.load(args.vocab)
    ff = feats.FeatFile(args.feats)
    gen_rows = replay.load(args.gen)
    labels = position_labels(gen_rows, ff)
    emb, head_rows, norm_w = train.load_flint(args.flint, vocab, args.device)
    t2d = train.t2d_table(vocab, emb.shape[0])
    summary = {}
    for hdir in args.heads:
        cfg = json.load(open(os.path.join(hdir, "config.json")))
        head = cfg["head"]
        profile = json.loads(json.dumps(drafter.PROFILES[head]))
        windows = dict(profile["windows"])
        windows.update(drafter.SCORE_EXTRA.get(head, {}))
        for spec in args.extra:
            name, anchors = spec.split("=")
            windows[name] = anchors.split(",")
        prof_all = {"committed": profile["committed"], "windows": windows}
        model = drafter.Drafter(emb, head_rows, norm_w, n_layer=cfg["args"].get("layers", 2)).to(args.device)
        if not args.init_only:
            model.load_state_dict(torch.load(os.path.join(hdir, "head.pt"), map_location=args.device), strict=False)
        model.eval()
        per = {name: {"coupled": [], "top": [], "conf": []} for name in windows}
        meta = {"seq": [], "j": [], "copy": [], "seg": [], "cat": []}
        for i in range(min(len(ff), args.max_seqs or len(ff))):
            rec = ff[i]
            b = train.Batch(rec, prof_all, t2d, min_target=0)
            d = train.to_device(b, args.device, dtype)
            committed, ca, win = train.build_inputs(d, prof_all, windows, rec.n_embd, dtype)
            with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=dtype):
                out = model(d["tokens"], committed, ca, win)
            p = feats.sampler_dist(d["top_lp"], min_p=args.min_p)
            for name, logits in out.items():
                c, t, conf = drafter.coupled_and_top(logits, d["top_draft"], p, alpha=args.alpha)
                per[name]["coupled"].append(c.cpu().numpy())
                per[name]["top"].append(t.cpu().numpy())
                per[name]["conf"].append(conf.cpu().numpy())
            j = b.tpos - rec.n_prompt + 1
            cat, in_copy, segs = labels[rec.seq_id]
            meta["seq"].append(np.full(len(j), rec.seq_id))
            meta["j"].append(j)
            meta["copy"].append(in_copy[j])
            meta["seg"].append(segs[j])
            meta["cat"].append(np.array([cat] * len(j)))
        arrs = {k: np.concatenate(v) for k, v in meta.items()}
        for name in windows:
            for m in ("coupled", "top", "conf"):
                arrs[f"{name}/{m}"] = np.concatenate(per[name][m])
        np.savez_compressed(os.path.join(args.out, f"scores-{head}.npz"), **arrs)
        res = {}
        for name in windows:
            r = {}
            for m in ("coupled", "top"):
                v = arrs[f"{name}/{m}"]
                fresh = ~arrs["copy"]
                r[m] = cluster_ci(v, arrs["seq"])
                r[m + "_fresh"] = cluster_ci(v[fresh], arrs["seq"][fresh])
                r[m + "_copy"] = cluster_ci(v[~fresh], arrs["seq"][~fresh])
                for cat in ("chat", "code", "edit", "agent"):
                    sel = fresh & (arrs["cat"] == cat)
                    r[f"{m}_fresh_{cat}"] = cluster_ci(v[sel], arrs["seq"][sel])
                for seg in np.unique(arrs["seg"]):
                    sel = arrs["seg"] == seg
                    r[f"{m}_{seg}"] = cluster_ci(v[sel], arrs["seq"][sel])
            res[name] = r
        summary[head] = {"dir": hdir, "train_gen_tokens": cfg.get("train_gen_tokens"), "windows": res,
                         "n_positions": int(len(arrs["seq"])), "fresh_share": float((~arrs["copy"]).mean())}
        print(f"\n## head {head} ({hdir}), {len(arrs['seq'])} positions, fresh {100 * (~arrs['copy']).mean():.1f} %")
        print("| Window | Coupled, all | Coupled, fresh | Coupled, copied | Top guess, all | Top, fresh | "
              "Fresh coupled: chat / code / edit / agent |")
        print("|---|---|---|---|---|---|---|")
        for name, r in res.items():
            f = lambda x: f"{x[0]:.3f} ({x[1]:.3f}–{x[2]:.3f})"   # noqa: E731
            per_cat = " / ".join(f"{r[f'coupled_fresh_{c}'][0]:.3f}" for c in ("chat", "code", "edit", "agent"))
            print(f"| {name} | {f(r['coupled'])} | {f(r['coupled_fresh'])} | {r['coupled_copy'][0]:.3f} | "
                  f"{r['top'][0]:.3f} | {r['top_fresh'][0]:.3f} | {per_cat} |")
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
