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
    """Per test sequence: category, per generated token in_copy and segment. Keyed by seq_id as feats.py seqs numbers
    it (the numeric end of the row's id), or by row index when those numbers repeat: Q51's ids number each category
    on its own (chat_0, code_0, ...), and its dump file numbers the rows 0, 1, 2, ..."""
    out = {}
    nums = [int(r["id"].split("_")[-1]) for r in gen_rows]
    keys = nums if len(set(nums)) == len(nums) else range(len(gen_rows))
    for sid, r in zip(keys, gen_rows):
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
    ap.add_argument("--window-group", type=int, default=4,
                    help="windows per forward pass (fewer needs less memory; the committed rows are recomputed per pass)")
    ap.add_argument("--spans", nargs="*", default=[],
                    help="feats.py seqs maps with generated spans (agent sessions): score only generated tokens")
    ap.add_argument("--ctx", type=int, default=-1,
                    help="committed positions the head attends to (-1: as trained, the head's --ctx; 0: all)")
    ap.add_argument("--chunk-targets", type=int, default=8192, help="score long records in pieces of N positions")
    ap.add_argument("--only-windows", help="comma-separated window names (e.g. g9,g9miss): score only these")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float32 if args.device == "cpu" else torch.float16 if args.fp16 else torch.bfloat16   # as train.py
    vocab = np.load(args.vocab)
    ff = feats.FeatFile(args.feats)
    gen_rows = replay.load(args.gen)
    labels = position_labels(gen_rows, ff)
    spans = feats.load_spans(args.spans)
    emb, head_rows, norm_w = train.load_flint(args.flint, vocab, args.device)
    t2d = train.t2d_table(vocab, emb.shape[0])
    summary = {}
    for hdir in args.heads:
        cfg = json.load(open(os.path.join(hdir, "config.json")))
        head = cfg["head"]
        profile = json.loads(json.dumps(drafter.PROFILES[head]))
        windows = dict(profile["windows"])
        windows.update(drafter.SCORE_EXTRA.get(head, {}))
        general = cfg["args"].get("general", False)
        if general:   # card-1 staircases per card count, with the test file's recorded layers
            rec_layers = ff[0].layers
            for n in (3, 4, 5, 6, 7, 8, 9, 10, 12):
                windows[f"g{n}"] = [drafter.depth_name(x, rec_layers) for x in drafter.split_depths(drafter.even_split(n))]
            windows["g9light"] = [drafter.depth_name(x, rec_layers)
                                  for x in drafter.split_depths(drafter.even_split(9, light_last=2))]
            for n in (6, 9):
                for kind in ("pre", "miss"):
                    windows[f"g{n}{kind}"] = [drafter.depth_name(x, rec_layers)
                                              for x in drafter.split_depths(drafter.even_split(n), kind)]
        for spec in args.extra:
            name, anchors = spec.split("=")
            windows[name] = anchors.split(",")
        prof_all = {"committed": profile["committed"], "windows": windows}
        fuse = cfg["args"].get("fuse", False)
        n_layer = cfg["args"].get("layers", 2)
        ctx = cfg["args"].get("ctx", 0) if args.ctx < 0 else args.ctx
        model = drafter.Drafter(emb, head_rows, norm_w, n_layer=n_layer, fuse=fuse, general=general).to(args.device)
        if not args.init_only:
            model.load_state_dict(torch.load(os.path.join(hdir, "head.pt"), map_location=args.device), strict=False)
        model.eval()
        if args.only_windows:
            keep = args.only_windows.split(",")
            windows = {n: w for n, w in windows.items() if n in keep}
            prof_all = {"committed": profile["committed"], "windows": windows}
        per = {name: {"coupled": [], "top": [], "conf": [], "uniform": [], "overlap": []} for name in windows}
        meta = {"seq": [], "j": [], "copy": [], "seg": [], "cat": [], "pmax": []}
        items = [(i, part) for i in range(min(len(ff), args.max_seqs or len(ff)))
                 for part in train.target_parts(ff[i], args.chunk_targets)]
        for i, part in items:
            rec = ff[i]
            b = train.Batch(rec, prof_all, t2d, min_target=0, all_anchors=fuse, spans=spans.get(rec.seq_id),
                            part=part, ctx=ctx, n_layer=n_layer)
            if not len(b.tpos):
                continue
            d = train.to_device(b, args.device, dtype)
            committed, ca, win, cextra = train.build_inputs(d, prof_all, windows, rec.n_embd, dtype, fuse)
            p = feats.sampler_dist(d["top_lp"], min_p=args.min_p)
            names = list(win)
            for g in range(0, len(names), args.window_group):   # a few windows per pass bounds the logits' memory
                with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=dtype,
                                                     enabled=dtype != torch.float32):
                    out = model(d["tokens"], committed, ca, {n: win[n] for n in names[g:g + args.window_group]},
                                committed_extra=cextra, ctx=ctx)
                for name, logits in out.items():
                    c, t, conf = drafter.coupled_and_top(logits, d["top_draft"], p, alpha=args.alpha)
                    per[name]["coupled"].append(c.cpu().numpy())
                    per[name]["top"].append(t.cpu().numpy())
                    per[name]["conf"].append(conf.cpu().numpy())
                    per[name]["uniform"].append(
                        drafter.uniform_match(logits, d["top_draft"], p, alpha=args.alpha).cpu().numpy())
                    per[name]["overlap"].append(drafter.overlap(logits, d["top_draft"], p, alpha=args.alpha).cpu().numpy())
                del out
            meta["pmax"].append(p.max(-1).values.float().cpu().numpy())
            j = b.tpos + b.p0 - rec.n_prompt + 1
            cat, in_copy, segs = labels[rec.seq_id]
            meta["seq"].append(np.full(len(j), rec.seq_id))
            meta["j"].append(j)
            meta["copy"].append(in_copy[j])
            meta["seg"].append(segs[j])
            meta["cat"].append(np.array([cat] * len(j)))
        arrs = {k: np.concatenate(v) for k, v in meta.items()}
        for name in windows:
            for m in ("coupled", "top", "conf", "uniform", "overlap"):
                arrs[f"{name}/{m}"] = np.concatenate(per[name][m])
        np.savez_compressed(os.path.join(args.out, f"scores-{head}.npz"), **arrs)
        res = {}
        for name in windows:
            r = {}
            for m in ("coupled", "top", "uniform", "overlap"):
                v = arrs[f"{name}/{m}"]
                fresh = ~arrs["copy"]
                r[m] = cluster_ci(v, arrs["seq"])
                r[m + "_fresh"] = cluster_ci(v[fresh], arrs["seq"][fresh])
                r[m + "_copy"] = cluster_ci(v[~fresh], arrs["seq"][~fresh])
                for cat in ("chat", "code", "edit", "agent", "agentic"):
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
              "Shared uniform, all / fresh | Fresh coupled: chat / code / edit / agent / agentic |")
        print("|---|---|---|---|---|---|---|---|")
        for name, r in res.items():
            f = lambda x: f"{x[0]:.3f} ({x[1]:.3f}–{x[2]:.3f})"   # noqa: E731
            per_cat = " / ".join(f"{r[f'coupled_fresh_{c}'][0]:.3f}"
                                 for c in ("chat", "code", "edit", "agent", "agentic"))
            print(f"| {name} | {f(r['coupled'])} | {f(r['coupled_fresh'])} | {r['coupled_copy'][0]:.3f} | "
                  f"{r['top'][0]:.3f} | {r['top_fresh'][0]:.3f} | {r['uniform'][0]:.3f} / {r['uniform_fresh'][0]:.3f} | "
                  f"{per_cat} |")
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
