"""Train one staircase-guesser head (Q20/D1) on llama-staircase-dump features.

usage: python train.py --head A --feats F1.bin [F2.bin ...] --vocab vocab.npy --flint FLINT_DIR --out OUT_DIR
                       [--minutes 180] [--lr 2e-4] [--accum-targets 8192] [--fp16] [--max-seqs N] [--seed 1]
FLINT_DIR holds flint_weights.py's output. The schedule is time-based: linear warm-up, then cosine decay to 10 % over
--minutes, so a run ends on time whatever the GPU. Every 3 % of the sequences (every 33rd) are held out for validation.
OUT_DIR gets head.pt (the weights with the best validation acceptance of the first window, without Flint's embedding
table), head-last.pt, config.json, log.jsonl and ckpt.pt (weights, optimizer and scaler state, step and elapsed time,
written every --ckpt-every minutes). --resume continues from ckpt.pt: the time budget counts the time already spent, so
a larger --minutes extends a run, and new --feats files continue it on more data.
"""
import argparse
import json
import math
import os
import queue
import random
import threading
import time

import numpy as np
import torch

import drafter
import feats


def load_flint(flint_dir, vocab, device):
    emb = torch.from_numpy(np.load(os.path.join(flint_dir, "token_embd.npy"))).to(device)
    out = np.load(os.path.join(flint_dir, "output.npy"), mmap_mode="r")
    head_rows = torch.from_numpy(np.ascontiguousarray(out[vocab]).astype(np.float32))
    norm_w = torch.from_numpy(np.load(os.path.join(flint_dir, "output_norm.npy")).astype(np.float32))
    return emb, head_rows, norm_w


def t2d_table(vocab, n_vocab):
    t2d = np.full(n_vocab, len(vocab), np.int64)
    t2d[vocab] = np.arange(len(vocab))
    return t2d


class Batch:
    """CPU side of one sequence for one head: copies out of the memory map, ready to move to the GPU."""

    def __init__(self, rec, profile, t2d, min_target=8):
        self.n_prompt, self.n_tok = rec.n_prompt, rec.n_tok
        self.tokens = np.array(rec.tokens, np.int64)
        wmax = max(len(w) for w in profile["windows"].values())
        t0 = max(rec.n_prompt, wmax, min_target)
        self.tpos = np.arange(t0, rec.n_tok - 1)
        j = self.tpos - rec.n_prompt + 1                      # generated token predicted by each target
        self.top_draft = t2d[rec.top_ids[j]]
        self.top_lp = np.array(rec.top_lp[j])
        self.sampled = t2d[self.tokens[self.tpos + 1]]
        need = {profile["committed"], "full"} | {a for w in profile["windows"].values() for a in w}
        self.arrays = {}
        for a in need - {"emb"}:
            if a == "full":
                self.arrays[a] = (0, np.array(rec.full), np.array(rec.full_scale))
            else:
                li = rec.layers.index(int(a[1:]))
                self.arrays[a] = (rec.lay_start[li], np.array(rec.lay[li]), np.array(rec.lay_scale[li]))


def to_device(b, device, dtype):
    out = {"tokens": torch.from_numpy(b.tokens).to(device), "tpos": torch.from_numpy(b.tpos).to(device),
           "top_draft": torch.from_numpy(b.top_draft).to(device), "sampled": torch.from_numpy(b.sampled).to(device),
           "top_lp": torch.from_numpy(b.top_lp).to(device), "arrays": {}}
    for a, (start, q, s) in b.arrays.items():
        qt = torch.from_numpy(q).to(device).view(torch.float8_e4m3fn).to(dtype)
        st = torch.from_numpy(s).to(device).to(dtype)
        out["arrays"][a] = (start, qt * st[:, None])
    return out


def anchor_rows(d, a, positions, n_embd, dtype):
    """Hidden states of anchor a at absolute positions (any shape); zeros for the embedding-only anchor."""
    if a == "emb":
        return torch.zeros(*positions.shape, n_embd, device=positions.device, dtype=dtype)
    start, x = d["arrays"][a]
    return x[positions - start]


def build_inputs(d, profile, windows, n_embd, dtype):
    N = d["tokens"].shape[0]
    allpos = torch.arange(N, device=d["tokens"].device)
    ca = profile["committed"]
    committed = anchor_rows(d, ca, allpos, n_embd, dtype) if ca == "emb" or d["arrays"][ca][0] == 0 else None
    if committed is None:
        raise ValueError(f"committed anchor {ca} is not stored for every position")
    win = {}
    for name, anchors in windows.items():
        if name == "plain":
            win[name] = (d["tpos"], anchors, None)
            continue
        hw = torch.stack([anchor_rows(d, a, d["tpos"] - j, n_embd, dtype) for j, a in enumerate(anchors)], 1)
        win[name] = (d["tpos"], anchors, hw)
    return committed, drafter.ANCHOR_IDS[ca], win


def loader(records, profile, t2d, order, q, stop):
    for ff, i in order:
        if stop.is_set():
            return
        try:
            q.put(Batch(records[ff][i], profile, t2d))
        except Exception as e:  # noqa: BLE001
            print(f"skip {ff}:{i}: {e}", flush=True)
    q.put(None)


@torch.no_grad()
def evaluate(model, files, val, profile, windows, t2d, device, dtype, n_embd, max_seqs=60):
    model.eval()
    alphas = (1.0, 1.5, 2.0, 3.0, 5.0)
    acc = {name: [0.0, 0.0, 0.0, 0] + [0.0] * len(alphas) for name in windows}
    for ff, i in val[:max_seqs]:
        b = Batch(files[ff][i], profile, t2d)
        d = to_device(b, device, dtype)
        committed, ca, win = build_inputs(d, profile, windows, n_embd, dtype)
        with torch.autocast(device.split(":")[0], dtype=dtype):
            out = model(d["tokens"], committed, ca, win)
        p = feats.sampler_dist(d["top_lp"])
        for name, logits in out.items():
            c, t, _ = drafter.coupled_and_top(logits, d["top_draft"], p)
            lq = torch.log_softmax(logits, -1)
            a = acc[name]
            a[0] += c.sum().item()
            a[1] += t.sum().item()
            a[2] += (p * (torch.log(p.clamp_min(1e-12)) - lq.gather(1, d["top_draft"]))).sum().item()
            a[3] += c.numel()
            for ai, al in enumerate(alphas):
                a[4 + ai] += drafter.coupled_and_top(logits, d["top_draft"], p, alpha=al)[0].sum().item()
    model.train()
    return {name: {"coupled": a[0] / a[3], "top": a[1] / a[3], "kl": a[2] / a[3], "n": a[3],
                   **{f"coupled_a{al}": a[4 + ai] / a[3] for ai, al in enumerate(alphas)}} for name, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=sorted(drafter.PROFILES))
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--flint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum-targets", type=int, default=8192)
    ap.add_argument("--fp16", action="store_true", help="float16 autocast with loss scaling (V100: no bfloat16)")
    ap.add_argument("--max-seqs", type=int, default=0, help="train on at most this many sequences (data scaling)")
    ap.add_argument("--max-tok", type=int, default=8192)
    ap.add_argument("--eval-every", type=float, default=15, help="minutes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-every", type=float, default=30, help="minutes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reg", type=float, default=0.0,
                    help="weight of the feature-regression loss (newest row onto Flint's final state of that token)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    profile = drafter.PROFILES[args.head]
    windows = profile["windows"]

    vocab = np.load(args.vocab)
    files = [feats.FeatFile(p) for p in args.feats]
    allrec = [(fi, i) for fi, ff in enumerate(files) for i in range(len(ff))]
    val = [r for k, r in enumerate(allrec) if k % 33 == 0]
    train = [r for k, r in enumerate(allrec) if (k % 33 != 0 or len(allrec) < 33)
             and files[r[0]][r[1]].n_tok <= args.max_tok]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    if args.max_seqs:
        train = train[:args.max_seqs]
    n_embd = files[0][0].n_embd
    emb, head_rows, norm_w = load_flint(args.flint, vocab, device)
    t2d = t2d_table(vocab, emb.shape[0])
    model = drafter.Drafter(emb, head_rows, norm_w, n_layer=args.layers).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=args.fp16)
    cfg = {"head": args.head, "profile": profile, "vocab": args.vocab, "n_params": n_params, "train_seqs": len(train),
           "val_seqs": len(val), "train_gen_tokens": int(sum(files[f][i].n_gen for f, i in train)), "args": vars(args)}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in cfg.items() if k != "args"}), flush=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    t_start = time.time()
    budget = args.minutes * 60
    step = epoch = seen = 0
    best = -1.0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch, seen, best = ck["step"], ck["epoch"], ck["seen"], ck["best"]
        t_start = time.time() - ck["elapsed"]
        print(json.dumps({"resumed": ckpt_path, "step": step, "seqs": seen, "elapsed_min": ck["elapsed"] / 60}),
              flush=True)
    last_ckpt = time.time()

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "epoch": epoch, "seen": seen, "best": best, "elapsed": time.time() - t_start}, tmp)
        os.replace(tmp, ckpt_path)
    acc_t = 0
    sums = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}
    last_eval = time.time()
    done = False
    while not done:
        epoch += 1
        order = train[:]
        rng.shuffle(order)
        q, stop = queue.Queue(maxsize=6), threading.Event()
        th = threading.Thread(target=loader, args=(files, profile, t2d, order, q, stop), daemon=True)
        th.start()
        while True:
            b = q.get()
            if b is None:
                break
            elapsed = time.time() - t_start
            if elapsed > budget:
                done = True
                stop.set()
                break
            frac = elapsed / budget
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            d = to_device(b, device, dtype)
            committed, ca, win = build_inputs(d, profile, windows, n_embd, dtype)
            with torch.autocast(device.split(":")[0], dtype=dtype):
                out = model(d["tokens"], committed, ca, win, return_feats=True)
            p = feats.sampler_dist(d["top_lp"])
            target_f = d["arrays"]["full"][1][d["tpos"]] if args.reg > 0 else None
            loss = 0.0
            for name, (logits, f) in out.items():
                l, kl, ce = drafter.kl_loss(logits, (d["top_draft"], p), d["sampled"])
                if args.reg > 0:
                    rl = drafter.feature_loss(f, target_f)
                    l = l + args.reg * rl
                    sums["reg"] = sums.get("reg", 0.0) + rl.item() / len(out)
                loss = loss + l / len(out)
                sums["kl"] += kl.item() / len(out)
                sums["ce"] += ce.item() / len(out)
            T = d["tpos"].shape[0]
            scaler.scale(loss * T / args.accum_targets).backward()
            sums["loss"] += loss.item()
            sums["n"] += 1
            acc_t += T
            seen += 1
            if acc_t >= args.accum_targets:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                acc_t = 0
                step += 1
                if step % 20 == 0:
                    rec = {"t_min": round(elapsed / 60, 2), "epoch": epoch, "step": step, "seqs": seen, "lr": lr,
                           **{k: sums[k] / max(1, sums["n"]) for k in ("loss", "kl", "ce", "reg") if k in sums}}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    sums = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}
            if time.time() - last_ckpt > args.ckpt_every * 60:
                last_ckpt = time.time()
                save_ckpt()
            if time.time() - last_eval > args.eval_every * 60:
                last_eval = time.time()
                ev = evaluate(model, files, val, profile, windows, t2d, device, dtype, n_embd)
                rec = {"t_min": round((time.time() - t_start) / 60, 2), "step": step, "seqs": seen, "val": ev}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save(model.state_dict(), os.path.join(args.out, "head-last.pt"))
                primary = ev[next(iter(windows))]["coupled"]
                if primary > best:
                    best = primary
                    torch.save(model.state_dict(), os.path.join(args.out, "head.pt"))
        stop.set()
    ev = evaluate(model, files, val, profile, windows, t2d, device, dtype, n_embd)
    torch.save(model.state_dict(), os.path.join(args.out, "head-last.pt"))
    if ev[next(iter(windows))]["coupled"] > best:
        best = ev[next(iter(windows))]["coupled"]
        torch.save(model.state_dict(), os.path.join(args.out, "head.pt"))
    save_ckpt()
    rec = {"t_min": round((time.time() - t_start) / 60, 2), "step": step, "seqs": seen, "epochs": epoch,
           "final_val": ev, "best_val_coupled": best}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
