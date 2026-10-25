"""Train one staircase-guesser head (Q20/D1) on llama-staircase-dump features.

usage: python train.py --head A --feats F1.bin [F2.bin ...] --vocab vocab.npy --flint FLINT_DIR --out OUT_DIR
                       [--minutes 180] [--lr 2e-4] [--accum-targets 8192] [--fp16] [--max-seqs N] [--seed 1]
FLINT_DIR holds flint_weights.py's output. The schedule is time-based: linear warm-up, then cosine decay to 10 % over
--minutes, so a run ends on time whatever the GPU. Every 3 % of the sequences (every 33rd) are held out for validation.
OUT_DIR gets head.pt (the weights with the best validation acceptance of the first window, or for --general the mean
over the staircases of its card counts, without Flint's embedding table), head-last.pt, config.json, log.jsonl and ckpt.pt (weights, optimizer and scaler state, step and elapsed time,
written every --ckpt-every minutes). Creating OUT_DIR/STOP ends the run early with its final evaluation and save.
--resume continues from ckpt.pt: the time budget counts the time already spent, so
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

    def __init__(self, rec, profile, t2d, min_target=8, all_anchors=False, sampler=None):
        """sampler: a function of the record's recorded layers returning one window ({"sample": names}); the window
        is then drawn here, in the loader thread, and only its layers are read."""
        self.n_prompt, self.n_tok = rec.n_prompt, rec.n_tok
        self.tokens = np.array(rec.tokens, np.int64)
        self.window = sampler(rec.layers) if sampler else None
        wmax = max([len(w) for w in profile["windows"].values()] + ([8] if all_anchors else [])
                   + ([len(self.window["sample"])] if self.window else []))
        # window rows reach back wmax - 1 positions; mid layers are stored from n_prompt - 8 (the dump's margin)
        t0 = max(rec.n_prompt + max(0, wmax - 9), wmax, min_target)
        self.tpos = np.arange(t0, rec.n_tok - 1)
        j = self.tpos - rec.n_prompt + 1                      # generated token predicted by each target
        self.top_draft = t2d[rec.top_ids[j]]
        self.top_lp = np.array(rec.top_lp[j])
        self.sampled = t2d[self.tokens[self.tpos + 1]]
        need = {profile["committed"], "full"}
        if not self.window:   # a sampled window is the only one trained: read only its layers
            need |= {a for w in profile["windows"].values() for a in w}
        if all_anchors:   # fused anchors or sampled windows read every six-card anchor
            need |= set(SIX_CARD)
        if self.window:
            need |= set(self.window["sample"])
        self.arrays = {}
        for a in need - {"emb"}:
            self.arrays[a] = rec.read_rows("full" if a == "full" else int(a[1:]))


def to_device(b, device, dtype):
    out = {"tokens": torch.from_numpy(b.tokens).to(device), "tpos": torch.from_numpy(b.tpos).to(device),
           "top_draft": torch.from_numpy(b.top_draft).to(device), "sampled": torch.from_numpy(b.sampled).to(device),
           "top_lp": torch.from_numpy(b.top_lp).to(device), "arrays": {}}
    for a, (start, q, s) in b.arrays.items():
        qt = torch.from_numpy(q).to(device).view(torch.float8_e4m3fn).to(dtype)
        st = torch.from_numpy(s).to(device).to(dtype)
        out["arrays"][a] = (start, qt * st[:, None])
    return out


SIX_CARD = ("L9", "L17", "L25", "L33", "L41", "full")


def sample_window(profile, rng, probs, p_emb=0.35, max_older=7):
    """One window for a sequence: the profile's nominal staircase, its after-miss window, or a random staircase. The
    random one covers any layer split of 5-9 cards under the six-card anchor map: the newest token at L9 or, with
    probability p_emb, as its embedding (card 1 holding fewer than 9 layers), then 0..max_older older tokens in flight
    at non-decreasing depths from L9, L17, L25, L33, L41 (older tokens beyond the window are complete).
    probs: (p_stair, p_miss, p_random)."""
    wins = profile["windows"]
    stair = next((w for n, w in wins.items() if n != "miss"), wins.get("miss"))
    u = rng.random()
    if u < probs[0]:
        return {"sample": stair}
    if u < probs[0] + probs[1] and "miss" in wins:
        return {"sample": wins["miss"]}
    newest = "emb" if rng.random() < p_emb else "L9"
    older = sorted((rng.choice(SIX_CARD[:5]) for _ in range(rng.randint(0, max_older))),
                   key=lambda a: drafter.ANCHOR_IDS[a])
    return {"sample": [newest] + older}


# card counts an owner may run and how often the general sampler draws them (6 and 9 P106 cards first, then 3-5 large
# cards and 10-12 small ones)
CARD_WEIGHTS = {3: 0.08, 4: 0.08, 5: 0.08, 6: 0.25, 7: 0.06, 8: 0.06, 9: 0.25, 10: 0.05, 11: 0.04, 12: 0.05}


def random_split(n, rng, p_light=0.3):
    """Layers per card for n cards: as even as possible, with probability p_light a light last card (1-3 layers next to
    the output head), then up to two single layers moved between random cards (-ts tweaks)."""
    split = drafter.even_split(n, rng.randint(1, 3) if n >= 3 and rng.random() < p_light else 0)
    for _ in range(rng.randint(0, 2)):
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j and split[i] > 1:
            split[i] -= 1
            split[j] += 1
    return split


# the splits owners run per card count, for heads trained on a few card counts only (--cards): the even split, and
# for nine cards also the one with two layers next to the output head (6/6/6/6/6/6/6/5/2)
STANDARD_SPLITS = {9: [drafter.even_split(9), drafter.even_split(9, 2)]}


def card_splits(token):
    """(window name, split) pairs of a --cards token: "N" the standard splits of N cards, "Ne" the even split only,
    "Nl" the split with two layers on the last card only (named as in score.py: gN, gNlight)."""
    if token.endswith("e"):
        return [(f"g{token[:-1]}", drafter.even_split(int(token[:-1])))]
    if token.endswith("l"):
        return [(f"g{token[:-1]}light", drafter.even_split(int(token[:-1]), 2))]
    n = int(token)
    return [(f"g{n}" + ("light" if k else ""), s) for k, s in enumerate(STANDARD_SPLITS.get(n, [drafter.even_split(n)]))]


def sample_general(recorded, rng, kinds=(0.6, 0.15, 0.25), cards=None):
    """One window of a random 3-12-card split: card count by CARD_WEIGHTS, a random split, then the head on card 1
    after its stage, a spare card before the cycle, or the first guess after a miss (probabilities kinds); each depth
    becomes the deepest recorded layer at or below it. With cards (a list of --cards tokens, see card_splits;
    repeating one weights it), the token is drawn from the list uniformly and the split from its splits."""
    if cards:
        token = rng.choice(cards)
        kind = rng.choices(("stair", "pre", "miss"), weights=kinds)[0]
        split = rng.choice([s for _, s in card_splits(token)])
    else:
        n = rng.choices(list(CARD_WEIGHTS), weights=list(CARD_WEIGHTS.values()))[0]
        kind = rng.choices(("stair", "pre", "miss"), weights=kinds)[0]
        split = random_split(n, rng)
    return {"sample": [drafter.depth_name(x, recorded) for x in drafter.split_depths(split, kind)]}


def fuse_extra(d, anchors, positions):
    """For fused anchors: (k, states, mask) for L9..L41 (k = 0..4) at positions [T, W], where the row's own anchor
    (anchors[j] for column j) is deeper than anchor k."""
    ids = torch.tensor([drafter.ANCHOR_IDS[a] for a in anchors], device=positions.device)
    out = []
    for k, name in enumerate(SIX_CARD[:5]):
        mask = (ids > k + 1).expand(positions.shape)
        if mask.any():
            out.append((k, anchor_rows(d, name, positions, None, None), mask))
    return out


def anchor_rows(d, a, positions, n_embd, dtype):
    """Hidden states of anchor a at absolute positions (any shape); zeros for the embedding-only anchor."""
    if a == "emb":
        return torch.zeros(*positions.shape, n_embd, device=positions.device, dtype=dtype)
    start, x = d["arrays"][a]
    return x[positions - start]


def build_inputs(d, profile, windows, n_embd, dtype, fuse=False):
    """Committed rows, their anchor id, the window inputs and (with fuse) the committed rows' extra anchors: L9 next to
    the final state, the only shallower anchor stored for every position."""
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
        if fuse:
            wpos = d["tpos"][:, None] - torch.arange(len(anchors), device=allpos.device)[None, :]
            win[name] = (d["tpos"], anchors, hw, fuse_extra(d, anchors, wpos))
        else:
            win[name] = (d["tpos"], anchors, hw)
    cextra = None
    if fuse and ca == "full":
        cextra = [(0, anchor_rows(d, "L9", allpos, None, None), torch.ones(N, dtype=torch.bool, device=allpos.device))]
    return committed, ca, win, cextra


def loader(records, profile, t2d, order, q, stop, all_anchors=False, sampler=None):
    for ff, i in order:
        if stop.is_set():
            return
        try:
            q.put(Batch(records[ff][i], profile, t2d, all_anchors=all_anchors, sampler=sampler))
        except Exception as e:  # noqa: BLE001
            print(f"skip {ff}:{i}: {e}", flush=True)
    q.put(None)


@torch.no_grad()
def evaluate(model, files, val, profile, windows, t2d, device, dtype, n_embd, max_seqs=60, fuse=False):
    model.eval()
    alphas = (1.0, 1.5, 2.0, 3.0, 5.0)
    acc = {name: [0.0, 0.0, 0.0, 0] + [0.0] * len(alphas) for name in windows}
    for ff, i in val[:max_seqs]:
        b = Batch(files[ff][i], profile, t2d, all_anchors=fuse)
        d = to_device(b, device, dtype)
        committed, ca, win, cextra = build_inputs(d, profile, windows, n_embd, dtype, fuse)
        with torch.autocast(device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
            out = model(d["tokens"], committed, ca, win, committed_extra=cextra)
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
    ap.add_argument("--init-from", help="head.pt or ckpt.pt to start from (weights only; new parameters keep their "
                                         "zero initialization; fresh optimizer and schedule)")
    ap.add_argument("--window-sample", help="P_STAIR,P_MISS,P_RANDOM: train one window per sequence, drawn this way")
    ap.add_argument("--loss", choices=("kl", "lk"), default="kl")
    ap.add_argument("--eta", type=float, default=3.0, help="LK loss: lambda = exp(-eta * coupled match)")
    ap.add_argument("--fuse", action="store_true", help="give every row each shallower anchor its token has reached")
    ap.add_argument("--noise", type=float, default=0.0, help="uniform noise amplitude after each anchor's RMS norm")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-seqs", type=int, default=60, help="validation sequences for the final evaluation")
    ap.add_argument("--general", action="store_true",
                    help="depth-general head: anchors at any recorded layer, windows of random 3-12-card splits "
                         "(sample_general); no fused anchors")
    ap.add_argument("--cards", help="tokens: with --general, train on these splits only: N = N cards' standard splits "
                                    "(STANDARD_SPLITS), Ne = the even split, Nl = two layers on the last card; repeats "
                                    "weight (e.g. 6,9 or 9e,9l,9l)")
    ap.add_argument("--reg", type=float, default=0.0,
                    help="weight of the feature-regression loss (newest row onto Flint's final state of that token)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    # FP32 on the CPU (checks): bfloat16 autocast on CPUs without bfloat16 instructions runs nearly single-threaded
    dtype = torch.float32 if device == "cpu" else torch.float16 if args.fp16 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    profile = drafter.PROFILES[args.head]
    windows = profile["windows"]

    vocab = np.load(args.vocab)
    files = [feats.FeatFile(p) for p in args.feats]
    allrec = [(fi, i) for fi, ff in enumerate(files) for i in range(len(ff))]
    val = [r for k, r in enumerate(allrec) if k % 33 == 0 and files[r[0]][r[1]].n_tok <= args.max_tok]
    train = [r for k, r in enumerate(allrec) if (k % 33 != 0 or len(allrec) < 33)
             and files[r[0]][r[1]].n_tok <= args.max_tok]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    if args.max_seqs:
        train = train[:args.max_seqs]
    n_embd = files[0][0].n_embd
    emb, head_rows, norm_w = load_flint(args.flint, vocab, device)
    t2d = t2d_table(vocab, emb.shape[0])
    if args.general and args.fuse:
        raise SystemExit("--general does not support --fuse")
    if args.cards and not args.general:
        raise SystemExit("--cards needs --general")
    model = drafter.Drafter(emb, head_rows, norm_w, n_layer=args.layers, fuse=args.fuse, noise=args.noise,
                            dropout=args.dropout, general=args.general).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.init_general(sd)
        print(json.dumps({"init_from": args.init_from, "missing": missing, "unexpected": unexpected}), flush=True)
    sample_probs = [float(x) for x in args.window_sample.split(",")] if args.window_sample else None
    all_anchors = args.fuse or sample_probs is not None
    general_sampler = None
    if args.general:
        srng = random.Random(args.seed + 7)
        cards = args.cards.split(",") if args.cards else None
        general_sampler = lambda layers: sample_general(layers, srng, cards=cards)   # noqa: E731
    # validation windows; the best checkpoint is the one with the highest mean acceptance over the primary ones
    eval_profile, primary = profile, [next(iter(windows))]
    if args.general:   # the staircases of the card counts the head is for (6 and 9 for the 3-12 head)
        common = sorted(set.intersection(*(set(ff[0].layers) for ff in files)))
        eval_profile, primary = {"committed": profile["committed"], "windows": dict(windows)}, []
        for token in cards or ("6", "9"):   # a repeated token weights the choice of checkpoint as it weights training
            for name, split in card_splits(token):
                eval_profile["windows"][name] = [drafter.depth_name(x, common) for x in drafter.split_depths(split)]
                primary.append(name)
    score_of = lambda ev: sum(ev[w]["coupled"] for w in primary) / len(primary)   # noqa: E731
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
        th = threading.Thread(target=loader, args=(files, profile, t2d, order, q, stop, all_anchors, general_sampler),
                              daemon=True)
        th.start()
        while True:
            tw = time.time()
            b = q.get()
            sums["wait"] = sums.get("wait", 0.0) + time.time() - tw
            if b is None:
                break
            elapsed = time.time() - t_start
            if elapsed > budget or os.path.exists(os.path.join(args.out, "STOP")):
                done = True
                stop.set()
                break
            frac = elapsed / budget
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            d = to_device(b, device, dtype)
            if b.window:
                step_windows = b.window
            else:
                step_windows = sample_window(profile, rng, sample_probs) if sample_probs else windows
            committed, ca, win, cextra = build_inputs(d, profile, step_windows, n_embd, dtype, args.fuse)
            with torch.autocast(device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
                out = model(d["tokens"], committed, ca, win, return_feats=True, committed_extra=cextra)
            p = feats.sampler_dist(d["top_lp"])
            target_f = d["arrays"]["full"][1][d["tpos"]] if args.reg > 0 else None
            loss = 0.0
            for name, (logits, f) in out.items():
                if args.loss == "lk":
                    l, kl, ce = drafter.lk_loss(logits, (d["top_draft"], p), d["sampled"], eta=args.eta)
                else:
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
                           **{k: sums[k] / max(1, sums["n"]) for k in ("loss", "kl", "ce", "reg", "wait") if k in sums}}
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    sums = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0, "wait": 0.0}
            if time.time() - last_ckpt > args.ckpt_every * 60:
                last_ckpt = time.time()
                save_ckpt()
            if time.time() - last_eval > args.eval_every * 60:
                last_eval = time.time()
                ev = evaluate(model, files, val, eval_profile, eval_profile["windows"], t2d, device, dtype, n_embd,
                              fuse=args.fuse)
                rec = {"t_min": round((time.time() - t_start) / 60, 2), "step": step, "seqs": seen, "val": ev}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save(model.state_dict(), os.path.join(args.out, "head-last.pt"))
                if score_of(ev) > best:
                    best = score_of(ev)
                    torch.save(model.state_dict(), os.path.join(args.out, "head.pt"))
        stop.set()
    ev = evaluate(model, files, val, eval_profile, eval_profile["windows"], t2d, device, dtype, n_embd,
                  max_seqs=args.val_seqs, fuse=args.fuse)
    torch.save(model.state_dict(), os.path.join(args.out, "head-last.pt"))
    if score_of(ev) > best:
        best = score_of(ev)
        torch.save(model.state_dict(), os.path.join(args.out, "head.pt"))
    save_ckpt()
    rec = {"t_min": round((time.time() - t_start) / 60, 2), "step": step, "seqs": seen, "epochs": epoch,
           "final_val": ev, "best_val_coupled": best}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
