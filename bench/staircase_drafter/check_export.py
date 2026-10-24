"""Checks for the rig path of the staircase guesser (Q20/D3).

  python check_export.py case OUT_DIR [--fuse] [--head A] [--type f16] ...
      a random-weight head (every weight random, so every path counts), exported with export_gguf.py, and a test case
      for llama-pipe-drafter-check: a prompt committed in bulk, then guesses with commits and windows of tokens in
      flight (anchors from the head's staircase), each with the logits drafter.Drafter.forward gives (float32, CPU)
  python check_export.py real OUT_DIR --weights heads/A --vocab vocab.npy --flint FLINT_DIR --feats q51.bin [--seq 0]
      the same with a trained head and Flint's real hidden states from a llama-staircase-dump file (the anchors the
      dump stored in FP8, decoded to float32 on both sides)
  python check_export.py head OUT.gguf [--fuse] [--head A] [--type q8_0] [--target-vocab 150144]
      a random-weight head of the real shape (2,048 wide, 40,000 draft ids among Flint's ids) for exactness and speed
      checks of the executor (its guesses are wrong, the output must still be plain decoding's)

needs numpy, torch and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
case.bin layout (little-endian):
  int32 header[8] = {0x53434831, n_vocab_table, d, n_prompt, n_steps, n_vocab_draft + 1, n_tap, fuse}
  float32 table[n_vocab_table][d]                  the embedding rows (the case's token ids index this table)
  int32 tok[n_prompt]; float32 full[n_prompt][d]; float32 l9[n_prompt][d]
  per step: int32 n_commit, int32 n_window, then n_commit + n_window rows (commits first, the window oldest first) of
            {int32 token, int32 pos, float32 taps[n_tap][d]} (taps: the inputs of layers 9, 17, 25, 33, 41 and the final
            state), then float32 expected[n_vocab_draft + 1] (the newest window row's logits)
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drafter  # noqa: E402
import export_gguf  # noqa: E402

N_TAP = 6   # L9, L17, L25, L33, L41, full


def random_model(d, n_layer, fuse, n_vocab_table, n_vocab_draft, ffn, seed):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(n_vocab_table, d, generator=g) * 0.02
    head_rows = torch.randn(n_vocab_draft, d, generator=g) * 0.02
    norm_w = 1 + 0.1*torch.randn(d, generator=g)
    model = drafter.Drafter(emb, head_rows, norm_w, n_layer=n_layer, ffn=ffn, fuse=fuse)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.1*torch.randn(p.shape, generator=g))
            elif name.startswith("anchor_emb") or name == "head.bias":
                p.copy_(0.1*torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(p.shape[-1]))
    model.eval()
    return model, emb


def state_numpy(model):
    return {k: v.detach().float().numpy() for k, v in model.state_dict().items()}


def write_case(path, model, table, case_tok, model_tok, P, windows, info, taps_of, full, l9, fuse):
    """Writes case.bin: the prompt (positions 0..P-1, committed in bulk), then one guess per entry of windows (tokens in
    flight), each after committing the oldest token of the previous window. case_tok indexes `table`, model_tok is
    what the reference model embeds (the same tokens). taps_of(pos, committed) -> float32 [N_TAP, d]; full and l9 are
    the committed rows' final state and layer-9 input by position."""
    stair, deepest = info["stair"], "L41"
    d = table.shape[1]
    V1 = model.head.weight.shape[0]
    f = open(path, "wb")
    f.write(struct.pack("<8i", 0x53434831, table.shape[0], d, P, len(windows), V1, N_TAP, int(fuse)))
    f.write(np.ascontiguousarray(table, dtype=np.float32).tobytes())
    f.write(case_tok[:P].astype(np.int32).tobytes())
    f.write(np.ascontiguousarray(full[:P], dtype=np.float32).tobytes())
    f.write(np.ascontiguousarray(l9[:P], dtype=np.float32).tobytes())

    def row(tok, pos, taps):
        return struct.pack("<2i", int(tok), int(pos)) + np.ascontiguousarray(taps, dtype=np.float32).tobytes()

    tok_t = torch.from_numpy(model_tok.astype(np.int64))
    c = P            # first position not committed
    results = []
    for s, w in enumerate(windows):
        commits = []
        if s > 0:
            commits.append((case_tok[c], c, taps_of(c, True)))
            c += 1
        win = [(case_tok[p], p, taps_of(p, False)) for p in range(c, c + w)]
        f.write(struct.pack("<2i", len(commits), len(win)))
        for r in commits + win:
            f.write(row(*r))

        # reference: committed rows 0 .. c-1 (later rows are masked out for the window), the window newest first
        N = c + w
        tpos = torch.tensor([N - 1])
        anchors = [stair[j] if j < len(stair) else deepest for j in range(w)]
        hw = torch.zeros(1, w, d)
        for j in range(w):                      # offset j = 0 is the newest
            taps = win[w - 1 - j][2]
            if anchors[j] != "emb":
                hw[0, j] = torch.from_numpy(np.ascontiguousarray(taps[drafter.ANCHOR_IDS[anchors[j]] - 1]))
        window = [tpos, anchors, hw]
        cextra = None
        if fuse:
            ids = torch.tensor([drafter.ANCHOR_IDS[a] for a in anchors])
            extra = []
            for k in range(5):
                hk = torch.zeros(1, w, d)
                for j in range(w):
                    hk[0, j] = torch.from_numpy(np.ascontiguousarray(win[w - 1 - j][2][k]))
                extra.append((k, hk, (ids > k + 1)[None, :]))
            window.append(extra)
            cextra = [(0, torch.from_numpy(np.ascontiguousarray(l9[:N], dtype=np.float32)), torch.ones(N, dtype=torch.bool))]
        with torch.no_grad():
            out = model(tok_t[:N], torch.from_numpy(np.ascontiguousarray(full[:N], dtype=np.float32)),
                        drafter.ANCHOR_IDS[info["committed"]], {"w": tuple(window)}, committed_extra=cextra)
        logits = out["w"][0].float().numpy()
        f.write(logits.astype(np.float32).tobytes())
        results.append({"step": s, "commits": len(commits), "window": w, "top": int(logits.argmax())})
    f.close()
    return results


def cmd_head(args):
    rng = np.random.default_rng(args.seed)
    vocab = np.sort(rng.choice(args.target_vocab, args.draft_vocab, replace=False)).astype(np.int64)
    # the embedding table is the target's; a small stand-in is enough to build the head
    model, _ = random_model(args.d, args.layers, args.fuse, 8, args.draft_vocab, args.ffn, args.seed)
    info = export_gguf.export(state_numpy(model), args.head, args.layers, args.fuse, vocab, args.out, qtype=args.type)
    info["bytes"] = os.path.getsize(args.out)
    print(json.dumps(info))


def cmd_case(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    V = args.draft_vocab
    vocab = np.sort(rng.choice(args.table_vocab, V, replace=False)).astype(np.int64)
    model, emb = random_model(args.d, args.layers, args.fuse, args.table_vocab, V, args.ffn, args.seed)
    gguf_path = os.path.join(args.out, "drafter.gguf")
    info = export_gguf.export(state_numpy(model), args.head, args.layers, args.fuse, vocab, gguf_path, qtype=args.type)

    d, P = args.d, args.prompt
    windows = [int(x) for x in args.windows.split(",")]
    T = P + len(windows) + max(windows) + 8
    tokens = rng.integers(0, args.table_vocab, T).astype(np.int64)
    full = (rng.standard_normal((T, d))*2).astype(np.float32)
    l9 = (rng.standard_normal((T, d))*5).astype(np.float32)

    def taps_of(pos, committed_row):
        t = (rng.standard_normal((N_TAP, d))*4).astype(np.float32)
        if committed_row:
            t[0] = l9[pos]
            t[N_TAP - 1] = full[pos]
        return t

    results = write_case(os.path.join(args.out, "case.bin"), model, emb.numpy(), tokens, tokens, P, windows, info,
                         taps_of, full, l9, args.fuse)
    print(json.dumps({"gguf": gguf_path, "case": os.path.join(args.out, "case.bin"), "export": info, "steps": results}))


def cmd_real(args):
    import feats
    import train
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.weights, "config.json")))
    sd_t = torch.load(os.path.join(args.weights, "head.pt"), map_location="cpu")
    sd_t = sd_t.get("model", sd_t)
    vocab = np.load(args.vocab)
    emb, head_rows, norm_w = train.load_flint(args.flint, vocab, "cpu")
    emb = emb.float()
    fuse = cfg["args"].get("fuse", False)
    layers = cfg["args"].get("layers", 2)
    model = drafter.Drafter(emb, head_rows, norm_w, n_layer=layers, fuse=fuse)
    model.load_state_dict(sd_t, strict=False)
    model.eval()
    gguf_path = os.path.join(args.out, "drafter.gguf")
    info = export_gguf.export({k: v.float().numpy() for k, v in sd_t.items() if hasattr(v, "float")}, cfg["head"],
                              layers, fuse, vocab, gguf_path, qtype=args.type)

    rec = feats.FeatFile(args.feats)[args.seq]
    windows = [int(x) for x in args.windows.split(",")]
    # positions committed in bulk before the guesses: the prompt and the first generated tokens
    P = max(1, min(args.prompt_max, rec.n_tok - (len(windows) + max(windows) + 4)))
    T = min(rec.n_tok, P + len(windows) + max(windows) + 4)
    pos_all = np.arange(T)
    full = rec.dequant("full", pos_all)
    l9 = rec.dequant(9, pos_all)
    deep = {L: rec.dequant(L, np.arange(max(P, rec.lay_start[rec.layers.index(L)]), T)) for L in (17, 25, 33, 41)}

    def taps_of(pos, committed_row):
        t = np.zeros((N_TAP, rec.n_embd), np.float32)
        t[0] = l9[pos]
        for k, L in enumerate((17, 25, 33, 41)):
            b = max(P, rec.lay_start[rec.layers.index(L)])
            t[k + 1] = deep[L][pos - b] if pos >= b else 0.0
        t[N_TAP - 1] = full[pos]
        return t

    model_tok = np.array(rec.tokens[:T], np.int64)
    used, case_tok = np.unique(model_tok, return_inverse=True)
    table = emb[torch.from_numpy(used)].numpy()
    results = write_case(os.path.join(args.out, "case.bin"), model, table, case_tok.astype(np.int64), model_tok, P,
                         windows, info, taps_of, full, l9, fuse)
    print(json.dumps({"gguf": gguf_path, "case": os.path.join(args.out, "case.bin"), "export": info,
                      "seq": int(rec.seq_id), "prompt": P, "steps": results}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("case", "real", "head"))
    ap.add_argument("out")
    ap.add_argument("--head", default="A", choices=sorted(drafter.PROFILES))
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--type", default=None, choices=("q8_0", "f16", "f32"))
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--draft-vocab", type=int, default=None)
    ap.add_argument("--table-vocab", type=int, default=6000)
    ap.add_argument("--target-vocab", type=int, default=150144)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--windows", default="4,4,4,5,6,1,2,4,4", help="tokens in flight per guess")
    ap.add_argument("--seed", type=int, default=7)
    # real
    ap.add_argument("--weights", help="train.py output folder (config.json, head.pt)")
    ap.add_argument("--vocab")
    ap.add_argument("--flint", help="flint_weights.py output folder")
    ap.add_argument("--feats", help="llama-staircase-dump file (e.g. Q51's held-out transcripts)")
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--prompt-max", type=int, default=600, help="real: positions committed in bulk before the guesses")
    args = ap.parse_args()
    if args.cmd == "head":
        args.type = args.type or "q8_0"
        args.draft_vocab = args.draft_vocab or 40000
        cmd_head(args)
    elif args.cmd == "real":
        args.type = args.type or "f16"
        cmd_real(args)
    else:
        args.type = args.type or "f16"
        args.draft_vocab = args.draft_vocab or 4000
        cmd_case(args)


if __name__ == "__main__":
    main()
