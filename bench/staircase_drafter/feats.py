"""Read llama-staircase-dump output, write its input, and turn Flint's top-20 log-probabilities into its sampler's
distribution.

usage:
  python feats.py seqs OUT_SEQS.txt OUT_MAP.jsonl GEN.jsonl [GEN.jsonl ...] [--exclude MAP.jsonl ...] [--take N]
                       [--max-tok N] [--seed N] [--only IDS_FILE]
      seqs.txt lines for the dump ("<seq_id> <n_prompt> <tokens...>") from generate.py's rows, plus a map
      seq_id -> row id and category. The seq_id is the prompt's number (the numeric end of its id), so it is stable
      across calls; rows already in an --exclude map are left out, and --take picks a random subset. Agent sessions
      (bench/agentic_data's sessions_seq.jsonl) keep their generated spans in the map (gen_ranges): the dump replays
      the tool results too, and training and scoring read only the spans (train.py and score.py --spans).
  python feats.py info FEAT.bin [FEAT.bin ...]
      sequences, tokens and bytes per file, and a decode check of the first record.
  python feats.py pack OUT.bin FEAT.bin [FEAT.bin ...] --layers 9,17,25,33,41
      copy the records with only the given layers (and the final state, tokens and top-k) to a smaller file, e.g. a
      six-card subset of a superset dump onto a fast disk for training.
"""
import argparse
import json
import mmap
import os
import sys

import numpy as np

MAGICS = (0x51320002, 0x51320003)   # v2: layer starts implied (first layer from 0, others from a0); v3: stored
ANCHORS = ("emb", "L9", "L17", "L25", "L33", "L41", "full")   # anchor index 0..6; 0 = the token embedding only


def e4m3_table():
    """FP8 E4M3 (float8_e4m3fn) code -> float32."""
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8 * 2.0 ** -6
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


class Record:
    """One dumped sequence: numpy views into the memory-mapped file."""

    def __init__(self, mm, off, fd=None):
        """fd: the file opened for reading, for read_rows (None: copy out of the map)."""
        self.fd = fd
        h = np.frombuffer(mm, np.int32, 8, off)
        assert h[7] in MAGICS, f"bad record at {off}"
        self.off = off
        self.seq_id, self.n_prompt, self.n_tok, self.a0, nl, self.k, self.n_embd = (int(x) for x in h[:7])
        n, e = self.n_tok, self.n_embd
        self.n_gen = n - self.n_prompt
        p = off + 32
        self.layers = [int(x) for x in np.frombuffer(mm, np.int32, nl, p)]
        p += 4 * nl
        if h[7] == 0x51320003:
            starts = [int(x) for x in np.frombuffer(mm, np.int32, nl, p)]
            p += 4 * nl
        else:
            starts = [0] + [self.a0] * (nl - 1)
        self.tokens = np.frombuffer(mm, np.int32, n, p)
        p += 4 * n
        self.full_scale = np.frombuffer(mm, np.float16, n, p)
        p += 2 * n
        self.full_off = p
        self.full = np.frombuffer(mm, np.uint8, n * e, p).reshape(n, e)
        p += n * e
        self.lay_start, self.lay_scale, self.lay, self.lay_off = [], [], [], []
        for li in range(nl):
            b = starts[li]
            self.lay_start.append(b)
            self.lay_scale.append(np.frombuffer(mm, np.float16, n - b, p))
            p += 2 * (n - b)
            self.lay_off.append(p)
            self.lay.append(np.frombuffer(mm, np.uint8, (n - b) * e, p).reshape(n - b, e))
            p += (n - b) * e
        g, k = max(0, self.n_gen), self.k
        self.top_ids = np.frombuffer(mm, np.int32, g * k, p).reshape(g, k)
        p += 4 * g * k
        self.top_lp = np.frombuffer(mm, np.float32, g * k, p).reshape(g, k)
        p += 4 * g * k
        self.end = p

    def read_rows(self, which, lo=0, hi=None):
        """(first position, FP8 rows [rows, n_embd] uint8, row scales float16) of an anchor ("full" or a layer
        number) for the stored positions in [lo, hi), copied out. Through FeatFile the rows come with one exact read,
        without the page cache's read-ahead, which on an HDD set to 8 MB of read-ahead would read several times the
        block."""
        if which == "full":
            b, q, s, off = 0, self.full, self.full_scale, self.full_off
        else:
            li = self.layers.index(which)
            b, q, s, off = self.lay_start[li], self.lay[li], self.lay_scale[li], self.lay_off[li]
        r0, r1 = max(0, lo - b), max(0, min(len(q), (self.n_tok if hi is None else hi) - b))
        r1 = max(r0, r1)
        b, q, s, off = b + r0, q[r0:r1], s[r0:r1], off + r0 * self.n_embd
        if self.fd is None:
            return b, np.array(q), np.array(s)
        buf = bytearray(q.nbytes)
        view, got = memoryview(buf), 0
        while got < q.nbytes:
            k = os.preadv(self.fd, [view[got:]], off + got)
            if k <= 0:
                raise IOError(f"short read at {off + got}")
            got += k
        return b, np.frombuffer(buf, np.uint8).reshape(q.shape), np.array(s)

    def dequant(self, which, rows=None):
        """float32 [rows, n_embd] of an anchor ("full" or a layer number) at absolute positions `rows`."""
        if which == "full":
            q, s, b = self.full, self.full_scale, 0
        else:
            li = self.layers.index(which)
            q, s, b = self.lay[li], self.lay_scale[li], self.lay_start[li]
        rows = np.arange(b, self.n_tok) if rows is None else np.asarray(rows)
        return E4M3[q[rows - b]] * s[rows - b, None].astype(np.float32)


class FeatFile:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, np.uint8, mode="r")
        # each record header would otherwise pull the device's whole read-ahead window (8 MB on our HDDs) into memory;
        # bulk rows go through read_rows instead
        self.mm._mmap.madvise(mmap.MADV_RANDOM)
        self.fd = os.open(path, os.O_RDONLY)
        self.offsets = []
        off = 0
        while off + 32 <= len(self.mm):
            try:
                r = Record(self.mm, off)
            except (AssertionError, ValueError):
                break
            if r.end > len(self.mm):
                break
            self.offsets.append(off)
            off = r.end

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        return Record(self.mm, self.offsets[i], self.fd)


def sampler_dist(lp, top_p=0.95, min_p=0.0):
    """Flint's sampler chain at temperature 1 on sorted top-k log-probabilities (numpy or torch, [..., k]): top-k (the k
    given), top-p keeping the token that crosses p, min-p relative to the most likely token, renormalized. Returns the
    kept probabilities, zero for dropped tokens (pipeline_sim.sampler_probs, vectorized)."""
    if isinstance(lp, np.ndarray):
        w = np.exp(lp - lp[..., :1])
        p = w / w.sum(-1, keepdims=True)
        before = np.cumsum(p, -1) - p
        keep = (before < top_p) & (p >= min_p * p[..., :1])
        p = np.where(keep, p, 0.0)
        return p / p.sum(-1, keepdims=True)
    import torch
    p = torch.softmax(lp - lp[..., :1], -1)
    before = torch.cumsum(p, -1) - p
    keep = (before < top_p) & (p >= min_p * p[..., :1])
    p = torch.where(keep, p, torch.zeros_like(p))
    return p / p.sum(-1, keepdim=True)


def load_gen(paths):
    rows = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            rows[r["id"]] = r
    return [rows[k] for k in sorted(rows, key=lambda s: int(s.split("_")[-1]))]


def cmd_seqs(args):
    import random
    done = set()
    for path in args.exclude or []:
        done |= {json.loads(l)["id"] for l in open(path)}
    only = {x.strip() for x in open(args.only) if x.strip()} if args.only else None
    rows = [r for r in load_gen(args.gen) if r["id"] not in done and len(r["gen_tokens"]) >= 2
            and len(r["prompt_tokens"]) + len(r["gen_tokens"]) <= args.max_tok
            and (only is None or r["id"] in only)]
    if args.take and args.take < len(rows):
        rows = sorted(random.Random(args.seed).sample(rows, args.take), key=lambda r: int(r["id"].split("_")[-1]))
    nums = [int(r["id"].split("_")[-1]) for r in rows]
    if len(set(nums)) != len(nums):   # e.g. Q51 (chat_0, code_0, ...): a dump of it needs its own row numbering
        raise SystemExit("rows share an id number, so their seq_ids would repeat: number the rows instead")
    with open(args.out_seqs, "w") as f, open(args.out_map, "w") as m:
        for r in rows:
            sid = int(r["id"].split("_")[-1])
            f.write(f"{sid} {len(r['prompt_tokens'])} " + " ".join(map(str, r["prompt_tokens"] + r["gen_tokens"])) + "\n")
            m.write(json.dumps({"seq_id": sid, "id": r["id"], "category": r["category"],
                                "n_prompt": len(r["prompt_tokens"]), "n_gen": len(r["gen_tokens"]),
                                **({"gen_ranges": r["gen_ranges"]} if "gen_ranges" in r else {})}) + "\n")
    n_gen = sum(gen_count(r) for r in rows)
    print(f"{len(rows)} sequences, {n_gen} generated tokens")


def gen_count(r):
    """Generated tokens of a row: all of gen_tokens, or for an agent session (gen_ranges) the tokens of its spans."""
    return sum(e - s for s, e in r["gen_ranges"]) if "gen_ranges" in r else len(r["gen_tokens"])


def load_spans(paths):
    """{seq_id: [[start, end), ...]}: the generated spans of agent sessions (absolute token positions) from feats.py
    seqs or feed_split.py maps, or from the sessions_seq.jsonl rows themselves (seq_id = the id's number); sequences
    without spans are generated after their prompt throughout."""
    spans = {}
    for path in paths or []:
        for line in open(path):
            m = json.loads(line)
            if "gen_ranges" in m:
                spans[m["seq_id"] if "seq_id" in m else int(m["id"].split("_")[-1])] = m["gen_ranges"]
    return spans


def cmd_info(args):
    for path in args.feat:
        ff = FeatFile(path)
        n_tok = n_gen = 0
        for i in range(len(ff)):
            r = ff[i]
            n_tok += r.n_tok
            n_gen += r.n_gen
        print(f"{path}: {len(ff)} sequences, {n_tok} tokens, {n_gen} generated, {len(ff.mm) / 1e9:.2f} GB")
        if len(ff):
            r = ff[0]
            x = r.dequant("full", [r.n_prompt - 1])
            print(f"  seq {r.seq_id}: layers {r.layers}, full-state rms {np.sqrt((x ** 2).mean()):.3f}, "
                  f"top-1 of the first generated token {r.top_ids[0, 0]} (actual {r.tokens[r.n_prompt]}), "
                  f"p {np.exp(r.top_lp[0, 0]):.3f}")


def cmd_pack(args):
    import os
    keep = [int(x) for x in args.layers.split(",")]
    have = set()
    if os.path.exists(args.out):   # resume: drop a cut-off last record, skip the sequences already packed
        prev = FeatFile(args.out)
        have = {prev[i].seq_id for i in range(len(prev))}
        end = prev[len(prev) - 1].end if len(prev) else 0
        del prev
        os.truncate(args.out, end)
    n_seq = 0
    with open(args.out, "ab") as out:
        for path in args.feat:
            ff = FeatFile(path)
            for i in range(len(ff)):
                r = ff[i]
                if r.seq_id in have:
                    continue
                idx = [r.layers.index(L) for L in keep]
                hdr = np.array([r.seq_id, r.n_prompt, r.n_tok, r.a0, len(keep), r.k, r.n_embd, 0x51320003], np.int32)
                out.write(hdr.tobytes())
                out.write(np.array(keep, np.int32).tobytes())
                out.write(np.array([r.lay_start[j] for j in idx], np.int32).tobytes())
                out.write(r.tokens.tobytes())
                out.write(r.full_scale.tobytes())
                out.write(r.full.tobytes())
                for j in idx:
                    out.write(r.lay_scale[j].tobytes())
                    out.write(r.lay[j].tobytes())
                out.write(r.top_ids.tobytes())
                out.write(r.top_lp.tobytes())
                n_seq += 1
            print(f"{path}: {len(ff)} records", flush=True)
    print(f"packed {n_seq} records -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seqs")
    a.add_argument("out_seqs")
    a.add_argument("out_map")
    a.add_argument("gen", nargs="+")
    a.add_argument("--exclude", nargs="*")
    a.add_argument("--take", type=int, default=0)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--max-tok", type=int, default=8192)
    a.add_argument("--only", help="file of row ids (one per line): only these rows")
    a = sub.add_parser("info")
    a.add_argument("feat", nargs="+")
    a = sub.add_parser("pack")
    a.add_argument("out")
    a.add_argument("feat", nargs="+")
    a.add_argument("--layers", default="9,17,25,33,41")
    args = ap.parse_args()
    {"seqs": cmd_seqs, "info": cmd_info, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
