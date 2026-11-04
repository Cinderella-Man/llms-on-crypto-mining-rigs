#!/usr/bin/env python3
"""Fit on nine 6 GB cards of Flash-Next files that keep N of their 512 routed experts in every layer, from GGUF headers
only: bench/file_choice/fit.py's grid and costs (imported) on headers whose routed-expert tensors (`ffn_gate_exps`,
`ffn_up_exps`, `ffn_down_exps`) and router rows (`ffn_gate_inp`, `exp_probs_b`) are scaled to N/512, which is exact
for a file pruned by slicing those tensors.
  python3 -I prune_fit.py --keep 512,480,448,384 [--kv f16] [--drafts none,dflash2] [--bytes bytes.tsv] \
    [--md fit.md] NAME=SHARD[,SHARD...] ... > fit.tsv
fit.tsv has fit.py's columns plus `keep`; bytes.tsv has, per file and N, the MiB of routed experts and in all on the
cards; fit.md, per file and N, the largest micro-batch that leaves 150 MiB on every card at each context (whole layers
/ routed-expert tensors moved by -ot).
Standard library only.
"""
import argparse, functools, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "bench", "file_choice")))
import fit  # noqa: E402  (puts scripts/ on the path and caches split_plan.read_header)
import split_plan  # noqa: E402

SCALED = re.compile(r"^blk\.\d+\.(ffn_(gate|up|down)_exps\.weight|ffn_gate_inp\.weight|exp_probs_b(\.bias)?)$")
MARGIN = 150
raw_header = split_plan.read_header.__wrapped__
STATE = {"keep": 512, "experts": 512}


@functools.lru_cache(maxsize=None)
def cached(path):
    return raw_header(path)


@functools.lru_cache(maxsize=None)
def scaled(path, keep, experts):
    kv, sizes = cached(path)
    f = keep / experts
    return kv, {k: (v * f if SCALED.match(k) else v) for k, v in sizes.items()}


def read_header(path):
    return scaled(path, STATE["keep"], STATE["experts"])


split_plan.read_header = read_header


def mib(shards):
    routed = cards = 0.0
    for s in shards:
        _, sizes = read_header(s)
        for k, v in sizes.items():
            if split_plan.EXPERTS.match(k):
                routed += v
            if not split_plan.HOST_ONLY.match(k):
                cards += v
    return routed / 2 ** 20, cards / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="NAME=SHARD[,SHARD...]")
    ap.add_argument("--keep", default="512,480,448,384,320,256")
    ap.add_argument("--cards", type=int, default=9)
    ap.add_argument("--free", default="6001,6003,6003,6003,6003,6003,6001,6007,6007")
    ap.add_argument("--kv", default="f16", help="cache types of fit.COSTS to plan (f16, q8_0)")
    ap.add_argument("--drafts", default="none,dflash2")
    ap.add_argument("--bytes")
    ap.add_argument("--md")
    a = ap.parse_args()
    keeps = [int(k) for k in a.keep.split(",")]
    kvs = set(a.kv.split(","))
    rows, sizes = [], []
    print("file\tkeep\tctx\tkv\tub\tdraft\tmode\tsmallest_free_mib\tts\tmoved\tot")
    for spec in a.files:
        name, shards = spec.split("=", 1)
        shards = shards.split(",")
        kv0, _ = cached(shards[0])
        STATE["experts"] = kv0[f"{kv0['general.architecture']}.expert_count"]
        for keep in keeps:
            STATE["keep"] = keep
            sizes.append((name, keep) + mib(shards))
            for line in fit.COSTS.strip().splitlines():
                kv, ub, ctx, k, c0, c, d = line.split()
                if kv not in kvs:
                    continue
                ctx = int(ctx)
                for draft in a.drafts.split(","):
                    extra = {"none": 0,
                             "dflash2": fit.DFLASH2_WEIGHTS + fit.DFLASH2_COMPUTE + fit.DFLASH2_CTX_64K * ctx / 65536,
                             "mtp": float(d), "mtp-shr": float(d) - fit.MTP_HEAD}[draft]
                    state = fit.STATE if draft == "none" else fit.STATE_DRAFT
                    base = ["--cards", str(a.cards), "--free", a.free, "--kv-mib", k, "--state-mib", str(state),
                            "--compute", f"{c0},{c}", "--last-extra", f"{extra:.0f}"] + shards
                    for mode, more in (("whole", []), ("ot", ["--split-experts"])):
                        free, ts, ot = fit.plan(base + more)
                        moved = len(ot.split(",")) if ot else 0
                        rows.append((name, keep, ctx, kv, int(ub), draft, mode, free, moved))
                        print(f"{name}\t{keep}\t{ctx}\t{kv}\t{ub}\t{draft}\t{mode}\t{free}\t{ts}\t{moved}\t{ot}",
                              flush=True)
    if a.bytes:
        with open(a.bytes, "w") as f:
            f.write("file\tkeep\trouted_experts_mib\tcards_mib\n")
            for name, keep, routed, cards in sizes:
                f.write(f"{name}\t{keep}\t{routed:.0f}\t{cards:.0f}\n")
    if a.md:
        ctxs = sorted({r[2] for r in rows})
        with open(a.md, "w") as f:
            for kv in sorted(kvs):
                for draft in a.drafts.split(","):
                    f.write(f"\n### {kv} cache, draft: {draft}\n\n| File | Keep | On the cards | "
                            + " | ".join("262k" if c == 262144 else f"{c // 1024}k" for c in ctxs) + " |\n|---|---|---|"
                            + "---|" * len(ctxs) + "\n")
                    for name, keep, _routed, cards in sizes:
                        cells = []
                        for ctx in ctxs:
                            cell = []
                            for mode in ("whole", "ot"):
                                got = [r for r in rows if r[:7] == (name, keep, ctx, kv, r[4], draft, mode)
                                       and r[7] is not None]
                                ok = [r for r in got if r[7] >= MARGIN]
                                if ok:
                                    best = max(ok, key=lambda r: (r[4], r[7]))
                                    cell.append(f"ub {best[4]}: {best[7]}" + (f" ({best[8]} moved)" if best[8] else ""))
                                else:
                                    cell.append(f"no ({max(r[7] for r in got)})" if got else "no")
                            cells.append(" / ".join(cell))
                        f.write(f"| {name} | {keep} | {cards:,.0f} MiB | " + " | ".join(cells) + " |\n")


if __name__ == "__main__":
    main()
