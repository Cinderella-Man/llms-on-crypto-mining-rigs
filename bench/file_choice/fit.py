#!/usr/bin/env python3
"""Fit of each Flash-Next file on nine 6 GB cards over a grid of context, cache type, micro-batch and draft, from GGUF
headers only, through scripts/split_plan.py (imported; each file's headers are read once).
  python3 -I fit.py [--cards 9] [--free 6001,6003,...] NAME=SHARD[,SHARD...] ... > fit.tsv
For every file and every row of COSTS (the per-layer costs llama.cpp projected for Flash-Next on Miner Llama v2.0 with
its default split, Q213: docs/data/2026-11-03_flash_next_fit/run/plan_splits.sh), it plans the best whole-layer `-ts`
and the best split with routed-expert tensors moved by `-ot` (--split-experts), for each draft:
  none     the main model alone (3 MiB of recurrent state per Gated DeltaNet layer);
  dflash2  Inco AI's DFlash2 drafter on the last card: 611 MiB of weights + 64 MiB compute + its cache (45 MiB at 64k,
           scaled with the context: est.), measured by the fork's fit at 64k (Q740); the main model keeps a copy of the
           recurrent state per drafted token (12.3 MiB per layer at three drafted, Q213);
  mtp      the MTP draft as Q213 measured it on the last card (its token embeddings in host memory), same state;
  mtp-shr  the same with LLAMA_DRAFT_SHARE=embd,head: the draft uses the target's head, 497 MiB less (Q635).
Prints one TSV row per plan: the smallest free MiB on any card, the -ts and the -ot moves.
Standard library only.
"""
import argparse, contextlib, functools, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "scripts")))
import split_plan  # noqa: E402

split_plan.read_header = functools.lru_cache(maxsize=None)(split_plan.read_header)

# kv ub ctx: KV MiB per attention layer, compute MiB on card 0, on the other cards, MTP draft MiB on the last card
COSTS = """
f16 64 32768 81 60 70 2053
f16 64 49152 121 81 91 2114
f16 64 65536 161 102 112 2175
f16 64 131072 322 186 195 2419
f16 64 262144 644 353 363 2909
f16 128 32768 81 110 129 2112
f16 128 49152 121 131 150 2173
f16 128 65536 161 161 180 2243
f16 128 131072 322 285 304 2528
f16 128 262144 644 532 551 3097
f16 256 32768 81 220 257 2225
f16 256 49152 121 260 298 2306
f16 256 65536 161 307 344 2392
f16 256 131072 322 527 564 2773
f16 256 262144 644 966 1004 3535
f16 512 65536 161 600 674 2692
f16 512 131072 322 1030 1105 3284
f16 512 262144 644 1886 1961 4462
q8_0 64 32768 43 115 125 2070
q8_0 64 49152 65 160 170 2137
q8_0 64 65536 86 205 215 2203
q8_0 64 131072 172 385 394 2468
q8_0 64 262144 344 744 754 3000
q8_0 128 32768 43 168 187 2132
q8_0 128 49152 65 233 252 2219
q8_0 128 65536 86 294 313 2301
q8_0 128 131072 172 538 557 2631
q8_0 128 262144 344 1025 1044 3290
q8_0 256 32768 43 270 308 2238
q8_0 256 49152 65 355 393 2345
q8_0 256 65536 86 440 477 2450
q8_0 256 131072 172 796 833 2892
q8_0 256 262144 344 1491 1529 3760
q8_0 512 65536 86 748 823 2766
q8_0 512 131072 172 1295 1370 3399
q8_0 512 262144 344 2423 2498 4699
"""
DFLASH2_WEIGHTS, DFLASH2_COMPUTE, DFLASH2_CTX_64K = 611, 64, 45  # MiB, the fork's fit at 64k (Q740)
MTP_HEAD = 497  # MiB of the draft's own head, skipped with LLAMA_DRAFT_SHARE=embd,head (Q635)
STATE, STATE_DRAFT = 3.0, 12.3  # MiB of recurrent state per Gated DeltaNet layer, without and with a draft (Q213)


def plan(args):
    sys.argv = ["split_plan.py"] + args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        split_plan.main()
    m = re.search(r"^#1: -ts (\S+)(?: -ot (\S+))?  smallest free (-?\d+) MiB", buf.getvalue(), re.M)
    if not m:
        return None, "", ""
    return int(m.group(3)), m.group(1), m.group(2) or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="NAME=SHARD[,SHARD...]")
    ap.add_argument("--cards", type=int, default=9)
    ap.add_argument("--free", default="6001,6003,6003,6003,6003,6003,6001,6007,6007",
                    help="free MiB per card once llama.cpp's CUDA context is up (Q213's nine cards)")
    ap.add_argument("--drafts", default="none,dflash2,mtp,mtp-shr")
    a = ap.parse_args()
    print("file\tctx\tkv\tub\tdraft\tmode\tsmallest_free_mib\tts\tmoved\tot")
    for spec in a.files:
        name, shards = spec.split("=", 1)
        shards = shards.split(",")
        for line in COSTS.strip().splitlines():
            kv, ub, ctx, k, c0, c, d = line.split()
            ctx = int(ctx)
            for draft in a.drafts.split(","):
                extra = {"none": 0, "dflash2": DFLASH2_WEIGHTS + DFLASH2_COMPUTE + DFLASH2_CTX_64K * ctx / 65536,
                         "mtp": float(d), "mtp-shr": float(d) - MTP_HEAD}[draft]
                state = STATE if draft == "none" else STATE_DRAFT
                base = ["--cards", str(a.cards), "--free", a.free, "--kv-mib", k, "--state-mib", str(state),
                        "--compute", f"{c0},{c}", "--last-extra", f"{extra:.0f}"] + shards
                for mode, more in (("whole", []), ("ot", ["--split-experts"])):
                    free, ts, ot = plan(base + more)
                    moved = len(ot.split(",")) if ot else 0
                    print(f"{name}\t{ctx}\t{kv}\t{ub}\t{draft}\t{mode}\t{free}\t{ts}\t{moved}\t{ot}", flush=True)


if __name__ == "__main__":
    main()
