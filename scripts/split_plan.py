#!/usr/bin/env python3
# split_plan.py [options] SHARD.gguf [SHARD.gguf ...]: the llama.cpp `-ts` layer split that leaves the most free memory
# on the fullest card, for any number of cards. It reads only the GGUF headers (all shards of a split file), sums each
# block's tensor bytes, and adds per card: the KV cache of its attention layers (--kv-mib per layer), the recurrent
# state of its other layers (--state-mib), its compute buffer (--compute, one value or one per card) and, on the last
# card, the output layer (every tensor outside the blocks except the token embeddings and the n-gram table, which stay
# in host memory) and --last-extra (e.g. a draft model that follows the -ts onto the last card). The split is exact:
# each card's -ts value is its layer count, the last card's plus one for the output layer, so llama.cpp assigns the
# same layers. Take --kv-mib, --state-mib and --compute from one scripts/fit_probe.sh run of the same context,
# micro-batch and cache type (its memory breakdown), then check the planned split with another probe.
#   python3 -I split_plan.py --cards 9 --free 6001 --kv-mib 161 --state-mib 3 --compute 600,674 --top 3 \
#     F-00001-of-00002.gguf F-00002-of-00002.gguf
# --compute with fewer values than cards repeats its last value. --split-experts also lets a card boundary fall inside
# a layer: the routed-expert tensors (gate, up, down, in that order) after the cut go to the next card through -ot,
# for files whose layers differ too much in size to balance whole. --layers prints each block's MiB instead.
# Standard library only.
import argparse, collections, re, struct

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
HOST_ONLY = re.compile(r"^(token_embd|per_layer_token_embd)\.")
EXPERTS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def read_header(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    rstr = lambda: f.read(rd("<Q")).decode("utf-8", "replace")

    def rval(t):
        if t in SCALAR:
            return rd(SCALAR[t])
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [rval(et) for _ in range(n)]
        raise ValueError(f"type {t}")

    assert f.read(4) == b"GGUF"
    _ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k, t = rstr(), rd("<I")
        kv[k] = rval(t)
    infos = []
    for _ in range(nt):
        name = rstr()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        infos.append([name, rd("<I"), rd("<Q")])
    align = kv.get("general.alignment", 32)
    data0 = (f.tell() + align - 1) // align * align
    end = f.seek(0, 2) - data0
    infos.sort(key=lambda x: x[2])
    sizes = {x[0]: (infos[i + 1][2] if i + 1 < len(infos) else end) - x[2] for i, x in enumerate(infos)}
    return kv, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--cards", type=int, required=True)
    ap.add_argument("--free", default="6001", help="free MiB per card, one value or one per card")
    ap.add_argument("--kv-mib", type=float, required=True, help="KV cache MiB per attention layer")
    ap.add_argument("--state-mib", type=float, default=0.0, help="recurrent state MiB per non-attention layer")
    ap.add_argument("--compute", default="0", help="compute buffer MiB, one value or one per card")
    ap.add_argument("--last-extra", type=float, default=0.0, help="MiB more on the last card")
    ap.add_argument("--top", type=int, default=1, help="also print the next best splits")
    ap.add_argument("--ts", help="score this -ts split (layer counts, the last one plus the output) instead of searching")
    ap.add_argument("--split-experts", action="store_true", help="allow cuts inside a layer (expert tensors via -ot)")
    ap.add_argument("--target", type=float, help="with --split-experts: move the fewest expert tensors that leave "
                    "every card this many MiB, instead of maximizing the smallest free")
    ap.add_argument("--layers", action="store_true", help="print each block's MiB (all, routed experts) as TSV and stop")
    a = ap.parse_args()

    kv, sizes = {}, {}
    for s in a.shards:
        k, z = read_header(s)
        kv = kv or k
        sizes.update(z)
    arch = kv["general.architecture"]
    n_layer = kv[f"{arch}.block_count"]
    rec = kv.get(f"{arch}.attention.recurrent_layers")
    interval = kv.get(f"{arch}.full_attention_interval")
    attn = [not rec[i] for i in range(n_layer)] if rec else \
        [(i + 1) % interval == 0 for i in range(n_layer)] if interval else [True] * n_layer
    MiB = 1024 * 1024
    block = collections.defaultdict(float)
    exps = collections.defaultdict(dict)  # layer -> {gate|up|down: MiB}
    out = 0.0
    for name, size in sizes.items():
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            block[int(m.group(1))] += size / MiB
            e = EXPERTS.match(name)
            if e:
                exps[int(e.group(1))][e.group(2)] = size / MiB
        elif not HOST_ONLY.match(name):
            out += size / MiB
    if a.layers:
        print("layer\tattention\tmib\texperts_mib")
        for i in range(n_layer):
            print(f"{i}\t{'yes' if attn[i] else 'no'}\t{block[i]:.1f}\t{sum(exps[i].values()):.1f}")
        print(f"output\t\t{out:.1f}\t")
        return
    per = lambda f: [float(x) for x in f.split(",")]
    expand = lambda v: [v[min(i, len(v) - 1)] for i in range(a.cards)]
    free, comp = expand(per(a.free)), expand(per(a.compute))

    # units in execution order: each layer's core (all but the split-off expert tensors, plus its cache or state),
    # then, with --split-experts, its expert tensors; a card holds a contiguous run of units
    units = []  # (layer, expert kind or None, MiB)
    for i in range(n_layer):
        split = [k for k in ("gate", "up", "down") if k in exps[i]] if a.split_experts else []
        units.append((i, None, block[i] - sum(exps[i][k] for k in split) + (a.kv_mib if attn[i] else a.state_mib)))
        units += [(i, k, exps[i][k]) for k in split]
    n = len(units)
    pre = [0.0]
    for u in units:
        pre.append(pre[-1] + u[2])
    core_at = [p for p, u in enumerate(units) if u[1] is None]
    cores_before = [0] * (n + 1)
    for p in range(n):
        cores_before[p + 1] = cores_before[p] + (units[p][1] is None)

    def card_used(c, lo, hi):
        u = pre[hi] - pre[lo] + comp[c]
        return u + out + a.last_extra if c == a.cards - 1 else u

    if a.ts:
        counts = [int(x) for x in a.ts.split(",")]
        counts[-1] -= 1
        assert sum(counts) == n_layer, f"-ts {a.ts} covers {sum(counts)} layers plus the output, not {n_layer}"
        cuts = [core_at[sum(counts[:i + 1])] if i < a.cards - 1 else n for i in range(a.cards)]
        lo, worst = 0, float("inf")
        for c, hi in enumerate(cuts):
            worst, lo = min(worst, free[c] - card_used(c, lo, hi)), hi
        best = {n: [(worst, cuts)]}
    else:
        # every contiguous split with at least one layer core per card: keep the a.top best per prefix of cards, by
        # largest smallest-free, or with --target by fewest moved tensors among those that keep the target
        def moved(lo, hi):
            p = lo
            while p < hi and units[p][1] is not None:
                p += 1
            return p - lo

        key = (lambda x: -x[0]) if a.target is None else (lambda x: (x[2], -x[0]))
        best = {0: [(float("inf"), [], 0)]}
        for c in range(a.cards):
            nxt = collections.defaultdict(list)
            for lo, cands in best.items():
                his = [n] if c == a.cards - 1 else range(lo + 1, n)
                for hi in his:
                    if cores_before[hi] - cores_before[lo] < 1 or n_layer - cores_before[hi] < a.cards - 1 - c:
                        continue
                    slack = free[c] - card_used(c, lo, hi)
                    if a.target is not None and slack < a.target:
                        continue
                    for worst, cuts, mv in cands:
                        nxt[hi].append((min(worst, slack), cuts + [hi], mv + moved(lo, hi)))
            best = {hi: sorted(v, key=key)[:a.top] for hi, v in nxt.items()}
        if n not in best:
            print(f"no split leaves {a.target:.0f} MiB on every card")
            return
        best = {n: [(w, cuts) for w, cuts, _ in best[n]]}
    print(f"{arch}: {n_layer} layers, attention at {[i for i in range(n_layer) if attn[i]]}, output {out:.0f} MiB "
          f"on the last card; weights {sum(block.values()):.0f} MiB in the blocks")
    for rank, (worst, cuts) in enumerate(best[n], 1):
        lo, ts, ot, rows = 0, [], [], []
        for c, hi in enumerate(cuts):
            layers = [units[p][0] for p in range(lo, hi) if units[p][1] is None]
            ts.append(len(layers) + (1 if c == a.cards - 1 else 0))
            moved = [f"blk.{units[p][0]}.ffn_{units[p][1]}_exps" for p in range(lo, hi)
                     if units[p][1] is not None and units[p][0] not in layers]
            ot += [rf"blk\.{m.split('.')[1]}\.{m.split('.')[2]}=CUDA{c}" for m in moved]
            u = card_used(c, lo, hi)
            rows.append(f"  card {c}: layers {layers[0]}-{layers[-1]}, {sum(attn[i] for i in layers)} attention"
                        + (f", plus {', '.join(moved)}" if moved else "") + f", {u:.0f} MiB, {free[c] - u:.0f} free")
            lo = hi
        print(f"#{rank}: -ts {','.join(map(str, ts))}" + (f" -ot {','.join(ot)}" if ot else "")
              + f"  smallest free {worst:.0f} MiB")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
