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
# --fewest finds the fewest cards that hold every layer with --margin MiB free on each. Every card boundary is a host
# round trip per token, so each card left out is one hop fewer. It scores every card count from --cards down, using the
# first k cards of --devices (default 0,1,...) and leaving the rest free: one TSV row per count, then the plan of the
# fewest with its -dev, -ts and -ot. --free is per card (its CUDA index), --compute per position in the split (the first
# value for the first card). --own-card MiB is a piece that needs a free card of its own (a draft model's whole pass:
# -devd and -otd onto that card), checked on the first free card; --extra-any MiB is a piece the planner puts on the
# card that leaves the most free (a draft's token embeddings: -otd token_embd=<that card>). Whole layers come first:
# any -ot turns off llama.cpp's pipelined prompt (llama_context runs no pipeline parallelism with tensor overrides).
#   python3 -I split_plan.py --cards 9 --free 6023 --kv-mib 161 --state-mib 12.3 --compute 344,419 --fewest \
#     --margin 300 --own-card 2781 --split-experts F-00001-of-00003.gguf F-00002-of-00003.gguf F-00003-of-00003.gguf
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


def load(shards):
    """The model's blocks from its headers: arch, layer count, attention per layer, MiB per block, routed-expert MiB per
    layer and kind, and the output layer's MiB."""
    kv, sizes = {}, {}
    for s in shards:
        k, z = read_header(s)
        kv = kv if "general.architecture" in kv else k  # the model's keys, from whichever shard holds them
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
    return arch, n_layer, attn, block, exps, out


def make_units(n_layer, attn, block, exps, kv_mib, state_mib, split_experts):
    """Units in execution order: each layer's core (all but the split-off expert tensors, plus its cache or state),
    then, with split_experts, its expert tensors; a card holds a contiguous run of units."""
    units = []  # (layer, expert kind or None, MiB)
    for i in range(n_layer):
        split = [k for k in ("gate", "up", "down") if k in exps[i]] if split_experts else []
        units.append((i, None, block[i] - sum(exps[i][k] for k in split) + (kv_mib if attn[i] else state_mib)))
        units += [(i, k, exps[i][k]) for k in split]
    return units


class Splitter:
    """Contiguous runs of units over cards; room[c] is the MiB the c-th card of the split has for its units once its
    compute buffer and extras are taken off."""

    def __init__(self, units):
        self.units, self.n = units, len(units)
        self.pre = [0.0]
        for u in units:
            self.pre.append(self.pre[-1] + u[2])
        self.core_at = [p for p, u in enumerate(units) if u[1] is None]
        self.n_layer = len(self.core_at)
        self.cores_before = [0] * (self.n + 1)
        for p in range(self.n):
            self.cores_before[p + 1] = self.cores_before[p] + (units[p][1] is None)

    def mib(self, lo, hi):
        return self.pre[hi] - self.pre[lo]

    def moved(self, lo, hi):
        p = lo
        while p < hi and self.units[p][1] is not None:
            p += 1
        return p - lo

    def score(self, room, counts):
        """A -ts split's layer counts (the last one without the output) as (smallest free, cuts, moved)."""
        assert sum(counts) == self.n_layer, f"the -ts covers {sum(counts)} layers plus the output, not {self.n_layer}"
        cuts = [self.core_at[sum(counts[:i + 1])] if i < len(room) - 1 else self.n for i in range(len(room))]
        lo, worst = 0, float("inf")
        for c, hi in enumerate(cuts):
            worst, lo = min(worst, room[c] - self.mib(lo, hi)), hi
        return worst, cuts, 0

    def search(self, room, top=1, target=None):
        """Every contiguous split with at least one layer core per card: the top best as (smallest free, cuts, moved),
        by largest smallest free, or with target by fewest moved tensors among the splits that leave every card that
        many MiB. Empty when no split does."""
        k, n = len(room), self.n
        if self.n_layer < k:
            return []
        key = (lambda x: -x[0]) if target is None else (lambda x: (x[2], -x[0]))
        best = {0: [(float("inf"), [], 0)]}
        for c in range(k):
            nxt = collections.defaultdict(list)
            for lo, cands in best.items():
                his = [n] if c == k - 1 else range(lo + 1, n)
                for hi in his:
                    if self.cores_before[hi] - self.cores_before[lo] < 1 or \
                            self.n_layer - self.cores_before[hi] < k - 1 - c:
                        continue
                    slack = room[c] - self.mib(lo, hi)
                    if target is not None and slack < target:
                        continue
                    mv = self.moved(lo, hi)
                    for worst, cuts, m in cands:
                        nxt[hi].append((min(worst, slack), cuts + [hi], m + mv))
            best = {hi: sorted(v, key=key)[:top] for hi, v in nxt.items()}
        return best.get(n, [])


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
    ap.add_argument("--devices", help="the cards to split over, in order (CUDA indices; default all --cards)")
    ap.add_argument("--fewest", action="store_true", help="the fewest cards that keep --margin free on each")
    ap.add_argument("--margin", type=float, default=300.0, help="with --fewest: MiB every card keeps free")
    ap.add_argument("--own-card", type=float, default=0.0, help="MiB of a piece that needs a free card of its own")
    ap.add_argument("--extra-any", type=float, default=0.0, help="MiB of a piece placed on the best card")
    a = ap.parse_args()

    arch, n_layer, attn, block, exps, out = load(a.shards)
    if a.layers:
        print("layer\tattention\tmib\texperts_mib")
        for i in range(n_layer):
            print(f"{i}\t{'yes' if attn[i] else 'no'}\t{block[i]:.1f}\t{sum(exps[i].values()):.1f}")
        print(f"output\t\t{out:.1f}\t")
        return
    per = lambda f: [float(x) for x in f.split(",")]
    expand = lambda v, k: [v[min(i, len(v) - 1)] for i in range(k)]
    free = expand(per(a.free), a.cards)
    devices = [int(x) for x in a.devices.split(",")] if a.devices else list(range(a.cards))
    assert len(set(devices)) == len(devices) and all(0 <= d < a.cards for d in devices), f"--devices {a.devices}"
    whole = Splitter(make_units(n_layer, attn, block, exps, a.kv_mib, a.state_mib, False))
    split = Splitter(make_units(n_layer, attn, block, exps, a.kv_mib, a.state_mib, True))

    def layout(used, at):
        """Compute and extra MiB per card of the split, and the room left for its units (--extra-any on card at)."""
        comp = expand(per(a.compute), len(used))
        extra = [0.0] * len(used)
        extra[-1] += out + a.last_extra
        if at is not None:
            extra[at] += a.extra_any
        return comp, extra, [free[d] - comp[c] - extra[c] for c, d in enumerate(used)]

    def plans(sp, used, top=1, target=None, counts=None):
        """The best splits over the cards used, as (smallest free, cuts, moved, the card of --extra-any)."""
        res = []
        for at in range(len(used)) if a.extra_any else [None]:
            room = layout(used, at)[2]
            found = [sp.score(room, counts)] if counts else sp.search(room, top, target)
            res += [(w, cuts, mv, at) for w, cuts, mv in found]
        key = (lambda x: -x[0]) if target is None else (lambda x: (x[2], -x[0]))
        return sorted(res, key=key)[:top]

    def show(sp, used, rank, plan, dev):
        """Print one plan: its arguments and a row per card."""
        worst, cuts, _, at = plan
        comp, extra, _ = layout(used, at)
        lo, ts, ot, rows = 0, [], [], []
        for c, hi in enumerate(cuts):
            units = sp.units
            layers = [units[p][0] for p in range(lo, hi) if units[p][1] is None]
            ts.append(len(layers) + (1 if c == len(used) - 1 else 0))
            moved = [f"blk.{units[p][0]}.ffn_{units[p][1]}_exps" for p in range(lo, hi)
                     if units[p][1] is not None and units[p][0] not in layers]
            ot += [rf"blk\.{m.split('.')[1]}\.{m.split('.')[2]}=CUDA{used[c]}" for m in moved]
            u = sp.mib(lo, hi) + comp[c] + extra[c]
            rows.append(f"  card {used[c]}: layers {layers[0]}-{layers[-1]}, {sum(attn[i] for i in layers)} attention"
                        + (f", plus {', '.join(moved)}" if moved else "")
                        + (f", plus the {a.extra_any:.0f} MiB piece" if c == at else "")
                        + f", {u:.0f} MiB, {free[used[c]] - u:.0f} free")
            lo = hi
        print(f"#{rank}: " + (f"-dev {','.join(f'CUDA{d}' for d in used)} " if dev else "") + f"-ts {','.join(map(str, ts))}"
              + (f" -ot {','.join(ot)}" if ot else "") + f"  smallest free {worst:.0f} MiB"
              + (f"; the {a.extra_any:.0f} MiB piece on CUDA{used[at]}" if at is not None else ""))
        print("\n".join(rows))

    head = lambda: print(f"{arch}: {n_layer} layers, attention at {[i for i in range(n_layer) if attn[i]]}, output "
                         f"{out:.0f} MiB on the last card; weights {sum(block.values()):.0f} MiB in the blocks")
    sp = split if a.split_experts else whole
    if not a.fewest:
        counts = None
        if a.ts:
            counts = [int(x) for x in a.ts.split(",")]
            counts[-1] -= 1
        found = plans(sp, devices, 1 if counts else a.top, None if counts else a.target, counts)
        if not found:
            print(f"no split leaves {a.target:.0f} MiB on every card" if a.target is not None else
                  f"no split: {n_layer} layers on {len(devices)} cards")
            return
        head()
        for rank, plan in enumerate(found, 1):
            show(sp, devices, rank, plan, a.devices is not None)
        return

    head()
    print(f"fewest cards keeping {a.margin:.0f} MiB free on every card"
          + (f", a {a.own_card:.0f} MiB piece on the first free card" if a.own_card else "")
          + (f", a {a.extra_any:.0f} MiB piece on the best card" if a.extra_any else ""))
    print("cards\tfree_cards\town_free\twhole_free\twhole_ts\tot_free\tot_moved")
    rows, chosen = [], None
    for k in range(len(devices), 0, -1):
        used, freed = devices[:k], devices[k:]
        if k > n_layer:
            continue
        own = (free[freed[0]] - a.own_card if freed else float("-inf")) if a.own_card else None
        w = plans(whole, used)
        o = plans(split, used) if a.split_experts else []
        ts = lambda p: ",".join(str(len([u for u in whole.units[lo:hi] if u[1] is None]) + (c == k - 1))
                                for c, (lo, hi) in enumerate(zip([0] + p[1][:-1], p[1])))
        print(f"{k}\t{','.join(f'CUDA{d}' for d in freed) or '-'}\t{'-' if own is None else 'none' if own == float('-inf') else f'{own:.0f}'}\t"
              + (f"{w[0][0]:.0f}\t{ts(w[0])}" if w else "-\t-") + "\t"
              + (f"{o[0][0]:.0f}\t{o[0][2]}" if o else "-\t-"))
        ok_own = own is None or own >= a.margin
        if ok_own and ((w and w[0][0] >= a.margin) or (o and o[0][0] >= a.margin)):
            chosen = (k, used, freed, own)
    if chosen is None:
        print(f"no card count keeps {a.margin:.0f} MiB free on every card")
        return
    k, used, freed, own = chosen
    print(f"fewest: {k} cards ({','.join(f'CUDA{d}' for d in used)}), {len(freed)} free"
          + (f" ({','.join(f'CUDA{d}' for d in freed)})" if freed else "")
          + (f"; the {a.own_card:.0f} MiB piece on CUDA{freed[0]}, {own:.0f} free" if a.own_card else ""))
    w = [p for p in plans(whole, used, a.top) if p[0] >= a.margin]
    for rank, plan in enumerate(w, 1):
        show(whole, used, rank, plan, True)
    if not w:
        print("whole layers do not keep the margin; the fewest -ot moves that do (no pipelined prompt with -ot):")
        for rank, plan in enumerate(plans(split, used, a.top, a.margin), 1):
            show(split, used, rank, plan, True)


if __name__ == "__main__":
    main()
