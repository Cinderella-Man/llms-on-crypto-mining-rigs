#!/usr/bin/env python3
"""A "Pascal mix" of a Qwen3.8-Flash-Next GGUF: the makers' routed experts byte for byte, every other weight in the
type that is fastest on a P106 at its quality. From one file's header dump (bench/file_choice/dump.py) and a recipe,
writes the per-tensor plan, the inputs of the build and the est. size and decode time.
  python3 -I pmix.py --dump BASE.json --recipe fast|spine6 --name NAME --out DIR [--eff efficiency.tsv]
      [--nonweight 33.48]
Recipes (what each tensor class becomes; a tensor not named keeps its type and bytes):
  both     routed experts (`*_exps`), hyper-connections (`hc_*`, BF16), output layer, token embeddings, n-gram table,
           norms and every other F32/F16/BF16 tensor: kept. Router `ffn_gate_inp` F32 -> BF16 (exact for these files:
           their F32 routers hold BF16 values; on the Pascal BF16 kernel instead of llama.cpp's F32 one).
  fast     dense matrices (attention, recurrent, PLE) in Q3_K or IQ3_S -> IQ4_XS: the slowest types per weight on a
           P106 (one-column paths at 25-35 GB/s against 88 for IQ4_XS), and IQ4_XS has more bits. Shared expert:
           down in IQ4_NL (no Pascal one-token kernel) -> Q5_0; gate and up of a layer get one type, so they fuse into
           one launch: the slow types go to IQ4_XS first, then both take the type of the two with more bits. A layer
           whose shared gate and up already have the routed experts' type keeps them (they fuse with the experts).
           No tensor gets fewer bits than the makers gave it.
  spine6   every dense matrix below Q6_K -> Q6_K; shared gate/up below Q6_K -> Q6_K; shared down below Q5_0 -> Q5_0
           (640-wide rows hold no 256-value blocks). The near-lossless spine at about the published file's speed.
New types are made by llama-quantize --allow-requantize from the Swift-1.5 Q8_0 of the same model with an importance
matrix; the router by scripts/gguf_retype.py. Writes to --out:
  plan.tsv       every tensor: class, layer, shape, old and new type, how (keep, quantize, retype), old and new bytes
  types.txt      llama-quantize --tensor-type-file: every matrix anchored by name, new type or its own (= copied)
  take.txt       the names scripts/gguf_compose.py takes from llama-quantize's output (one per line)
  retype.txt     the gguf_retype.py rule for the router
  summary.tsv    per role and type: MB a decoded token reads and est. ms, base against mix (bench/file_choice's model:
                 bytes over each type's GB/s on a P106 + --nonweight), MB on the cards
  NAME.json      the mix's planned header dump (dump.py's format, for bench/file_choice/model.py)
Standard library only.
"""
import argparse, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "file_choice"))
from model import gbps, load_eff, role  # noqa: E402

BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q5_0": (32, 22), "Q8_0": (32, 34),
         "IQ4_NL": (32, 18), "Q2_0": (32, 9), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
         "Q6_K": (256, 210), "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98), "IQ2_S": (256, 82),
         "IQ2_XS": (256, 74), "IQ2_XXS": (256, 66), "IQ1_M": (256, 56)}
SLOW = ("Q3_K", "IQ3_S")
DENSE = re.compile(r"\.(attn_(q|k|v|qkv|gate|output)|ssm_out|ple_key|ple_query)\.weight$")
MiB = 2 ** 20


def bits(t):
    k, b = BLOCK[t]
    return 8 * b / k


def nbytes(t, dims):
    k, b = BLOCK[t]
    assert dims[0] % k == 0, f"{t} needs rows of whole {k}-value blocks, not {dims[0]}"
    n = 1
    for d in dims[1:]:
        n *= d
    return n * dims[0] // k * b


def tclass(name, typ, dims):
    if "_exps." in name:
        return "routed"
    if name.endswith("ffn_gate_inp.weight"):
        return "router"
    if re.search(r"\.ffn_(gate|up)_shexp\.weight$", name):
        return "shared_gu"
    if name.endswith("ffn_down_shexp.weight"):
        return "shared_down"
    if typ in ("F32", "F16", "BF16") or len(dims) < 2 or "hc_" in name:
        return "keep"
    if DENSE.search(name):
        return "dense"
    return "keep"  # output, token embeddings, the n-gram table


def plan(tensors, recipe):
    """name -> (class, new type, how)."""
    types = {n: t for n, t, _d, _s, _sh in tensors}
    out = {}
    for name, typ, dims, _size, _shard in tensors:
        c = tclass(name, typ, dims)
        new = typ
        if c == "router" and typ == "F32":
            new = "BF16"
        elif c == "dense":
            if recipe == "fast" and typ in SLOW:
                new = "IQ4_XS"
            elif recipe == "spine6" and bits(typ) < bits("Q6_K"):
                new = "Q6_K"
        elif c == "shared_down":
            if recipe == "fast" and typ == "IQ4_NL":
                new = "Q5_0"
            elif recipe == "spine6" and bits(typ) < bits("Q5_0"):
                new = "Q5_0"
        out[name] = [c, new]
    for name, typ, dims, _size, _shard in tensors:  # shared gate/up, per layer
        m = re.match(r"(blk\.\d+)\.ffn_gate_shexp\.weight$", name)
        if not m:
            continue
        g, u, e = name, f"{m[1]}.ffn_up_shexp.weight", f"{m[1]}.ffn_gate_exps.weight"
        tg, tu = types[g], types[u]
        if recipe == "fast":
            if tg == tu == types.get(e):
                continue
            tg, tu = ("IQ4_XS" if t in SLOW else t for t in (tg, tu))
            pair = max((tg, tu), key=bits)
        else:
            pair = max((tg, tu, "Q6_K"), key=bits)
        out[g][1] = out[u][1] = pair
    for name, typ, _dims, _size, _shard in tensors:
        c, new = out[name]
        out[name] = (c, new, "keep" if new == typ else ("retype" if c == "router" else "quantize"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--recipe", required=True, choices=("fast", "spine6"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eff", default=os.path.join(HERE, "..", "file_choice", "efficiency.tsv"))
    ap.add_argument("--nonweight", type=float, default=33.48, help="ms of GPU time per token outside weight products")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.dump))
    eff = load_eff(a.eff)
    ne, nu = d["keys"]["expert_count"], d["keys"]["expert_used_count"]
    p = plan(d["tensors"], a.recipe)

    rows, new_tensors = [], []
    tok = {"base": collections.Counter(), "mix": collections.Counter()}
    cards = collections.Counter()
    for name, typ, dims, size, shard in d["tensors"]:
        c, new, how = p[name]
        nsize = size if how == "keep" else (nbytes(new, dims) + 31) // 32 * 32
        layer = re.match(r"blk\.(\d+)\.", name)
        rows.append((name, c, layer[1] if layer else "", "x".join(map(str, dims)), typ, new, how, size, nsize))
        new_tensors.append([name, new, dims, nsize, shard])
        rl = role(name)
        if rl == "host":
            continue
        f = nu / ne if rl.startswith("routed") else 1
        tok["base"][(rl, typ)] += size * f
        tok["mix"][(rl, new)] += nsize * f
        cards["base"] += size
        cards["mix"] += nsize

    with open(os.path.join(a.out, "plan.tsv"), "w") as f:
        f.write("tensor\tclass\tlayer\tshape\told_type\tnew_type\thow\told_bytes\tnew_bytes\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(a.out, "types.txt"), "w") as f:
        for name, typ, dims, _size, _shard in d["tensors"]:
            if len(dims) >= 2 and p[name][0] != "router":
                f.write(f"^{re.escape(name)}$={p[name][1].lower()}\n")
    with open(os.path.join(a.out, "take.txt"), "w") as f:
        f.writelines(f"{n}\n" for n, _t, _d, _s, _sh in d["tensors"] if p[n][2] == "quantize")
    with open(os.path.join(a.out, "retype.txt"), "w") as f:
        f.write(r"ffn_gate_inp\.weight$=BF16@F32" + "\n")
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(dict(d, name=a.name, tensors=new_tensors), f, separators=(",", ":"))
        f.write("\n")

    ms = {}
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write("file\trole\ttype\tmb_per_token\ttimed_gbps\texpected_gbps\ttimed_ms\texpected_ms\n")
        for which, label in (("base", d["name"]), ("mix", a.name)):
            tt = te = 0.0
            for (rl, typ), b in sorted(tok[which].items(), key=lambda kv: (kv[0][0], -kv[1])):
                g = gbps(eff, typ, rl)
                t1, t2 = b / 1e6 / g[0], b / 1e6 / g[1]
                tt, te = tt + t1, te + t2
                f.write(f"{label}\t{rl}\t{typ}\t{b / 1e6:.1f}\t{g[0]}\t{g[1]}\t{t1:.2f}\t{t2:.2f}\n")
            ms[which] = (a.nonweight + tt, a.nonweight + te)
            f.write(f"{label}\ttotal\t\t{sum(tok[which].values()) / 1e6:.1f}\t\t\t{ms[which][0]:.1f}\t{ms[which][1]:.1f}"
                    f"\n{label}\tcards_mib\t\t{cards[which] / MiB:.0f}\t\t\t\t\n")
    n = collections.Counter((c, typ, new) for (_n, c, _l, _s, typ, new, how, _o, _ns) in rows if how != "keep")
    print(f"{a.name}: {sum(n.values())} tensors change; cards {cards['base'] / MiB:.0f} -> {cards['mix'] / MiB:.0f} MiB; "
          f"est. ms/token timed {ms['base'][0]:.1f} -> {ms['mix'][0]:.1f}, expected {ms['base'][1]:.1f} -> "
          f"{ms['mix'][1]:.1f}")
    for (c, typ, new), k in sorted(n.items()):
        print(f"  {c:12s} {typ:7s} -> {new:7s} {k:4d}")


if __name__ == "__main__":
    main()
