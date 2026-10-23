"""Expert overlap between consecutive tokens (Q51), from llama-route-dump's routes.bin and generate.py's gen.jsonl.

usage: python analyze.py ROUTES.bin GEN.jsonl OUT.json [N_EXPERT]

Reports, for generated tokens (and for prompt tokens, for comparison), per category and per layer:
- distinct experts touched by n consecutive tokens (n = 1, 2, 4, 8, 16), against independent random routing;
- experts shared by adjacent tokens;
- the router's top-k weights after renormalization (how light the last experts are);
- how concentrated each layer's expert use is.
"""
import json
import struct
import sys
from collections import defaultdict

WINDOWS = (1, 2, 4, 8, 16)


def read_routes(path):
    seqs = {}  # seq -> {"n_prompt", "n", "ids": {layer: [mask per pos]}, "w": {layer: [sorted weights]}}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off < len(data):
        kind, seq, layer, pos0, n_tok, k = struct.unpack_from("<6i", data, off)
        off += 24
        if kind == 2:
            seqs[seq] = {"n_prompt": pos0, "n": n_tok, "k": 0, "ids": defaultdict(dict), "w": defaultdict(dict)}
            continue
        s = seqs[seq]
        s["k"] = k
        cnt = n_tok * k
        if kind == 0:
            vals = struct.unpack_from(f"<{cnt}i", data, off)
            for i in range(n_tok):
                m = 0
                for e in vals[i * k:(i + 1) * k]:
                    m |= 1 << e
                s["ids"][layer][pos0 + i] = m
        else:
            vals = struct.unpack_from(f"<{cnt}f", data, off)
            for i in range(n_tok):
                s["w"][layer][pos0 + i] = vals[i * k:(i + 1) * k]
        off += cnt * 4
    return seqs


def random_distinct(n, k, n_expert):
    return n_expert * (1 - (1 - k / n_expert) ** n)


def window_stats(masks, acc, key_prefix):
    """masks: one int bitmask per consecutive position; adds distinct-count sums for each window size to acc."""
    level = masks
    width = 1
    unions = {1: masks}
    while width < max(WINDOWS):
        level = [level[i] | level[i + width] for i in range(len(level) - width)]
        width *= 2
        unions[width] = level
    for n in WINDOWS:
        u = unions.get(n, [])
        acc[(key_prefix, n)][0] += sum(x.bit_count() for x in u)
        acc[(key_prefix, n)][1] += len(u)
    shared = [(masks[i] & masks[i + 1]).bit_count() for i in range(len(masks) - 1)]
    acc[(key_prefix, "adj")][0] += sum(shared)
    acc[(key_prefix, "adj")][1] += len(shared)


def main():
    routes_path, gen_path, out_path = sys.argv[1:4]
    n_expert = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    gen = [json.loads(l) for l in open(gen_path)]
    seqs = read_routes(routes_path)
    k = next(s["k"] for s in seqs.values() if s["k"])
    layers = sorted({l for s in seqs.values() for l in s["ids"]})

    acc = defaultdict(lambda: [0, 0])        # (scope, n) -> [sum distinct, windows]
    acc_layer = defaultdict(lambda: [0, 0])  # (layer, n) over generated tokens
    usage = defaultdict(lambda: defaultdict(int))  # layer -> expert -> routed count (generated tokens)
    wsum = [0.0] * k
    wn = 0
    light = defaultdict(int)  # threshold -> (token, layer) pairs whose lightest renormalized weight is below it
    mass = 0.0
    n_gen_tokens = defaultdict(int)

    for i, g in enumerate(gen):
        s = seqs.get(i)
        if s is None:
            continue
        cat = g["category"]
        n_prompt, n = s["n_prompt"], s["n"]
        n_gen_tokens[cat] += n - n_prompt
        for layer in layers:
            ids = s["ids"][layer]
            if len(ids) != n:
                print(f"seq {i} layer {layer}: {len(ids)} of {n} positions recorded, skipped", file=sys.stderr)
                continue
            gen_masks = [ids[p] for p in range(n_prompt, n)]
            prm_masks = [ids[p] for p in range(0, n_prompt)]
            for scope in (("gen", cat), ("gen", "all")):
                window_stats(gen_masks, acc, scope)
            window_stats(prm_masks, acc, ("prompt", "all"))
            window_stats(gen_masks, acc_layer, layer)
            for m in gen_masks:
                while m:
                    low = m & -m
                    usage[layer][low.bit_length() - 1] += 1
                    m ^= low
            for p in range(n_prompt, n):
                w = s["w"][layer].get(p)
                if w is None:
                    continue
                tot = sum(w)
                mass += tot
                ws = sorted((x / tot for x in w), reverse=True)
                for j in range(k):
                    wsum[j] += ws[j]
                wn += 1
                for t in (0.02, 0.05, 0.10, 0.15):
                    if ws[-1] < t:
                        light[t] += 1

    def table(key_prefix):
        row = {}
        for n in WINDOWS:
            a = acc[(key_prefix, n)]
            row[str(n)] = a[0] / a[1] if a[1] else None
        a = acc[(key_prefix, "adj")]
        row["adjacent_shared"] = a[0] / a[1] if a[1] else None
        return row

    out = {"n_expert": n_expert, "k": k, "moe_layers": len(layers), "generated_tokens": dict(n_gen_tokens),
           "random": {str(n): random_distinct(n, k, n_expert) for n in WINDOWS},
           "random_adjacent_shared": k * k / n_expert,
           "distinct": {}, "per_layer": {}, "weights": {}, "usage": {}}
    for cat in sorted(n_gen_tokens) + ["all"]:
        out["distinct"][f"gen/{cat}"] = table(("gen", cat))
    out["distinct"]["prompt/all"] = table(("prompt", "all"))
    for layer in layers:
        row = {}
        for n in WINDOWS:
            a = acc_layer[(layer, n)]
            row[str(n)] = a[0] / a[1] if a[1] else None
        a = acc_layer[(layer, "adj")]
        row["adjacent_shared"] = a[0] / a[1] if a[1] else None
        out["per_layer"][str(layer)] = row
    out["weights"] = {"mean_sorted_renormalized": [x / wn for x in wsum] if wn else None,
                      "lightest_below": {str(t): light[t] / wn for t in sorted(light)} if wn else None,
                      "mean_topk_probability_mass": mass / wn if wn else None}
    cover = defaultdict(list)
    for layer in layers:
        counts = sorted(usage[layer].values(), reverse=True)
        tot = sum(counts)
        used = len(counts)
        for frac in (0.1, 0.25, 0.5):
            top = int(round(frac * n_expert))
            cover[str(frac)].append(sum(counts[:top]) / tot if tot else 0)
        cover["unused"].append(n_expert - used)
    out["usage"] = {f: sum(v) / len(v) for f, v in cover.items()}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"k={k}, {len(layers)} MoE layers, generated tokens {dict(n_gen_tokens)}")
    print("distinct experts per layer for n consecutive tokens (random routing in the last row):")
    print(f"{'scope':16s}" + "".join(f"{'n=' + str(n):>9s}" for n in WINDOWS) + f"{'adj':>9s}")
    for name, row in out["distinct"].items():
        print(f"{name:16s}" + "".join(f"{row[str(n)]:9.2f}" for n in WINDOWS) + f"{row['adjacent_shared']:9.2f}")
    print(f"{'random':16s}" + "".join(f"{out['random'][str(n)]:9.2f}" for n in WINDOWS)
          + f"{out['random_adjacent_shared']:9.2f}")
    print("renormalized weights, mean by rank:", [round(x, 3) for x in out["weights"]["mean_sorted_renormalized"]])
    print("lightest expert below threshold:", out["weights"]["lightest_below"])
    print("top-k probability mass:", round(out["weights"]["mean_topk_probability_mass"], 3))
    print("usage: share of routed reads on the top 10/25/50 % of a layer's experts, unused experts:", out["usage"])


if __name__ == "__main__":
    main()
