"""Which routed experts the tool-call spans use, and what a usage cut drops of them (Q897), from llama-imatrix's
span matrices (--span-mask: one matrix per token class, <out>.span<c>.gguf, beside the usual <out>.gguf).
  span_counts.py --full OUT.gguf --span OUT.span0.gguf OUT.span1.gguf ... --experts experts.tsv.gz --out DIR
                 [--keep 464,448,416] [--names other,call,argument,response,assistant]
--experts is the ranking the cut files were made by (Q232's experts.tsv.gz: columns layer, expert, routings); a cut at
N keeps each layer's N most-routed experts there, ties to the lower id (scripts/gguf_prune.py's rule). Routings are
read from each layer's ffn_gate_exps counts (the router's picks; up and down count the same).
Writes to DIR:
  span_experts.tsv.gz  per layer and expert: the ranking's routings and rank; this pass's routings, all and per
                       class, and the class's share of the layer (%); the call class's share over the expert's share
                       of all routings (enrichment)
  span_layers.tsv      per layer and class: tokens' routings, experts covering 50/90/99 % of them, the top five with
                       their share and their rank in the ranking
  span_cut.tsv         per kept N, class and layer: the share of the class's routings that go to experts the cut drops
  span_cut_summary.tsv per kept N and class: that share over all layers, the worst layer, and the layers above 1 %
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import argparse
import gzip
import math
import os
import re
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

GATE = re.compile(r"^blk\.(\d+)\.ffn_(gate|gate_up)_exps\.weight\.counts$")


def routings(path):
    """{layer: [routings per expert]} and the header."""
    h = Header(path)
    out = {}
    with open(path, "rb") as f:
        for name, dims, typ, off in h.tensors:
            m = GATE.match(name)
            if not m:
                continue
            assert typ == 0
            f.seek(h.data0 + off)
            a = array("f")
            a.frombytes(f.read(4 * math.prod(dims)))
            out[int(m[1])] = [int(v) for v in a]
    return out, h


def ranking(path):
    per = {}
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        head = f.readline().rstrip("\n").split("\t")
        li, ei, ri = head.index("layer"), head.index("expert"), head.index("routings")
        for line in f:
            c = line.rstrip("\n").split("\t")
            per.setdefault(int(c[li]), {})[int(c[ei])] = float(c[ri])
    rank = {}
    for layer, d in per.items():
        order = sorted(d, key=lambda e: (-d[e], e))
        rank[layer] = {e: i + 1 for i, e in enumerate(order)}
    return per, rank


def cover(vals, p):
    tot, run = sum(vals), 0
    for i, v in enumerate(sorted(vals, reverse=True), 1):
        run += v
        if run >= p * tot:
            return i
    return len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--span", nargs="+", required=True, help="the class matrices, class 0 first")
    ap.add_argument("--experts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default="464,448,416")
    ap.add_argument("--names", default="other,call,argument,response,assistant")
    a = ap.parse_args()
    names = a.names.split(",")
    full, _ = routings(a.full)
    spans, tokens = [], []
    for c, p in enumerate(a.span):
        r, h = routings(p)
        assert h.value("imatrix.span.class") == c, f"{p}: class {h.value('imatrix.span.class')}, expected {c}"
        spans.append(r)
        tokens.append(h.value("imatrix.span.tokens"))
    names = (names + [f"class{c}" for c in range(len(names), len(spans))])[:len(spans)]
    per, rank = ranking(a.experts)
    layers = sorted(full)
    for layer in layers:
        assert [sum(s[layer][e] for s in spans) for e in range(len(full[layer]))] == full[layer], \
            f"layer {layer}: the classes do not add up to the full matrix"
    os.makedirs(a.out, exist_ok=True)
    call = names.index("call") if "call" in names else 1

    with gzip.open(os.path.join(a.out, "span_experts.tsv.gz"), "wt") as f:
        f.write("layer\texpert\tranking_routings\tranking_rank\troutings\t"
                + "\t".join(f"{n}\t{n}_share_pct" for n in names) + "\tcall_enrichment\n")
        for layer in layers:
            tot = sum(full[layer])
            ctot = [sum(s[layer]) for s in spans]
            for e in range(len(full[layer])):
                cols = []
                for c, s in enumerate(spans):
                    cols += [str(s[layer][e]), f"{100 * s[layer][e] / ctot[c]:.4f}" if ctot[c] else "0"]
                share = full[layer][e] / tot if tot else 0
                enr = (spans[call][layer][e] / ctot[call]) / share if share and ctot[call] else 0
                f.write(f"{layer}\t{e}\t{per[layer].get(e, 0):.0f}\t{rank[layer].get(e, 0)}\t{full[layer][e]}\t"
                        + "\t".join(cols) + f"\t{enr:.3f}\n")

    with open(os.path.join(a.out, "span_layers.tsv"), "w") as f:
        f.write("# tokens per class: " + ", ".join(f"{n} {t}" for n, t in zip(names, tokens)) + "\n")
        f.write("layer\tclass\troutings\tn50\tn90\tn99\ttop5 (expert:share%:ranking rank)\n")
        for layer in layers:
            for c, s in enumerate(spans):
                v = s[layer]
                tot = sum(v)
                if not tot:
                    continue
                top = sorted(range(len(v)), key=lambda e: (-v[e], e))[:5]
                f.write(f"{layer}\t{names[c]}\t{tot}\t{cover(v, .5)}\t{cover(v, .9)}\t{cover(v, .99)}\t"
                        + " ".join(f"{e}:{100 * v[e] / tot:.2f}:{rank[layer].get(e, 0)}" for e in top) + "\n")

    keeps = [int(k) for k in a.keep.split(",")]
    with open(os.path.join(a.out, "span_cut.tsv"), "w") as f, \
            open(os.path.join(a.out, "span_cut_summary.tsv"), "w") as g:
        f.write("keep\tclass\tlayer\troutings\tdropped\tdropped_pct\n")
        g.write("keep\tclass\troutings\tdropped_pct\tworst_layer\tworst_layer_dropped_pct\tlayers_over_1pct\n")
        for n in keeps:
            for c, s in [(len(spans), full)] + list(enumerate(spans)):
                name = "all" if c == len(spans) else names[c]
                tot = drop = 0
                worst, over = (None, -1.0), 0
                for layer in layers:
                    kept = {e for e, r in rank[layer].items() if r <= n}
                    lt = sum(s[layer])
                    ld = sum(v for e, v in enumerate(s[layer]) if e not in kept)
                    tot, drop = tot + lt, drop + ld
                    pct = 100 * ld / lt if lt else 0.0
                    worst = max(worst, (layer, pct), key=lambda x: x[1])
                    over += pct > 1
                    f.write(f"{n}\t{name}\t{layer}\t{lt}\t{ld}\t{pct:.3f}\n")
                g.write(f"{n}\t{name}\t{tot}\t{100 * drop / tot if tot else 0:.3f}\t{worst[0]}\t{worst[1]:.3f}\t{over}\n")
    print(open(os.path.join(a.out, "span_cut_summary.tsv")).read(), end="")


if __name__ == "__main__":
    main()
