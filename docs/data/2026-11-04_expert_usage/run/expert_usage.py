#!/usr/bin/env python3
"""How a MoE model's routed experts are used, per layer, from an importance matrix (llama-imatrix's GGUF output).
  python3 -I expert_usage.py IMATRIX.gguf OUT_DIR
For every routed-expert tensor (ffn_gate_exps, ffn_up_exps, ffn_down_exps) llama-imatrix stores, per expert,
`.counts`: how many (token, slot) routings reached it, and `.in_sum2`: the sum over those routings of each input column
squared (tools/imatrix/imatrix.cpp, the GGML_OP_MUL_MAT_ID branch). Gate and up read the token's hidden state x; down
reads the expert's own SwiGLU activation h, so down's in_sum2 summed over its columns is the activation energy the
expert hands to its down projection, the sum over routed tokens of ||h||^2. Three ways to rank a layer's experts:
  routings  how often the router picks the expert (frequency pruning);
  energy    sum over routed tokens of ||h||^2, frequency times magnitude (close to EAN, which sums ||f(x)||);
  mean      ||h||^2 per routing, magnitude alone (REAP's average over routed tokens, without the router weight and
            the down projection, which the matrix does not hold).
Writes to OUT_DIR:
  experts.tsv   per layer and expert: routings, share, rank; energy, share, rank; mean ||h||^2 and ||x||^2 per routing
  layers.tsv    per layer: routings, unused experts, experts covering 50/80/90/95/99/99.9 % of routings and of energy,
                entropy, effective experts, Gini, the busiest expert against the mean, the top five experts, the
                Spearman correlation of routings and mean, and how many of the top 256 by routings are in the top 256
                by energy and by mean
  prune.tsv     per layer, kept N and ranking: the share of routings and of energy the kept experts carry
  summary.tsv   per kept N and ranking, over all layers: routings and energy kept (all layers, the worst layer)
  coverage.tsv  layers.tsv's coverage counts, unused experts, effective experts and Gini over the layers: mean,
                median, min, max, and the sum over layers (the experts a per-layer cut at that coverage keeps)
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import math, os, re, sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

NAME = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.(counts|in_sum2)$")
COVER = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
KEEP = (64, 128, 192, 256, 288, 320, 352, 384, 416, 448, 464, 472, 480, 496)
TOP = 256


def read(f, h, t):
    name, dims, typ, off = t
    assert typ == 0, f"{name}: type {typ}, expected F32"
    f.seek(h.data0 + off)
    a = array("f")
    a.frombytes(f.read(4 * math.prod(dims)))
    if sys.byteorder != "little":
        a.byteswap()
    return dims, a


def per_expert(dims, a):
    ne0, n = dims[0], dims[1]
    return [math.fsum(a[e * ne0:(e + 1) * ne0]) for e in range(n)]


def cover(desc, total, p):
    run = 0.0
    for i, v in enumerate(desc, 1):
        run += v
        if run >= p * total * (1 - 1e-12):
            return i
    return len(desc)


def gini(xs):
    s = sorted(xs)
    n, tot = len(s), sum(s)
    return (2 * sum(i * v for i, v in enumerate(s, 1))) / (n * tot) - (n + 1) / n if tot else 0.0


def ranks(xs):
    """1 = largest; ties share their mean rank."""
    order = sorted(range(len(xs)), key=lambda i: -xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def top(xs, n):
    return set(sorted(range(len(xs)), key=lambda i: (-xs[i], i))[:n])


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    h = Header(path)
    chunks, chunk = h.value("imatrix.chunk_count"), h.value("imatrix.chunk_size")
    raw = {}
    with open(path, "rb") as f:
        for t in h.tensors:
            m = NAME.match(t[0])
            if m:
                raw[(int(m.group(1)), m.group(2), m.group(3))] = per_expert(*read(f, h, t))
    layers = sorted({k[0] for k in raw})
    tokens = chunks * chunk if chunks and chunk else None

    ex = open(os.path.join(out, "experts.tsv"), "w")
    ex.write("layer\texpert\troutings\tshare_pct\trank\tenergy\tenergy_share_pct\tenergy_rank\tmean_h2\tmean_x2\n")
    ly = open(os.path.join(out, "layers.tsv"), "w")
    ly.write("layer\troutings\trouted_per_token\tunused\t" + "\t".join(f"n{p * 100:g}" for p in COVER) + "\t"
             + "\t".join(f"energy_n{p * 100:g}" for p in COVER)
             + "\tentropy_bits\teffective_experts\tgini\tmax_over_mean\ttop5_pct\tspearman_routings_mean"
             f"\ttop{TOP}_energy_overlap\ttop{TOP}_mean_overlap\n")
    pr = open(os.path.join(out, "prune.tsv"), "w")
    pr.write("layer\tkeep\tranking\troutings_kept_pct\tenergy_kept_pct\n")
    tot = {}  # (keep, ranking) -> [routings kept, routings, energy kept, energy, worst layer routings %, worst energy %]
    per_layer, n_all = {}, 0  # measure -> its value in each layer
    for L in layers:
        cnt = raw[(L, "gate", "counts")]
        for kind in ("up", "down"):
            other = raw.get((L, kind, "counts"))
            assert other is None or other == cnt, f"layer {L}: {kind} counts differ from gate's"
        n = len(cnt)
        energy = raw[(L, "down", "in_sum2")]
        x2 = raw[(L, "gate", "in_sum2")]
        mean = [energy[e] / cnt[e] if cnt[e] else 0.0 for e in range(n)]
        T, E = sum(cnt), math.fsum(energy)
        r_cnt, r_en = ranks(cnt), ranks(energy)
        for e in range(n):
            ex.write(f"{L}\t{e}\t{cnt[e]:.0f}\t{100 * cnt[e] / T:.4f}\t{r_cnt[e]:g}\t{energy[e]:.6g}\t"
                     f"{100 * energy[e] / E:.4f}\t{r_en[e]:g}\t{mean[e]:.6g}\t"
                     f"{x2[e] / cnt[e] if cnt[e] else 0.0:.6g}\n")
        desc_c, desc_e = sorted(cnt, reverse=True), sorted(energy, reverse=True)
        p = [c / T for c in cnt if c]
        H = -sum(q * math.log(q) for q in p)
        top5 = sorted(range(n), key=lambda e: (-cnt[e], e))[:5]
        used = [e for e in range(n) if cnt[e]]
        rho = spearman([cnt[e] for e in used], [mean[e] for e in used])
        tc = top(cnt, TOP)
        cov_c, cov_e = [cover(desc_c, T, q) for q in COVER], [cover(desc_e, E, q) for q in COVER]
        for q, v in zip(COVER, cov_c):
            per_layer.setdefault(f"n{q * 100:g}", []).append(v)
        for q, v in zip(COVER, cov_e):
            per_layer.setdefault(f"energy_n{q * 100:g}", []).append(v)
        per_layer.setdefault("unused", []).append(n - len(used))
        per_layer.setdefault("effective_experts", []).append(math.exp(H))
        per_layer.setdefault("gini", []).append(gini(cnt))
        n_all += n
        ly.write(f"{L}\t{T:.0f}\t{T / tokens if tokens else float('nan'):.3f}\t{n - len(used)}\t"
                 + "\t".join(map(str, cov_c)) + "\t" + "\t".join(map(str, cov_e))
                 + f"\t{H / math.log(2):.3f}\t{math.exp(H):.1f}\t{gini(cnt):.3f}\t{max(cnt) / (T / n):.2f}\t"
                 + ",".join(f"{e}:{100 * cnt[e] / T:.2f}" for e in top5)
                 + f"\t{rho:.3f}\t{len(tc & top(energy, TOP))}\t{len(tc & top(mean, TOP))}\n")
        for keep in KEEP:
            for name, score in (("routings", cnt), ("energy", energy), ("mean", mean)):
                kept = top(score, keep)
                kc, ke = sum(cnt[e] for e in kept), math.fsum(energy[e] for e in kept)
                pr.write(f"{L}\t{keep}\t{name}\t{100 * kc / T:.3f}\t{100 * ke / E:.3f}\n")
                s = tot.setdefault((keep, name), [0.0, 0.0, 0.0, 0.0, 100.0, 100.0])
                s[0] += kc; s[1] += T; s[2] += ke; s[3] += E
                s[4] = min(s[4], 100 * kc / T); s[5] = min(s[5], 100 * ke / E)
    with open(os.path.join(out, "summary.tsv"), "w") as sm:
        sm.write(f"# {len(layers)} layers, {chunks} chunks of {chunk} tokens ({tokens} tokens)\n")
        sm.write("keep\tranking\troutings_kept_pct\tworst_layer_routings_kept_pct\tenergy_kept_pct\t"
                 "worst_layer_energy_kept_pct\n")
        for (keep, name), s in sorted(tot.items()):
            sm.write(f"{keep}\t{name}\t{100 * s[0] / s[1]:.3f}\t{s[4]:.3f}\t{100 * s[2] / s[3]:.3f}\t{s[5]:.3f}\n")
    with open(os.path.join(out, "coverage.tsv"), "w") as cv:
        cv.write("measure\tmean\tmedian\tmin\tmax\tsum_over_layers\tsum_pct_of_all_experts\n")
        for k, v in per_layer.items():
            s = sorted(v)
            med = (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2
            cv.write(f"{k}\t{sum(v) / len(v):.3f}\t{med:g}\t{s[0]:g}\t{s[-1]:g}\t{sum(v):.6g}\t"
                     f"{100 * sum(v) / n_all:.2f}\n")
    for fh in (ex, ly, pr):
        fh.close()


if __name__ == "__main__":
    main()
