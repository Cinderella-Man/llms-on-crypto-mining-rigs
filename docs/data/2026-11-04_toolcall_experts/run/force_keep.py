"""The experts a usage cut must keep for tool calls (Q897): per layer, the experts the cut drops that carry a share of
the tool-call spans' routings, as an id list for scripts/gguf_prune.py --force-ids.
  force_keep.py --span-experts DIR/span_experts.tsv.gz --experts experts.tsv.gz --keep 464 --out DIR
                [--classes call,argument] [--max-per-layer 8] [--min-share PCT] [--min-routings 20]
--span-experts is span_counts.py's table; --experts the ranking the cut is made by (Q232's experts.tsv.gz). An expert
the cut at --keep drops (rank > N) is forced when it carries at least X % of its layer's span routings (the classes
in --classes, summed) and at least --min-routings of them. Without --min-share, X is the smallest of a grid
(0.05-10 %) at which no layer forces more than --max-per-layer experts; when even 10 % forces more, each layer keeps
its --max-per-layer largest. The cut stays at N experts a layer: a forced expert takes the place of the lowest-ranked
expert the cut would keep (gguf_prune.py's --force-ids rule), so the file keeps its size and its fit.
Writes to DIR:
  force_<N>.tsv        `layer<TAB>id,id,...` for every layer (the --force-ids format)
  force_<N>_ids.tsv    every forced expert: its span routings and share, its rank and routings in the ranking, and
                       the expert it displaces
  force_<N>_sweep.tsv  per X: forced experts (most in a layer, mean, total) and the share of span routings and of all
                       routings (the ranking's text and this pass) the cut drops, without and with the forced ids
Standard library only; gguf_prune.py from the repository's scripts/ checks that its own selection gives the same ids.
"""
import argparse
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_prune import rank_ids  # noqa: E402

GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)


def load(path, classes):
    """{layer: {expert: (rank, ranking routings, this pass's routings, span routings)}}"""
    per = {}
    with gzip.open(path, "rt") as f:
        head = f.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(head)}
        missing = [c for c in classes if c not in col]
        assert not missing, f"{path}: no class columns {missing}"
        for line in f:
            c = line.rstrip("\n").split("\t")
            per.setdefault(int(c[0]), {})[int(c[1])] = (
                int(c[col["ranking_rank"]]), float(c[col["ranking_routings"]]), int(c[col["routings"]]),
                sum(int(c[col[k]]) for k in classes))
    return per


def kept(experts, n, forced):
    others = sorted((r, e) for e, (r, *_x) in experts.items() if e not in forced)
    return set(forced) | {e for _r, e in others[:n - len(forced)]}


def dropped(per, n, force):
    """Shares (%) the cut drops: span routings, the ranking's routings, this pass's routings."""
    tot, drop = [0.0] * 3, [0.0] * 3
    for layer, ex in per.items():
        k = kept(ex, n, force.get(layer, []))
        for e, (_r, rr, pr, sr) in ex.items():
            for i, v in enumerate((sr, rr, pr)):
                tot[i] += v
                if e not in k:
                    drop[i] += v
    return [100 * d / t if t else 0.0 for d, t in zip(drop, tot)]


def choose(per, n, x, min_routings, cap=None):
    force = {}
    for layer, ex in per.items():
        span = sum(v[3] for v in ex.values())
        cand = sorted(((-v[3], e) for e, v in ex.items()
                       if v[0] > n and v[3] >= min_routings and span and 100 * v[3] / span >= x))
        force[layer] = [e for _v, e in cand[:cap]]
    return force


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-experts", required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="call,argument")
    ap.add_argument("--max-per-layer", type=int, default=8)
    ap.add_argument("--min-share", type=float)
    ap.add_argument("--min-routings", type=int, default=20)
    a = ap.parse_args()
    n = a.keep
    per = load(a.span_experts, a.classes.split(","))
    os.makedirs(a.out, exist_ok=True)
    base = dropped(per, n, {})
    sweep, pick = [], None
    for x in ([a.min_share] if a.min_share is not None else GRID):
        force = choose(per, n, x, a.min_routings)
        most = max(len(v) for v in force.values())
        sweep.append((x, force, most))
        if pick is None and most <= a.max_per_layer:
            pick = (x, force)
    if pick is None:
        x = sweep[-1][0]
        pick = (x, choose(per, n, x, a.min_routings, a.max_per_layer))
        print(f"even {x} % forces {sweep[-1][2]} experts in a layer: each layer keeps its {a.max_per_layer} largest")
    x, force = pick
    with open(os.path.join(a.out, f"force_{n}_sweep.tsv"), "w") as f:
        f.write(f"# keep {n}, span classes {a.classes}, min routings {a.min_routings}; dropped shares in %: span, the "
                f"ranking's routings, this pass's routings; without forcing: {base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}\n")
        f.write("min_share_pct\tmost_in_a_layer\tmean_per_layer\ttotal\tspan_dropped_pct\tranking_dropped_pct\t"
                "pass_dropped_pct\n")
        for sx, sf, most in sweep:
            d = dropped(per, n, sf)
            tot = sum(len(v) for v in sf.values())
            f.write(f"{sx:g}\t{most}\t{tot / len(sf):.2f}\t{tot}\t{d[0]:.3f}\t{d[1]:.3f}\t{d[2]:.3f}\n")
    with open(os.path.join(a.out, f"force_{n}.tsv"), "w") as f:
        f.write("layer\tforced_ids\n" + "".join(f"{l}\t{','.join(map(str, sorted(force[l])))}\n" for l in sorted(force)))
    with open(os.path.join(a.out, f"force_{n}_ids.tsv"), "w") as f:
        f.write("layer\texpert\tspan_routings\tspan_share_pct\tranking_rank\tranking_routings\tdisplaces\n")
        for layer in sorted(force):
            ex = per[layer]
            span = sum(v[3] for v in ex.values())
            plain = kept(ex, n, [])
            out = sorted(plain - kept(ex, n, force[layer]), key=lambda e: -ex[e][0])
            for e in sorted(force[layer], key=lambda e: -ex[e][3]):
                f.write(f"{layer}\t{e}\t{ex[e][3]}\t{100 * ex[e][3] / span:.3f}\t{ex[e][0]}\t{ex[e][1]:.0f}\t"
                        f"{out.pop(0) if out else '-'}\n")
    # the file gguf_prune.py would build keeps the same ids
    ids = rank_ids(a.experts, "routings", n, force)
    assert all(set(ids[l]) == kept(per[l], n, force[l]) for l in per), "gguf_prune.py's selection differs"
    d = dropped(per, n, force)
    tot = sum(len(v) for v in force.values())
    print(f"keep {n}: forced {tot} experts at {x:g} % ({max(len(v) for v in force.values())} at most in a layer); span "
          f"routings dropped {base[0]:.3f} -> {d[0]:.3f} %, the ranking's {base[1]:.3f} -> {d[1]:.3f} %, this pass's "
          f"{base[2]:.3f} -> {d[2]:.3f} %")
    print(f"gguf_prune.py prune --keep {n} --experts {a.experts} --force-ids {os.path.join(a.out, f'force_{n}.tsv')} "
          f"--out-dir <dir> SHARD...")


if __name__ == "__main__":
    main()
