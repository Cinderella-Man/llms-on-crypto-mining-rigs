#!/usr/bin/env python3
"""kshape_est.py: the k-column launch shapes of Q823 (kcol::pick_shape's new defaults) against the bench's variants,
and from them the inside-ggml estimate behind Q824's dispatch table.

From bench/mmvq_pascal's mmvq_kcol CSVs (variants pascal = the default before Q823, r1, r2, itt1, s32k =
GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768): per (type, shape, k) the shape the new rule picks, named by the measured variant
that has it, and its time; '?' where no variant has it. The rule (mmvq-pascal-kcol-impl.cuh): R = 2 only for >= 2 columns,
>= 8 blocks per SM, registers for 3 blocks per SM, within 32 KB, not a grid type; else R = 1 within 32 KB where the
registers allow 3 blocks and the grid has > 2 blocks per SM, else within 48 KB; where nothing fits, the old default.
Registers -> blocks: 65536 / (128 threads x registers rounded up to 8), at most 16.

  python3 -I kshape_est.py shapes NSM kcol_dense.csv kcol_iq.csv > shapes.tsv
  python3 -I kshape_est.py ggml NSM kcol_dense.csv kcol_iq.csv -- perf_dense.tsv perf_cov.tsv > ggml_est.tsv"""
import collections, csv, math, sys

GRID = {"iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_m"}


def blocks_by_regs(regs):
    if regs <= 0:
        return 0
    return min(16, 65536 // (128 * (int(math.ceil(regs / 8.0)) * 8)))


def load(files):
    cases = collections.defaultdict(dict)
    calls = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r.get("check") != "PASS" or not r["us_median"] or r["variant"] == "prod":
                continue
            key = (r["type"], r["shape"], int(r["K"]), int(r["N"]), int(r["tokens"]))
            cases[key][r["variant"]] = r
            calls[key] = float(r["calls_per_token"])
    return cases, calls


def new_shape(key, v, nsm):
    """(variant name, row) of the bench variant with the new default's shape, or ('?', None)."""
    typ, k = key[0], key[4]
    s32, r1, r2, pas = v.get("s32k"), v.get("r1"), v.get("r2"), v.get("pascal")
    if s32 is not None and int(s32["R"]) == 2:
        regs2 = int(r2["regs"]) if r2 else int(s32["regs"])
        if typ not in GRID and k >= 2 and int(s32["grid"]) >= 8 * nsm and blocks_by_regs(regs2) >= 3:
            return "s32k", s32
    if r1 is None:
        return ("pascal", pas) if pas else ("?", None)
    budget32 = blocks_by_regs(int(r1["regs"])) >= 3 and int(r1["grid"]) > 2 * nsm
    if not budget32:
        return "r1", r1
    if s32 is not None and int(s32["R"]) == 1:
        return "s32k", s32
    if int(r1["smem"]) <= 32768:
        return "r1", r1
    if s32 is None:
        # nothing fits 32 KB at R = 1: the old default (pascal)
        return ("pascal", pas) if pas else ("?", None)
    return "?", None


def main():
    mode, nsm = sys.argv[1], int(sys.argv[2])
    args = sys.argv[3:]
    perf_files = []
    if "--" in args:
        i = args.index("--")
        args, perf_files = args[:i], args[i + 1:]
    cases, calls = load(args)
    us = lambda r: float(r["us_median"]) if r is not None else float("nan")
    if mode == "shapes":
        print("type\tshape\tK\tN\tk\tcalls_per_token\tnew_variant\tnew_R\tnew_itt\tnew_smem\tnew_us\ts32k_us\tpascal_us\tbest_variant\tbest_us\tnew_vs_s32k\tnew_vs_best")
        tot = collections.Counter()
        for key in sorted(cases, key=lambda k: (k[4], k[0], k[1])):
            v = cases[key]
            name, r = new_shape(key, v, nsm)
            s32 = v.get("s32k")
            pas = v.get("pascal")
            best = min(v.values(), key=lambda x: float(x["us_median"]))
            n_us = us(r)
            # where the 32 KB arm declined, ggml ran the earlier kernels: count it at the default's time here
            s_us = us(s32) if s32 is not None else us(pas)
            print("\t".join(str(x) for x in (key[0], key[1], key[2], key[3], key[4], f"{calls[key]:g}", name,
                  r["R"] if r else "", r["itt"] if r else "", r["smem"] if r else "", f"{n_us:.1f}", f"{s_us:.1f}",
                  f"{us(pas):.1f}", best["variant"], f"{us(best):.1f}",
                  f"{n_us / s_us:.3f}" if r else "", f"{n_us / us(best):.3f}" if r else "")))
            if r is not None and pas is not None:
                c = calls[key] / 1000
                tot[(key[4], "new")] += c * n_us
                tot[(key[4], "s32k")] += c * s_us
                tot[(key[4], "pascal")] += c * us(pas)
                tot[(key[4], "best")] += c * us(best)
        print("\n# k\tpascal_ms_per_token\ts32k_ms_per_token\tnew_ms_per_token\tbest_ms_per_token")
        for k in sorted({k for k, _ in tot}):
            print(f"# {k}\t{tot[(k, 'pascal')]:.2f}\t{tot[(k, 's32k')]:.2f}\t{tot[(k, 'new')]:.2f}\t{tot[(k, 'best')]:.2f}")
        return
    # ggml: perf rows (kind type m k n us_off us_on ...): m rows, k the row width, n tokens
    shape_of = {}
    for key in cases:
        shape_of[(key[0], key[2], key[3], key[4])] = key
    print("type\trows\tK\tk\tbench_shape\tus_off\tus_on_old\tnew_variant\tratio_new_vs_old_default\tus_on_new_est\tspeedup_new_est\tus_on_s32k_est\tspeedup_s32k_est\tkcol_wins_after")
    for f in perf_files:
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["kind"] != "kcol":
                continue
            typ, m, kk, n = r["type"], int(r["m"]), int(r["k"]), int(r["n"])
            key = shape_of.get((typ, kk, m, n))
            off, on = float(r["us_off"]), float(r["us_on"])
            if key is None:
                print(f"{typ}\t{m}\t{kk}\t{n}\t-\t{off:.1f}\t{on:.1f}\t-\t\t\t\t\t\t{'yes' if on < off else 'no'}")
                continue
            v = cases[key]
            name, nr = new_shape(key, v, nsm)
            pas, s32 = v.get("pascal"), v.get("s32k")
            ratio = us(nr) / us(pas) if nr is not None and pas is not None else float("nan")
            rs = us(s32) / us(pas) if s32 is not None and pas is not None else float("nan")
            est = on * ratio
            ests = on * rs
            wins = "?" if math.isnan(est) else ("yes" if est < off else "no")
            print(f"{typ}\t{m}\t{kk}\t{n}\t{key[1]}\t{off:.1f}\t{on:.1f}\t{name}\t{ratio:.3f}\t{est:.1f}\t{off / est:.3f}\t{ests:.1f}\t{off / ests:.3f}\t{wins}")


main()
