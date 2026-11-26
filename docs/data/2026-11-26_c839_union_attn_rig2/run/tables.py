#!/usr/bin/env python3
"""C839's tables from run/c839.sh's outputs (test-backend-ops perf, --output sql, one file a run).

Usage: python3 -I tables.py <collected folder> <output folder>
Writes calls.tsv (every timed call: arm, type, kv, nb, splits, repeat, us a call, runs), clocks.tsv (the card's clocks
while each run's cases were timed), union_fit.tsv (the union kernel's time against its list's length) and table.md.
"""
import math
import os
import re
import statistics
import sys

FIELDS = ["test_time", "build_commit", "backend_name", "op_name", "op_params", "test_mode", "supported", "passed",
          "error_message", "time_us", "flops", "bandwidth_gb_s", "memory_kb", "n_runs", "device_description",
          "backend_reg_name"]
N_KV_MAX = 2051
LAYERS = 12  # Flash-Next's attention layers: one call each a verify pass
# the engine's split count at nsm 5 (fasp_pick_splits): per-token lists 1 token 10, 4 tokens 3; union lists 10
ENGINE_S = {("token", 1): 10, ("token", 4): 3, ("union", 4): 10}


def parse_sql(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            vals = re.findall(r"'([^']*)'", line.split("VALUES", 1)[1])
            r = dict(zip(FIELDS, vals))
            p = dict(kv.split("=", 1) for kv in re.findall(r"(\w+=[^,\[]+|\w+=\[[^\]]*\])", r["op_params"]))
            rows.append({"type": p["type_K"], "kv": int(p["kv"]), "nb": int(p["nb"]), "us": float(r["time_us"]),
                         "runs": int(r["n_runs"]), "supported": r["supported"], "commit": r["build_commit"]})
    return rows


def union_cells(kv, nb):
    """expected cells of one union list over nb query rows whose kept cells test-backend-ops draws independently
    (init_tensor_kq_mask_sparse: row r keeps 2,051 - r % 17 random cells)"""
    miss = 1.0
    for r in range(nb):
        miss *= 1.0 - (N_KV_MAX - r % 17)/kv
    return kv*(1.0 - miss)


def fmt(xs, nd=1):
    return ", ".join(f"{x:.{nd}f}" for x in xs)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    calls = []  # arm, splits, type, kv, nb, repeat, us, runs
    commits = set()
    for name in sorted(os.listdir(src)):
        m = re.fullmatch(r"(dense|token|union)-r(\d+)\.sql|sweep-(token|union)-S(\d+)-r(\d+)\.sql", name)
        if not m:
            continue
        if m.group(1):
            arm, splits, rep = m.group(1), 0, int(m.group(2))
        else:
            arm, splits, rep = m.group(3), int(m.group(4)), int(m.group(5))
        for r in parse_sql(os.path.join(src, name)):
            if r["supported"] != "1":
                raise SystemExit(f"{name}: a case not supported")
            commits.add(r["commit"])
            calls.append((arm, splits, r["type"], r["kv"], r["nb"], rep, r["us"], r["runs"]))
    calls.sort(key=lambda c: (c[0], c[1], c[2], c[3], c[4], c[5]))
    with open(os.path.join(out, "calls.tsv"), "w", encoding="utf-8") as f:
        f.write("arm\tsplits\ttype\tkv\tnb\trepeat\tus_call\truns\n")
        for c in calls:
            f.write("\t".join(str(x) for x in c[:6]) + f"\t{c[6]:.3f}\t{c[7]}\n")

    def get(arm, splits, typ, kv, nb):
        return [c[6] for c in sorted(calls, key=lambda c: c[5])
                if c[0] == arm and c[1] == splits and c[2] == typ and c[3] == kv and c[4] == nb]

    reps = sorted({c[5] for c in calls})
    kvs = sorted({c[3] for c in calls if c[1] == 0})
    lines = [f"# C839 tables (test-backend-ops perf, build {', '.join(sorted(commits))}, {len(reps)} rounds)", ""]

    # 1. every arm, us a call, each round by value
    lines += ["## Flash-Next's attention call, us a call (each round by value)", "",
              "| Cache | Cells | Tokens | dense | per-token | union | union list (est. cells) |",
              "|---|---:|---:|---|---|---|---:|"]
    for typ in ("f16", "q8_0"):
        for kv in kvs:
            for nb in (1, 4):
                d, t, u = get("dense", 0, typ, kv, nb), get("token", 0, typ, kv, nb), get("union", 0, typ, kv, nb)
                lines.append(f"| {typ} | {kv:,} | {nb} | {fmt(d)} | {fmt(t)} | {fmt(u) if u else '-'} | "
                             f"{union_cells(kv, nb):,.0f} |" if nb > 1 else
                             f"| {typ} | {kv:,} | {nb} | {fmt(d)} | {fmt(t)} | - | - |")
    lines.append("")

    # 2. a verify pass's 12 calls, ms, at R121's depths
    lines += ["## A verify pass's 12 attention calls, ms (12 x us a call, each round by value)", "",
              "| Cache | Cells | Tokens | dense | per-token | union |", "|---|---:|---:|---|---|---|"]
    for typ in ("f16", "q8_0"):
        for kv in (4096, 8192):
            for nb in (1, 4):
                row = []
                for arm in ("dense", "token", "union"):
                    xs = get(arm, 0, typ, kv, nb)
                    row.append(fmt([x*LAYERS/1000 for x in xs], 2) if xs else "-")
                lines.append(f"| {typ} | {kv:,} | {nb} | " + " | ".join(row) + " |")
    lines.append("")

    # 3. the union kernel against its list's length (F16 and Q8_0): least squares over every round
    fit_rows = []
    lines += ["## The union kernel's time against its list's length (4 tokens; least squares over every round)", ""]
    for typ in ("f16", "q8_0"):
        xs, ys = [], []
        for kv in kvs:
            for y in get("union", 0, typ, kv, 4):
                xs.append(union_cells(kv, 4))
                ys.append(y)
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        b = sum((x - mx)*(y - my) for x, y in zip(xs, ys))/sum((x - mx)**2 for x in xs)
        a = my - b*mx
        res = [y - (a + b*x) for x, y in zip(xs, ys)]
        rmax = max(abs(r) for r in res)
        at = {c: a + b*c for c in (2051, 2655, 3841)}
        fit_rows.append((typ, a, b, rmax, at))
        lines.append(f"- {typ}: {a:.1f} us + {b*1000:.2f} us per 1,000 cells of the list (largest residual "
                     f"{rmax:.1f} us over {n} calls); at 2,051 cells {at[2051]:.1f} us, at 2,655 {at[2655]:.1f}, "
                     f"at 3,841 {at[3841]:.1f}")
    lines.append("")
    with open(os.path.join(out, "union_fit.tsv"), "w", encoding="utf-8") as f:
        f.write("type\tus_fixed\tus_per_cell\tmax_residual_us\tus_at_2051\tus_at_2655\tus_at_3841\n")
        for typ, a, b, rmax, at in fit_rows:
            f.write(f"{typ}\t{a:.3f}\t{b:.6f}\t{rmax:.3f}\t{at[2051]:.3f}\t{at[2655]:.3f}\t{at[3841]:.3f}\n")

    # 4. the split counts
    lines += ["## Split counts (F16, us a call, each round by value; * the engine's count at nsm 5)", "",
              "| Kernel | Cells | Tokens | " + " | ".join(f"S {s}" for s in sorted({c[1] for c in calls if c[1]})) + " |",
              "|---|---:|---:|" + "---|"*len({c[1] for c in calls if c[1]})]
    splits = sorted({c[1] for c in calls if c[1]})
    for arm, nb in (("token", 1), ("token", 4), ("union", 4)):
        for kv in (4096, 8192):
            cells = []
            for s in splits:
                xs = get(arm, s, "f16", kv, nb)
                star = "*" if ENGINE_S[(arm, nb)] == s else ""
                cells.append((star + fmt(xs)) if xs else "-")
            lines.append(f"| {arm} | {kv:,} | {nb} | " + " | ".join(cells) + " |")
    lines.append("")

    # 5. the clocks while the cases were timed (samples in P0, each run)
    clk = []
    for name in sorted(os.listdir(src)):
        if not name.endswith(".clocks"):
            continue
        sm, mem = [], []
        with open(os.path.join(src, name), encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4 or parts[3] != "P0":
                    continue
                sm.append(int(parts[1].split()[0]))
                mem.append(int(parts[2].split()[0]))
        if sm:
            clk.append((name[:-7], len(sm), statistics.median(sm), min(sm), max(sm), statistics.median(mem), min(mem),
                        max(mem)))
    with open(os.path.join(out, "clocks.tsv"), "w", encoding="utf-8") as f:
        f.write("run\tp0_samples\tsm_median_mhz\tsm_min\tsm_max\tmem_median_mhz\tmem_min\tmem_max\n")
        for c in clk:
            f.write("\t".join(f"{x:.0f}" if isinstance(x, float) else str(x) for x in c) + "\n")
    if clk:
        def counts(xs):
            vals = sorted({round(x) for x in xs})
            return ", ".join(f"{v:,} MHz in {sum(1 for x in xs if round(x) == v)}" for v in vals)
        lines += ["## Clocks", "",
                  f"Median SM clock of each run's P0 samples: {counts([c[2] for c in clk])} of {len(clk)} runs; "
                  f"memory {counts([c[5] for c in clk])} ([clocks.tsv](clocks.tsv)).", ""]
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
