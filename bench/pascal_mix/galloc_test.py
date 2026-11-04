#!/usr/bin/env python3
"""Self-test of galloc.py solve: on a switch table (galloc.py switches' output) restricted to three groups, with a
synthetic KLD table, the solver's pick at each budget must be the best of every combination of the groups' options
(found by brute force), within the byte and ms budgets.
  python3 -I galloc_test.py SWITCHES.tsv WORKDIR
Prints `ok` or the first difference and exits 1. Standard library only.
"""
import csv, itertools, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sw_path, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    rows = list(csv.DictReader(open(sw_path), delimiter="\t"))
    base = [r for r in rows if r["name"] == "base"][0]
    keep = [r for r in rows if r["name"] != "base" and r["group"] in ("1", "3", "5") and int(r["type_changes"]) > 0]
    sub = os.path.join(work, "switches.tsv")
    with open(sub, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(keep + [base])
    rnd = random.Random(7)
    ref = {"agent": 0.2792, "wiki": 0.4264, "code": 0.1080}
    kld = {(n, t): v for t, v in ref.items() for n in ["pmix-q2-fast"]}
    for r in keep:
        for t, v in ref.items():
            kld[(r["name"], t)] = v * (1 + rnd.uniform(-0.08, 0.02))
    kpath = os.path.join(work, "kld.tsv")
    with open(kpath, "w") as f:
        f.write("name\ttext\tkld\tkld_se\tsame_top_pct\tsame_top_se\n")
        for (n, t), v in kld.items():
            f.write(f"{n}\t{t}\t{v:.6f}\t0.005\t90\t0.1\n")
    on_card, ms0 = int(base["delta_bytes"]), float(base["delta_ms_timed"])
    budgets = [on_card + x for x in (5e8, 2e9, 4e9)]
    ms_budget = ms0 + 6.0
    out = os.path.join(work, "solve.tsv")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "galloc.py"), "solve", "--switches", sub, "--kld", kpath,
                    "--out", out, "--budgets", ",".join(f"{b:.0f}" for b in budgets), "--ms-budget", str(ms_budget)],
                   check=True, stdout=subprocess.DEVNULL)
    got = {r["budget_bytes"]: r for r in csv.DictReader(open(out), delimiter="\t")}
    # brute force: per group and class, none or one switch
    cls = {}
    for r in keep:
        cls.setdefault((r["group"], r["class"]), []).append(r)
    choices = [[None] + v for v in cls.values()]
    damage = {r["name"]: sum((kld[(r["name"], t)] - v) / v for t, v in ref.items()) for r in keep}
    for b in budgets:
        best = None
        for combo in itertools.product(*choices):
            picks = [r for r in combo if r]
            by = sum(int(r["delta_bytes"]) for r in picks)
            ms = sum(float(r["delta_ms_timed"]) for r in picks)
            if on_card + by > b or ms0 + ms > ms_budget:
                continue
            d = sum(damage[r["name"]] for r in picks)
            if best is None or d < best[0] - 1e-12:
                best = (d, sorted(r["name"] for r in picks))
        g = got.get(f"{b:.0f}")
        mine = sorted(x for x in g["switches"].split(",") if x != "-") if g else None
        if mine != best[1] and abs(float(g["damage"]) - best[0]) > 1e-4:
            print(f"budget {b:.0f}: solve picked {mine} (damage {g['damage']}), brute force {best[1]} ({best[0]:.4f})")
            sys.exit(1)
        if int(g["on_card_bytes"]) > b or float(g["est_ms"]) > ms_budget + 1e-6:
            print(f"budget {b:.0f}: the pick breaks a budget: {g}")
            sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
