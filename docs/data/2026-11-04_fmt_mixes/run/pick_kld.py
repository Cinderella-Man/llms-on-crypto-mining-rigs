#!/usr/bin/env python3
"""A built pick's measured KLD against galloc.py solve's prediction, one row per text (standard library).

For each text: the start's measured KLD and its ±, the pick's predicted KLD (solve.tsv) and the change it predicts, the
pick's measured KLD, its ± and same-top-token share (llama-perplexity's log <logs>/<name>.<text>.log), the measured
change, measured minus predicted in KLD and in the measured ± (llama-perplexity's standard error of the mean KLD),
whether that is within two of them (Q167's test), the measured change over the predicted one, and the root sum of
squares of the picked switches' own ± from solve.tsv. Changes are taken from the unrounded scores.

  python3 -I pick_kld.py --solve solve.tsv --budget 37470389760 --logs <score folder> --name gmix-q2 > kld_pick.tsv
"""
import argparse
import csv
import re
import sys

KLD = re.compile(r"^Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)")
TOP = re.compile(r"^Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s*%")


def score(path):
    kld = top = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = KLD.match(line)
            if m:
                kld = (float(m.group(1)), float(m.group(2)))
            m = TOP.match(line)
            if m:
                top = (float(m.group(1)), float(m.group(2)))
    if kld is None or top is None:
        sys.exit(f"{path}: no Mean KLD / Same top p line")
    return kld, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", required=True, help="galloc.py solve's table (solve.tsv)")
    ap.add_argument("--budget", required=True, help="the budget_bytes of the solve's row that was built")
    ap.add_argument("--logs", required=True, help="folder of <name>.<text>.log, the start's under its own name")
    ap.add_argument("--name", required=True, help="the built pick's name in the logs")
    ap.add_argument("--start", default="pmix-q2-fast")
    ap.add_argument("--texts", default="agent,wiki,code")
    a = ap.parse_args()
    with open(a.solve, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["budget_bytes"] == a.budget]
    if len(rows) != 1:
        sys.exit(f"{a.solve}: {len(rows)} rows with budget_bytes {a.budget}")
    r = rows[0]
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["text", "start_kld", "start_se", "pred_kld", "pred_change", "kld", "kld_se", "change", "diff",
                "diff_in_se", "within_2se", "change_ratio", "same_top_pct", "same_top_se", "switch_se"])
    for t in a.texts.split(","):
        (s, s_se), _ = score(f"{a.logs}/{a.start}.{t}.log")
        (k, k_se), (top, top_se) = score(f"{a.logs}/{a.name}.{t}.log")
        p = float(r[f"pred_{t}"])
        d = k - p
        w.writerow([t, f"{s:.6f}", f"{s_se:.6f}", f"{p:.5f}", f"{p - s:+.5f}", f"{k:.6f}", f"{k_se:.6f}",
                    f"{k - s:+.6f}", f"{d:+.6f}", f"{d / k_se:+.2f}", "yes" if abs(d) <= 2 * k_se else "no",
                    f"{(k - s) / (p - s):.3f}", f"{top:.3f}", f"{top_se:.3f}", r[f"switch_se_{t}"]])


if __name__ == "__main__":
    main()
