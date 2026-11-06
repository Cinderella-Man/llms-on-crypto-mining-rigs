#!/usr/bin/env python3
"""One text's KLD table of the single-switch files, from llama-perplexity's logs (<logs>/<name>.<text>.log).

Each file's mean KLD and its ±, the change from the start (pmix-q2-fast) and from the control (dniq23), the same-top-token
share and two classes: against the start, a gain is below it by more than the start's ± (llama-perplexity's standard error
of the mean KLD), a loss above it by more, the rest negligible; against the control, the same with the control's ±. The
first six columns (class, layers, donor, MiB and est. ms added) are copied from META, the agent-text table, so the rows
and their order are its own. Changes are taken from the unrounded scores.

  python3 -I kld_table.py --meta kld_agent.tsv --logs <score folder> --text wiki > kld_wiki.tsv
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


def verdict(d, se):
    return "gain" if d < -se else "loss" if d > se else "negligible"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="the agent-text table (kld_agent.tsv): rows and first six columns")
    ap.add_argument("--logs", required=True, help="folder of <name>.<text>.log")
    ap.add_argument("--text", required=True, help="wiki, agent or code")
    ap.add_argument("--start", default="pmix-q2-fast")
    ap.add_argument("--control", default="dniq23")
    a = ap.parse_args()
    with open(a.meta, encoding="utf-8") as f:
        meta = list(csv.DictReader(f, delimiter="\t"))
    scores = {r["name"]: score(f"{a.logs}/{r['name']}.{a.text}.log") for r in meta}
    s, s_se = scores[a.start][0]
    c, c_se = scores[a.control][0]
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["name", "class", "layers", "donor", "delta_mib", "delta_ms_timed", f"kld_{a.text}", "kld_se", "d_kld",
                "same_top_pct", "same_top_se", "verdict", "d_ctrl", "verdict_ctrl"])
    for r in meta:
        n = r["name"]
        (k, k_se), (t, t_se) = scores[n]
        if r["class"] == "file":
            v = "start" if n == a.start else "reference"
            vc = "-"
        else:
            v = verdict(k - s, s_se)
            vc = "control" if n == a.control else verdict(k - c, c_se)
        w.writerow([n, r["class"], r["layers"], r["donor"], r["delta_mib"], r["delta_ms_timed"], f"{k:.4f}",
                    f"{k_se:.4f}", f"{k - s:+.4f}", f"{t:.2f}", f"{t_se:.2f}", v, f"{k - c:+.4f}", vc])


if __name__ == "__main__":
    main()
