"""How much of DFlash2's acceptance comes from copied text: tokens per step of each request against the request's
copy share, one weighted least-squares line, with a bootstrap over requests.

    TABLES=$WORK/tables ACC=draft_acceptance.csv RUN=dflash2 ARM=dflash2b python3 dflash_fit.py

ACC is the server's per-request acceptance log of that arm (docs/data/2026-11-03_flash_next_dflash2/agent/dflash2b/
draft_acceptance.csv: accepted and generated draft tokens, mean_accepted_len = tokens per step), one row per request
in order. The line y = a + b x (y = tokens per step, x = share of the request's tokens at least 8 tokens into a copy
of the prompt or of its own earlier text) gives a, the tokens per step on text with no copy, and a + b for text that
is all copy. Requests are matched to the rows by order and checked by their generated-token counts.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import quantile  # noqa: E402


def fit(pts):
    sw = sum(w for _, _, w in pts)
    mx = sum(x * w for x, _, w in pts) / sw
    my = sum(y * w for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    b = sum(w * (x - mx) * (y - my) for x, y, w in pts) / sxx
    return my - b * mx, b


def main():
    tables = os.environ["TABLES"]
    rows = [r for r in csv.DictReader(open(os.path.join(tables, "per_request.csv")))
            if r["run"] == os.environ["RUN"] and r["arm"] == os.environ["ARM"]]
    acc = list(csv.DictReader(open(os.environ["ACC"])))
    if len(rows) != len(acc):
        raise SystemExit("%d requests but %d acceptance rows" % (len(rows), len(acc)))
    col = os.environ.get("COL", "covered_own_L8")
    pts, bad = [], 0
    for r, a in zip(rows, acc):
        steps = float(a["generated"]) / float(os.environ.get("DRAFTED", "2"))
        tokens = float(a["accepted"]) + steps
        if abs(tokens - float(r["tokens"])) > 3:
            bad += 1
        pts.append((int(r[col]) / int(r["tokens"]), float(a["mean_accepted_len"]), steps))
    print("%d requests; %d whose token counts differ by more than 3 from the log" % (len(rows), bad))
    a0, b0 = fit(pts)
    rng = random.Random(3)
    boots = []
    for _ in range(2000):
        s = [pts[rng.randrange(len(pts))] for _ in pts]
        try:
            boots.append(fit(s))
        except ZeroDivisionError:
            pass
    lo = sorted(x[0] for x in boots)
    hi = sorted(x[0] + x[1] for x in boots)
    tw = sum(w for _, _, w in pts)
    mean_x = sum(x * w for x, _, w in pts) / tw
    mean_y = sum(y * w for _, y, w in pts) / tw
    print("copy share weighted by steps: %.3f; tokens per step %.3f" % (mean_x, mean_y))
    open(os.path.join(tables, "tau_out.txt"), "w").write("%.2f\n" % a0)
    print("line: tokens per step = %.3f + %.3f x share" % (a0, b0))
    print("no copy (x = 0): %.2f (95%% interval %.2f-%.2f); all copy (x = 1): %.2f (%.2f-%.2f)" % (
        a0, quantile(lo, 0.025), quantile(lo, 0.975), a0 + b0, quantile(hi, 0.025), quantile(hi, 0.975)))
    with open(os.path.join(tables, "dflash2_vs_copy_share.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["copy_share", "tokens_per_step", "steps"])
        for x, y, s in pts:
            w.writerow(["%.4f" % x, "%.3f" % y, "%.1f" % s])


if __name__ == "__main__":
    main()
