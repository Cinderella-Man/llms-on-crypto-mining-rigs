#!/usr/bin/env python3
"""How a failed greedy check differs from the card's stock result: the text, or only the probabilities.

  greedy_diff.py OUT_DIR TABLE_CSV [STOCK_DIR] [RUN]

OUT_DIR is climb.py's out folder (card*/greedy-*.json and .out). For every check whose .out says DIFFERS, compares
its JSON with the same card's greedy-stock.json, prompt by prompt, and writes one row per failed check and prompt:
whether the generated text is identical, the first token whose chosen token differs, how many chosen tokens differ,
the first token whose top-10 log-probabilities differ in any digit, and the largest change of the chosen token's
log-probability before the text parts (0 when the text never parts).
STOCK_DIR (default OUT_DIR) holds cardN/greedy-stock.json, for a soak folder whose card folders keep only the stock hash;
RUN names the run in a first column.
"""
import csv
import glob
import json
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    out, table = sys.argv[1], sys.argv[2]
    stock_dir = sys.argv[3] if len(sys.argv) > 3 else out
    run = sys.argv[4] if len(sys.argv) > 4 else "climb"
    rows = []
    for o in sorted(glob.glob(os.path.join(out, "card*", "greedy-*.out"))):
        with open(o) as f:
            text = f.read()
        if "DIFFERS" not in text:
            continue
        card = int(os.path.basename(os.path.dirname(o))[4:])
        tag = os.path.basename(o)[len("greedy-"):-len(".out")]
        ref = load(os.path.join(stock_dir, f"card{card}", "greedy-stock.json"))
        got = load(o[:-len(".out")] + ".json")
        for i, (r, g) in enumerate(zip(ref, got)):
            rp = r["response"]["completion_probabilities"] or []
            gp = g["response"]["completion_probabilities"] or []
            first_tok = first_prob = None
            n_tok = 0
            max_d = 0.0
            for j, (a, b) in enumerate(zip(rp, gp)):
                if a["id"] != b["id"]:
                    n_tok += 1
                    if first_tok is None:
                        first_tok = j
                elif first_tok is None:
                    max_d = max(max_d, abs(a["logprob"] - b["logprob"]))
                if first_prob is None and a.get("top_logprobs") != b.get("top_logprobs"):
                    first_prob = j
            rows.append({"run": run, "card": card, "check": tag, "prompt": i + 1, "tokens": len(gp),
                         "text_identical": r["response"]["content"] == g["response"]["content"],
                         "first_token_differs": "" if first_tok is None else first_tok,
                         "tokens_differ": n_tok,
                         "first_probability_differs": "" if first_prob is None else first_prob,
                         "max_chosen_logprob_change_before": f"{max_d:.3g}"})
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["card"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
