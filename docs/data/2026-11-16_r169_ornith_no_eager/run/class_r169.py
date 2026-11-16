"""R169's class (Q1496): o4ne (R169a o4nea, R169b o4neb, the released Ornith v4 preset less
GGML_CUDA_KEEPALIVE_EAGER=1) against o4sn (R166a o4sna, R166b o4snb, the released preset), R166's and R160's rule:
the means of each arm's two runs; a gain or a loss when the difference is larger than the larger of the two arms'
spreads (the gap between an arm's two runs), negligible otherwise. Reads agent.tsv (agent_tables.py's output) from
the folder given; prints a TSV."""
import csv
import sys

MEASURES = [  # column, label, lower is better
    ("equal_work_min_per_task", "minutes a task at equal work", True),
    ("gen_tok_s", "generation tok/s", False),
    ("prompt_tok_s", "prompt tok/s", False),
    ("ttft_median_s", "first token median s", True),
    ("ttft_p90_s", "first token p90 s", True),
    ("ttft_turn1_median_s", "first token on a task's first turn, median s", True),
]
NEW, REF = ("o4nea", "o4neb"), ("o4sna", "o4snb")


def main(folder):
    with open(f"{folder}/agent.tsv", newline="") as f:
        rows = {r["arm"]: r for r in csv.DictReader(f, delimiter="\t")}
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["measure", *REF, *NEW, "o4sn_mean", "o4ne_mean", "o4sn_spread", "o4ne_spread", "difference",
                "difference_pct", "class"])
    for col, label, lower in MEASURES:
        ref = [float(rows[a][col]) for a in REF]
        new = [float(rows[a][col]) for a in NEW]
        mr, mn = sum(ref) / 2, sum(new) / 2
        sr, sn = abs(ref[0] - ref[1]), abs(new[0] - new[1])
        diff = mn - mr
        pct = 100 * diff / mr
        if abs(diff) > max(sr, sn) + 1e-9:
            cls = "gain" if (diff < 0) == lower else "loss"
        else:
            cls = "negligible"
        w.writerow([label, *[rows[a][col] for a in REF], *[rows[a][col] for a in NEW], f"{mr:.4f}", f"{mn:.4f}",
                    f"{sr:.2f}", f"{sn:.2f}", f"{diff:+.4f}", f"{pct:+.1f}", cls])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
