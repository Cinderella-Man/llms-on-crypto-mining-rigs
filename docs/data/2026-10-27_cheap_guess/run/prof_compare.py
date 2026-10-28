"""Per arm and card, the GPU time of a served replay from GGML_CUDA_PROFILE_HOST traces (replay.sh PROFILE=1): graph
evaluations, their GPU milliseconds in all and per generated token, and the host's time inside graph_compute. On the
head's card (CUDA0) the head's own graphs count with Flint's first stage, so the difference between the arms there is
the head's GPU work; copies outside the graphs (the taps) show only in the replay's wall time.

usage: python prof_compare.py --tokens N ARM=prof-ARM.csv ...   (N: the generated tokens of the replay, equal per arm)
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--tokens", type=int, required=True)
    a = ap.parse_args()
    per = {}
    for spec in a.arms:
        arm, path = spec.split("=", 1)
        d = defaultdict(lambda: [0, 0.0, 0.0])   # device -> evaluations, GPU ms, host ms in graph_compute
        for r in csv.DictReader(open(path)):
            x = d[int(r["device"])]
            x[0] += 1
            x[1] += float(r["gpu_us"]) / 1000.0
            x[2] += float(r["host_exit_ms"]) - float(r["host_enter_ms"])
        per[arm] = d
    devs = sorted({dv for d in per.values() for dv in d})
    print(f"GPU ms per generated token ({a.tokens} tokens), per card; evaluations in brackets\n")
    print("| card | " + " | ".join(per) + " |")
    print("|---|" + "---|" * len(per))
    for dv in devs:
        cells = [f"{per[arm][dv][1] / a.tokens:.3f} ({per[arm][dv][0]})" for arm in per]
        print(f"| CUDA{dv} | " + " | ".join(cells) + " |")
    tot = [f"{sum(x[1] for x in per[arm].values()) / a.tokens:.3f}" for arm in per]
    print("| all | " + " | ".join(tot) + " |")
    host = [f"{sum(x[2] for x in per[arm].values()) / a.tokens:.3f}" for arm in per]
    print("| host in graph_compute | " + " | ".join(host) + " |")


main()
