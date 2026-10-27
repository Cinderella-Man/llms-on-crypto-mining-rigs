#!/usr/bin/env python3
"""Every llama-bench row of scripts/launch_overhead_probe.sh's JSONL files as one CSV.

usage: tg64_grid.py DIR > tg64_grid.csv
DIR holds <model>.idle.jsonl, <model>.busy1.jsonl and <model>.busy2.jsonl (busy1: a busy loop on CPU core 1;
busy2: on both cores).
"""
import csv
import glob
import json
import os
import sys

BUSY = {"idle": 0, "busy1": 1, "busy2": 2}


def main():
    rows = []
    for p in glob.glob(os.path.join(sys.argv[1], "*.jsonl")):
        model, cond = os.path.basename(p)[:-6].rsplit(".", 1)
        for line in open(p):
            if not line.strip():
                continue
            j = json.loads(line)
            rows.append([model, j["model_type"], j["model_size"], cond, BUSY[cond], j["n_threads"], j["flash_attn"],
                         j["n_depth"], j["n_gen"], round(j["avg_ts"], 2), round(j["stddev_ts"], 2),
                         len(j["samples_ts"]), " ".join(f"{x:.2f}" for x in j["samples_ts"]), j["build_commit"],
                         j["gpu_info"]])
    rows.sort(key=lambda r: (r[0].lower(), r[4], r[5], r[6], r[7]))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["model", "model_type", "model_bytes", "cpu_load", "busy_cores", "threads", "flash_attn", "depth",
                "n_gen", "tok_s", "tok_s_sd", "reps", "samples_tok_s", "build", "gpu"])
    w.writerows(rows)


if __name__ == "__main__":
    main()
