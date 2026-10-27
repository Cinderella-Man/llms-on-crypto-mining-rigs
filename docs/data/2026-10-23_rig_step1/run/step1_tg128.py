#!/usr/bin/env python3
"""Registry row Q34 step 1 (2026-10-23): one row per run A-M from its llama-bench JSONL (pp512 and tg128, -fa on).

usage: step1_tg128.py DIR > step1_tg128.csv     (DIR holds s1-A/ ... s1-M/, each with one gpus_*.jsonl)
"""
import csv
import glob
import json
import os
import sys

GRAPHS = "graphs61"
W2 = "graphs61 + 2 warps"
WK = "graphs61 + 2 warps for rows >= 2048 (wide-K)"
STOCK = "stock d77256999"
# run: (build, build directory under /opt/llama-builds, file, rescale)
RUNS = {
    "A": (GRAPHS, "llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type",
          "makers' Q4_1 (flint-int4-q4_1.gguf)", "on"),
    "B": (W2, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps",
          "makers' Q4_1 (flint-int4-q4_1.gguf)", "on"),
    "C": (GRAPHS, "llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type",
          "makers-fast (Q4_1_64, fused QKV, F16 router)", "off"),
    "D": (W2, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps",
          "makers-fast (Q4_1_64, fused QKV, F16 router)", "off"),
    "E": (GRAPHS, "llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type",
          "Unsloth UD-Q4_K_XL", "on"),
    "F": (W2, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps",
          "Unsloth UD-Q4_K_XL", "on"),
    "G": (GRAPHS, "llama.cpp-20261022-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type",
          "Miner S4-L0", "off"),
    "H": (W2, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps",
          "Miner S4-L0", "off"),
    "I": (WK, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps-widek",
          "Miner S4-L0", "off"),
    "J": (WK, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps-widek",
          "Unsloth UD-Q4_K_XL", "on"),
    "K": (STOCK, "llama.cpp-20261022-d77256999", "Unsloth UD-Q4_K_XL", "on (no switch)"),
    "L": (WK, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps-widek",
          "Unsloth Q8_0", "on"),
    "M": (WK, "llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps-widek",
          "Unsloth Q8_0", "off"),
}


def main():
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["run", "build", "build_dir", "file", "file_bytes", "rescale", "cards", "pp512_tok_s", "pp512_sd",
                "tg128_tok_s", "tg128_sd", "tg128_samples"])
    for run in sorted(RUNS):
        (p,) = glob.glob(os.path.join(sys.argv[1], f"s1-{run}", "*.jsonl"))
        cards = len(os.path.basename(p)[5:-6].split(","))
        res = {}
        size = None
        for line in open(p):
            if line.strip():
                j = json.loads(line)
                res["pp" if j["n_prompt"] else "tg"] = j
                size = j["model_size"]
        build, bdir, fname, rescale = RUNS[run]
        pp, tg = res["pp"], res["tg"]
        w.writerow([run, build, bdir, fname, size, rescale, cards, round(pp["avg_ts"], 2), round(pp["stddev_ts"], 2),
                    round(tg["avg_ts"], 2), round(tg["stddev_ts"], 2),
                    " ".join(f"{x:.2f}" for x in tg["samples_ts"])])


if __name__ == "__main__":
    main()
