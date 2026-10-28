"""Per agent session, what llama-server's lookups would guess at each generated position, as train.py's
--source-labels: the copy lookup (0), the request's continuation of a 2-token or longer context (1), of a 1-token
context (2), none (3), and 4 for tool results and each request's first token (no guess is made there). The lookups are
cost_sim.py's port of common/pipe-guess.cpp on the true prefix (source_split.py does the same per scored position).

usage: python source_labels.py SESSIONS_SEQ.jsonl OUT.npz [--jobs 8]
OUT.npz: one uint8 array per session, keyed by the seq id (the id's numeric end), one label per token of gen_tokens.
"""
import argparse
import importlib.util
import json
import os
from multiprocessing import Pool

import numpy as np

_spec = importlib.util.spec_from_file_location("cost_sim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      "cost_sim.py"))
cost_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cost_sim)
LABEL = {"copy": 0, "cont2": 1, "cont1": 2, "none": 3}


def label_row(line):
    r = json.loads(line)
    seq = r["prompt_tokens"] + r["gen_tokens"]
    n_prompt = len(r["prompt_tokens"])
    ranges = r.get("gen_ranges") or [(n_prompt, len(seq))]
    out = np.full(len(r["gen_tokens"]), 4, np.uint8)
    g = cost_sim.ServerGuesser()
    k = 0
    for s, e in ranges:
        while k < s:
            g.push(seq[k])
            k += 1
        for a in range(s, e):
            if a > s:
                g.count_to(len(g.hist) - 2)
                if g.copy_guess() is not None:
                    src = "copy"
                elif g.cont_guess(2) is not None:
                    src = "cont2"
                elif g.cont_guess(1) is not None:
                    src = "cont1"
                else:
                    src = "none"
                out[a - n_prompt] = LABEL[src]
            g.push(seq[a])
            k += 1
    return str(int(r["id"].split("_")[-1])), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    with open(a.sessions) as f, Pool(a.jobs) as pool:
        res = dict(pool.imap_unordered(label_row, f, chunksize=4))
    np.savez_compressed(a.out, **res)
    counts = np.bincount(np.concatenate(list(res.values())), minlength=5)
    names = ("copy", "cont2", "cont1", "none", "other")
    print(f"{len(res)} sessions; " + ", ".join(f"{n} {c} ({c / counts.sum():.3f})" for n, c in zip(names, counts)))


main()
