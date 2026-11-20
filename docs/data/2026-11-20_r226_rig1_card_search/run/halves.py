#!/usr/bin/env python3
"""R226: one row per half of the card search.

    python3 -I halves.py load MODEL_LOG > <half>/load.tsv
    python3 -I halves.py table STOCK_ROUNDS a:<half a folder> [b:<half b folder> ...] > halves.md

`load` takes one half's model_check.sh log (OUT/model.log, profile rounds only) and writes its load and end lines as a
table: when the server was ready, the load in s, MiB used on cards 0 to 8, host memory available, and the done line's
rounds, seconds and exit. R209's model_rounds.py parses the log.

`table` reads each half folder's model_rounds.tsv (model_rounds.py `rounds`) and load.tsv, and STOCK_ROUNDS, the rounds
table that holds the stock rounds every half is compared with (R224's model_rounds.tsv: its stock round 1 is the stored
reference, its round 2 the stock speed). Per half: the load, the rounds and how many gave the stock answer, the first
differing round and when it came (minutes after the server was ready), and generation tok/s (the median of rounds 2 on,
against stock's round 2). Standard library only.
"""
import csv
import datetime
import os
import statistics
import sys


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def pct(a, b):
    return f"{(b - a) / a * 100:+.2f} %"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cmd_load(log):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "2026-11-19_r209_oc_v5",
                                    "run"))
    from model_rounds import parse
    _, ready, done = parse(log)
    r, d = ready["profile"], done.get("profile")
    print("\t".join(("ready", "load_s", "mib_cards_0_to_8", "host_mib", "done", "rounds", "secs", "exit")))
    print("\t".join(map(str, (r["time"], r["load_s"], " ".join(r["mib"]), r["host"]) + (
        (d["time"], d["rounds"], d["secs"], d["exit"]) if d else ("", "", "", "")))))


def cmd_table(stock_tsv, halves):
    st = {int(r["round"]): r for r in rows(stock_tsv) if r["tag"] == "stock"}
    ref, s2 = st[1]["hash"], st[2]
    print("| half | load, s | rounds | the stock answer | first differing round | minutes after the load | "
          "the rounds' time, s | generation tok/s, median of rounds 2 on (prompts 1, 2) | against stock's round 2 |")
    print("|---|---:|---:|---:|---|---:|---:|---|---|")
    for name, folder in halves:
        p = [r for r in rows(os.path.join(folder, "model_rounds.tsv")) if r["tag"] == "profile"]
        ld = rows(os.path.join(folder, "load.tsv"))[0]
        same = [r for r in p if r["hash"] == ref]
        diff = [r for r in p if r["hash"] != ref]
        first = (f"round {diff[0]['round']} (`{diff[0]['hash']}`)",
                 f"{(ts(diff[0]['time']) - ts(ld['ready'])).seconds / 60:.1f}") if diff else ("none", "")
        later = [r for r in p if int(r["round"]) >= 2]
        gen = vs = ""
        if later:
            g1 = statistics.median(float(r["gen1"]) for r in later)
            g2 = statistics.median(float(r["gen2"]) for r in later)
            gen = f"{g1:.2f}, {g2:.2f} ({len(later)} rounds)"
            vs = f"{pct(float(s2['gen1']), g1)}, {pct(float(s2['gen2']), g2)}"
        print(f"| {name} | {ld['load_s']} | {len(p)} | {len(same)} of {len(p)} | {first[0]} | {first[1]} | "
              f"{ld['secs']} | {gen} | {vs} |")
    print(f"\nThe stock reference is `{ref}`, stock's round 1 in {stock_tsv.split('docs/data/')[-1]}; stock's round 2 "
          f"generated {float(s2['gen1']):.2f} and {float(s2['gen2']):.2f} tok/s.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "load":
        cmd_load(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "table" and all(":" in a for a in sys.argv[3:]):
        cmd_table(sys.argv[2], [tuple(a.split(":", 1)) for a in sys.argv[3:]])
    else:
        sys.exit(__doc__)
