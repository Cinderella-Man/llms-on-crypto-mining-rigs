#!/usr/bin/env python3
"""common_turns.py ARM=TURNS_CSV ...: the first token over the task-turns every run has.

A run's first-token median and p90 are taken over all its turns, so a run whose agent took more short late turns gets a
lower median from its work, not from its speed. This prints, per run, its turns, its turns at turn 5 or later, and the
first-token median and p90 over the (task, attempt, turn) keys present in every run given, with those turns' new prompt
tokens, so the runs compare on the same turns. Standard library only (python3 -I).
"""
import csv
import statistics
import sys


def pct(values, q):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main(argv):
    runs = []
    for a in argv:
        name, path = a.split("=", 1)
        with open(path, newline="") as f:
            rows = {(r["task_id"], r["attempt"], int(r["turn"])): r for r in csv.DictReader(f)}
        runs.append((name, rows))
    if not runs:
        sys.exit("usage: common_turns.py ARM=TURNS_CSV ...")
    common = set.intersection(*(set(rows) for _, rows in runs))
    print(f"{len(common)} task-turns in every run of {', '.join(n for n, _ in runs)}\n")
    print("| run | turns | turns at turn 5 or later | first token median s, common turns | p90 s, common turns "
          "| new prompt tokens of the common turns |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, rows in runs:
        t = [float(rows[k]["ttft_s"]) for k in common]
        n = sum(int(rows[k]["prompt_n"]) for k in common)
        late = sum(1 for k in rows if k[2] >= 5)
        print(f"| {name} | {len(rows)} | {late} | {statistics.median(t):.2f} | {pct(t, 0.9):.2f} | {n} |")


if __name__ == "__main__":
    main(sys.argv[1:])
