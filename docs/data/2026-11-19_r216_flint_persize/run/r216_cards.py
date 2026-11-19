#!/usr/bin/env python3
"""r216_cards.py: each card's memory in agent runs, from the harness's card log (cards.csv, every card every 0.5 s).

    python3 -I r216_cards.py LABEL=cards.csv[@HH:MM:SS] ...

For each run (LABEL) and each card: the MiB in use at the first sample at or after HH:MM:SS (the moment the client
first reached the server, so the loaded model before any request; omitted: not shown), the peak MiB over the whole log
and when it was first reached, and the card's total from --total (default 6144 MiB, a P106-090) less the peak. Prints a
TSV on stdout. Standard library only.
"""
import csv
import sys

TOTAL = 6144


def read(path):
    rows = []
    for r in csv.reader(open(path)):
        if not r or r[0].startswith("#") or r[0] == "epoch":
            continue
        rows.append(r)
    return rows


def main():
    total = TOTAL
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--total="):
            total = int(a.split("=", 1)[1])
        else:
            args.append(a)
    print("run\tcard\tloaded_mib\tpeak_mib\tpeak_first_at\tfree_at_peak_mib")
    for a in args:
        label, rest = a.split("=", 1)
        path, _, at = rest.partition("@")
        rows = read(path)
        n = (len(rows[0]) - 3) // 8
        for g in range(n):
            col = 3 + g * 8 + 5
            loaded = ""
            if at:
                for r in rows:
                    if r[1].split(" ")[1][:8] >= at and r[col].isdigit():
                        loaded = int(r[col])
                        break
            peak, when = -1, ""
            for r in rows:
                if r[col].isdigit() and int(r[col]) > peak:
                    peak, when = int(r[col]), r[1].split(" ")[1][:12]
            print(f"{label}\t{g}\t{loaded}\t{peak}\t{when}\t{total - peak}")


if __name__ == "__main__":
    main()
