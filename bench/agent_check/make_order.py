"""Write tasks/polyglot/order.txt, the fixed order in which run.py takes the tasks.

    python3 make_order.py [--calibration attempts.csv] [--seed 1]

Every prefix of the order is a proportional sample of the strata, so a pilot that runs the first few tasks sees both
tracks and the whole range of difficulty. Strata are track × tercile, where the tercile is taken within the track of
the reference solution's length (before calibration) or of the calibration pass rate (with --calibration, the
attempts.csv written by report.py; the tasks every attempt passed and those every attempt failed form two more
strata per track). Within a stratum the order is a seeded shuffle. The next task is always taken from the
track that is furthest behind its share, and within the track from the stratum that is furthest behind (systematic
sampling at both levels), so the tracks alternate from the start.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks" / "polyglot"


def terciles(items, key):
    s = sorted(items, key=key)
    n = len(s)
    return [s[: n // 3], s[n // 3: 2 * n // 3], s[2 * n // 3:]]


def systematic(lists):
    """Merge lists so that every prefix takes from each list in proportion to its length: the next item comes from
    the list whose next item's ideal position ((k + 0.5) / length) is earliest."""
    taken = [0] * len(lists)
    out = []
    for _ in range(sum(map(len, lists))):
        i = min((k for k in range(len(lists)) if taken[k] < len(lists[k])),
                key=lambda k: ((taken[k] + 0.5) / len(lists[k]), k))
        out.append(lists[i][taken[i]])
        taken[i] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tasks = [json.loads(p.read_text()) for p in sorted(TASKS.glob("*/task.json"))]
    rate = {}
    if a.calibration:
        per = defaultdict(list)
        for r in csv.DictReader(a.calibration.open()):
            per[r["task_id"]].append(int(r["pass"]))
        rate = {k: sum(v) / len(v) for k, v in per.items()}
    rng = random.Random(a.seed)
    tracks = {}
    for track in ("python", "javascript"):
        tt = [t for t in tasks if t["track"] == track]
        if rate:
            mixed = [t for t in tt if 0 < rate.get(t["id"], 0.5) < 1]
            groups = terciles(mixed, lambda t: -rate.get(t["id"], 0.5))
            groups += [[t for t in tt if rate.get(t["id"], 0.5) == 1], [t for t in tt if rate.get(t["id"], 0.5) == 0]]
        else:
            groups = terciles(tt, lambda t: t["reference_lines"])
        strata = []
        for g in groups:
            if g:
                ids = sorted(t["id"] for t in g)
                rng.shuffle(ids)
                strata.append(ids)
        tracks[track] = systematic(strata)
    order = systematic(list(tracks.values()))
    (TASKS / "order.txt").write_text("\n".join(order) + "\n")
    print(f"{len(order)} tasks; first 9: {' '.join(order[:9])}")


if __name__ == "__main__":
    main()
