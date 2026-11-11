#!/usr/bin/env python3
"""long_stages.py: where a step's long host waits sit, from LLAMA_STEP_TRACE beside GGML_CUDA_PROFILE_HOST (R86, Q1313).

  python3 -I long_stages.py TRACE.jsonl HOST_PROFILE.csv --profile-t0 MS [--over-ms 20] [--main-device 0] > long.md

For every generation line (drafted tokens and an acceptance) in order: the request it belongs to (a line without a draft
starts a new one), the step's length (s.upd → s.upd.end), the gap's host part (the last drafting pass's wait's end →
v.ub.in), v.ub.in → the main device's graph_compute entry (the profile, moved onto the trace's clock by
--profile-t0, align_t0.py's value), and every interval between consecutive marks over --over-ms other than the waits
for the cards (*.sync → *.sync.end) and the verify pass's queued work (v.ub.in → v.ub.comp, split by the main device's
entry instead). Prints the lines that hold such an interval or a card-0 lag over --over-ms, and a count per stage.
"""
import argparse
import bisect
import csv
import json
import re
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("--profile-t0", type=float, required=True)
    ap.add_argument("--over-ms", type=float, default=20.0)
    ap.add_argument("--main-device", type=int, default=0)
    a = ap.parse_args()

    with open(a.trace) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    header, lines = rows[0], rows[1:]
    shift = a.profile_t0 - header["t0_steady_ms"]
    with open(a.profile) as f:
        entries = sorted(float(r["host_enter_ms"]) + shift for r in csv.DictReader(f) if int(r["device"]) == a.main_device)

    req, k_in_req, out, per_stage = 0, 0, [], Counter()
    for o in lines:
        names = [m[0] for m in o["t"]]
        if not (o["n"].get("drf", 0) > 0 and "s.acc" in names):
            req += 1
            k_in_req = 0
            continue
        k_in_req += 1
        ts = dict()
        for n, t in o["t"]:
            ts.setdefault(n, t)
        t_vin = ts.get("v.ub.in")
        i_drf = names.index("s.drf.out") if "s.drf.out" in names else None
        t_dse = max((t for n, t in o["t"][:i_drf] if n == "d.sync.end"), default=None) if i_drf else None
        j = bisect.bisect_left(entries, t_vin) if t_vin is not None else len(entries)
        lag = entries[j] - t_vin if j < len(entries) else float("nan")
        long = []
        for (n0, t0), (n1, t1) in zip(o["t"], o["t"][1:]):
            if t1 - t0 <= a.over_ms or n1 == n0 + ".end" or (n0 == "v.ub.in" and n1 == "v.ub.comp"):
                continue
            stage = re.sub(r"#\d+$", "", n0) + " → " + re.sub(r"#\d+$", "", n1)
            long.append(f"{stage} {t1 - t0:.1f}")
            per_stage[stage] += 1
        if lag > a.over_ms:
            per_stage["v.ub.in → card's graph_compute entry"] += 1
        if long or lag > a.over_ms:
            out.append([str(req), str(k_in_req), str(o["step"]), f"{o['t'][-1][1] - o['t'][0][1]:.1f}",
                        f"{t_vin - t_dse:.2f}" if t_dse is not None and t_vin is not None else "—", f"{lag:.2f}",
                        "; ".join(long) or "—"])
    print(f"# {a.trace}: intervals over {a.over_ms} ms\n")
    print(f"| request | step in the request | line | the line, ms | gap's host part, ms | v.ub.in → card {a.main_device}'s "
          f"entry, ms | other intervals over {a.over_ms} ms |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for r in out:
        print("| " + " | ".join(r) + " |")
    print("\n| stage | lines |\n|---|---:|")
    for s, c in per_stage.most_common():
        print(f"| {s} | {c} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
