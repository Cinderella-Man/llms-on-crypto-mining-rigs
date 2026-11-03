#!/usr/bin/env python3
"""Tables from rig_kcol.sh's perf logs: per product and token count, us per call with the new kernels off and on, the
speed-up, and each arm's cost relative to its own one-token call. Standard library only.
Usage: perf_table.py <out folder> > perf.tsv   (reads perf_kcol_{off,on}.log and perf_bf16_{off,on}.log)"""
import re
import sys
from pathlib import Path

LINE = re.compile(r"^\s+(\w+)\((.*)\):\s+(\d+) runs -\s+([\d.]+) us/run")


def read(path):
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            params = dict(kv.split("=", 1) for kv in re.split(r",(?![^\[]*\])", m.group(2)) if "=" in kv)
            out[m.group(2)] = (params, float(m.group(4)))
    return out


def main():
    d = Path(sys.argv[1])
    print("kind\ttype\tm\tk\tn\tus_off\tus_on\tspeedup\toff_vs_n1\ton_vs_n1")
    for kind in ("kcol", "bf16"):
        off, on = read(d / f"perf_{kind}_off.log"), read(d / f"perf_{kind}_on.log")
        rows = []
        for key, (p, t_on) in on.items():
            if key not in off:
                continue
            rows.append((p["type_a"], int(p["m"]), int(p["k"]), int(p["n"]), off[key][1], t_on))
        base = {}
        for t, m, k, n, a, b in rows:
            if n == 1:
                base[(t, m, k)] = (a, b)
        for t, m, k, n, a, b in sorted(rows):
            ba, bb = base.get((t, m, k), (0.0, 0.0))
            print(f"{kind}\t{t}\t{m}\t{k}\t{n}\t{a:.2f}\t{b:.2f}\t{a / b:.3f}\t"
                  f"{a / ba if ba else 0:.3f}\t{b / bb if bb else 0:.3f}")


if __name__ == "__main__":
    main()
