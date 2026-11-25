#!/usr/bin/env python3
"""C1025's rig check (Q1758): the tables from rig_probe.py's output folder, each run by value, nothing averaged.

  python3 -I docs/data/2026-11-24_c1025_nvml_probe/run/rig_tables.py <rig_probe's out folder> <folder for the tables>

Reads results.csv and logs/ and writes:
  rig_runs.tsv   one row per run: its place in the pair, wall time, peak resident memory, the probe line's figures, and
                 from the log's own timestamps the CUDA start's span from "found N CUDA devices" to ggml_cuda_init's
                 last line (the probe's line, printed whether it ran or was skipped); for a load also the tensor read
                 (from "loading model tensors" to "constructing llama_context"), the time to /health, the peak at
                 /health, the host's MemAvailable before and lowest, and whether its reply equals the other arm's
  rig_pairs.tsv  one row per device-list pair and one for the two loads: probe minus unset for each of those figures
Standard library only.
"""
import csv, os, re, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
TS = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")


def t(line):
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def spans(path):
    found = probe = tens = ctx = None
    with open(path, errors="replace") as f:
        for l in f:
            if found is None and "found" in l and "CUDA devices" in l:
                found = t(l)
            elif probe is None and "NVML probe" in l:
                probe = t(l)
            elif tens is None and "loading model tensors" in l:
                tens = t(l)
            elif ctx is None and "constructing llama_context" in l:
                ctx = t(l)
    init = "" if None in (found, probe) else f"{(probe - found) * 1e3:.3f}"
    read = "" if None in (tens, ctx) else f"{ctx - tens:.3f}"
    return init, read


with open(os.path.join(src, "results.csv")) as f:
    rows = list(csv.DictReader(f))

out = []
for i, r in enumerate(rows):
    log = os.path.join(src, "logs", f"{r['phase']}-{r['rep']}-{r['arm']}.log" if r["phase"] == "list" else f"load-{r['arm']}.log")
    init, read = spans(log)
    same_phase = [x for x in rows[:i] if x["phase"] == r["phase"] and x["rep"] == r["rep"]]
    place = "second" if same_phase else "first"  # each pair (and the two loads) in the order they ran
    out.append(dict(r, place=place, init_span_ms=init, tensor_read_s=read))

loads = [o for o in out if o["phase"] == "load"]
for o in loads:
    other = [x for x in loads if x is not o]
    o["reply_same"] = ("yes" if other and o["reply_tokens"] and o["reply_tokens"] == other[0]["reply_tokens"] else "no") if other else ""

COLS = ["phase", "rep", "arm", "place", "rc", "wall_s", "maxrss_kib", "probe", "open_init_ms", "close_ms", "rss_init_mib",
        "rss_close_mib", "init_span_ms", "tensor_read_s", "health_s", "vmhwm_health_kib", "memavail_before_kib",
        "memavail_min_kib", "stopped", "reply_same"]
with open(os.path.join(dst, "rig_runs.tsv"), "w") as f:
    f.write("\t".join(COLS) + "\n")
    for o in out:
        f.write("\t".join(str(o.get(c, "")) for c in COLS) + "\n")


def diff(a, b, key, fmt):
    try:
        return fmt % (float(b[key]) - float(a[key]))
    except (ValueError, KeyError):
        return ""


PC = ["phase", "rep", "order", "wall_s", "maxrss_kib", "init_span_ms", "tensor_read_s", "health_s", "vmhwm_health_kib",
      "memavail_before_kib", "memavail_min_kib"]
with open(os.path.join(dst, "rig_pairs.tsv"), "w") as f:
    f.write("# probe minus unset, each pair by value\n" + "\t".join(PC) + "\n")
    groups = {}
    for o in out:
        groups.setdefault((o["phase"], o["rep"]), []).append(o)
    for (phase, rep), g in groups.items():
        u = [x for x in g if x["arm"] == "unset"]
        p = [x for x in g if x["arm"] == "probe"]
        if not (u and p):
            continue
        u, p = u[0], p[0]
        order = "unset first" if u["place"] == "first" else "probe first"
        f.write("\t".join([phase, rep, order, diff(u, p, "wall_s", "%+.4f"), diff(u, p, "maxrss_kib", "%+.0f"),
                           diff(u, p, "init_span_ms", "%+.3f"), diff(u, p, "tensor_read_s", "%+.3f"),
                           diff(u, p, "health_s", "%+.3f"), diff(u, p, "vmhwm_health_kib", "%+.0f"),
                           diff(u, p, "memavail_before_kib", "%+.0f"), diff(u, p, "memavail_min_kib", "%+.0f")]) + "\n")

for name in ("rig_runs.tsv", "rig_pairs.tsv"):
    print(f"== {name}")
    print(open(os.path.join(dst, name)).read())
