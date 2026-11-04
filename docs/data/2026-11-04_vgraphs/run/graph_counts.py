#!/usr/bin/env python3
"""C151 (Q701): the tables of rig_vgraphs.sh's output.

Usage: python3 -I graph_counts.py OUT [--tables DIR]

OUT holds one folder per arm (rig_vgraphs.sh). Per arm and window (the load, then each request, from marks.tsv's
line counts of the server log) it counts:
  - the CUDA graph events that GGML_CUDA_GRAPH_DEBUG=1 logs (the same messages on -ladder2 and on the new build):
    'warmup complete' (a graph is captured: a new key, or a key whose graph changed and then ran unchanged twice),
    'warmup reset' (a captured key's graph changed: the next evaluations run uncaptured until it is stable again),
    'update failed' (cudaGraphExecUpdate refused the new capture: re-instantiated), 'reused' (the llama graph was
    reused, so the CUDA graph launched without a property check), 'disabled' (graphs off for a key);
  - the server's steps (debug log): 'slot decode token' is a step of 1 token, 'generate_draft ... #draft=N' one of
    1 + N; the steps per width and how often the width changed from one step to the next;
  - the scheduler's graph reallocations and the per-size schedulers created (new build).
Then the memory: per stage (quiet, loaded, after each request) each card's VRAM, the host's MemAvailable, CmaFree,
usable (MemAvailable - CmaFree) and swap, the server's RSS (anonymous, file, shared), peak RSS and anonymous PSS; and
the memory breakdown llama-server prints at exit (with -lv 4 or more), per device. And per reply its tokens, speed and
draft acceptance.

Also each context's graph size (max_nodes, the reserved graphs' nodes and splits: one row per reserve, the
main model's first). Writes graph_counts.csv, widths.csv, memory.csv, vram.csv, breakdown.csv, replies.csv, graphs.csv
and summary.md into DIR (default OUT/tables).
"""
import csv
import json
import os
import re
import sys

EVENTS = [
    ("captures", re.compile(r"CUDA graph warmup complete")),
    ("warmup_resets", re.compile(r"CUDA graph warmup reset")),
    ("update_failed", re.compile(r"CUDA graph update failed")),
    ("reused", re.compile(r"CUDA Graph id \d+ reused")),
    ("disabled", re.compile(r"disabling CUDA graphs")),
    ("reallocs", re.compile(r"graph reallocation: nodes")),
    ("sched_created", re.compile(r"per-size schedulers")),
]
STEP1 = re.compile(r"slot decode token, id=")
STEPN = re.compile(r"generate_draft: id=.*#draft=(\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# the breakdown's columns are padded to a common width, so every gap is one or more spaces
GPU_ROW = re.compile(r"\|\s+- (\S+) \((.*?)\)\s+\|\s+(\d+)\s+=\s+(\d+)\s+\+\s+\((\d+)\s+=\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\)"
                     r"\s+\+\s+(-?\d+)\s+\|")
OTHER_ROW = re.compile(r"\|\s+- (.+?)\s+\|\s+(\d+)\s+=\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\s+\|")
RESERVE = re.compile(r"sched_reserve: graph.*nodes = ([^,]+), splits = ([^,]+)")
MAX_NODES = re.compile(r"max_nodes = (\d+)")


def read_marks(path):
    marks = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            marks.append((row["stage"], int(row["log_lines"])))
    return marks


def windows(marks, n_lines):
    """[(name, first line, last line)]: the load up to the first mark, then each request, then the exit."""
    out, prev = [], 0
    for stage, line in marks:
        out.append(("load" if stage == "loaded" else stage, prev, line))
        prev = line
    out.append(("exit", prev, n_lines))
    return out


def meminfo(path):
    vals, section = {}, "meminfo"
    with open(path) as f:
        for line in f:
            if line.startswith("--- "):
                section = line[4:].strip()
                continue
            m = re.match(r"(\w+):\s+(\d+) kB", line)
            if m:
                vals[(section, m.group(1))] = int(m.group(2)) / 1024.0
    return vals


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    out = args[0]
    tables = os.path.join(out, "tables")
    if "--tables" in args:
        tables = args[args.index("--tables") + 1]
    os.makedirs(tables, exist_ok=True)
    arms = sorted(d for d in os.listdir(out) if os.path.isfile(os.path.join(out, d, "marks.tsv")))
    order = ["ladder2", "vg", "vg-gps0", "vg-sps1"]
    arms.sort(key=lambda a: (order.index(a) if a in order else len(order), a))

    counts, widths, memory, vram, breakdown, replies, graphs = [], [], [], [], [], [], []
    for arm in arms:
        d = os.path.join(out, arm)
        logs = [f for f in os.listdir(d) if f.startswith("server-") and f.endswith(".log")]
        if not logs:
            continue
        with open(os.path.join(d, logs[0]), errors="replace") as f:
            lines = [ANSI.sub("", x) for x in f]
        for name, a, b in windows(read_marks(os.path.join(d, "marks.tsv")), len(lines)):
            row = {"arm": arm, "window": name}
            for key, rx in EVENTS:
                row[key] = sum(1 for x in lines[a:b] if rx.search(x))
            seq = []
            for x in lines[a:b]:
                if STEP1.search(x):
                    seq.append(1)
                else:
                    m = STEPN.search(x)
                    if m:
                        seq.append(1 + int(m.group(1)))
            row["steps"] = len(seq)
            row["width_changes"] = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            row["captures_per_100_steps"] = round(100.0 * row["captures"] / len(seq), 1) if seq else ""
            counts.append(row)
            for w in sorted(set(seq)):
                widths.append({"arm": arm, "window": name, "width": w, "steps": seq.count(w)})
        mx = [m.group(1) for m in (MAX_NODES.search(x) for x in lines) if m]
        rs = [m for m in (RESERVE.search(x) for x in lines) if m]
        for i, m in enumerate(rs):
            graphs.append({"arm": arm, "context": i, "max_nodes": mx[i] if i < len(mx) else "",
                           "nodes": m.group(1).strip(), "splits": m.group(2).strip()})
        for x in lines:
            m = GPU_ROW.search(x)
            if m:
                breakdown.append({"arm": arm, "device": m.group(1), "total": m.group(3), "free": m.group(4),
                                  "self": m.group(5), "model": m.group(6), "context": m.group(7),
                                  "compute": m.group(8), "unaccounted": m.group(9)})
                continue
            m = OTHER_ROW.search(x)
            if m:
                breakdown.append({"arm": arm, "device": m.group(1), "total": "", "free": "", "self": m.group(2),
                                  "model": m.group(3), "context": m.group(4), "compute": m.group(5),
                                  "unaccounted": ""})
        stages = ["quiet"] + [s for s, _ in read_marks(os.path.join(d, "marks.tsv"))]
        for st in stages:
            p = os.path.join(d, "vram_%s.csv" % st)
            if os.path.isfile(p):
                with open(p) as f:
                    for r in csv.reader(f):
                        if len(r) >= 3:
                            vram.append({"arm": arm, "stage": st, "card": r[0].strip(),
                                         "used_mib": r[1].strip().split()[0], "total_mib": r[2].strip().split()[0]})
            p = os.path.join(d, "host_%s.txt" % st)
            if os.path.isfile(p):
                v = meminfo(p)
                g = lambda s, k: round(v[(s, k)], 1) if (s, k) in v else ""
                avail, cma = g("meminfo", "MemAvailable"), g("meminfo", "CmaFree")
                memory.append({"arm": arm, "stage": st, "mem_available_mib": avail, "cma_free_mib": cma,
                               "usable_mib": round(avail - cma, 1) if avail != "" and cma != "" else "",
                               "swap_used_mib": round(v[("meminfo", "SwapTotal")] - v[("meminfo", "SwapFree")], 1)
                               if ("meminfo", "SwapTotal") in v else "",
                               "rss_mib": g("status", "VmRSS"), "rss_anon_mib": g("status", "RssAnon"),
                               "rss_file_mib": g("status", "RssFile"), "rss_shmem_mib": g("status", "RssShmem"),
                               "hwm_mib": g("status", "VmHWM"), "pss_anon_mib": g("smaps_rollup", "Pss_Anon")})
        n = 1
        while os.path.isfile(os.path.join(d, "reply_%d.json" % n)):
            try:
                with open(os.path.join(d, "reply_%d.json" % n)) as f:
                    t = json.load(f).get("timings", {})
            except ValueError:
                t = {}
            dn, da = t.get("draft_n"), t.get("draft_n_accepted")
            replies.append({"arm": arm, "request": n, "predicted_n": t.get("predicted_n", ""),
                            "predicted_per_second": round(t["predicted_per_second"], 2) if "predicted_per_second" in t else "",
                            "draft_n": dn if dn is not None else "", "draft_n_accepted": da if da is not None else "",
                            "acceptance": round(da / dn, 3) if dn else ""})
            n += 1

    def write(name, rows):
        if not rows:
            return
        with open(os.path.join(tables, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write("graph_counts.csv", counts)
    write("widths.csv", widths)
    write("memory.csv", memory)
    write("vram.csv", vram)
    write("breakdown.csv", breakdown)
    write("replies.csv", replies)
    write("graphs.csv", graphs)

    # summary: per arm the requests' captures, resets, re-instantiations, steps and width changes, the host once
    # loaded and the largest VRAM per card after the requests
    with open(os.path.join(tables, "summary.md"), "w") as f:
        f.write("| arm | captures (load / requests) | warmup resets (requests) | re-instantiations (requests) | "
                "steps (requests) | width changes | captures per 100 steps | usable host MiB loaded / last | "
                "server RSS MiB loaded / last | VRAM MiB per card, highest (card) after the requests |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for arm in arms:
            cs = [c for c in counts if c["arm"] == arm]
            load = sum(c["captures"] for c in cs if c["window"] == "load")
            req = [c for c in cs if c["window"].startswith("request_")]
            cap = sum(c["captures"] for c in req)
            rst = sum(c["warmup_resets"] for c in req)
            upd = sum(c["update_failed"] for c in req)
            steps = sum(c["steps"] for c in req)
            wch = sum(c["width_changes"] for c in req)
            per = round(100.0 * cap / steps, 1) if steps else ""
            ms = [m for m in memory if m["arm"] == arm]
            ml = next((m for m in ms if m["stage"] == "loaded"), None)
            mz = ms[-1] if ms else None
            vs = [v for v in vram if v["arm"] == arm and v["stage"].startswith("request_")]
            vmax = max(vs, key=lambda v: int(v["used_mib"])) if vs else None
            f.write("| %s | %d / %d | %d | %d | %d | %d | %s | %s / %s | %s / %s | %s |\n" % (
                arm, load, cap, rst, upd, steps, wch, per,
                ml["usable_mib"] if ml else "", mz["usable_mib"] if mz else "",
                ml["rss_mib"] if ml else "", mz["rss_mib"] if mz else "",
                "%s (%s)" % (vmax["used_mib"], vmax["card"]) if vmax else ""))
    print("tables in", tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
