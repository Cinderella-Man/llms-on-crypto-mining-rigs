#!/usr/bin/env python3
"""Q550: the per-device memory of pipelined speculation from memory.sh's llama-server logs.

For each log server-d<D>.log: whether pipelining came up ("on", "failed" to allocate, or "off" at depth 0), per
device the model buffer, the KV buffers (attention cache and indexer cache), the recurrent-state buffer before
llama_pipe_init (one group) and after it (its state groups), the main compute buffer, the pipe slots' compute buffers
and, from gpu-d<D>.txt, nvidia-smi's used and total MiB of card <i> for device CUDA<i> (meaningful on the rig, where
the devices are the cards); for the host (device "host"), the server's mapped file pages in model_mib, its anonymous
memory (pinned buffers included) in compute_mib and the CPU backend's pipe-slot buffers. Writes a CSV (one row per
depth and device) and prints what pipelining adds over the devices, at most on one device, and the fullest card.

Usage: python3 memory_table.py <dir with server-d*.log> <out.csv>
"""
import csv
import re
import sys
from pathlib import Path

RE_MODEL = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB")
RE_KV = re.compile(r"llama_kv_cache:\s+(\S+) KV buffer size =\s+([\d.]+) MiB")
RE_RS = re.compile(r"(?:llama_memory_recurrent|alloc_tensors):\s+(\S+) RS buffer size =\s+([\d.]+) MiB")
RE_GROUPS = re.compile(r"set_n_groups: (\d+) state groups")
RE_COMP = re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB")
RE_PIPE = re.compile(r"pipe_init:\s+(\S+) pipe slots' compute buffers =\s+([\d.]+) MiB")
RE_SLOTS = re.compile(r"pipe_init: (\d+) slots \+ (\d+) chunk slots")
RE_PIPE_ON = re.compile(r"pipelined speculation: up to \d+ guesses .*?(\d+) pipe slots")
RE_PIPE_FAIL = re.compile(r"failed to set up the pipe slots")


def parse(path):
    dev = {}
    groups = 1
    slots = 0
    rs_seen = {}
    comp_seen = set()
    state = "off"
    seen_groups = False
    for line in path.read_text(errors="replace").splitlines():
        if m := RE_PIPE_ON.search(line):
            state = "on"
            slots = slots or int(m[1])
        elif RE_PIPE_FAIL.search(line):
            state = "failed"
        if m := RE_MODEL.search(line):
            dev.setdefault(m[1], {})["model"] = float(m[2])
        elif m := RE_KV.search(line):
            d = dev.setdefault(m[1], {})
            d["kv"] = d.get("kv", 0.0) + float(m[2])
        elif m := RE_GROUPS.search(line):
            groups = int(m[1])
            seen_groups = True
        elif m := RE_RS.search(line):
            d = dev.setdefault(m[1], {})
            # the first RS line per device is the one-group buffer, a later one llama_pipe_init's groups
            if m[1] in rs_seen:
                d["rs_groups"] = float(m[2])
            else:
                d["rs_1"] = float(m[2])
                rs_seen[m[1]] = True
        elif m := RE_COMP.search(line):
            # sched_reserve runs again in llama_pipe_init: keep the first size, they are the same
            if m[1] not in comp_seen:
                dev.setdefault(m[1], {})["compute"] = float(m[2])
                comp_seen.add(m[1])
        elif m := RE_PIPE.search(line):
            dev.setdefault(m[1], {})["pipe"] = float(m[2])
        elif m := RE_SLOTS.search(line):
            slots = int(m[1]) + int(m[2])
    if state == "on" and not seen_groups:
        groups = slots + 1  # no set_n_groups line (below -lv 5): 2 D + 3 groups for 2 (D + 1) slots
    return dev, groups, slots, state


def main():
    src = Path(sys.argv[1])
    rows = []
    for log in sorted(src.glob("server-d*.log"), key=lambda p: int(p.stem.split("-d")[1])):
        depth = int(log.stem.split("-d")[1])
        dev, groups, slots, state = parse(log)
        smi = {}
        gpu = src / f"gpu-d{depth}.txt"
        if gpu.exists():
            for line in gpu.read_text().splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) == 3 and f[0].isdigit():
                    smi[f"CUDA{f[0]}"] = (float(f[1]), float(f[2]))
        worst = (0.0, "")
        fullest = (-1.0, "")
        for name in sorted(dev, key=lambda n: (not n.startswith("CUDA"), int(re.sub(r"\D", "", n) or 0))):
            d = dev[name]
            if "model" not in d or name.startswith("CPU") or name.endswith("Host"):
                continue
            rs_1 = d.get("rs_1", 0.0)
            rs_g = d.get("rs_groups", rs_1)
            added = rs_g - rs_1 + d.get("pipe", 0.0)
            worst = max(worst, (added, name))
            used, total = smi.get(name, (0.0, 0.0))
            fullest = max(fullest, (used, name))
            rows.append({"depth": depth, "pipelining": state, "pipe_slots": slots, "state_groups": groups,
                         "device": name,
                         "model_mib": d.get("model", 0.0), "kv_mib": round(d.get("kv", 0.0), 2), "rs_1group_mib": rs_1,
                         "rs_groups_mib": rs_g, "compute_mib": d.get("compute", 0.0),
                         "pipe_slots_compute_mib": d.get("pipe", 0.0), "pipelining_adds_mib": round(added, 2),
                         "smi_used_mib": used, "smi_total_mib": total})
        if not any(r["depth"] == depth for r in rows):
            # a log without libllama's buffer lines (llama-server below -lv 5): nvidia-smi's use per card only
            for name, (used, total) in sorted(smi.items(), key=lambda kv: int(kv[0][4:])):
                fullest = max(fullest, (used, name))
                rows.append({"depth": depth, "pipelining": state, "pipe_slots": slots, "state_groups": groups,
                             "device": name, "model_mib": "", "kv_mib": "", "rs_1group_mib": "", "rs_groups_mib": "",
                             "compute_mib": "", "pipe_slots_compute_mib": "", "pipelining_adds_mib": "",
                             "smi_used_mib": used, "smi_total_mib": total})
        status = src / f"status-d{depth}.txt"
        host = {}
        if status.exists():
            for line in status.read_text().splitlines():
                k, v = line.split(":", 1)
                host[k] = int(v.split()[0]) / 1024.0
            rows.append({"depth": depth, "pipelining": state, "pipe_slots": slots, "state_groups": groups,
                         "device": "host",
                         "model_mib": round(host.get("RssFile", 0.0), 1), "kv_mib": 0.0, "rs_1group_mib": 0.0,
                         "rs_groups_mib": 0.0, "compute_mib": round(host.get("RssAnon", 0.0), 1),
                         "pipe_slots_compute_mib": dev.get("CPU", {}).get("pipe", 0.0), "pipelining_adds_mib": 0.0,
                         "smi_used_mib": 0.0, "smi_total_mib": 0.0})
        total = sum(r["pipelining_adds_mib"] or 0.0 for r in rows if r["depth"] == depth)
        top = f"; fullest card {fullest[1]} at {fullest[0]:.0f} MiB (nvidia-smi)" if smi else ""
        print(f"depth {depth}: pipelining {state}, {slots} pipe slots, {groups} state groups; adds {total:.1f} MiB over "
              f"the devices, at most {worst[0]:.1f} MiB on {worst[1]}{top}")
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
