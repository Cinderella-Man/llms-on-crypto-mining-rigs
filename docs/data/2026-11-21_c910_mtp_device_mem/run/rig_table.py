# C910's rig check table: reads the collected deep_lane folder of c910_rig.sh and writes, per load, what --mtp-device
# did to memory: the head copy's log line, card 0's memory once the server was up (and every card's), the host's lowest
# MemAvailable and usable memory (MemAvailable minus CmaFree) while loading and over the server's whole life (the RAM
# watchdog's samples), whether the watchdog stopped it, and the replies' token ids against the first load's.
# Usage: python3 -I rig_table.py <collected OUT folder> > rig_table.md
import glob
import hashlib
import json
import os
import re
import sys

d = sys.argv[1]


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


loads = []
text = read(os.path.join(d, "loads.tsv")) or ""
rows = [line.split("\t") for line in text.splitlines()]
if rows:
    head = rows[0]
    for r in rows[1:]:
        loads.append(dict(zip(head, r)))


def server_facts(name):
    log = read(os.path.join(d, f"server-{name}.log"))
    if log is None:
        for p in glob.glob(os.path.join(d, "stopped", f"{name}-*", f"server-{name}.log")):
            log = read(p)
    if log is None:
        return {"copy": "no log", "bufs": "", "errors": ""}
    m = re.search(r"MTP output head copied to (\S+), ([\d.]+) MiB", log)
    copy = f"{m.group(1)}, {m.group(2)} MiB" if m else "none"
    if "no room for the MTP output head copy" in log:
        copy += "; NO ROOM"
    bufs = re.findall(r"(CUDA\d+|CPU\S*) model buffer size = +([\d.]+) MiB", log)
    card0 = [b for n, b in bufs if n == "CUDA0"]
    errs = [line.strip() for line in log.splitlines()
            if re.search(r"out of memory|failed to allocate|cudaMalloc|GGML_ASSERT|error:|abort", line)]
    return {"copy": copy, "bufs": " + ".join(card0) + " MiB" if card0 else "-", "errors": "; ".join(errs[:2])}


def fit_facts(name):
    t = read(os.path.join(d, f"fit_{name}.txt"))
    if t is None:
        return {"card0": "-", "cards": "-", "avail_up": "-"}
    cards = re.findall(r"^\s*(\d+),\s*(\d+),\s*(\d+)\s*$", t, re.M)
    c0 = next(((u, tot) for i, u, tot in cards if i == "0"), None)
    m = re.search(r"MemAvailable:\s+(\d+) kB", t)
    return {
        "card0": f"{c0[0]} of {c0[1]} MiB used, {int(c0[1]) - int(c0[0])} free" if c0 else "-",
        "cards": " ".join(f"{int(tot) - int(u)}" for i, u, tot in cards),
        "avail_up": f"{int(m.group(1)) // 1024} MiB" if m else "-",
    }


def watchdog_facts(name, load_s):
    paths = [os.path.join(d, f"ram_watchdog-{name}.tsv")] + \
        glob.glob(os.path.join(d, "stopped", f"{name}-*", f"ram_watchdog-{name}.tsv"))
    t = next((x for x in (read(p) for p in paths) if x), None)
    if t is None:
        return {"load_min": "-", "all_min": "-"}
    samples = []
    for line in t.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        try:
            samples.append((int(f[0]), int(f[1]), int(f[3])))
        except (ValueError, IndexError):
            continue
    if not samples:
        return {"load_min": "-", "all_min": "-"}
    t0 = samples[0][0]
    during = [s for s in samples if load_s is None or s[0] - t0 <= load_s] or samples

    def fmt(ss):
        a = min(ss, key=lambda s: s[1])
        u = min(ss, key=lambda s: s[2])
        return f"{a[1]} avail (at {a[0] - t0} s), {u[2]} usable (at {u[0] - t0} s)"

    return {"load_min": fmt(during), "all_min": fmt(samples)}


def reply_hashes(name):
    t = read(os.path.join(d, f"{name}.jsonl"))
    if t is None:
        return []
    out = []
    for line in t.splitlines():
        try:
            ids = json.loads(line).get("ids")
        except json.JSONDecodeError:
            continue
        if ids is not None:
            out.append(hashlib.md5(json.dumps(ids).encode()).hexdigest()[:8])
    return out


print("| load | arm | started | load s | stopped | head copy (log) | card 0 model buffers | card 0 once up | free MiB per card "
      "once up | host MemAvailable once up | lowest while loading (MiB) | lowest over the server's life (MiB) | replies "
      "(md5 of ids) | errors |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
first = None
for ld in loads:
    name = ld.get("load", "?")
    try:
        load_s = int(ld.get("load_s") or 0) or None
    except ValueError:
        load_s = None
    s, f, w = server_facts(name), fit_facts(name), watchdog_facts(name, load_s)
    h = reply_hashes(name)
    if first is None and h:
        first = h
    same = "" if first is None or h == first or not h else " (DIFFER from the first load)"
    print(f"| {name} | {ld.get('arm', '')} | {ld.get('started', '')} | {ld.get('load_s', '')} | "
          f"{'yes' if ld.get('timed') == 'stopped' else 'no'} | {s['copy']} | {s['bufs']} | {f['card0']} | {f['cards']} | "
          f"{f['avail_up']} | {w['load_min']} | {w['all_min']} | {' '.join(h) or '-'}{same} | {s['errors'] or '-'} |")
