#!/usr/bin/env python3
"""s31_tables.py: S31's smoke arms on a rig (the -draftmem build's switches on the MTP draft line) as one table.
Run where the smoke folders are, with python3 -I:

  python3 -I s31_tables.py --out rig_smoke.csv ARM=DIR [ARM=DIR ...]

A DIR is smoke_dflash.sh's OUT (docs/data/2026-11-03_flash_next_dflash2/run/): smoke.log, server-smoke.log,
ram_watchdog-smoke.tsv (the RAM watchdog's one-second samples: epoch, MemAvailable, CmaFree, usable, swap used, MiB
swapped out in the window, pressure some/full avg10, pid), host_loaded.txt, vram_loaded.csv and, for each request that
got a reply, reply_long.json (the agent's first-turn request) and reply_1.json (S12's smoke request). One row per arm:
how it ended, the floor and the kill reason, the load time, host memory once loaded (usable, Shmem, swap used), the
lowest usable after the load and the most swap-out in 10 s over the whole run, the highest pressure, cards 4 and 8's
VRAM once loaded, and per request the generated tokens, tok/s, drafted and accepted tokens and the acceptance.
"""
import argparse
import csv
import json
import os
import re


def meminfo(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"(\w+):\s+(\d+) kB", line)
            if m:
                out[m.group(1)] = int(m.group(2)) // 1024
    return out


def watchdog(path):
    rows, floor, kill = [], "", ""
    if not os.path.exists(path):
        return rows, floor, kill
    for line in open(path):
        if line.startswith("#"):
            m = re.search(r"MemAvailable < (-?\d+) MiB", line)
            if m:
                floor = m.group(1)
            if " KILL " in line:
                kill = line.split(": ", 1)[1].strip() if ": " in line else line.strip()
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) >= 8 and f[0].isdigit():
            rows.append([int(f[0])] + [float(x) if x not in ("", "NA") else 0.0 for x in f[1:8]])
    return rows, floor, kill


def reply(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        r = json.load(open(path))
    except ValueError:
        return None
    t, u = r.get("timings", {}), r.get("usage", {})
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {"gen": u.get("completion_tokens", t.get("predicted_n")), "prompt": u.get("prompt_tokens", t.get("prompt_n")),
            "tps": round(t.get("predicted_per_second", 0), 2), "drafted": dn, "accepted": da,
            "acc": round(da / dn, 3) if dn else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    hdr = ["arm", "ended", "floor_mib", "kill", "load_s", "usable_loaded_mib", "shmem_loaded_mib", "swap_used_loaded_mib",
           "usable_lowest_after_load_mib", "swapout_10s_max_mib", "psi_full_max", "card4_loaded_mib", "card8_loaded_mib"]
    for q in ("long", "smoke"):
        hdr += [f"{q}_prompt", f"{q}_gen", f"{q}_tps", f"{q}_drafted", f"{q}_accepted", f"{q}_acceptance"]
    out = []
    for spec in a.arms:
        arm, d = spec.split("=", 1)
        log = open(os.path.join(d, "smoke.log")).read() if os.path.exists(os.path.join(d, "smoke.log")) else ""
        m = re.search(r"up after (\d+) s", log)
        load = int(m.group(1)) if m else ""
        rows, floor, kill = watchdog(os.path.join(d, "ram_watchdog-smoke.tsv"))
        h = meminfo(os.path.join(d, "host_loaded.txt"))
        after = [r for r in rows if rows and load != "" and r[0] - rows[0][0] >= load]
        vram = {}
        vp = os.path.join(d, "vram_loaded.csv")
        if os.path.exists(vp):
            for line in open(vp):
                f = [x.strip() for x in line.split(",")]
                if len(f) >= 2 and f[0].isdigit():
                    vram[int(f[0])] = f[1].split()[0]
        rl, r1 = reply(os.path.join(d, "reply_long.json")), reply(os.path.join(d, "reply_1.json"))
        ended = "answered both" if rl and r1 else ("stopped by the watchdog" if kill else "no reply")
        row = [arm, ended, floor, kill, load,
               h.get("MemAvailable", 0) - h.get("CmaFree", 0) if h else "", h.get("Shmem", "") if h else "",
               h.get("SwapTotal", 0) - h.get("SwapFree", 0) if h else "",
               int(min(r[3] for r in after)) if after else "", int(max((r[5] for r in rows), default=0)),
               max((r[7] for r in rows), default=0.0), vram.get(4, ""), vram.get(8, "")]
        for r in (rl, r1):
            row += [r["prompt"], r["gen"], r["tps"], r["drafted"], r["accepted"], r["acc"]] if r else [""] * 6
        out.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out)
    for r in out:
        print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main()
