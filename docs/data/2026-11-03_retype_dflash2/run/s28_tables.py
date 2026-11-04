#!/usr/bin/env python3
"""s28_tables.py: the tables of Q551's stacked step (the retyped file with the DFlash2 drafter) from the smoke test
and the timed agent-check arm: Q561's s18_tables.py, with smoke tests of any number of requests. Run where the files
are (the agent check's client), with python3 -I:

  python3 -I s28_tables.py --out OUT --smoke 2=DIR --attempts CLIENT_OUT/res/attempts.csv \
      --server-log CLIENT_OUT/rig/server-hcq8df2.log --guard mem_guard_timed.log --ref nodraft --arms dflash2b,hcq8df2

A smoke DIR is smoke_dflash.sh's OUT (reply_<n>.json, host_*.txt, mem_guard.log, server-smoke.log); its label is
--spec-draft-n-max. Writes to OUT:
  smoke.csv            per smoke test and request: generated tokens, decode seconds, tok/s, drafted and accepted
                       tokens, acceptance, verification steps (accepted / (mean length - 1), from the server's log),
                       tokens and milliseconds per step
  smoke_memory.csv     per smoke test: how it ended, MemAvailable minus CmaFree (usable) at its lowest and the most
                       swap-out in 10 s (mem_guard.log), usable, Shmem and swap once loaded and after each request
  draft_acceptance.csv the timed arm's server log: per request, acceptance, accepted, drafted, mean accepted length
  session_memory.csv   the timed arm's guard log, split at the server's "model loaded": usable (median, lowest), swap
                       (mean, highest), most swap-out in 10 s, highest full memory pressure, kills
  common_tasks.csv     the tasks the reference arm and every arm in --arms ran (attempts.csv): per arm, sums of turns,
                       generated tokens, decode, new prompt tokens, prefill, fixed, other and wall seconds; and the wall
                       time each arm would take at the reference's generated tokens (its decode rate), with its own
                       non-decode seconds (low) and with the reference's (high), as Q226 did
"""
import argparse
import csv
import json
import os
import re
import statistics
from datetime import datetime

TIMING = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.\d+ .*task (\d+) \| draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) "
                    r"generated\), mean len =\s*([\d.]+)")


def offset_s(line):  # the server log's m.ss.mmm.uuu prefix, in seconds
    m, s, ms = line.split(" ", 1)[0].split(".")[:3]
    return int(m) * 60 + int(s) + int(ms) / 1000


def meminfo(path):
    v = {}
    for line in open(path):
        k, _, rest = line.partition(":")
        if rest.strip().endswith("kB"):
            v.setdefault(k, int(rest.split()[0]) // 1024)
    return v


def guard_rows(path):  # epoch avail cma usable swap swapout psi-some psi-full pid
    rows, kills = [], []
    for line in open(path):
        if line.startswith("#"):
            if "KILL" in line:
                kills.append(line.strip("# \n"))
            continue
        f = line.split()
        if len(f) == 9 and f[8] != "-":
            rows.append((int(f[0]), int(f[3]), int(f[4]), int(f[5]), float(f[7])))
    return rows, kills


def server_start_epoch(log):
    first = open(log).readline()
    return datetime.fromisoformat(first.split()[1]).timestamp()


def loaded_offset(log):
    for line in open(log):
        if "llama_server: model loaded" in line:
            return offset_s(line)
    return None


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def smoke_tables(out, smokes):
    rows, mem = [], []
    for spec in smokes:
        label, d = spec.split("=", 1)
        log = os.path.join(d, "server-smoke.log")
        lens = [TIMING.match(x) for x in open(log)]
        lens = [m for m in lens if m]
        n = 0
        while os.path.exists(os.path.join(d, f"reply_{n + 1}.json")):
            n += 1
        for i in range(1, n + 1):
            p = os.path.join(d, f"reply_{i}.json")
            if os.path.getsize(p) == 0:
                rows.append([label, i, "", "", "", "", "", "", "", "", ""])
                continue
            t = json.load(open(p))["timings"]
            drafted, accepted = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
            m = lens[i - 1] if i <= len(lens) else None
            mean_len = float(m.group(8)) if m else None
            # without a draft every pass is one token
            steps = round(accepted / (mean_len - 1)) if mean_len and mean_len > 1 else None if drafted else t["predicted_n"]
            rows.append([label, i, t["predicted_n"], round(t["predicted_ms"] / 1000, 1),
                         round(t["predicted_per_second"], 2), drafted, accepted,
                         round(accepted / drafted, 3) if drafted else "", steps,
                         round(t["predicted_n"] / steps, 2) if steps else "",
                         round(t["predicted_ms"] / steps) if steps else ""])
        g, kills = guard_rows(os.path.join(d, "mem_guard.log"))
        watchdog = [x.strip() for x in open(log) if x.startswith("watchdog:")]
        ended = "; ".join(kills + watchdog) or "completed"
        r = [label, ended, min(x[1] for x in g), max(x[3] for x in g), max(x[4] for x in g)]
        for h in ["loaded"] + [f"after_request_{i}" for i in range(1, n + 1)]:
            v = meminfo(os.path.join(d, f"host_{h}.txt"))
            r += [v["MemAvailable"] - v["CmaFree"], v["Shmem"], v["SwapTotal"] - v["SwapFree"]]
        mem.append(r)
    write(os.path.join(out, "smoke.csv"),
          ["n_max", "request", "generated", "decode_s", "gen_tok_s", "drafted", "accepted", "acceptance", "steps",
           "tokens_per_step", "ms_per_step"], rows)
    hdr = ["n_max", "ended", "lowest_usable_mib", "most_swapout_10s_mib", "highest_full_avg10_pct"]
    for h in ["loaded"] + [f"after_request_{i}" for i in range(1, (len(mem[0]) - 5) // 3)]:
        hdr += [f"{h}_usable_mib", f"{h}_shmem_mib", f"{h}_swap_mib"]
    write(os.path.join(out, "smoke_memory.csv"), hdr, mem)


def arm_tables(out, server_log, guard):
    rows = []
    for line in open(server_log):
        m = TIMING.match(line)
        if m:
            rows.append([round(offset_s(line)), m.group(4), m.group(5), m.group(6), m.group(7), m.group(8)])
    write(os.path.join(out, "draft_acceptance.csv"),
          ["seconds_from_server_start", "server_task", "acceptance", "accepted", "generated", "mean_accepted_len"], rows)
    acc = sum(int(r[3]) for r in rows)
    gen = sum(int(r[4]) for r in rows)
    t_loaded = server_start_epoch(server_log) + (loaded_offset(server_log) or 0)
    g, kills = guard_rows(guard)
    mem = []
    for phase, sel in (("load", [x for x in g if x[0] < t_loaded]), ("session", [x for x in g if x[0] >= t_loaded])):
        if not sel:
            continue
        mem.append([phase, len(sel), statistics.median(x[1] for x in sel), min(x[1] for x in sel),
                    round(statistics.mean(x[2] for x in sel)), max(x[2] for x in sel), max(x[3] for x in sel),
                    max(x[4] for x in sel), "; ".join(kills) or "none"])
    write(os.path.join(out, "session_memory.csv"),
          ["phase", "samples", "usable_median_mib", "usable_lowest_mib", "swap_mean_mib", "swap_highest_mib",
           "most_swapout_10s_mib", "highest_full_avg10_pct", "kills"], mem)
    return len(rows), acc, gen


def common_tasks(out, attempts, ref, arms):
    by = {}
    for r in csv.DictReader(open(attempts)):
        by[(r["file"], r["task_id"])] = r  # the last row of an arm and task is the scored attempt
    tasks = sorted({t for a, t in by if a == ref})
    for a in arms:
        tasks = [t for t in tasks if (a, t) in by]
    cols = ["turns", "output_tokens", "decode_s", "prompt_new", "prefill_s", "fixed_s", "other_s", "wall_s"]
    sums = {a: {c: sum(float(by[(a, t)][c]) for t in tasks) for c in cols} for a in [ref] + arms}
    rs = sums[ref]
    rows = []
    for a in [ref] + arms:
        s = sums[a]
        rate = s["output_tokens"] / s["decode_s"]
        dec = rs["output_tokens"] / rate
        low = s["prefill_s"] + s["fixed_s"] + s["other_s"] + dec
        high = rs["prefill_s"] + rs["fixed_s"] + rs["other_s"] + dec
        rows.append([a, len(tasks), int(s["turns"]), int(s["output_tokens"]), round(s["decode_s"]), round(rate, 2),
                     int(s["prompt_new"]), round(s["prefill_s"]), round(s["prompt_new"] / s["prefill_s"], 1),
                     round(s["fixed_s"]), round(s["other_s"]), round(s["wall_s"]), round(low), round(high),
                     round(rs["wall_s"] / high, 3), round(rs["wall_s"] / low, 3)])
    write(os.path.join(out, "common_tasks.csv"),
          ["arm", "tasks", "turns", "output_tokens", "decode_s", "generation_tok_s", "new_prompt_tokens", "prefill_s",
           "prompt_tok_s", "fixed_s", "other_s", "wall_s", "wall_at_ref_tokens_low_s", "wall_at_ref_tokens_high_s",
           "speedup_at_ref_tokens_low", "speedup_at_ref_tokens_high"], rows)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="append", default=[])
    ap.add_argument("--attempts")
    ap.add_argument("--server-log")
    ap.add_argument("--guard")
    ap.add_argument("--ref", default="nodraft")
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        smoke_tables(a.out, a.smoke)
    if a.server_log and a.guard:
        n, acc, gen = arm_tables(a.out, a.server_log, a.guard)
        print(f"timed arm: {n} drafting requests, acceptance {acc / max(gen, 1):.3f} ({acc} of {gen})")
    if a.attempts and a.arms:
        tasks = common_tasks(a.out, a.attempts, a.ref, a.arms.split(","))
        print(f"common tasks: {len(tasks)}")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".csv"):
            print(f"== {f}")
            print(open(os.path.join(a.out, f)).read(), end="")


if __name__ == "__main__":
    main()
