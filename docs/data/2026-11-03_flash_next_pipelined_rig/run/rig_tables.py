#!/usr/bin/env python3
"""R29a / Q550 on the rig: tables from the agent check's rig logs.

Reads the llama-server logs (rig/server-<arm>.log, bench/agent_check/rig_server.sh) and mem_guard.sh's logs
(mem_guard-<arm>.log) and writes:
  requests.csv     one row per request: arm, task, prompt tokens processed and ms, generated tokens and ms, ms per
                   generated token, pipelined guesses, right, wrong, tokens sampled outside the pipeline and how many
                   of those with the tool-call grammar active
  host_memory.csv  per arm, while llama-server ran: MemAvailable minus CmaFree (mean, lowest), swap in use (highest),
                   the most swapped out in a guard window, memory pressure (PSI some avg10 mean, full avg10 highest)
and prints, per arm, the decode speed over all requests and, for pipelined arms, a least-squares fit of each request's
decode seconds to its generated tokens, right guesses and wrong guesses.

Usage: python3 rig_tables.py <rig log dir> <guard log dir> <out dir> <arm> [<arm> ...]
"""
import csv
import re
import sys
from pathlib import Path

RE_PROMPT = re.compile(r"task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_EVAL = re.compile(r"task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_PIPE = re.compile(r"task (\d+) \|\s+pipelined =\s+(\d+) guesses, (\d+) right, (\d+) wrong")
RE_SAMP = re.compile(r"task (\d+) \|\s+sampling =\s+(\d+) tokens, \d+ drawn on the device, (\d+) while the grammar")


def requests(path, arm):
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        for rx, keys in ((RE_PROMPT, ("prompt_ms", "prompt_tokens")), (RE_EVAL, ("gen_ms", "gen_tokens")),
                         (RE_PIPE, ("guesses", "right", "wrong")), (RE_SAMP, ("sampled", "sampled_grammar"))):
            m = rx.search(line)
            if m:
                r = rows.setdefault(m[1], {"arm": arm, "task": int(m[1])})
                for k, v in zip(keys, m.groups()[1:]):
                    r[k] = float(v) if k.endswith("_ms") else int(v)
                break
    out = []
    for r in rows.values():
        if "gen_tokens" not in r:
            continue
        for k in ("guesses", "right", "wrong", "sampled", "sampled_grammar", "prompt_ms", "prompt_tokens"):
            r.setdefault(k, 0)
        r["ms_per_token"] = round(r["gen_ms"] / r["gen_tokens"], 1) if r["gen_tokens"] else ""
        out.append(r)
    return sorted(out, key=lambda r: r["task"])


def solve(x, y):  # least squares through the normal equations (Gauss-Jordan; a few columns)
    k = len(x[0])
    m = [[sum(a[i] * a[j] for a in x) for j in range(k)] + [sum(a[i] * b for a, b in zip(x, y))] for i in range(k)]
    for i in range(k):
        p = m[i][i]
        m[i] = [v / p for v in m[i]]
        for r in range(k):
            if r != i:
                f = m[r][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    return [row[k] for row in m]


def guard(path, arm):
    s = [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]
    s = [f for f in s if len(f) == 9 and f[8] != "-"]
    if not s:
        return None
    real = [int(f[3]) for f in s]
    return {"arm": arm, "samples_s": len(s), "avail_minus_cma_mean_mib": round(sum(real) / len(real)),
            "avail_minus_cma_lowest_mib": min(real), "mem_available_lowest_mib": min(int(f[1]) for f in s),
            "swap_highest_mib": max(int(f[4]) for f in s), "swapout_window_highest_mib": max(int(f[5]) for f in s),
            "psi_some_avg10_mean_pct": round(sum(float(f[6]) for f in s) / len(s), 2),
            "psi_full_avg10_highest_pct": max(float(f[7]) for f in s)}


def main():
    logs, guards, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rows, mem = [], []
    for arm in sys.argv[4:]:
        rs = requests(logs / f"server-{arm}.log", arm)
        rows += rs
        n = sum(r["gen_tokens"] for r in rs)
        t = sum(r["gen_ms"] for r in rs) / 1000
        g = sum(r["sampled_grammar"] for r in rs)
        print(f"{arm}: {len(rs)} requests, {n} tokens generated in {t:.0f} s, {n / t:.2f} tok/s; guesses "
              f"{sum(r['guesses'] for r in rs)}, right {sum(r['right'] for r in rs)}, wrong "
              f"{sum(r['wrong'] for r in rs)}; sampled outside the pipeline {sum(r['sampled'] for r in rs)} "
              f"({g} with the tool grammar active)")
        if len(rs) >= 5 and sum(r["right"] for r in rs) and sum(r["wrong"] for r in rs):
            x = [[r["gen_tokens"], r["right"], r["wrong"]] for r in rs]
            y = [r["gen_ms"] / 1000 for r in rs]
            c = solve(x, y)
            pred = [sum(a * b for a, b in zip(row, c)) for row in x]
            my = sum(y) / len(y)
            r2 = 1 - sum((a - b) ** 2 for a, b in zip(y, pred)) / sum((a - my) ** 2 for a in y)
            print(f"  fit: decode s = {c[0] * 1000:.0f} ms x tokens {c[1] * 1000:+.0f} ms x right "
                  f"{c[2] * 1000:+.0f} ms x wrong (R2 {r2:.3f})")
        gp = guards / f"mem_guard-{arm}.log"
        if gp.exists() and (gm := guard(gp, arm)):
            mem.append(gm)
    cols = ["arm", "task", "prompt_tokens", "prompt_ms", "gen_tokens", "gen_ms", "ms_per_token", "guesses", "right",
            "wrong", "sampled", "sampled_grammar"]
    with open(out / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if mem:
        with open(out / "host_memory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mem[0].keys()))
            w.writeheader()
            w.writerows(mem)
        for m in mem:
            print(m)


if __name__ == "__main__":
    main()
