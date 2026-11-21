#!/usr/bin/env python3
# C878: which generation passes of a llama-server run were shorter than a full verify pass, and which graph each one
# was on card 0. Reads the server's log at -lv 5 (the slot's "max possible draft", "generate_draft" and "decode" lines,
# the "print_timing" block that ends each request) and the profile GGML_CUDA_PROFILE_MODE=graph wrote in the same load
# (device,eval,host_ms,...,graph,phase,how,groups,eval_groups,eval_us). A generation pass is the decode after a "max
# possible draft" line: one token plus the drafts of the "generate_draft" line between them (none when the most possible
# is 0). The profile's clock starts later than the log's; the offset is the one that puts the most of card 0's
# evaluations of rarely run graphs within 25 ms of a short pass.
# Usage: python3 -I short_passes.py <server log (.gz or plain)> <profile csv (.gz or plain)> [FULL]
#   FULL: the token count of a full verify pass (default 4: one token and three drafts)
# Prints two tables (TSV): every short pass, and card 0's evaluations of those graphs with the pass each one matches.
import csv, gzip, io, re, sys


def opener(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


TS = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ")


def ts(line):
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi*60 + s + ms/1e3 + us/1e6


def passes(log):
    out, task, pending, drafts, n_tokens, i_req = [], None, False, 0, None, 0
    for line in opener(log):
        m = re.search(r"\| task (\d+) \|", line)
        if m:
            task = int(m.group(1))
        if "get_n_draft_: " in line and "max possible draft:" in line:
            pending, drafts = True, 0
            continue
        m = re.search(r"generate_draft: .*#tokens=(\d+), #draft=(\d+)", line)
        if m and pending:
            n_tokens, drafts = int(m.group(1)), int(m.group(2))
            continue
        m = re.search(r"new n_tokens = (\d+)", line)
        if m:
            n_tokens = int(m.group(1))
            continue
        if " decode: n_batch" in line and pending:
            out.append({"task": task, "req": i_req, "tokens": 1 + drafts, "cells": n_tokens, "t": ts(line)})
            pending = False
            continue
        if "print_timing" in line and "prompt eval time" in line:
            i_req += 1
    return out


def main():
    log, prof = sys.argv[1], sys.argv[2]
    full = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ps = passes(log)
    last = {}
    for p in ps:
        last[p["task"]] = p
    short = [p for p in ps if p["tokens"] < full]
    print("# every generation pass shorter than a full verify pass")
    print("task\ttokens\tcells_before\tlast_of_reply\tlog_s")
    for p in short:
        print(f"{p['task']}\t{p['tokens']}\t{p['cells']}\t{'yes' if last[p['task']] is p else 'no'}\t{p['t']:.3f}")

    rows = [r for r in csv.DictReader(opener(prof)) if r["device"] == "0" and r["phase"] == "0" and r["label"] == "graph"]
    count = {}
    for r in rows:
        count[r["graph"]] = count.get(r["graph"], 0) + 1
    rare = [r for r in rows if count[r["graph"]] <= 4]
    best, best_n = None, -1
    for r in rare:
        for p in short:
            off = p["t"] - float(r["host_ms"])/1e3
            n = sum(1 for q in rare if any(abs(s["t"] - off - float(q["host_ms"])/1e3) < 0.025 for s in short))
            if n > best_n:
                best, best_n = off, n
    print()
    print(f"# card 0's evaluations of graphs written at most 4 times; log clock = profile clock + {best:.3f} s "
          f"({best_n} of {len(rare)} matched within 25 ms)")
    print("eval\thost_ms\tgraph\thow\tlaunch_groups\tgpu_ms\tpass_task\tpass_tokens\tpass_cells_before\tlast_of_reply")
    for r in sorted(rare, key=lambda r: float(r["host_ms"])):
        t = float(r["host_ms"])/1e3 + best
        m = min(short, key=lambda p: abs(p["t"] - t)) if short else None
        ok = m is not None and abs(m["t"] - t) < 0.025
        print(f"{r['eval']}\t{r['host_ms']}\t{r['graph']}\t{r['how']}\t{r['eval_groups']}\t{float(r['eval_us'])/1e3:.2f}\t"
              + (f"{m['task']}\t{m['tokens']}\t{m['cells']}\t{'yes' if last[m['task']] is m else 'no'}" if ok else "-\t-\t-\t-"))


if __name__ == "__main__":
    main()
