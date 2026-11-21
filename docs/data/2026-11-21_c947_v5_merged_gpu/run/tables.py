#!/usr/bin/env python3
"""C947 (Q1687): tables from a copy of run.sh's WORK folder (standard library only; run with python3 -I).

Usage: python3 -I tables.py <work copy> <out folder> [<reference mbox folder> [<reference full test log, .gz allowed>]]

Writes to <out folder>:
  mbox.tsv          each mailbox-test arm and setting: result, rounds, hops by mailbox; with a reference folder (pull
                    request 4's run of the same arms), its result and hops beside them
  mbox-arms.tsv     per arm: exit code, the test's own check count, and lines that would mean lost bytes or a hang
  load-replies.tsv  every greedy reply: tokens, md5 of the text and of the token ids, drafted and accepted tokens
  reply-agree.tsv   per model and request: the merged build (new) against the base (ref)
  load-memory.tsv   MiB per card after the load and after the two requests, both builds
  bops.tsv          test-backend-ops per card's share: exit code, the passed line, OK, FAIL, illegal-access lines
  bops-coverage.tsv with a reference full test log: cases per operator there and in the four shares together
"""
import csv
import gzip
import hashlib
import json
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SETTING = re.compile(r"^(\S+)\s+(ok|FAIL)\s*:\s*(\d+) rounds over (\d+) cards, (\d+) hops by mailbox")
CHECKS = re.compile(r"test-hop-mailbox: (\d+) checks, (\d+) failed")
ARMS = ["off", "g16-b8192", "g16-b4096", "g1-b64", "g64-b4096-r480"]
MODELS = ["flashnext", "flintq8", "ornith"]
BUILDS = ["new", "ref"]
PROMPTS = ["code", "prose"]


def read(path):
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            return ANSI.sub("", f.read())
    except OSError:
        return None


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def settings(text):
    out = {}
    for line in (text or "").splitlines():
        m = SETTING.match(line)
        if m:
            out[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(5)))
    return out


def mbox(work, out, ref):
    rows, arms = [], []
    rc = {}
    for line in (read(os.path.join(work, "mbox", "rc.txt")) or "").splitlines():
        parts = line.split(" rc=")
        if len(parts) == 2:
            rc[parts[0]] = parts[1]
    for arm in ARMS:
        text = read(os.path.join(work, "mbox", arm + ".log"))
        mine = settings(text)
        theirs = settings(read(os.path.join(ref, arm + ".log"))) if ref else {}
        for s, (res, rounds, hops) in mine.items():
            r = theirs.get(s)
            rows.append([arm, s, res, rounds, hops] +
                        ([r[0], r[2], "yes" if r == (res, rounds, hops) else "NO"] if ref else []))
        m = CHECKS.search(text or "")
        low = (text or "").lower()
        arms.append([arm, rc.get(arm, "?"), m.group(1) if m else "?", m.group(2) if m else "?",
                     low.count("differ"), low.count("timed out"), low.count("cuda error"), len(mine)])
    write_tsv(os.path.join(out, "mbox.tsv"),
              ["arm", "setting", "result", "rounds", "hops_by_mailbox"] +
              (["pr4_result", "pr4_hops_by_mailbox", "same_as_pr4"] if ref else []), rows)
    write_tsv(os.path.join(out, "mbox-arms.tsv"),
              ["arm", "exit", "checks", "checks_failed", "lines_differ", "lines_timed_out", "lines_cuda_error",
               "settings"], arms)


def reply(path):
    try:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return None
    content = j.get("content", "")
    toks = j.get("tokens") or []
    t = j.get("timings") or {}
    return {
        "tokens": j.get("tokens_predicted", len(toks)),
        "chars": len(content),
        "md5": hashlib.md5(content.encode("utf-8")).hexdigest(),
        "tok_md5": hashlib.md5(",".join(str(x) for x in toks).encode()).hexdigest() if toks else "",
        "n_tok_ids": len(toks),
        "drafted": t.get("draft_n", ""),
        "accepted": t.get("draft_n_accepted", ""),
        "stop": j.get("stop_type", ""),
        "start": content[:60].replace("\n", "\\n").replace("\t", " "),
    }


def mem(path):
    vals = []
    for line in (read(path) or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            vals.append(parts[1])
    return " ".join(vals)


def load(work, out):
    rows, agree, memrows = [], [], []
    got = {}
    for m in MODELS:
        for b in BUILDS:
            d = os.path.join(work, "load", f"{m}-{b}")
            up = (read(os.path.join(d, "load.txt")) or "").strip()
            memrows.append([m, b, up, mem(os.path.join(d, "mem.csv")), mem(os.path.join(d, "mem-after.csv"))])
            for p in PROMPTS:
                r = reply(os.path.join(d, f"reply-{p}.json"))
                got[(m, b, p)] = r
                if r:
                    rows.append([m, b, p, r["tokens"], r["n_tok_ids"], r["chars"], r["md5"], r["tok_md5"],
                                 r["drafted"], r["accepted"], r["stop"], r["start"]])
                else:
                    rows.append([m, b, p, "", "", "", "no reply", "", "", "", "", ""])
        for p in PROMPTS:
            n, r = got[(m, "new", p)], got[(m, "ref", p)]
            if n and r:
                agree.append([m, p, "yes" if n["md5"] == r["md5"] else "NO",
                              "yes" if n["tok_md5"] == r["tok_md5"] and n["tok_md5"] else "NO",
                              "yes" if (n["drafted"], n["accepted"]) == (r["drafted"], r["accepted"]) else "NO",
                              n["tokens"], n["chars"]])
            else:
                agree.append([m, p, "missing", "missing", "missing", "", ""])
    write_tsv(os.path.join(out, "load-replies.tsv"),
              ["model", "build", "prompt", "tokens", "token_ids", "chars", "text_md5", "token_ids_md5", "drafted",
               "accepted", "stop", "start"], rows)
    write_tsv(os.path.join(out, "reply-agree.tsv"),
              ["model", "prompt", "same_text", "same_token_ids", "same_drafted_accepted", "tokens", "chars"], agree)
    write_tsv(os.path.join(out, "load-memory.tsv"),
              ["model", "build", "server", "MiB_per_card_20s_after_up", "MiB_per_card_after_requests"], memrows)


# a passing case prints "  OP(...): OK"; a failing one "[OP] ERR = a > b   OP(...): FAIL"
CASE = re.compile(r"^(?:  |\[[A-Za-z0-9_]+\] ERR = \S+ > \S+\s+)([A-Za-z0-9_]+)\(.*: (OK|FAIL)$")


def case_counts(text):
    counts = {}
    for line in (text or "").splitlines():
        m = CASE.match(line.rstrip())
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def bops(work, out, ref_log):
    rows = []
    mine = {}
    rc = {}
    for line in (read(os.path.join(work, "bops", "rc.txt")) or "").splitlines():
        parts = line.split(" rc=")
        if len(parts) == 2:
            rc[parts[0]] = parts[1]
    for t in ["mul_mat", "mul_mat_id", "rest", "fusion"]:
        text = read(os.path.join(work, "bops", t + ".log")) or ""
        for k, v in case_counts(text).items():
            mine[k] = mine.get(k, 0) + v
        m = re.search(r"(\d+)/(\d+) tests passed", text)
        lines = text.splitlines()
        rows.append([t, rc.get(t, "?"), m.group(1) if m else "", m.group(2) if m else "",
                     sum(1 for x in lines if x.rstrip().endswith(": OK")),
                     sum(1 for x in lines if ": FAIL" in x and "Backend" not in x),
                     sum(1 for x in lines if "illegal memory access" in x)])
    write_tsv(os.path.join(out, "bops.tsv"), ["share", "exit", "passed", "tests", "OK_lines", "FAIL_cases",
                                              "illegal_access_lines"], rows)
    if ref_log:
        theirs = case_counts(read(ref_log))
        cov = [[k, theirs.get(k, 0), mine.get(k, 0), "yes" if theirs.get(k, 0) == mine.get(k, 0) else "NO"]
               for k in sorted(set(theirs) | set(mine))]
        cov.append(["total", sum(theirs.values()), sum(mine.values()),
                    "yes" if sum(theirs.values()) == sum(mine.values()) else "NO"])
        write_tsv(os.path.join(out, "bops-coverage.tsv"), ["operator", "reference_cases", "c947_cases", "same"], cov)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    work, out = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    ref_log = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    if os.path.isdir(os.path.join(work, "mbox")):
        mbox(work, out, ref)
    if os.path.isdir(os.path.join(work, "load")):
        load(work, out)
    if os.path.isdir(os.path.join(work, "bops")):
        bops(work, out, ref_log)


if __name__ == "__main__":
    main()
