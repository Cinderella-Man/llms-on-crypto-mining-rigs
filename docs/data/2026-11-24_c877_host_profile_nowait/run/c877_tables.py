#!/usr/bin/env python3
"""C877 (Q1583): the tables of run.sh's session step. Counts only; no time is reported.

Usage: python3 -I c877_tables.py <session folder> <tables folder> <arm> [<arm> ...]

Each <session folder>/<arm>/ holds session.tsv (C950's session: round, request, kind, prompt characters, generated
tokens, prompt tokens, cached tokens, md5 of the token ids, reply start), marks.tsv (session_marks.py: round, index,
the steady clock's ms when each /completion request was sent and when its reply had arrived), server.log and, for a
profiled arm, hp.csv (GGML_CUDA_PROFILE_HOST: device, eval, host_enter_ms, host_exit_ms, gpu_us and, from C877 on,
queued). The profile's log line gives the steady clock's reading at the file's 0 ms, so each line is given to the
request whose marks hold its host_enter_ms.

Writes:
  tokens.tsv    per request: kind, tokens and the md5 of the token ids per arm, and whether every arm gave the same
  pipe.tsv      per request and arm: the server's pipelined line (guesses, right, wrong), and whether the arms agree
  profile.tsv   per profiled arm and card: lines, evaluation ids (first, last, missing, repeated), the count line the
                profile writes when the backend is freed (evaluations, written, entries with earlier ones still
                running, the most at once, entries the host waited at)
  evals.tsv     per request and card: the evaluations the profile wrote for it in each profiled arm, and whether the
                arms agree
  queued.tsv    per profiled arm, card and kind of request (the pipe's plain-text requests, the tool calls under a
                schema, the load before the first request): entries, entries with an earlier evaluation of the card
                still running, the most at once, and the entries by that number (0, 1, 2, 3, 4, 5 or more)
"""
import collections
import csv
import os
import re
import sys

SES, OUT, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)

T0_RE = re.compile(r"its 0 ms is ([0-9.]+) ms on the steady clock")
PIPE_RE = re.compile(r"pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong")
FLUSH_RE = re.compile(r"ggml_cuda_profile_host_flush: CUDA(\d+): (\d+) evaluations, (\d+) written; (\d+) entered with "
                      r"earlier ones still running \(at most (\d+)\), the host waited at (\d+)")


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def kind_class(kind):
    return "pipe" if kind.startswith("plain") else "tool"


arm = {}
for a in ARMS:
    d = os.path.join(SES, a)
    ses = read_tsv(os.path.join(d, "session.tsv"))
    marks = read_tsv(os.path.join(d, "marks.tsv"))
    log = open(os.path.join(d, "server.log"), errors="replace").read() if os.path.exists(os.path.join(d, "server.log")) else ""
    # the requests in order, each with its marks (the session writes one row and one mark per /completion request)
    reqs = []
    by_round = collections.defaultdict(list)
    for m in marks:
        by_round[m[0]].append((float(m[2]), float(m[3])))
    seen = collections.Counter()
    for r in ses:
        rnd, name, kind = r[0], r[1], r[2]
        i = seen[rnd]
        seen[rnd] += 1
        t = by_round[rnd][i] if i < len(by_round[rnd]) else (None, None)
        reqs.append({"round": rnd, "name": name, "kind": kind, "n": r[4], "md5": r[7], "start": t[0], "end": t[1]})
    # the server's timing blocks, one per request, in order; a block's pipelined line, if any, is the request's
    blocks = re.split(r"prompt eval time =", log)[1:]
    pipes = []
    for b in blocks:
        m = PIPE_RE.search(b.split("prompt eval time")[0])
        pipes.append(m.groups() if m else None)
    t0 = T0_RE.search(log)
    flush = {int(m.group(1)): tuple(int(x) for x in m.groups()[1:]) for m in FLUSH_RE.finditer(log)}
    prof = None
    hp = os.path.join(d, "hp.csv")
    if os.path.exists(hp):
        with open(hp, newline="") as f:
            rd = csv.DictReader(f)
            # a server killed mid-write leaves a partial last line: only whole lines count
            prof = [r for r in rd if all(r.get(k) not in (None, "") for k in rd.fieldnames)]
    arm[a] = {"reqs": reqs, "pipes": pipes, "t0": float(t0.group(1)) if t0 else None, "flush": flush, "prof": prof,
              "n_blocks": len(blocks)}


def where(a, enter_ms):
    """the request (index) whose marks hold the line's entry, 'load' before the first, 'between' otherwise"""
    x = arm[a]
    t = x["t0"] + enter_ms
    for i, r in enumerate(x["reqs"]):
        if r["start"] is not None and r["start"] <= t <= r["end"]:
            return i
    first = x["reqs"][0]["start"] if x["reqs"] else None
    return "load" if first is not None and t < first else "between"


keys = [(r["round"], r["name"]) for r in arm[ARMS[0]]["reqs"]]
lines = []

# tokens.tsv
with open(os.path.join(OUT, "tokens.tsv"), "w") as f:
    f.write("round\trequest\tkind\t" + "\t".join(f"{a}_tokens\t{a}_md5" for a in ARMS) + "\tsame\n")
    n_same = 0
    for i, k in enumerate(keys):
        row = [k[0], k[1], arm[ARMS[0]]["reqs"][i]["kind"]]
        md5s = []
        for a in ARMS:
            rs = arm[a]["reqs"]
            r = rs[i] if i < len(rs) and (rs[i]["round"], rs[i]["name"]) == k else None
            row += [r["n"], r["md5"]] if r else ["", ""]
            md5s.append(r["md5"] if r else None)
        same = len(set(md5s)) == 1 and md5s[0] is not None
        n_same += same
        f.write("\t".join(row + ["yes" if same else "no"]) + "\n")
lines.append(f"tokens: {n_same} of {len(keys)} requests give one token list in every arm ({' '.join(ARMS)})")

# pipe.tsv
with open(os.path.join(OUT, "pipe.tsv"), "w") as f:
    f.write("round\trequest\tkind\t" + "\t".join(f"{a}_guesses\t{a}_right\t{a}_wrong" for a in ARMS) + "\tsame\n")
    n_pipe, n_agree = 0, 0
    for i, k in enumerate(keys):
        row = [k[0], k[1], arm[ARMS[0]]["reqs"][i]["kind"]]
        vals = []
        for a in ARMS:
            p = arm[a]["pipes"][i] if i < len(arm[a]["pipes"]) else None
            row += list(p) if p else ["", "", ""]
            vals.append(p)
        if any(vals):
            n_pipe += 1
            n_agree += len(set(vals)) == 1
        f.write("\t".join(row + ["yes" if len(set(vals)) == 1 else "no"]) + "\n")
lines.append(f"pipe: {n_pipe} requests with a pipelined line; the arms' counts agree on {n_agree}; timing blocks per arm: "
             + ", ".join(f"{a} {arm[a]['n_blocks']} for {len(arm[a]['reqs'])} requests" for a in ARMS))

profiled = [a for a in ARMS if arm[a]["prof"] is not None]

# profile.tsv
with open(os.path.join(OUT, "profile.tsv"), "w") as f:
    f.write("arm\tdevice\tlines\tfirst_eval\tlast_eval\tmissing_ids\trepeated_ids\tqueued_column\t"
            "count_line_evals\tcount_line_written\tcount_line_queued_entries\tcount_line_max_queued\tcount_line_waited\n")
    for a in profiled:
        by_dev = collections.defaultdict(list)
        for r in arm[a]["prof"]:
            by_dev[int(r["device"])].append(int(r["eval"]))
        has_q = bool(arm[a]["prof"]) and "queued" in arm[a]["prof"][0]
        for dev in sorted(set(by_dev) | set(arm[a]["flush"])):
            ev = by_dev.get(dev, [])
            c = collections.Counter(ev)
            lo, hi = (min(ev), max(ev)) if ev else (None, None)
            missing = (hi - lo + 1 - len(c)) if ev else 0
            repeated = sum(v - 1 for v in c.values())
            fl = arm[a]["flush"].get(dev)
            f.write("\t".join(str(x) for x in [a, dev, len(ev), lo, hi, missing, repeated, "yes" if has_q else "no"]
                              + (list(fl) if fl else ["", "", "", "", ""])) + "\n")
            lines.append(f"profile {a} CUDA{dev}: {len(ev)} lines, ids {lo}..{hi}, {missing} missing, {repeated} repeated; "
                         + (f"count line: {fl[0]} evaluations, {fl[1]} written, {fl[2]} entered with earlier ones running "
                            f"(at most {fl[3]}), waited at {fl[4]}" if fl else "no count line"))

# evals.tsv and queued.tsv
devs = sorted({int(r["device"]) for a in profiled for r in arm[a]["prof"]})
per = {}  # (arm, request index or 'load'/'between', device) -> list of queued values (None without the column)
for a in profiled:
    if arm[a]["t0"] is None:
        lines.append(f"{a}: no 0 ms line in the server log: lines not placed")
        continue
    for r in arm[a]["prof"]:
        w = where(a, float(r["host_enter_ms"]))
        q = int(r["queued"]) if "queued" in r and r["queued"] not in (None, "") else None
        per.setdefault((a, w, int(r["device"])), []).append(q)

with open(os.path.join(OUT, "evals.tsv"), "w") as f:
    f.write("round\trequest\tkind\tdevice\t" + "\t".join(profiled) + "\tsame\n")
    n_rows, n_same = 0, 0
    for i, k in enumerate(["load"] + list(range(len(keys))) + ["between"]):
        for dev in devs:
            vals = [len(per.get((a, k, dev), [])) for a in profiled]
            if isinstance(k, int):
                name = keys[k]
                kind = arm[ARMS[0]]["reqs"][k]["kind"]
            else:
                name, kind = ("", k), k
            same = len(set(vals)) == 1
            n_rows += 1
            n_same += same
            f.write("\t".join([name[0], name[1], kind, str(dev)] + [str(v) for v in vals] + ["yes" if same else "no"]) + "\n")
lines.append(f"evaluations per request and card: the profiled arms ({' '.join(profiled)}) agree on {n_same} of {n_rows} rows")

with open(os.path.join(OUT, "queued.tsv"), "w") as f:
    f.write("arm\tdevice\trequests\tentries\tentries_with_earlier_running\tmost_at_once\tq0\tq1\tq2\tq3\tq4\tq5_or_more\n")
    for a in profiled:
        for dev in devs:
            for cls in ("pipe", "tool", "load"):
                qs = []
                for (aa, w, dd), v in per.items():
                    if aa != a or dd != dev:
                        continue
                    c = w if isinstance(w, str) else kind_class(arm[a]["reqs"][w]["kind"])
                    if c == cls:
                        qs += v
                if not qs or qs[0] is None:
                    if qs:
                        f.write(f"{a}\t{dev}\t{cls}\t{len(qs)}\t\t\t\t\t\t\t\t\n")
                    continue
                h = collections.Counter(min(q, 5) for q in qs)
                f.write("\t".join(str(x) for x in [a, dev, cls, len(qs), sum(1 for q in qs if q > 0), max(qs)]
                                  + [h.get(j, 0) for j in range(6)]) + "\n")
                if cls == "pipe":
                    lines.append(f"queued {a} CUDA{dev} on the pipe's requests: {len(qs)} entries, "
                                 f"{sum(1 for q in qs if q > 0)} with an earlier evaluation still running, at most {max(qs)}; "
                                 f"by number 0..5+: {' '.join(str(h.get(j, 0)) for j in range(6))}")

print("\n".join(lines))
