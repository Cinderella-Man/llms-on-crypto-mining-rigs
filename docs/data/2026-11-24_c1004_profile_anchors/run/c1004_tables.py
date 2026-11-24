#!/usr/bin/env python3
"""C1004 (Q1737): the tables of run.sh's session step. Correctness only: no speed is reported. The microsecond figures
below are the anchors' own error (how far a placed time can sit from the host's clock), not how fast anything ran.

Usage: python3 -I c1004_tables.py <session folder> <tables folder> <arm> [<arm> ...]

Each <session folder>/<arm>/ holds session.tsv (C950's session: round, request, kind, prompt characters, generated
tokens, prompt tokens, cached tokens, md5 of the token ids, reply start), server.log and, for a profiled arm, hp.csv
(GGML_CUDA_PROFILE_HOST: device, eval, host_enter_ms, host_exit_ms, gpu_us, queued and, with
GGML_CUDA_PROFILE_HOST_ANCHORS, gpu_start_ms and gpu_end_ms: the evaluation's start and end on the card, on the file's
clock).

Writes:
  tokens.tsv   per request: kind, tokens and the md5 of the token ids per arm, and whether every arm gave the same
  profile.tsv  per profiled arm and card: the file's columns, lines, evaluation ids (first, last, missing, repeated), the
               count line written at the backend's free (evaluations, written, entries with earlier ones still running,
               the most at once, entries the host waited at) and the anchors' line (anchors, the longest an evaluation
               was placed after the newest, checks, the largest difference of the card's clock from the host's between two
               anchors and the time between them, the mean drift in ppm)
  anchors.tsv  per arm with the two columns and card: lines placed and lines before the card's first anchor; and the
               checks, each with the lines below zero (beyond the file's rounding), the lines beyond BOUND_US and the
               worst case in microseconds:
                 span    gpu_end - gpu_start against gpu_us (the same two events: equal up to the file's rounding)
                 order   each evaluation starts on the card no earlier than the card's previous one ended (one stream;
                         the two may be placed from different anchors)
                 enter   each evaluation starts on the card no earlier than the host entered it (the host records the
                         start event after its entry reading, so a start placed before it is the anchor's error)
  waits.tsv    per arm that waits and has the two columns, and card: the entries at which the host waited for the
               card's previous evaluation (queued >= 1) and, for each, the host's entry reading after that wait minus the
               previous evaluation's placed end: never below zero beyond the anchors' jitter if the anchors are right, and
               small (the host's wake-up and its writes); count, below zero, beyond BOUND_US, smallest, median and largest
               in microseconds
"""
import collections
import csv
import os
import re
import statistics
import sys

SES, OUT, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
TOL_MS = 0.0015  # the file's %.3f rounding of two numbers
FLUSH_RE = re.compile(r"ggml_cuda_profile_host_flush: CUDA(\d+): (\d+) evaluations, (\d+) written; (\d+) entered with "
                      r"earlier ones still running \(at most (\d+)\), the host waited at (\d+)")
ANCH_RE = re.compile(r"ggml_cuda_profile_host_flush: CUDA(\d+): (\d+) anchors, events placed at most ([0-9.]+) ms from "
                     r"theirs; (\d+) checked against the one before: card's clock minus host's (-?[0-9.]+) us at most "
                     r"\(over ([0-9.]+) ms\), (-?[0-9.]+) ppm over all")
# the anchors' bound, fixed before the run: an anchor is the host's reading when the card reaches an event on an idle
# stream, which the first pass (an anchor on the backend's own stream) found within 18 us of the one before over
# 36 us; a placed time that contradicts a host reading by more than this is a failure, less is the anchor's jitter
BOUND_US = 100.0


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


arm = {}
for a in ARMS:
    d = os.path.join(SES, a)
    ses = read_tsv(os.path.join(d, "session.tsv"))
    log_path = os.path.join(d, "server.log")
    log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
    reqs = [{"round": r[0], "name": r[1], "kind": r[2], "n": r[4], "md5": r[7]} for r in ses]
    flush = {int(m.group(1)): m.groups()[1:] for m in FLUSH_RE.finditer(log)}
    anch = {int(m.group(1)): m.groups()[1:] for m in ANCH_RE.finditer(log)}
    prof, cols = None, None
    hp = os.path.join(d, "hp.csv")
    if os.path.exists(hp):
        with open(hp, newline="") as f:
            rd = csv.reader(f)
            cols = next(rd)
            # a server killed mid-write leaves a partial last line: only whole lines count
            prof = [dict(zip(cols, r)) for r in rd if len(r) == len(cols)]
    arm[a] = {"reqs": reqs, "flush": flush, "anch": anch, "prof": prof, "cols": cols}

lines = []
keys = [(r["round"], r["name"]) for r in arm[ARMS[0]]["reqs"]]

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

profiled = [a for a in ARMS if arm[a]["prof"] is not None]


def by_card(a):
    cards = collections.defaultdict(list)
    for r in arm[a]["prof"]:
        cards[int(r["device"])].append(r)
    for c in cards:
        cards[c].sort(key=lambda r: int(r["eval"]))
    return dict(sorted(cards.items()))


with open(os.path.join(OUT, "profile.tsv"), "w") as f:
    f.write("arm\tcard\tcolumns\tlines\tfirst_eval\tlast_eval\tmissing\trepeated\tlog_evaluations\tlog_written\t"
            "log_entered_with_earlier\tlog_most_at_once\tlog_host_waited\tlog_anchors\tlog_farthest_from_anchor_ms\t"
            "log_checks\tlog_max_clock_diff_us\tlog_over_ms\tlog_drift_ppm\n")
    for a in profiled:
        for c, rows in by_card(a).items():
            ids = [int(r["eval"]) for r in rows]
            cnt = collections.Counter(ids)
            missing = (max(ids) - min(ids) + 1) - len(cnt)
            repeated = sum(v - 1 for v in cnt.values())
            fl = arm[a]["flush"].get(c, ("",) * 5)
            an = arm[a]["anch"].get(c, ("",) * 6)
            f.write("\t".join([a, str(c), " ".join(arm[a]["cols"]), str(len(rows)), str(min(ids)), str(max(ids)),
                               str(missing), str(repeated), *fl, *an]) + "\n")
            lines.append(f"profile {a} CUDA{c}: {len(arm[a]['cols'])} columns, {len(rows)} lines, ids {min(ids)}-{max(ids)}"
                         f" missing {missing} repeated {repeated}; count line {' '.join(fl) or 'none'};"
                         f" anchors line {' '.join(an) or 'none'}")

anchored = [a for a in profiled if "gpu_start_ms" in arm[a]["cols"]]
waits_rows = []


class Check:
    """one check: values in microseconds that should not be below zero"""

    def __init__(self):
        self.n, self.below, self.beyond, self.worst = 0, 0, 0, None

    def add(self, us):
        self.n += 1
        self.below += us < -TOL_MS * 1000
        self.beyond += us < -BOUND_US
        self.worst = us if self.worst is None else min(self.worst, us)

    def cells(self):
        return [self.n, self.below, self.beyond, f"{self.worst:.1f}" if self.worst is not None else ""]

    def text(self):
        return (f"{self.n} checked, {self.below} below 0, {self.beyond} beyond {BOUND_US:.0f} us, worst "
                + (f"{self.worst:.1f} us" if self.worst is not None else "none"))


with open(os.path.join(OUT, "anchors.tsv"), "w") as f:
    f.write("arm\tcard\tlines\tplaced\tbefore_first_anchor\tunplaced_after\tspan_worst_us\t"
            "order_n\torder_below0\torder_beyond\torder_worst_us\tenter_n\tenter_below0\tenter_beyond\tenter_worst_us\n")
    for a in anchored:
        for c, rows in by_card(a).items():
            placed = [r for r in rows if r["gpu_start_ms"] != ""]
            first = next((i for i, r in enumerate(rows) if r["gpu_start_ms"] != ""), len(rows))
            unplaced_late = sum(1 for r in rows[first:] if r["gpu_start_ms"] == "")
            span_worst = 0.0
            order, enter = Check(), Check()
            prev = None
            for r in rows:
                if r["gpu_start_ms"] == "":
                    prev = None
                    continue
                s, e = float(r["gpu_start_ms"]), float(r["gpu_end_ms"])
                span_worst = max(span_worst, abs((e - s) - float(r["gpu_us"]) / 1000.0) * 1000)
                enter.add((s - float(r["host_enter_ms"])) * 1000)
                if prev is not None and int(r["eval"]) == int(prev["eval"]) + 1:
                    order.add((s - float(prev["gpu_end_ms"])) * 1000)
                    if int(r["queued"]) >= 1 and "nowait" not in a:
                        # the waiting profile: the host waited for prev's end, then read its clock for this entry
                        waits_rows.append((a, c, (float(r["host_enter_ms"]) - float(prev["gpu_end_ms"])) * 1000))
                prev = r
            f.write("\t".join(str(x) for x in [a, c, len(rows), len(placed), first, unplaced_late, f"{span_worst:.1f}",
                                                *order.cells(), *enter.cells()]) + "\n")
            lines.append(f"anchors {a} CUDA{c}: {len(placed)} of {len(rows)} lines placed ({first} before the first anchor,"
                         f" {unplaced_late} unplaced after it); span worst {span_worst:.1f} us; order: {order.text()};"
                         f" enter: {enter.text()}")

with open(os.path.join(OUT, "waits.tsv"), "w") as f:
    f.write("arm\tcard\twaits\tbelow0\tbeyond\tsmallest_us\tmedian_us\tlargest_us\n")
    groups = collections.defaultdict(list)
    for a, c, v in waits_rows:
        groups[(a, c)].append(v)
    for (a, c), vs in sorted(groups.items()):
        below = sum(v < -TOL_MS * 1000 for v in vs)
        beyond = sum(v < -BOUND_US for v in vs)
        f.write(f"{a}\t{c}\t{len(vs)}\t{below}\t{beyond}\t{min(vs):.1f}\t{statistics.median(vs):.1f}\t{max(vs):.1f}\n")
        lines.append(f"waits {a} CUDA{c}: {len(vs)} waits, host entry after the wait minus the placed end: {below} below 0,"
                     f" {beyond} beyond {BOUND_US:.0f} us; smallest {min(vs):.1f} us, median {statistics.median(vs):.1f} us,"
                     f" largest {max(vs):.1f} us")

print("\n".join(lines))
