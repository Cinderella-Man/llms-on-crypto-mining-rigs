#!/usr/bin/env python3
"""C954 (Q1687 point 8): the tables of run.sh's output. For each arm (one llama-server process that loads, sleeps,
loads again and sleeps again) it reads the server log, phases.tsv, rss.tsv and the two cycles' replies, and writes:
  fit.tsv       the memory fit's projection per card, per load (the fit runs before each load)
  buffers.tsv   the loader's lines per load: each "model buffer size" line and the "MTP output head copied" line
  cards.tsv     every card's memory use (nvidia-smi, MiB) at each phase
  rss.tsv       the server process's RssAnon per load: at the load's start, its largest value during the load (and
                how long after the start), and once the server answered; RssFile's largest value for reference
  replies.tsv   every request's token count, md5 of the token ids and of the text, drafted and accepted tokens
  compare.tsv   per request, each arm's md5 of the token ids, and whether it equals the first arm's
Usage: python3 -I tables.py <run.sh's WORK/load folder> <output folder>
"""
import csv
import os
import re
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
ORDER = ["new-unset", "new-dev0", "base-dev0", "base-unset"]
arms = [a for a in ORDER if os.path.isdir(os.path.join(SRC, a))] + \
    sorted(a for a in os.listdir(SRC) if a not in ORDER and os.path.isdir(os.path.join(SRC, a)))

SLEEP = "server is entering sleeping state"
FIT_HEAD = "projected memory use with initial parameters"
FIT_ROW = re.compile(r"-\s+(\S+) \(([^)]*)\)\s*:\s+(-?\d+) total,\s+(-?\d+) used,\s+(-?\d+) free vs\. target of\s+(-?\d+)")
BUF = re.compile(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB")
COPY = re.compile(r"MTP output head copied to (\S+), ([\d.]+) MiB")
NOROOM = re.compile(r"no room for the MTP output head copy|output head could not be copied")


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


fit_rows, buf_rows, card_rows, rss_rows, rep_rows = [], [], [], [], []
replies = {}  # arm -> {(cycle, round, prompt): md5 of ids}

for a in arms:
    d = os.path.join(SRC, a)
    log = read(os.path.join(d, "server.log")).splitlines()
    cycle, fit_block = 1, 0
    for line in log:
        if SLEEP in line:
            cycle += 1
            continue
        if FIT_HEAD in line:
            fit_block += 1
            continue
        m = FIT_ROW.search(line)
        if m:
            fit_rows.append([a, fit_block, m.group(1), m.group(3), m.group(4), m.group(5), m.group(6)])
            continue
        m = BUF.search(line)
        if m:
            buf_rows.append([a, cycle, m.group(1) + " model buffer", m.group(2)])
            continue
        m = COPY.search(line)
        if m:
            buf_rows.append([a, cycle, "MTP output head copied to " + m.group(1), m.group(2)])
            continue
        if NOROOM.search(line):
            buf_rows.append([a, cycle, line.strip()[-120:], ""])

    phases = []
    for line in read(os.path.join(d, "phases.tsv")).splitlines():
        f = line.split("\t")
        if len(f) < 3:
            continue
        phases.append((int(f[0]), f[1]))
        cards = dict(x.split(":", 1) for x in f[2].split())
        card_rows.append([a, f[1]] + [cards[k].split("/")[0] for k in sorted(cards)])
    t = dict((p, ts) for ts, p in phases)

    samples = []
    for line in read(os.path.join(d, "rss.tsv")).splitlines()[1:]:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        samples.append((int(f[0]), int(f[1]), int(f[2])))

    def window(c, t0, t1):
        s = [x for x in samples if t0 <= x[0] <= t1]
        if not s:
            return [a, c, "-", "-", "-", "-", "-"]
        pk = max(s, key=lambda x: x[1])
        return [a, c, s[0][1] // 1024, pk[1] // 1024, f"{(pk[0] - t0) / 1000:.2f}", s[-1][1] // 1024,
                max(x[2] for x in s) // 1024]

    if "started" in t and "up1" in t:
        rss_rows.append(window(1, t["started"], t["up1"]))
    if "asleep1" in t and "up2" in t:
        rss_rows.append(window(2, t["asleep1"], t["up2"]))

    replies[a] = {}
    for c in ("c1", "c2"):
        for line in read(os.path.join(d, c, "replies.tsv")).splitlines():
            f = line.split("\t")
            if len(f) < 8:
                continue
            rep_rows.append([a, c] + f[:8])
            replies[a][(c, f[0], f[1])] = f[3]

write("fit.tsv", ["arm", "load", "card", "total_mib", "projected_used_mib", "projected_free_mib", "target_free_mib"],
      fit_rows)
write("buffers.tsv", ["arm", "load", "line", "mib"], buf_rows)
ncards = max((len(r) - 2 for r in card_rows), default=0)
write("cards.tsv", ["arm", "phase"] + [f"card{i}_used_mib" for i in range(ncards)], card_rows)
write("rss.tsv", ["arm", "load", "rss_anon_start_mib", "rss_anon_peak_mib", "peak_after_s", "rss_anon_up_mib",
                  "rss_file_peak_mib"], rss_rows)
write("replies.tsv", ["arm", "cycle", "round", "prompt", "tokens", "md5_ids", "md5_text", "drafted", "accepted",
                      "stop"], rep_rows)

keys = sorted({k for a in arms for k in replies[a]})
cmp_rows = []
ref = arms[0] if arms else None
for k in keys:
    vals = [replies[a].get(k, "-") for a in arms]
    first = replies[ref].get(("c1", k[1], k[2])) if ref else None
    cmp_rows.append(list(k) + vals + ["yes" if all(v == first for v in vals) else "NO"])
write("compare.tsv", ["cycle", "round", "prompt"] + [f"md5_ids_{a}" for a in arms] +
      [f"all_equal_{ref}_c1"], cmp_rows)

print(f"arms {arms}: fit rows {len(fit_rows)}, buffer lines {len(buf_rows)}, phases {len(card_rows)}, "
      f"rss windows {len(rss_rows)}, replies {len(rep_rows)}, requests compared {len(cmp_rows)}, "
      f"all equal {sum(r[-1] == 'yes' for r in cmp_rows)}")
