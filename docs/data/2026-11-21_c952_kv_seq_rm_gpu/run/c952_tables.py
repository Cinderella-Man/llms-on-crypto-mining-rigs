#!/usr/bin/env python3
"""C952 (Q1687 point 6): tables from run.sh's output folder. Nothing timed is read or written: the server's checkpoint
lines carry host times, which are cut off here; only positions and counts are kept.

Usage: python3 -I c952_tables.py <WORK/out> <output folder>
Writes:
  replies.tsv   one line per arm, round and request: prompt tokens read, cached tokens used, generated tokens, the md5
                of the token ids (12 hex digits), drafted and accepted drafts, the stop reason
  events.tsv    per arm, the server's cache events in order, positions only: checkpoints created (pos_min, pos_max,
                n_tokens), restored (pos_min, pos_max, n_tokens, n_past), erased, full re-reads (shared tokens), the
                text-reuse lines (a draft's restore-checkpoint lines are left out: the cancelled reply's share of them
                would depend on when the server sees the hang-up; the MTP drafts here log none)
  compare.tsv   per model, setting, round and request: base against branch on the token ids, prompt tokens read,
                cached tokens and drafts; for cancel only the CANCEL_AT chunks read
  verdicts.tsv  per model and setting: replies, positions and event lists the same on base and branch?
  mask.tsv      per arm, the device-mask check's lines (setting mask): masks identical to the host's, of them built
                from the resident copy (updated from the changed-cell list) and after a full upload, the steps whose
                changed-cell list was not empty, the changed cells summed and the largest list; mismatches
  settings.tsv  per model and build, each setting's token ids against def's (information: another head rule may put
                cells in another order, and attention then sums them in another order)
"""
import csv
import os
import re
import sys

OUT, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
REQS = ["fresh", "turn", "edit", "cancel", "after", "back"]

EV = [
    ("created", re.compile(r"created context checkpoint (\d+) of (\d+)( inside the call)? \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("restored", re.compile(r"restored context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), n_past = (-?\d+)")),
    ("erased-old", re.compile(r"erasing old context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("erased-invalid", re.compile(r"erased invalidated context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("full-reread", re.compile(r"forcing full prompt re-processing due to lack of cache data \(shared (\d+) tokens, no checkpoint below (-?\d+)")),
    ("text-reuse", re.compile(r"prompt ids differ from the cache at (\d+), same text: (\d+) more cached tokens kept \((\d+) -> (\d+)")),
    ("draft-restore", re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens \(restore checkpoint\)")),
    ("kq-mask-mismatch", re.compile(r"mask values differ from the host mask|: device (keep|drop), host ")),
    ("assert", re.compile(r"GGML_ASSERT|GGML_ABORT")),
]


def events(log):
    out = []
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                for name, rx in EV:
                    m = rx.search(line)
                    if m:
                        g = [x for x in m.groups() if x is not None and not x.startswith(" inside")]
                        if name == "created" and m.group(3):
                            name = "created-incall"
                        if name in ("kq-mask-mismatch", "assert"):
                            g = []
                        out.append(name + ":" + ",".join(g))
                        break
    except FileNotFoundError:
        pass
    return out


MASK = re.compile(r"check_kq_cells: (\S+) copy (\d+): (\d+) tokens x (\d+) cells x \d+ streams \(swa type \d+, (resident|all cells uploaded), (\d+) changed cells, (\d+) old cells\): identical to the host mask")


def mask_summary(log):
    n = res = full = nz = tot = mx = 0
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = MASK.search(line)
                if not m:
                    continue
                n += 1
                res += m.group(5) == "resident"
                full += m.group(5) != "resident"
                c = int(m.group(6))
                nz += c > 0
                tot += c
                mx = max(mx, c)
    except FileNotFoundError:
        pass
    return [n, res, full, nz, tot, mx]


def arms():
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if os.path.isdir(p) and d.count("-") == 2:
            yield d, p


rows, ev, mk = {}, {}, {}
with open(os.path.join(DST, "replies.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "setting", "build", "round", "request", "prompt_n", "cache_n", "n_gen", "tokens_md5",
                "drafted", "accepted", "stop"])
    for d, p in arms():
        m, s, b = d.split("-")
        e = events(os.path.join(p, "server.log"))
        ev[(m, s, b)] = [x for x in e if not x.startswith("draft-restore:")]
        mk[(m, s, b)] = mask_summary(os.path.join(p, "server.log"))
        try:
            with open(os.path.join(p, "session.tsv"), encoding="utf-8") as g:
                for line in g:
                    x = line.rstrip("\n").split("\t")
                    r = dict(round=x[0], request=x[1], prompt_n=x[2], cache_n=x[3], n_gen=x[4], md5=x[5][:12],
                             drafted=x[7], accepted=x[8], stop=x[9])
                    rows[(m, s, b, r["round"], r["request"])] = r
                    w.writerow([m, s, b, r["round"], r["request"], r["prompt_n"], r["cache_n"], r["n_gen"], r["md5"],
                                r["drafted"], r["accepted"], r["stop"]])
        except FileNotFoundError:
            pass

with open(os.path.join(DST, "events.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "setting", "build", "n", "event", "values"])
    for (m, s, b), e in sorted(ev.items()):
        for i, x in enumerate(e):
            k, v = x.split(":", 1)
            w.writerow([m, s, b, i + 1, k, v])

with open(os.path.join(DST, "mask.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "setting", "build", "masks identical to the host's", "from the resident copy",
                "after a full upload", "steps with changed cells listed", "changed cells summed", "largest list",
                "mismatches"])
    for (m, s_, b), x in sorted(mk.items()):
        if x[0] or s_ == "mask":
            w.writerow([m, s_, b] + x + [sum(1 for e in ev[(m, s_, b)] if e.startswith("kq-mask-mismatch"))])

models = sorted({k[0] for k in rows})
settings = [s for s in ["def", "tail0", "mask"] if any(k[1] == s for k in rows)]
rounds = sorted({k[3] for k in rows})
verd = []
with open(os.path.join(DST, "compare.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "setting", "round", "request", "tokens", "prompt_n", "cache_n", "drafts", "base_md5",
                "new_md5", "base_prompt_n/cache_n", "new_prompt_n/cache_n", "base_drafted/accepted",
                "new_drafted/accepted"])
    for m in models:
        for s in settings:
            if not any(k[0] == m and k[1] == s for k in rows):
                continue
            n = same = 0
            for r in rounds:
                for q in REQS:
                    a, c = rows.get((m, s, "base", r, q)), rows.get((m, s, "new", r, q))
                    if not a or not c:
                        w.writerow([m, s, r, q, "missing"] + [""] * 9)
                        n += 1
                        continue
                    t = "same" if a["md5"] and a["md5"] == c["md5"] else "DIFFERENT"
                    pn = "same" if a["prompt_n"] == c["prompt_n"] else "DIFFERENT"
                    cn = "same" if a["cache_n"] == c["cache_n"] else "DIFFERENT"
                    dr = "same" if (a["drafted"], a["accepted"]) == (c["drafted"], c["accepted"]) else "DIFFERENT"
                    if q == "cancel":  # the cancel's server-side length depends on when it sees the hang-up
                        pn = pn if pn == "same" else "DIFFERENT"
                        cn, dr = "n/a", "n/a"
                    ok = t == "same" and pn == "same" and cn in ("same", "n/a") and dr in ("same", "n/a")
                    n += 1
                    same += ok
                    w.writerow([m, s, r, q, t, pn, cn, dr, a["md5"], c["md5"], f"{a['prompt_n']}/{a['cache_n']}",
                                f"{c['prompt_n']}/{c['cache_n']}", f"{a['drafted']}/{a['accepted']}",
                                f"{c['drafted']}/{c['accepted']}"])
            eb, en = ev.get((m, s, "base"), []), ev.get((m, s, "new"), [])
            same_ev = eb == en
            same_mk = mk.get((m, s, "base")) == mk.get((m, s, "new"))
            first = next((i for i, (x, y) in enumerate(zip(eb, en)) if x != y), min(len(eb), len(en)))
            nr = lambda e: sum(1 for x in e if x.startswith("restored:"))
            nf = lambda e: sum(1 for x in e if x.startswith("full-reread:"))
            na = lambda e: sum(1 for x in e if x.startswith(("assert", "kq-mask-mismatch")))
            verd.append([m, s, f"{same} of {n}", "same" if same_ev else f"DIFFERENT from event {first + 1}",
                         f"{len(eb)}/{len(en)}", f"{nr(eb)}/{nr(en)}", f"{nf(eb)}/{nf(en)}", f"{na(eb)}/{na(en)}",
                         "same" if same_mk else "DIFFERENT",
                         "PASS" if same == n and same_ev and same_mk and na(eb) + na(en) == 0 else "CHECK"])

with open(os.path.join(DST, "verdicts.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "setting", "requests the same (tokens, prompt_n, cache_n, drafts)", "event lists",
                "events base/new", "restores base/new", "full re-reads base/new", "asserts or mask mismatches base/new",
                "verdict"])
    w.writerows(verd)

with open(os.path.join(DST, "settings.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["model", "build", "setting", "requests with def's token ids", "first request that differs"])
    for m in models:
        for b in ["base", "new"]:
            for s in settings:
                if s == "def":
                    continue
                n = same = 0
                first = ""
                for r in rounds:
                    for q in REQS:
                        a, c = rows.get((m, "def", b, r, q)), rows.get((m, s, b, r, q))
                        if not a or not c:
                            continue
                        n += 1
                        if a["md5"] == c["md5"]:
                            same += 1
                        elif not first:
                            first = f"{r} {q}"
                if n:
                    w.writerow([m, b, s, f"{same} of {n}", first or "none"])

for name in ["verdicts.tsv", "settings.tsv"]:
    print(open(os.path.join(DST, name)).read())
