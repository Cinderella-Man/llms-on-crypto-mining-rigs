#!/usr/bin/env python3
"""C1023 (Q1756): the tables of the kept card-suite record from run.sh's outputs.

    python3 -I tables.py <WORK/bops> <out folder> [build tag ...]

Reads <share>-record.log (the recording build's live check), <share>-replay-<tag>.log and .tsv (each replay), the .sec
files (wall seconds), record.txt and manifest-<share>.md5. Writes to the out folder:
  steps.tsv         one line per step: share, step, exit, the summary line, OK/FAIL/NOREC/not supported/illegal counts,
                    the record's or replay's own line (the record's folder written as $REC), wall seconds
  cases.tsv.gz      one line per recorded case: share, record id, case, the live verdict while recording, each replay's
                    verdict and largest normalized error
  verdict.tsv       the pass criteria: every case the live check passed passes in each replay, no case missing
  record.tsv        per share: cases, files, bytes, the manifest's own md5
  manifest-<share>.md5.gz   md5 of every file of the record (paths relative to the record's folder), for md5sum -c
The build tags default to every <share>-replay-<tag>.tsv found, in name order.
"""
import collections
import glob
import gzip
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^  ((?:MUL_MAT|MUL_MAT_ID)\(.*\)): (OK|FAIL|NOREC.*|not supported.*)$")
SHARES = ["mul_mat_id", "mul_mat"]


def lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield ANSI.sub("", line.rstrip("\n"))


def log_counts(path):
    c = collections.Counter()
    passed, rr, cpu = "", "", ""
    for line in lines(path):
        if line.endswith(": OK"):
            c["OK"] += 1
        if ": FAIL" in line:
            c["FAIL"] += 1
        if ": NOREC" in line:
            c["NOREC"] += 1
        if "not supported" in line:
            c["not supported"] += 1
        if "illegal memory access" in line:
            c["illegal"] += 1
        if "tests passed" in line and not passed:
            passed = line.strip()
        if line.startswith("  record: ") or line.startswith("  replay: "):
            # the record's folder is the machine's own: written as $REC
            rr = re.sub(r"/\S+/(mul_mat(?:_id)?):", r"$REC/\1:", line.strip())
        if line.startswith("  replay cpu-check: "):
            cpu = line.strip()
    return c, passed, rr, cpu


def live_verdicts(path):
    """Per case name, the record step's verdicts in order of appearance (a name can repeat)."""
    v = collections.defaultdict(list)
    for line in lines(path):
        m = LINE.match(line)
        if m and m.group(2) in ("OK", "FAIL"):
            v[m.group(1)].append(m.group(2))
    return v


def replay_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            errs = [float(x) for x in p[3].split(",") if x] if len(p) > 3 and p[3] else []
            rows.append((p[0], p[1], p[2], max(errs) if errs else None))
    return rows


def main():
    src, out = sys.argv[1], sys.argv[2]
    tags = sys.argv[3:]
    if not tags:
        found = set()
        for f in glob.glob(os.path.join(src, "*-replay-*.tsv")):
            found.add(re.sub(r"^.*?-replay-", "", os.path.basename(f))[:-4])
        tags = sorted(found)
    os.makedirs(out, exist_ok=True)

    rc = {}
    for line in open(os.path.join(src, "rc.txt"), encoding="utf-8"):
        k, _, v = line.strip().partition(" rc=")
        rc[k] = v

    steps = []
    for share in SHARES:
        for step in ["record"] + [f"replay-{t}" for t in tags]:
            log = os.path.join(src, f"{share}-{step}.log")
            if not os.path.exists(log):
                continue
            c, passed, rr, cpu = log_counts(log)
            sec = open(os.path.join(src, f"{share}-{step}.sec")).read().strip() if os.path.exists(
                os.path.join(src, f"{share}-{step}.sec")) else ""
            steps.append([share, step, rc.get(f"{share}-{step}", ""), passed, c["OK"], c["FAIL"], c["NOREC"],
                          c["not supported"], c["illegal"], rr + (f"; {cpu}" if cpu else ""), sec])
    with open(os.path.join(out, "steps.tsv"), "w", encoding="utf-8") as f:
        f.write("share\tstep\texit\tsummary\tOK\tFAIL\tNOREC\tnot supported\tillegal access\trecord or replay line\twall s\n")
        for s in steps:
            f.write("\t".join(str(x) for x in s) + "\n")

    verdict = []
    with gzip.open(os.path.join(out, "cases.tsv.gz"), "wt", encoding="utf-8", compresslevel=9) as g:
        g.write("share\trecord id\tcase\tlive (recording)\t" +
                "\t".join(f"{t}\t{t} max err" for t in tags) + "\n")
        for share in SHARES:
            rlog = os.path.join(src, f"{share}-record.log")
            if not os.path.exists(rlog):
                continue
            live = live_verdicts(rlog)
            n_live_ok = sum(v.count("OK") for v in live.values())
            per_tag = {}
            for t in tags:
                p = os.path.join(src, f"{share}-replay-{t}.tsv")
                per_tag[t] = replay_rows(p) if os.path.exists(p) else []
            base = per_tag[tags[0]] if tags else []
            seen = collections.Counter()
            idx = {t: {r[0]: r for r in per_tag[t]} for t in tags}
            for rid, name, _, _ in base:
                k = seen[name]
                seen[name] += 1
                lv = live[name][k] if k < len(live[name]) else ""
                cols = []
                for t in tags:
                    r = idx[t].get(rid)
                    cols += [r[2] if r else "", "" if not r or r[3] is None else f"{r[3]:.3e}"]
                g.write("\t".join([share, rid, name, lv] + cols) + "\n")
            for t in tags:
                rows = per_tag[t]
                ok_names = collections.Counter(r[1] for r in rows if r[2] == "OK")
                live_ok = collections.Counter({n: v.count("OK") for n, v in live.items()})
                lost = sum((live_ok - ok_names).values())
                norec = sum(1 for r in rows if r[2] == "NOREC")
                fail = sum(1 for r in rows if r[2] == "FAIL")
                verdict.append([share, t, n_live_ok, sum(ok_names.values()), fail, norec, lost,
                                "passed" if lost == 0 and norec == 0 and fail == 0 else "FAILED"])
    with open(os.path.join(out, "verdict.tsv"), "w", encoding="utf-8") as f:
        f.write("share\treplay on\tlive OK while recording\treplay OK\treplay FAIL\tNOREC\tlive OK cases not OK in the replay\tverdict\n")
        for v in verdict:
            f.write("\t".join(str(x) for x in v) + "\n")

    rec = os.path.join(src, "record.txt")
    if os.path.exists(rec):
        with open(os.path.join(out, "record.tsv"), "w", encoding="utf-8") as f:
            f.write("share\tcases (.meta files)\tfiles\tbytes\tmanifest md5\n")
            for line in lines(rec):
                p = line.split("\t")
                if len(p) >= 5 and p[0] in SHARES:
                    f.write("\t".join([p[0]] + [x.split(" ", 1)[1] if " " in x else x for x in p[1:4]] +
                                      [p[4].split()[-1]]) + "\n")
                elif line.startswith("largest file"):
                    f.write(f"# {line}\n")
    for share in SHARES:
        m = os.path.join(src, f"manifest-{share}.md5")
        if os.path.exists(m):
            with open(m, "rb") as fi, gzip.open(os.path.join(out, f"manifest-{share}.md5.gz"), "wb", compresslevel=9) as fo:
                fo.write(fi.read())
    for v in verdict:
        print("\t".join(str(x) for x in v))


if __name__ == "__main__":
    main()
