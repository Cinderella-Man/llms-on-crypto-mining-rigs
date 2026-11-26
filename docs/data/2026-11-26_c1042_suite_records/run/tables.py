#!/usr/bin/env python3
"""C1042 (Q1775): the tables of the kept FLASH_ATTN_EXT and fused-case records from run.sh's outputs.

    python3 -I tables.py <WORK/bops> <out folder> <build tag> [<build tag> ...]

Reads <share>-record.log (the recording build's live check), <share>-replay-<build>-<arm>.log and .tsv (each replay),
the .cmd and .sec files (arm, wall seconds), record.txt, manifest-<share>.md5 and list-ops.tsv. The build tags name the
order of the builds (the first one also ran the cpu-check replay). Writes to the out folder:
  steps.tsv         one line per step: share, step, arm, exit, the summary line, OK/FAIL/NOREC/not supported/illegal
                    counts, the record's or replay's own line (the record's folder written as $REC), wall seconds
  verdict.tsv       the pass criteria per share and replay: every case the live check passed passes, none missing
  errors.tsv        per share and arm: cases whose largest normalized error is the same number on every build, and
                    per share and arm against the plain (or mmv) arm on the first build: cases whose error the arm
                    changes, the cases another kernel computed
  cases.tsv.gz      one line per recorded case: share, record id, case, the live verdict while recording, each replay's
                    verdict and largest normalized error
  record.tsv        per share: cases, files, bytes, the manifest's own md5; the disk read of the largest file
  list_ops.tsv      the list step: every operator name test mode runs, the cases the card checks and their record
                    bytes, the cases it does not support, and whether it is a ggml operation or a test-mode-only
                    (fused-case) name
  manifest-<share>.md5.gz   md5 of every file of the record (paths relative to the record's folder), for md5sum -c
"""
import collections
import glob
import gzip
import os
import re
import shutil
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^  ([A-Z0-9_]+\(.*\)): (OK|FAIL|NOREC.*|not supported.*)$")
SHARES = ["flash_attn_ext", "fused", "mul_mat_id", "mul_mat"]
ARMS = ["all", "tile", "plain", "forced", "mmv", "cpucheck"]


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
            rr = re.sub(r"/\S+/(\w+):", r"$REC/\1:", line.strip())
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


def arm_of(src, share, step):
    p = os.path.join(src, f"{share}-{step}.cmd")
    if os.path.exists(p):
        for line in lines(p):
            if line.startswith("arm: "):
                return line[5:]
    return ""


def main():
    src, out, builds = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)

    rc = {}
    for line in open(os.path.join(src, "rc.txt"), encoding="utf-8"):
        k, _, v = line.strip().partition(" rc=")
        rc[k] = v

    def order(tag):
        b, _, a = tag.rpartition("-")
        return (builds.index(b) if b in builds else len(builds), ARMS.index(a) if a in ARMS else len(ARMS), tag)

    tags = {}
    for share in SHARES:
        found = [os.path.basename(f)[len(share) + len("-replay-"):-4]
                 for f in glob.glob(os.path.join(src, f"{share}-replay-*.tsv"))]
        tags[share] = sorted(found, key=order)

    steps = []
    for share in SHARES:
        for step in ["record"] + [f"replay-{t}" for t in tags[share]]:
            log = os.path.join(src, f"{share}-{step}.log")
            if not os.path.exists(log):
                continue
            c, passed, rr, cpu = log_counts(log)
            secf = os.path.join(src, f"{share}-{step}.sec")
            sec = open(secf).read().strip() if os.path.exists(secf) else ""
            steps.append([share, step, arm_of(src, share, step), rc.get(f"{share}-{step}", ""), passed, c["OK"],
                          c["FAIL"], c["NOREC"], c["not supported"], c["illegal"], rr + (f"; {cpu}" if cpu else ""), sec])
    with open(os.path.join(out, "steps.tsv"), "w", encoding="utf-8") as f:
        f.write("share\tstep\tarm\texit\tsummary\tOK\tFAIL\tNOREC\tnot supported\tillegal access\t"
                "record or replay line\twall s\n")
        for s in steps:
            f.write("\t".join(str(x) for x in s) + "\n")

    verdict, errors = [], []
    with gzip.open(os.path.join(out, "cases.tsv.gz"), "wt", encoding="utf-8", compresslevel=9) as g:
        all_tags = []
        for share in SHARES:
            all_tags += [t for t in tags[share] if t not in all_tags]
        g.write("share\trecord id\tcase\tlive (recording)\t" + "\t".join(f"{t}\t{t} max err" for t in all_tags) + "\n")
        for share in SHARES:
            rlog = os.path.join(src, f"{share}-record.log")
            if not os.path.exists(rlog) or not tags[share]:
                continue
            ts = tags[share]
            live = live_verdicts(rlog)
            n_live_ok = sum(v.count("OK") for v in live.values())
            per_tag = {t: replay_rows(os.path.join(src, f"{share}-replay-{t}.tsv")) for t in ts}
            idx = {t: {r[0]: r for r in per_tag[t]} for t in ts}
            base = per_tag[ts[0]]
            seen = collections.Counter()
            for rid, name, _, _ in base:
                k = seen[name]
                seen[name] += 1
                lv = live[name][k] if k < len(live[name]) else ""
                cols = []
                for t in all_tags:
                    r = idx[t].get(rid) if t in idx else None
                    cols += [r[2] if r else "", "" if not r or r[3] is None else f"{r[3]:.3e}"]
                g.write("\t".join([share, rid, name, lv] + cols) + "\n")
            live_ok = collections.Counter({n: v.count("OK") for n, v in live.items()})
            for t in ts:
                rows = per_tag[t]
                ok_names = collections.Counter(r[1] for r in rows if r[2] == "OK")
                lost = sum((live_ok - ok_names).values())
                norec = sum(1 for r in rows if r[2] == "NOREC")
                fail = sum(1 for r in rows if r[2] == "FAIL")
                failing = ";".join(r[1] for r in rows if r[2] == "FAIL")
                verdict.append([share, t, arm_of(src, share, f"replay-{t}"), n_live_ok, sum(ok_names.values()), fail,
                                norec, lost, "passed" if lost == 0 and norec == 0 and fail == 0 else "FAILED",
                                failing or "-"])
            # the same arm on every build: is each case's error the same number?
            arms = sorted({t.rpartition("-")[2] for t in ts if not t.endswith("-cpucheck")}, key=ARMS.index)
            for a in arms:
                bt = [t for t in ts if t.rpartition("-")[2] == a]
                if len(bt) < 2:
                    continue
                same = differ = 0
                for rid, r0 in idx[bt[0]].items():
                    if r0[2] == "NOREC":
                        continue
                    vals = {idx[t][rid][3] for t in bt if rid in idx[t]}
                    if len(vals) == 1:
                        same += 1
                    else:
                        differ += 1
                errors.append([share, f"{a} on every build ({', '.join(t.rpartition('-')[0] for t in bt)})",
                               same, differ])
            # each arm against the reference arm (plain for attention, mmv for the fused cases) on the first build:
            # a case whose error changes was computed by another kernel
            ref = "plain" if "plain" in arms else ("mmv" if "mmv" in arms else None)
            first = ts[0].rpartition("-")[0]
            if ref:
                rt = f"{first}-{ref}"
                for a in arms:
                    if a == ref or f"{first}-{a}" not in idx:
                        continue
                    t = f"{first}-{a}"
                    changed = kept = 0
                    for rid, r0 in idx[rt].items():
                        if r0[2] == "NOREC" or rid not in idx[t]:
                            continue
                        if idx[t][rid][3] == r0[3]:
                            kept += 1
                        else:
                            changed += 1
                    errors.append([share, f"{a} against {ref} on {first}", kept, changed])
            # the cpu-check replay against the same arm's plain replay on the first build
            ct = [t for t in ts if t.endswith("-cpucheck")]
            if ct:
                ra = arm_of(src, share, f"replay-{ct[0]}")
                st = f"{first}-{ra}"
                if st in idx:
                    same = differ = 0
                    for rid, r0 in idx[st].items():
                        if r0[2] == "NOREC" or rid not in idx[ct[0]]:
                            continue
                        if idx[ct[0]][rid][3] == r0[3]:
                            same += 1
                        else:
                            differ += 1
                    errors.append([share, f"{ra} with --cpu-check against without, on {first}", same, differ])
    with open(os.path.join(out, "verdict.tsv"), "w", encoding="utf-8") as f:
        f.write("share\treplay\tarm\tlive OK while recording\treplay OK\treplay FAIL\tNOREC\t"
                "live OK cases not OK in the replay\tverdict\tfailing cases\n")
        for v in verdict:
            f.write("\t".join(str(x) for x in v) + "\n")
    with open(os.path.join(out, "errors.tsv"), "w", encoding="utf-8") as f:
        f.write("share\tcomparison\tcases with the same largest error\tcases whose largest error differs\n")
        for e in errors:
            f.write("\t".join(str(x) for x in e) + "\n")

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
    lo = os.path.join(src, "list-ops.tsv")
    if os.path.exists(lo):
        with open(os.path.join(out, "list_ops.tsv"), "w", encoding="utf-8") as f:
            f.write("operator name\tcases the card checks\trecord bytes\tcases not supported\tkind\n")
            for line in lines(lo):
                f.write(line + "\n")
    for share in SHARES:
        m = os.path.join(src, f"manifest-{share}.md5")
        if os.path.exists(m):
            with open(m, "rb") as fi, gzip.open(os.path.join(out, f"manifest-{share}.md5.gz"), "wb",
                                                compresslevel=9) as fo:
                shutil.copyfileobj(fi, fo)
    for v in verdict:
        print("\t".join(str(x) for x in v))
    for e in errors:
        print("\t".join(str(x) for x in e))


if __name__ == "__main__":
    main()
