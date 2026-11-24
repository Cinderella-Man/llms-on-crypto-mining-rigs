#!/usr/bin/env python3
"""C1018's tables from run.sh's work folder (copied back without the model files).

Usage: python3 -I tables.py <work folder> <data folder> [<more work folders>]
More work folders (a later call of run.sh with other ROUNDS and its own WORK, such as a re-timing on a quiet machine)
add their rows to rounds.tsv and builds.tsv only; the first column names the work folder.
Writes into the data folder:
  rounds.tsv    run.sh's rounds: builds (tags), arms, layouts, checks, exit status, seconds, verdict counts
  builds.tsv    per round and build, the seconds from the build's first line to the next one (the models and every arm)
  verdicts.tsv  every verdict but the profile lines: round, tree, arm, layout, arch, check, verdict, detail, plain hashes
  hashes.tsv    reach round: per architecture and kind of model, the host check's plain logits hash under each setting,
                and whether a setting changed it against the one it adds to
  kernels.tsv   prof round: per arm and architecture, every matrix-product kernel the host check's runs launched, with
                its launch count (types by name)
  reach.tsv     prof round: per arm, the matrix-product kernels it runs that the arm it adds to does not, and the ones
                it drops (across the five architectures)
  types.tsv     each copy's tensors by type, as llama-quantize wrote them
  models.tsv    the md5 of every model file per round and build
"""
import csv
import re
import sys
from pathlib import Path

GGML = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1", 10: "q2_K",
        11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K", 15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
        19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m", 30: "bf16", 41: "q1_0",
        42: "q2_0", 500: "n2p", 501: "n3p", 502: "n2p_128", 503: "n3p_128"}
# the k-column kernel's own type index (kcol::ktype in mmvq-pascal-kcol-impl.cuh)
KCOL = ["q8_0", "q2_0", "q3_K", "q4_K", "q5_K", "q6_K", "iq4_xs", "q4_0", "q5_0", "iq4_nl", "iq2_xxs", "iq2_xs",
        "iq2_s", "iq3_xxs", "iq3_s", "iq1_m"]
MATRIX = re.compile(r"mul_mat|gemm|gemv|cutlass")
PATH = re.compile(r"/(srv|mnt|home|opt|tmp|root)/\S*")
SETTINGS = ["stock", "p106", "fn", "p106-c4", "fn-c4"]
# each setting against the one it adds to
BASE = {"p106": "stock", "fn": "p106", "p106-c4": "p106", "fn-c4": "p106-c4"}


def tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def clean(s):
    return PATH.sub("<path>", s)


def kname(n):
    """the kernel name with its type parameters named"""
    n = re.sub(r"\(ggml_type\)(\d+)", lambda m: GGML.get(int(m.group(1)), "type" + m.group(1)), n)
    m = re.match(r"kcol::mul_mat_kcol<\(int\)(\d+)", n)
    if m and int(m.group(1)) < len(KCOL):
        n = n.replace(m.group(0), "kcol::mul_mat_kcol<" + KCOL[int(m.group(1))], 1)
    return n


def short(n):
    """the kernel with its type: the name and the first template parameter"""
    return re.sub(r"<([^,>]*)[,>].*", r"<\1>", n)


def split_arm(arm):
    kind, setting = arm.split("-", 1)
    return kind, setting


def main():
    work, dest = Path(sys.argv[1]), Path(sys.argv[2])
    works = [work] + [Path(a) for a in sys.argv[3:]]
    allrounds, builds = [], []
    for w in works:
        rws = [r for r in tsv(w / "rounds.tsv")[1:] if r]
        allrounds += [[w.name] + [clean(c) for c in r] for r in rws]
        # per build, from the round's log: "HH:MM:SS <tag>: <bin>" starts it; the next such line or "done:" ends it
        for rn in [r[0] for r in rws]:
            lines = (w / f"{rn}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            marks = []
            for ln in lines:
                m = re.match(r"(\d\d):(\d\d):(\d\d) (\S+): (/\S+|done)", ln)
                m2 = re.match(r"(\d\d):(\d\d):(\d\d) done:", ln)
                if m2:
                    marks.append(("done", int(m2[1]) * 3600 + int(m2[2]) * 60 + int(m2[3])))
                elif m and m[5] != "done":
                    marks.append((m[4], int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])))
            for (tag, t0), (_, t1) in zip(marks, marks[1:]):
                builds.append([w.name, rn, tag, t1 - t0])
    write(dest / "rounds.tsv", ["work", "round", "builds", "arms", "layouts", "checks", "exit", "seconds", "verdicts"],
          allrounds)
    write(dest / "builds.tsv", ["work", "round", "tree", "seconds"], builds)
    names = [r[0] for r in tsv(work / "rounds.tsv")[1:] if r]

    verdicts, host = [], {}
    for rn in names:
        f = work / rn / "summary.tsv"
        if not f.exists():
            continue
        for r in tsv(f)[1:]:
            if len(r) < 9 or r[4] == "profile":
                continue
            verdicts.append([rn] + [clean(c) for c in r])
            if rn == "reach" and r[4] == "host":
                host[(r[3], r[1])] = r[8]
    write(dest / "verdicts.tsv", ["round", "tree", "arm", "layout", "arch", "check", "verdict", "detail",
                                  "plain tokens hash", "plain logits hash"], verdicts)

    rows = []
    archs = sorted({a for a, _ in host})
    kinds = []
    for _, arm in host:
        k = arm.split("-", 1)[0]
        if k not in kinds:
            kinds.append(k)
    for a in archs:
        for k in kinds:
            h = {s: host.get((a, f"{k}-{s}"), "") for s in SETTINGS}
            if not any(h.values()):
                continue
            cmp = []
            for s, b in BASE.items():
                cmp.append("-" if not h[s] or not h[b] else ("same" if h[s] == h[b] else "other"))
            rows.append([a, k] + [h[s] or "-" for s in SETTINGS] + cmp)
    write(dest / "hashes.tsv", ["arch", "models"] + [f"plain logits: {s}" for s in SETTINGS] +
          [f"{s} against {b}" for s, b in BASE.items()], rows)

    krows, per_arm = [], {}
    pdir = work / "prof" / "tip"
    if pdir.exists():
        for d in sorted(pdir.glob("*-one")):
            arm = d.name[: -len("-one")]
            seen = per_arm.setdefault(arm, set())
            for f in sorted(d.glob("prof-*.tsv")):
                a = f.name[len("prof-"): -len(".tsv")]
                for r in tsv(f):
                    if len(r) < 2 or not MATRIX.search(r[1]):
                        continue
                    n = kname(r[1])
                    krows.append([arm, a, n, r[0]])
                    seen.add(short(n))
    write(dest / "kernels.tsv", ["arm", "arch", "kernel", "launches"], krows)

    rrows = []
    for arm in sorted(per_arm):
        kind, setting = split_arm(arm)
        b = BASE.get(setting)
        ref = per_arm.get(f"{kind}-{b}") if b else None
        mine = per_arm[arm]
        new = sorted(mine - ref) if ref is not None else []
        gone = sorted(ref - mine) if ref is not None else []
        rrows.append([arm, f"{kind}-{b}" if b else "-", " ".join(sorted(mine)),
                      " ".join(new) if ref is not None else "-", " ".join(gone) if ref is not None else "-"])
    write(dest / "reach.tsv", ["arm", "against", "matrix-product kernels (name<type>)", "new against it",
                               "not run any more"], rrows)

    trows = []
    tf = work / "reach" / "tip" / "types.tsv"
    if tf.exists():
        trows = [r for r in tsv(tf) if r]
    write(dest / "types.tsv", ["arch", "models", "tensors by type"], trows)

    mrows = []
    for rn in names:
        for t in sorted((work / rn).glob("*/models.md5")) if (work / rn).exists() else []:
            for ln in t.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) == 2 and p[1].endswith(".gguf"):
                    mrows.append([rn, t.parent.name, p[1], p[0]])
    write(dest / "models.tsv", ["round", "tree", "file", "md5"], mrows)


if __name__ == "__main__":
    main()
