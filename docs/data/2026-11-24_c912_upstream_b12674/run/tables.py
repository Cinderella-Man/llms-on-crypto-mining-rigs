#!/usr/bin/env python3
"""C912: the data folder's tables from a copy of run.sh's work folder (standard library only).

Usage: python3 -I tables.py <copy of the work folder> <output folder>
Reads: rig/md5.txt, rig/bin-md5-{base,head}.txt, checks/{rig,cpu}-warnings-*.txt, checks/archs/*.log, checks/ctest.tsv,
checks/hand.tsv, cpu/summary.tsv, cpu-trace/summary.tsv, gpupipe/summary.tsv, load/<model>-<build>/, bops/summary.tsv,
bops/*.log and sass/{summary.txt,kq.txt}.
Writes: checks.tsv, ctest.tsv, pipe.tsv, gpu_pipe.tsv, replies.tsv, bops.tsv, binaries.tsv.
"""
import csv
import json
import os
import re
import sys

W, OUT = sys.argv[1], sys.argv[2]


def rd(*p):
    path = os.path.join(W, *p)
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""


def lines(*p):
    return [l for l in rd(*p).splitlines() if l.strip()]


def tsv(*p):
    return [l.split("\t") for l in lines(*p)]


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


checks = []

# rig build and warnings
md5 = lines("rig", "md5.txt")
checks.append(["rig build (TESTS=1)", "-", md5[0].split()[1] if md5 else "-", "OK" if md5 else "FAIL",
               f"md5 {md5[0].split()[0]}, {md5[1]}" if len(md5) > 1 else "no tarball"])
rb, rh = set(lines("checks", "rig-warnings-base.txt")), set(lines("checks", "rig-warnings-head.txt"))
checks.append(["rig build warnings", str(len(rb)), str(len(rh)), "OK" if not rh - rb else "NEW",
               f"only in head: {len(rh - rb)}; only in base: {len(rb - rh)}"])
cb, ch = set(lines("checks", "cpu-warnings-base.txt")), set(lines("checks", "cpu-warnings-head.txt"))
checks.append(["CPU build warnings", str(len(cb)), str(len(ch)), "OK" if not ch - cb else "NEW",
               f"only in head: {len(ch - cb)}; only in base: {len(cb - ch)}"])

# rig builds' logits files
cmp_log = rd("checks", "archs", "head-compare.log")
n_files = rd("checks", "archs", "n_files.txt").strip() or "?"
compared = [l for l in cmp_log.splitlines() if "logits against" in l]
differ = [l for l in compared if ", 0 of " not in l]
summ = {k: (re.search(r"^Summary.*$", rd("checks", "archs", f"{k}.log"), re.M) or [""])[0]
        for k in ("head-plain", "base-write", "head-compare")}
checks.append(["test-llama-archs -b CPU -s 1, rig builds: logits files", f"{n_files} written", f"{len(compared)} compared",
               "SAME" if compared and not differ else "DIFFERENT",
               f"differ: {len(differ)}; head {summ['head-plain']}; head against base {summ['head-compare']}"])

# CPU trees (cpu_tests.sh's summary)
summary = tsv("cpu", "summary.tsv")[1:]
by = {}
for r in summary:
    if len(r) >= 3:
        by[(r[0], r[1])] = r
for c in ("build", "archs files", "ident vs base"):
    b, h = by.get(("base", c)), by.get(("cand", c))
    checks.append([f"CPU tree: {c}", b[2] if b else "-", h[2] if h else "-", h[2] if h else "-",
                   (h[3] if h and len(h) > 3 else "").strip()])

ct = tsv("checks", "ctest.tsv")
diffs = [r for r in ct if len(r) == 3 and r[1] != r[2]]
checks.append(["ctest, test by test", f"{sum(1 for r in ct if r[1] != '-')} tests, {sum(1 for r in ct if r[1] == 'Passed')} pass",
               f"{sum(1 for r in ct if r[2] != '-')} tests, {sum(1 for r in ct if r[2] == 'Passed')} pass",
               "SAME but named" if diffs else "SAME", "; ".join(f"{r[0]}: {r[1]} -> {r[2]}" for r in diffs)])
write("ctest.tsv", ["test", "base af00bb13b", "candidate"], ct)

# the trace: the candidate without the automatic weight-scale lookup, its logits against the base's
tr = {(r[0], r[1]): r for r in tsv("cpu-trace", "summary.tsv")[1:] if len(r) >= 3}
t = tr.get(("noscale", "ident vs base"))
if t:
    checks.append(["trace: the candidate without the automatic '<name>.scale' lookup, logits against the base's CPU files",
                   "-", "-", t[2], (t[3] if len(t) > 3 else "").strip()])

hand = tsv("checks", "hand.tsv")[1:]
hd = [r for r in hand if len(r) > 4 and r[4] != "same"]
hb = [r for r in summary if r[1].startswith("hand ")]
hv = {(r[0], r[1]): r[2] for r in hb}
hand_names = sorted({r[1] for r in hb})
checks.append(["hand-run tests, verdicts", f"{sum(1 for n in hand_names if hv.get(('base', n)) == 'OK')} of {len(hand_names)} OK",
               f"{sum(1 for n in hand_names if hv.get(('cand', n)) == 'OK')} of {len(hand_names)} OK",
               "SAME" if all(hv.get(('base', n)) == hv.get(('cand', n)) for n in hand_names) else "DIFFERENT", ""])
checks.append(["hand-run tests, logs line by line (compare_hand.py)", "-", "-",
               "SAME but named" if hd else "SAME",
               f"{len(hand)} comparisons, {len(hand) - len(hd)} the same; " + "; ".join(f"{r[0]}: {r[5] if len(r) > 5 else r[4]}" for r in hd)])

pipe = [[r[0], r[1].replace("pipe ", "", 1), r[2], (r[3] if len(r) > 3 else "").strip()] for r in summary if r[1].startswith("pipe ")]
pipe.sort(key=lambda r: (r[1], r[0] != "base"))
write("pipe.tsv", ["tree", "architecture and arm", "verdict", "detail"], pipe)
pb = [r for r in pipe if r[0] == "base"]
ph = [r for r in pipe if r[0] == "cand"]
checks.append(["CPU pipe step (C935, C1010): 5 architectures x 4 arms",
               f"{sum(1 for r in pb if r[2] == 'OK')} of {len(pb)} OK",
               f"{sum(1 for r in ph if r[2] == 'OK')} of {len(ph)} OK",
               "OK" if ph and all(r[2] == "OK" for r in ph) else "DIFFERS", ""])

# card pipe step
gp = tsv("gpupipe", "summary.tsv")
if gp:
    hdr, rows = gp[0], [r for r in gp[1:] if len(r) >= 6]
    keep = [r[:6] + [r[7] if len(r) > 7 else "", r[8] if len(r) > 8 else "", (r[6] if len(r) > 6 else "").strip()[:200]] for r in rows]
    write("gpu_pipe.tsv", ["build", "arm", "layout", "architecture", "check", "verdict", "plain tokens hash",
                           "plain logits hash", "detail"], keep)
    for t in ("base", "head"):
        tr = [r for r in rows if r[0] == t and r[5] in ("OK", "DIFFERS", "FAIL")]
        bad = sorted({f"{r[3]} {r[4]}" for r in tr if r[5] != "OK"})
        checks.append([f"card pipe step (C1009), {t}", "-", f"{sum(1 for r in tr if r[5] == 'OK')} of {len(tr)} OK",
                       "OK" if not bad else "DIFFERS", ("not OK: " + ", ".join(bad)) if bad else ""])
    hb_ = {(r[1], r[2], r[3], r[4]): (r[7] if len(r) > 7 else "", r[8] if len(r) > 8 else "") for r in rows if r[0] == "base"}
    hh_ = {(r[1], r[2], r[3], r[4]): (r[7] if len(r) > 7 else "", r[8] if len(r) > 8 else "") for r in rows if r[0] == "head"}
    both = [k for k in hb_ if k in hh_ and k[3] == "host" and hb_[k][0] not in ("", "-")]
    same = [k for k in both if hb_[k] == hh_[k]]
    checks.append(["card pipe step: plain decoding's hashes, head against base", "-", f"{len(same)} of {len(both)} the same",
                   "SAME" if both and len(same) == len(both) else "DIFFERENT",
                   "; ".join(f"{'/'.join(k)}" for k in both if k not in same)])

# release-file replies: three files, two requests each
NAMES = {"flashnext": "Swift Qwen3.8-Flash-Next Pascal Q2 GGUF", "flintq8": "Flint Pascal Q8 GGUF",
         "ornith": "Ornith-1.5-122B Pascal N2P GGUF"}
rep = []
for m in ("flashnext", "flintq8", "ornith"):
    for p in ("code", "prose"):
        got = {}
        for b in ("head", "base"):
            d = os.path.join("load", f"{m}-{b}")
            try:
                j = json.loads(rd(d, f"reply-{p}.json"))
            except ValueError:
                j = {}
            tm = j.get("timings", {})
            got[b] = (j.get("tokens") or [], j.get("content", ""), (tm.get("draft_n"), tm.get("draft_n_accepted")))
            mem = ";".join(l.replace(" ", "") for l in lines(d, "mem.csv"))
            mem2 = ";".join(l.replace(" ", "") for l in lines(d, "mem-after.csv"))
            bin_md5 = (re.search(r"^([0-9a-f]{32})\s", rd(d, "command.txt"), re.M) or [None, "-"])[1]
            dr = "-" if got[b][2][0] is None else f"{got[b][2][0]} drafted, {got[b][2][1]} accepted"
            rep.append([m, p, b, rd(d, "load.txt").strip(), str(len(got[b][0])), str(len(got[b][1])), dr, mem, mem2, bin_md5])
        same = got["head"][0] == got["base"][0] and got["head"][0]
        checks.append([f"greedy reply, {NAMES[m]}, {p} request (v5 preset, -ts 1,1,1, the fork's kernels forced)",
                       f"{len(got['base'][0])} tokens", f"{len(got['head'][0])} tokens", "SAME" if same and got["head"][2] == got["base"][2] else "DIFFERENT",
                       ("the same token ids and text" if same and got["head"][1] == got["base"][1] else "other tokens or text")
                       + ("" if got["head"][2][0] is None else f"; drafts {got['head'][2]} and {got['base'][2]}")])
write("replies.tsv", ["model", "request", "build", "load", "tokens", "characters", "draft", "MiB per card after the load (card,used,total)",
                      "after the requests", "llama-server md5"], rep)

# SASS
ss = rd("sass", "summary.txt").splitlines()
if ss:
    checks.append(["cuobjdump -sass, every sm_61 kernel of llama-server", "-", "-", "EXPLAINED",
                   ss[0].strip() + "; " + rd("sass", "kq.txt").strip() + " (see sass_families.tsv and the README)"])

# card suite
bo = tsv("bops", "summary.tsv")
envs = {"mul_mat-p106": "P106_ENV", "mul_mat-p106-mt2": "P106_ENV GGML_CUDA_MMVQ_PASCAL_MT=2", "mul_mat_id-p106": "P106_ENV",
        "fa-plain": "none", "fa-forced": "GGML_CUDA_FATTN_TILE_PASCAL=2",
        "fa-all": "GGML_CUDA_FATTN_TILE_PASCAL=2 _LDS=1 _KDP=1 _MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"}
brows = []
for r in bo[1:]:
    if len(r) < 7:
        continue
    fails = [l.strip()[:200] for l in rd("bops", f"{r[0]}.log").splitlines() if ": FAIL" in l and "Backend" not in l]
    brows.append([r[0], envs.get(r[0], "?"), r[1], r[2], r[3], r[4], r[5], r[6], " | ".join(fails[:5]) or "-"])
    checks.append([f"test-backend-ops {r[0]}", "-", r[2] or "-", "OK" if r[4] == "0" and r[1] == "0" else "FAIL",
                   f"settings: {envs.get(r[0], '?')}; FAIL lines {r[4]}; not supported {r[5]}; illegal access {r[6]}"])
write("bops.tsv", ["arm", "settings", "exit", "passed", "OK lines", "FAIL lines", "not supported", "illegal access",
                   "failing cases"], brows)

write("checks.tsv", ["check", "base af00bb13b", "candidate", "verdict", "detail"], checks)

# binaries
mb = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-base.txt") if len(l.split()) == 2}
mh = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-head.txt") if len(l.split()) == 2}
bins = [["tarball", "base llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz, head " + (md5[0].split()[1] if md5 else "-"),
         "b51ad2740d5822e7e14af656c413ec22", md5[0].split()[0] if md5 else "-"]]
for n in sorted(set(mb) | set(mh)):
    bins.append(["bin/" + n, "", mb.get(n, "-"), mh.get(n, "-")])
write("binaries.tsv", ["file", "name", "base md5", "head md5"], bins)
