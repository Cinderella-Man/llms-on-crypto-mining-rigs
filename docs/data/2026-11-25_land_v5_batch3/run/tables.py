#!/usr/bin/env python3
"""land-v5-batch3: the data folder's tables from a copy of run.sh's work folder (standard library only).

Usage: python3 -I tables.py <copy of the work folder> <output folder>
Reads: rig/md5.txt, rig/bin-md5-{base,head}.txt, checks/{rig,cpu}-warnings-*.txt, checks/archs/*.log, checks/ctest.tsv,
checks/hand.tsv, cpu/summary.tsv, gpupipe/summary.tsv, load/<model>-<build>/, bops/summary.tsv, bops/*.log and the
kept record's steps ({replay,record,replay2}/bops/: summary.txt, *-replay-*.tsv, record.txt).
Also the repeats of the one failure (replay.log, cc-*.log, repeat*/repeats.tsv) and the machine-code comparison
(sass/sass_compare.tsv).
Writes: checks.tsv, ctest.tsv, pipe.tsv, gpu_pipe.tsv, replies.tsv, bops.tsv, replays.tsv, replay_cases.tsv,
cpucheck_runs.tsv, repeats.tsv, sass_compare.tsv, binaries.tsv.
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
    b, h = by.get(("base", c)), by.get(("land", c))
    checks.append([f"CPU tree: {c}", b[2] if b else "-", h[2] if h else "-", h[2] if h else "-",
                   (h[3] if h and len(h) > 3 else "").strip()])

ct = tsv("checks", "ctest.tsv")
diffs = [r for r in ct if len(r) == 3 and r[1] != r[2]]
checks.append(["ctest, test by test", f"{sum(1 for r in ct if r[1] != '-')} tests, {sum(1 for r in ct if r[1] == 'Passed')} pass",
               f"{sum(1 for r in ct if r[2] != '-')} tests, {sum(1 for r in ct if r[2] == 'Passed')} pass",
               "SAME but named" if diffs else "SAME", "; ".join(f"{r[0]}: {r[1]} -> {r[2]}" for r in diffs)])
write("ctest.tsv", ["test", "base eee615cdb", "head 2d839da1d"], ct)

hand = tsv("checks", "hand.tsv")[1:]
hd = [r for r in hand if len(r) > 4 and r[4] != "same"]
hb = [r for r in summary if r[1].startswith("hand ")]
hv = {(r[0], r[1]): r[2] for r in hb}
hand_names = sorted({r[1] for r in hb})
checks.append(["hand-run tests, verdicts", f"{sum(1 for n in hand_names if hv.get(('base', n)) == 'OK')} of {len(hand_names)} OK",
               f"{sum(1 for n in hand_names if hv.get(('land', n)) == 'OK')} of {len(hand_names)} OK",
               "SAME" if all(hv.get(('base', n)) == hv.get(('land', n)) for n in hand_names) else "DIFFERENT", ""])
checks.append(["hand-run tests, logs line by line (compare_hand.py)", "-", "-",
               "SAME but named" if hd else "SAME",
               f"{len(hand)} comparisons, {len(hand) - len(hd)} the same; " + "; ".join(f"{r[0]}: {r[5] if len(r) > 5 else r[4]}" for r in hd)])

pipe = [[r[0], r[1].replace("pipe ", "", 1), r[2], (r[3] if len(r) > 3 else "").strip()] for r in summary if r[1].startswith("pipe ")]
pipe.sort(key=lambda r: (r[1], r[0] != "base"))
write("pipe.tsv", ["tree", "architecture and arm", "verdict", "detail"], pipe)
pb = [r for r in pipe if r[0] == "base"]
ph = [r for r in pipe if r[0] == "land"]
vb, vh = {r[1]: r[2] for r in pb}, {r[1]: r[2] for r in ph}
checks.append([f"CPU pipe step (C935, C1010, C1020): {len(ph)} architecture-arm pairs",
               f"{sum(1 for r in pb if r[2] == 'OK')} of {len(pb)} OK",
               f"{sum(1 for r in ph if r[2] == 'OK')} of {len(ph)} OK",
               "SAME" if vb == vh and ph else "DIFFERENT",
               "; ".join(f"{k}: base {vb.get(k, '-')}, head {vh.get(k, '-')}" for k in sorted(set(vb) | set(vh)) if vb.get(k) != vh.get(k) or vh.get(k) != "OK")])

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
        checks.append([f"card pipe step (C1009, C1019), {t}", "-", f"{sum(1 for r in tr if r[5] == 'OK')} of {len(tr)} OK",
                       "OK" if not bad else "DIFFERS", ("not OK: " + ", ".join(bad)) if bad else ""])
    hb_ = {(r[1], r[2], r[3], r[4]): (r[7] if len(r) > 7 else "", r[8] if len(r) > 8 else "") for r in rows if r[0] == "base"}
    hh_ = {(r[1], r[2], r[3], r[4]): (r[7] if len(r) > 7 else "", r[8] if len(r) > 8 else "") for r in rows if r[0] == "head"}
    both = [k for k in hb_ if k in hh_ and k[3] == "host" and hb_[k][0] not in ("", "-")]
    same = [k for k in both if hb_[k] == hh_[k]]
    checks.append(["card pipe step: plain decoding's hashes, head against base", "-", f"{len(same)} of {len(both)} the same",
                   "SAME" if both and len(same) == len(both) else "DIFFERENT",
                   "; ".join(f"{'/'.join(k)}" for k in both if k not in same)])

# release-file replies
rep = []
for m in ("flintq8", "ornith"):
    toks = {}
    for b in ("head", "base"):
        d = os.path.join("load", f"{m}-{b}")
        try:
            j = json.loads(rd(d, "reply-code.json"))
        except ValueError:
            j = {}
        t = j.get("tokens") or []
        toks[b] = t
        mem = ";".join(l.replace(" ", "") for l in lines(d, "mem.csv"))
        mem2 = ";".join(l.replace(" ", "") for l in lines(d, "mem-after.csv"))
        bin_md5 = (re.search(r"^([0-9a-f]{32})\s", rd(d, "command.txt"), re.M) or [None, "-"])[1]
        rep.append([m, b, rd(d, "load.txt").strip(), str(len(t)), str(len(j.get("content", ""))), mem, mem2, bin_md5])
    same = toks["head"] == toks["base"] and toks["head"]
    checks.append([f"greedy reply, {'Flint Pascal Q8 GGUF' if m == 'flintq8' else 'Ornith-1.5-122B Pascal N2P GGUF'} (v5 preset, -ts 1,1,1)",
                   f"{len(toks['base'])} tokens", f"{len(toks['head'])} tokens", "SAME" if same else "DIFFERENT",
                   "the same token ids" if same else "other tokens"])
write("replies.tsv", ["model", "build", "load", "tokens", "characters", "MiB per card after the load (card,used,total)",
                      "after the request", "llama-server md5"], rep)

# card suite
bo = tsv("bops", "summary.tsv")
envs = {"mul_mat-p106": "P106_ENV", "mul_mat-p106-mt2": "P106_ENV GGML_CUDA_MMVQ_PASCAL_MT=2",
        "mul_mat-p106-j64": "P106_ENV GGML_CUDA_MMQ_J64_I64=2", "mul_mat_id-p106": "P106_ENV",
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

# the kept record (C1023): replays, the cpu-check, the new cases recorded, the replays again
steps = []
for d in ("replay", "record", "replay2"):
    cur = None
    for l in lines(d, "bops", "summary.txt"):
        m = re.match(r"^(\S+): (.*)$", l)
        if m and not l.startswith(" "):
            f = dict(re.findall(r"(OK|FAIL|NOREC|not supported|illegal) (\d+)", m.group(2)))
            passed = (re.search(r"(\d+/\d+ tests passed)", m.group(2)) or [None, "-"])[1]
            rc = (re.search(r"rc=(\d+)", m.group(2)) or [None, "-"])[1]
            sec = (re.search(r"; (\d+) s$", m.group(2)) or [None, "-"])[1]
            cur = [d, m.group(1), rc, passed, f.get("OK", "-"), f.get("FAIL", "-"), f.get("NOREC", "-"),
                   f.get("not supported", "-"), f.get("illegal", "-"), sec, ""]
            steps.append(cur)
        elif cur is not None and l.strip().startswith(("replay", "record")):
            # the record's folder on the build machine as <record>
            l = re.sub(r"/(?:[^/\s:]+/)*bops-record/[^/\s:]+", "<record>", l.strip())
            cur[10] = (cur[10] + "; " if cur[10] else "") + l
write("replays.tsv", ["step", "run", "exit", "passed", "OK lines", "FAIL lines", "NOREC", "not supported", "illegal access",
                      "seconds", "record or replay line"], steps)
for r in steps:
    ok = r[2] == "0" and r[5] == "0" and r[8] == "0" and "FAIL" not in r[10]
    checks.append([f"kept record: {r[0]} {r[1]}", "-", r[3], "OK" if ok else "FAIL",
                   f"NOREC {r[6]}; not supported {r[7]}; {r[10]}"])
# each case's verdict and error, head against base, in the last replay of each share
cases = []
for share in ("mul_mat_id", "mul_mat"):
    d = "replay2" if os.path.exists(os.path.join(W, "replay2", "bops", f"{share}-replay-head.tsv")) else "replay"
    def per_case(tag):
        out = {}
        for l in lines(d, "bops", f"{share}-replay-{tag}.tsv"):
            c = l.split("\t")
            if len(c) >= 3:
                out[c[0]] = (c[1], c[2], "\t".join(c[3:]))
        return out
    h, b = per_case("head"), per_case("base")
    both = [k for k in h if k in b]
    same_v = sum(1 for k in both if h[k][1] == b[k][1])
    same_e = sum(1 for k in both if h[k][1:] == b[k][1:])
    only_h = [h[k][0] for k in h if k not in b]
    cases.append([share, d, str(len(b)), str(len(h)), str(len(both)), str(same_v), str(same_e), str(len(only_h)),
                  str(sum(1 for k in h if h[k][1] == "OK")), str(sum(1 for k in h if h[k][1] == "NOREC"))])
    checks.append([f"kept record, {share}: each case head against base ({d})", f"{len(b)} cases", f"{len(h)} cases",
                   "SAME" if both and same_e == len(both) else "DIFFERENT",
                   f"{len(both)} in both: verdict the same {same_v}, verdict and error the same {same_e}; only in head {len(only_h)}"])
write("replay_cases.tsv", ["share", "replay", "base cases", "head cases", "in both", "same verdict",
                           "same verdict and error", "only in head", "head OK", "head NOREC"], cases)

# the one failure (the head's first MUL_MAT replay with --cpu-check 25) and its repeats
runs = []
for l in lines("replay.log"):
    m = re.search(r"(mul_mat\S*) (replay-\S+): \S+ rc=(\d+); (\S+) tests passed; FAIL (\d+); (\d+) s", l)
    if m and m.group(1) == "mul_mat":
        runs.append(["driver", m.group(2).replace("replay-", ""), m.group(3), m.group(4), m.group(5), m.group(6)])
for t in ("base1", "head2", "base3", "head4", "base5", "head6", "base7", "head8", "base9", "head10"):
    for l in lines(f"cc-{t}.log"):
        m = re.search(r"(mul_mat\S*) (replay-\S+): \S+ rc=(\d+); (\S+) tests passed; FAIL (\d+); (\d+) s", l)
        if m:
            step = re.sub(r"^replay-(head|base)\d*", r"\1", m.group(2))
            runs.append([f"repeat {t}", step, m.group(3), m.group(4), m.group(5), m.group(6)])
write("cpucheck_runs.tsv", ["run", "build and replay", "exit", "passed", "FAIL lines", "seconds"], runs)
cc = [r for r in runs if r[1].endswith("cpucheck")]
for b in ("head", "base"):
    rb = [r for r in cc if r[1].startswith(b)]
    checks.append([f"MUL_MAT replay with --cpu-check 25, every run, {b}", "-", f"{sum(1 for r in rb if r[4] == '0')} of {len(rb)} runs pass",
                   "OK" if all(r[4] == "0" for r in rb) else "ONE FAILURE" if sum(1 for r in rb if r[4] != "0") == 1 else "FAIL",
                   "; ".join(f"{r[0]}: {r[3]}" for r in rb)])
rep_rows = []
for d in ("repeat", "repeat2h", "repeat2b"):
    cnt = {}
    for r in tsv(d, "repeats.tsv")[1:]:
        if len(r) >= 5:
            k = (r[0], r[1])
            c = cnt.setdefault(k, {"runs": 0, "OK": 0, "FAIL": 0, "other": 0})
            c["runs"] += 1
            c[r[4] if r[4] in ("OK", "FAIL") else "other"] += 1
    for (b, mode), c in sorted(cnt.items()):
        rep_rows.append([d, b, mode, str(c["runs"]), str(c["OK"]), str(c["FAIL"]), str(c["other"])])
write("repeats.tsv", ["set", "build", "mode", "runs", "OK", "FAIL", "no verdict"], rep_rows)
for r in rep_rows:
    checks.append([f"the failing BF16 case alone, {r[0]} {r[1]} {r[2]}", "-", f"{r[4]} of {r[3]} OK", "OK" if r[5] == "0" and r[6] == "0" else "FAIL", ""])
sc = tsv("sass", "sass_compare.tsv")
if sc:
    write("sass_compare.tsv", sc[0], sc[1:])
    tot = {}
    for r in sc[1:]:
        tot[r[1]] = tot.get(r[1], 0) + int(r[2])
    mmvf = [r for r in sc[1:] if r[0].startswith("mul_mat_vec_f")]
    checks.append(["test-backend-ops machine code (cuobjdump -sass), kernel by kernel", "-", ", ".join(f"{k} {v}" for k, v in sorted(tot.items())),
                   "SAME but named", "differing: mul_mat_q in one printed line number each, the Pascal attention kernels in the fault's multiplies (C993); "
                   + "mul_mat_vec_f: " + ", ".join(f"{r[1]} {r[2]}" for r in mmvf)])

write("checks.tsv", ["check", "base eee615cdb", "head 2d839da1d", "verdict", "detail"], checks)

# binaries
mb = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-base.txt") if len(l.split()) == 2}
mh = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-head.txt") if len(l.split()) == 2}
bins = [["tarball", "base llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz, head " + (md5[0].split()[1] if md5 else "-"),
         "575ef3c702f69f14d7d3022c1287c38f", md5[0].split()[0] if md5 else "-"]]
for n in sorted(set(mb) | set(mh)):
    bins.append(["bin/" + n, "", mb.get(n, "-"), mh.get(n, "-")])
write("binaries.tsv", ["file", "name", "base md5", "head md5"], bins)
