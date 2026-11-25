#!/usr/bin/env python3
"""upstream-b12723 (from land-v5-batch3's): the data folder's tables from a copy of run.sh's work folder (standard
library only).

Usage: python3 -I tables.py <copy of the work folder> <output folder>
Reads: rig/md5.txt, rig/bin-md5-{base,head}.txt, checks/{rig,cpu}-warnings-*.txt, checks/archs/*.log, checks/ctest.tsv,
cpu/summary.tsv, cpu/{base,land}/hand/*.log, gpupipe/summary.tsv, load/<model>-<build>/, bops/summary.tsv, bops/*.log,
bops/ops-*.tsv, the kept record's steps ({replay,record,replay2}/bops/: summary.txt, *-replay-*.tsv), the machine-code
comparisons (sass/sass_compare.tsv, sass2/sass_pairs.tsv), the logits trace (cpu-trace/summary.tsv) and the final
build's check (final/: rig/md5.txt, rig/bin-md5-*.txt, checks/bin_compare.tsv from bin_compare.sh, checks/archs/,
sass/ and sass-server/sass_compare.tsv for test-backend-ops and llama-server, load/).
Writes: checks.tsv, ctest.tsv, hand.tsv, pipe.tsv, gpu_pipe.tsv, replies.tsv, bops.tsv, ops_cases.tsv, replays.tsv,
replay_cases.tsv, logits.tsv, sass_compare.tsv, sass_pairs.tsv, bin_compare.tsv, binaries.tsv.
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
# run.sh's warnings() drops gcc's, nvcc's and CMakeLists.txt's line numbers; a CMake script's (ui-assets.cmake:621) too here
def cmake_ln(ls):
    return {re.sub(r"(\.cmake):\d+", r"\1", l) for l in ls}
cb, ch = cmake_ln(lines("checks", "cpu-warnings-base.txt")), cmake_ln(lines("checks", "cpu-warnings-head.txt"))
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
write("ctest.tsv", ["test", "base 2d839da1d", "head 0a6a04e22"], ct)

# the hand-run logs line by line with compare_hand.py's rules (docs/data/2026-11-05_c565_ladder4fix/run/): the time stamp
# and level letter dropped, the trees' paths as <tree>, the numbers masked in timing lines, and thread-safety and
# context-model (two threads write each) compared as sorted lists with every stamp dropped; and without the line b12723
# adds to every context's log ("rs_replay = false", #31669)
STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ?")
ANY_STAMP = re.compile(r"\d+\.\d{2}\.\d{3}\.\d{3} ")
LEVEL = re.compile(r"^[IWED] ")
TIMING = re.compile(r"( ms\b| ms$|ms per token|tokens per second| t/s|time =|seconds)")
ORDERLESS = {"thread-safety", "context-model"}
def norm_log(path, orderless=False):
    out = []
    for l in open(path, encoding="utf-8", errors="replace"):
        if re.search(r"rs_replay\s+=", l):
            continue
        l = STAMP.sub("", l.rstrip("\n")).rstrip()
        if orderless:
            l = LEVEL.sub("", ANY_STAMP.sub("", l))
        l = re.sub(r"/cpu/(base|land)/", "/cpu/<tree>/", l)
        if TIMING.search(l):
            l = re.sub(r"\d+(\.\d+)?", "N", l)
        out.append(l)
    return sorted(out) if orderless else out
hand = []
hd_dir = os.path.join(W, "cpu", "base", "hand")
for n in sorted(os.listdir(hd_dir)) if os.path.isdir(hd_dir) else []:
    if not n.endswith(".log"):
        continue
    b, h = os.path.join(hd_dir, n), os.path.join(W, "cpu", "land", "hand", n)
    if not os.path.exists(h):
        hand.append([n[:-4], "-", "-", "only in base"])
        continue
    lb, lh = norm_log(b, n[:-4] in ORDERLESS), norm_log(h, n[:-4] in ORDERLESS)
    nd = sum(1 for x, y in zip(lb, lh) if x != y) + abs(len(lb) - len(lh))
    hand.append([n[:-4], str(len(lb)), str(len(lh)), "same" if nd == 0 else f"{nd} lines differ"])
write("hand.tsv", ["log", "base lines", "head lines", "without times, paths and the rs_replay line"], hand)
hd = [r for r in hand if r[3] != "same"]
hb = [r for r in summary if r[1].startswith("hand ")]
hv = {(r[0], r[1]): r[2] for r in hb}
hand_names = sorted({r[1] for r in hb})
checks.append(["hand-run tests, verdicts", f"{sum(1 for n in hand_names if hv.get(('base', n)) == 'OK')} of {len(hand_names)} OK",
               f"{sum(1 for n in hand_names if hv.get(('land', n)) == 'OK')} of {len(hand_names)} OK",
               "SAME" if all(hv.get(('base', n)) == hv.get(('land', n)) for n in hand_names) else "DIFFERENT", ""])
checks.append(["hand-run tests, logs line by line (hand.tsv)", "-", "-",
               "SAME" if not hd else "DIFFERENT",
               f"{len(hand)} logs, {len(hand) - len(hd)} the same without times, paths and b12723's rs_replay line; "
               + "; ".join(f"{r[0]}: {r[3]}" for r in hd)])

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

# release-file replies: head and base, and the head with upstream's three switches (headup)
NAMES = {"flashnext": "Swift Qwen3.8-Flash-Next Pascal Q2 GGUF", "flintq8": "Flint Pascal Q8 GGUF",
         "ornith": "Ornith-1.5-122B Pascal N2P GGUF"}
rep_rows = []
def reply(root, m, b, p):
    try:
        j = json.loads(rd(root, "load", f"{m}-{b}", f"reply-{p}.json"))
    except ValueError:
        j = {}
    t = j.get("timings", {})
    return j.get("tokens") or [], j.get("content", ""), (t.get("draft_n"), t.get("draft_n_accepted"))
def mem(root, m, b, f):
    return ";".join(l.replace(" ", "") for l in lines(root, "load", f"{m}-{b}", f))
for root, builds in (("", ("head", "base", "headup")), ("final", ("head", "base", "headup"))):
    for m in NAMES:
        if not os.path.isdir(os.path.join(W, root, "load", f"{m}-head")):
            continue
        got = {}
        for b in builds:
            d = os.path.join(root, "load", f"{m}-{b}")
            if not os.path.isdir(os.path.join(W, d)):
                continue
            bin_md5 = (re.search(r"^([0-9a-f]{32})\s", rd(d, "command.txt"), re.M) or [None, "-"])[1]
            for p in ("code", "prose"):
                t, c, dr = reply(root, m, b, p)
                got[(b, p)] = (t, c, dr)
                rep_rows.append([root or "first", m, b, p, rd(d, "load.txt").strip(), str(len(t)), str(len(c)),
                                 f"{dr[0]}/{dr[1]}" if dr[0] is not None else "-", mem(root, m, b, "mem.csv"),
                                 mem(root, m, b, "mem-after.csv"), bin_md5])
        # the layer split the load ran with (three cards -ts 1,1,1; the final build's load ran on two, -ts 1,1)
        ts = (re.search(r"-ts (\S+)", rd(root, "load", f"{m}-head", "command.txt")) or [None, "?"])[1]
        if root:  # the final run: its head is caf27fdae, its base the first head 0a6a04e22
            label = {"head": "final caf27fdae", "base": "first head 0a6a04e22", "headup": "final with UP_ENV"}
        else:
            label = {"head": "head", "base": "base", "headup": "head with UP_ENV"}
        for x, y in (("head", "base"), ("headup", "head")):
            for p in ("code", "prose"):
                if (x, p) not in got or (y, p) not in got:
                    continue
                a_, b_ = got[(x, p)], got[(y, p)]
                same = a_[0] == b_[0] and len(a_[0]) > 0
                n = next((i for i, (u, v) in enumerate(zip(a_[0], b_[0])) if u != v), min(len(a_[0]), len(b_[0])))
                ma = mem(root, m, x, "mem-after.csv") == mem(root, m, y, "mem-after.csv") and mem(root, m, x, "mem.csv") == mem(root, m, y, "mem.csv")
                checks.append([f"{'final build: ' if root else ''}greedy reply, {NAMES[m]}, {p} (v5 preset, -ts {ts}), {label[x]} against {label[y]}",
                               f"{len(b_[0])} tokens", f"{len(a_[0])} tokens" if a_[0] else "no reply",
                               "SAME" if same else ("NO REPLY" if not a_[0] else "DIFFERENT"),
                               ("the same token ids" if same else f"first difference at token {n}") +
                               (f"; drafted/accepted {a_[2][0]}/{a_[2][1]} and {b_[2][0]}/{b_[2][1]}" if a_[2][0] is not None else "") +
                               ("; MiB per card the same after the load and after the requests" if ma and same else "")])
write("replies.tsv", ["build check", "model", "build", "request", "load", "tokens", "characters", "drafted/accepted",
                      "MiB per card after the load (card,used,total)", "after the requests", "llama-server md5"], rep_rows)

# card suite
bo = tsv("bops", "summary.tsv")
envs = {"mul_mat-p106": "P106_ENV", "mul_mat-p106-mt2": "P106_ENV GGML_CUDA_MMVQ_PASCAL_MT=2",
        "mul_mat-p106-j64": "P106_ENV GGML_CUDA_MMQ_J64_I64=2", "mul_mat_id-p106": "P106_ENV",
        "fa-plain": "none", "fa-forced": "GGML_CUDA_FATTN_TILE_PASCAL=2",
        "fa-all": "GGML_CUDA_FATTN_TILE_PASCAL=2 _LDS=1 _KDP=1 _MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4",
        "ops-head": "P106_ENV (head)", "ops-base": "P106_ENV (base)", "ops-headup": "P106_ENV UP_ENV (head)"}
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
# the operators b12723 touched, case by case: op name, cases on each side, verdicts compared
ops = []
for f, (xa, ya) in (("ops-head-base.tsv", ("head", "base")), ("ops-headup-head.tsv", ("head with UP_ENV", "head"))):
    per = {}
    for r in tsv("bops", f):
        if len(r) < 3:
            continue
        op = r[0].split("(")[0]
        c = per.setdefault(op, {"x": 0, "y": 0, "both": 0, "same": 0, "x OK": 0, "y OK": 0, "differ": []})
        if r[1] != "-":
            c["x"] += 1
            c["x OK"] += r[1] == "OK"
        if r[2] != "-":
            c["y"] += 1
            c["y OK"] += r[2] == "OK"
        if r[1] != "-" and r[2] != "-":
            c["both"] += 1
            if r[1] == r[2]:
                c["same"] += 1
            else:
                c["differ"].append(r[0][:160])
    for op, c in sorted(per.items()):
        ops.append([f"{xa} against {ya}", op, str(c["x"]), str(c["x OK"]), str(c["y"]), str(c["y OK"]), str(c["both"]),
                    str(c["same"]), " | ".join(c["differ"][:3]) or "-"])
    tb = sum(c["both"] for c in per.values())
    ts = sum(c["same"] for c in per.values())
    checks.append([f"test-backend-ops, the operators b12723 touched, {xa} against {ya}",
                   f"{sum(c['y'] for c in per.values())} cases, {sum(c['y OK'] for c in per.values())} OK",
                   f"{sum(c['x'] for c in per.values())} cases, {sum(c['x OK'] for c in per.values())} OK",
                   "SAME" if tb and ts == tb else "DIFFERENT", f"{tb} cases on both sides, the same verdict {ts}"])
write("ops_cases.tsv", ["comparison", "operator", "cases (first)", "OK (first)", "cases (second)", "OK (second)",
                        "cases on both", "same verdict", "differing cases"], ops)

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

# the random models' processor logits: the rig builds (archs) and the CPU trees (ident), and the trace tree
lg = []
for l in compared:
    m = re.match(r"(\S+): logits against \S+: nmse (\S+), max abs diff (\S+), (\d+) of (\d+) differ", l)
    if m and m.group(4) != "0":
        lg.append(["rig builds, head against base", m.group(1), m.group(2), m.group(3), f"{m.group(4)} of {m.group(5)}"])
idt = by.get(("land", "ident vs base"))
if idt:
    for a_ in (re.search(r"differ: ([^;]*);", idt[3]) or [None, ""])[1].split():
        lg.append(["CPU trees, head against base", a_, "-", "-", "differ"])
tr = [r for r in tsv("cpu-trace", "summary.tsv") if len(r) >= 3 and r[1] == "ident vs base"]
for r in tr:
    lg.append(["CPU trees, trace (softplus helper and inkling2 graph as b12674) against base", "all 152 files",
               "-", "-", r[2] + ": " + (re.search(r"differ: ([^;]*);", r[3]) or [None, "?"])[1]])
    checks.append(["test-llama-archs trace: head with #32655's softplus helper and the three inkling2 graph commits "
                   "reverted, CPU tree, against base", "-", "-", r[2], r[3].strip()])
write("logits.tsv", ["comparison", "architecture", "nmse", "max abs diff", "logits differing"], lg)
sc = tsv("sass", "sass_compare.tsv")
if sc:
    write("sass_compare.tsv", sc[0], sc[1:])
    tot = {}
    for r in sc[1:]:
        tot[r[1]] = tot.get(r[1], 0) + int(r[2])
    fam = {}
    for r in sc[1:]:
        if r[1] != "same bytes":
            fam.setdefault(r[0], []).append(f"{r[1]} {r[2]}")
    checks.append(["test-backend-ops machine code (cuobjdump -sass), kernel by kernel", "-",
                   ", ".join(f"{k} {v}" for k, v in sorted(tot.items())), "SAME but named",
                   "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(fam.items()))])
sp = tsv("sass2", "sass_pairs.tsv")
if sp:
    write("sass_pairs.tsv", sp[0], sp[1:])
    for famname in ("gated_delta_net_cuda", "rms_norm_mul_rope_f32"):
        rr = [r for r in sp[1:] if famname in r[0]]
        same = [r for r in rr if r[3].endswith(" 0")]
        checks.append([f"renamed kernels, {famname}: each head instance (the new template argument false) against its base twin",
                       f"{len(rr)} pairs", f"{len(same)} the same instructions past parameter offsets",
                       "SAME" if len(same) == len(rr) else "SAME but named",
                       "; ".join(f"{r[0][:60]}...: {r[2]}, {r[3]}" for r in rr if r not in same)])

# the final build (caf27fdae, the web page pinned to b12722) against the first head (0a6a04e22)
fm = lines("final", "rig", "md5.txt")
if fm:
    checks.append(["final build (TESTS=1, web page b12722)", "-", fm[0].split()[1], "OK", f"md5 {fm[0].split()[0]}, {fm[1] if len(fm) > 1 else ''}"])
    fb = {l.split()[1]: l.split()[0] for l in lines("final", "rig", "bin-md5-base.txt") if len(l.split()) == 2}
    fh = {l.split()[1]: l.split()[0] for l in lines("final", "rig", "bin-md5-head.txt") if len(l.split()) == 2}
    dif = sorted(n for n in set(fb) | set(fh) if fb.get(n) != fh.get(n))
    checks.append(["final build's programs against the first head's: md5", f"{len(fb)} programs", f"{len(fh)} programs",
                   "DIFFERENT" if dif else "SAME",
                   f"{len(dif)} of {len(fh)} md5s differ (every program carries its commit id and an ELF build id: "
                   "bin_compare.tsv shows where)"])
    # bin_compare.sh: bytes that differ and the defined symbols whose size differs, program by program
    bc = tsv("final", "checks", "bin_compare.tsv")
    if bc:
        write("bin_compare.tsv", bc[0], bc[1:])
        rows_ = [r for r in bc[1:] if len(r) >= 7]
        nolib = [r for r in rows_ if r[6] == "0"]
        withf = [r for r in rows_ if r[6] != "0"]
        funcs = sorted({f for r in withf for f in (r[7] if len(r) > 7 else "").split("|") if f})
        checks.append(["final build's programs against the first head's: bytes and functions (bin_compare.sh)",
                       f"{len(rows_)} programs", f"{len(nolib)} with no function changed, {len(withf)} with one",
                       "SAME but named",
                       f"no function changed: {', '.join(r[0] for r in nolib)} "
                       f"(bytes differing: {', '.join(sorted({r[3] for r in nolib}))}: the short commit id and the ELF build id); "
                       f"the rest link libllama and differ in the size of {'; '.join(funcs)} only "
                       "(the code after it moves, so millions of bytes differ: bin_compare.tsv lists each program)"
                       if withf else ""])
    fl = rd("final", "checks", "archs", "head-compare.log")
    fc = [l for l in fl.splitlines() if "logits against" in l]
    fd = [l for l in fc if ", 0 of " not in l]
    checks.append(["final build: test-llama-archs -b CPU -s 1 logits against the first head's", "-", f"{len(fc)} compared",
                   "SAME" if fc and not fd else "DIFFERENT", f"differ: {len(fd)}"])
    for sd, prog in (("sass", "test-backend-ops"), ("sass-server", "llama-server")):
        fs = tsv("final", sd, "sass_compare.tsv")
        if fs:
            ft = {}
            for r in fs[1:]:
                ft[r[1]] = ft.get(r[1], 0) + int(r[2])
            checks.append([f"final build: {prog} machine code (cuobjdump -sass) against the first head's, kernel by kernel", "-",
                           ", ".join(f"{k} {v}" for k, v in sorted(ft.items())),
                           "SAME" if set(ft) == {"same bytes"} else "DIFFERENT", ""])

write("checks.tsv", ["check", "base 2d839da1d", "head 0a6a04e22 (final: caf27fdae)", "verdict", "detail"], checks)

# binaries
mb = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-base.txt") if len(l.split()) == 2}
mh = {l.split()[1]: l.split()[0] for l in lines("rig", "bin-md5-head.txt") if len(l.split()) == 2}
fh2 = {l.split()[1]: l.split()[0] for l in lines("final", "rig", "bin-md5-head.txt") if len(l.split()) == 2}
bins = [["tarball", "base llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3.tar.gz, head " + (md5[0].split()[1] if md5 else "-")
         + ", final " + (fm[0].split()[1] if fm else "-"),
         "7d774f05c29460a6993b19d38796047c", md5[0].split()[0] if md5 else "-", fm[0].split()[0] if fm else "-"]]
for n in sorted(set(mb) | set(mh) | set(fh2)):
    bins.append(["bin/" + n, "", mb.get(n, "-"), mh.get(n, "-"), fh2.get(n, "-")])
write("binaries.tsv", ["file", "name", "base md5 (2d839da1d)", "head md5 (0a6a04e22)", "final md5 (caf27fdae)"], bins)
