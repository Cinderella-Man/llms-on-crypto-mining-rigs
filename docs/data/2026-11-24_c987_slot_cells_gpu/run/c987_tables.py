#!/usr/bin/env python3
"""C987 (Q1720, LLMRig point 17): tables from run.sh's output folder. Nothing timed is read or written: the server's
lines carry host times, which are cut off here; only token ids, log-probabilities, cells and counts are kept.

Usage: python3 -I c987_tables.py <WORK/out> <output folder>
Writes:
  loads.tsv     per load: the KV cache's cells, sequences and streams (the server's llama_kv_cache line), the CUDA
                backend's features that name the fork's Pascal kernels, asserts, requests answered
  replies.tsv   per load and request: slot, prompt tokens read, cached tokens used, generated tokens, the md5 of the
                token ids (12 hex digits), the stop reason, and the cache's placements for it: how many, the stream,
                the head and used cells before the first, the highest used cell + 1 before the last. A placement is one
                LLAMA_KV_CACHE_DEBUG line printed before a micro-batch is placed; the search for free cells starts at
                the head
  cells.tsv     per load, the fixed request (fix-1): where its cells went (stream, first cell, the highest used cell + 1
                at its end, the other slot's cells in the same stream) and the same for fix-2 and fix-p
  compare.tsv   per load: fix-1's token ids against the reference (one slot, first load), with the reply token where
                they first differ; fix-2 and fix-p against fix-1 of the same load; fix-1 against the same arm's other
                load
  probs.tsv     per load: fix-p's log-probabilities against the reference's fix-p, step by step up to the first step
                whose token differs: steps compared, steps whose chosen token's log-probability differs at all, the
                largest difference, the first step that differs, the smallest gap between the two most likely tokens
  verdicts.tsv  per mode: distinct fix-1 replies over its histories and loads, whether each equals the reference
"""
import csv
import json
import os
import re
import sys

OUT, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
MODES = ["one", "kvu", "streams"]
HISTS = ["none", "long", "short"]
TASK = re.compile(r"processing task, is_child = 0")
FS = re.compile(r"find_slot: stream\[(\d+)\], n = +(\d+), used = +(\d+), head = +(\d+), size = +(\d+)")
FSQ = re.compile(r"find_slot: stream\[(\d+)\] min\[(\d+)\] = +(-?\d+), max\[\d+\] = +(-?\d+)")
KV = re.compile(r"llama_kv_cache: size = .*\(\s*(\d+) cells,\s*(\d+) layers,\s*(\d+)/(\d+) seqs\)")
FEAT = re.compile(r"(MMVQ_PASCAL|MMVF_PASCAL|COMPUTE_6_1|ARCHS) = (\S+)")
ASSERT = re.compile(r"GGML_ASSERT|GGML_ABORT")


def read_log(log):
    """KV cache lines, features, asserts, and the placements split by request (segment 0 = the load)"""
    kv, feat, asserts, segs = [], {}, 0, [[]]
    try:
        f = open(log, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return kv, feat, asserts, segs
    with f:
        for line in f:
            if TASK.search(line):
                segs.append([])
                continue
            m = FS.search(line)
            if m:
                segs[-1].append(dict(stream=int(m.group(1)), n=int(m.group(2)), used=int(m.group(3)),
                                     head=int(m.group(4)), size=int(m.group(5)), seqs={}))
                continue
            m = FSQ.search(line)
            if m and segs[-1]:
                segs[-1][-1]["seqs"][int(m.group(2))] = (int(m.group(3)), int(m.group(4)))
                continue
            m = KV.search(line)
            if m:
                kv.append("%s cells, %s/%s seqs/streams" % (m.group(1), m.group(3), m.group(4)))
                continue
            for k, v in FEAT.findall(line):
                feat[k] = v
            if ASSERT.search(line):
                asserts += 1
    return kv, feat, asserts, segs


def tokens(p, tag, req):
    try:
        with open(os.path.join(p, f"{tag}-{req}.json"), encoding="utf-8") as f:
            return json.load(f).get("tokens") or []
    except (FileNotFoundError, ValueError):
        return []


def probs(p, tag):
    rows = []
    try:
        with open(os.path.join(p, f"{tag}-fix-p.probs.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                x = line.rstrip("\n").split("\t")
                rows.append((int(x[1]), float(x[2]), x[3], float(x[4]) if x[4] not in ("", "''") else None))
    except (FileNotFoundError, ValueError, StopIteration, IndexError):
        pass
    return rows


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return "" if len(a) == len(b) else min(len(a), len(b))


loads, rows, segs_of, tok, prb, kvl = [], {}, {}, {}, {}, {}
for d in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, d)
    m = re.fullmatch(r"(one|kvu|streams)-(none|long|short)-r(\d+)", d)
    if not (os.path.isdir(p) and m):
        continue
    mode, hist, rep = m.group(1), m.group(2), int(m.group(3))
    key = (mode, hist, rep)
    kv, feat, asserts, segs = read_log(os.path.join(p, "server.log"))
    reqs = []
    try:
        with open(os.path.join(p, "session.tsv"), encoding="utf-8") as g:
            for line in g:
                x = line.rstrip("\n").split("\t")
                rows[key + (x[1],)] = dict(slot=x[2], prompt_n=x[3], cache_n=x[4], n_gen=x[5], md5=x[6][:12], stop=x[8])
                reqs.append(x[1])
    except FileNotFoundError:
        pass
    # segment i (1-based) is the i-th request sent; a request that failed before reaching a slot would shift this
    segs_of[key] = {q: (segs[i + 1] if i + 1 < len(segs) else None) for i, q in enumerate(reqs)}
    for q in reqs:
        tok[key + (q,)] = tokens(p, d, q)
    prb[key] = probs(p, d)
    kvl[key] = kv
    loads.append((key, reqs, kv, feat, asserts, len(segs) - 1))

loads.sort(key=lambda t: (MODES.index(t[0][0]), HISTS.index(t[0][1]), t[0][2]))
REF = ("one", "none", 1)
ref_tok = tok.get(REF + ("fix-1",), [])
ref_prb = prb.get(REF, [])


def name(key):
    return "%s-%s-r%d" % key


with open(os.path.join(DST, "loads.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["load", "mode", "history", "rep", "kv_cache", "MMVQ_PASCAL", "MMVF_PASCAL", "COMPUTE_6_1", "asserts",
                "requests", "tasks_logged"])
    for key, reqs, kv, feat, asserts, ntask in loads:
        w.writerow([name(key), *key, "; ".join(sorted(set(kv))), feat.get("MMVQ_PASCAL", ""),
                    feat.get("MMVF_PASCAL", ""), feat.get("COMPUTE_6_1", ""), asserts, len(reqs), ntask])


def place(seg):
    if not seg:
        return dict(n="", stream="", head="", used="", top="")
    return dict(n=len(seg), stream=seg[0]["stream"], head=seg[0]["head"], used=seg[0]["used"], top=seg[-1]["n"])


with open(os.path.join(DST, "replies.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["load", "request", "slot", "prompt_n", "cache_n", "n_gen", "tokens_md5", "stop", "placements",
                "stream", "head_first", "used_first", "top_last"])
    for key, reqs, *_ in loads:
        for q in reqs:
            r, pl = rows[key + (q,)], place(segs_of[key].get(q))
            w.writerow([name(key), q, r["slot"], r["prompt_n"], r["cache_n"], r["n_gen"], r["md5"], r["stop"], pl["n"],
                        pl["stream"], pl["head"], pl["used"], pl["top"]])

with open(os.path.join(DST, "cells.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["load", "stream", "fix1_first_cell", "fix1_top_cell_plus1", "other_slot_cells", "fix2_first_cell",
                "fixp_first_cell", "same_cells_in_load"])
    for key, reqs, *_ in loads:
        s = segs_of[key]
        a, b, c = place(s.get("fix-1")), place(s.get("fix-2")), place(s.get("fix-p"))
        w.writerow([name(key), a["stream"], a["head"], a["top"], a["used"], b["head"], c["head"],
                    "yes" if a["head"] == b["head"] == c["head"] and a["top"] == b["top"] == c["top"] else "no"])

with open(os.path.join(DST, "compare.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["load", "fix1_tokens", "fix1_md5", "same_as_reference", "first_diff_token", "fix2_same_as_fix1",
                "fixp_same_as_fix1", "same_as_other_load"])
    for key, reqs, *_ in loads:
        t1, t2, tp = tok.get(key + ("fix-1",), []), tok.get(key + ("fix-2",), []), tok.get(key + ("fix-p",), [])
        other = [k for k, *_ in loads if k[:2] == key[:2] and k != key]
        to = tok.get(other[0] + ("fix-1",), None) if other else None
        w.writerow([name(key), len(t1), rows.get(key + ("fix-1",), {}).get("md5", ""),
                    "yes" if t1 and t1 == ref_tok else "no", first_diff(t1, ref_tok) if t1 != ref_tok else "",
                    "yes" if t2 == t1 and t1 else "no", "yes" if tp == t1 and t1 else "no",
                    "" if to is None else ("yes" if to == t1 else "no (token %s)" % first_diff(t1, to))])

with open(os.path.join(DST, "probs.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["load", "steps", "first_token_diff", "steps_compared", "steps_logprob_differs", "largest_abs_diff",
                "first_step_differs", "smallest_top2_gap_compared"])
    for key, reqs, *_ in loads:
        pr = prb.get(key, [])
        fd = first_diff([x[0] for x in pr], [x[0] for x in ref_prb])
        n = len(pr) if fd == "" else fd
        diffs = [abs(pr[i][1] - ref_prb[i][1]) for i in range(n)]
        nz = [i for i, x in enumerate(diffs) if x != 0.0]
        gaps = [pr[i][1] - pr[i][3] for i in range(n) if pr[i][3] is not None]
        w.writerow([name(key), len(pr), fd, n, len(nz), "%.3g" % max(diffs) if diffs else "", nz[0] if nz else "",
                    "%.4g" % min(gaps) if gaps else ""])

with open(os.path.join(DST, "verdicts.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["mode", "loads", "distinct_fix1_replies", "loads_same_as_reference", "histories_that_change_it",
                "repeats_in_load_same", "loads_repeat_each_other"])
    for mode in MODES:
        ks = [k for k, *_ in loads if k[0] == mode]
        t = {k: tuple(tok.get(k + ("fix-1",), [])) for k in ks}
        moved = sorted({k[1] for k in ks if list(t[k]) != ref_tok})
        inload = sum(tok.get(k + ("fix-2",), []) == list(t[k]) == tok.get(k + ("fix-p",), []) for k in ks)
        pairs = {}
        for k in ks:
            pairs.setdefault(k[:2], []).append(t[k])
        rep = sum(len(set(v)) == 1 for v in pairs.values())
        w.writerow([mode, len(ks), len(set(t.values())), sum(list(v) == ref_tok for v in t.values()),
                    " ".join(moved) if moved else "none", "%d of %d" % (inload, len(ks)), "%d of %d" % (rep, len(pairs))])
