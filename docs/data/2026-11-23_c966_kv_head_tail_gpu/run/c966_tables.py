#!/usr/bin/env python3
"""C966 (Q1696, LLMRig point 13): tables from run.sh's output folder. Nothing timed is read or written: the server's
lines carry host times, which are cut off here; only positions, cells and counts are kept.

Usage: python3 -I c966_tables.py <WORK/out> <output folder>
Writes:
  replies.tsv     one line per arm, round and request: prompt tokens read, cached tokens used, generated tokens, the
                  md5 of the token ids (12 hex digits), drafted and accepted drafts, the stop reason
  events.tsv      per arm, the server's cache events in order with their slot, positions only: checkpoints created
                  (pos_min, pos_max, n_tokens), restored (pos_min, pos_max, n_tokens, n_past), erased, full re-reads,
                  text reuse (a draft's restore-checkpoint lines are left out)
  compare.tsv     per setting, round and request: base against branch on the token ids, prompt tokens read, cached
                  tokens, drafts and the cache's placements (below); for a cancel only the chunks read
  verdicts.tsv    per setting: replies, events, mask counts and placements the same on base and branch?
  mask.tsv        per arm with the card-built mask: masks identical to the host's, of them from the resident copy
                  (kept up to date from the changed-cell list) and after a full upload, the steps with changed cells
                  listed, the changed cells summed, the largest list, mismatches
  heads.tsv       per build, the cache's placements under LLAMA_KV_HEAD_TAIL=0 against the default (tail0 against
                  def, mask0 against mask), request by request: placements logged in each, placements whose head or
                  used-cell count differ, the head and used cells at the first that differs. A placement is one
                  LLAMA_KV_CACHE_DEBUG line printed before a micro-batch is placed (head, used cells, the highest used
                  cell + 1, each slot's lowest and highest position)
  rules.tsv       the same, one line per build and pair: where the two rules first part, how many requests differ
  settings.tsv    per build, each setting's token ids against another's (information: another head rule puts cells
                  in another place, and attention then sums them in another order)
"""
import csv
import json
import os
import re
import sys

OUT, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
SETTINGS = ["def", "tail0", "mask", "mask0"]
PAIRS = [("def", "tail0"), ("mask", "mask0")]
SLOT = re.compile(r"\| id\s+(\d+) \|| id\s+(\d+) \|")
EV = [
    ("created", re.compile(r"created context checkpoint (\d+) of (\d+)( inside the call)? \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("restored", re.compile(r"restored context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), n_past = (-?\d+)")),
    ("erased-old", re.compile(r"erasing old context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("erased-close", re.compile(r"erasing context checkpoint too close to an earlier one \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("erased-invalid", re.compile(r"erased invalidated context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+)")),
    ("full-reread", re.compile(r"forcing full prompt re-processing due to lack of cache data \(shared (\d+) tokens, no checkpoint below (-?\d+)")),
    ("text-reuse", re.compile(r"prompt ids differ from the cache at (\d+), same text: (\d+) more cached tokens kept \((\d+) -> (\d+)")),
    ("draft-restore", re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens \(restore checkpoint\)")),
    ("kq-mask-mismatch", re.compile(r"mask values differ from the host mask|: device (keep|drop), host ")),
    ("assert", re.compile(r"GGML_ASSERT|GGML_ABORT")),
]
MASK = re.compile(r"check_kq_cells: (\S+) copy (\d+): (\d+) tokens x (\d+) cells x \d+ streams \(swa type \d+, (resident|all cells uploaded), (\d+) changed cells, (\d+) old cells\): identical to the host mask")
TASK = re.compile(r"processing task, is_child = 0")
FS = re.compile(r"find_slot: stream\[(\d+)\], n = +(\d+), used = +(\d+), head = +(\d+), size = +(\d+)")
FSQ = re.compile(r"find_slot: stream\[(\d+)\] min\[(\d+)\] = +(-?\d+), max\[\d+\] = +(-?\d+)")


def read_log(log):
    """events, mask summary, and the placements split by request (segment 0 = the load's warm-up)"""
    ev, segs = [], [[]]
    n = res = full = nz = tot = mx = 0
    try:
        f = open(log, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ev, [0] * 6, segs
    with f:
        for line in f:
            if TASK.search(line):
                segs.append([])
                continue
            m = FS.search(line)
            if m:
                segs[-1].append([int(m.group(3)), int(m.group(4)), int(m.group(2)), int(m.group(5)), ()])
                continue
            m = FSQ.search(line)
            if m and segs[-1]:
                segs[-1][-1][4] += ((int(m.group(2)), int(m.group(3)), int(m.group(4))),)
                continue
            m = MASK.search(line)
            if m:
                n += 1
                res += m.group(5) == "resident"
                full += m.group(5) != "resident"
                c = int(m.group(6))
                nz += c > 0
                tot += c
                mx = max(mx, c)
                continue
            for name, rx in EV:
                m = rx.search(line)
                if m:
                    g = [x for x in m.groups() if x is not None and not x.startswith(" inside")]
                    if name == "created" and m.group(3):
                        name = "created-incall"
                    if name in ("kq-mask-mismatch", "assert"):
                        g = []
                    s = SLOT.search(line)
                    slot = (s.group(1) or s.group(2)) if s else ""
                    ev.append((name, slot, ",".join(g)))
                    break
    return ev, [n, res, full, nz, tot, mx], segs


rows, order, ev, mk, pl = {}, {}, {}, {}, {}
for d in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, d)
    if not (os.path.isdir(p) and d.count("-") == 2):
        continue
    _, s, b = d.split("-")
    e, k, segs = read_log(os.path.join(p, "server.log"))
    ev[(s, b)] = [x for x in e if x[0] != "draft-restore"]
    mk[(s, b)] = k
    reqs = []
    try:
        with open(os.path.join(p, "session.tsv"), encoding="utf-8") as g:
            for line in g:
                x = line.rstrip("\n").split("\t")
                r = dict(round=x[0], request=x[1], prompt_n=x[2], cache_n=x[3], n_gen=x[4], md5=x[5][:12],
                         drafted=x[7], accepted=x[8], stop=x[9])
                rows[(s, b, r["round"], r["request"])] = r
                reqs.append((r["round"], r["request"]))
    except FileNotFoundError:
        pass
    order[(s, b)] = reqs
    # segment i (1-based) is the i-th request sent; a request that failed before reaching a slot would shift this
    pl[(s, b)] = {"load": segs[0]}
    for i, q in enumerate(reqs):
        pl[(s, b)][q] = segs[i + 1] if i + 1 < len(segs) else None
    pl[(s, b)]["extra"] = len(segs) - 1 - len(reqs)

with open(os.path.join(DST, "replies.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "build", "round", "request", "prompt_n", "cache_n", "n_gen", "tokens_md5", "drafted",
                "accepted", "stop", "placements"])
    for (s, b), reqs in sorted(order.items()):
        for q in reqs:
            r = rows[(s, b) + q]
            seg = pl[(s, b)].get(q)
            w.writerow([s, b, q[0], q[1], r["prompt_n"], r["cache_n"], r["n_gen"], r["md5"], r["drafted"],
                        r["accepted"], r["stop"], "" if seg is None else len(seg)])

with open(os.path.join(DST, "events.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "build", "n", "slot", "event", "values"])
    for (s, b), e in sorted(ev.items()):
        for i, (k, slot, v) in enumerate(e):
            w.writerow([s, b, i + 1, slot, k, v])

with open(os.path.join(DST, "mask.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "build", "masks identical to the host's", "from the resident copy", "after a full upload",
                "steps with changed cells listed", "changed cells summed", "largest list", "mismatches"])
    for (s, b), x in sorted(mk.items()):
        if x[0] or s.startswith("mask"):
            w.writerow([s, b] + x + [sum(1 for e in ev[(s, b)] if e[0] == "kq-mask-mismatch")])


def same_pl(a, c, cancel):
    if a is None or c is None:
        return "missing"
    if cancel:  # how far the server ran past the chunks read depends on when it saw the hang-up
        k = min(len(a), len(c))
        return "same" if a[:k] == c[:k] else "DIFFERENT"
    return "same" if a == c else "DIFFERENT"


settings = [s for s in SETTINGS if any(k[0] == s for k in order)]
verd = []
with open(os.path.join(DST, "compare.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "round", "request", "tokens", "prompt_n", "cache_n", "drafts", "placements", "base_md5",
                "new_md5", "base_prompt_n/cache_n", "new_prompt_n/cache_n", "base_drafted/accepted",
                "new_drafted/accepted", "placements base/new"])
    for s in settings:
        reqs = order.get((s, "base")) or order.get((s, "new")) or []
        n = same = plsame = 0
        for q in reqs:
            a, c = rows.get((s, "base") + q), rows.get((s, "new") + q)
            if not a or not c:
                w.writerow([s, q[0], q[1], "missing"] + [""] * 11)
                n += 1
                continue
            cancel = q[1].endswith("cancel")
            t = "same" if a["md5"] and a["md5"] == c["md5"] else "DIFFERENT"
            pn = "same" if a["prompt_n"] == c["prompt_n"] else "DIFFERENT"
            cn = "same" if a["cache_n"] == c["cache_n"] else "DIFFERENT"
            dr = "same" if (a["drafted"], a["accepted"]) == (c["drafted"], c["accepted"]) else "DIFFERENT"
            if cancel:
                cn, dr = "n/a", "n/a"
            pa, pc = pl[(s, "base")].get(q), pl[(s, "new")].get(q)
            ps = same_pl(pa, pc, cancel)
            ok = t == "same" and pn == "same" and cn in ("same", "n/a") and dr in ("same", "n/a")
            n += 1
            same += ok
            plsame += ps == "same"
            w.writerow([s, q[0], q[1], t, pn, cn, dr, ps, a["md5"], c["md5"], f"{a['prompt_n']}/{a['cache_n']}",
                        f"{c['prompt_n']}/{c['cache_n']}", f"{a['drafted']}/{a['accepted']}",
                        f"{c['drafted']}/{c['accepted']}",
                        f"{'' if pa is None else len(pa)}/{'' if pc is None else len(pc)}"])
        eb, en = ev.get((s, "base"), []), ev.get((s, "new"), [])
        same_ev = eb == en
        same_mk = mk.get((s, "base")) == mk.get((s, "new"))
        same_load = pl[(s, "base")]["load"] == pl[(s, "new")]["load"] if (s, "new") in pl and (s, "base") in pl else False
        first = next((i for i, (x, y) in enumerate(zip(eb, en)) if x != y), min(len(eb), len(en)))
        cnt = lambda e, k: sum(1 for x in e if x[0] in k)
        na = cnt(eb, ("assert", "kq-mask-mismatch")) + cnt(en, ("assert", "kq-mask-mismatch"))
        verd.append([s, f"{same} of {n}", "same" if same_ev else f"DIFFERENT from event {first + 1}",
                     f"{len(eb)}/{len(en)}", f"{cnt(eb, ('restored',))}/{cnt(en, ('restored',))}",
                     f"{cnt(eb, ('full-reread',))}/{cnt(en, ('full-reread',))}",
                     f"{cnt(eb, ('assert', 'kq-mask-mismatch'))}/{cnt(en, ('assert', 'kq-mask-mismatch'))}",
                     "same" if same_mk else "DIFFERENT", f"{plsame} of {n}" + ("" if same_load else ", load DIFFERENT"),
                     "PASS" if same == n and plsame == n and same_load and same_ev and same_mk and na == 0 else "CHECK"])

with open(os.path.join(DST, "verdicts.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "requests the same (tokens, prompt_n, cache_n, drafts)", "event lists", "events base/new",
                "restores base/new", "full re-reads base/new", "asserts or mask mismatches base/new", "mask counts",
                "requests with the same placements", "verdict"])
    w.writerows(verd)

rules = []
with open(os.path.join(DST, "heads.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["build", "default", "head rule 0", "round", "request", "placements default/0", "head differs",
                "used cells differ", "first differing: head default/0", "first differing: used default/0",
                "first differing: highest used cell + 1 default/0"])
    for b in ["base", "new"]:
        for s0, s1 in PAIRS:
            if (s0, b) not in pl or (s1, b) not in pl:
                continue
            reqs = order[(s0, b)]
            nreq = ndiff = 0
            first = ""
            for q in reqs:
                a, c = pl[(s0, b)].get(q), pl[(s1, b)].get(q)
                if a is None or c is None:
                    continue
                k = min(len(a), len(c))
                hd = sum(1 for i in range(k) if a[i][1] != c[i][1])
                ud = sum(1 for i in range(k) if a[i][0] != c[i][0])
                j = next((i for i in range(k) if a[i][:3] != c[i][:3]), None)
                nreq += 1
                if hd or ud:
                    ndiff += 1
                    if not first:
                        first = f"{q[0]} {q[1]}: head {a[j][1]}/{c[j][1]}, used {a[j][0]}/{c[j][0]}"
                w.writerow([b, s0, s1, q[0], q[1], f"{len(a)}/{len(c)}", hd, ud,
                            "" if j is None else f"{a[j][1]}/{c[j][1]}", "" if j is None else f"{a[j][0]}/{c[j][0]}",
                            "" if j is None else f"{a[j][2]}/{c[j][2]}"])
            rules.append([b, s0, s1, f"{ndiff} of {nreq}", first or "none"])

with open(os.path.join(DST, "rules.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["build", "default", "head rule 0", "requests whose placements differ", "where they first part"])
    w.writerows(rules)

def tokens(s, b, q):
    """the reply's token ids from the saved JSON (a cancel: its chunks)"""
    p = os.path.join(OUT, f"ornmtp2-{s}-{b}", f"{q[0]}-{q[1]}.json")
    try:
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
    except (FileNotFoundError, ValueError):
        return []
    if "chunks" in j:
        return [t for c in j["chunks"] for t in (c.get("tokens") or [])]
    return j.get("tokens") or []


def parts(s, b, ref, q):
    a, c = tokens(ref, b, q), tokens(s, b, q)
    return next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), min(len(a), len(c)))


with open(os.path.join(DST, "settings.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["build", "setting", "against", "requests with the same token ids",
                "requests that differ (at the reply's token index, from 0)"])
    for b in ["base", "new"]:
        for s, ref in [("tail0", "def"), ("mask", "def"), ("mask0", "mask"), ("mask0", "tail0")]:
            if (s, b) not in order or (ref, b) not in order:
                continue
            n = same = 0
            diff = []
            for q in order[(ref, b)]:
                a, c = rows.get((ref, b) + q), rows.get((s, b) + q)
                if not a or not c:
                    continue
                n += 1
                if a["md5"] == c["md5"]:
                    same += 1
                else:
                    diff.append(f"{q[0]} {q[1]} @{parts(s, b, ref, q)}")
            w.writerow([b, s, ref, f"{same} of {n}", ", ".join(diff) or "none"])

with open(os.path.join(DST, "rounds.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["setting", "build", "requests whose round-2 token ids are round 1's",
                "requests whose round-2 placements are round 1's (heads and used cells)", "round-2 requests that differ in tokens"])
    for (s, b), reqs in sorted(order.items()):
        r1 = [q for q in reqs if q[0] == "r1"]
        n = st = sp = 0
        diff = []
        for q in r1:
            q2 = ("r2", q[1])
            a, c = rows.get((s, b) + q), rows.get((s, b) + q2)
            if not a or not c:
                continue
            n += 1
            if a["md5"] == c["md5"]:
                st += 1
            else:
                diff.append(q[1])
            pa, pc = pl[(s, b)].get(q), pl[(s, b)].get(q2)
            if pa is not None and pc is not None:
                if q[1].endswith("cancel"):
                    k = min(len(pa), len(pc))
                    pa, pc = pa[:k], pc[:k]
                sp += [x[:3] for x in pa] == [x[:3] for x in pc]
        w.writerow([s, b, f"{st} of {n}", f"{sp} of {n}", ", ".join(diff) or "none"])

for name in ["verdicts.tsv", "rules.tsv", "settings.tsv", "rounds.tsv"]:
    print(open(os.path.join(DST, name)).read())
