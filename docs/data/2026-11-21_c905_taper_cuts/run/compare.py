# C905: compares the server runs of run.sh (step server). Per run: client.py's jsonl (token ids, top-3 probabilities,
# prompt_n, cache_n, restored checkpoint) and the "ubatch taper:" lines of its server log (one per prompt call, in
# order). Pairs as in PAIRS; for each call of a run with a taper the pieces under the floor (9 tokens) that are not a
# whole segment between the server's in-call cuts (4 + 512 and 4 tokens before the call's end).
# Usage: python3 -I compare.py <dir> > server.tsv   (writes <dir>/calls.tsv too)
import json
import os
import re
import sys

d = sys.argv[1]
MIN = 9
N_UBATCH = 512

# run, reference run, what has to hold
PAIRS = [
    ("head-off", "base-off", "same", "taper unset: replies, probabilities, prompt_n, cache_n, no taper line"),
    ("head-v5", "base-v5", "same", "short=1400:64 (cuts unset): replies, probabilities, counts and every call's pieces the base's"),
    ("head-v5c", "head-v5", "tokens", "short=1400:64,cuts=1 against short=1400:64: the same replies and counts"),
    ("head-v5c-req", "head-v5c", "same", "cuts=1 sent per request against set on the server: everything the same"),
    ("head-v5", "head-off", "tokens", "short=1400:64 against the taper unset (not part of the verdict: no change of this branch)"),
]
INFO = {("head-v5", "head-off")}


def load_rows(name):
    path = os.path.join(d, name + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(x) for x in f]


ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$")


def load_calls(name):
    path = os.path.join(d, name + ".log")
    if not os.path.exists(path):
        return None
    res = []
    for line in open(path, errors="replace"):
        m = pat.search(ansi.sub("", line))
        if m:
            res.append((int(m.group(1)), m.group(5).strip()))
    return res


def sizes(plan):
    out = []
    for t in plan.split():
        n, _, c = t.partition("x")
        out += [int(n)] * (int(c) if c else 1)
    return out


def bad_pieces(n, plan):
    # pieces under MIN that are not a whole segment between the cuts (n - 516, n - 4) and the call's ends
    bounds = sorted({0, n} | {n - off for off in (4 + N_UBATCH, 4) if n > off})
    bad, pos = [], 0
    for s in sizes(plan):
        if s < MIN:
            whole = pos in bounds and pos + s in bounds and not any(pos < b < pos + s for b in bounds)
            if not whole:
                bad.append(f"{s}@{pos}")
        pos += s
    return bad


runs = sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl"))
rows = {r: load_rows(r) for r in runs}
calls = {r: load_calls(r) for r in runs}

with open(os.path.join(d, "calls.tsv"), "w") as f:
    f.write("run\tcall\tcall_tokens\tpieces\tpieces_under_9_not_a_whole_segment\n")
    for r in runs:
        for i, (n, plan) in enumerate(calls[r] or []):
            b = bad_pieces(n, plan)
            f.write(f"{r}\t{i}\t{n}\t{plan}\t{' '.join(b) if b else '-'}\n")

print("run\treference\trule\tverdict\tdetail")
fail = 0
for run, ref, rule, what in PAIRS:
    a, b = rows.get(run), rows.get(ref)
    if a is None or b is None:
        print(f"{run}\t{ref}\t{what}\tMISSING\t")
        fail = 1
        continue
    errs = [x["req"] for x in a + b if "error" in x]
    keys = ["tokens", "prompt_n", "cache_n", "cache_ckpt_n"] + (["probs"] if rule == "same" else [])
    diff = [x["req"] for x, y in zip(a, b) if any(x.get(k) != y.get(k) for k in keys)]
    detail = f"{len(a)} requests"
    ok = not errs and not diff and len(a) == len(b)
    if rule == "tokens":
        # probabilities: the largest difference of a top-3 probability
        dmax = 0.0
        for x, y in zip(a, b):
            for px, py in zip(x.get("probs") or [], y.get("probs") or []):
                for tx, ty in zip(px.get("top_probs", []), py.get("top_probs", [])):
                    if tx.get("id") == ty.get("id"):
                        dmax = max(dmax, abs(tx.get("prob", 0.0) - ty.get("prob", 0.0)))
                    else:
                        dmax = max(dmax, 1.0)
        same_p = all(x.get("probs") == y.get("probs") for x, y in zip(a, b))
        detail += f"; top-3 probabilities {'the same' if same_p else f'largest difference {dmax:.3e}'}"
    if rule == "same":
        ca, cb = calls[run] or [], calls[ref] or []
        if ca != cb:
            ok = False
            detail += f"; taper lines differ ({len(ca)} against {len(cb)})"
        else:
            detail += f"; {len(ca)} taper lines the same"
    if errs:
        detail += "; errors: " + ",".join(errs)
    if diff:
        detail += "; differ: " + ",".join(diff)
    info = (run, ref) in INFO
    print(f"{run}\t{ref}\t{what}\t{('OK' if ok else 'DIFFERS') if info else ('OK' if ok else 'FAIL')}\t{detail}")
    fail |= ok is False and not info

# the floor per run
for r in runs:
    cs = calls[r] or []
    n_bad = sum(1 for n, p in cs if bad_pieces(n, p))
    print(f"{r}\t-\tcalls with a piece under 9 tokens that is not a whole segment\tINFO\t{n_bad} of {len(cs)} calls")
c = calls.get("head-v5c") or []
cr = calls.get("head-v5c-req") or []
okc = bool(c) and not any(bad_pieces(n, p) for n, p in c + cr)
print(f"head-v5c\t-\tcuts=1: no piece under 9 tokens but whole segments the cuts make\t{'OK' if okc else 'FAIL'}\t{len(c)} and {len(cr)} calls")
fail |= not okc
print(f"all\t-\tverdict\t{'PASS' if not fail else 'FAIL'}\t")
