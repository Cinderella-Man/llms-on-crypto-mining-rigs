# C936 (Q1675): the tables from pipe_mtp.sh's output folder. Usage: python3 -I tables.py <OUT folder> <data folder>
# Writes loads.tsv, mib.tsv, replies.tsv, parts.tsv and guesses.tsv into the data folder and prints them.
import json
import os
import re
import sys
from datetime import datetime

src, dst = sys.argv[1], sys.argv[2]
CONFIGS = ["preset", "plain", "d1", "d2", "d3", "walk", "grammar", "plain_probs"]
REQS = ["reply256", "reply256_probs", "reply512", "schema"]
present = [c for c in CONFIGS if os.path.isdir(os.path.join(src, c))]


def read(c, name, default=""):
    p = os.path.join(src, c, name)
    return open(p, encoding="utf-8", errors="replace").read().strip() if os.path.exists(p) else default


def requests(c):
    p = os.path.join(src, c, "requests.jsonl")
    rows = [json.loads(line) for line in open(p)] if os.path.exists(p) else []
    return {r["req"]: r for r in rows if r["req"] != "prompts"}


def blocks(c):
    """One block per finished request, in order: the task's print_timing lines."""
    out, cur, task = [], None, None
    log = read(c, "server.log").splitlines()
    for line in log:
        if "print_timing" in line and "prompt eval time" in line:
            task = re.search(r"\| task (\d+) \|", line).group(1)
            cur = {"task": task, "lines": []}
            out.append(cur)
        if cur is not None and "print_timing" in line and f"| task {task} |" in line:
            cur["lines"].append(line.split(f"| task {task} |", 1)[1].strip())
    return out, log


def stat(lines, key):
    for line in lines:
        if line.startswith(key):
            return line
    return ""


def write(name, header, rows):
    with open(os.path.join(dst, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"== {name}")
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(x) for x in r))


def ts(c, name):
    s = read(c, name)
    return datetime.fromisoformat(s) if s else None


# loads.tsv
rows = []
for c in present:
    cmd = read(c, "command.txt")
    extra = " ".join(re.findall(r"(--pipe-[a-z-]+ \S+|LLAMA_PIPE_GRAMMAR=\S+)", cmd)) or "-"
    if c == "plain":
        extra = "draft off"
    a, b = ts(c, "t_launch"), ts(c, "t_loaded")
    _, log = blocks(c)
    pipe = [l.split("srv ", 1)[-1].strip() for l in log if "pipelined speculation" in l or "--pipe-guess" in l
            or "pipe slots" in l]
    rows.append([c, extra, read(c, "status"), f"{(b - a).total_seconds():.0f}" if a and b else "-",
                 " / ".join(re.sub(r"^\S+\s+", "", p) for p in pipe) or "-"])
write("loads.tsv", ["config", "added to the preset", "status", "load s", "server's pipe lines"], rows)

# mib.tsv
rows = []
for c in present:
    for phase in ("load", "end"):
        t = read(c, f"mib_{phase}.csv")
        if not t:
            continue
        cards = [l.split(",") for l in t.splitlines() if l[:1].isdigit()]
        host = re.search(r"(\d+) kB", t)
        rows.append([c, phase] + [x[1].strip() for x in cards] + [str(int(host.group(1)) // 1024) if host else "-"])
write("mib.tsv", ["config", "when"] + [f"card {i} MiB" for i in range(9)] + ["host MiB available"], rows)

# replies.tsv and parts.tsv
R = {c: requests(c) for c in present}


def part(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def top2(c, i, q):
    """The top two tokens at reply index i of request q on config c's path: from its own copy with token
    probabilities (reply256_probs), or, for plain, from the plain_probs load when that load gave plain's tokens."""
    cands = [R.get(c, {}).get(q + "_probs"), R.get(c, {}).get(q)]
    if c == "plain":
        cands += [R.get("plain_probs", {}).get(q)]
    own = R.get(c, {}).get(q)
    for r in cands:
        if r and "top2" in r and i < len(r["top2"]) and own and r.get("tokens", [])[:i + 1] == own["tokens"][:i + 1]:
            return " ".join(f"{tid}:{json.dumps(tok)}:{lp:.4f}" for tid, tok, lp in r["top2"][i])
    return "-"


rows, prow = [], []
for c in present:
    for q in REQS:
        r = R[c].get(q)
        if r is None:
            continue
        toks = r.get("tokens")
        vs = []
        for ref in ("preset", "plain"):
            rr = R.get(ref, {}).get("reply256" if q == "reply256_probs" else q)
            p = part(toks, rr.get("tokens") if rr else None)
            vs.append("-" if p is None else ("same" if p == -1 else f"parts at {p}"))
            if p is not None and p >= 0 and q in ("reply256", "reply512", "schema") and c != ref:
                prow.append([c, q, ref, p, toks[p] if p < len(toks) else "end", rr["tokens"][p]
                             if p < len(rr["tokens"]) else "end", top2(ref, p, q), top2(c, p, q)])
        t = r.get("timings") or {}
        rows.append([c, q, r.get("n", "-"), r.get("stop_type", "-"), f"{r['t_end'] - r['t_start']:.1f}",
                     r.get("error", "-"), vs[0], vs[1],
                     f"{t['predicted_per_second']:.2f}" if "predicted_per_second" in t else "-",
                     (r.get("content") or "")[:60].replace("\n", " ").replace("\t", " ")])
write("replies.tsv", ["config", "request", "tokens", "stop", "client s", "error", "vs preset", "vs plain",
                      "tok/s (server)", "text start"], rows)
write("parts.tsv", ["config", "request", "against", "index", "its token", "the reference's token",
                    "the reference's top two (id:text:logprob)", "this config's top two"], prow)

# guesses.tsv: per finished request, in order (the blocks follow the requests that finished)
# The server's "graphs reused" and "pipe reallocs" counts run over the load: each request's own is the difference.
rows = []
for c in present:
    bl, _ = blocks(c)
    done = [q for q in REQS if q in R[c] and "error" not in R[c][q]]
    prev = [0, 0, 0]
    for q, b in zip(done, bl):
        g = re.search(r"graphs reused =\s+(\d+)", stat(b["lines"], "graphs reused"))
        pr = re.search(r"pipe reallocs =\s+(\d+)\s+/\s+(\d+)", stat(b["lines"], "pipe reallocs"))
        cum = [int(g.group(1)) if g else 0] + ([int(pr.group(1)), int(pr.group(2))] if pr else [0, 0])
        own = [x - y for x, y in zip(cum, prev)]
        prev = cum
        pipe = stat(b["lines"], "pipelined =")
        m = re.search(r"(\d+) guesses, (\d+) right, (\d+) wrong, (\d+) from the store.*?(\d+) grammar rejections", pipe)
        mtp = re.search(r"pipe mtp\s+=\s+(\d+) rows, ([\d.]+) ms per row", stat(b["lines"], "pipe mtp"))
        acc = re.search(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)",
                        stat(b["lines"], "draft acceptance"))
        rows.append([c, q] + (list(m.groups()) if m else ["-"] * 5)
                    + (list(mtp.groups()) if mtp else ["-", "-"])
                    + ([acc.group(2), acc.group(3)] if acc else ["-", "-"])
                    + [own[0], f"{own[1]} / {own[2]}" if pr else "-"])
write("guesses.tsv", ["config", "request", "guesses", "right", "wrong", "from the store", "grammar rejections",
                      "MTP rows", "ms per MTP row", "draft accepted", "draft generated", "graphs reused",
                      "pipe re-plans (same / changed cards)"], rows)
