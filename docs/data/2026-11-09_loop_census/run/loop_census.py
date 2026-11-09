"""Loop census of the agent check's replies (Q1138, with Q1158's repeated-token class). Standard library only.

    python3 -I loop_census.py selftest --tokenizer BIN --vocab flint=FILE.gguf
    python3 -I loop_census.py vocab    FILE.gguf OTHER.gguf        # are two files' tokenizers the same?
    python3 -I loop_census.py tokenize --in IN --arms arms.tsv --work WORK --tokenizer BIN --vocab MODEL=FILE.gguf …
                                       [-j 8]
    python3 -I loop_census.py census   --in IN --arms arms.tsv --work WORK --tokenizer BIN --vocab MODEL=FILE.gguf …
                                       --out OUT --private PRIVATE

IN holds the agent check's result folders as bench/agent_check wrote them: `<folder>/proxy.jsonl`, the replies in
`<folder>/proxy.jsonl.bodies.gz` and the attempts in `<folder>/res/<arm>/<task>/a<N>/result.json`. arms.tsv lists the
arms to count (group, model, folder, arm): a proxy log is appended to across arms, so one arm's replies sit in several
folders and each arm is read from the one folder its row names.

A reply is the text the proxy assembled from the stream: the reasoning, the answer and each tool call. Each part is
tokenized on its own with the model's vocabulary by llama.cpp's llama-tokenize (`--no-escape`, so a backslash in code
stays a backslash); a tool call is tokenized as `{"name": "<name>", "arguments": <arguments>`. No window crosses from
one part into the next.

Three classes per reply:
1. the weight-space LP post's labels: a token repeats when the 16 tokens ending at it appeared earlier in the reply;
   a loop starts at the first point after which at least 80 % of the remaining tokens repeat. The post's "looping
   tokens" are repeats at least 64 tokens into a loop, so a reply loops when it has at least one of those. Loop tokens
   are the repeats from the loop's start to the reply's end;
2. arXiv 2606.02011's loop rescue: a 20-gram at least 4 times in the last 1,024 tokens. Loop tokens are the tokens
   from the first position where that holds to the reply's end (what a rescue at that point would cut);
3. the reply ends in one token repeated to the end. Re-tokenized text does not give the generated tokens back here
   (64 slashes come back as one token of Flash-Next's vocabulary and two of Flint's), so the test is on the text: the
   last part ends in at least 16 copies of a unit of at most 16 characters, and the unit is a single token of the
   model's vocabulary. A run of 16 equal token ids at the end of the re-tokenized part counts too.

`census` writes tables of counts and task ids to OUT (no reply text) and one row per reply to PRIVATE.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W1, FRAC1, DEEP1 = 16, 0.8, 64  # rule 1: window, share of the rest that repeats, tokens into a loop
N2, K2, WIN2 = 20, 4, 1024      # rule 2: n-gram, occurrences, window
REPS3, PERIOD3 = 16, 16         # class 3: copies of the unit, the unit's longest length in characters
TAG_RE = re.compile(r"^([^/]+)/([^/]+)/a(\d+)/t(\d+)/(\d+)$")
FINISHED = ("stop", "tool_calls")


# ---------------------------------------------------------------------------------------------------------- input

def load_arms(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_folder(root, folder):
    d = Path(root) / folder
    rows = [json.loads(l) for l in (d / "proxy.jsonl").open() if l.strip()]
    replies = {}
    with gzip.open(d / "proxy.jsonl.bodies.gz", "rt") as g:
        for l in g:
            b = json.loads(l)
            replies[b["id"]] = b.get("reply") or {}
    results = {}
    for p in d.glob("res/*/*/a*/result.json"):
        r = json.loads(p.read_text())
        results[r["tag"]] = r
    return rows, replies, results


def parts_of(reply):
    """The reply's parts in the order the model wrote them: (kind, text)."""
    out = []
    if reply.get("reasoning_content"):
        out.append(("reasoning", reply["reasoning_content"]))
    if reply.get("content"):
        out.append(("content", reply["content"]))
    for tc in reply.get("tool_calls") or []:
        if tc.get("name") or tc.get("arguments"):
            out.append(("tool", '{"name": "%s", "arguments": %s' % (tc.get("name") or "", tc.get("arguments") or "")))
    return out


def arm_replies(root, arms):
    """Yield (arm row, proxy row, reply, attempt result or None, tag fields) for every reply of every listed arm."""
    cache = {}
    for a in arms:
        if a["folder"] not in cache:
            cache[a["folder"]] = load_folder(root, a["folder"])
        rows, replies, results = cache[a["folder"]]
        mine = []
        for r in rows:
            m = TAG_RE.match(r.get("tag") or "")
            if m and m.group(1) == a["arm"]:
                mine.append((m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), r["id"], r, m))
        for task, att, try_n, turn, _, r, m in sorted(mine, key=lambda x: x[:5]):
            trace = r["tag"].rsplit("/", 1)[0]
            yield a, r, replies.get(r["id"], {}), results.get(trace), (task, att, try_n, turn)


def check_coverage(root, arms):
    """Every tagged row of every proxy log under root belongs to a listed arm, or is a copy of a listed row."""
    listed = {(a["folder"], a["arm"]) for a in arms}
    covered, extra = set(), []
    logs = sorted(p.parent for p in Path(root).rglob("proxy.jsonl.bodies.gz"))
    per_folder = {}
    for d in logs:
        folder = str(d.relative_to(root))
        rows = [json.loads(l) for l in (d / "proxy.jsonl").open() if l.strip()]
        per_folder[folder] = rows
        for r in rows:
            arm = (r.get("tag") or "").split("/")[0]
            if (folder, arm) in listed:
                covered.add((r["id"], r.get("tag")))
    copies = 0
    for folder, rows in per_folder.items():
        for r in rows:
            arm = (r.get("tag") or "").split("/")[0]
            if (folder, arm) in listed:
                continue
            if (r["id"], r.get("tag")) in covered:
                copies += 1
            else:
                extra.append((folder, r.get("tag")))
    return len(logs), len(covered), copies, extra


# ------------------------------------------------------------------------------------------------------ tokenizer

class Tokenizer:
    """llama-tokenize per text, cached in WORK/tokens_<model>.jsonl.gz by the text's sha256."""

    def __init__(self, binary, vocabs, work):
        self.binary, self.vocabs, self.work = binary, vocabs, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.cache = defaultdict(dict)
        for model in vocabs:
            p = self.path(model)
            if p.exists():
                with gzip.open(p, "rt") as g:
                    for l in g:
                        d = json.loads(l)
                        self.cache[model][d["k"]] = d["ids"]

    def path(self, model):
        return self.work / f"tokens_{model}.jsonl.gz"

    @staticmethod
    def key(text):
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()

    def run(self, model, text):
        p = subprocess.run([self.binary, "-m", self.vocabs[model], "--stdin", "--no-bos", "--no-escape", "--ids",
                            "--log-disable"], input=text.encode("utf-8", "replace"), capture_output=True)
        lines = [l for l in p.stdout.decode().splitlines() if l.startswith("[")]
        if p.returncode or not lines:
            raise RuntimeError(f"llama-tokenize failed ({p.returncode}): {p.stderr.decode()[-400:]}")
        return json.loads(lines[-1])

    def ids(self, model, text):
        k = self.key(text)
        if k not in self.cache[model]:
            self.cache[model][k] = self.run(model, text)
            with gzip.open(self.path(model), "at") as g:
                g.write(json.dumps({"k": k, "ids": self.cache[model][k]}) + "\n")
        return self.cache[model][k]

    def fill(self, model, texts, jobs):
        todo = {self.key(t): t for t in texts if self.key(t) not in self.cache[model]}
        with ThreadPoolExecutor(jobs) as ex:
            got = list(ex.map(lambda kv: (kv[0], self.run(model, kv[1])), todo.items()))
        with gzip.open(self.path(model), "at") as g:
            for k, ids in got:
                self.cache[model][k] = ids
                g.write(json.dumps({"k": k, "ids": ids}) + "\n")
        return len(todo)


def gguf_tokenizer(path):
    """sha256 of every tokenizer.ggml.* value in a GGUF file's metadata (the tensors are not read)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))

        def value(t, h):
            if t == 8:
                n = f.read(8)
                h.update(n)
                h.update(f.read(struct.unpack("<Q", n)[0]))
            elif t == 9:
                hdr = f.read(12)
                et, cnt = struct.unpack("<IQ", hdr)
                h.update(hdr)
                if et in sizes:
                    h.update(f.read(sizes[et] * cnt))
                else:
                    for _ in range(cnt):
                        value(et, h)
            else:
                h.update(f.read(sizes[t]))

        for _ in range(n_kv):
            key = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
            h = hashlib.sha256()
            value(struct.unpack("<I", f.read(4))[0], h)
            if key.startswith("tokenizer.ggml."):
                out[key] = h.hexdigest()[:16]
    return out


# ---------------------------------------------------------------------------------------------------------- rules

def rule1(ids, part):
    """The post's labels. Returns the repeat flags, the loop's start (or None), the repeats from the start on, the
    post's looping tokens, and how often the most repeated 16-token window of the loop occurs in the reply."""
    n = len(ids)
    rep = bytearray(n)
    seen = {}
    for t in range(W1 - 1, n):
        if part[t - W1 + 1] != part[t]:
            continue
        g = tuple(ids[t - W1 + 1:t + 1])
        c = seen.get(g, 0)
        if c:
            rep[t] = 1
        seen[g] = c + 1
    suf = [0] * (n + 1)
    for t in range(n - 1, -1, -1):
        suf[t] = suf[t + 1] + rep[t]
    start = next((s for s in range(n) if suf[s] and suf[s] >= FRAC1 * (n - s)), None)
    if start is None or not suf[min(n, start + DEEP1)]:
        return rep, None, 0, 0, 0
    most = max(seen[tuple(ids[t - W1 + 1:t + 1])] for t in range(start, n) if rep[t])
    return rep, start, suf[start], suf[start + DEEP1], most


def rule2(ids, part):
    """The paper's rule at every position: 1 where some 20-gram occurs at least 4 times in the last 1,024 tokens."""
    n = len(ids)
    fire = bytearray(n)
    grams = [None] * n
    counts = {}
    hot = 0
    for t in range(n):
        if t >= N2 - 1 and part[t - N2 + 1] == part[t]:
            g = grams[t] = tuple(ids[t - N2 + 1:t + 1])
            c = counts[g] = counts.get(g, 0) + 1
            if c == K2:
                hot += 1
        old = t - (WIN2 - N2 + 1)
        if old >= 0 and grams[old] is not None:
            g = grams[old]
            c = counts[g] - 1
            if c == K2 - 1:
                hot -= 1
            if c:
                counts[g] = c
            else:
                del counts[g]
        if hot:
            fire[t] = 1
    return fire


def tail_unit(text):
    """The shortest unit (at most PERIOD3 characters) the text's tail repeats at least REPS3 times: (unit, copies)."""
    n = len(text)
    for p in range(1, PERIOD3 + 1):
        i = n - p - 1
        while i >= 0 and text[i] == text[i + p]:
            i -= 1
        run = n - 1 - i
        if run >= p * REPS3:
            return text[n - run:n - run + p], run // p
    return None, 0


def tail_same(ids):
    """Length of the run of one token id at the end, and the longest such run anywhere."""
    best = cur = 0
    for i, t in enumerate(ids):
        cur = cur + 1 if i and t == ids[i - 1] else 1
        best = max(best, cur)
    return (cur if ids else 0), best


def tail_char_run(text):
    """Length of the run of one character at the text's end."""
    n = len(text)
    i = n - 1
    while i > 0 and text[i - 1] == text[n - 1]:
        i -= 1
    return n - i if n else 0


def longest_char_run(text):
    best = cur = 0
    prev = None
    for ch in text:
        cur = cur + 1 if ch == prev else 1
        prev = ch
        if cur > best and not ch.isspace():
            best = cur
    return best


def classify(parts, part_ids, single_token):
    """Everything the census records about one reply's text. parts: (kind, text); part_ids: the ids of each part."""
    ids, part = [], []
    for i, p in enumerate(part_ids):
        ids += p
        part += [i] * len(p)
    n = len(ids)
    kinds = [k for k, _ in parts]
    d = dict(n_tok=n, n_reasoning=sum(len(p) for k, p in zip(kinds, part_ids) if k == "reasoning"),
             n_content=sum(len(p) for k, p in zip(kinds, part_ids) if k == "content"),
             n_tool=sum(len(p) for k, p in zip(kinds, part_ids) if k == "tool"))
    rep, start, loop_rep, looping, most = rule1(ids, part)
    d.update(rep_tokens=sum(rep), r1=int(start is not None), r1_start="" if start is None else start,
             r1_tokens=loop_rep, r1_looping=looping, r1_most=most,
             r1_part="" if start is None else kinds[part[next(t for t in range(start, n) if rep[t])]])
    fire = rule2(ids, part)
    first = next((t for t in range(n) if fire[t]), None)
    d.update(r2=int(first is not None), r2_first="" if first is None else first, r2_fire=sum(fire),
             r2_tokens=0 if first is None else n - first, r2_part="" if first is None else kinds[part[first]])
    unit, copies, same, longest = None, 0, 0, 0
    if parts:
        unit, copies = tail_unit(parts[-1][1])
        same, _ = tail_same(part_ids[-1])
        longest = max(tail_same(p)[1] for p in part_ids)
        if unit is not None:
            turns = [unit[i:] + unit[:i] for i in range(len(unit))]
            if not any(single_token(u) for u in turns):
                unit, copies = None, 0
    c3 = int(unit is not None or same >= REPS3)
    token = unit if unit is not None else f"token id {part_ids[-1][-1]}" if c3 else ""
    d.update(c3=c3, c3_token=json.dumps(token, ensure_ascii=True) if c3 else "", c3_copies=copies if c3 else "",
             tail_same=same, tail_char_run=tail_char_run(parts[-1][1]) if parts else 0, longest_same=longest,
             char_run=max((longest_char_run(t) for _, t in parts), default=0),
             last_part=kinds[-1] if kinds else "")
    return d


# ----------------------------------------------------------------------------------------------------- self-test

def selftest(a):
    ids = list(range(300)) + [1000 + i % 10 for i in range(500)]
    part = [0] * len(ids)
    rep, start, loop_rep, looping, most = rule1(ids, part)
    assert rep[:325] == bytearray(325) and all(rep[325:]), "rule 1: repeats start 16 tokens into the second cycle"
    assert start == 207, start  # the 475 repeats are 80 % of the last 593 tokens
    assert loop_rep == 475 and looping == 475 and most == 49, (loop_rep, looping, most)
    assert rule1(list(range(300)) + list(range(100, 140)), [0] * 340)[1] is None, "rule 1: a short restatement"
    assert sum(rule1(list(range(30)) * 2, [0] * 60)[0]) == 15
    assert not any(rule1(list(range(30)) * 2, [0] * 45 + [1] * 15)[0]), "rule 1: no window crosses two parts"
    fire = rule2(ids, part)
    first = next(t for t in range(len(ids)) if fire[t])
    assert first == 300 + 19 + 30, first  # the 4th copy of the first 20-gram ends 30 tokens after its first copy
    assert not any(rule2(list(range(5000)), [0] * 5000))
    far = ([7] * 20 + list(range(100, 1200))) * 5
    assert not any(rule2(far, [0] * len(far))), "rule 2: copies more than 1,024 tokens apart"
    near = ([7] * 20 + list(range(100, 300))) * 5
    assert next(t for t, x in enumerate(rule2(near, [0] * len(near))) if x) == 3 * 220 + 19, "rule 2: the 4th copy"
    assert tail_unit("abc" + "/" * 40) == ("/", 40) and tail_unit("x" + "ab" * 20) == ("ab", 20)
    assert tail_unit("abc" + "/" * 15) == (None, 0) and tail_unit("") == (None, 0)
    assert tail_same([1, 2, 2, 2, 3, 3]) == (2, 3)
    d = classify([("content", "abc" + "/" * 40)], [[5, 6, 6]], lambda u: u == "/")
    assert d["c3"] == 1 and d["c3_token"] == '"/"' and d["c3_copies"] == 40, d
    assert classify([("content", "x" + "ab" * 20)], [[5, 6, 6]], lambda u: False)["c3"] == 0
    print("rules: ok")
    if a.tokenizer:
        tok = Tokenizer(a.tokenizer, dict(v.split("=", 1) for v in a.vocab), a.work)
        for model in tok.vocabs:
            lit, real = tok.run(model, "a\\nb"), tok.run(model, "a\nb")
            assert lit != real and len(lit) >= 3, f"{model}: --no-escape is not honoured"
            assert tok.run(model, "") == [], model
            print(f"tokenizer {model}: ok ({len(tok.run(model, '/' * 64))} tokens for 64 slashes, "
                  f"{len(tok.run(model, '/'))} for one)")


# --------------------------------------------------------------------------------------------------------- stages

def cmd_vocab(a):
    ref = gguf_tokenizer(a.files[0])
    for k, v in sorted(ref.items()):
        print(f"{k}\t{v}")
    # the keys that decide how text is split; the others (special-token ids, add_bos, the NFC normalizer) do not
    # change the ids of plain NFC text
    core = ["tokenizer.ggml." + k for k in ("model", "pre", "tokens", "token_type", "merges", "scores")]
    same = True
    for i, other in enumerate(a.files[1:], 2):
        o = gguf_tokenizer(other)
        diff = sorted(k for k in set(ref) | set(o) if ref.get(k) != o.get(k))
        split = [k for k in diff if k in core]
        print(f"file {i} against file 1: " + ("identical" if not diff else
              ("DIFFERENT VOCABULARY: " if split else "same vocabulary and merges; other keys differ: ") + ", ".join(diff)))
        same = same and not split
    sys.exit(0 if same else 1)


def cmd_tokenize(a):
    arms = load_arms(a.arms)
    tok = Tokenizer(a.tokenizer, dict(v.split("=", 1) for v in a.vocab), a.work)
    texts = defaultdict(set)
    for arm, _, reply, _, _ in arm_replies(a.inp, arms):
        for _, text in parts_of(reply):
            texts[arm["model"]].add(text)
    for model, ts in sorted(texts.items()):
        print(f"{model}: {len(ts)} texts, {tok.fill(model, ts, a.jobs)} tokenized now", flush=True)


def pct(x, y):
    return f"{100 * x / y:.2f}" if y else ""


def cmd_census(a):
    arms = load_arms(a.arms)
    tok = Tokenizer(a.tokenizer, dict(v.split("=", 1) for v in a.vocab), a.work)
    out, priv = Path(a.out), Path(a.private)
    out.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    n_logs, n_rows, n_copies, extra = check_coverage(a.inp, arms)
    if extra:
        sys.exit(f"{len(extra)} proxy rows belong to no listed arm, e.g. {extra[:5]}")
    rows, attempts = [], {}
    for arm, r, reply, res, (task, att, try_n, turn) in arm_replies(a.inp, arms):
        parts = parts_of(reply)
        model = arm["model"]
        d = classify(parts, [tok.ids(model, t) for _, t in parts], lambda u: len(tok.ids(model, u)) == 1)
        finish = r.get("finish_reason") or ""
        cut = ("length" if finish == "length" else "client_gone" if r.get("client_gone") else
               "server_error" if r.get("upstream_error") or r.get("status") != 200 else "" if finish in FINISHED
               else "other")
        if r.get("predicted_n") and r.get("predicted_ms"):
            ms = r["predicted_ms"] / r["predicted_n"]
        elif d["n_tok"] and r.get("t_first_token") is not None and r.get("t_end"):
            ms = 1000 * (r["t_end"] - r["t_first_token"]) / d["n_tok"]
        else:
            ms = 0.0
        key = (arm["folder"], arm["arm"], task, att, try_n)
        if res is not None:
            attempts[key] = res
        row = dict(group=arm["group"], model=model, folder=arm["folder"], arm=arm["arm"], task=task, attempt=att,
                   try_n=try_n, turn=turn, id=r["id"], finish=finish, cut=cut,
                   predicted_n="" if r.get("predicted_n") is None else r["predicted_n"], **d,
                   ms_per_token=round(ms, 3), r1_s=round(d["r1_tokens"] * ms / 1000, 2),
                   r2_s=round(d["r2_tokens"] * ms / 1000, 2),
                   stream_s="" if r.get("t_first_token") is None else round(r["t_end"] - r["t_first_token"], 2),
                   scored=int(res is not None and res.get("status") == "scored"),
                   attempt_runaway=(res or {}).get("runaway", ""), attempt_pass=(res or {}).get("success", ""),
                   attempt_wall_s=(res or {}).get("wall_s", ""))
        rows.append(row)
    if len(rows) != n_rows:
        sys.exit(f"{n_rows} proxy rows in the listed arms, {len(rows)} read: a tag does not parse")
    # replies after a class-3 reply in the same arm: a NaN in the state would leave every one of them class 3 too
    by_arm_order = defaultdict(list)
    for row in rows:
        by_arm_order[(row["folder"], row["arm"])].append(row)
    for lst in by_arm_order.values():
        lst.sort(key=lambda x: x["id"])
        for i, row in enumerate(lst):
            later = [x for x in lst[i + 1:] if x["n_tok"]]
            row["later_replies"] = len(later) if row["c3"] else ""
            row["later_class3"] = sum(x["c3"] for x in later) if row["c3"] else ""

    def write(path, table, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(table)

    write(priv / "replies.tsv", rows, list(rows[0]))
    ident = ["group", "model", "folder", "arm", "task", "attempt", "try_n", "turn"]

    def summary(sel, att):
        scored = [r for r in sel if r["scored"]]
        fin = [r for r in sel if r["predicted_n"]]
        wall = sum(x["wall_s"] for x in att if x.get("status") == "scored")
        tokens = sum(r["n_tok"] for r in sel)
        cutr = [r for r in sel if r["cut"] == "client_gone"]
        capped = sum(x["wall_s"] for x in att if x.get("status") == "scored" and x.get("runaway"))
        return dict(
            replies=sum(1 for r in sel if r["n_tok"]), tokens=tokens,
            tokens_per_server_count=(f"{sum(r['n_tok'] for r in fin) / sum(r['predicted_n'] for r in fin):.4f}"
                                     if fin else ""),
            rule1_replies=sum(r["r1"] for r in sel), rule2_replies=sum(r["r2"] for r in sel),
            both_replies=sum(r["r1"] and r["r2"] for r in sel), class3_replies=sum(r["c3"] for r in sel),
            rule1_finished=sum(r["r1"] for r in sel if r["finish"] in FINISHED),
            rule2_finished=sum(r["r2"] for r in sel if r["finish"] in FINISHED),
            repeat_tokens=sum(r["rep_tokens"] for r in sel),
            looping_tokens=sum(r["r1_looping"] for r in sel),
            repeat_token_pct=pct(sum(r["rep_tokens"] for r in sel), tokens),
            rule1_tokens=sum(r["r1_tokens"] for r in sel), rule2_tokens=sum(r["r2_tokens"] for r in sel),
            rule2_fire_tokens=sum(r["r2_fire"] for r in sel),
            rule2_fire_pct=pct(sum(r["r2_fire"] for r in sel), tokens),
            rule2_in_passed=sum(r["r2"] for r in sel if r["attempt_pass"] == 1),
            rule1_token_pct=pct(sum(r["r1_tokens"] for r in sel), tokens),
            rule2_token_pct=pct(sum(r["r2_tokens"] for r in sel), tokens),
            attempts=sum(1 for x in att if x.get("status") == "scored"),
            passed=sum(1 for x in att if x.get("status") == "scored" and x.get("success")),
            runaway_attempts=sum(1 for x in att if x.get("status") == "scored" and x.get("runaway")),
            runaway_s=round(capped, 1), runaway_time_pct=pct(capped, wall),
            task_s=round(wall, 1), rule1_s=round(sum(r["r1_s"] for r in scored), 1),
            rule2_s=round(sum(r["r2_s"] for r in scored), 1),
            rule1_time_pct=pct(sum(r["r1_s"] for r in scored), wall),
            rule2_time_pct=pct(sum(r["r2_s"] for r in scored), wall),
            cut_replies=len(cutr), cut_rule1=sum(r["r1"] for r in cutr), cut_rule2=sum(r["r2"] for r in cutr),
            cut_class3=sum(r["c3"] for r in cutr), cut_tokens=sum(r["n_tok"] for r in cutr),
            cut_repeat_tokens=sum(r["rep_tokens"] for r in cutr),
            length_stops=sum(1 for r in sel if r["cut"] == "length"),
            server_errors=sum(1 for r in sel if r["cut"] == "server_error"),
            longest_tail_same=max((r["tail_same"] for r in sel), default=0),
            longest_tail_char_run=max((r["tail_char_run"] for r in sel), default=0))

    def grouped(keys):
        sel, att = defaultdict(list), defaultdict(list)
        for r in rows:
            sel[tuple(r[k] for k in keys)].append(r)
        for (folder, arm, task, _, _), res in attempts.items():
            meta = next(x for x in arms if x["folder"] == folder and x["arm"] == arm)
            full = dict(meta, task=task)
            att[tuple(full[k] for k in keys)].append(res)
        return [dict(zip(keys, k), **summary(v, att.get(k, []))) for k, v in sel.items()]

    cols = list(summary([], []))
    by_arm = grouped(["group", "model", "folder", "arm"])
    by_model = sorted(grouped(["group", "model"]), key=lambda x: (x["group"], x["model"]))
    by_task = sorted(grouped(["group", "model", "task"]), key=lambda x: (x["group"], x["model"], x["task"]))
    write(out / "by_arm.tsv", by_arm, ["group", "model", "folder", "arm"] + cols)
    write(out / "by_model.tsv", by_model, ["group", "model"] + cols)
    write(out / "by_task.tsv", by_task, ["group", "model", "task"] + cols)
    loops = [r for r in rows if r["r1"] or r["r2"] or r["c3"]]
    write(out / "loops.tsv", loops, ident + ["finish", "cut", "n_tok", "rep_tokens", "r1", "r1_start", "r1_tokens",
                                             "r1_looping", "r1_most", "r1_part", "r2", "r2_first", "r2_fire",
                                             "r2_tokens", "r2_part", "c3",
                                             "r1_s", "r2_s", "attempt_runaway", "attempt_pass"])
    write(out / "class3.tsv", [r for r in rows if r["c3"]],
          ident + ["finish", "cut", "n_tok", "c3_token", "c3_copies", "tail_same", "last_part", "later_replies",
                   "later_class3"])
    # every reply that did not end on its own, and every attempt the agent check stopped at a cap
    runaways = []
    for r in rows:
        if r["cut"] or r["attempt_runaway"]:
            last = max(x["turn"] for x in rows if all(x[k] == r[k] for k in ident[:7]))
            if r["cut"] or r["turn"] == last:
                att_rows = [x for x in rows if all(x[k] == r[k] for k in ident[:7])]
                runaways.append(dict(r, attempt_tokens=sum(x["n_tok"] for x in att_rows),
                                     attempt_rule1_s=round(sum(x["r1_s"] for x in att_rows), 1),
                                     attempt_rule2_s=round(sum(x["r2_s"] for x in att_rows), 1)))
    write(out / "runaways.tsv", runaways,
          ident + ["finish", "cut", "attempt_runaway", "attempt_pass", "attempt_wall_s", "stream_s", "n_tok",
                   "n_reasoning", "n_content", "n_tool", "last_part", "rep_tokens", "r1", "r1_start", "r1_tokens",
                   "r1_most", "r2", "r2_first", "r2_fire", "r2_tokens", "c3", "r1_s", "r2_s", "attempt_tokens", "attempt_rule1_s",
                   "attempt_rule2_s"])
    with open(out / "summary.md", "w") as f:
        f.write(f"{n_logs} proxy logs, {n_rows} replies in {len(arms)} arms ({n_copies} more rows are copies of "
                f"those in a later log), {len(attempts)} attempts with a result.\n\n")
        show = ["replies", "tokens", "repeat_token_pct", "rule1_replies", "rule2_replies", "rule2_finished",
                "class3_replies", "rule1_token_pct", "rule2_fire_pct", "rule2_token_pct", "rule1_time_pct",
                "rule2_time_pct", "attempts", "runaway_attempts", "runaway_time_pct", "cut_replies", "cut_rule1",
                "cut_rule2",
                "cut_class3", "length_stops", "longest_tail_same", "longest_tail_char_run"]
        for title, table, keys in (("Per group and model", by_model, ["group", "model"]),
                                   ("Per arm", by_arm, ["group", "model", "folder", "arm"])):
            f.write(f"## {title}\n\n| " + " | ".join(keys + show) + " |\n|" + "---|" * len(keys + show) + "\n")
            for x in table:
                f.write("| " + " | ".join(str(x[k]) for k in keys + show) + " |\n")
            f.write("\n")
    print((out / "summary.md").read_text().split("\n## Per arm")[0])
    print(f"class 3: {sum(r['c3'] for r in rows)} replies; cut replies: "
          f"{Counter(r['cut'] for r in rows if r['cut']).most_common()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocab")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_vocab)
    for name, fn in (("selftest", selftest), ("tokenize", cmd_tokenize), ("census", cmd_census)):
        p = sub.add_parser(name)
        p.add_argument("--tokenizer", help="llama.cpp's llama-tokenize (a CPU build)")
        p.add_argument("--vocab", action="append", default=[], metavar="MODEL=FILE.gguf")
        p.add_argument("--work", default="work", help="token cache")
        if name != "selftest":
            p.add_argument("--in", dest="inp", required=True)
            p.add_argument("--arms", required=True)
        if name == "tokenize":
            p.add_argument("-j", "--jobs", type=int, default=8)
        if name == "census":
            p.add_argument("--out", required=True)
            p.add_argument("--private", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
