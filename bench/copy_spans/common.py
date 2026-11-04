"""Shared pieces of the copy-span analysis (standard library only).

- load_run(): the agent check's proxy log (proxy.jsonl and proxy.jsonl.bodies.gz, bench/agent_check/proxy.py).
- render_prompt(): the prompt llama-server builds from a request (the chat template of Qwen3.8-Flash-Next and its
  fine-tunes, written out by hand; check_render.py compares it with the Jinja template byte for byte).
- build_output(): the text the model generated for a reply, rebuilt from the server's parsed reply (reasoning,
  answer, tool calls in the model's <tool_call> format), with the region of every byte (reasoning, answer, call
  scaffolding, each tool argument) and the markers a copy can start after.
- Vocab: token id -> bytes, read from a GGUF; tokenize(): llama-tokenize driven in batches.
- build_sam() / match_profile(): longest earlier match of the text's suffix (suffix automaton over the context, a
  hash index over the response's own earlier text).
"""
import gzip
import json
import os
import struct
import subprocess
import sys
from collections import Counter

XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
         "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
TOOL_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOL_FORMAT = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")

TOJSON_SEPARATORS = (", ", ": ")   # llama.cpp's template engine prints JSON the way Python's json.dumps does


# ---------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------

def load_run(path, run_name):
    """Every answered chat request of one agent-check run: dict(run, arm, task, attempt, try_, turn, session, id,
    body, reply, finish, predicted_n, prompt_tokens)."""
    meta = {}
    with open(os.path.join(path, "proxy.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["id"]] = r
    out = []
    with gzip.open(os.path.join(path, "proxy.jsonl.bodies.gz"), "rt", encoding="utf-8") as g:
        for line in g:
            d = json.loads(line)
            r = meta.get(d["id"])
            if r is None or not isinstance(d.get("body"), dict) or not isinstance(d.get("reply"), dict):
                continue
            if r.get("status") != 200 or r.get("finish_reason") not in ("stop", "tool_calls", "length"):
                continue
            tag = (r.get("tag") or "").split("/")
            if len(tag) != 5:
                continue
            arm, task, attempt, try_, turn = tag
            usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
            out.append({
                "run": run_name, "arm": arm, "task": task, "attempt": attempt, "try_": try_, "turn": int(turn),
                "session": "/".join([run_name, arm, task, attempt, try_]), "id": r["id"], "body": d["body"],
                "reply": d["reply"], "finish": r["finish_reason"], "predicted_n": r.get("predicted_n") or 0,
                "prompt_tokens": usage.get("prompt_tokens") or ((r.get("prompt_n") or 0) + (r.get("cache_n") or 0)),
            })
    return out


# ---------------------------------------------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------------------------------------------

def tojson(v):
    return json.dumps(v, ensure_ascii=False, separators=TOJSON_SEPARATORS)


def render_content(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(c)


def parse_arguments(a):
    """A tool call's arguments as a dict (the server hands the template an object); None for an empty call."""
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        if a == "":
            return {}
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {"_raw": a}
        except ValueError:
            return {"_raw": a}
    return {}


def render_prompt(body, generation_prompt=True):
    """The prompt text the server feeds the model for this request, and the strings a response could quote that the
    prompt holds (user messages and tool results, trimmed, for the JSON-escaped variant)."""
    msgs, tools = body["messages"], body.get("tools")
    parts, quotable = [], []
    if tools:
        parts.append("<|im_start|>system\n" + XHIGH + "\n\n" + TOOL_HEAD)
        for t in tools:
            parts.append("\n" + tojson(t))
        parts.append("\n</tools>" + TOOL_FORMAT)
        if msgs[0]["role"] == "system":
            c = render_content(msgs[0].get("content")).strip()
            if c:
                parts.append("\n\n" + c)
        parts.append("<|im_end|>\n")
    else:
        c = render_content(msgs[0].get("content")).strip() if msgs[0]["role"] == "system" else ""
        if c:
            parts.append("<|im_start|>system\n" + XHIGH + "\n\n" + c + "<|im_end|>\n")
        else:
            parts.append("<|im_start|>system\n" + XHIGH + "<|im_end|>\n")
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = m["role"]
        content = render_content(m.get("content")).strip()
        if role == "system":
            continue
        if role == "user":
            parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
            quotable.append(content)
        elif role == "assistant":
            rc = m.get("reasoning_content")
            rc = rc.strip() if isinstance(rc, str) else ""
            parts.append("<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content)
            for k, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function", tc)
                if k == 0:
                    parts.append("\n\n" if content else "")
                else:
                    parts.append("\n")
                parts.append("<tool_call>\n<function=" + fn["name"] + ">\n")
                args = parse_arguments(fn.get("arguments"))
                for key, val in args.items():
                    parts.append("<parameter=" + key + ">\n" + (val if isinstance(val, str) else tojson(val))
                                 + "\n</parameter>\n")
                parts.append("</function>\n</tool_call>")
            parts.append("<|im_end|>\n")
        elif role == "tool":
            if i > 0 and msgs[i - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if i == n - 1 or msgs[i + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
            quotable.append(content)
        else:
            raise ValueError("unexpected role " + role)
    if generation_prompt:
        parts.append("<|im_start|>assistant\n<think>\n")
    return "".join(parts), quotable


# ---------------------------------------------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------------------------------------------

class OutBuilder:
    """Builds the generated text from pieces labelled with a region; keeps byte offsets of region starts and the
    markers a copy can start after: (byte offset where the marker ends, kind)."""

    def __init__(self):
        self.texts, self.starts, self.regions, self.markers = [], [], [], []
        self.blen = 0

    def add(self, region, text):
        if not text:
            return
        if self.regions and self.regions[-1] == region:
            self.texts[-1] += text
        else:
            self.texts.append(text)
            self.starts.append(self.blen)
            self.regions.append(region)
        self.blen += len(text.encode("utf-8"))

    def mark(self, kind):
        self.markers.append((self.blen, kind))

    def json_value(self, region, v):
        """A non-string tool argument the way the template prints it (json.dumps style), with the string leaves
        labelled region + '.' + key (the key of the object they sit in) and the rest region + '.scaf'."""
        self._json(region, v, None)

    def _json(self, region, v, key):
        scaf = region + ".scaf"
        if isinstance(v, dict):
            self.add(scaf, "{")
            for j, (k, x) in enumerate(v.items()):
                if j:
                    self.add(scaf, ", ")
                self.add(scaf, json.dumps(k, ensure_ascii=False) + ": ")
                self._json(region, x, k)
            self.add(scaf, "}")
        elif isinstance(v, list):
            self.add(scaf, "[")
            for j, x in enumerate(v):
                if j:
                    self.add(scaf, ", ")
                self._json(region, x, key)
            self.add(scaf, "]")
        elif isinstance(v, str):
            self.add(scaf, '"')
            self.mark("jstr:" + (key or ""))
            self.add(region + "." + (key or "str"), json.dumps(v, ensure_ascii=False)[1:-1])
            self.add(scaf, '"')
        else:
            self.add(scaf, json.dumps(v, ensure_ascii=False))


def build_output(reply, finish):
    """The generated text of one reply and its labelled regions. Returns (text, starts, regions, markers)."""
    b = OutBuilder()
    reasoning = reply.get("reasoning_content") or ""
    content = reply.get("content") or ""
    tcs = reply.get("tool_calls") or []
    b.add("reasoning", reasoning)
    b.add("struct", "</think>\n\n")
    b.mark("think_close")
    b.add("content", content)
    for k, tc in enumerate(tcs):
        fn = tc.get("function", tc)
        name = fn.get("name", "?")
        b.add("struct", ("\n\n" if (k == 0 and content.strip()) else "\n" if k else "") + "<tool_call>\n<function="
              + name + ">\n")
        b.mark("call")
        for key, val in parse_arguments(fn.get("arguments")).items():
            b.add("struct", "<parameter=" + key + ">\n")
            b.mark("param")
            region = "arg:" + name + "." + key
            if isinstance(val, str):
                b.add(region, val)
            else:
                b.json_value(region, val)
            b.add("struct", "\n</parameter>\n")
        b.add("struct", "</function>\n</tool_call>")
    if finish in ("stop", "tool_calls"):
        b.add("struct", "<|im_end|>")
    return "".join(b.texts), b.starts, b.regions, b.markers


# ---------------------------------------------------------------------------------------------------------------
# Vocabulary and tokenizer
# ---------------------------------------------------------------------------------------------------------------

def read_gguf_kv(path, keys=None):
    """The key-value section of a GGUF file as a dict; with `keys`, only those (others are skipped, not stored)."""
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    def rstr():
        n, = rd("<Q")
        return f.read(n).decode("utf-8", "replace")

    types = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, = rd("<I")
            n, = rd("<Q")
            return [rval(et) for _ in range(n)]
        return rd(types[t])[0]

    if f.read(4) != b"GGUF":
        raise ValueError(path + " is not a GGUF file")
    rd("<I")
    rd("<Q")
    nkv, = rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        t, = rd("<I")
        v = rval(t)
        if keys is None or k in keys:
            kv[k] = v
    f.close()
    return kv


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_vocab(gguf_path):
    """token id -> the bytes the token stands for (byte-level BPE; control and user-defined tokens are literal)."""
    kv = read_gguf_kv(gguf_path, {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})
    toks, types = kv["tokenizer.ggml.tokens"], kv.get("tokenizer.ggml.token_type") or [1] * len(kv["tokenizer.ggml.tokens"])
    dec = {c: b for b, c in bytes_to_unicode().items()}
    table = []
    for t, ty in zip(toks, types):
        if ty in (3, 4):
            table.append(t.encode("utf-8"))
        else:
            try:
                table.append(bytes(dec[c] for c in t))
            except KeyError:
                table.append(t.encode("utf-8"))
    return table


def tokenize(texts, tok_bin, vocab_gguf, table, batch_bytes=3_000_000, log=sys.stderr):
    """Token ids of every text, by llama-tokenize (special tokens parsed, no BOS). Texts go through the tool in
    batches joined by a control token that none of them contains; each result is checked to decode back to its text."""
    sentinel = None
    for lit in (b"<|fim_pad|>", b"<|repo_name|>", b"<|file_sep|>", b"<|video_pad|>"):
        if lit in table and not any(lit.decode() in t for t in texts):
            sentinel = lit
            break
    if sentinel is None:
        raise RuntimeError("no free separator token")
    sid = table.index(sentinel)
    sent = sentinel.decode()
    out = [None] * len(texts)
    i = 0
    while i < len(texts):
        j, size = i, 0
        while j < len(texts) and (j == i or size + len(texts[j]) < batch_bytes):
            size += len(texts[j]) + len(sent)
            j += 1
        data = sent.join(texts[i:j]).encode("utf-8")
        p = subprocess.run([tok_bin, "-m", vocab_gguf, "--stdin", "--ids", "--no-bos", "--no-escape", "--log-disable"],
                           input=data, capture_output=True, check=True)
        line = [x for x in p.stdout.decode("utf-8", "replace").splitlines() if x.startswith("[")][-1]
        ids = json.loads(line)
        cur, k = [], i
        for t in ids:
            if t == sid:
                out[k] = cur
                cur, k = [], k + 1
            else:
                cur.append(t)
        out[k] = cur
        if k != j - 1:
            raise RuntimeError("batch %d-%d: %d texts came back as %d" % (i, j, j - i, k - i + 1))
        for x in range(i, j):
            if b"".join(table[t] for t in out[x]) != texts[x].encode("utf-8"):
                raise RuntimeError("text %d does not decode back to itself" % x)
        print("tokenized texts %d-%d (%d tokens)" % (i, j, len(ids)), file=log)
        i = j
    return out


# ---------------------------------------------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------------------------------------------

def build_sam(seq):
    """Suffix automaton of seq: (next, link, len, last) where last[v] is the largest end position of the substrings
    of state v. Standard online construction; `last` by one pass over the link tree."""
    nxt, link, ln, lp = [{}], [-1], [0], [-1]
    cur_last = 0
    for i, c in enumerate(seq):
        cur = len(nxt)
        nxt.append({})
        ln.append(ln[cur_last] + 1)
        link.append(0)
        lp.append(i)
        p = cur_last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][c]
            if ln[p] + 1 == ln[q]:
                link[cur] = q
            else:
                cl = len(nxt)
                nxt.append(dict(nxt[q]))
                ln.append(ln[p] + 1)
                link.append(link[q])
                lp.append(-1)
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = cl
                    p = link[p]
                link[q] = link[cur] = cl
        cur_last = cur
    order = sorted(range(1, len(nxt)), key=ln.__getitem__, reverse=True)
    for v in order:
        u = link[v]
        if lp[v] > lp[u]:
            lp[u] = lp[v]
    return nxt, link, ln, lp


def match_profile(ctx, g, tail=16, own=True, sam=None, maxn=48):
    """For every position e of the response g: (L, q) where L is the length of the longest suffix of the stream
    ctx[-tail:] + g[:e+1] that also occurs earlier in the history, and q the end position (in ctx + g coordinates)
    of its newest earlier occurrence (-1 if none). The history is ctx without its last `tail` tokens (a suffix
    automaton; pass `sam` to reuse it) and, with own=True, the stream itself before the current token (an index of
    n-grams up to maxn tokens). Ties go to the stream: it is the newer text. With tail=0 and own=False this is the
    plain "longest match in the context" of every position."""
    tail = min(tail, len(ctx))
    base = len(ctx) - tail
    if sam is None:
        sam = build_sam(ctx[:base])
    nxt, link, ln, lp = sam
    stream = ctx[base:] + g
    kt = tail
    res_l, res_q = [0] * len(g), [-1] * len(g)
    v = 0
    l = 0
    idx = {}
    for t, c in enumerate(stream):
        while v and c not in nxt[v]:
            v = link[v]
            l = ln[v]
        if c in nxt[v]:
            v = nxt[v][c]
            l += 1
        else:
            v = 0
            l = 0
        if t >= kt:
            bl, bq = (l, lp[v]) if l > 0 else (0, -1)
            if own and t > 0:
                hi = min(maxn, t)
                if tuple(stream[t:t + 1]) in idx:
                    lo = 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if tuple(stream[t - mid + 1:t + 1]) in idx:
                            lo = mid
                        else:
                            hi = mid - 1
                    if lo >= bl:
                        bl, bq = lo, base + idx[tuple(stream[t - lo + 1:t + 1])]
            res_l[t - kt], res_q[t - kt] = bl, bq
        if own:
            for n in range(1, min(maxn, t + 1) + 1):
                idx[tuple(stream[t - n + 1:t + 1])] = t
    return res_l, res_q


def covered_flags(prof_len, min_len):
    """Which positions of the response sit inside a verbatim repeat of at least min_len tokens: the union of the
    matches [e - L + 1, e] (clipped at the response start) over positions whose clipped match length is >= min_len."""
    n = len(prof_len)
    cov = [False] * n
    reach = -1                      # lowest covered position so far is propagated by marking ranges
    for e in range(n):
        L = min(prof_len[e], e + 1)
        if L >= min_len:
            for p in range(max(e - L + 1, reach + 1), e + 1):
                cov[p] = True
            reach = e
    return cov


def runs(flags):
    """(start, length) of every maximal run of True."""
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - s))
            s = None
    if s is not None:
        out.append((s, len(flags) - s))
    return out


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


__all__ = [n for n in dir() if not n.startswith("_")]
Counter  # noqa: B018  (re-exported for the scripts)
