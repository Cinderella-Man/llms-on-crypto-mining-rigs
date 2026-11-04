"""A class for every token of an importance-matrix input, for llama-imatrix --span-mask (Q897), from its --span-dump.
  span_mask.py find DUMP.tsv --format template|plain --out MASK.txt [--summary SUMMARY.tsv] [--text TEXT]
                    [--ctx 512] [--chunks 200] [--show N]
  span_mask.py synth DUMP.tsv --kind all|none|id|ngram|chunk|random --out MASK.txt [--ctx 512] [--chunks 4]
                    [--chunk K] [--width W] [--seed S]
  span_mask.py text OUT.txt [--bytes 300000] [--seed S]
find rebuilds the input's bytes from the tokens' pieces (with --text it checks they are that file's bytes, less the
last newline, which llama.cpp's -f drops), finds the
tool-call spans and gives each token a class:
  0 other
  1 call, the tool-call scaffolding. template (the chat template, template_text.py): all of a <tool_call> ...
    </tool_call> block but the parameter values: the tags, the function and parameter names, the newlines between.
    plain (calib_text.py's `name({json})` lines in an assistant message): the name, the parentheses and the JSON's
    structure (braces, brackets, keys with their quotes, colons, commas, the quotes around string values)
  2 argument, the values. template: a parameter's value, between `<parameter=...>` and its newline and the newline
    before `</parameter>`. plain: the contents of the JSON's string values, its numbers, true, false and null
  3 response, the tool results. template: each <tool_response> ... </tool_response> block with its tags. plain: the
    body of a `### tool` message
  4 assistant, the rest of what the model writes: an assistant message but its tool calls (the reasoning with its
    <think> tags, the answer, the role line). plain: a `### assistant` message's body but its call lines
A token that covers bytes of more than one class takes the first of 1, 2, 3, 4, 0 among them, so a token that holds
part of a tag counts as scaffolding. An unterminated block (a session cut at its length limit) ends at the next
<|im_end|> or at the end of the text. The mask file is '#classes 5', then '<token id> <class>' per token; --summary
writes the tokens of each class and the calls, parameters and responses found, in the whole text and in the part
the matrix reads (the first CHUNKS chunks of CTX tokens). --show prints the first N calls token by token.
synth writes masks for the collector's checks: all (every token class 1) and none (every token class 0), both
'#classes 2'; id (class 1: every token whose id is the most frequent one in the first CHUNKS chunks; class 2: the token
after each of them, unless it has that id too; '#classes 3'); ngram (the same for the most frequent window of W tokens
that lies inside one chunk: class 1 its last token, class 2 the token after it; for models whose first layer reads
the preceding tokens too, as qwen4exp's n-gram embeddings do); chunk (class 1: the tokens of chunk K; '#classes 2');
random (classes 0-3 at random, '#classes 4').
text writes a text for those checks with the test vocabulary of test-llama-archs (a token per 5 bytes): a fixed
sentence of 40 bytes (8 tokens) after each filler of 10-40 random letters (a multiple of 5), so the sentence's tokens
repeat with the same 7 tokens before them.
Standard library only.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter

NAMES = ("other", "call", "argument", "response", "assistant")
PRIORITY = (1, 2, 3, 4, 0)
SCALAR = re.compile(rb"-?\d[\d.eE+-]*|true|false|null")
HEADER = re.compile(rb"(?m)^### (user|assistant|tool)\n")
PLAIN_CALL = re.compile(rb"(?m)^([A-Za-z_][A-Za-z0-9_.-]*)\((\{.*\})\)$")


def read_dump(path):
    """(header fields, token ids, piece bytes) of a --span-dump file."""
    ids, pieces, head = [], [], {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                cols = line[1:].split("\t")
                head.update(zip(cols[1::2], cols[2::2]))
                continue
            i, tid, hexes = (line.split("\t") + [""])[:3]
            assert int(i) == len(ids), f"{path}: token {i} out of order"
            ids.append(int(tid))
            pieces.append(bytes.fromhex(hexes))
    assert "tokens" not in head or int(head["tokens"]) == len(ids), f"{path}: {len(ids)} tokens, header {head}"
    return head, ids, pieces


def mark(cls, a, b, c):
    if b > a:
        cls[a:b] = bytes([c]) * (b - a)


def mark_json(cls, text, a, b):
    """A JSON text in text[a:b], marked call (1) already: string values' contents and scalars become argument (2)."""
    i = a
    while i < b:
        c = text[i]
        if c == 0x22:
            j = i + 1
            while j < b and text[j] != 0x22:
                j += 2 if text[j] == 0x5C else 1
            k = j + 1
            while k < b and text[k] in b" \t\r\n":
                k += 1
            if not (k < b and text[k] == 0x3A):  # a value, not a key
                mark(cls, i + 1, min(j, b), 2)
            i = j + 1
        elif c in b"-0123456789tfn":
            m = SCALAR.match(text, i, b)
            if m:
                mark(cls, i, m.end(), 2)
                i = m.end()
            else:
                i += 1
        else:
            i += 1


def block_end(text, start, close):
    """End of the block opened at start: after its close tag, or at the next <|im_end|> or the text's end."""
    e = text.find(close, start)
    im = text.find(b"<|im_end|>", start)
    if e >= 0 and (im < 0 or e < im):
        return e + len(close), True
    return (im if im >= 0 else len(text)), False


def find_template(text, cls):
    calls, params, resps = [], [], []
    pos = 0
    while (pos := text.find(b"<|im_start|>assistant", pos)) >= 0:
        end = text.find(b"<|im_end|>", pos)
        end = len(text) if end < 0 else end + len(b"<|im_end|>")
        mark(cls, pos, end, 4)
        pos = end
    pos = 0
    while (pos := text.find(b"<tool_response>", pos)) >= 0:
        end, _ok = block_end(text, pos, b"</tool_response>")
        mark(cls, pos, end, 3)
        resps.append(pos)
        pos = end
    pos = 0
    while (pos := text.find(b"<tool_call>", pos)) >= 0:
        end, ok = block_end(text, pos, b"</tool_call>")
        mark(cls, pos, end, 1)
        calls.append((pos, end, ok))
        q = pos
        while (q := text.find(b"<parameter=", q, end)) >= 0:
            gt = text.find(b">", q, end)
            if gt < 0:
                break
            vs = gt + 1 + (text[gt + 1:gt + 2] == b"\n")
            ve = text.find(b"\n</parameter>", vs, end)
            if ve < 0:
                ve = text.find(b"</parameter>", vs, end)
            if ve < 0:
                ve = end
            mark(cls, vs, ve, 2)
            params.append(q)
            q = ve
        pos = end
    return calls, params, resps


def find_plain(text, cls):
    calls, params, resps = [], [], []
    heads = list(HEADER.finditer(text))
    for k, h in enumerate(heads):
        a, b = h.end(), heads[k + 1].start() if k + 1 < len(heads) else len(text)
        if h[1] == b"tool":
            while b > a and text[b - 1] == 0x0A:
                b -= 1
            mark(cls, a, b, 3)
            resps.append(a)
        elif h[1] == b"assistant":
            mark(cls, a, b, 4)
            for m in PLAIN_CALL.finditer(text, a, b):
                try:
                    args = json.loads(m[2])
                except ValueError:
                    continue
                if not isinstance(args, dict):
                    continue
                mark(cls, m.start(), m.end(), 1)
                mark_json(cls, text, m.start(2), m.end(2))
                calls.append((m.start(), m.end(), True))
                params.extend([m.start()] * len(args))
    return calls, params, resps


def token_classes(pieces, cls):
    out, off = [], 0
    for p in pieces:
        seen = set(cls[off:off + len(p)])
        out.append(next((c for c in PRIORITY if c in seen), 0))
        off += len(p)
    return out


def write_mask(path, ids, classes, n_classes, note):
    with open(path, "w") as f:
        f.write(f"# span mask: {note}\n#classes {n_classes}\n")
        f.writelines(f"{t} {c}\n" for t, c in zip(ids, classes))


def cmd_find(a):
    head, ids, pieces = read_dump(a.dump)
    text = b"".join(pieces)
    if a.text:
        raw = open(a.text, "rb").read()
        if raw.endswith(b"\n") and raw[:-1] == text:
            raw = raw[:-1]  # llama.cpp's -f drops the file's last newline
        if raw != text:
            n = next((i for i, (x, y) in enumerate(zip(raw, text)) if x != y), min(len(raw), len(text)))
            sys.exit(f"{a.dump}: the pieces rebuild {len(text)} bytes, {a.text} has {len(raw)}; first difference at "
                     f"byte {n}: {raw[max(0, n - 20):n + 20]!r} vs {text[max(0, n - 20):n + 20]!r}")
    cls = bytearray(len(text))
    calls, params, resps = (find_template if a.format == "template" else find_plain)(text, cls)
    classes = token_classes(pieces, cls)
    write_mask(a.out, ids, classes, len(NAMES), f"{a.format} tool-call spans of {a.dump}; classes "
               + ", ".join(f"{c} {n}" for c, n in enumerate(NAMES)))
    n_proc = min(len(ids), a.ctx * a.chunks) // a.ctx * a.ctx
    proc_bytes = sum(len(p) for p in pieces[:n_proc])
    rows = [("tokens", len(ids), n_proc)]
    whole, proc = Counter(classes), Counter(classes[:n_proc])
    for c, name in enumerate(NAMES):
        rows.append((f"tokens_{name}", whole[c], proc[c]))
    rows.append(("calls", len(calls), sum(s < proc_bytes for s, _e, _ok in calls)))
    rows.append(("calls_unterminated", sum(not ok for *_x, ok in calls), sum(s < proc_bytes and not ok for s, _e, ok in calls)))
    rows.append(("parameters", len(params), sum(s < proc_bytes for s in params)))
    rows.append(("responses", len(resps), sum(s < proc_bytes for s in resps)))
    summary = "measure\twhole_text\tprocessed\n" + "".join(f"{m}\t{w}\t{p}\n" for m, w, p in rows)
    if a.summary:
        with open(a.summary, "w") as f:
            f.write(f"# {a.format} spans of {len(ids)} tokens (parse_special {head.get('parse_special', '?')}); "
                    f"processed: the first {a.chunks} chunks of {a.ctx} tokens\n" + summary)
    sys.stdout.write(summary)
    starts, off = [], 0
    for p in pieces:
        starts.append(off)
        off += len(p)
    for s, e, _ok in calls[:a.show]:
        toks = [k for k in range(len(ids)) if starts[k] < e and starts[k] + len(pieces[k]) > s]
        print(" ".join(f"{classes[k]}:{pieces[k].decode('utf-8', 'replace')!r}" for k in toks))


def cmd_synth(a):
    _head, ids, _pieces = read_dump(a.dump)
    n = len(ids)
    if a.kind in ("all", "none"):
        classes, n_cls = [1 if a.kind == "all" else 0] * n, 2
    elif a.kind == "id":
        mode = min(Counter(ids[:a.ctx * a.chunks]).most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
        classes = [1 if t == mode else 0 for t in ids]
        for k in range(n - 1):
            if classes[k] == 1 and classes[k + 1] == 0:
                classes[k + 1] = 2
        n_cls = 3
        print(f"id {mode}: {classes[:a.ctx * a.chunks].count(1)} tokens of it in the first {a.chunks} chunks, "
              f"{classes[:a.ctx * a.chunks].count(2)} after them")
    elif a.kind == "ngram":
        w = a.width
        seen = Counter(tuple(ids[k - w + 1:k + 1]) for k in range(min(n, a.ctx * a.chunks)) if k % a.ctx >= w - 1)
        gram, count = min(seen.most_common(), key=lambda kv: (-kv[1], kv[0]))
        classes = [0] * n
        for k in range(n):
            if k % a.ctx >= w - 1 and tuple(ids[k - w + 1:k + 1]) == gram:
                classes[k] = 1
        for k in range(n - 1):
            if classes[k] == 1 and classes[k + 1] == 0:
                classes[k + 1] = 2
        n_cls = 3
        print(f"window of {w} ending in {gram[-1]}: {count} in the first {a.chunks} chunks, "
              f"{classes[:a.ctx * a.chunks].count(2)} tokens after them")
    elif a.kind == "chunk":
        classes = [1 if a.chunk * a.ctx <= k < (a.chunk + 1) * a.ctx else 0 for k in range(n)]
        n_cls = 2
    else:
        r = random.Random(a.seed)
        classes, n_cls = [r.randrange(4) for _ in range(n)], 4
    write_mask(a.out, ids, classes, n_cls, f"synthetic, kind {a.kind}, of {a.dump}")


def cmd_text(a):
    r = random.Random(a.seed)
    sentence = "the same forty bytes, every time: done. "
    assert len(sentence) == 40
    parts, size = [], 0
    while size < a.bytes:
        filler = "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5 * r.randint(2, 8)))
        parts.append(filler + sentence)
        size += len(parts[-1])
    with open(a.out, "w") as f:
        f.write("".join(parts))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("dump")
    f.add_argument("--format", choices=("template", "plain"), required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--summary")
    f.add_argument("--text")
    f.add_argument("--ctx", type=int, default=512)
    f.add_argument("--chunks", type=int, default=200)
    f.add_argument("--show", type=int, default=0)
    s = sub.add_parser("synth")
    s.add_argument("dump")
    s.add_argument("--kind", choices=("all", "none", "id", "ngram", "chunk", "random"), required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ctx", type=int, default=512)
    s.add_argument("--chunks", type=int, default=4)
    s.add_argument("--chunk", type=int, default=0)
    s.add_argument("--width", type=int, default=8)
    s.add_argument("--seed", type=int, default=20261104)
    t = sub.add_parser("text")
    t.add_argument("out")
    t.add_argument("--bytes", type=int, default=300_000)
    t.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    {"find": cmd_find, "synth": cmd_synth, "text": cmd_text}[a.cmd](a)


if __name__ == "__main__":
    main()
