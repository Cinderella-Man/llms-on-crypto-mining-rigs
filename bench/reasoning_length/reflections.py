#!/usr/bin/env python3
"""Count the reflection words in Flint's reasoning (registry row Q94).

Reads transcripts with exact token ids (bench/expert_overlap/generate.py writes them) and Flint's vocabulary from a GGUF
header, decodes the reasoning block (<think> ... </think>, or to the end when the output ran out of tokens) and counts:
- word-initial tokens that are reflection words ("Wait", " Hmm", " Actually", ...), the kind NoWait suppresses with a
  logit bias during reasoning, per word, per 1,000 reasoning tokens, at the start of a sentence or inside one;
- where in the reasoning they occur (tenths of its length);
- the reasoning tokens in sentences and paragraphs that a reflection word opens: the self-checking text such a bias aims
  at, an upper bound on what it can remove directly;
- the most frequent sentence openers, to choose the bias list from Flint's own text.

Text mode (--text) reads outputs without token ids (llama-server's `content`, e.g. the BFCL runs of Q41) and counts the
same words per 1,000 words of reasoning.

usage: python reflections.py GEN.jsonl MODEL.gguf [--out RESULTS.json]
       python reflections.py --text OUTPUTS.jsonl [...] [--out RESULTS.json]
"""
import argparse
import json
import re
import struct
from collections import Counter, defaultdict

THINK_OPEN, THINK_CLOSE = 150006, 150007

# Openers of a self-check or a change of course. "but" and "however" also join ordinary clauses, so they are counted
# separately as contrast words.
REFLECTION = ("wait", "hmm", "hmmm", "actually", "alternatively", "oh", "ah", "double-check", "recheck", "re-check",
              "verify", "reconsider", "rethink")
CONTRAST = ("but", "however", "maybe", "perhaps")
PHRASES = ("but wait", "wait,", "hold on", "let me double-check", "let me double check", "let me check",
           "let me verify", "let me re-check", "let me recheck", "let me re-read", "let me reread", "let me reconsider",
           "let me think again", "let me re-examine", "double-check", "on second thought", "actually,")


def gguf_tokens(path):
    """tokenizer.ggml.tokens from a GGUF header (metadata only)."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def rstr():
            return f.read(rd("Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("I"), rd("Q")
                return [rval(et) for _ in range(n)]
            return rd(sizes[t])
        assert f.read(4) == b"GGUF"
        rd("I")
        rd("Q")
        for _ in range(rd("Q")):
            key, t = rstr(), rd("I")
            val = rval(t)
            if key == "tokenizer.ggml.tokens":
                return val
    raise ValueError("no tokenizer.ggml.tokens")


def byte_decoder():
    """GPT-2 byte-level BPE: the printable character that stands for each byte, inverted."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_texts(vocab):
    dec = byte_decoder()
    out = []
    for tok in vocab:
        if all(ch in dec for ch in tok):
            out.append(bytes(dec[ch] for ch in tok).decode("utf-8", "replace"))
        else:
            out.append(tok)
    return out


WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def analyze(rows, texts):
    res = {}
    by = defaultdict(Counter)                 # class -> counters
    words = defaultdict(Counter)              # class -> word -> count
    words_initial = defaultdict(Counter)      # class -> word -> sentence-initial count
    tenths = defaultdict(lambda: [0] * 10)    # class -> reflection words per tenth of the reasoning
    openers = Counter()
    for r in rows:
        gen, cls = r["gen_tokens"], r["category"]
        if THINK_OPEN not in gen:
            continue
        a = gen.index(THINK_OPEN) + 1
        b = gen.index(THINK_CLOSE) if THINK_CLOSE in gen else len(gen)
        toks = gen[a:b]
        n = len(toks)
        if not n:
            continue
        # decoded text with each token's start offset
        starts, parts, pos = [], [], 0
        for t in toks:
            s = texts[t]
            starts.append(pos)
            parts.append(s)
            pos += len(s)
        text = "".join(parts)

        def tok_at(off):    # index of the token containing character offset off
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if starts[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        for key in (cls, "all"):
            by[key]["reasoning_tokens"] += n
            by[key]["transcripts"] += 1
        # word-initial reflection tokens: the token's text, without leading space, is the word (any case)
        for i, t in enumerate(toks):
            s = parts[i]
            w = s.strip().lower()
            if not s or not s[:1].isspace() and i and not parts[i - 1][-1:].isspace() and parts[i - 1][-1:].isalpha():
                continue    # inside a word
            w = w.rstrip(",.!?:;")
            if w in REFLECTION or w in CONTRAST:
                before = text[:starts[i]].rstrip(" ")
                initial = not before or before[-1] in ".!?:\n*" or before.endswith("--")
                kind = "reflection" if w in REFLECTION else "contrast"
                for key in (cls, "all"):
                    words[key][w] += 1
                    by[key][kind + "_tokens"] += 1
                    if initial:
                        words_initial[key][w] += 1
                        by[key][kind + "_initial"] += 1
                    if kind == "reflection":
                        tenths[key][min(9, i * 10 // n)] += 1
        # phrases
        low = text.lower()
        for ph in PHRASES:
            for m in re.finditer(re.escape(ph), low):
                if m.start() and low[m.start() - 1].isalpha():
                    continue
                for key in (cls, "all"):
                    words[key]["phrase: " + ph] += 1
        # sentences and paragraphs opened by a reflection word ("but" only as "but wait")
        for unit, pattern in (("sentence", r"(?<=[.!?:])\s+|\n+"), ("paragraph", r"\n\s*\n")):
            off = 0
            for piece in re.split(pattern, text):
                start = text.find(piece, off) if piece else off
                end = start + len(piece)
                off = end
                if not piece.strip():
                    continue
                if unit == "sentence":
                    m = WORD.match(piece.lstrip(" -*#>\t0123456789.)"))
                    if m:
                        openers[m.group(0)] += 1
                if opens_reflection(piece):
                    k = tok_at(end - 1) - tok_at(start) + 1
                    for key in (cls, "all"):
                        by[key][unit + "_tokens"] += k
                        by[key][unit + "_count"] += 1
    for key, c in by.items():
        n = c["reasoning_tokens"]
        res[key] = {
            "transcripts": c["transcripts"], "reasoning_tokens": n,
            "reflection_per_1000": 1000 * c["reflection_tokens"] / n,
            "reflection_initial_per_1000": 1000 * c["reflection_initial"] / n,
            "contrast_per_1000": 1000 * c["contrast_tokens"] / n,
            "contrast_initial_per_1000": 1000 * c["contrast_initial"] / n,
            "reflection_sentences": c["sentence_count"], "reflection_sentence_share": c["sentence_tokens"] / n,
            "reflection_paragraphs": c["paragraph_count"], "reflection_paragraph_share": c["paragraph_tokens"] / n,
            "words": dict(words[key].most_common()), "words_initial": dict(words_initial[key].most_common()),
            "reflection_by_tenth": tenths[key],
        }
    res["sentence_openers"] = dict(openers.most_common(60))
    return res


def opens_reflection(s):
    s = s.lstrip(" -*#>\t0123456789.)").lower()
    m = WORD.match(s)
    if not m:
        return False
    if m.group(0).rstrip("'") in REFLECTION:
        return True
    return s.startswith(("but wait", "let me double", "let me re", "let me verify", "let me check", "on second thought",
                         "hold on"))


def analyze_text(items):
    """items: (class, reasoning text). Counts per 1,000 words; spans as shares of the reasoning's characters."""
    c, words = defaultdict(Counter), defaultdict(Counter)
    for cls, text in items:
        for key in (cls, "all"):
            c[key]["outputs"] += 1
            c[key]["chars"] += len(text)
        prev_end = 0
        for m in WORD.finditer(text):
            w = m.group(0).lower().rstrip("'")
            for key in (cls, "all"):
                c[key]["words"] += 1
            if w in REFLECTION or w in CONTRAST:
                kind = "reflection" if w in REFLECTION else "contrast"
                before = text[:m.start()].rstrip(" ")
                initial = not before or before[-1] in ".!?:\n*"
                for key in (cls, "all"):
                    words[key][w] += 1
                    c[key][kind] += 1
                    c[key][kind + "_initial"] += initial
            prev_end = m.end()
        for unit, pattern in (("sentence", r"(?<=[.!?:])\s+|\n+"), ("paragraph", r"\n\s*\n")):
            for piece in re.split(pattern, text):
                if piece.strip() and opens_reflection(piece):
                    for key in (cls, "all"):
                        c[key][unit + "_chars"] += len(piece)
                        c[key][unit + "_count"] += 1
    res = {}
    for key, k in c.items():
        nw = max(1, k["words"])
        res[key] = {"outputs": k["outputs"], "reasoning_words": k["words"],
                    "reflection_per_1000_words": 1000 * k["reflection"] / nw,
                    "reflection_initial_per_1000_words": 1000 * k["reflection_initial"] / nw,
                    "contrast_per_1000_words": 1000 * k["contrast"] / nw,
                    "contrast_initial_per_1000_words": 1000 * k["contrast_initial"] / nw,
                    "reflection_sentence_share": k["sentence_chars"] / max(1, k["chars"]),
                    "reflection_paragraph_share": k["paragraph_chars"] / max(1, k["chars"]),
                    "words": dict(words[key].most_common())}
    return res


def reasoning_of(content):
    a = content.find("<think>")
    if a < 0:
        return None
    b = content.find("</think>", a)
    return content[a + 7:b if b >= 0 else len(content)]


def main_text(paths, out):
    items = []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            text = reasoning_of(r.get("content") or r.get("text") or "")
            if text is not None:
                items.append((r.get("category", "all").split("_")[0] if "gen_tokens" not in r else r["category"], text))
    res = analyze_text(items)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    print("| Class | Outputs | Reasoning words | Reflection words per 1,000 words (sentence-initial) | Contrast words per"
          " 1,000 (sentence-initial) | Reasoning in sentences a reflection word opens | in paragraphs it opens |")
    print("|---|---|---|---|---|---|---|")
    for key, r in sorted(res.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        print(f"| {key} | {r['outputs']:,} | {r['reasoning_words']:,} | {r['reflection_per_1000_words']:.1f}"
              f" ({r['reflection_initial_per_1000_words']:.1f}) | {r['contrast_per_1000_words']:.1f}"
              f" ({r['contrast_initial_per_1000_words']:.1f}) | {100 * r['reflection_sentence_share']:.1f} % |"
              f" {100 * r['reflection_paragraph_share']:.1f} % |")
    print("\nWords (all):", ", ".join(f"{w} {n}" for w, n in res["all"]["words"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen", nargs="?")
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("--text", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.text:
        return main_text(args.text, args.out)
    rows = [json.loads(line) for line in open(args.gen)]
    texts = token_texts(gguf_tokens(args.gguf))
    res = analyze(rows, texts)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("| Class | Reasoning tokens | Reflection words per 1,000 (sentence-initial) | Contrast words per 1,000"
          " (sentence-initial) | Reasoning in sentences a reflection word opens | in paragraphs it opens |")
    print("|---|---|---|---|---|---|")
    for key in ("chat", "code", "edit", "agent", "all"):
        r = res.get(key)
        if not r:
            continue
        print(f"| {key} | {r['reasoning_tokens']:,} | {r['reflection_per_1000']:.1f} ({r['reflection_initial_per_1000']:.1f}) |"
              f" {r['contrast_per_1000']:.1f} ({r['contrast_initial_per_1000']:.1f}) |"
              f" {100 * r['reflection_sentence_share']:.1f} % ({r['reflection_sentences']}) |"
              f" {100 * r['reflection_paragraph_share']:.1f} % ({r['reflection_paragraphs']}) |")
    a = res["all"]
    print("\nWords (all classes; sentence-initial in brackets):")
    for w, c in a["words"].items():
        print(f"  {w}: {c} ({a['words_initial'].get(w, 0)})")
    print("\nReflection words by tenth of the reasoning:", a["reflection_by_tenth"])
    print("\nMost frequent sentence openers:", ", ".join(f"{w} {c}" for w, c in list(res["sentence_openers"].items())[:40]))


if __name__ == "__main__":
    main()
