"""Exact token counts for Flint's tokenizer in pure Python (standard library only).

Reads Flint's Hugging Face `tokenizer.json` (byte-level BPE, 150,000 tokens, 149,744 merges, added tokens such as
<think> and <tool_call>) and reproduces its pre-tokenizer: digits in runs of 1-3, CJK runs, then the GPT-4-style
split. The Unicode classes (\\p{L}, \\p{Lu}, \\p{M}, \\p{P}, \\p{S}, ...) are built from Python's unicodedata, since the
standard `re` module has no \\p{...}.
"""
import json
import re
import sys
import unicodedata
from functools import lru_cache


def _ranges(pred):
    out, start, prev = [], None, None
    for cp in range(sys.maxunicode + 1):
        if pred(cp):
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def _cls(rs):
    esc = lambda cp: re.escape(chr(cp)) if cp > 0x20 else "\\x%02x" % cp
    return "".join(esc(a) if a == b else esc(a) + "-" + esc(b) for a, b in rs if not 0xD800 <= a <= 0xDFFF)


def _build_patterns():
    cat = [unicodedata.category(chr(cp)) for cp in range(sys.maxunicode + 1)]
    has = lambda cats: _cls(_ranges(lambda cp: cat[cp] in cats))
    L = has({"Lu", "Ll", "Lt", "Lm", "Lo"})
    N = has({"Nd", "Nl", "No"})
    up = has({"Lu", "Lt", "Lm", "Lo", "Mn", "Mc", "Me"})
    lo = has({"Ll", "Lm", "Lo", "Mn", "Mc", "Me"})
    ps = has({"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So"})
    sfx = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    main = (f"[^\\r\\n{L}{N}]?[{up}]*[{lo}]+{sfx}|[^\\r\\n{L}{N}]?[{up}]+[{lo}]*{sfx}"
            f"| ?[{ps}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    return [re.compile(f"[{N}]{{1,3}}"), re.compile("[一-龥぀-ゟ゠-ヿ]+"), re.compile(main)]


def _byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class FlintTokenizer:
    def __init__(self, path):
        t = json.load(open(path))
        assert t["model"]["type"] == "BPE"
        self.vocab = t["model"]["vocab"]
        self.ranks = {tuple(m if isinstance(m, list) else m.split(" ", 1)): i for i, m in enumerate(t["model"]["merges"])}
        added = sorted((a["content"] for a in t["added_tokens"]), key=len, reverse=True)
        self.added = re.compile("|".join(re.escape(a) for a in added))
        self.splits = _build_patterns()
        self.bmap = _byte_map()
        self._bpe = lru_cache(maxsize=1 << 20)(self._bpe_uncached)

    def _bpe_uncached(self, piece):
        word = [self.bmap[b] for b in piece.encode("utf-8")]
        while len(word) > 1:
            best, bi = None, -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            a, b = word[bi], word[bi + 1]
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return len(word)

    def _pretokenize(self, text):
        pieces = [text]
        for pat in self.splits:
            nxt = []
            for p in pieces:
                pos = 0
                for m in pat.finditer(p):
                    if m.start() > pos:
                        nxt.append(p[pos:m.start()])
                    if m.end() > m.start():
                        nxt.append(m.group())
                    pos = m.end()
                if pos < len(p):
                    nxt.append(p[pos:])
            pieces = nxt
        return pieces

    def count(self, text):
        """Number of tokens in text, added tokens (<think>, <tool_call>, ...) counted as one each."""
        n, pos = 0, 0
        for m in self.added.finditer(text):
            n += sum(self._bpe(p) for p in self._pretokenize(text[pos:m.start()])) + 1
            pos = m.end()
        return n + sum(self._bpe(p) for p in self._pretokenize(text[pos:]))
