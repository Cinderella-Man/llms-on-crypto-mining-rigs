#!/usr/bin/env python3
"""Print the token ids that tree_replay.py splits a dump by, for a GGUF model: the markers that open and close a tool
call and the reasoning, read from the tokenizer's vocabulary in the file's header (the first shard is enough).

    python3 gguf_markers.py MODEL.gguf
    --tool-open 248058 --tool-close 248059 --think-open 248068 --think-close 248069

The output is the argument list of tree_replay.py; a marker the vocabulary does not hold as a single token is left
out, with a note on stderr. Standard library only; it reads the header, not the weights.
"""
import struct
import sys

MARKERS = {"tool-open": "<tool_call>", "tool-close": "</tool_call>", "think-open": "<think>", "think-close": "</think>"}
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def read_str(f):
    (n,) = read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def skip_str(f):
    (n,) = read(f, "<Q")
    f.seek(n, 1)


def read_value(f, t, keep):
    """One metadata value of GGUF type t; arrays are returned only for `keep` (the vocabulary), else skipped."""
    if t in SIZES:
        f.seek(SIZES[t], 1)
        return None
    if t == 8:
        return read_str(f) if keep else skip_str(f)
    if t == 9:
        (et,) = read(f, "<I")
        (n,) = read(f, "<Q")
        if et in SIZES:
            f.seek(n * SIZES[et], 1)
            return None
        if et == 8:
            if keep:
                return [read_str(f) for _ in range(n)]
            for _ in range(n):
                skip_str(f)
            return None
        for _ in range(n):  # arrays of arrays: not used by tokenizers
            read_value(f, et, False)
        return None
    raise SystemExit(f"unknown GGUF value type {t}")


def vocabulary(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        (_version,) = read(f, "<I")
        (_n_tensors,) = read(f, "<Q")
        (n_kv,) = read(f, "<Q")
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = read(f, "<I")
            value = read_value(f, t, key == "tokenizer.ggml.tokens")
            if key == "tokenizer.ggml.tokens":
                return value
    raise SystemExit(f"{path}: no tokenizer.ggml.tokens")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    tokens = vocabulary(sys.argv[1])
    index = {}
    for i, tok in enumerate(tokens):
        index.setdefault(tok, i)
    out = []
    for flag, text in MARKERS.items():
        if text in index:
            out.append(f"--{flag} {index[text]}")
        else:
            print(f"{text} is not a single token of this vocabulary", file=sys.stderr)
    print(" ".join(out))


if __name__ == "__main__":
    main()
