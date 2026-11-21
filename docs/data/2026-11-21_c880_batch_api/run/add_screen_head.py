#!/usr/bin/env python3
# C880 (2026-11-21): copies a GGUF and adds a screened output head, so that llama-screen-check runs on a small random
# model. The fork loads an optional tensor output_screen.weight of the output head's shape (llama_set_screen_head picks
# M candidates with it, then computes their logits with the full head) and reads M from <arch>.output_screen.top_m.
# The added tensor is a copy of output.weight (token_embd.weight when the head is tied), so the screen picks the full
# head's own top M: the point is to run llama-screen-check's decode path, not to judge a screen.
#
# The file is extended in place of being rewritten (gguf-py's writer refuses the empty arrays a random model's
# tokenizer keys can hold): the header's two counts go up by one, the new key goes after the last key, the new tensor's
# entry after the last entry with its offset at the end of the data, and its bytes (the source tensor's) after the data.
# Every existing offset in the data section stays valid. Little-endian GGUF v2 or v3 only.
#
# Usage: python3 -I add_screen_head.py <in.gguf> <out.gguf> [--top-m 64]
import argparse
import struct
import sys

# GGUF value types: fixed sizes; 8 is a string, 9 an array
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_U32, T_STRING, T_ARRAY = 4, 8, 9
# ggml types the random models use: (block size, bytes per block)
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24)}


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.p = buf, 0

    def take(self, fmt: str):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        n = self.take("Q")
        s = self.b[self.p:self.p + n].decode("utf-8")
        self.p += n
        return s

    def value(self, t: int):
        if t in FIXED:
            raw = self.b[self.p:self.p + FIXED[t]]
            self.p += FIXED[t]
            return raw
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et, n = self.take("I"), self.take("Q")
            for _ in range(n):
                self.value(et)
            return None
        raise ValueError(f"unknown GGUF value type {t}")


def gguf_string(s: str) -> bytes:
    e = s.encode("utf-8")
    return struct.pack("<Q", len(e)) + e


def main() -> int:
    ap = argparse.ArgumentParser(description="copy a GGUF and add output_screen.weight (a copy of the output head)")
    ap.add_argument("--top-m", type=int, default=64, help="<arch>.output_screen.top_m (candidates per token)")
    ap.add_argument("input")
    ap.add_argument("output")
    a = ap.parse_args()

    buf = open(a.input, "rb").read()
    r = Reader(buf)
    magic, version = r.take("4s"), r.take("I")
    if magic != b"GGUF" or version not in (2, 3):
        print(f"{a.input}: not a little-endian GGUF v2/v3 file", file=sys.stderr)
        return 1
    n_tensors, n_kv = r.take("Q"), r.take("Q")
    arch, align = None, 32
    for _ in range(n_kv):
        key, t = r.string(), r.take("I")
        v = r.value(t)
        if key == "general.architecture":
            arch = v
        elif key == "general.alignment" and t == T_U32:
            align = struct.unpack("<I", v)[0]
    kv_end = r.p
    infos = {}
    for _ in range(n_tensors):
        start = r.p
        name, nd = r.string(), r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        gtype, off = r.take("I"), r.take("Q")
        infos[name] = (start, r.p, dims, gtype, off)
    ti_end = r.p
    data_start = (ti_end + align - 1) // align * align
    if arch is None or "output_screen.weight" in infos:
        print(f"{a.input}: no architecture key, or output_screen.weight already there", file=sys.stderr)
        return 1
    src_name = "output.weight" if "output.weight" in infos else "token_embd.weight"
    if src_name not in infos:
        print(f"{a.input} has neither output.weight nor token_embd.weight", file=sys.stderr)
        return 1
    _, _, dims, gtype, off = infos[src_name]
    if gtype not in TYPE_SIZE:
        print(f"{src_name}: ggml type {gtype} not in this script's size table", file=sys.stderr)
        return 1
    bs, ts = TYPE_SIZE[gtype]
    n_el = 1
    for d in dims:
        n_el *= d
    nbytes = n_el // bs * ts
    src = buf[data_start + off:data_start + off + nbytes]
    data = buf[data_start:]
    data_len = (len(data) + align - 1) // align * align

    key = gguf_string(f"{arch}.output_screen.top_m") + struct.pack("<II", T_U32, a.top_m)
    info = (gguf_string("output_screen.weight") + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", gtype, data_len))
    head = (buf[:4] + struct.pack("<IQQ", version, n_tensors + 1, n_kv + 1) + buf[24:kv_end] + key
            + buf[kv_end:ti_end] + info)
    with open(a.output, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % align))
        f.write(data + b"\0" * (data_len - len(data)))
        f.write(src)
    print(f"{a.output}: {arch}, {n_tensors + 1} tensors, {n_kv + 1} keys; output_screen.weight = copy of {src_name} "
          f"(ggml type {gtype}, dims {dims}, {nbytes} bytes), {arch}.output_screen.top_m = {a.top_m}, alignment {align}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
