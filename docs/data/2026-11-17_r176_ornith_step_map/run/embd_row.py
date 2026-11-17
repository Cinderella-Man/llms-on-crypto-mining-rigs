#!/usr/bin/env python3
"""embd_row.py: one row of a GGUF file's token embedding, read alone (FN114, Q1544).

  nice -n 19 python3 -I embd_row.py <model.gguf> 248077 248076 1000

Reads the GGUF header (the key-value pairs and the tensor table), then only the named rows' bytes of
`token_embd.weight`: never the rest of the file. Per row: the tensor's type, the row's offset and size in bytes, how many
of its bytes are not zero, its token text (`tokenizer.ggml.tokens`), and its values dequantized (the first 16, the
largest magnitude, how many are zero, and each block's scale for the block types). Types read: F32, F16, BF16, Q4_0,
Q5_0, Q8_0, Q4_K, Q5_K, Q6_K; any other type gets its raw bytes only.
Standard library only, so it runs on a rig's bare Python.
"""
import struct
import sys

GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# ggml type -> (name, elements a block, bytes a block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(n))[0]

    def string(self):
        n = self.take("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t, want=None):
        if t in GGUF_SCALARS:
            return self.take(GGUF_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            if want is not None:  # keep only the wanted indices (the token table holds ~248k strings)
                out = {}
                for i in range(n):
                    v = self.value(et)
                    if i in want:
                        out[i] = v
                return out
            if et in GGUF_SCALARS:  # skip a numeric array without decoding it
                self.f.seek(n * struct.calcsize("<" + GGUF_SCALARS[et]), 1)
                return f"<array of {n}>"
            for _ in range(n):
                self.value(et)
            return f"<array of {n}>"
        raise ValueError(f"unknown GGUF value type {t}")


def f16(b, o):
    return struct.unpack_from("<e", b, o)[0]


def k_scale_min(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant(tname, b):
    """-> (values, per-block scales) for the types listed in the docstring, else (None, None)"""
    vals, scales = [], []
    if tname == "F32":
        return list(struct.unpack(f"<{len(b) // 4}f", b)), []
    if tname == "F16":
        return list(struct.unpack(f"<{len(b) // 2}e", b)), []
    if tname == "BF16":
        return [struct.unpack("<f", b"\0\0" + b[i:i + 2])[0] for i in range(0, len(b), 2)], []
    if tname == "Q8_0":
        for o in range(0, len(b), 34):
            d = f16(b, o)
            scales.append(d)
            vals += [d * q for q in struct.unpack_from("<32b", b, o + 2)]
        return vals, scales
    if tname == "Q4_0":
        for o in range(0, len(b), 18):
            d = f16(b, o)
            scales.append(d)
            qs = b[o + 2:o + 18]
            vals += [d * ((x & 0xF) - 8) for x in qs] + [d * ((x >> 4) - 8) for x in qs]
        return vals, scales
    if tname == "Q5_0":
        for o in range(0, len(b), 22):
            d = f16(b, o)
            scales.append(d)
            qh = struct.unpack_from("<I", b, o + 2)[0]
            qs = b[o + 6:o + 22]
            lo = [((x & 0xF) | (((qh >> j) & 1) << 4)) - 16 for j, x in enumerate(qs)]
            hi = [((x >> 4) | (((qh >> (j + 12)) & 1) << 4)) - 16 for j, x in enumerate(qs)]
            vals += [d * v for v in lo + hi]
        return vals, scales
    if tname in ("Q4_K", "Q5_K"):
        bs = 144 if tname == "Q4_K" else 176
        for o in range(0, len(b), bs):
            d, dmin = f16(b, o), f16(b, o + 2)
            scales.append(d)
            sc = b[o + 4:o + 16]
            if tname == "Q4_K":
                qh, qs = None, b[o + 16:o + 144]
            else:
                qh, qs = b[o + 16:o + 48], b[o + 48:o + 176]
            is_, u1, u2 = 0, 1, 2
            for j in range(0, 256, 64):
                s1, m1 = k_scale_min(is_, sc)
                s2, m2 = k_scale_min(is_ + 1, sc)
                q = qs[j // 2:j // 2 + 32]
                for l in range(32):
                    h = ((qh[l] & u1) and 16 or 0) if qh else 0
                    vals.append(d * s1 * ((q[l] & 0xF) + h) - dmin * m1)
                for l in range(32):
                    h = ((qh[l] & u2) and 16 or 0) if qh else 0
                    vals.append(d * s2 * ((q[l] >> 4) + h) - dmin * m2)
                is_ += 2
                u1 <<= 2
                u2 <<= 2
        return vals, scales
    if tname == "Q6_K":
        for o in range(0, len(b), 210):
            ql, qh = b[o:o + 128], b[o + 128:o + 192]
            sc = struct.unpack_from("<16b", b, o + 192)
            d = f16(b, o + 208)
            scales.append(d)
            out = [0.0] * 256
            for n in (0, 128):
                lq, hq, s = ql[n // 2:n // 2 + 64], qh[n // 4:n // 4 + 32], sc[n // 16:n // 16 + 8]
                for l in range(32):
                    i = l // 16
                    q1 = ((lq[l] & 0xF) | (((hq[l] >> 0) & 3) << 4)) - 32
                    q2 = ((lq[l + 32] & 0xF) | (((hq[l] >> 2) & 3) << 4)) - 32
                    q3 = ((lq[l] >> 4) | (((hq[l] >> 4) & 3) << 4)) - 32
                    q4 = ((lq[l + 32] >> 4) | (((hq[l] >> 6) & 3) << 4)) - 32
                    out[n + l] = d * s[i] * q1
                    out[n + l + 32] = d * s[i + 2] * q2
                    out[n + l + 64] = d * s[i + 4] * q3
                    out[n + l + 96] = d * s[i + 6] * q4
            vals += out
        return vals, scales
    return None, None


def main():
    path, rows = sys.argv[1], [int(x) for x in sys.argv[2:]]
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file")
        version, n_t, n_kv = r.take("I"), r.take("Q"), r.take("Q")
        align, tokens, n_vocab = 32, {}, None
        for _ in range(n_kv):
            key, t = r.string(), r.take("I")
            if key == "tokenizer.ggml.tokens":
                tokens = r.value(t, want=set(rows))
            else:
                v = r.value(t)
                if key == "general.alignment":
                    align = v
        info = None
        for _ in range(n_t):
            name = r.string()
            nd = r.take("I")
            dims = [r.take("Q") for _ in range(nd)]
            tt, off = r.take("I"), r.take("Q")
            if name == "token_embd.weight":
                info = (dims, tt, off)
        data0 = (f.tell() + align - 1) // align * align
        if info is None:
            sys.exit("no token_embd.weight")
        dims, tt, off = info
        tname, be, bb = TYPES.get(tt, (f"ggml type {tt}", None, None))
        print(f"GGUF v{version}, {n_t} tensors, {n_kv} keys; token_embd.weight: type {tname} ({tt}), dims {dims}")
        if be is None:
            sys.exit("unknown block size for this type")
        row_bytes = dims[0] // be * bb
        for row in rows:
            f.seek(data0 + off + row * row_bytes)
            b = f.read(row_bytes)
            nz = sum(1 for x in b if x)
            print(f"\nrow {row} (token {tokens.get(row, '?')!r}): {row_bytes} bytes at {data0 + off + row * row_bytes}, "
                  f"{nz} of them not zero")
            vals, scales = dequant(tname, b)
            if vals is None:
                print("  first 32 bytes:", b[:32].hex())
                continue
            zeros = sum(1 for v in vals if v == 0.0)
            amax = max(abs(v) for v in vals)
            print(f"  {len(vals)} values: {zeros} are exactly zero; largest magnitude {amax:.6g}")
            print("  first 16:", " ".join(f"{v:.4g}" for v in vals[:16]))
            if scales:
                zs = sum(1 for s in scales if s == 0.0)
                print(f"  {len(scales)} block scales: {zs} are zero; first 8:", " ".join(f"{s:.4g}" for s in scales[:8]))


if __name__ == "__main__":
    main()
