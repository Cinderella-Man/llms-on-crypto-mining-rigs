#!/usr/bin/env python3
# table_info.py: where Flash-Next's n-gram table (`per_layer_token_embd.weight`) sits in a split GGUF, and the hash
# heads' row ranges, for table_bits. Standard library only.
#   python3 -I table_info.py SHARD... > table_info.txt
# Prints `key value` lines: file, offset (absolute byte offset of the tensor data), type, type_pos (byte offset of the
# tensor's type field in that file's header), ne0 (row width), rows, end (offset + data size), file_size, heads, then
# `head <h> <first row> <rows>` per head, and every `*.ple.*` key.
import struct, sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 20: ("IQ4_NL", 32, 18), 30: ("BF16", 1, 2)}
NAME = "per_layer_token_embd.weight"


def read(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    rstr = lambda: f.read(rd("<Q")).decode()

    def val(t):
        if t in SCALAR:
            return rd(SCALAR[t])
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [val(et) for _ in range(n)]
        raise ValueError(f"{path}: value type {t}")

    assert f.read(4) == b"GGUF", f"{path}: not a GGUF"
    _, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = val(rd("<I"))
    tensors = {}
    for _ in range(nt):
        name = rstr()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        tpos = f.tell()
        tensors[name] = (dims, rd("<I"), rd("<Q"), tpos)
    align = kv.get("general.alignment", 32)
    data0 = (f.tell() + align - 1) // align * align
    f.seek(0, 2)
    return kv, tensors, data0, f.tell()


kv = {}
out = []
for path in sys.argv[1:]:
    k, tensors, data0, size = read(path)
    kv.update(k)
    if NAME in tensors:
        dims, t, off, tpos = tensors[NAME]
        tname, blk, bsz = TYPES[t]
        nbytes = dims[1] * dims[0] // blk * bsz
        out += [("file", path), ("offset", data0 + off), ("type", tname), ("type_pos", tpos), ("ne0", dims[0]),
                ("rows", dims[1]), ("end", data0 + off + nbytes), ("file_size", size)]
arch = kv["general.architecture"]
offs, sizes = kv[f"{arch}.ple.head_offsets"], kv[f"{arch}.ple.head_vocab_sizes"]
out.append(("heads", len(offs)))
for k, v in out:
    print(k, v)
for h, (o, n) in enumerate(zip(offs, sizes)):
    print("head", h, o, n)
for k in sorted(kv):
    if ".ple." in k:
        print("#", k, kv[k])
