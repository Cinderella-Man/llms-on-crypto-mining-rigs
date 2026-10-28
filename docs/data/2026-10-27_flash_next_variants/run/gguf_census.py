#!/usr/bin/env python3
# What a GGUF file holds, read from its header alone: the metadata, and the bytes per tensor class (the tensor name with
# its layer number taken out) and format. Formats the local gguf-py doesn't know (custom types such as trellis codes)
# still count, by their id, since sizes come from the tensor offsets and not from block sizes.
#
# The closing summary splits the bytes into what a GPU must hold and what one generated token reads: every tensor
# once, except the token embeddings and the n-gram table (a few rows per token), the routed experts (used / count of
# them) and any MTP layer (read only while drafting).
#
#   gguf_census.py FILE.gguf [FILE-00002-of-00002.gguf ...]    one report over all the shards given
#
# GGUF_PY=<llama.cpp>/gguf-py names the formats; without it they print as type ids.
import mmap
import os
import re
import struct
import sys
from collections import defaultdict

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def type_names():
    path = os.environ.get("GGUF_PY")
    if not path:
        return {}
    # constants.py alone, so the package's numpy imports aren't needed
    import importlib.util
    spec = importlib.util.spec_from_file_location("gguf_constants", os.path.join(path, "gguf", "constants.py"))
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    return {t.value: t.name for t in constants.GGMLQuantizationType}


class Header:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        self.size = os.path.getsize(path)

    def take(self, fmt):
        v = struct.unpack_from(fmt, self.m, self.pos)
        self.pos += struct.calcsize(fmt)
        return v[0]

    def string(self):
        n = self.take("<Q")
        s = self.m[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return s

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.take("<I"), self.take("<Q")
            if etype in SCALAR and count > 64:
                self.pos += struct.calcsize(SCALAR[etype]) * count
                return f"[{count} values]"
            items = [self.value(etype) for _ in range(count)]
            return f"[{count} values]" if count > 64 else items
        return self.take(SCALAR[vtype])


def read(path):
    h = Header(path)
    if h.m[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    h.pos = 4
    h.take("<I")
    n_tensors, n_kv = h.take("<Q"), h.take("<Q")
    meta = {}
    for _ in range(n_kv):
        key = h.string()
        meta[key] = h.value(h.take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = h.string()
        dims = [h.take("<Q") for _ in range(h.take("<I"))]
        infos.append((name, dims, h.take("<I"), h.take("<Q")))
    align = int(meta.get("general.alignment", 32))
    data_start = (h.pos + align - 1) // align * align
    ends = sorted(o for *_, o in infos)[1:] + [h.size - data_start]
    size_of = dict(zip(sorted(o for *_, o in infos), ends))
    tensors = [(name, dims, t, size_of[o] - o) for name, dims, t, o in infos]
    return meta, tensors


def summary(by_class, meta, n_layers):
    gb = lambda n: f"{n / 1e9:.2f}"
    experts = int(next((v for k, v in meta.items() if k.endswith(".expert_count")), 0) or 0)
    used = int(next((v for k, v in meta.items() if k.endswith(".expert_used_count")), 0) or 0)
    parts = defaultdict(int)
    for (cls, _), (_, n) in by_class.items():
        if cls == "per_layer_token_embd.weight":
            parts["ngram"] += n
        elif cls == "token_embd.weight":
            parts["embd"] += n
        elif ".nextn." in cls or cls.startswith("mtp"):
            parts["mtp"] += n
        elif re.match(r"blk\.N\.ffn_(gate|up|down|gate_up)_exps", cls):
            parts["experts"] += n
        else:
            parts["dense"] += n
    total = sum(parts.values())
    per_token = parts["dense"] + (parts["experts"] * used / experts if experts else parts["experts"])
    print("\nsummary\tGB")
    print(f"layers\t{n_layers}\nexperts\t{experts} (used {used})")
    print(f"total\t{gb(total)}\nngram_table\t{gb(parts['ngram'])}\nmtp\t{gb(parts['mtp'])}")
    print(f"on_gpu (total - ngram_table)\t{gb(total - parts['ngram'])}")
    print(f"routed_experts\t{gb(parts['experts'])}\ndense (all but embeddings, experts, ngram, mtp)\t{gb(parts['dense'])}")
    print(f"read_per_token (dense + experts x used/count)\t{gb(per_token)}")


def main():
    names = type_names()
    by_class = defaultdict(lambda: [0, 0])
    by_type = defaultdict(int)
    shapes = {}
    meta_all = {}
    for path in sys.argv[1:]:
        meta, tensors = read(path)
        meta_all.update(meta)
        print(f"== {os.path.basename(path)}: {len(tensors)} tensors")
        for k, v in meta.items():
            if not k.startswith("tokenizer.ggml.") or not isinstance(v, str) or len(v) < 80:
                print(f"  {k} = {v}")
        for name, dims, t, n in tensors:
            cls = re.sub(r"^blk\.\d+\.", "blk.N.", name)
            tname = names.get(t, f"type{t}")
            by_class[(cls, tname)][0] += 1
            by_class[(cls, tname)][1] += n
            by_type[tname] += n
            shapes.setdefault(cls, "x".join(str(d) for d in dims))
    print("\nclass\tformat\tcount\tshape\tGB")
    for (cls, tname), (count, n) in sorted(by_class.items(), key=lambda kv: -kv[1][1]):
        print(f"{cls}\t{tname}\t{count}\t{shapes[cls]}\t{n / 1e9:.3f}")
    print("\nformat\tGB")
    for tname, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"{tname}\t{n / 1e9:.3f}")
    print(f"total\t{sum(by_type.values()) / 1e9:.3f}")
    n_layers = next((v for k, v in meta_all.items() if k.endswith(".block_count")), "?")
    summary(by_class, meta_all, n_layers)


if __name__ == "__main__":
    main()
