#!/usr/bin/env python3
# C865: the types and shapes of a GGUF file's FFN gate and up weights, routed (_exps) and shared (_shexp), grouped by
# role (the layer number replaced by N). The shared-expert match (one launch for both triples) needs the routed and
# the shared weights of one type and the same row size. Reads the header only, standard library only (no numpy);
# prints the ggml type id, named from the GGML_TYPE_ enum of the ggml.h passed in.
# Usage: ffn_types.py GGML_H FILE [FILE ...]  (all shards of one model)
import sys, re, struct, collections

names = {}
for m in re.finditer(r"GGML_TYPE_(\w+)\s*=\s*(\d+)", open(sys.argv[1]).read()):
    names.setdefault(int(m.group(2)), m.group(1))

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

def rd_str(f):
    return f.read(rd(f, "Q")).decode("utf-8", "replace")

def skip_val(f, t):
    if t == 8:
        f.seek(rd(f, "Q"), 1)
    elif t == 9:
        et, n = rd(f, "I"), rd(f, "Q")
        if et in SIZES:
            f.seek(SIZES[et] * n, 1)
        else:
            for _ in range(n):
                skip_val(f, et)
    else:
        f.seek(SIZES[t], 1)

roles = collections.Counter()
per_layer = collections.defaultdict(dict)  # layer -> {role suffix: type}
for path in sys.argv[2:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        rd(f, "I")  # version
        n_tensors, n_kv = rd(f, "Q"), rd(f, "Q")
        for _ in range(n_kv):
            rd_str(f)
            skip_val(f, rd(f, "I"))
        for _ in range(n_tensors):
            name = rd_str(f)
            dims = [rd(f, "Q") for _ in range(rd(f, "I"))]
            ty = rd(f, "I")
            rd(f, "Q")  # offset
            if re.search(r"ffn_(gate|up)(_exps|_shexp)?\.weight$", name):
                role = re.sub(r"\.\d+\.", ".N.", name)
                roles[(role, names.get(ty, str(ty)), "x".join(map(str, dims)))] += 1
                il = re.search(r"\.(\d+)\.", name)
                if il:
                    per_layer[int(il.group(1))][role.split(".")[2]] = names.get(ty, str(ty))
print("role\ttype\tshape\ttensors")
for (role, ty, shape), n in sorted(roles.items()):
    print(f"{role}\t{ty}\t{shape}\t{n}")
print()
print("layers whose gate and up share a type (the triple can fuse): routed, shared")
for kind in ("exps", "shexp"):
    pairs = collections.Counter()
    for il, d in per_layer.items():
        g, u = d.get(f"ffn_gate_{kind}"), d.get(f"ffn_up_{kind}")
        if g and u:
            pairs[(g, u)] += 1
    print(kind + "\t" + ", ".join(f"gate {g} up {u}: {n}" for (g, u), n in sorted(pairs.items())))
