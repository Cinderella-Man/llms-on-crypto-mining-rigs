#!/usr/bin/env python3
# C861 (a): which tensors of a GGUF file are Q4_K, Q5_K or Q6_K (the types GGML_CUDA_MMQ_KQ_FP32SCALE changes) or
# Q3_K (an int scale multiply too, not changed), grouped by role (the name with its layer number replaced by N).
# Reads the header only. Usage: kquant_census.py GGUFPY_DIR LABEL FILE [FILE ...]  (all shards of one model)
# Prints: per type the tensor count and bytes, then each K-quant role with its count, type and shape.
import sys, re, collections, types
sys.path.insert(0, sys.argv[1])
try:
    import yaml  # noqa: F401  gguf-py imports it for model cards; reading tensors does not use it
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
from gguf import GGUFReader  # noqa: E402

label, files = sys.argv[2], sys.argv[3:]
by_type = collections.Counter()
bytes_type = collections.Counter()
roles = collections.OrderedDict()
total = 0
for f in files:
    r = GGUFReader(f)
    for t in r.tensors:
        ty = t.tensor_type.name
        by_type[ty] += 1
        bytes_type[ty] += int(t.n_bytes)
        total += int(t.n_bytes)
        if ty in ("Q3_K", "Q4_K", "Q5_K", "Q6_K"):
            role = re.sub(r"\.\d+\.", ".N.", t.name)
            key = (role, ty, tuple(int(x) for x in t.shape))
            roles[key] = roles.get(key, 0) + 1

print(f"## {label}")
print("type\ttensors\tbytes\tshare")
for ty, n in sorted(by_type.items(), key=lambda kv: -bytes_type[kv[0]]):
    print(f"{ty}\t{n}\t{bytes_type[ty]}\t{100.0 * bytes_type[ty] / total:.2f}%")
print("k-quant role\ttype\tshape\ttensors")
for (role, ty, shape), n in roles.items():
    print(f"{role}\t{ty}\t{'x'.join(map(str, shape))}\t{n}")
print()
