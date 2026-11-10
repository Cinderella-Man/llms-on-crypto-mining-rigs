#!/usr/bin/env python3
"""compare_heads.py: two MTP draft GGUFs of one architecture, tensor by tensor.

    PYTHONPATH=<llama.cpp>/gguf-py python3 compare_heads.py REF NEW [--exact NEW_BF16] [--target SHARD1] [--out TSV] \
        [--same-types]

REF is the draft in use, NEW the new draft in the same types, NEW_BF16 the new draft before llama-quantize (the
checkpoint's own values). Every tensor is decoded to float32 (gguf-py's dequantize) and compared as one long vector:

  cos           the cosine between the two
  rel           |a - b| / |a|, with a the first of the pair
  max_abs       the largest difference of one value

per pair: REF against NEW; with --exact also REF against NEW_BF16 (how far the two heads are apart, with only REF's
rounding in it) and NEW against NEW_BF16 (what the rounding of these types costs: the floor under the other two).
A head that is the reference's own weights shows the same numbers for REF against NEW_BF16 as for NEW against
NEW_BF16; a head that was trained further shows more. `same_values` counts the decoded values of REF and NEW that
are equal bit for bit, `same_bytes` says whether the two tensors' stored bytes are equal, and `same_blocks` counts
the type's blocks (256 values for the K types, 32 for Q8_0 and IQ4_NL, one value for F32) whose bytes are equal: a
block rounded twice from the same values with the same scales comes out the same, one rounded from other values
does not. Tensors over 2^27 values are read in slices along their first axis.

--target: the first shard of the model the draft serves; prints whether the tokenizer arrays and ids of REF equal
the target's (the new file copies REF's key/value pairs).
"""
import argparse
import sys

import numpy as np

import gguf
from gguf import quants

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ref")
ap.add_argument("new")
ap.add_argument("--exact")
ap.add_argument("--target")
ap.add_argument("--out")
ap.add_argument("--same-types", action="store_true", help="end with an error when a tensor's type differs")
A = ap.parse_args()


def tensors(path):
    r = gguf.GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def decode(t, lo, hi):
    """float32 values of the slices lo..hi along the tensor's first (checkpoint-order) axis."""
    d = t.data[lo:hi] if t.data.ndim > 1 else t.data
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.asarray(d, dtype=np.float32).ravel()
    if t.tensor_type == gguf.GGMLQuantizationType.F16:
        return np.asarray(d).astype(np.float32).ravel()
    return quants.dequantize(np.asarray(d), t.tensor_type).astype(np.float32, copy=False).ravel()


def pair(a, b):
    """cos, rel, max_abs, values, values equal bit for bit, of two tensors of one shape."""
    n0 = a.data.shape[0] if a.data.ndim > 1 else 1
    count = int(np.prod([int(x) for x in a.shape]))
    step = max(1, n0 // max(1, count >> 27)) if a.data.ndim > 1 else 1
    ab = aa = bb = dd = 0.0
    mx = 0.0
    same = 0
    for lo in range(0, n0, step):
        x, y = decode(a, lo, lo + step), decode(b, lo, lo + step)
        xd, yd = x.astype(np.float64), y.astype(np.float64)
        ab += float(xd @ yd)
        aa += float(xd @ xd)
        bb += float(yd @ yd)
        d = xd - yd
        dd += float(d @ d)
        mx = max(mx, float(np.abs(d).max()))
        same += int((x.view(np.uint32) == y.view(np.uint32)).sum())
    return ab / (aa * bb) ** 0.5, (dd / aa) ** 0.5, mx, count, same


ref_r, ref = tensors(A.ref)
new_r, new = tensors(A.new)
exact = tensors(A.exact)[1] if A.exact else None
if sorted(ref) != sorted(new):
    raise SystemExit(f"other tensor names: {sorted(set(ref) ^ set(new))}")

cols = ["tensor", "shape", "values", "ref_type", "new_type", "ref_new_cos", "ref_new_rel", "ref_new_max_abs"]
if exact:
    cols += ["ref_exact_cos", "ref_exact_rel", "ref_exact_max_abs", "new_exact_cos", "new_exact_rel",
             "new_exact_max_abs"]
cols += ["same_values", "same_bytes", "blocks", "same_blocks"]
lines =["\t".join(cols)]
for name, t in ref.items():
    n = new[name]
    if [int(x) for x in t.shape] != [int(x) for x in n.shape]:
        raise SystemExit(f"{name}: shape {list(t.shape)} against {list(n.shape)}")
    cos, rel, mx, count, same = pair(t, n)
    row = [name, "x".join(str(int(x)) for x in t.shape), str(count), t.tensor_type.name, n.tensor_type.name,
           f"{cos:.6f}", f"{rel:.5f}", f"{mx:.5g}"]
    if exact:
        e = exact[name]
        c1, r1, m1, _, _ = pair(t, e)
        c2, r2, m2, _, _ = pair(n, e)
        row += [f"{c1:.6f}", f"{r1:.5f}", f"{m1:.5g}", f"{c2:.6f}", f"{r2:.5f}", f"{m2:.5g}"]
    same_bytes = t.tensor_type == n.tensor_type and np.array_equal(t.data, n.data)
    row += [str(same), "yes" if same_bytes else "no"]
    if t.tensor_type == n.tensor_type:
        size = gguf.GGML_QUANT_SIZES[t.tensor_type][1]  # bytes of one block of the type
        eq = (np.ascontiguousarray(t.data).view(np.uint8).reshape(-1, size) ==
              np.ascontiguousarray(n.data).view(np.uint8).reshape(-1, size)).all(axis=1)
        row += [str(len(eq)), str(int(eq.sum()))]
    else:
        row += ["", ""]
    lines.append("\t".join(row))
    print(lines[-1], flush=True)
if A.out:
    with open(A.out, "w") as f:
        f.write("\n".join(lines) + "\n")

kv_ref = {k: v for k, v in ref_r.fields.items() if not k.startswith("GGUF.")}
kv_new = {k: v for k, v in new_r.fields.items() if not k.startswith("GGUF.")}
diff = sorted(k for k in set(kv_ref) | set(kv_new)
              if k not in kv_ref or k not in kv_new or kv_ref[k].contents() != kv_new[k].contents())
print(f"keys: {len(kv_ref)} in REF, {len(kv_new)} in NEW; with another value or in one file only: {diff}")
if A.target:
    tgt = gguf.GGUFReader(A.target).fields
    for k in sorted(k for k in kv_ref if k.startswith("tokenizer.") and k != "tokenizer.chat_template"):
        state = "missing in the target" if k not in tgt else \
            "the same" if kv_ref[k].contents() == tgt[k].contents() else "DIFFERENT"
        print(f"target {k}: {state}")
    k = "tokenizer.chat_template"
    if k in tgt and k in kv_ref:
        print(f"target {k}: {'the same' if kv_ref[k].contents() == tgt[k].contents() else 'different'}")
other = sorted(n for n in ref if ref[n].tensor_type != new[n].tensor_type)
print(f"types: {len(ref) - len(other)} of {len(ref)} tensors in REF's type; of another type: {other}")
print(f"sizes: REF {ref_r.data.nbytes:,} bytes, NEW {new_r.data.nbytes:,} bytes")
sys.stdout.flush()
if A.same_types and other:
    sys.exit(1)
