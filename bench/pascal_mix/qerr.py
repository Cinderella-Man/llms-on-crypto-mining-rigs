#!/usr/bin/env python3
"""How much error each weight type puts into Qwen3.8-Flash-Next's dense tensors, measured on the real tensors: for a
sample of layers, every dense matrix (attention and recurrent projections, the shared expert, PLE key, output layer)
of each given file is dequantized and compared with the same tensor of the reference (the Swift-1.5 Q8_0, the
reference of every KLD in this project), weighted by an importance matrix (llama-imatrix's mean squared activation
per input column, the weights llama-quantize itself minimises):
    noise(t, file) = sum_j imp_j sum_i (w_ij - ref_ij)^2,   power(t) = sum_j imp_j sum_i ref_ij^2,   rel = noise / power
where j runs over a row's inputs. Files hold different types for the same tensor (the published builds' own choices,
the Pascal mixes' requantizations), so a two-way fit log rel(t, type) = a(t) + c(type) gives every type's error
factor across tensors, with which bench/pascal_mix/vmix.py prices a type a tensor was never quantized to.
  python3 qerr.py --ref Q8_FIRST_SHARD --imatrix IMATRIX.gguf --file NAME=FIRST_SHARD [--file ...]
      [--layers 0,4,8,...] [--out-dir DIR]
Writes DIR/qerr.tsv (tensor, class, layer, file, type, weights, power, noise, rel) and DIR/types.tsv (type, factor
c relative to IQ4_XS, observations, the fit's RMS residual in log2 units). Needs numpy and gguf-py (PYTHONPATH: a
llama.cpp tree's gguf-py that knows every type in the files, Q2_0 included).
"""
import argparse, collections, glob, math, os, re

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

DENSE = re.compile(r"(^output\.weight$|\.(attn_(q|k|v|qkv|gate|output)|ssm_out|ple_key|ffn_(gate|up|down)_shexp)\.weight$)")


def tclass(name):
    if name == "output.weight":
        return "output"
    m = re.search(r"\.(attn_(?:q|k|v)|attn_output|attn_qkv|attn_gate|ssm_out|ple_key|ffn_(?:gate|up|down)_shexp)\.", name)
    return {"attn_q": "fa_qkv", "attn_k": "fa_qkv", "attn_v": "fa_qkv", "attn_output": "fa_out", "attn_qkv": "gdn_qkv",
            "attn_gate": "gdn_gate", "ssm_out": "gdn_out", "ple_key": "ple"}.get(m.group(1), "shexp")


class Model:
    def __init__(self, first):
        self.where = {}
        for p in sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-0000?-of-\1.gguf", first))):
            r = GGUFReader(p)
            for t in r.tensors:
                self.where[t.name] = t

    def weights(self, name):
        t = self.where[name]
        w = dequantize(t.data, t.tensor_type)
        return w.reshape(-1, int(t.shape[0])).astype(np.float64), t.tensor_type.name


def importance(path):
    r = GGUFReader(path)
    s2 = {t.name[:-len(".in_sum2")]: np.asarray(t.data, dtype=np.float64) for t in r.tensors if t.name.endswith(".in_sum2")}
    cnt = {t.name[:-len(".counts")]: float(np.asarray(t.data).reshape(-1)[0]) for t in r.tensors if t.name.endswith(".counts")}
    return {n: s2[n] / max(cnt.get(n, 1.0), 1.0) for n in s2}


def fit(rows):
    """log2 rel = a(tensor) + c(type), least squares by alternating means; c relative to IQ4_XS (or the first type)."""
    obs = [(r[0], r[4], math.log2(r[8])) for r in rows if r[8] > 0]
    types = sorted({o[1] for o in obs})
    a = collections.defaultdict(float)
    c = {t: 0.0 for t in types}
    for _ in range(200):
        by_t = collections.defaultdict(list)
        for n, t, y in obs:
            by_t[n].append(y - c[t])
        a = {n: sum(v) / len(v) for n, v in by_t.items()}
        by_c = collections.defaultdict(list)
        for n, t, y in obs:
            by_c[t].append(y - a[n])
        c = {t: sum(v) / len(v) for t, v in by_c.items()}
    ref = c.get("IQ4_XS", c[types[0]])
    res = collections.defaultdict(list)
    for n, t, y in obs:
        res[t].append(y - a[n] - c[t])
    return [(t, 2 ** (c[t] - ref), len(res[t]), math.sqrt(sum(x * x for x in res[t]) / len(res[t]))) for t in
            sorted(types, key=lambda t: c[t])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--file", action="append", required=True)
    ap.add_argument("--layers", default="0,1,3,4,8,11,12,16,20,24,28,32,36,40,44,47")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    layers = {int(x) for x in a.layers.split(",")}
    imp = importance(a.imatrix)
    ref = Model(a.ref)
    files = [(f.split("=", 1)[0], Model(f.split("=", 1)[1])) for f in a.file]
    names = [n for n in ref.where if DENSE.search(n) and (n == "output.weight" or
             (re.match(r"blk\.(\d+)\.", n) and int(re.match(r"blk\.(\d+)\.", n)[1]) in layers))]
    rows = []
    out = open(os.path.join(a.out_dir, "qerr.tsv"), "w")
    out.write("tensor\tclass\tlayer\tfile\ttype\tweights\tpower\tnoise\trel\n")
    for name in names:
        w0, _ = ref.weights(name)
        im = imp.get(name)
        if im is None or im.shape[0] != w0.shape[1]:
            im = np.ones(w0.shape[1])
        power = float(((w0 * w0) @ im).sum())
        m = re.match(r"blk\.(\d+)\.", name)
        seen = set()
        for fname, mod in files:
            if name not in mod.where:
                continue
            w, typ = mod.weights(name)
            if typ == "Q8_0" or (typ, mod.where[name].data.tobytes()[:64]) in seen:
                continue  # the reference's own type, or the same tensor as a file already compared
            seen.add((typ, mod.where[name].data.tobytes()[:64]))
            d = w - w0
            noise = float(((d * d) @ im).sum())
            r = (name, tclass(name), m[1] if m else "", fname, typ, w0.size, power, noise, noise / power)
            rows.append(r)
            out.write("\t".join(str(x) for x in r[:6]) + f"\t{power:.6g}\t{noise:.6g}\t{r[8]:.6g}\n")
            out.flush()
    out.close()
    with open(os.path.join(a.out_dir, "types.tsv"), "w") as f:
        f.write("type\tfactor_vs_iq4_xs\tobservations\trms_residual_log2\n")
        for t, fac, n, rms in fit(rows):
            f.write(f"{t}\t{fac:.4g}\t{n}\t{rms:.3f}\n")
    print(open(os.path.join(a.out_dir, "types.tsv")).read())


if __name__ == "__main__":
    main()
