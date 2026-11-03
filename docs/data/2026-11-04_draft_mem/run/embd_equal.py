#!/usr/bin/env python3
"""Does a draft model hold the same token embeddings (and output head) as its target?

Compares two GGUF files tensor by tensor on the CPU, for Q564 (the MTP draft reads the
target's token embeddings instead of loading its own copy):

  - the vocabulary: token strings, types and merges hashed in both files
  - token_embd.weight and output.weight: type, shape, bytes; when the types differ, both
    are dequantized row block by row block and compared per row (cosine, relative RMS
    error), against a shuffled pairing of the same rows as the "unrelated" baseline

Usage (gguf-py from a llama.cpp tree on PYTHONPATH, numpy installed):
  DRAFT=<draft.gguf> TARGET=<target shard 1 .gguf> [TARGET_SHARDS="<shard 2> ..."] \
  [ROWS=8192] [OUT=<dir>] python3 embd_equal.py
Writes <OUT>/embd_equal.md (summary) and <OUT>/embd_equal_rows.tsv (per row block).
"""

import hashlib
import os
import sys
import time

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def field_str(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    if v.dtype == np.uint8:
        return bytes(v).decode("utf-8", "replace")
    return v.tolist()[0] if len(v) == 1 else v.tolist()


def field_hash(r, key):
    f = r.fields.get(key)
    if f is None:
        return None, 0
    h = hashlib.sha256()
    for i in f.data:
        h.update(bytes(f.parts[i]))
        h.update(b"\0")
    return h.hexdigest()[:16], len(f.data)


def find_tensor(readers, name):
    for r in readers:
        for t in r.tensors:
            if t.name == name:
                return t
    return None


def rows_f32(t, r0, r1):
    data = t.data[r0:r1]
    qt = GGMLQuantizationType(t.tensor_type)
    if qt in (GGMLQuantizationType.F32,):
        return np.asarray(data, dtype=np.float32)
    if qt == GGMLQuantizationType.F16:
        return np.asarray(data, dtype=np.float16).astype(np.float32)
    return dequantize(np.asarray(data), qt).astype(np.float32)


def compare(name, a, b, rows, out_rows):
    lines = [f"### `{name}`", ""]
    qa, qb = GGMLQuantizationType(a.tensor_type).name, GGMLQuantizationType(b.tensor_type).name
    sa, sb = [int(x) for x in a.shape], [int(x) for x in b.shape]
    lines.append(f"- draft: {qa} {sa}, {a.n_bytes:,} bytes; target: {qb} {sb}, {b.n_bytes:,} bytes")
    if sa != sb:
        lines.append("- **shapes differ**: the draft cannot read the target's tensor")
        return lines, False
    if a.tensor_type == b.tensor_type:
        same = np.array_equal(np.asarray(a.data).view(np.uint8), np.asarray(b.data).view(np.uint8))
        lines.append(f"- same type; bytes identical: **{same}**")
        if same:
            return lines, True
    n_rows = sa[1]
    rng = np.random.default_rng(0)
    cos_all, rel_all, cos_shuf = [], [], []
    max_abs = 0.0
    t0 = time.time()
    for r0 in range(0, n_rows, rows):
        r1 = min(n_rows, r0 + rows)
        x = rows_f32(a, r0, r1)
        y = rows_f32(b, r0, r1)
        nx = np.linalg.norm(x, axis=1)
        ny = np.linalg.norm(y, axis=1)
        dot = np.einsum("ij,ij->i", x, y)
        cos = dot / np.maximum(nx * ny, 1e-30)
        rel = np.linalg.norm(x - y, axis=1) / np.maximum(nx, 1e-30)
        perm = rng.permutation(r1 - r0)
        cs = np.einsum("ij,ij->i", x, y[perm]) / np.maximum(nx * ny[perm], 1e-30)
        max_abs = max(max_abs, float(np.max(np.abs(x - y))))
        cos_all.append(cos)
        rel_all.append(rel)
        cos_shuf.append(cs)
        out_rows.append((name, r0, r1, float(np.mean(cos)), float(np.min(cos)), float(np.mean(rel)),
                         float(np.max(rel)), float(np.mean(cs))))
        if (r0 // rows) % 8 == 0:
            print(f"{name}: rows {r1}/{n_rows} ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    cos = np.concatenate(cos_all)
    rel = np.concatenate(rel_all)
    cs = np.concatenate(cos_shuf)
    q = lambda v, p: float(np.percentile(v, p))
    lines += [
        f"- per-row cosine, draft vs target: mean {np.mean(cos):.5f}, min {np.min(cos):.5f}, "
        f"p1 {q(cos, 1):.5f}, p50 {q(cos, 50):.5f}",
        f"- per-row relative RMS error |d - t| / |d|: mean {np.mean(rel):.4f}, p50 {q(rel, 50):.4f}, "
        f"p99 {q(rel, 99):.4f}, max {np.max(rel):.4f}; largest element difference {max_abs:.4g}",
        f"- baseline, each draft row against a shuffled target row: mean cosine {np.mean(cs):.5f}",
        f"- rows with cosine < 0.9: {int(np.sum(cos < 0.9))} of {n_rows}",
    ]
    return lines, bool(np.min(cos) > 0.9)


def main():
    draft = os.environ["DRAFT"]
    target = os.environ["TARGET"]
    shards = os.environ.get("TARGET_SHARDS", "").split()
    rows = int(os.environ.get("ROWS", "8192"))
    out = os.environ.get("OUT", ".")
    os.makedirs(out, exist_ok=True)

    rd = GGUFReader(draft)
    rts = [GGUFReader(p) for p in [target] + shards]
    rt = rts[0]

    md = ["# Draft vs target: token embeddings and output head", ""]
    md.append(f"- draft: `{os.path.basename(draft)}`, architecture `{field_str(rd, 'general.architecture')}`")
    md.append(f"- target: `{os.path.basename(target)}`" + (f" + {len(shards)} shards" if shards else "")
              + f", architecture `{field_str(rt, 'general.architecture')}`")
    for arch_r, who in ((rd, "draft"), (rt, "target")):
        arch = field_str(arch_r, "general.architecture")
        md.append(f"- {who}: embedding_length {field_str(arch_r, f'{arch}.embedding_length')}, "
                  f"block_count {field_str(arch_r, f'{arch}.block_count')}, "
                  f"nextn_predict_layers {field_str(arch_r, f'{arch}.nextn_predict_layers')}")
    md += ["", "## Vocabulary", ""]
    vocab_same = True
    for key in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
        a, b = field_str(rd, key), field_str(rt, key)
        md.append(f"- `{key}`: draft `{a}`, target `{b}`")
        vocab_same &= a == b
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"):
        (ha, na), (hb, nb) = field_hash(rd, key), field_hash(rt, key)
        md.append(f"- `{key}`: draft {na:,} entries sha256 `{ha}`, target {nb:,} entries `{hb}`: "
                  f"**{'same' if ha == hb else 'DIFFERENT'}**")
        vocab_same &= ha == hb
    md += ["", "## Tensors", ""]
    out_rows = []
    verdicts = {}
    for name in ("token_embd.weight", "output.weight"):
        a = find_tensor([rd], name)
        b = find_tensor(rts, name)
        if a is None or b is None:
            md += [f"### `{name}`", "", f"- missing in {'draft' if a is None else 'target'}", ""]
            continue
        lines, ok = compare(name, a, b, rows, out_rows)
        verdicts[name] = ok
        md += lines + [""]
    draft_names = sorted(t.name for t in rd.tensors if not t.name.startswith("blk."))
    md += ["## Draft tensors outside the blocks", "", ", ".join(f"`{n}`" for n in draft_names), ""]
    md += ["## Verdict", "", f"- vocabulary identical: **{vocab_same}**"]
    for k, v in verdicts.items():
        md.append(f"- `{k}`: {'same matrix (every row cosine > 0.9)' if v else 'not the same matrix'}")
    with open(os.path.join(out, "embd_equal.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out, "embd_equal_rows.tsv"), "w") as f:
        f.write("tensor\trow0\trow1\tcos_mean\tcos_min\trel_mean\trel_max\tcos_shuffled\n")
        for r in out_rows:
            f.write("\t".join(str(x) if isinstance(x, (int, str)) else f"{x:.6f}" for x in r) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
