#!/usr/bin/env python3
# Merges each layer's attn_q, attn_k and attn_v weights into one attn_qkv tensor (rows q, then k, then v), which
# llama.cpp's loader accepts for Flint and many other architectures. The model computes exactly the same thing, but
# every token runs one matrix-vector product (and one activation quantization) per layer instead of three, which saves
# ~200 kernel launches per token on Flint. Only layers whose q, k and v share one format are merged; the bytes are
# copied, nothing is requantized.
#
#   fuse_qkv.py <in.gguf> <out.gguf>
# Needs: numpy and gguf-py from the llama.cpp source tree (PYTHONPATH=<llama.cpp>/gguf-py).
import re, sys
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
T = {t.name: t for t in r.tensors}

fused, skipped = {}, []
for name in T:
    m = re.match(r'blk\.(\d+)\.attn_q\.weight$', name)
    if not m:
        continue
    b = f'blk.{m.group(1)}.'
    q, k, v = T[b + 'attn_q.weight'], T[b + 'attn_k.weight'], T[b + 'attn_v.weight']
    if not (q.tensor_type == k.tensor_type == v.tensor_type) or any(b + f'attn_{x}.bias' in T for x in 'qkv'):
        skipped.append(m.group(1))
        continue
    fused[b] = (q, k, v)

def out_tensors():
    """(name, data, type) in the original order, with q/k/v replaced by attn_qkv at q's position"""
    for t in r.tensors:
        b = t.name[: t.name.index('attn_') ] if '.attn_' in t.name else None
        if b in fused and t.name.endswith(('attn_k.weight', 'attn_v.weight')):
            continue
        if b in fused and t.name.endswith('attn_q.weight'):
            q, k, v = fused[b]
            yield b + 'attn_qkv.weight', (q, k, v), q.tensor_type
        else:
            yield t.name, (t,), t.tensor_type

def shape_of(parts):
    """numpy shape, dtype and size of the concatenation along the row axis (axis 0 of the numpy view)"""
    d0 = parts[0].data
    rows = sum(p.data.shape[0] for p in parts)
    return (rows, *d0.shape[1:]), d0.dtype, sum(p.data.nbytes for p in parts)

w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
for name, parts, ttype in out_tensors():
    shape, dtype, nbytes = shape_of(parts)
    w.add_tensor_info(name, shape, dtype, nbytes, ttype)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for name, parts, ttype in out_tensors():
    data = parts[0].data if len(parts) == 1 else np.concatenate([p.data for p in parts], axis=0)
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print(f'fused q/k/v in {len(fused)} layers; left separate (formats differ or biases present): {skipped or "none"}')
