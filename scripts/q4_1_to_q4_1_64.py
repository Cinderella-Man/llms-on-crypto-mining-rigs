#!/usr/bin/env python3
# Rewrites Q4_1 tensors whose neighbouring 32-weight blocks share their scale and offset as Q4_1_64 tensors (one scale
# and offset per 64 weights, 4.5 instead of 5.0 bits per weight), copying the 4-bit codes unchanged. Flint's official
# Q4_1 GGUF is group-64 INT4 stored as group-32, so this conversion is exact for every one of its Q4_1 tensors. Tensors
# where any block pair differs stay Q4_1. The output needs a llama.cpp built with patches/q4_1_64-type.diff.
#
#   q4_1_to_q4_1_64.py <in.gguf> <out.gguf>
# Needs numpy and the patched gguf-py (PYTHONPATH=<patched llama.cpp>/gguf-py), which knows the Q4_1_64 type.
import sys
import numpy as np
import gguf

Q41, Q4164 = gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q4_1_64
r = gguf.GGUFReader(sys.argv[1])
arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()

def pairs(t):
    """the tensor's blocks as (n_pairs, 2, 20) bytes, or None if it can't be converted exactly"""
    if t.tensor_type != Q41 or int(t.shape[0]) % 64:
        return None
    b = np.asarray(t.data).reshape(-1, 20).reshape(-1, 2, 20)
    return b if np.array_equal(b[:, 0, :4], b[:, 1, :4]) else None

plan = [(t, pairs(t)) for t in r.tensors]
w = gguf.GGUFWriter(sys.argv[2], arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
for t, p in plan:
    if p is None:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    else:
        rows = t.data.shape[:-1]; row_bytes = t.data.shape[-1] // 20 // 2 * 36
        w.add_tensor_info(t.name, (*rows, row_bytes), np.dtype(np.uint8), int(np.prod(rows)) * row_bytes, Q4164)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
n = kept = 0
for t, p in plan:
    if p is None:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        kept += t.tensor_type == Q41
        continue
    out = np.concatenate([p[:, 0, :4], p[:, 0, 4:], p[:, 1, 4:]], axis=1)    # dm, first half's nibbles, second half's
    w.write_tensor_data(out.reshape(*t.data.shape[:-1], -1), tensor_endianess=r.endianess)
    n += 1
w.close()
print(f'converted {n} Q4_1 tensors to Q4_1_64; {kept} Q4_1 tensors kept (block pairs differ or rows not a multiple of 64)')
