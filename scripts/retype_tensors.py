#!/usr/bin/env python3
# Converts selected tensors of a GGUF to another format and copies every other tensor byte for byte. Unlike
# llama-quantize, which requantizes the whole file, this keeps a distillation-tuned quant (such as Flint's makers'
# Q4_1) exactly as it is except for the tensors you name, e.g. widening the expert down-projections to Q8_0 so that
# llama.cpp's rescale step can be switched off safely.
#
#   retype_tensors.py <in.gguf> <out.gguf> <regex>=<type>[@<from type>] [...]
#   e.g. retype_tensors.py flint-int4-q4_1.gguf flint-q4_1-downq8.gguf 'ffn_down_exps=Q8_0'
#        retype_tensors.py qwen-Q4_0.gguf qwen-Q4_0-noK.gguf '.=Q8_0@Q6_K'   (every Q6_K tensor to Q8_0, nothing else)
# Types: any format gguf-py can quantize (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, ...); the first matching
# pattern wins (matched with re.search against the tensor name). Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import re, sys
import numpy as np
import gguf
from gguf.quants import quantize, dequantize

src, dst, rules = sys.argv[1], sys.argv[2], sys.argv[3:]
def rule(x):
    k, v = x.split('=', 1)
    v, _, frm = v.partition('@')
    return re.compile(k), gguf.GGMLQuantizationType[v.upper()], gguf.GGMLQuantizationType[frm.upper()] if frm else None
rules = [rule(x) for x in rules]
r = gguf.GGUFReader(src)
arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()

def target(t):
    for pat, ty, frm in rules:
        if pat.search(t.name) and (frm is None or t.tensor_type == frm):
            return ty
    return None

def converted(t, ty):
    """the tensor in format ty, with the numpy shape GGUFWriter expects"""
    x = dequantize(t.data, t.tensor_type)                     # float32, shape (..., ne0)
    if ty == gguf.GGMLQuantizationType.F32:
        return x
    if ty == gguf.GGMLQuantizationType.F16:
        return x.astype(np.float16)
    return quantize(x, ty)

plan = []
for t in r.tensors:
    ty = target(t)
    if ty is None or ty == t.tensor_type:
        plan.append((t, None))
        continue
    blk = gguf.GGML_QUANT_SIZES[ty][0]
    if int(t.shape[0]) % blk:
        sys.exit(f'{t.name}: row length {int(t.shape[0])} is not a multiple of {ty.name} blocks ({blk})')
    plan.append((t, ty))

w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
for t, ty in plan:
    if ty is None:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    else:
        ne = [int(x) for x in reversed(t.shape)]
        bs, ts = gguf.GGML_QUANT_SIZES[ty]
        nbytes = int(np.prod(ne)) // bs * ts
        shape = (*ne[:-1], ne[-1] // bs * ts) if ty not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16) else tuple(ne)
        dtype = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.F16: np.float16}.get(ty, np.uint8)
        w.add_tensor_info(t.name, shape, np.dtype(dtype), nbytes, ty)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
n = 0
for t, ty in plan:
    w.write_tensor_data(t.data if ty is None else converted(t, ty), tensor_endianess=r.endianess)
    if ty is not None:
        n += 1
        print(f'{t.name}: {t.tensor_type.name} -> {ty.name}', flush=True)
w.close()
print(f'converted {n} tensors, copied the rest')
