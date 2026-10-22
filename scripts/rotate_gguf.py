#!/usr/bin/env python3
# Folds orthogonal Hadamard rotations into a 16-bit Flint GGUF so that its weights quantize better, without changing
# what the model computes and without any llama.cpp change (QuaRot-style "R1" and "R2" rotations):
#
#   R1  rotates the residual stream by Q = D*H (random signs D, normalized 2048-point Walsh-Hadamard H). Every RMSNorm
#       weight is folded into the matrices that read the norm's output and set to 1, so the norms stay exact; the token
#       embedding and every matrix reading the residual get Q on their input side, every matrix writing to the
#       residual gets Q^T on its output side.
#   R2  (optional) rotates each attention value head by a 128-point Hadamard and undoes it on the input side of the
#       attention output projection (attention is linear in V, so this is exact too).
#   --online-down B  (optional) stores every FFN down-projection as W*H (block Hadamard of size B on its input) and
#       writes llama.cpp's `prism.hadamard` metadata, so llama.cpp applies H to the down-projection's input at run time
#       (one fast Walsh-Hadamard kernel per layer and branch). These inputs follow a SwiGLU, so no offline fold exists.
#       Needs a llama.cpp whose loader accepts the architecture for prism.hadamard (Flint needs a one-line patch).
#
# Rotated weights spread outliers across each row, which is what 4-bit block formats such as Q4_0 handle badly.
# The input must be a 16/32-bit GGUF (e.g. made by convert_hf_to_gguf.py --outtype bf16); quantize the output with
# llama-quantize as usual (compute a new importance matrix on the rotated model first: activations rotate too).
#
#   rotate_gguf.py <in.gguf> <out.gguf> [--no-r1] [--r2] [--online-down B] [--seed N]
# Needs: numpy, torch (CPU is enough), and gguf-py from the llama.cpp source tree (set PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, math, re, sys
import numpy as np
import torch
import gguf

p = argparse.ArgumentParser()
p.add_argument('inp'); p.add_argument('out')
p.add_argument('--no-r1', dest='r1', action='store_false', help='skip the residual-stream rotation')
p.add_argument('--r2', action='store_true', help='also rotate each value head and the attention output input side')
p.add_argument('--online-down', type=int, default=0, metavar='B', help='block size of the run-time Hadamard on down-projection inputs')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

r = gguf.GGUFReader(args.inp)
arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
if arch != 'flint':
    sys.exit(f'only checked for flint, not {arch}')
kv = lambda k: r.get_field(f'{arch}.{k}').contents()
n_embd, n_layer = kv('embedding_length'), kv('block_count')
head_dim = kv('attention.key_length')

T = {t.name: t for t in r.tensors}
FT = gguf.GGMLQuantizationType

def load(name):
    t = T[name]
    shape = [int(x) for x in reversed(t.shape)]
    if t.tensor_type == FT.BF16:
        u = torch.from_numpy(np.frombuffer(t.data, dtype=np.uint16).astype(np.int32)).reshape(shape)
        return (u << 16).view(torch.float32)
    if t.tensor_type in (FT.F32, FT.F16):
        return torch.from_numpy(np.asarray(t.data).astype(np.float32)).reshape(shape)
    sys.exit(f'{name}: type {t.tensor_type.name} is not 16/32-bit')

def store(name, x):
    """returns a numpy array in the tensor's original type"""
    t = T[name]
    x = x.contiguous()
    if t.tensor_type == FT.BF16:
        b = x.view(torch.int32)
        b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16   # round to nearest even
        return b.to(torch.int16).numpy().view(np.uint16).view(np.uint8).reshape(*x.shape[:-1], -1)
    if t.tensor_type == FT.F16:
        return x.to(torch.float16).numpy()
    return x.numpy()

def fwht(x, dim):
    """normalized Walsh-Hadamard transform along dim (size a power of two); H is symmetric and H*H = I"""
    x = x.movedim(dim, -1).contiguous()
    n = x.shape[-1]; h = 1
    while h < n:
        y = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        x = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return (x / math.sqrt(n)).movedim(-1, dim)

g = torch.Generator().manual_seed(args.seed)
D = torch.randint(0, 2, (n_embd,), generator=g).float() * 2 - 1      # residual signs
Dv = torch.randint(0, 2, (head_dim,), generator=g).float() * 2 - 1   # value-head signs (R2)

def rot_in(w, gamma=None):
    """W -> W diag(gamma) D H on the input (last) axis: the matrix now reads the rotated residual"""
    if gamma is not None:
        w = w * gamma
    return fwht(w * D, -1)

def rot_out(w):
    """W -> H D W on the output axis (second to last): the matrix now writes the rotated residual"""
    return fwht(w * D.view(-1, 1), -2)

def heads(w, axis, signs):
    """per-head D H on an axis of size n_heads*head_dim"""
    w = w.movedim(axis, -1)
    s = w.shape
    w = fwht(w.reshape(*s[:-1], s[-1] // head_dim, head_dim) * signs, -1).reshape(s)
    return w.movedim(-1, axis)

def norm_of(name):
    """the RMSNorm weight folded into a matrix that reads the norm's output, or None if the matrix isn't rotated"""
    if name == 'output.weight':
        return 'output_norm.weight'
    m = re.match(r'blk\.(\d+)\.(\w+)\.weight$', name)
    if not m:
        return None
    k = m.group(2)
    if k in ('attn_q', 'attn_k', 'attn_v'):
        return f'blk.{m.group(1)}.attn_norm.weight'
    if k in ('ffn_gate', 'ffn_up', 'ffn_gate_inp', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_gate_shexp', 'ffn_up_shexp'):
        return f'blk.{m.group(1)}.ffn_norm.weight'
    return None

WRITERS = ('attn_output', 'ffn_down', 'ffn_down_exps', 'ffn_down_shexp')
KEEP = ('attn_q_norm', 'attn_k_norm', 'attn_sinks')

DOWNS = ('ffn_down', 'ffn_down_exps', 'ffn_down_shexp')

def transform(name):
    """the tensor's new data in its original type, or None to copy it unchanged"""
    m = re.match(r'blk\.\d+\.(\w+)\.weight$', name)
    kind = m.group(1) if m else name
    w = None
    if args.r1:
        if name == 'token_embd.weight':
            w = fwht(load(name) * D, -1)
        elif name == 'output_norm.weight' or kind in ('attn_norm', 'ffn_norm'):
            w = torch.ones([int(x) for x in reversed(T[name].shape)])
        elif norm_of(name) is not None:
            w = rot_in(load(name), load(norm_of(name)))
        elif kind in WRITERS:
            w = rot_out(load(name))
        elif kind not in KEEP:
            sys.exit(f'unexpected tensor, not handled: {name}')
    if args.r2 and kind == 'attn_v':
        w = heads(load(name) if w is None else w, -2, Dv)     # rows of V: n_head_kv * head_dim
    if args.r2 and kind == 'attn_output':
        w = heads(load(name) if w is None else w, -1, Dv)     # input columns: n_head * head_dim
    if args.online_down and kind in DOWNS:
        w = load(name) if w is None else w
        n_in = w.shape[-1]
        w = fwht(w.reshape(*w.shape[:-1], n_in // args.online_down, args.online_down), -1).reshape(w.shape)
    return None if w is None else store(name, w)

w = gguf.GGUFWriter(args.out, arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
w.add_key_value('rotation.r1', args.r1, gguf.GGUFValueType.BOOL)
w.add_key_value('rotation.r2', args.r2, gguf.GGUFValueType.BOOL)
w.add_key_value('rotation.seed', args.seed, gguf.GGUFValueType.UINT32)
if args.online_down:
    downs = [t.name for t in r.tensors if re.match(r'blk\.\d+\.(ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$', t.name)]
    for t in downs:
        if int(T[t].shape[0]) % args.online_down:
            sys.exit(f'{t}: input size {int(T[t].shape[0])} is not a multiple of {args.online_down}')
    w.add_key_value('prism.hadamard.version', 1, gguf.GGUFValueType.UINT32)
    w.add_key_value('prism.hadamard.block_size', args.online_down, gguf.GGUFValueType.UINT32)
    w.add_key_value('prism.hadamard.transform', 'normalized-sylvester-walsh-hadamard', gguf.GGUFValueType.STRING)
    w.add_key_value('prism.hadamard.axis', 'input-last-dimension', gguf.GGUFValueType.STRING)
    w.add_key_value('prism.hadamard.sign_mode', 'identity', gguf.GGUFValueType.STRING)
    w.add_key_value('prism.hadamard.weight_names', downs, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for i, t in enumerate(r.tensors):
    d = transform(t.name)
    if d is not None:
        assert d.shape == t.data.shape and d.dtype == t.data.dtype, t.name
    w.write_tensor_data(t.data if d is None else d, tensor_endianess=r.endianess)
    if i % 50 == 0:
        print(f'{i}/{len(r.tensors)} {t.name}', flush=True)
w.close()
print('wrote', args.out)
