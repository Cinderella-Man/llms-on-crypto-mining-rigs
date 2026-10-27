#!/usr/bin/env python3
# Q107 (2026-10-24): G3g and K2hg = G3 and K2h with every attention tensor (q, k, v, output) of Flint's 14 global
# layers (no sliding window, no RoPE) in Q8_0, quantized from the BF16 GGUF like the rest of the mixes.
# Steps: check that gguf-py's Q8_0
# reproduces llama-quantize's bytes on tensors the mixes already hold in Q8_0; write a donor GGUF with the 56 tensors;
# compose_gguf.py; check every output tensor is the source's or the donor's, byte for byte.
# Env: BF16, G3, K2H (input GGUFs), OUT (folder for the donor and the two files), COMPOSE (default
# scripts/compose_gguf.py, run from the repository root); needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import os, re, subprocess, sys
import numpy as np
import gguf
from gguf.quants import quantize

BF16 = os.environ['BF16']        # flint-BF16.gguf
OUT = os.environ['OUT']          # output folder
GLOBAL = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
NAMES = [f'blk.{l}.attn_{k}.weight' for l in GLOBAL for k in ('q', 'k', 'v', 'output')]
SRC = {'G3g': os.environ['G3'], 'K2hg': os.environ['K2H']}   # the two source mixes
COMPOSE = os.environ.get('COMPOSE', 'scripts/compose_gguf.py')
Q8 = gguf.GGMLQuantizationType.Q8_0
os.makedirs(OUT, exist_ok=True)

rb = gguf.GGUFReader(BF16)
bt = {t.name: t for t in rb.tensors}


def to_q8(t):
    x = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(int(t.shape[1]), -1)
    return quantize(x, Q8)


# 1. gguf-py Q8_0 against llama-quantize's, on every global attention tensor G3 already holds in Q8_0
g3 = {t.name: t for t in gguf.GGUFReader(SRC['G3g']).tensors}
checked = 0
for n in NAMES:
    if g3[n].tensor_type == Q8:
        mine = to_q8(bt[n]).reshape(-1)
        if not np.array_equal(mine, np.asarray(g3[n].data).reshape(-1)):
            sys.exit(f'{n}: gguf-py Q8_0 differs from llama-quantize\'s ({(mine != np.asarray(g3[n].data).reshape(-1)).sum()} bytes)')
        checked += 1
print(f'gguf-py Q8_0 = llama-quantize Q8_0 on {checked} tensors', flush=True)

# 2. donor
donor = os.path.join(OUT, 'q107-global-attn-q8_0-donor.gguf')
w = gguf.GGUFWriter(donor, arch=rb.get_field(gguf.Keys.General.ARCHITECTURE).contents())
data = {n: to_q8(bt[n]) for n in NAMES}
for n in NAMES:
    w.add_tensor_info(n, data[n].shape, np.dtype(np.uint8), data[n].nbytes, Q8)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for n in NAMES:
    w.write_tensor_data(data[n])
w.close()
print(f'donor: {donor}', flush=True)

# 3./4. compose and check
rule = r'blk\.(' + '|'.join(map(str, GLOBAL)) + r')\.attn_(q|k|v|output)\.weight=' + donor
for name, src in SRC.items():
    out = os.path.join(OUT, f'flint-q107-{name}.gguf')
    open(out + '.partial', 'w').close()
    subprocess.run([sys.executable, COMPOSE, src, out, rule], check=True)
    rs, ro = gguf.GGUFReader(src), gguf.GGUFReader(out)
    so = {t.name: t for t in ro.tensors}
    n_src = n_don = 0
    for t in rs.tensors:
        o = so[t.name]
        want = data[t.name] if t.name in data else np.asarray(t.data)
        wt = Q8 if t.name in data else t.tensor_type
        if o.tensor_type != wt or not np.array_equal(np.asarray(o.data).reshape(-1), want.reshape(-1)):
            sys.exit(f'{out}: {t.name} is neither the source\'s nor the donor\'s')
        n_don += t.name in data
        n_src += t.name not in data
    changed = sum(1 for n in NAMES if {t.name: t for t in rs.tensors}[n].tensor_type != Q8)
    print(f'{out}: {n_src} tensors from {os.path.basename(src)}, {n_don} from the donor ({changed} changed type), checked', flush=True)
    os.remove(out + '.partial')
    open(out + '.ready', 'w').close()
print('Q107_FILES_DONE')
