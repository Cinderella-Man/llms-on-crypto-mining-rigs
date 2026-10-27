#!/usr/bin/env python3
# Q127 swap probes (2026-10-24; long_context_mix.md, ranked table row 2): which tensor class carries UD-Q4_K_XL's edge on tool-schema
# tokens at 8k. Three probe files from G3 (Q8_0 head, F32 router; probes, not candidates):
#   SW-A = G3 with UD's attention (q, k, v, output of all 49 layers: 196 tensors, UD's own bytes);
#   SW-X = G3 with UD's routed experts (gate, up, down of layers 2-48: 141 tensors; score with the rescale on);
#   G3a  = G3 with Q8_0 for the 19 sliding-layer attention tensors UD keeps at Q8_0 and G3 at Q5_0 (q in 27, 28, 30-32,
#          34, 36, 38, 40, 42, 44, 47; output in 34, 36, 38, 40, 42, 44, 47), quantized from the BF16 GGUF with gguf-py.
# gguf-py's Q8_0 is first checked against llama-quantize's bytes on sliding-layer attention G3 already holds in Q8_0.
# compose_gguf.py writes each file; every output tensor is then checked to be G3's or the donor's, byte for byte.
# Env: BF16, G3, UD (input GGUFs), OUT (output folder), COMPOSE (default scripts/compose_gguf.py, run from the
# repository root); needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).   make_q127.py [SW-A] [SW-X] [G3a]
import os, re, subprocess, sys
import numpy as np
import gguf
from gguf.quants import quantize

BF16 = os.environ['BF16']      # flint-BF16.gguf
G3 = os.environ['G3']          # flint-q73-G3.gguf
UD = os.environ['UD']          # Unsloth's flint-UD-Q4_K_XL.gguf
OUT = os.environ['OUT']        # output folder
COMPOSE = os.environ.get('COMPOSE', 'scripts/compose_gguf.py')
GLOBAL = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48}
G3A = [f'blk.{l}.attn_q.weight' for l in (27, 28, 30, 31, 32, 34, 36, 38, 40, 42, 44, 47)] + \
      [f'blk.{l}.attn_output.weight' for l in (34, 36, 38, 40, 42, 44, 47)]
Q8 = gguf.GGMLQuantizationType.Q8_0
os.makedirs(OUT, exist_ok=True)
only = sys.argv[1:]  # optional subset of probe names

bt = {t.name: t for t in gguf.GGUFReader(BF16).tensors}
g3 = {t.name: t for t in gguf.GGUFReader(G3).tensors}
ud = {t.name: t for t in gguf.GGUFReader(UD).tensors}


def to_q8(t):
    x = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(int(t.shape[1]), -1)
    return quantize(x, Q8)


assert not set(int(n.split('.')[1]) for n in G3A) & GLOBAL, 'G3a must touch sliding layers only'
for n in G3A:
    assert g3[n].tensor_type != Q8 and ud[n].tensor_type == Q8, f'{n}: G3 {g3[n].tensor_type.name}, UD {ud[n].tensor_type.name}'

PROBES = {
    'SW-A': (r'^blk\.\d+\.attn_(q|k|v|output)\.weight$', UD),
    'SW-X': (r'^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$', UD),
    'G3a': ('^(' + '|'.join(re.escape(n) for n in G3A) + ')$', os.path.join(OUT, 'q127-g3a-q8_0-donor.gguf')),
}
if not only or 'G3a' in only:
    sliding_q8 = [n for n, t in g3.items() if re.match(r'^blk\.\d+\.attn_(q|output)\.weight$', n) and t.tensor_type == Q8
                  and int(n.split('.')[1]) not in GLOBAL]
    for n in sliding_q8:
        mine = to_q8(bt[n]).reshape(-1)
        if not np.array_equal(mine, np.asarray(g3[n].data).reshape(-1)):
            sys.exit(f'{n}: gguf-py Q8_0 differs from llama-quantize\'s')
    print(f'gguf-py Q8_0 = llama-quantize Q8_0 on {len(sliding_q8)} sliding-layer attention tensors of G3', flush=True)
    donor = PROBES['G3a'][1]
    data = {n: to_q8(bt[n]) for n in G3A}
    w = gguf.GGUFWriter(donor, arch=gguf.GGUFReader(G3).get_field(gguf.Keys.General.ARCHITECTURE).contents())
    for n in G3A:
        w.add_tensor_info(n, data[n].shape, np.dtype(np.uint8), data[n].nbytes, Q8)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for n in G3A:
        w.write_tensor_data(data[n])
    w.close()
    print(f'donor: {donor} ({len(G3A)} tensors)', flush=True)

for name, (rx, donor) in PROBES.items():
    if only and name not in only:
        continue
    out = os.path.join(OUT, f'flint-q127-G3-{name}.gguf')
    open(out + '.partial', 'w').close()
    subprocess.run([sys.executable, COMPOSE, G3, out, f'{rx}={donor}'], check=True)
    dt = {t.name: t for t in gguf.GGUFReader(donor).tensors}
    so = {t.name: t for t in gguf.GGUFReader(out).tensors}
    pat = re.compile(rx)
    n_src = n_don = n_type = 0
    for n, t in g3.items():
        want = dt[n] if pat.search(n) else t
        o = so[n]
        if o.tensor_type != want.tensor_type or not np.array_equal(np.asarray(o.data).reshape(-1), np.asarray(want.data).reshape(-1)):
            sys.exit(f'{out}: {n} is neither G3\'s nor the donor\'s')
        if pat.search(n):
            n_don += 1
            n_type += want.tensor_type != t.tensor_type
        else:
            n_src += 1
    if len(so) != len(g3):
        sys.exit(f'{out}: {len(so)} tensors, G3 has {len(g3)}')
    print(f'{out}: {n_src} tensors from G3, {n_don} from {os.path.basename(donor)} ({n_type} of another type), checked; '
          f'{os.path.getsize(out) / 1e9:.2f} GB', flush=True)
    os.remove(out + '.partial')
    open(out + '.ready', 'w').close()
print('Q127_FILES_DONE')
