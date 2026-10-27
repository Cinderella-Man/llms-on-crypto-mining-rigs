#!/usr/bin/env python3
# Q127 follow-up (2026-10-24): magnitudes of every down-projection input that act_hessian dumped (HESS_DUMP rows, float32 from a CPU
# run): routed experts (ffn_moe_swiglu-L, 512 values per row) and shared expert / dense FFN (ffn_swiglu-L, 1024 or
# 6144). Per layer: the largest |x|; rows whose largest |x| passes 1024 (where the MoE rescale divides) and 65504
# (FP16 range: cuBLAS's FP16 path); 32-value blocks whose sum passes 65504 (block_q8_1's FP16 sum); and the largest
# |x| / 127 against FP16's range (block_q8_1's FP16 scale d).
#   mag.py DUMP_DIR
import glob, os, re, sys
import numpy as np

F16 = 65504.0
d = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(d, '*.rows.f32')):
    m = re.match(r'(ffn_moe_swiglu|ffn_swiglu)-(\d+)\.rows\.f32$', os.path.basename(p))
    if not m:
        continue
    kind, l = m.group(1), int(m.group(2))
    n = 512 if kind == 'ffn_moe_swiglu' else (6144 if l < 2 else 1024)
    x = np.fromfile(p, np.float32)
    x = x[:len(x) // n * n].reshape(-1, n)
    a = np.abs(x)
    amax = a.max(axis=1)
    bs = np.abs(x.reshape(len(x), n // 32, 32).astype(np.float64).sum(axis=2))
    per_tok = 6 if kind == 'ffn_moe_swiglu' else 1           # routed rows: 6 (token, expert) rows per token
    late = amax[4096 * per_tok:]                              # positions 4096 and later (row order = token order)
    rows.append((kind, l, len(x), float(a.max()), int((amax > 1024).sum()), int((amax > F16).sum()),
                 float(bs.max()), int((bs > F16).sum()), float(a.max() / 127), float(late.max()) if len(late) else float('nan')))
rows.sort(key=lambda r: (r[0], r[1]))
print('| Input | Layer | Rows | max abs | max abs, positions 4096+ | rows max > 1024 | rows max > 65504 | max block sum | '
      'blocks sum > 65504 | max/127 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4g} | {r[9]:.4g} | {r[4]} | {r[5]} | {r[6]:.4g} | {r[7]} | {r[8]:.4g} |')
for kind in ('ffn_moe_swiglu', 'ffn_swiglu'):
    sel = [r for r in rows if r[0] == kind]
    if sel:
        hot = [r[1] for r in sel if r[7] > 0]
        print(f'\n{kind}: {len(sel)} layers; largest |x| {max(r[3] for r in sel):.4g} (layer '
              f'{max(sel, key=lambda r: r[3])[1]}); layers with a 32-value block sum past 65504: {hot or "none"}; '
              f'with a value past 65504: {[r[1] for r in sel if r[5] > 0] or "none"}')
