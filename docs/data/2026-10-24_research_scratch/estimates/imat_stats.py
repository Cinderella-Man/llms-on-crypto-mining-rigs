# 2026-10-24 estimate: from an importance-matrix GGUF (in_sum2 and counts per tensor): per-expert token counts and the
# input structure per tensor class and layer band (max channel energy over mean, participation ratio, top-1 % share),
# the "Input structure" table of docs/research/notes/gptq_body.md. Env LLAMA_SRC: a llama.cpp tree (for gguf-py).
#   python3 imat_stats.py imatrix_unsloth.gguf
import os, sys, re, collections
sys.path.insert(0, os.path.join(os.environ.get('LLAMA_SRC', '.'), 'gguf-py'))   # a llama.cpp tree's gguf-py
import numpy as np
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = {k: r.fields[k] for k in r.fields}
def fv(k):
    fld = f[k]; return fld.parts[fld.data[0]].tolist() if fld.types else None
print('chunk_count', fv('imatrix.chunk_count'), 'chunk_size', fv('imatrix.chunk_size'))
T = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
print('shapes', [(n, T[n].shape) for n in list(T)[:0]])
for n in ['blk.10.ffn_gate_exps.weight.in_sum2','blk.10.ffn_gate_exps.weight.counts','blk.10.ffn_down_exps.weight.in_sum2','blk.10.ffn_down_exps.weight.counts','output.weight.in_sum2','output.weight.counts']:
    if n in T: print(n, T[n].shape, T[n].ravel()[:4])

def metr(e):
    e = np.maximum(e, 0); n = e.size; m = e.mean()
    pr = e.sum()**2 / (n * (e**2).sum())
    top = np.sort(e)[::-1]; k = max(1, n // 100)
    return e.max()/m, pr, top[:k].sum()/e.sum()
print('\n== per-expert token counts (gate_exps), 351,744 tokens x 6 / 256 = 8,244 mean')
allc = []
for L in range(49):
    n = f'blk.{L}.ffn_gate_exps.weight.counts'
    if n not in T: continue
    c = T[n].ravel(); allc.append(c)
    if L in (2,3,5,8,12,16,20,24,28,32,36,40,44,46,47,48):
        q = np.percentile(c, [0,1,5,50,95,100])
        print(f'L{L:2d} min {q[0]:7.0f} p1 {q[1]:7.0f} p5 {q[2]:7.0f} med {q[3]:7.0f} p95 {q[4]:7.0f} max {q[5]:7.0f}  <2048: {(c<2048).sum():3d}  <512: {(c<512).sum():3d}  top25%share {np.sort(c)[::-1][:64].sum()/c.sum():.2f}')
A = np.concatenate(allc)
print('all layers: experts', A.size, 'min', A.min(), 'p1', np.percentile(A,1), 'p10', np.percentile(A,10), '<2048', (A<2048).sum(), '<4096', (A<4096).sum(), '<512', (A<512).sum(), 'zero', (A==0).sum(), 'min/mean', A.min()/A.mean())
print('\n== diag(H) anisotropy: max/mean, participation ratio, top-1% energy share')
cls = collections.defaultdict(list)
for name in T:
    if not name.endswith('.in_sum2'): continue
    base = name[:-8]
    s = T[name]; c = T[base + '.counts']
    m = re.match(r'blk\.(\d+)\.(.*)\.weight', base)
    L, kind = (int(m.group(1)), m.group(2)) if m else (-1, base)
    if s.ndim == 2:  # per expert
        ok = c.ravel() > 0
        per = [metr(s[i]/c.ravel()[i]) for i in np.where(ok)[0]]
        agg = metr(s[ok].sum(0)/c.ravel()[ok].sum())
        cls[kind].append((L, np.median([p[0] for p in per]), np.median([p[1] for p in per]), np.median([p[2] for p in per]), agg))
    else:
        cls[kind].append((L,) + metr(s/c.ravel()[0]))
for kind, rows in sorted(cls.items()):
    rows.sort()
    def band(lo, hi):
        rr = [x for x in rows if lo <= x[0] <= hi]
        if not rr: return ''
        a = np.array([x[1:4] for x in rr], dtype=float)
        return f'{lo}-{hi}: max/mean {np.median(a[:,0]):6.1f} (max {a[:,0].max():7.1f}) PR {np.median(a[:,1]):.2f} top1% {np.median(a[:,2]):.2f}'
    print(kind)
    for lo, hi in [(-1,-1),(0,1),(2,8),(9,16),(17,24),(25,32),(33,40),(41,45),(46,48)]:
        b = band(lo, hi)
        if b: print('   ', b)
