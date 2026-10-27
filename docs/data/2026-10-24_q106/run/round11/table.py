# Round 11 (2026-10-24), the byte model behind docs/research/notes/long_context_mix.md (its tensor-class and ranked
# tables): per-tensor types of UD-Q4_K_XL (from its header), G3, K2 and K1; MB per token and est. ms per format.
# Run from this folder with PYTHONPATH=../../../2026-10-24_research_scratch/estimates (for ggufhdr.py);
# env UD_GGUF = path of Unsloth's flint-UD-Q4_K_XL.gguf (only its header is read).
import sys; sys.path.insert(0,'.')
from mix import *
U = ud_map(); G = ours('G3'); K2 = ours('K2'); K1 = ours('K1')
def mb(m, c, layers):
    return sum(NP[c]*BPW[m[(c,l)]] for l in layers if (c,l) in m)/1e6
def types(m, c, layers):
    cnt = collections.Counter(m[(c,l)] for l in layers if (c,l) in m)
    return ', '.join(f"{t}×{n}" for t,n in cnt.most_common())
GL = sorted(GLOBAL); SL = [l for l in range(49) if l not in GLOBAL]
rows = [('q','global',GL),('q','sliding',SL),('k','all',range(49)),('v','all',range(49)),('o','global',GL),('o','sliding',SL),
        ('shexp','2-24',range(2,25)),('shexp','25-48',range(25,49)),('dense','0-1',range(2)),
        ('gu','2-18',range(2,19)),('gu','19-48',range(19,49)),('down','2-18',range(2,19)),('down','19-48',range(19,49)),('router','all',range(49))]
for c,band,ls in rows:
    ls=list(ls)
    print(f"{c:6s} {band:8s} | UD {types(U,c,ls):40s} {mb(U,c,ls):6.1f} | G3 {types(G,c,ls):22s} {mb(G,c,ls):6.1f} | K2 {types(K2,c,ls):22s} {mb(K2,c,ls):6.1f} | K1 {mb(K1,c,ls):6.1f}")
# where UD has more bits than G3 (per tensor), MB of difference
more_ud = collections.defaultdict(float); more_g3 = collections.defaultdict(float)
for k,t in U.items():
    c,l = k
    if c in ('head',): continue
    if k not in G: continue
    d = NP[c]*(BPW[t]-BPW[G[k]])/1e6
    if d>0: more_ud[c]+=d
    elif d<0: more_g3[c]+=-d
print('UD more bytes than G3 by class (MB):', {c:round(v,1) for c,v in more_ud.items()})
print('G3 more bytes than UD by class (MB):', {c:round(v,1) for c,v in more_g3.items()})
# same vs K2
more_ud = collections.defaultdict(float); more_k = collections.defaultdict(float)
for k,t in U.items():
    c,l = k
    if c in ('head',) or k not in K2: continue
    d = NP[c]*(BPW[t]-BPW[K2[k]])/1e6
    if d>0: more_ud[c]+=d
    elif d<0: more_k[c]+=-d
print('UD more bytes than K2:', {c:round(v,1) for c,v in more_ud.items()})
print('K2 more bytes than UD:', {c:round(v,1) for c,v in more_k.items()})
# per-tensor: attention tensors where UD Q8_0 and G3 Q5_0
lst = [(c,l) for (c,l),t in U.items() if c in ('q','o','k','v') and t=='Q8_0' and G.get((c,l))=='Q5_0']
print('attention tensors UD Q8_0 / G3 Q5_0:', len(lst), 'sliding:', sorted([x for x in lst if x[1] not in GLOBAL]))
lst2 = [(c,l) for (c,l),t in U.items() if c in ('q','o','k','v') and t=='Q8_0' and K2.get((c,l))=='Q5_0']
print('attention tensors UD Q8_0 / K2 Q5_0:', len(lst2), 'MB', sum(NP[c]*0.375 for c,l in lst2)/1e6, 'sliding MB', sum(NP[c]*0.375 for c,l in lst2 if l not in GLOBAL)/1e6)
