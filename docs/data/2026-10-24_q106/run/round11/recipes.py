# Round 11 (2026-10-24), the byte model behind docs/research/notes/long_context_mix.md (its tensor-class and ranked
# tables): per-tensor types of UD-Q4_K_XL (from its header), G3, K2 and K1; MB per token and est. ms per format.
# Run from this folder with PYTHONPATH=../../../2026-10-24_research_scratch/estimates (for ggufhdr.py);
# env UD_GGUF = path of Unsloth's flint-UD-Q4_K_XL.gguf (only its header is read).
import sys; sys.path.insert(0,'.')
from mix import *
U = ud_map()
def setr(m, c, layers, t):
    for l in layers:
        if (c,l) in m: m[(c,l)] = t
    return m
def show(name, m, head='SCREEN', router='F16'):
    tot, bc, bf, ms = cost(m, head, router)
    print(f"{name:34s} {tot/1e9:.3f} GB   " + ' '.join(f"{f}={v/1e6:.0f}" for f,v in sorted(bf.items())))
    return tot
show('UD + screen', U, 'SCREEN', None)
show('K2sr = K2 + screen + F16 router', ours('K2'))
show('K1sr', ours('K1'))
show('G3sr', ours('G3'))
m = ours('G3'); setr(m,'shexp',range(25,49),'Q6_K'); setr(m,'k',range(33,49),'Q6_K'); setr(m,'v',range(33,49),'Q6_K'); setr(m,'q',range(0,9),'Q6_K')
show('G3-6sr (Q6_K: sh25-48, kv33-48, q0-8)', m)
# attention-first: G3 + Q8_0 where UD has Q8_0 in attention (29 tensors) 
GLs = sorted(GLOBAL)
def g3a():
    m = ours('G3')
    for (c,l),t in U.items():
        if c in ('q','o','k','v') and t=='Q8_0' and m.get((c,l))=='Q5_0': m[(c,l)]='Q8_0'
    return m
show('G3a (G3 + 29 attention tensors Q8_0)', g3a())
m = g3a(); setr(m,'shexp',range(2,49),'Q6_K'); setr(m,'dense',range(2),'Q6_K'); setr(m,'gu',range(2,9),'Q6_K'); setr(m,'down',range(46,49),'Q6_K')
show('G3a-6sr: + Q6_K shexp/dense/gu2-8/down46-48', m)
setr(m,'shexp',range(33,46),'Q5_0'); setr(m,'q',range(0,9),'Q5_0')
show('  ... + sh33-45 Q5_0, q0-8 Q5_0', m)
# K2 attention-first: K2 + 19 sliding attention tensors UD-Q8_0
m = ours('K2')
for (c,l),t in U.items():
    if c in ('q','o') and t=='Q8_0' and m.get((c,l))=='Q5_0' and l not in GLOBAL and l>=25: m[(c,l)]='Q8_0'
show('K2a (K2 + 19 sliding q/o Q8_0)', m)
# swap probes
def swap(base, donor, classes):
    m = dict(base)
    for k,t in donor.items():
        if k[0] in classes: m[k]=t
    return m
show('SW-A: G3 + UD attention (Q8_0 head)', swap(ours('G3'),U,{'q','k','v','o'}), 'Q8_0', 'F32')
show('SW-X: G3 + UD routed experts', swap(ours('G3'),U,{'gu','down'}), 'Q8_0', 'F32')
show('SW-U: UD + G3 attention', swap(U,ours('G3'),{'q','k','v','o'}), 'Q8_0', 'BF16')
show('G3 (Q8_0 head, F32 router)', ours('G3'), 'Q8_0', 'F32')
show('UD (Q8_0 head)', U, 'Q8_0', None)
