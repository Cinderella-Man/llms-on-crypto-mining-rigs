# 2026-10-24 estimate: entropy of quantized codes (Q8_0, a symmetric 6-bit, Q5_0, Q4_0) for blocks of 32 Gaussian,
# Student-t (4) and Laplace weights: how many bits entropy coding could remove
# (docs/research/notes/unconventional_ideas.md, "Entropy-coded weights"). numpy; ent2.py is the same in pure Python.
import numpy as np
rng=np.random.default_rng(0)
def ent(codes):
    v,c=np.unique(codes,return_counts=True); p=c/c.sum(); return -(p*np.log2(p)).sum()
for name,gen in [("gauss",lambda n: rng.standard_normal(n)),("t4",lambda n: rng.standard_t(4,n)),("laplace",lambda n: rng.laplace(size=n))]:
    w=gen(32*200000).reshape(-1,32)
    amax=np.abs(w).max(1,keepdims=True)
    q8=np.round(w/amax*127).astype(int)
    # q4_0: d = max/-8 where max is signed value with largest abs
    idx=np.abs(w).argmax(1); mx=w[np.arange(len(w)),idx][:,None]
    d=mx/-8; q4=np.clip(np.round(w/d)+8,0,15).astype(int)
    d5=mx/-16; q5=np.clip(np.round(w/d5)+16,0,31).astype(int)
    # q6 symmetric 6-bit
    d6=mx/-32; q6=np.clip(np.round(w/d6)+32,0,63).astype(int)
    print(name,"Q8 %.2f/8  Q6 %.2f/6  Q5 %.2f/5  Q4 %.2f/4"%(ent(q8),ent(q6),ent(q5),ent(q4)))
