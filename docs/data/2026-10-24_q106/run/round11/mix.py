# Round 11 (2026-10-24), the byte model behind docs/research/notes/long_context_mix.md (its tensor-class and ranked
# tables): per-tensor types of UD-Q4_K_XL (from its header), G3, K2 and K1; MB per token and est. ms per format.
# Run from this folder with PYTHONPATH=../../../2026-10-24_research_scratch/estimates (for ggufhdr.py);
# env UD_GGUF = path of Unsloth's flint-UD-Q4_K_XL.gguf (only its header is read).
import os, sys, re, collections
sys.path.insert(0,'.')
from ggufhdr import header, T
BPW = {'Q8_0':34/32,'Q5_0':22/32,'Q4_1':20/32,'Q4_0':18/32,'Q5_K':176/256,'Q4_K':144/256,'Q6_K':210/256,'IQ4_XS':136/256,
       'IQ3_S':110/256,'BF16':2,'F16':2,'F32':4,'SCREEN':None}
# GB/s on the fork (P106-090, large shapes); est. where unmeasured
RATE = {'Q8_0':150,'Q5_0':138,'Q4_1':145,'Q4_0':130,'Q5_K':125,'IQ4_XS':100,'Q4_K':130,'Q6_K':140,'IQ3_S':90,'BF16':82,'F16':82,'F32':110}
GLOBAL = {0,1,5,9,13,17,21,25,29,33,37,41,45,48}
NP = {'q':2048*4096,'k':2048*512,'v':2048*512,'o':4096*2048,'shexp':3*2048*1024,'gu':6*2*2048*512,'down':6*512*2048,
      'router':2048*256,'dense':3*2048*6144,'head':2048*150144}
def ud_map():
    kv, ts = header(os.environ['UD_GGUF'])   # Unsloth's flint-UD-Q4_K_XL.gguf
    m = {}
    cmap = {'attn_q':'q','attn_k':'k','attn_v':'v','attn_output':'o','ffn_gate_exps':'gu','ffn_down_exps':'down','ffn_gate_inp':'router',
            'ffn_gate_shexp':'shexp','ffn_gate':'dense'}
    for name, dims, t in ts:
        if name == 'output.weight': m[('head',-1)] = T[t][0]
        mm = re.match(r'blk\.(\d+)\.(\w+)\.weight', name)
        if mm and mm.group(2) in cmap: m[(cmap[mm.group(2)], int(mm.group(1)))] = T[t][0]
    return m
def rng(a,b): return set(range(a,b+1))
def ours(name):
    m = {}
    for l in range(49):
        m[('q',l)] = 'Q8_0' if l<=24 else 'Q5_0'
        m[('k',l)] = m[('v',l)] = 'Q8_0'
        m[('o',l)] = 'Q8_0' if l<=32 else 'Q5_0'
        if l<=1: m[('dense',l)] = 'Q8_0'; continue
        m[('shexp',l)] = 'Q8_0'
        m[('gu',l)] = 'Q8_0' if l<=8 else 'Q5_0'
        m[('down',l)] = 'Q8_0' if l>=46 else 'Q5_0'
        m[('router',l)] = 'F32'
    m[('head',-1)] = 'Q8_0'
    if name in ('K2','K1'):
        for l in rng(33,45): m[('shexp',l)] = 'Q5_0'
        for l in rng(33,48): m[('k',l)] = m[('v',l)] = 'Q5_0'
        for l in rng(0,8): m[('q',l)] = 'Q5_0'
    if name == 'K1':
        for l in rng(25,45): m[('shexp',l)] = 'Q5_0'
        for l in rng(25,32): m[('o',l)] = 'Q5_0'
    return m
def cost(m, head=None, router=None):
    by_cls = collections.defaultdict(float); by_fmt = collections.defaultdict(float); ms = 0.0
    for (c,l),t in m.items():
        if c=='head' and head: t = head
        if c=='router' and router: t = router
        if t == 'SCREEN':
            b = 174e6; r = 150
            by_cls[c]+=b; by_fmt['screen']+=b; ms += b/(r*1e9)*1e3; continue
        b = NP[c]*BPW[t]
        by_cls[c]+=b; by_fmt[t]+=b
        r = RATE[t] if not (c=='head' and t=='Q8_0') else 165
        ms += b/(r*1e9)*1e3
    return sum(by_cls.values()), by_cls, by_fmt, ms
if __name__ == '__main__':
    U = ud_map()
    files = {'UD':(U,None,None),'G3':(ours('G3'),None,None),'K2h':(ours('K2'),'Q5_0',None),'K1h':(ours('K1'),'Q5_0',None),
             'UD+scr':(U,'SCREEN',None),'G3+scr':(ours('G3'),'SCREEN',None),'K2+scr':(ours('K2'),'SCREEN',None),'K1+scr':(ours('K1'),'SCREEN',None),
             'K1+scr+r16':(ours('K1'),'SCREEN','F16'),'K2+scr+r16':(ours('K2'),'SCREEN','F16'),'G3+scr+r16':(ours('G3'),'SCREEN','F16')}
    # S4-L0 for calibration
    s4 = {}
    for l in range(49):
        s4[('q',l)]='Q4_1'; s4[('k',l)]=s4[('v',l)]='Q8_0'; s4[('o',l)]='Q8_0' if l<=32 else 'Q4_1'
        if l<=1: s4[('dense',l)]='Q8_0'; continue
        s4[('shexp',l)]='Q8_0'; s4[('gu',l)]='Q5_0'; s4[('down',l)]='Q8_0'; s4[('router',l)]='F16'
    s4[('head',-1)]='Q8_0'
    tot, bc, bf, ms = cost(s4); t0 = 26.8 - ms
    print(f'S4-L0 {tot/1e9:.3f} GB, matrix {ms:.2f} ms -> t0 = {t0:.2f} ms')
    for n,(m,h,r) in files.items():
        tot, bc, bf, ms = cost(m,h,r)
        print(f"{n:12s} {tot/1e9:.3f} GB  est. {t0+ms:.2f} ms  " + ' '.join(f"{c}={v/1e6:.0f}" for c,v in sorted(bc.items())) + ' | ' + ' '.join(f"{f}={v/1e6:.0f}" for f,v in sorted(bf.items())))
