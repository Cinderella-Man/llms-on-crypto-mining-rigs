#!/usr/bin/env python3
# Main-loop cost of the fork's Pascal matrix-vector kernels from their sm_61 SASS: for each kernel, the warp
# instructions in its largest backward-branch loop, lane instructions per weight byte, and the counts of IDP, LDS, LDG,
# STS, PRMT, FFMA, F2F, I2F and BRA. Written for the multi-token kernels (Q78, 2026-10-23, fork branch
# pascal-p106-v2-multicol). Input: `cuobjdump -sass` of mmvq-pascal.cu compiled alone for sm_61 (see regs.py).
# Optional filters select kernels as kind:type:shape:fk, e.g. "mc:q8_0:R2C2s:0" (kind 1 = single token, mc = 2-4 tokens;
# shape R<rows>C<lane groups> plus s = staged, d = direct loads). Needs c++filt (binutils).
# Usage: loops.py mmvq-pascal.sass [kind:type:shape:fk ...]
import re,subprocess,sys,collections
txt=open(sys.argv[1]).read()
funcs=re.split(r'\n\s+Function : ',txt)[1:]
tn={'2':'q4_0','3':'q4_1','6':'q5_0','8':'q8_0','13':'q5_K','23':'iq4_xs'}
bs={'q4_0':18,'q4_1':20,'q5_0':22,'q8_0':34,'q5_K':176,'iq4_xs':136}
want=set(tuple(x.split(':')) for x in sys.argv[2:])
for f in funcs:
    name=f.split('\n',1)[0].strip()
    d=subprocess.run(['c++filt',name],capture_output=True,text=True).stdout.strip()
    m=re.search(r'(mul_mat_vec_q_pascal(?:_mc)?)<\(ggml_type\)(\d+), (\d), (\d), (true|false), (\d)>',d)
    if not m: continue
    k,t,R,C,st,fk=m.groups(); t=tn[t]; R=int(R); C=int(C); fk=int(fk)
    key=('mc' if k.endswith('_mc') else '1', t, f'R{R}C{C}{"s" if st=="true" else "d"}', str(fk))
    if want and key not in want: continue
    ins=[]
    for l in f.split('\n'):
        mm=re.match(r'\s+/\*([0-9a-f]{4,})\*/\s+(.*?);',l)
        if mm:
            a=int(mm.group(1),16); op=mm.group(2)
            ins.append((a,op))
    best=None
    for a,op in ins:
        mb=re.search(r'BRA\b.*?0x([0-9a-f]+)\s*$',op)
        if mb:
            tgt=int(mb.group(1),16)
            if tgt<a:
                span=[x for x in ins if tgt<=x[0]<=a]
                if best is None or len(span)>len(best[2]): best=(tgt,a,span)
    if not best: print(key,'no loop'); continue
    span=[op for _,op in best[2] if not op.startswith('NOP')]
    c=collections.Counter()
    for op in span:
        o=op.split()[0]
        if o.startswith('@'): o=op.split()[1]
        c[o.split('.')[0]]+=1
    M=2 if fk==2 else 1
    wbytes=32*R*M*C*bs[t]
    L=len(span)
    print(f"{key[0]:2s} {t:6s} {key[2]} fk{fk}: loop {L} instr, {L/wbytes:.3f}/B; IDP {c['IDP']} LDS {c['LDS']} LDG {c['LDG']} STS {c['STS']} PRMT {c['PRMT']} FFMA {c['FFMA']} F2F {c['F2F']} I2F {c['I2F']} BRA {c['BRA']}")
