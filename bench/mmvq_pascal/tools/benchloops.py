#!/usr/bin/env python3
# Main-loop cost of bench/mmvq_pascal's staged kernels (mb::pascal::mmvq_staged) for Q6_K, Q5_K, Q8_0 and IQ4_XS from
# the bench's sm_61 SASS: warp instructions in the largest backward-branch loop of each kernel, lane instructions per
# weight byte, and the counts of the instructions that matter (IDP, LDS, PRMT, LDG, STS, LOP, shifts). Written for the
# Q6_K kernel (Q101, 2026-10-24); benchloops_k.py in ../verify/ is the K-quant version.
# Usage: benchloops.py bench/mmvq_pascal/build/sm61.sass   (written by bench/mmvq_pascal/build.sh)
import re,sys,collections
txt=open(sys.argv[1]).read()
bs={'2':18,'3':20,'4':22,'5':176,'6':136,'7':210,'0':18,'1':20}
names={'0':'q4_0','1':'q4_1','2':'q8_0','3':'q4_1_64','4':'q5_0','5':'q5_K','6':'iq4_xs','7':'q6_K'}
bsz={'q4_0':18,'q4_1':20,'q8_0':34,'q4_1_64':36,'q5_0':22,'q5_K':176,'iq4_xs':136,'q6_K':210}
for f in re.split(r'\n\s+Function : ',txt)[1:]:
    name,body=f.split('\n',1); name=name.strip()
    m=re.match(r'_ZN2mb6pascal11mmvq_stagedILi(\d)ELi(\d)ELi(\d)ELi4ELb([01])ELi(\d+)EEEvNS0_4argsE',name)
    if not m: continue
    t,R,C,G,SU=m.groups(); t=names[t]; R=int(R); C=int(C); G=int(G)
    if t not in ('q6_K','q5_K','q8_0','iq4_xs'): continue
    ins=[]
    for l in body.split('\n'):
        mm=re.match(r'\s+/\*([0-9a-f]{4,})\*/\s+(.*?);',l)
        if mm: ins.append((int(mm.group(1),16),mm.group(2)))
    best=None
    for a,op in ins:
        mb=re.search(r'BRA\b.*?0x([0-9a-f]+)\s*$',op)
        if mb and int(mb.group(1),16)<a:
            span=[x for x in ins if int(mb.group(1),16)<=x[0]<=a]
            if best is None or len(span)>len(best): best=span
    if not best: continue
    span=[op for _,op in best if not op.startswith('NOP')]
    c=collections.Counter((op.split()[1] if op.startswith('@') else op.split()[0]).split('.')[0] for op in span)
    M=2 if G else 1
    wb=32*R*M*C*bsz[t]
    print(f"{t:6s} R{R}C{C} gate{G} SU{SU}: loop {len(span)} warp-instr, {32*len(span)/wb:.2f} lane-instr/B; IDP {c['IDP']} LDS {c['LDS']} PRMT {c['PRMT']} LDG {c['LDG']} STS {c['STS']} LOP {c['LOP']+c['LOP3']} SHF {c['SHF']+c['SHL']+c['SHR']}")
