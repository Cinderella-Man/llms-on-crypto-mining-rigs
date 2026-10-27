#!/usr/bin/env python3
# regs.py and loops.py in one table, for the launch shapes the fork picks only: per weight type, fusion kind and kernel
# (1tok = single token, mc = 2-4 tokens, mc-sa = 2-4 tokens with staged activations) the registers, spill bytes,
# main-loop instructions, instructions per weight byte and the LDG/LDS/STS counts. Written for Q78 (2026-10-23, fork
# branch pascal-p106-v2-multicol). Inputs as for regs.py and loops.py. Needs c++filt (binutils).
# Usage: loops2.py ptxas.log mmvq-pascal.sass
import re,subprocess,collections,sys
tn={'2':'q4_0','3':'q4_1','6':'q5_0','8':'q8_0','13':'q5_K','23':'iq4_xs'}
bs={'q4_0':18,'q4_1':20,'q5_0':22,'q8_0':34,'q5_K':176,'iq4_xs':136}
choose={('q4_1',2):(2,2,1),('q4_1',0):(2,4,1),('q4_0',2):(2,2,1),('q4_0',0):(2,2,1),('q5_0',2):(1,2,1),('q5_0',0):(2,2,1),
('q8_0',2):(1,2,1),('q8_0',0):(2,2,1),('q5_K',2):(1,1,1),('q5_K',0):(1,1,1),('iq4_xs',2):(1,1,0),('iq4_xs',0):(1,1,1)}
log,sass=sys.argv[1],sys.argv[2]
regs={};spill={}
cur=None
for l in open(log):
    m=re.search(r"Compiling entry function '(\S+)'",l)
    if m: cur=m.group(1); continue
    m=re.search(r"(\d+) bytes spill stores",l)
    if m and cur: spill[cur]=int(m.group(1))
    m=re.search(r"Used (\d+) registers",l)
    if m and cur: regs[cur]=int(m.group(1)); cur=None
txt=open(sass).read()
rows=[]
for f in re.split(r'\n\s+Function : ',txt)[1:]:
    name,body=f.split('\n',1); name=name.strip()
    d=subprocess.run(['c++filt',name],capture_output=True,text=True).stdout.strip()
    m=re.search(r'mul_mat_vec_q_pascal(_mc)?<\(ggml_type\)(\d+), (\d), (\d), (true|false), (\d)(?:, (true|false))?>',d)
    if not m: continue
    mc,t,R,C,st,fk,sa=m.groups(); t=tn[t]; R=int(R); C=int(C); fk=int(fk)
    if choose.get((t,fk))!=(R,C,1 if st=='true' else 0): continue
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
    span=[op for _,op in best if not op.startswith('NOP')]
    c=collections.Counter((op.split()[1] if op.startswith('@') else op.split()[0]).split('.')[0] for op in span)
    M=2 if fk==2 else 1
    kind='1tok' if not mc else ('mc-sa' if sa=='true' else 'mc')
    rows.append((t,fk,kind,f"{t:6s} fk{fk} R{R}C{C} {kind:5s} regs {regs.get(name,'?'):>3} spill {spill.get(name,0):3d}  loop {len(span):4d} instr {len(span)/(32*R*M*C*bs[t]):.3f}/B  LDG {c['LDG']:3d} LDS {c['LDS']:3d} STS {c['STS']:2d}"))
for r in sorted(rows): print(r[3])
