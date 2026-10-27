#!/usr/bin/env python3
# Registers and spills of the fork's Pascal matrix-vector kernels, one line per weight type and fusion kind (fk: 0 plain,
# 1 fused bias add, 2 fused gate/up + GLU) at the launch shape the fork picks (mmvqp_choose), single-token kernel against
# the 2-4-token one (_mc). Written for the multi-token kernels (Q78, 2026-10-23, fork branch pascal-p106-v2-multicol).
# Input: the ptxas log of ggml/src/ggml-cuda/mmvq-pascal.cu compiled alone for sm_61 with `nvcc -arch=sm_61 -Xptxas -v`
# (plus the include and define flags of the fork's CUDA build). Needs c++filt (binutils).
# Usage: regs.py ptxas.log
import re,subprocess,sys
lines=open(sys.argv[1]).read().splitlines()
cur=None; rows=[]
for l in lines:
    m=re.search(r"Compiling entry function '(\S+)'",l)
    if m: cur=m.group(1); spill=''; continue
    m=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",l)
    if m and cur: spill=m.group(2)
    m=re.search(r"Used (\d+) registers",l)
    if m and cur:
        rows.append((cur,int(m.group(1)),spill)); cur=None
dem=subprocess.run(['c++filt'],input='\n'.join(r[0] for r in rows),capture_output=True,text=True).stdout.splitlines()
tn={'2':'q4_0','3':'q4_1','6':'q5_0','8':'q8_0','13':'q5_K','23':'iq4_xs'}
# shapes chosen by mmvqp_choose: (type, gate) -> (R,C,staged)
choose={('q4_1',True):(2,2,1),('q4_1',False):(2,4,1),('q4_0',True):(2,2,1),('q4_0',False):(2,2,1),('q5_0',True):(1,2,1),('q5_0',False):(2,2,1),
('q8_0',True):(1,2,1),('q8_0',False):(2,2,1),('q5_K',True):(1,1,1),('q5_K',False):(1,1,1),('iq4_xs',True):(1,1,0),('iq4_xs',False):(1,1,1)}
res={}
for (n,r,sp),d in zip(rows,dem):
    m=re.search(r'(mul_mat_vec_q_pascal(?:_mc)?)<\(ggml_type\)(\d+), (\d), (\d), (true|false), (\d)>',d)
    k,t,R,C,st,fk=m.groups(); res[(k,tn[t],int(R),int(C),1 if st=='true' else 0,int(fk))]=(r,sp)
print("type    fk  shape   1-col(regs/spill)  mc(regs/spill)")
for (t,g),(R,C,st) in choose.items():
    for fk in ([2] if g else [0,1]):
        a=res.get(('mul_mat_vec_q_pascal',t,R,C,st,fk)); b=res.get(('mul_mat_vec_q_pascal_mc',t,R,C,st,fk))
        print(f"{t:7s} {fk}  R{R}C{C}{'s' if st else 'd'}  {a}  {b}")
