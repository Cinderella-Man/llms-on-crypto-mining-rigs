#!/usr/bin/env python3
# Compare the SASS of the fork's mmvq-pascal.cu between two single-file sm_61 builds (cuobjdump -sass), ignoring the
# staging-unit template argument (SU = 16) that the Q6_K branch adds, and list registers and spills of the Q6_K kernels
# (Q101, 2026-10-24): every kernel of the old file must be unchanged by the new type.
# Build each version of ggml/src/ggml-cuda/mmvq-pascal.cu alone (the old one copied as mmvq-pascal_before.cu), e.g.
#   nvcc -O3 -std=c++17 -arch=sm_61 -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/src/ggml-cuda \
#        -Xptxas -v -c mmvq-pascal.cu -o new.o 2> ptxas.txt && cuobjdump -sass new.o > new.sass
# Usage: forksass.py old.sass new.sass [ptxas.txt of the new build]   (needs c++filt for the register list)
import re,sys,subprocess
def funcs(p):
    t=open(p).read(); out={}
    for f in re.split(r'\n\s+Function : ',t)[1:]:
        name,body=f.split('\n',1); name=name.strip()
        name=re.sub(r'_ZN\d+_GLOBAL__N__[0-9a-f_]+mmvq_pascal(?:_before)?_cu_[0-9a-f]+','',name)
        name=re.sub(r'ELi16EEEvNS_','EEEvNS_',name)   # trailing SU = 16
        ins=[re.sub(r'/\*[0-9a-f]{4,}\*/','',l).split(';')[0].strip() for l in body.split('\n') if re.match(r'\s+/\*[0-9a-f]{4,}\*/',l)]
        out[name]=ins
    return out
a=funcs(sys.argv[1]); b=funcs(sys.argv[2])
same=sum(1 for k in a if k in b and a[k]==b[k]); diff=[k for k in a if k in b and a[k]!=b[k]]; miss=[k for k in a if k not in b]
print(len(a),'kernels before;',same,'identical;',len(diff),'differ;',len(miss),'not found;',len(b)-len(a)+len(miss),'new')
for k in diff[:8]: print('  differs',k)
for k in miss[:8]: print('  not found',k)
if len(sys.argv)>3:
    cur=None
    for l in open(sys.argv[3]):
        m=re.search(r"Compiling entry function '(\S+)'",l)
        if m: cur=m.group(1); continue
        m=re.search(r"(\d+) bytes spill stores",l)
        if m and cur and 'GGML_TYPE14' in subprocess.run(['c++filt',cur],capture_output=True,text=True).stdout.replace('(ggml_type)14','GGML_TYPE14'): sp=m.group(1)
        m=re.search(r"Used (\d+) registers",l)
        if m and cur:
            d=subprocess.run(['c++filt',cur],capture_output=True,text=True).stdout.strip()
            if '(ggml_type)14' in d:
                d=re.sub(r'\(anonymous namespace\)::','',d); d=re.sub(r'\(.*','',d)
                print(f"  {m.group(1):>4} regs  {d}")
            cur=None
