#!/usr/bin/env python3
# Compare the SASS of two bench/mmvq_pascal builds (the committed one and one with a new kernel type): every kernel of
# the committed build must be identical in the new one, after dropping the staging-unit template argument (Li16E) that
# the new mmvq_staged carries. Prints identical / differing / missing / new counts (Q101, 2026-10-24).
# Usage: sasscmp.py committed/sm61.sass new/sm61.sass   (build/sm61.sass of each bench/mmvq_pascal/build.sh build)
import re,sys
def funcs(p):
    t=open(p).read(); out={}
    for f in re.split(r'\n\s+Function : ',t)[1:]:
        name,body=f.split('\n',1); name=name.strip()
        # normalize: drop the staging-unit template argument (Li16E) that the new mmvq_staged carries
        n=re.sub(r'(mmvq_stagedILi\d+ELi\d+ELi\d+ELi\d+ELb[01]E)Li16E',r'\1',name)
        ins=[re.sub(r'/\*[0-9a-f]{4,}\*/','',l).split(';')[0].strip() for l in body.split('\n') if re.match(r'\s+/\*[0-9a-f]{4,}\*/',l)]
        out[n]=ins
    return out
a=funcs(sys.argv[1]); b=funcs(sys.argv[2])
same=sum(1 for k in a if k in b and a[k]==b[k]); diff=[k for k in a if k in b and a[k]!=b[k]]; miss=[k for k in a if k not in b]
print(len(a),'kernels in the committed build;',same,'identical;',len(diff),'differ;',len(miss),'missing;',len(b)-len(a),'new in the new build')
for k in diff[:10]: print('differs',k)
for k in miss[:10]: print('missing',k)
