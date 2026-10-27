#!/usr/bin/env python3
# Sorts the failures in a test-backend-ops log into 2-4-token cases (n = 2-4, or m = 2-4 for MUL_MAT_VEC_FUSION) of the
# six types with Pascal kernels (Q8_0, Q4_0, Q4_1, Q5_0, Q5_K, IQ4_XS) and everything else, and lists passing cases in
# scope. Used to check that a deliberately broken kernel (a mutation) fails exactly the cases it should (Q78,
# 2026-10-23; docs/research/notes/pascal_mmvq_integration.md, section 14).
# Usage: classify.py test-backend-ops.log
import re,sys,collections
T={'q8_0','q4_0','q4_1','q5_0','q5_K','iq4_xs'}
fail=collections.Counter(); okmc=collections.Counter(); failother=[]; ok_examples=[]
for l in open(sys.argv[1]):
    m=re.search(r'\s([A-Z_]+)\((.*)\): (.*)$',l)
    if not m: continue
    op,v,res=m.groups()
    if 'not supported' in res: continue
    d=dict(kv.split('=',1) for kv in re.split(r',(?=[a-z_]+=)',v) if '=' in kv)
    t=d.get('type_a',d.get('type'))
    n=d.get('m') if op=='MUL_MAT_VEC_FUSION' else d.get('n')
    isfail='FAIL' in res
    inscope = t in T and n is not None and n.isdigit() and 2<=int(n)<=4
    if isfail:
        if inscope: fail[op]+=1
        else: failother.append(op+'('+v[:140])
    elif inscope:
        okmc[op]+=1; ok_examples.append(op+'('+v[:160])
print('failed, 2-4 tokens of the six types:',dict(fail),sum(fail.values()))
print('failed, other:',len(failother)); print('\n'.join(failother[:10]))
print('passed, 2-4 tokens of the six types:',dict(okmc))
print('\n'.join(ok_examples[:30]))
