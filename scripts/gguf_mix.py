#!/usr/bin/env python3
# Prints which quantization format each kind of tensor uses in a GGUF file and its share of the bytes, reading only
# the header (pure Python, no numpy). Usage: scripts/gguf_mix.py <file.gguf>
import struct, sys, re, collections
T = {0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),3:('Q4_1',32,20),6:('Q5_0',32,22),7:('Q5_1',32,24),8:('Q8_0',32,34),
     10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),
     16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),
     21:('IQ3_S',256,110),22:('IQ2_S',256,82),23:('IQ4_XS',256,136),29:('IQ1_M',256,56),30:('BF16',1,2),39:('MXFP4',32,17)}
f = open(sys.argv[1], 'rb')
rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
def s(): return f.read(rd('Q')).decode('utf-8', 'replace')
SZ = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(t):
    if t == 8: return s()
    if t == 9:
        et, n = rd('I'), rd('Q')
        return [val(et) for _ in range(n)] if et in (8, 9) else f.read(struct.calcsize('<' + SZ[et]) * n)
    return rd(SZ[t])
assert f.read(4) == b'GGUF'; ver = rd('I'); nt, nkv = rd('Q'), rd('Q')
for _ in range(nkv): s(); val(rd('I'))
mix = collections.defaultdict(collections.Counter); by = collections.defaultdict(int); tot = 0
for _ in range(nt):
    name = s(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); rd('Q')
    n = 1
    for d in dims: n *= d
    tn, bs, bb = T.get(t, (f'type{t}', 1, 1)); b = n // bs * bb
    k = re.sub(r'blk\.\d+\.', '', name); mix[k][tn] += 1; by[k] += b; tot += b
for k in sorted(by, key=lambda k: -by[k]):
    if by[k] / tot >= 0.002: print(f"{k:24s} {100*by[k]/tot:5.1f}%  " + ", ".join(f"{a}x{c}" for a, c in mix[k].most_common()))
print(f"total tensor data {tot/1e9:.2f} GB")
