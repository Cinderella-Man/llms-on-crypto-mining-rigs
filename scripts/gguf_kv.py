#!/usr/bin/env python3
# Prints a GGUF file's KV pairs (arrays shortened to their first 8 values, tokenizer arrays omitted) without numpy.
#
#   scripts/gguf_kv.py model.gguf
import struct, sys
SIZES = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rd(f, fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))[0]
def rstr(f): n = rd(f,'Q'); return f.read(n).decode('utf-8','replace')
def val(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at = rd(f,'I'); n = rd(f,'Q')
        if at == 8: v = [rstr(f) for _ in range(n)]
        else: v = [rd(f, SIZES[at]) for _ in range(n)]
        return ('array', at, n, v[:8])
    return rd(f, SIZES[t])
with open(sys.argv[1],'rb') as f:
    f.read(4); rd(f,'I'); nt = rd(f,'Q'); nkv = rd(f,'Q')
    for _ in range(nkv):
        k = rstr(f); v = val(f, rd(f,'I'))
        if k.startswith('tokenizer.ggml.') and isinstance(v, tuple): v = ('array', v[1], v[2], '...')
        print(k, v)
