# 2026-10-24: a pure-Python GGUF header reader (metadata, tensor names, shapes and types; no tensor data). Imported by
# udmap.py and by docs/data/2026-10-24_q106/run/round11/mix.py.   python3 ggufhdr.py <file.gguf> prints the header.
import struct, re, sys, collections
T = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
     8: ('Q8_0', 32, 34), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
     14: ('Q6_K', 256, 210), 16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50),
     20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 29: ('IQ1_M', 256, 56),
     30: ('BF16', 1, 2)}
def header(path):
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    s = lambda: f.read(rd('Q')).decode('utf-8', 'replace')
    SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et, n = rd('I'), rd('Q')
            if et in (8,9): return [val(et) for _ in range(n)]
            f.seek(struct.calcsize('<'+SZ[et])*n, 1); return f'<array {n}>'
        return rd(SZ[t])
    assert f.read(4) == b'GGUF'
    rd('I'); nt, nkv = rd('Q'), rd('Q')
    kv = {}
    for _ in range(nkv):
        k = s(); kv[k] = val(rd('I'))
    ts = []
    for _ in range(nt):
        name = s(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); rd('Q')
        ts.append((name, dims, t))
    return kv, ts
if __name__ == '__main__':
    kv, ts = header(sys.argv[1])
    for k,v in kv.items():
        if 'token' in k and isinstance(v,str) and len(v)>200: continue
        if isinstance(v, list) and len(v) > 20: v = f'<list {len(v)}>'
        print(k, '=', v if not isinstance(v,str) or len(v)<200 else v[:200])
    for name, dims, t in ts:
        if name.startswith('blk.5.') or not name.startswith('blk.'):
            print(name, dims, T.get(t,(t,))[0])
