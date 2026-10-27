#!/usr/bin/env python3
# Writes a GGUF that keeps only layers 0..N-1 of a model (per-layer KV arrays of the flint architecture cut to N, the
# block count set to N), optionally taking the tensors whose names match REGEX from a second GGUF of the same model.
# Tensor data is copied as is, so the cut file loads in llama.cpp and runs the first N layers plus the output head:
# small enough for kernel and fusion checks on one GPU (registry row Q50 used a 5-layer cut of Flint). No numpy.
#
#   scripts/gguf_truncate.py <base.gguf> <out.gguf> <N> [<other.gguf> <regex>]
#   scripts/gguf_truncate.py flint-S4L0.gguf flint-5L.gguf 5
#
# For another architecture, change the "flint." prefix in the KV loop below to that architecture's name.
import os, re, struct, sys
base, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
other, rx = (sys.argv[4], re.compile(sys.argv[5])) if len(sys.argv) > 5 else (None, None)
FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rd(f, fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))[0]
def rstr(f): n = rd(f,'Q'); return f.read(n)
def rval(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        at = rd(f,'I'); n = rd(f,'Q')
        return (at, [rstr(f) if at == 8 else rd(f, FMT[at]) for _ in range(n)])
    return rd(f, FMT[t])
def wstr(o, b): o.write(struct.pack('<Q', len(b))); o.write(b)
def wval(o, t, v):
    if t == 8: wstr(o, v)
    elif t == 9:
        at, items = v; o.write(struct.pack('<IQ', at, len(items)))
        for x in items:
            wstr(o, x) if at == 8 else o.write(struct.pack('<'+FMT[at], x))
    else: o.write(struct.pack('<'+FMT[t], v))
def read(path):
    f = open(path, 'rb'); assert f.read(4) == b'GGUF'
    ver = rd(f,'I'); nt = rd(f,'Q'); nkv = rd(f,'Q')
    kvs = []
    for _ in range(nkv):
        k = rstr(f); t = rd(f,'I'); kvs.append((k, t, rval(f, t)))
    align = next((v for k, t, v in kvs if k == b'general.alignment'), 32)
    tis = []
    for _ in range(nt):
        name = rstr(f); nd = rd(f,'I'); dims = [rd(f,'Q') for _ in range(nd)]; t = rd(f,'I'); off = rd(f,'Q')
        tis.append([name, dims, t, off, 0])
    data = (f.tell() + align - 1) // align * align
    end = os.path.getsize(path) - data
    offs = sorted(ti[3] for ti in tis) + [end]
    nxt = {offs[i]: offs[i+1] for i in range(len(offs) - 1)}
    for ti in tis: ti[4] = nxt[ti[3]] - ti[3]
    return f, ver, kvs, align, tis, data
fb, ver, kvs, align, tis, data_b = read(base)
fo_ = read(other) if other else None
oth = {ti[0]: ti for ti in fo_[4]} if fo_ else {}
n_layer = next(v for k, t, v in kvs if k.endswith(b'.block_count'))
def keep(name):
    m = re.match(rb'blk\.(\d+)\.', name)
    return m is None or int(m.group(1)) < N
sel = []
for ti in tis:
    if not keep(ti[0]): continue
    if rx and rx.search(ti[0].decode()):
        sel.append((oth[ti[0]], fo_[0], fo_[5]))
    else:
        sel.append((ti, fb, data_b))
with open(out, 'wb') as o:
    o.write(b'GGUF'); o.write(struct.pack('<IQQ', ver, len(sel), len(kvs)))
    for k, t, v in kvs:
        if k.endswith(b'.block_count'): v = N
        elif t == 9 and len(v[1]) == n_layer and k.startswith(b'flint.'): v = (v[0], v[1][:N])
        wstr(o, k); o.write(struct.pack('<I', t)); wval(o, t, v)
    off = 0
    for ti, _, _ in sel:
        wstr(o, ti[0]); o.write(struct.pack('<I', len(ti[1])))
        for d in ti[1]: o.write(struct.pack('<Q', d))
        o.write(struct.pack('<IQ', ti[2], off)); off += ti[4]
    o.write(b'\0' * ((-o.tell()) % align))
    for ti, f, data in sel:
        f.seek(data + ti[3]); left = ti[4]
        while left:
            b = f.read(min(left, 64 << 20)); o.write(b); left -= len(b)
print('wrote', out, len(sel), 'tensors', os.path.getsize(out) >> 20, 'MiB')
