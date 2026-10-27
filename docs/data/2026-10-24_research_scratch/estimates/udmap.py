# 2026-10-24: per-layer tensor types of a Flint GGUF and its sliding/global layer pattern, e.g. Unsloth's UD-Q4_K_XL
# (the types listed for the global layers in docs/research/notes/long_context_quality.md). Reads the header only.
#   python3 udmap.py flint-UD-Q4_K_XL.gguf      (run from this folder: it imports ggufhdr.py)
import struct, re, sys, collections
sys.path.insert(0, '.')
from ggufhdr import T
def header_full(path):
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    s = lambda: f.read(rd('Q')).decode('utf-8', 'replace')
    SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def val(t, keep):
        if t == 8: return s()
        if t == 9:
            et, n = rd('I'), rd('Q')
            if et in (8,9): return [val(et, False) for _ in range(n)]
            if keep: return [rd(SZ[et]) for _ in range(n)]
            f.seek(struct.calcsize('<'+SZ[et])*n, 1); return None
        return rd(SZ[t])
    assert f.read(4) == b'GGUF'
    rd('I'); nt, nkv = rd('Q'), rd('Q')
    kv = {}
    for _ in range(nkv):
        k = s(); kv[k] = val(rd('I'), 'pattern' in k)
    ts = []
    for _ in range(nt):
        name = s(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); rd('Q')
        ts.append((name, dims, t))
    return kv, ts
kv, ts = header_full(sys.argv[1])
sw = kv['flint.attention.sliding_window_pattern']; rp = kv['flint.attention.rope_pattern']
print('sliding pattern', sw); print('rope pattern', rp)
glob = [i for i,v in enumerate(sw) if not v]
print('global (no sliding):', glob, len(glob))
classes = ['attn_q','attn_k','attn_v','attn_output','ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp','ffn_gate_exps','ffn_up_exps','ffn_down_exps','ffn_gate_inp','ffn_gate','ffn_up','ffn_down']
tab = collections.defaultdict(dict)
for name, dims, t in ts:
    m = re.match(r'blk\.(\d+)\.(\w+)\.weight', name)
    if m: tab[int(m.group(1))][m.group(2)] = T[t][0]
print('layer', ' '.join(c.replace('ffn_','').replace('attn_','a_') for c in classes))
for l in range(49):
    print(l, 'G' if l in glob else ' ', ' '.join(tab[l].get(c,'-') for c in classes))
