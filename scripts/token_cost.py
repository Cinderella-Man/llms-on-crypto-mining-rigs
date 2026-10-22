#!/usr/bin/env python3
# Bytes of weights read per generated token, by tensor kind and by format, from a GGUF header alone (pure Python),
# and a predicted decode time per token on the P106-090 rig:
#
#   time = sum over formats (bytes read in that format / (decode efficiency of the format * 173 GB/s)) + fixed time
#
# Mixture-of-Experts tensors (*_exps) count only the experts used per token; the token-embedding table is left out
# (llama.cpp keeps it in host memory and reads one row). Decode efficiencies are measured fractions of the card's
# 173 GB/s fitted to measured Flint speeds (below) and can be overridden; the fixed time covers kernel launches and
# other per-token work that doesn't scale with bytes.
#
#   token_cost.py <file.gguf> [<file.gguf> ...] [--eff Q4_1=0.40 ...] [--fixed-ms 11]
import argparse, collections, re, struct

T = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
     8: ('Q8_0', 32, 34), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
     14: ('Q6_K', 256, 210), 16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50),
     20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 29: ('IQ1_M', 256, 56),
     30: ('BF16', 1, 2), 39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36)}
# Fractions of 173 GB/s, fitted to llama-bench tg128 of four Flint files on the rig (llama.cpp d77256999 + CUDA graphs:
# makers' Q4_1 35.8 ms/token, imatrix Q5_0 44.8, Unsloth Q8_0 48.8, UD-Q4_K_XL 52.6) with 11 ms of fixed time per
# token. The K-/I-quant value is one number for Q4_K, Q5_K, Q6_K, IQ4_XS and IQ3_S together. Formats marked "guess"
# have no measurement on these cards yet. On a big dense model (Llama-2-7B) the same card reaches 0.42 with Q4_0 and
# 0.54 with Q8_0: Flint's many small expert matrices are decoded less efficiently.
EFF = {'Q4_1': 0.38, 'Q8_0': 0.41, 'Q5_0': 0.30, 'Q4_K': 0.19, 'Q5_K': 0.19, 'Q6_K': 0.19, 'IQ4_XS': 0.19, 'IQ3_S': 0.19,
       'F32': 0.60, 'BF16': 0.50, 'F16': 0.50,
       'Q4_0': 0.40, 'Q5_1': 0.29, 'IQ4_NL': 0.30, 'MXFP4': 0.30, 'NVFP4': 0.28}   # these five: guesses
GUESSED = {'Q4_0', 'Q5_1', 'IQ4_NL', 'MXFP4', 'NVFP4', 'F32', 'BF16', 'F16'}

def header(path):
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    s = lambda: f.read(rd('Q')).decode('utf-8', 'replace')
    SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et, n = rd('I'), rd('Q')
            return [val(et) for _ in range(n)] if et in (8, 9) else [rd(SZ[et]) for _ in range(n)]
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

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+')
ap.add_argument('--eff', nargs='*', default=[], help='FORMAT=fraction overrides')
ap.add_argument('--fixed-ms', type=float, default=11.0, help='per-token time that does not scale with bytes')
ap.add_argument('--peak', type=float, default=173.0, help='card memory bandwidth in GB/s')
a = ap.parse_args()
for e in a.eff:
    k, v = e.split('='); EFF[k.upper()] = float(v)

for path in a.files:
    kv, ts = header(path)
    arch = kv['general.architecture']
    n_exp = kv.get(f'{arch}.expert_count', 1); n_used = kv.get(f'{arch}.expert_used_count', 1)
    by_kind = collections.defaultdict(int); by_fmt = collections.defaultdict(int); fmts = collections.defaultdict(set)
    for name, dims, t in ts:
        if name == 'token_embd.weight':
            continue
        fmt, bs, bb = T.get(t, (f'type{t}', 1, 1))
        n = 1
        for d in dims: n *= d
        b = n // bs * bb
        if '_exps' in name and len(dims) == 3:
            b = b * n_used / n_exp
        kind = re.sub(r'blk\.\d+\.', '', name).replace('.weight', '')
        by_kind[kind] += b; by_fmt[fmt] += b; fmts[kind].add(fmt)
    tot = sum(by_fmt.values())
    ms_bytes = sum(b / (EFF.get(f, 0.35) * a.peak * 1e9) * 1e3 for f, b in by_fmt.items())
    ms = ms_bytes + a.fixed_ms
    print(f'== {path}')
    print(f'   bytes read per token: {tot/1e9:.3f} GB; predicted {ms:.1f} ms/token = {1000/ms:.1f} tok/s '
          f'({ms_bytes:.1f} ms reading weights + {a.fixed_ms:.1f} ms fixed)')
    for f, b in sorted(by_fmt.items(), key=lambda x: -x[1]):
        unk = ' (guess)' if f in GUESSED else ('' if f in EFF else ' (unknown, 0.35 assumed)')
        print(f'   {f:7s} {b/1e9:6.3f} GB  {100*b/tot:5.1f} %  at {EFF.get(f, 0.35):.2f} of peak{unk}')
    for k, b in sorted(by_kind.items(), key=lambda x: -x[1])[:12]:
        if b / tot >= 0.005:
            print(f'   {k:22s} {b/1e9:6.3f} GB  {100*b/tot:5.1f} %  {"/".join(sorted(fmts[k]))}')
