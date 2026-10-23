#!/usr/bin/env python3
# Screened output head, offline: how often a cheap screen's top-M candidates contain what the exact head needs.
#
#   screen_head.py <bf16.gguf> <exact.gguf> <H.npy> <x spec> [...] [--screen name=head.q5_0|head.q4_0 ...]
#                  [--ranks 64,128,256,512] [--M 64,128,256,512,1024] [--out results.tsv]
#
# The exact head is exact.gguf's output.weight (Q8_0 in our files), computed in float32. Screens:
#   - each --screen file: a precomputed head in raw Q5_0/Q4_0 blocks (gptq_head.py quant);
#   - q4_0-rtn and q4_1-rtn: llama-quantize's round-to-nearest Q4_0 / Q4_1 of the BF16 head;
#   - rank-r for each --ranks r: the best rank-r approximation of the BF16 head under H (W ~ A B with
#     A = W H^1/2 V_r, B = V_r^T H^-1/2, V_r the top r eigenvectors of H^1/2 W^T W H^1/2), A and B stored in Q8_0.
# For every held-out row (position >= 1 of the x specs, which use gptq_head.py's x.f32@ids syntax) and every M:
#   top1    the exact head's top token is among the screen's top M (greedy decoding is exact);
#   top20   all of the exact head's top 20 are among them (top-k 20 sampling is exact: the sampler needs those logits);
#   nucleus all tokens the makers' sampler can pick (top-k 20, then top-p 0.95, T 1.0) are among them;
#   p_samp  the probability that the token this sampler draws is among them;
#   kld     KL divergence from the exact head of a head that uses exact logits for the M candidates and the screen's
#           logits for the rest (what perplexity and KLD would see);
# and the M a row needs for top20, as percentiles. A control screen, the exact head itself, must score 1 everywhere. Bytes: the screen's bytes plus M rows of the exact Q8_0 head, as a
# share of the exact head's bytes.
# H must not contain the held-out rows. Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gptq_head as g

KMAX = 4096          # ranks are resolved up to here; a row needing more counts as > KMAX
Q8 = 34 / 32         # bytes per weight of Q8_0
TOPK, TOPP = 20, 0.95


def q8_0(x):
    from gguf.quants import quantize, dequantize
    import gguf
    t = gguf.GGMLQuantizationType.Q8_0
    return dequantize(quantize(np.ascontiguousarray(x, np.float32), t), t).reshape(x.shape)


def logsoftmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nucleus(p20):
    """p20 [n, 20] exact probabilities of the exact top 20, in decreasing order -> mask of the tokens top-p 0.95 keeps
    and their sampling probabilities (top-k 20 renormalizes, then top-p keeps the smallest prefix reaching 0.95)"""
    q = p20 / p20.sum(axis=1, keepdims=True)
    c = np.cumsum(q, axis=1)
    keep = np.ones_like(q, bool)
    keep[:, 1:] = c[:, :-1] < TOPP
    w = np.where(keep, q, 0)
    return keep, w / w.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bf16'); p.add_argument('exact'); p.add_argument('hessian'); p.add_argument('x', nargs='+')
    p.add_argument('--screen', action='append', default=[])
    p.add_argument('--ranks', default='64,128,256,512'); p.add_argument('--M', default='64,128,256,512,1024')
    p.add_argument('--kld-M', default='256,1024'); p.add_argument('--out')
    a = p.parse_args()
    Ms = [int(m) for m in a.M.split(',')]
    kld_Ms = [int(m) for m in a.kld_M.split(',')]

    Wx = g.head_f32(g.head_tensor(a.exact)[1])
    rows, n = Wx.shape
    exact_bytes = rows * n * Q8
    Wb = g.head_f32(g.head_tensor(a.bf16)[1])

    # control: the exact head as its own screen must contain everything at every M and give KLD 0
    screens = [('control-exact', exact_bytes, lambda X: X @ Wx.T)]   # (name, bytes, logits function)
    for s in a.screen:
        name, path = s.split('=', 1)
        raw = np.fromfile(path, np.uint8)
        screens.append((name, raw.size, lambda X, W=g.unpack(raw.reshape(rows, -1), n): X @ W.T))
    c, d = g.rtn_ref(Wb, 16)
    screens.append(('q4_0-rtn', rows * n // 32 * 18, lambda X, W=g.unpack(g.pack(c, d, 'q4_0'), n): X @ W.T))
    del c, d
    from gguf.quants import quantize, dequantize
    import gguf
    t41 = gguf.GGMLQuantizationType.Q4_1
    W41 = dequantize(quantize(Wb, t41), t41).reshape(rows, n)
    screens.append(('q4_1-rtn', rows * n // 32 * 20, lambda X, W=W41: X @ W.T))

    H = np.load(a.hessian)
    ev, Q = np.linalg.eigh(H)
    ev = np.maximum(ev, ev.max() * 1e-9)
    Hh, Hih = (Q * np.sqrt(ev)) @ Q.T, (Q / np.sqrt(ev)) @ Q.T          # H^1/2, H^-1/2
    WH = Wb.astype(np.float64) @ Hh
    G = WH.T @ WH
    gv, V = np.linalg.eigh(G)
    V = V[:, ::-1]
    print(f'energy of W H^1/2 in the top 64/128/256/512 directions: ' + ', '.join(
        f'{gv[::-1][:r].sum() / gv.sum():.3f}' for r in (64, 128, 256, 512)), flush=True)
    for r in [int(x) for x in a.ranks.split(',')]:
        A = q8_0((WH @ V[:, :r]).astype(np.float32))
        B = q8_0((V[:, :r].T @ Hih).astype(np.float32))
        screens.append((f'rank-{r}', (rows + n) * r * Q8, lambda X, A=A, B=B: (X @ B.T) @ A.T))
    del WH, G, Wb

    stats = {nm: dict(rows=0, top1={m: 0 for m in Ms}, top20={m: 0 for m in Ms}, nuc={m: 0 for m in Ms},
                      psamp={m: 0.0 for m in Ms}, kld={m: 0.0 for m in kld_Ms}, need=[]) for nm, _, _ in screens}
    for spec in a.x:
        x, idx, sel = g.load_x(spec)
        use = np.flatnonzero(sel & (idx[:, 1] >= 1))
        print(f'{spec}: {len(use)} rows', flush=True)
        for i in range(0, len(use), 512):
            X = np.asarray(x[use[i:i + 512]], np.float32)
            Z = X @ Wx.T
            lz = logsoftmax(Z)
            t20 = np.argpartition(-Z, TOPK, axis=1)[:, :TOPK]
            t20 = np.take_along_axis(t20, np.argsort(-np.take_along_axis(Z, t20, 1), axis=1), 1)
            keep, ps = nucleus(np.exp(np.take_along_axis(lz, t20, 1)))
            for nm, _, f in screens:
                st = stats[nm]
                S = f(X)
                top = np.argpartition(-S, KMAX, axis=1)[:, :KMAX]
                top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
                hit = top[:, None, :] == t20[:, :, None]                   # [rows, 20, KMAX]
                rank = np.where(hit.any(axis=2), hit.argmax(axis=2), KMAX)   # screen rank of each exact top-20 token
                st['rows'] += len(X)
                st['need'].append(rank.max(axis=1) + 1)
                for m in Ms:
                    inm = rank < m
                    st['top1'][m] += inm[:, 0].sum()
                    st['top20'][m] += inm.all(axis=1).sum()
                    st['nuc'][m] += (inm | ~keep).all(axis=1).sum()
                    st['psamp'][m] += (ps * inm).sum()
                for m in kld_Ms:
                    hyb = S.copy()
                    cand = top[:, :m]
                    np.put_along_axis(hyb, cand, np.take_along_axis(Z, cand, 1), 1)
                    st['kld'][m] += (np.exp(lz) * (lz - logsoftmax(hyb))).sum()
            print(f'  {i + len(X)}/{len(use)}', flush=True)

    lines = ['screen\tscreen_MB\tscreen_share\tM\tread_share\ttop1\ttop20\tnucleus\tp_sampled\tkld_hybrid']
    for nm, sb, _ in screens:
        st = stats[nm]
        nr = st['rows']
        need = np.concatenate(st['need'])
        for m in Ms:
            read = (sb + m * n * Q8) / exact_bytes
            kl = f"{st['kld'][m] / nr:.5f}" if m in st['kld'] else ''
            lines.append(f"{nm}\t{sb / 1e6:.1f}\t{sb / exact_bytes:.3f}\t{m}\t{read:.3f}\t{st['top1'][m] / nr:.5f}\t"
                         f"{st['top20'][m] / nr:.5f}\t{st['nuc'][m] / nr:.5f}\t{st['psamp'][m] / nr:.6f}\t{kl}")
        pc = np.percentile(need, [50, 90, 99, 99.9])
        lines.append(f"# {nm}: M needed for the full top-20, percentiles 50/90/99/99.9: "
                     + '/'.join(str(int(v)) for v in pc) + f"; rows needing > {KMAX}: {(need > KMAX).sum()} of {nr}")
    out = '\n'.join(lines)
    print(out)
    if a.out:
        open(a.out, 'w').write(out + '\n')


if __name__ == '__main__':
    main()
