#!/usr/bin/env python3
"""The Ornith-1.5-35B candidates' table from run/candidates.sh's logs and the files' header summaries.

  python3 -I run/table.py OUT KLD HEADERS RECIPE ... > kld_35b.tsv

OUT holds quant-<recipe>.log (llama-quantize's output, for bits per weight), KLD the scores
Ornith-1.5-35B-<recipe>.<text>.kld.log (llama-perplexity --kl-divergence), HEADERS the summaries
header-Ornith-1.5-35B-<recipe>.txt (scripts/gguf_summary.py). RECIPE Q8_0 is the reference (the makers' file). One row
per recipe: GB (decimal) of tensor data, bits per weight, GB read per decoded token (8 of 256 routed experts per layer,
every other block tensor and the output layer, without the token embedding, a row lookup, and without the MTP block,
1 of 41), and per text the mean KL divergence with its standard error, the share of tokens whose top token matches the
reference's, and the perplexity with the reference's. Standard library only.
"""
import re
import sys

TEXTS = ('wiki', 'agent', 'code')
USED, EXPERTS, BLOCKS, MAIN = 8, 256, 41, 40


def grab(pattern, text):
    m = re.search(pattern, text)
    return tuple(float(g) for g in m.groups()) if m else None


def header(path):
    """(tensor GB, GB read per decoded token) from a gguf_summary.py output"""
    text = open(path, errors='replace').read()
    total = grab(r'tensor data ([0-9.]+) GB', text)[0]
    routed = other = out = 0.0
    part = text.split('per role and type', 1)[1].split('per block', 1)[0]
    for line in part.splitlines()[1:]:
        f = line.split()
        if len(f) != 4:
            continue
        role, gb = f[0], float(f[3])
        if role in ('token_embd.weight', 'output_norm.weight'):
            continue
        if role == 'output.weight':
            out += gb
        elif role.endswith('_exps'):
            routed += gb
        else:
            other += gb
    per_token = (routed * USED / EXPERTS + other) * MAIN / BLOCKS + out
    return total, per_token


def main():
    out, kld, hdr, recipes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    cols = ['recipe', 'gb', 'bpw', 'gb_per_token']
    for t in TEXTS:
        cols += [f'{t}_kld', f'{t}_kld_se', f'{t}_same_top_pct', f'{t}_ppl', f'{t}_ppl_ref']
    print('\t'.join(cols))
    for r in recipes:
        gb, per_token = header(f'{hdr}/header-Ornith-1.5-35B-{r}.txt')
        try:
            bpw = grab(r'quant size\s*=\s*[0-9.]+ MiB \(([0-9.]+) BPW\)', open(f'{out}/quant-{r}.log').read())
        except OSError:
            bpw = None
        row = [r, f'{gb:.2f}', f'{bpw[0]:.2f}' if bpw else '', f'{per_token:.2f}']
        for t in TEXTS:
            if r == 'Q8_0':
                row += ['0', '', '100', '', '']
                continue
            try:
                k = open(f'{kld}/Ornith-1.5-35B-{r}.{t}.kld.log', errors='replace').read()
            except OSError:
                row += [''] * 5
                continue
            mean = grab(r'Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)', k)
            same = grab(r'Same top p:\s+([0-9.]+)\s+±', k)
            ppl = grab(r'Mean PPL\(Q\)\s*:\s*([0-9.]+)', k)
            pref = grab(r'Mean PPL\(base\)\s*:\s*([0-9.]+)', k)
            row += [f'{mean[0]:.4f}' if mean else '', f'{mean[1]:.4f}' if mean else '',
                    f'{same[0]:.2f}' if same else '', f'{ppl[0]:.4f}' if ppl else '', f'{pref[0]:.4f}' if pref else '']
        print('\t'.join(row))


if __name__ == '__main__':
    main()
