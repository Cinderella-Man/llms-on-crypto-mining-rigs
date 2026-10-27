#!/usr/bin/env python3
# Summarizes lowbit_proxy.py results: per format, bits per weight and the importance-weighted relative error of the
# expert gate, up and down matrices (geometric mean over the sampled matrices, and the paired ratio to a reference
# format on the same matrices), then the expert mixes given with --mix (gate/up format, down format) at their average
# bits per weight. Pure Python.
#
#   lowbit_summary.py <proxy.tsv> [--ref IQ3_XXS] [--mix IQ2_S/IQ3_XXS Q2_K/Q3_K ...]
import argparse, collections, math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv')
    ap.add_argument('--ref', default='IQ3_XXS')
    ap.add_argument('--mix', nargs='*', default=[])
    a = ap.parse_args()
    rows = [l.rstrip('\n').split('\t') for l in open(a.tsv)]
    hdr, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(hdr)}
    err = collections.defaultdict(dict)    # (format) -> {(layer, expert, kind): err}
    bpw = {}
    for r in rows:
        f = r[ix['format']]
        err[f][(r[ix['layer']], r[ix['expert']], r[ix['kind']])] = float(r[ix['err_imp']])
        bpw[f] = float(r[ix['bpw']])
    kinds = ('gate', 'up', 'down')
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')
    ref = err[a.ref]
    print('matrices per format: %s' % sorted({len(v) for v in err.values()}))
    print('%-16s %6s  %8s %8s %8s  %8s  %s' % ('format', 'bpw', 'gate', 'up', 'down', 'all', 'vs ' + a.ref))
    for f in sorted(err, key=lambda f: (-bpw[f], f)):
        per = [gm([v for k, v in err[f].items() if k[2] == kind]) for kind in kinds]
        common = [k for k in err[f] if k in ref]
        ratio = gm([err[f][k] / ref[k] for k in common])
        print('%-16s %6.4f  %8.5f %8.5f %8.5f  %8.5f  %.3f' % (f, bpw[f], *per, gm(list(err[f].values())), ratio))
    if a.mix:
        print('\nexpert mixes (gate and up / down): average bpw, error per kind, sum of the three')
        for m in a.mix:
            gu, dn = m.split('/')
            if gu not in err or dn not in err:
                continue
            b = (2 * bpw[gu] + bpw[dn]) / 3
            e = [gm([v for k, v in err[gu].items() if k[2] == 'gate']), gm([v for k, v in err[gu].items() if k[2] == 'up']),
                 gm([v for k, v in err[dn].items() if k[2] == 'down'])]
            print('%-24s %6.4f  %8.5f %8.5f %8.5f  %8.5f' % (m, b, *e, sum(e)))


if __name__ == '__main__':
    main()
