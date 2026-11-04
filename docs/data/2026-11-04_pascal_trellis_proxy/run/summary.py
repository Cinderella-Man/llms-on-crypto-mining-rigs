#!/usr/bin/env python3
# Summarizes ptq_proxy.py results as the table of registry Q929: per format, bits per weight, the importance-weighted
# error relative to a reference on the same matrices (geometric mean of the paired ratios; the reference = 1.0) for the
# gate/up matrices (2,560 wide), the down matrices (640 wide) and all of them, the same relative to a second reference
# (TQ2_T), and run/decode_cost.py's estimated one-token decode cost on a P106. Pure Python.
#
#   summary.py <proxy.tsv> [--ref file:q2_0_swift] [--ref2 tq2t+r128] [--md] [--three-term tcq8.L10,...]
import argparse, collections, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_cost  # noqa: E402

NAMES = {
    'file:q2_0_swift': "Q2_0, Swift-1.5's own file",
    'file:gyro_m': "TQ2_T, Gyro-M's own file (original checkpoint)",
    'file:gyro_m.alt': 'Gyro-M, rotation undone in the other order (check)',
    'file:iq3s_ista': "IQ3_S mix, ISTA's file (original checkpoint)",
    'file:iq3s_swift': "IQ3_S mix, Swift-1.5's own file",
    'ik2x_s4x8@128': 'N2P-128 (numpy)',
    'ik3x_s5x32@128': 'N3P-128 (numpy)',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv')
    ap.add_argument('--ref', default='file:q2_0_swift')
    ap.add_argument('--ref2', default='tq2t+r128')
    ap.add_argument('--md', action='store_true')
    ap.add_argument('--three-term', default='', help='tcq formats whose parity has three terms (their decode cost)')
    a = ap.parse_args()
    three = set(x for x in a.three_term.split(',') if x)
    rows = [l.rstrip('\n').split('\t') for l in open(a.tsv)]
    hdr, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(hdr)}
    err = collections.defaultdict(dict)
    bpw = {}
    for r in rows:
        f = r[ix['format']]
        err[f][(r[ix['layer']], r[ix['expert']], r[ix['kind']])] = float(r[ix['err_imp']])
        bpw[f] = float(r[ix['bpw']])
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

    def rel(f, ref, kinds):
        if ref not in err:
            return float('nan'), 0
        common = [k for k in err[f] if k in err[ref] and k[2] in kinds]
        return gm([err[f][k] / err[ref][k] for k in common]), len(common)

    GU, DN, ALL = ('gate', 'up'), ('down',), ('gate', 'up', 'down')
    out = []
    for f in err:
        gu, ngu = rel(f, a.ref, GU)
        dn, ndn = rel(f, a.ref, DN)
        al, nal = rel(f, a.ref, ALL) if ndn and ngu else (float('nan'), 0)
        r2, _ = rel(f, a.ref2, ALL if ndn and ngu else (GU if ngu else DN))
        ps = decode_cost.estimate(f, f.replace('+r128', '').replace('.i8', '') in three)
        out.append((f, bpw[f], gu, dn, al, r2, ps, ngu + ndn, gm(list(err[f].values()))))
    out.sort(key=lambda t: (t[4] if t[4] == t[4] else (t[2] if t[2] == t[2] else t[3])))
    fmt_ps = lambda p: '%.1f' % p if p else '—'
    fmt_r = lambda x: '%.3f' % x if x == x else '—'
    if a.md:
        print('| Format | bpw | gate/up | down | all | vs %s | est. ps | matrices |' % a.ref2)
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for f, b, gu, dn, al, r2, ps, n, _ in out:
            print('| %s | %s | %s | %s | %s | %s | %s | %d |' % (NAMES.get(f, '`%s`' % f), '%.4f' % b if b == b else 'mix',
                  fmt_r(gu), fmt_r(dn), fmt_r(al), fmt_r(r2), fmt_ps(ps), n))
    else:
        print('reference %s = 1.0; second reference %s' % (a.ref, a.ref2))
        print('%-24s %7s %7s %7s %7s %8s %7s %4s %9s' % ('format', 'bpw', 'gate/up', 'down', 'all', 'vs ref2', 'est.ps', 'n', 'gm err'))
        for f, b, gu, dn, al, r2, ps, n, g in out:
            print('%-24s %7.4f %7s %7s %7s %8s %7s %4d %9.5f' % (f, b, fmt_r(gu), fmt_r(dn), fmt_r(al), fmt_r(r2), fmt_ps(ps), n, g))


if __name__ == '__main__':
    main()
