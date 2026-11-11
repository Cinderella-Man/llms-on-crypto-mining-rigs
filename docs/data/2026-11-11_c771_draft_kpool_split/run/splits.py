#!/usr/bin/env python3
"""C771 (Q1357): the scheduler's splits of each evaluation, from llama-server's LLAMA_UBATCH_TRACE=1 lines.

Each evaluation logs a "sched-trace:" line (ggml_backend_sched: every split's device, inputs and nodes, in order) and
then a "ubatch-trace:" line (llama_context: the graph's kind, mtp for the MTP draft and dec for the target, with the
micro-batch's tokens and outputs). An arm's trace.txt holds both kinds of lines in the log's order.

Pairs are given as <arm a>:<trace a>:<arm b>:<trace b>, a without LLAMA_MTP_EMBD_FIRST and b with it. A pair passes when
- both arms have the same evaluations of each kind in the same order (tokens and outputs);
- every evaluation has the same nodes on each device in both arms (the sum of its splits' nodes there);
- each of b's draft evaluations has one split on the second device (--second, the endpoint of the draft's layer);
- the target's evaluations have the same splits in both arms (device and nodes, in order).
Writes --tsv: per arm, kind, tokens, outputs and split pattern, the evaluations and the splits on the second device.
"""
import argparse
import collections
import re
import sys

SPLIT = re.compile(r'(?:^|\s)(\S+?):i(\d+):n(\d+):w-?\d+:c-?\d+:l-?\d+')
PASS = re.compile(r'^ubatch-trace: (mtp|dec|enc) n_tokens = (\d+) n_seqs = \d+ n_outputs = (\d+)')


def device(name, second):
    if second in name:
        return 'dev2'
    if name.startswith('RPC'):
        return 'dev1'
    return name


def read(path, second):
    """The evaluations in order: (kind, n_tokens, n_outputs, [(device, inputs, nodes), ...])."""
    passes, pending, orphans = [], None, 0
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('sched-trace:'):
                if pending is not None:
                    orphans += 1
                body = line.split('|', 1)[1] if '|' in line else ''
                pending = [(device(m.group(1), second), int(m.group(2)), int(m.group(3))) for m in SPLIT.finditer(body)]
                continue
            m = PASS.match(line)
            if m is None:
                continue
            if pending is None:
                raise SystemExit(f'{path}: an evaluation without its scheduler line: {line.strip()}')
            passes.append((m.group(1), int(m.group(2)), int(m.group(3)), pending))
            pending = None
    return passes, orphans


def nodes_per_device(splits):
    c = collections.Counter()
    for dev, _, n in splits:
        c[dev] += n
    return dict(sorted(c.items()))


def pattern(splits):
    return ' '.join(f'{dev}:n{n}' for dev, _, n in splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--second', required=True, help="the second device's endpoint (the draft's layer)")
    ap.add_argument('--tsv', required=True)
    ap.add_argument('pairs', nargs='+')
    args = ap.parse_args()

    rows = []
    fail = False
    for spec in args.pairs:
        a, ta, b, tb = spec.split(':', 3) if spec.count(':') == 3 else (None,) * 4
        if a is None:
            raise SystemExit(f'bad pair {spec!r}')
        try:
            pa, oa = read(ta, args.second)
            pb, ob = read(tb, args.second)
        except OSError as e:
            print(f'{a} -> {b}: FAIL: {e}')
            fail = True
            continue
        for arm, ps in ((a, pa), (b, pb)):
            groups = collections.Counter((k, nt, no, pattern(s), sum(d == 'dev2' for d, _, _ in s)) for k, nt, no, s in ps)
            for (k, nt, no, pat, n2), n in sorted(groups.items()):
                rows.append((arm, k, nt, no, pat, n, n2))

        problems = []
        if oa or ob:
            problems.append(f'scheduler lines without an evaluation: {a} {oa}, {b} {ob}')
        if [p[:3] for p in pa] != [p[:3] for p in pb]:
            problems.append(f'the evaluations differ ({len(pa)} and {len(pb)})')
        mtp_a = [p for p in pa if p[0] == 'mtp']
        mtp_b = [p for p in pb if p[0] == 'mtp']
        dec_a = [p for p in pa if p[0] == 'dec']
        dec_b = [p for p in pb if p[0] == 'dec']
        if not mtp_a:
            problems.append(f'{a}: no draft evaluation traced')
        diff_nodes = sum(nodes_per_device(x[3]) != nodes_per_device(y[3]) for x, y in zip(pa, pb))
        if diff_nodes:
            problems.append(f'{diff_nodes} evaluations with other nodes on a device')
        b_not_one = sum(sum(d == 'dev2' for d, _, _ in p[3]) != 1 for p in mtp_b)
        if b_not_one:
            problems.append(f'{b}: {b_not_one} draft evaluations without exactly one split on the second device')
        dec_diff = sum([(d, n) for d, _, n in x[3]] != [(d, n) for d, _, n in y[3]] for x, y in zip(dec_a, dec_b))
        if dec_diff or len(dec_a) != len(dec_b):
            problems.append(f'{dec_diff} target evaluations with other splits')

        def hist(ps, f):
            return dict(sorted(collections.Counter(f(p) for p in ps).items()))
        n2 = lambda p: sum(d == 'dev2' for d, _, _ in p[3])
        ns = lambda p: len(p[3])
        msg = (f'{a} -> {b}: {len(mtp_a)} draft evaluations one to one; splits on the second device per evaluation '
               f'{hist(mtp_a, n2)} -> {hist(mtp_b, n2)}; splits per evaluation {hist(mtp_a, ns)} -> {hist(mtp_b, ns)}; '
               f'the same nodes on each device in all {len(pa)} evaluations; {len(dec_a)} target evaluations with the '
               f'same splits')
        if problems:
            fail = True
            print(f'{a} -> {b}: FAIL: ' + '; '.join(problems))
            print('  ' + msg)
        else:
            print(msg)

    with open(args.tsv, 'w', encoding='utf-8') as f:
        f.write('arm\tkind\tn_tokens\tn_outputs\tsplits\tevaluations\tsplits_on_second\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
