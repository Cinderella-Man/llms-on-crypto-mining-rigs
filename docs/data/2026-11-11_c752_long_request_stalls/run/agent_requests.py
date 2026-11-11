#!/usr/bin/env python3
"""Per request of an agent check: the step time against a fit of the arms' steps, and the host's swap around it.

Usage: agent_requests.py OUT --arms a,b [--tsv per_request.tsv] [--tz +01:00] > requests.md

OUT is an agent check's collected folder (bench/agent_check): proxy.jsonl (one line per request with the server's
timings), rig/ram_watchdog-<arm>.tsv (the rig's memory once a second) and rig/vmstat.log (`vmstat -t 10`, local
time in --tz).

A generation step is one verify pass: steps = predicted_n - draft_n_accepted (each step gives its accepted drafts
plus one token). The fit is one line for all arms, generation time = c0 + steps * (b0 + b1 * depth), depth the
context at the reply's middle, by least squares, refitted twice without the requests whose residual is above three
scaled median absolute deviations. A request's excess is its generation time minus the fit; an outlier is a request
above the last fit's cut. A wait like R86's (about 117 ms before a verify pass, Q1328) adds 117 ms per step it hits,
so `excess / 117` is the number of such steps a request could hold.

The swap columns: swap used (MiB) at the request's start and its generation end from the watchdog, the swap-out
during the request (the rise of swap used), the swap-in during its generation from vmstat (si, KiB/s over each
10-second line, weighted by the overlap), and the share of CPU time in iowait over the generation (wa, the same way).
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys


def load_watchdog(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip('\n').split('\t')
            rows.append((float(p[0]), float(p[1]), float(p[4]), float(p[6]), float(p[7])))
    return rows


def load_vmstat(path, tz):
    sign = 1 if tz[0] == '+' else -1
    hh, mm = tz[1:].split(':')
    off = dt.timezone(sign * dt.timedelta(hours=int(hh), minutes=int(mm)))
    rows, first = [], True
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 19 or not p[0].isdigit():
                continue
            if first:  # the first line averages since boot
                first = False
                continue
            t = dt.datetime.strptime(p[-2] + ' ' + p[-1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=off).timestamp()
            rows.append((t - 10.0, t, int(p[6]), int(p[7]), int(p[8]), int(p[15])))
    return rows


def vm_over(vm, t0, t1):
    """si (KiB), so (KiB), bi (KiB) summed and wa averaged over [t0, t1], each 10 s line weighted by its overlap."""
    si = so = bi = wa = w = 0.0
    for a, b, s_in, s_out, b_in, w_a in vm:
        o = min(b, t1) - max(a, t0)
        if o <= 0:
            continue
        si += s_in * o
        so += s_out * o
        bi += b_in * o
        wa += w_a * o
        w += o
    return si, so, bi, (wa / w if w else float('nan'))


def wd_at(wd, t):
    best = None
    for r in wd:
        if r[0] <= t:
            best = r
        else:
            break
    return best or (wd[0] if wd else None)


def wd_span(wd, t0, t1):
    return [r for r in wd if t0 <= r[0] <= t1]


def solve3(a, b):
    m = [row[:] + [v] for row, v in zip(a, b)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        for r in range(3):
            if r != i:
                f = m[r][i] / m[i][i]
                for c in range(i, 4):
                    m[r][c] -= f * m[i][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit(reqs):
    xs = [(1.0, r['steps'], r['steps'] * r['depth_mid']) for r in reqs]
    ys = [r['predicted_ms'] for r in reqs]
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    return solve3(ata, aty)


def pred(c, r):
    return c[0] + r['steps'] * (c[1] + c[2] * r['depth_mid'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--arms', required=True)
    ap.add_argument('--tsv')
    ap.add_argument('--tz', default='+01:00')
    ap.add_argument('--wait-ms', type=float, default=117.0)
    a = ap.parse_args()
    arms = a.arms.split(',')

    vm = load_vmstat(os.path.join(a.out, 'rig', 'vmstat.log'), a.tz)
    wds = {arm: load_watchdog(os.path.join(a.out, 'rig', f'ram_watchdog-{arm}.tsv')) for arm in arms}

    reqs = []
    with open(os.path.join(a.out, 'proxy.jsonl')) as f:
        for line in f:
            x = json.loads(line)
            tag = x['tag'].split('/')
            arm = tag[0]
            t = x.get('timings') or {}
            if arm not in arms or not t or not t.get('predicted_n'):
                continue
            steps = t['predicted_n'] - t.get('draft_n_accepted', 0)
            if steps <= 0:
                continue
            t_start = x['t_wall']
            t_gen = t_start + x['t_first_token']
            t_end = t_start + x['t_end']
            depth0 = t['prompt_n'] + t['cache_n']
            wd = wds[arm]
            w0, wg, we = wd_at(wd, t_start), wd_at(wd, t_gen), wd_at(wd, t_end)
            span = wd_span(wd, t_start, t_end)
            si, so, bi, wa = vm_over(vm, t_gen, t_end)
            pso = vm_over(vm, t_start, t_gen)[1]
            reqs.append(dict(
                arm=arm, task=tag[1], turn=int(tag[-1]), t_start=t_start,
                prompt_n=t['prompt_n'], cache_n=t['cache_n'], predicted_n=t['predicted_n'],
                accepted=t.get('draft_n_accepted', 0), steps=steps, predicted_ms=t['predicted_ms'],
                depth_mid=depth0 + t['predicted_n'] / 2.0, ckpt_n=t.get('ckpt_created_n', 0),
                ckpt_ms=t.get('ckpt_save_ms', 0.0), restore_ms=t.get('cache_restore_ms', 0.0),
                avail_start=w0[1], avail_min=min([r[1] for r in span] or [w0[1]]),
                swap_start=w0[2], swap_gen=wg[2], swap_end=we[2],
                swap_out=max(0.0, we[2] - w0[2]), so_prompt_kib=pso, si_gen_kib=si, so_gen_kib=so,
                bi_gen_kib=bi, wa_gen=wa,
                psi_full_max=max([r[4] for r in span] or [0.0])))
    reqs.sort(key=lambda r: r['t_start'])

    use = [r for r in reqs if r['steps'] >= 10]
    c = fit(use)
    for _ in range(2):
        res = [r['predicted_ms'] - pred(c, r) for r in use]
        med = statistics.median(res)
        mad = statistics.median([abs(v - med) for v in res]) * 1.4826
        cut = med + 3 * mad
        keep = [r for r, v in zip(use, res) if v <= cut]
        c = fit(keep)
    res = [r['predicted_ms'] - pred(c, r) for r in use]
    med = statistics.median(res)
    mad = statistics.median([abs(v - med) for v in res]) * 1.4826
    cut = med + 3 * mad
    for r in reqs:
        r['fit_ms'] = pred(c, r)
        r['excess_ms'] = r['predicted_ms'] - r['fit_ms']
        r['outlier'] = int(r['steps'] >= 10 and r['excess_ms'] > cut)
        r['ms_step'] = r['predicted_ms'] / r['steps']

    print(f"# Arms {', '.join(arms)}: generation time per request against one fit of the arms\n")
    print(f"Fit on {len(keep)} of {len(use)} requests with 10 steps or more: generation ms = {c[0]:.1f} + steps x "
          f"({c[1]:.2f} + {c[2] * 1000:.3f} x depth / 1000); residual median {med:.1f} ms, scaled MAD {mad:.1f} ms, "
          f"outlier cut {cut:.1f} ms.\n")

    tasks = {arm: len({r['task'] for r in reqs if r['arm'] == arm}) for arm in arms}
    print('| arm | tasks | requests | steps | median ms a step | requests above the cut | their excess, s | '
          'excess over all requests, s | excess a task, s | as 117 ms waits | swap-out, MiB | swap-in in generation, MiB | '
          'iowait in generation, % (mean) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for arm in arms:
        rs = [r for r in reqs if r['arm'] == arm]
        out = [r for r in rs if r['outlier']]
        tot_ex = sum(r['excess_ms'] for r in rs if r['steps'] >= 10)
        steps = sum(r['steps'] for r in rs)
        gen_ms = sum(r['predicted_ms'] for r in rs)
        wa = sum(r['wa_gen'] * r['predicted_ms'] for r in rs if r['wa_gen'] == r['wa_gen']) / gen_ms
        print(f"| {arm} | {tasks[arm]} | {len(rs)} | {steps} | {statistics.median([r['ms_step'] for r in rs]):.1f} | "
              f"{len(out)} | {sum(r['excess_ms'] for r in out) / 1000:.2f} | {tot_ex / 1000:.2f} | "
              f"{tot_ex / 1000 / tasks[arm]:.2f} | {tot_ex / a.wait_ms:.0f} | "
              f"{sum(r['swap_out'] for r in rs):.0f} | {sum(r['si_gen_kib'] for r in rs) / 1024:.1f} | {wa:.2f} |")

    print('\n## The requests above the cut\n')
    print('| arm | task | turn | depth at start | steps | ms a step | excess, ms | as 117 ms waits | checkpoints saved, ms | '
          'MemAvailable start, min | swap at start, at the end | swap-out in the prompt, MiB | swap-in in generation, MiB | iowait, % |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|')
    for r in reqs:
        if not r['outlier']:
            continue
        print(f"| {r['arm']} | {r['task']} | {r['turn']} | {r['prompt_n'] + r['cache_n']} | {r['steps']} | "
              f"{r['ms_step']:.1f} | {r['excess_ms']:.0f} | {r['excess_ms'] / a.wait_ms:.1f} | {r['ckpt_n']}, "
              f"{r['ckpt_ms']:.0f} | {r['avail_start']:.0f}, {r['avail_min']:.0f} | {r['swap_start']:.0f}, "
              f"{r['swap_end']:.0f} | {r['so_prompt_kib'] / 1024:.1f} | {r['si_gen_kib'] / 1024:.2f} | {r['wa_gen']:.1f} |")

    print('\n## Each arm\'s first request (the agent\'s first turn, the first request after the load)\n')
    print('| arm | task | prompt | steps | ms a step | excess, ms | checkpoints, save ms | MemAvailable start, min | '
          'swap start, end | swap-in in generation, MiB | iowait, % |')
    print('|---|---|---:|---:|---:|---:|---|---|---|---:|---:|')
    for arm in arms:
        r = next(r for r in reqs if r['arm'] == arm)
        print(f"| {arm} | {r['task']} | {r['prompt_n']} | {r['steps']} | {r['ms_step']:.1f} | {r['excess_ms']:.0f} | "
              f"{r['ckpt_n']}, {r['ckpt_ms']:.0f} | {r['avail_start']:.0f}, {r['avail_min']:.0f} | "
              f"{r['swap_start']:.0f}, {r['swap_end']:.0f} | {r['si_gen_kib'] / 1024:.2f} | {r['wa_gen']:.1f} |")

    print('\n## Excess against swap traffic\n')
    print('Requests with 10 steps or more, split by the swap-in during their generation.\n')
    print('| arm | swap-in in generation | requests | median excess, ms | mean excess, ms | above the cut |')
    print('|---|---|---:|---:|---:|---:|')
    for arm in arms:
        rs = [r for r in reqs if r['arm'] == arm and r['steps'] >= 10]
        for name, sel in (('none (< 0.1 MiB)', lambda r: r['si_gen_kib'] < 102.4),
                          ('0.1 to 1 MiB', lambda r: 102.4 <= r['si_gen_kib'] < 1024),
                          ('1 MiB or more', lambda r: r['si_gen_kib'] >= 1024)):
            g = [r for r in rs if sel(r)]
            if not g:
                print(f'| {arm} | {name} | 0 | | | |')
                continue
            ex = [r['excess_ms'] for r in g]
            print(f"| {arm} | {name} | {len(g)} | {statistics.median(ex):.0f} | {statistics.mean(ex):.0f} | "
                  f"{sum(r['outlier'] for r in g)} |")

    print('\n## Excess against swap-out in the prompt\n')
    print('Requests with 10 steps or more, split by the swap-out (vmstat so) between the request\'s start and its first '
          'token: R86\'s waits followed a swap-out at the end of the prompt.\n')
    print('| arm | swap-out in the prompt | requests | median excess, ms | their excess, s | excess a task, s | above the cut |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for arm in arms:
        rs = [r for r in reqs if r['arm'] == arm and r['steps'] >= 10]
        for name, sel in (('under 20 MiB', lambda r: r['so_prompt_kib'] < 20 * 1024),
                          ('20 MiB or more', lambda r: r['so_prompt_kib'] >= 20 * 1024)):
            g = [r for r in rs if sel(r)]
            if not g:
                print(f'| {arm} | {name} | 0 | | | | |')
                continue
            ex = [r['excess_ms'] for r in g]
            print(f"| {arm} | {name} | {len(g)} | {statistics.median(ex):.0f} | {sum(ex) / 1000:.2f} | "
                  f"{sum(ex) / 1000 / tasks[arm]:.2f} | {sum(r['outlier'] for r in g)} |")

    if a.tsv:
        cols =['arm', 'task', 'turn', 't_start', 'prompt_n', 'cache_n', 'predicted_n', 'accepted', 'steps',
                'predicted_ms', 'ms_step', 'depth_mid', 'fit_ms', 'excess_ms', 'outlier', 'ckpt_n', 'ckpt_ms',
                'restore_ms', 'avail_start', 'avail_min', 'swap_start', 'swap_gen', 'swap_end', 'swap_out',
                'so_prompt_kib', 'si_gen_kib', 'so_gen_kib', 'bi_gen_kib', 'wa_gen', 'psi_full_max']
        with open(a.tsv, 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in reqs:
                f.write('\t'.join(f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k]) for k in cols) + '\n')


if __name__ == '__main__':
    sys.exit(main())
