#!/usr/bin/env python3
"""Per request of a diagnostic load: the host waits, the swap and the disk traffic while it ran.

Usage: load_vm.py --load NAME=DIR[,REQ1,REQ2,...] ... --waits WAITS.tsv ... [--tz +01:00] > loads_vm.md

DIR is one load's output folder from bench/agent_check/rig_server.sh's monitors: server-<NAME>.log (the first line
"=== <ISO time> ..." is the server's start; the log's times count from there), vmstat.log (`vmstat -t 10`, local
time in --tz) and ram_watchdog-<NAME>.tsv (MemAvailable and swap used once a second). REQ1,... name the load's
requests in the order they ran (default greedy,long). WAITS.tsv are S37's waits.py tables (load, request,
steps_with_a_long_turn, median_long_turn_ms, ...), joined by load and request.

For each request: its prompt window (launch to the first generated token) and its generation window (to the
release), from the server's log; the swap used at the start and at the generation's start and end; the swap-in, the
disk reads and the iowait share over the generation from vmstat, each 10 s line weighted by its overlap. vmstat's
lines are 10 s wide, so a short generation borrows from its neighbours: read them as the traffic around it.
"""
import argparse
import datetime as dt
import os
import re
import sys

TS = re.compile(r'^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ')


def rel_s(line):
    m = TS.match(line)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6


def tzinfo(tz):
    sign = 1 if tz[0] == '+' else -1
    hh, mm = tz[1:].split(':')
    return dt.timezone(sign * dt.timedelta(hours=int(hh), minutes=int(mm)))


def server_requests(path):
    t0, reqs, cur = None, [], {}
    with open(path, errors='replace') as f:
        for line in f:
            if line.startswith('=== ') and t0 is None:
                t0 = dt.datetime.fromisoformat(line.split()[1]).timestamp()
                continue
            t = rel_s(line)
            if t is None:
                continue
            m = re.search(r'task (\d+) \| processing task', line)
            if m:
                cur = {'task': int(m.group(1)), 'launch': t}
                continue
            m = re.search(r'task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens', line)
            if m and cur:
                cur['eval_ms'] = float(m.group(2))
                continue
            m = re.search(r'task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens', line)
            if m and cur:
                cur['prompt_ms'] = float(m.group(2))
                cur['prompt_n'] = int(m.group(3))
                continue
            if 'stop processing' in line and cur:
                cur['release'] = t
                reqs.append(cur)
                cur = {}
    for r in reqs:
        r['launch'] += t0
        r['release'] += t0
        r['gen'] = r['release'] - r.get('eval_ms', 0.0) / 1e3
    return reqs


def vmstat(path, tz):
    rows, first = [], True
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 19 or not p[0].isdigit():
                continue
            if first:
                first = False
                continue
            t = dt.datetime.strptime(p[-2] + ' ' + p[-1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()
            rows.append((t - 10.0, t, int(p[6]), int(p[7]), int(p[8]), int(p[15])))
    return rows


def vm_over(vm, t0, t1):
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
    return si / 1024, so / 1024, bi / 1024, (wa / w if w else float('nan'))


def watchdog(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split('\t')
            rows.append((float(p[0]), float(p[1]), float(p[4])))
    return rows


def at(wd, t):
    best = wd[0]
    for r in wd:
        if r[0] <= t:
            best = r
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', action='append', required=True)
    ap.add_argument('--waits', action='append', default=[])
    ap.add_argument('--build', action='append', default=[], help='NAME=label, the build a load ran')
    ap.add_argument('--tz', default='+01:00')
    a = ap.parse_args()
    tz = tzinfo(a.tz)

    waits = {}
    for path in a.waits:
        with open(path) as f:
            head = f.readline().rstrip('\n').split('\t')
            for line in f:
                d = dict(zip(head, line.rstrip('\n').split('\t')))
                waits[(d['load'], d['request'])] = d
    builds = dict(b.split('=', 1) for b in a.build)

    print('| build | load | request | prompt tokens | MemAvailable at launch, min (MiB) | swap used at launch, '
          'at the first token, at the end (MiB) | steps with a wait over 50 ms | median wait, ms | '
          'swap-in in generation, MiB | disk reads in generation, MiB | iowait in generation, % |')
    print('|---|---|---|---:|---|---|---:|---:|---:|---:|---:|')
    for spec in a.load:
        name, rest = spec.split('=', 1)
        parts = rest.split(',')
        d, names = parts[0], (parts[1:] or ['greedy', 'long'])
        reqs = server_requests(os.path.join(d, f'server-{name}.log'))
        vm = vmstat(os.path.join(d, 'vmstat.log'), tz)
        wd = watchdog(os.path.join(d, f'ram_watchdog-{name}.tsv'))
        for rname, r in zip(names, reqs):
            si, so, bi, wa = vm_over(vm, r['gen'], r['release'])
            span = [x for x in wd if r['launch'] <= x[0] <= r['release']] or [at(wd, r['launch'])]
            w = waits.get((name, rname), {})
            n = w.get('steps_with_a_long_turn', '')
            med = w.get('median_long_turn_ms', '') if n not in ('', '0') else ''
            print(f"| {builds.get(name, '')} | {name} | {rname} | {r.get('prompt_n', '')} | "
                  f"{at(wd, r['launch'])[1]:.0f}, {min(x[1] for x in span):.0f} | "
                  f"{at(wd, r['launch'])[2]:.0f}, {at(wd, r['gen'])[2]:.0f}, {at(wd, r['release'])[2]:.0f} | "
                  f"{n} | {med} | {si:.1f} | {bi:.1f} | {wa:.1f} |")


if __name__ == '__main__':
    sys.exit(main())
