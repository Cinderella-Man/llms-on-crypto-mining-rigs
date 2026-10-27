#!/usr/bin/env python3
# Research round 16 (power and tuning, 2026-10-24): per llama-bench test the power, clocks and power-cap share of
# every card, from the nvidia-smi trace (smi.csv) next to the llama-bench jsonl of each run folder; output phases.txt.
"""Per-phase power/clock/throttle statistics from nvidia-smi logs next to llama-bench jsonl files.

usage: smi_phases.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR holds smi.csv (timestamp,index,pstate,sm,mem,power,temp,util,mem.used,reasons) and *.jsonl
from llama-bench. smi timestamps are local time (+02:00); jsonl test_time is UTC.
"""
import sys, glob, json, os, datetime as dt
from collections import defaultdict

TZ = dt.timedelta(hours=2)

def parse_smi(path):
    rows = []
    with open(path) as f:
        hdr = f.readline()
        for line in f:
            p = [x.strip() for x in line.split(',')]
            if len(p) < 10:
                continue
            try:
                t = dt.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f') - TZ
                rows.append(dict(t=t, gpu=int(p[1]), ps=p[2], sm=int(p[3].split()[0]), mem=int(p[4].split()[0]),
                                 pw=float(p[5].split()[0]), temp=int(p[6]), util=int(p[7].split()[0]),
                                 used=int(p[8].split()[0]), reasons=int(p[9], 16)))
            except Exception:
                pass
    return rows

def windows(run):
    out = []
    for jf in glob.glob(os.path.join(run, '*.jsonl')):
        for line in open(jf):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if 'test_time' not in d:
                continue
            t0 = dt.datetime.strptime(d['test_time'], '%Y-%m-%dT%H:%M:%SZ')
            s = d['samples_ns']
            dur = sum(s) / 1e9 + (s[0] / 1e9 if d['n_prompt'] > 0 else 0.05)  # + warmup
            kind = 'pp' if d['n_prompt'] > 0 and d['n_gen'] == 0 else ('tg' if d['n_gen'] > 0 and d['n_prompt'] == 0 else 'mix')
            ntok = (d['n_prompt'] + d['n_gen'])
            out.append(dict(kind=kind, t0=t0, t1=t0 + dt.timedelta(seconds=dur), n_prompt=d['n_prompt'], n_gen=d['n_gen'],
                            depth=d.get('n_depth', 0), ub=d.get('n_ubatch'), ts=d['avg_ts'], ntok=ntok,
                            model=os.path.basename(d['model_filename']), gpus=d.get('gpu_info', '').count('P106'),
                            fa=d.get('flash_attn'), tsplit=d.get('tensor_split'), jf=os.path.basename(jf)))
    return sorted(out, key=lambda w: w['t0'])

def stats(rows, w):
    sel = [r for r in rows if w['t0'] + dt.timedelta(seconds=0.6) <= r['t'] <= w['t1']]
    if not sel:
        return None
    by = defaultdict(list)
    for r in sel:
        by[r['gpu']].append(r)
    used = [g for g in by if max(r['used'] for r in by[g]) > 200]
    if not used:
        return None
    tot_pw = defaultdict(float)  # per sampling round total power over all logged cards
    rnd = -1
    for r in sel:
        if r['gpu'] == 0 or rnd < 0:
            rnd += 1
        r['rnd'] = rnd
        tot_pw[rnd] += r['pw']
    busy = [r for g in used for r in by[g] if r['util'] >= 5]
    allu = [r for g in used for r in by[g]]
    nb = max(len(busy), 1)
    cap = sum(1 for r in busy if r['reasons'] & 0x4) / nb
    hw = sum(1 for r in allu if r['reasons'] & (0x8 | 0x40 | 0x80))
    th = sum(1 for r in allu if r['reasons'] & (0x20 | 0x40))
    masks = sorted({r['reasons'] for r in allu})
    per_card_mean = [sum(r['pw'] for r in by[g]) / len(by[g]) for g in used]
    duty = len(busy) / max(len(allu), 1)
    # simultaneous busy cards per timestamp
    sim = defaultdict(int)
    for g in used:
        for r in by[g]:
            sim[r['rnd']] += (r['util'] >= 50)
    util_mean = sum(r['util'] for r in allu) / len(allu)
    simv = list(sim.values())
    return dict(n_used=len(used), duty=duty, pw_all=sum(r['pw'] for r in allu) / len(allu),
                pw_busy=sum(r['pw'] for r in busy) / nb, pw_peak=max(r['pw'] for r in allu),
                card_mean_lo=min(per_card_mean), card_mean_hi=max(per_card_mean),
                sm_busy=sum(r['sm'] for r in busy) / nb, sm_min=min((r['sm'] for r in busy), default=0),
                cap=cap, hw=hw, th=th, temp=max(r['temp'] for r in allu), masks=masks,
                tot9=sum(tot_pw.values()) / max(len(tot_pw), 1),
                sim_mean=(sum(simv) / len(simv)) if simv else 0, sim_max=max(simv) if simv else 0, util=util_mean)

def main():
    for run in sys.argv[1:]:
        smi = os.path.join(run, 'smi.csv')
        if not os.path.exists(smi):
            continue
        rows = parse_smi(smi)
        for w in windows(run):
            s = stats(rows, w)
            if not s:
                continue
            jpt = s['tot9'] / w['ts'] if w['ts'] else 0
            print(f"{run.split('results/')[-1]:55s} {w['model'][:28]:28s} {w['kind']}{w['n_prompt'] or w['n_gen']:<5d} d{w['depth']:<6d} ub{w['ub']} "
                  f"cards{s['n_used']} {w['ts']:7.1f}t/s util{s['util']:4.0f}% P(all){s['pw_all']:5.1f} P(busy){s['pw_busy']:5.1f} "
                  f"peak{s['pw_peak']:5.1f} card{s['card_mean_lo']:4.1f}-{s['card_mean_hi']:4.1f} sm{s['sm_busy']:5.0f}(min{s['sm_min']}) "
                  f"cap{s['cap']*100:4.0f}% hw{s['hw']} th{s['th']} T{s['temp']} totGPU {s['tot9']:5.0f}W {jpt:5.2f}J/t "
                  f"sim{s['sim_mean']:.1f}/{s['sim_max']} masks{[hex(m) for m in s['masks']]}")

main()
