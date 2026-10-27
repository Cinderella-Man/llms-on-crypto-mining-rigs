#!/usr/bin/env python3
# Research round 16 (power and tuning, 2026-10-24): idle power by P-state, and per run the samples at >= 80 %
# utilization (power, SM clock, power-cap share, cards busy at once), over nvidia-smi traces.
#   usage: busy_samples.py RUN_DIR/smi.csv [...]   (columns: timestamp,index,pstate,sm,mem,power,temp,util,mem.used,reasons)
import sys, os
from collections import defaultdict
def rows(path):
    with open(path) as f:
        f.readline()
        for line in f:
            p=[x.strip() for x in line.split(',')]
            if len(p)<10: continue
            try:
                yield dict(gpu=int(p[1]),ps=p[2],sm=int(p[3].split()[0]),mem=int(p[4].split()[0]),pw=float(p[5].split()[0]),
                           temp=int(p[6]),util=int(p[7].split()[0]),used=int(p[8].split()[0]),rs=int(p[9],16))
            except Exception: pass
agg=defaultdict(lambda: defaultdict(list))
for path in sys.argv[1:]:
    run=path.split('results/')[-1].rsplit('/smi.csv',1)[0]
    rr=list(rows(path))
    # rounds
    rnd=-1; R=defaultdict(list)
    for r in rr:
        if r['gpu']==0: rnd+=1
        R[rnd].append(r)
    for k,rs in R.items():
        nb=sum(1 for r in rs if r['util']>=80)
        for r in rs:
            if r['ps']=='P8' and r['util']==0: agg['P8idle'][run].append(r)
            if r['ps']=='P0' and r['util']==0 and r['used']>200: agg['P0idle_loaded'][run].append(r)
            if r['util']>=80: agg['busy80'][run].append((r,nb))
def summ(lst):
    n=len(lst); 
    return n, sum(r['pw'] for r in lst)/n, sum(r['sm'] for r in lst)/n, sum(1 for r in lst if r['rs']&4)/n, max(r['temp'] for r in lst), sorted({hex(r['rs']) for r in lst})
for key in ('P8idle','P0idle_loaded'):
    allr=[r for run in agg[key] for r in agg[key][run]]
    n,pw,sm,cap,t,m=summ(allr)
    pws=sorted(r['pw'] for r in allr)
    print(key, n, f"mean {pw:.1f} W p10 {pws[n//10]:.1f} p50 {pws[n//2]:.1f} p90 {pws[9*n//10]:.1f} sm {sm:.0f} masks {m}")
    sms=defaultdict(int)
    for r in allr: sms[r['sm']]+=1
    print('  sm clocks', sorted(sms.items(), key=lambda x:-x[1])[:6])
print('busy80 by run (util>=80% samples): n, mean W, mean SM, cap share, maxT, n-simultaneous-busy mean')
for run in sorted(agg['busy80']):
    lst=[r for r,nb in agg['busy80'][run]]
    if len(lst)<5: continue
    n,pw,sm,cap,t,m=summ(lst)
    nbm=sum(nb for r,nb in agg['busy80'][run])/n
    print(f"  {run[:70]:70s} n{n:5d} {pw:5.1f}W sm{sm:5.0f} cap{cap*100:4.0f}% T{t} sim{nbm:.2f} {m}")
