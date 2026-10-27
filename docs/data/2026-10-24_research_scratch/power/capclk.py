#!/usr/bin/env python3
# Research round 16 (power and tuning, 2026-10-24): clock-event bitmask counts, SM clock and power in capped and
# uncapped busy samples (>= 90 % utilization), memory clock under the cap, samples above 75 W, uncapped clock by
# temperature.   usage: capclk.py RUN_DIR/smi.csv [...]   (the smi.csv layout of busy_samples.py)
import sys
from collections import defaultdict, Counter
masks=Counter(); 
capped=defaultdict(list); uncapped=defaultdict(list); memc=Counter(); peak=[]
bytemp=defaultdict(list)
for path in sys.argv[1:]:
    with open(path) as f:
        f.readline()
        for line in f:
            p=[x.strip() for x in line.split(',')]
            if len(p)<10: continue
            try:
                g=int(p[1]); ps=p[2]; sm=int(p[3].split()[0]); mem=int(p[4].split()[0]); pw=float(p[5].split()[0]); t=int(p[6]); u=int(p[7].split()[0]); rs=int(p[9],16)
            except Exception: continue
            masks[rs]+=1
            if u>=90:
                (capped if rs&4 else uncapped)[g].append((sm,pw,t,mem))
                memc[(rs&4, mem)]+=1
                if not rs&4: bytemp[(t//5)*5].append(sm)
            if pw>75: peak.append(pw)
print('mask counts', {hex(k):v for k,v in masks.items()})
for name,d in (('capped',capped),('uncapped',uncapped)):
    allv=[x for g in d for x in d[g]]
    if not allv: continue
    print(name, len(allv), 'sm %.0f'%(sum(x[0] for x in allv)/len(allv)), 'pw %.1f'%(sum(x[1] for x in allv)/len(allv)), 'smmin', min(x[0] for x in allv))
print('mem clocks in util>=90 samples (cap,mem):', dict(memc))
print('samples >75 W:', len(peak), 'max %.1f'%max(peak), 'p99 of those', sorted(peak)[int(len(peak)*0.99)-1] if peak else None)
print('uncapped util>=90 SM by temperature bucket:', {k:(len(v), round(sum(v)/len(v))) for k,v in sorted(bytemp.items())})
