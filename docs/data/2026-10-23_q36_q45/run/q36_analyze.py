# Q36 (2026-10-23): per class and 8-layer group, the KLD change against S4 on the same GPU set, the decode time the
# group saves (est.: bytes saved at the efficiencies of scripts/token_cost.py, 173 GB/s), and the KLD cost per ms
# saved; then the Q44 probes. Reads this folder's scores.tsv.
#   q36_analyze.py docs/data/2026-10-23_q36_q45/scores.tsv
import csv, sys
S = {}
for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
    S[r['file']] = (float(r['kld']), float(r['same_top_pct']), float(r['ppl_ratio']))
q41, q80, q50 = 20/32/(0.38*173e9)*1e3, 34/32/(0.41*173e9)*1e3, 22/32/(0.30*173e9)*1e3
per = {'a': 4096*2048*(q80-q41), 'd': 6*512*2048*(q80-q41), 'u': 2*6*512*2048*(q50-q41)}
groups = {1: range(0, 9), 2: range(9, 17), 3: range(17, 25), 4: range(25, 33), 5: range(33, 41), 6: range(41, 49)}
ok = {'a': lambda l: True, 'd': lambda l: 2 <= l <= 45, 'u': lambda l: l >= 2}
for c in 'adu':
    b = S.get(f'flint-mixS4_nr_q36{c}')
    print(f'class {c}: lane baseline S4', b)
    if not b: continue
    for g, ls in groups.items():
        n = sum(1 for l in ls if ok[c](l)); ms = n * per[c]
        s = S.get(f'q36-S4-{c}{g}_nr')
        if s:
            dk = s[0] - b[0]
            print(f'  {c}{g} layers {ls.start}-{ls.stop-1}: saves {ms:.3f} ms  dKLD {dk:+.4f}  dtop {s[1]-b[1]:+.2f}  ppl {s[2]:.4f}  KLD/ms {dk/ms:.4f}')
for k, v in S.items():
    if 'hmk' in k: print(k, v)
