"""merge_rows.py V3DIR COLLECTED OUTDIR ARM: the table folder for agent_tables.py. res/turns.csv and res/attempts.csv
hold v3's Q5 run (docs/data/2026-11-05_release_regression/agent/flint-pq5-v3base, its arm rel renamed v3q5) and ARM's
rows from the collected run (COLLECTED/res/turns.csv rows with arm == ARM, attempts rows whose tag starts with ARM/).
Also writes OUTDIR/agent/<ARM>/{turns,attempts}.csv and OUTDIR/agent/v3q5/turns.csv for prompt_by_size.py."""
import csv, os, sys
v3, col, out, arm = sys.argv[1:5]
os.makedirs(os.path.join(out, 'res'), exist_ok=True)
for a in (arm, 'v3q5'):
    os.makedirs(os.path.join(out, 'agent', a), exist_ok=True)
def rd(p): return list(csv.DictReader(open(p, newline='')))
def wr(p, rows, fields):
    w = csv.DictWriter(open(p, 'w', newline=''), fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
vt, va = rd(os.path.join(v3, 'turns.csv')), rd(os.path.join(v3, 'attempts.csv'))
for r in vt: r['arm'] = 'v3q5'
for r in va:
    r['file'] = 'v3q5'; r['tag'] = 'v3q5/' + r['tag'].split('/', 1)[1]
ct = [r for r in rd(os.path.join(col, 'res', 'turns.csv')) if r['arm'] == arm]
ca = [r for r in rd(os.path.join(col, 'res', 'attempts.csv')) if r['tag'].startswith(arm + '/')]
tf, af = list(ct[0].keys()), list(ca[0].keys())
assert set(tf) >= set(vt[0].keys()) and set(af) >= set(va[0].keys()), 'columns differ'
wr(os.path.join(out, 'res', 'turns.csv'), vt + ct, tf)
wr(os.path.join(out, 'res', 'attempts.csv'), va + ca, af)
wr(os.path.join(out, 'agent', arm, 'turns.csv'), ct, tf)
wr(os.path.join(out, 'agent', arm, 'attempts.csv'), ca, af)
wr(os.path.join(out, 'agent', 'v3q5', 'turns.csv'), vt, list(vt[0].keys()))
print('v3q5 turns', len(vt), 'attempts', len(va), '|', arm, 'turns', len(ct), 'attempts', len(ca))
