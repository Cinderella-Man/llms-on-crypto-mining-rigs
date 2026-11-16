"""merge_rows.py V3DIR COLLECTED OUTDIR ARM [COLLECTED2 ARM2 ...]: the table folder for agent_tables.py. res/turns.csv
and res/attempts.csv hold v3's Q5 run (docs/data/2026-11-05_release_regression/agent/flint-pq5-v3base, its arm rel
renamed v3q5) and each ARM's rows from its collected run (COLLECTED/res/turns.csv rows with arm == ARM, attempts rows
whose tag starts with ARM/), in the order given. R170a ran it with one pair (col q5f4t), R170b with two (R170a's
collected run with q5f4t, its own with q5f3). Also writes OUTDIR/agent/<ARM>/{turns,attempts}.csv for each ARM and
OUTDIR/agent/v3q5/turns.csv for prompt_by_size.py."""
import csv, os, sys
v3, out = sys.argv[1], sys.argv[3]
pairs = [(sys.argv[2], sys.argv[4])] + list(zip(sys.argv[5::2], sys.argv[6::2]))
assert len(sys.argv) % 2 == 1, 'COLLECTED and ARM come in pairs'
os.makedirs(os.path.join(out, 'res'), exist_ok=True)
for a in [arm for _, arm in pairs] + ['v3q5']:
    os.makedirs(os.path.join(out, 'agent', a), exist_ok=True)
def rd(p): return list(csv.DictReader(open(p, newline='')))
def wr(p, rows, fields):
    w = csv.DictWriter(open(p, 'w', newline=''), fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
vt, va = rd(os.path.join(v3, 'turns.csv')), rd(os.path.join(v3, 'attempts.csv'))
for r in vt: r['arm'] = 'v3q5'
for r in va:
    r['file'] = 'v3q5'; r['tag'] = 'v3q5/' + r['tag'].split('/', 1)[1]
allt, alla, tf, af, counts = list(vt), list(va), None, None, ['v3q5 turns %d attempts %d' % (len(vt), len(va))]
for col, arm in pairs:
    ct = [r for r in rd(os.path.join(col, 'res', 'turns.csv')) if r['arm'] == arm]
    ca = [r for r in rd(os.path.join(col, 'res', 'attempts.csv')) if r['tag'].startswith(arm + '/')]
    t, a = list(ct[0].keys()), list(ca[0].keys())
    assert set(t) >= set(vt[0].keys()) and set(a) >= set(va[0].keys()), 'columns differ'
    assert tf is None or (t == tf and a == af), 'collected runs have different columns'
    tf, af = t, a
    allt += ct; alla += ca
    wr(os.path.join(out, 'agent', arm, 'turns.csv'), ct, tf)
    wr(os.path.join(out, 'agent', arm, 'attempts.csv'), ca, af)
    counts.append('%s turns %d attempts %d' % (arm, len(ct), len(ca)))
wr(os.path.join(out, 'res', 'turns.csv'), allt, tf)
wr(os.path.join(out, 'res', 'attempts.csv'), alla, af)
wr(os.path.join(out, 'agent', 'v3q5', 'turns.csv'), vt, list(vt[0].keys()))
print(' | '.join(counts))
