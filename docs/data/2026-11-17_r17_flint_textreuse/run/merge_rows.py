"""merge_rows.py OUTDIR NAME=DIR[:ARM] ...: the table folder for agent_tables.py, prompt_by_size.py and report.py work.

Each DIR holds an arm's turns.csv and attempts.csv: a published per-arm folder (docs/data/<run>/agent/<arm>/), or a
collected client folder (its res/turns.csv and res/attempts.csv, which hold every arm of a shared client folder). ARM
(default NAME) picks the rows: turns.csv rows with arm == ARM, attempts.csv rows whose tag starts with ARM/. They are
renamed NAME (turns: arm; attempts: file and the tag's first part), so v3.0's reference run, whose arm is rel, joins as
v3p065. Writes OUTDIR/res/{turns,attempts}.csv with every NAME in the order given and OUTDIR/agent/<NAME>/{turns,
attempts}.csv for each; tables with more columns than others (C852's) keep them, the others' cells left empty."""
import csv, os, sys
out = sys.argv[1]
def rd(p): return list(csv.DictReader(open(p, newline='')))
def wr(p, rows, fields):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    w = csv.DictWriter(open(p, 'w', newline=''), fieldnames=fields, restval='')
    w.writeheader()
    for r in rows: w.writerow(r)
def union(a, b): return a + [k for k in b if k not in a]
allt, alla, tf, af, counts = [], [], [], [], []
for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1)
    d, _, arm = rest.partition(':')
    arm = arm or name
    base = os.path.join(d, 'res') if os.path.exists(os.path.join(d, 'res', 'turns.csv')) else d
    ct = [r for r in rd(os.path.join(base, 'turns.csv')) if r['arm'] == arm]
    ca = [r for r in rd(os.path.join(base, 'attempts.csv')) if r['tag'].split('/', 1)[0] == arm]
    assert ct and ca, 'no rows for %s in %s' % (arm, base)
    for r in ct: r['arm'] = name
    for r in ca: r['file'] = name; r['tag'] = name + '/' + r['tag'].split('/', 1)[1]
    t, a = list(ct[0].keys()), list(ca[0].keys())
    tf, af = union(tf, t), union(af, a)
    wr(os.path.join(out, 'agent', name, 'turns.csv'), ct, t)
    wr(os.path.join(out, 'agent', name, 'attempts.csv'), ca, a)
    allt += ct; alla += ca
    counts.append('%s turns %d attempts %d' % (name, len(ct), len(ca)))
wr(os.path.join(out, 'res', 'turns.csv'), allt, tf)
wr(os.path.join(out, 'res', 'attempts.csv'), alla, af)
print(' | '.join(counts))
