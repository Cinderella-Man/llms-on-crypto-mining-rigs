#!/usr/bin/env bash
# test_c778.sh: C778's (Q1366) checks on any Linux machine with bash, curl and python3; no GPU, no model, nothing
# timed. NEW and OLD are two copies of the files the deep lane's self-test needs (bench/deep_lane,
# bench/agent_check/mock_server.py, release/flash-next-v3/settings.env, scripts/ram_watchdog.sh): NEW with C778, OLD
# without it (the public v4 at 406a939).
#   NEW=<tree with C778> OLD=<tree before it> OUT=<results> nice -n 19 bash test_c778.sh
# 1. OLD copied to OUT/tree and its selftest.sh run in OUT/work, then moved to OUT/work-old (its output in OUT/old.out);
# 2. NEW copied to the same OUT/tree and its selftest.sh run in the same OUT/work, so that every path in their files is
#    the same, then moved to OUT/work-new (OUT/new.out): every case of OLD's, plus stop1, stop2 and stop0;
# 3. the cases both ran compared file by file (OUT/compare.txt): the same files in each case folder, and the same
#    content once the clock and the host's free memory are taken out (ISO times and HH:MM:SS, loads.tsv's started and
#    load_s, requests.csv's wall_s and time, the .jsonl files' time and wall_s, a server's load seconds, MemAvailable
#    and SwapFree).
# Exit 0 when both self-tests pass and nothing differs.
set -uo pipefail
: "${NEW:?NEW=<tree with C778>}" "${OLD:?OLD=<tree before C778>}" "${OUT:?OUT=<results folder>}"
NEW=$(realpath "$NEW"); OLD=$(realpath "$OLD"); OUT=$(realpath -m "$OUT")
mkdir -p "$OUT"
rm -rf "$OUT/work" "$OUT/work-old" "$OUT/work-new"
st=0
for v in old new; do
  t=OLD; [ "$v" = new ] && t=NEW
  rm -rf "$OUT/tree" && cp -a "${!t}" "$OUT/tree"
  t0=$(date +%s)
  bash "$OUT/tree/bench/deep_lane/selftest.sh" "$OUT/work" > "$OUT/$v.out" 2>&1
  rc=$?
  echo "$v: selftest.sh exited $rc after $(($(date +%s) - t0)) s" | tee -a "$OUT/$v.out"
  [ "$rc" = 0 ] || st=1
  mv "$OUT/work" "$OUT/work-$v"
done
python3 -I - "$OUT/work-old" "$OUT/work-new" > "$OUT/compare.txt" << 'EOF' || st=1
import csv, io, json, re, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
CLOCK = [(re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?([+-]\d\d:?\d\d|Z)?'), 'T'),
         (re.compile(r'(?m)^\d\d:\d\d:\d\d '), 'HH:MM:SS '),
         (re.compile(r'up after \d+ s'), 'up after N s'),
         (re.compile(r'(MemAvailable|SwapFree):\s+\d+ kB'), r'\1: N kB')]
DROP = {'loads.tsv': ('started', 'load_s'), 'requests.csv': ('wall_s', 'time')}
def norm(p):
    s = p.read_text(errors='replace')
    if p.name in DROP or (p.name.startswith('requests-') and p.suffix == '.csv'):
        tab = p.suffix == '.tsv'
        rows = list(csv.DictReader(io.StringIO(s), delimiter='\t' if tab else ',', quoting=csv.QUOTE_NONE if tab else csv.QUOTE_MINIMAL))
        drop = DROP.get(p.name, DROP['requests.csv'])
        return '\n'.join(json.dumps({k: v for k, v in r.items() if k not in drop}) for r in rows)
    if p.suffix == '.jsonl':
        return '\n'.join(json.dumps({k: v for k, v in json.loads(x).items() if k not in ('time', 'wall_s')}) for x in s.splitlines())
    for rx, to in CLOCK:
        s = rx.sub(to, s)
    return s
cases = sorted({p.name for p in old.iterdir() if p.is_dir()} & {p.name for p in new.iterdir() if p.is_dir()} - {'bin', 'text', 'models'})
consoles = sorted({p.name for p in old.glob('*.console')} & {p.name for p in new.glob('*.console')})
bad = 0
for c in cases:
    fo = {str(p.relative_to(old / c)) for p in (old / c).rglob('*') if p.is_file()}
    fn = {str(p.relative_to(new / c)) for p in (new / c).rglob('*') if p.is_file()}
    diff = sorted(f for f in fo & fn if norm(old / c / f) != norm(new / c / f))
    print(f'{c}: {len(fo)} files before, {len(fn)} after; only before: {sorted(fo - fn) or "none"}; only after: '
          f'{sorted(fn - fo) or "none"}; different content: {diff or "none"}')
    bad += bool(fo ^ fn or diff)
diff = [c for c in consoles if norm(old / c) != norm(new / c)]
print(f'consoles of the cases both ran ({", ".join(consoles)}): different: {diff or "none"}')
bad += bool(diff)
print('the same files and content in every case both ran' if not bad else f'{bad} differences')
sys.exit(1 if bad else 0)
EOF
cat "$OUT/compare.txt"
exit $st
