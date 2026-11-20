#!/usr/bin/env bash
# FN101 (Q1518): the cards' energy beside the agent check's speeds, for R220's four runs, R227's four runs and R209's
# stock and overclocked pairs. Standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-21_fn101_joules_token/run/tables.sh TABLES
# TABLES holds, per run (c10a r075a c10b r075b, c600a t1400a c600b t1400b, f5oc0 f5oc1, o5oc0 o5oc1), from the run's
# raw output:
#   rig_<run>/cards.csv        the rig's card log (scripts/card_log.py's: every card's power every 0.5 s)
#   proxy_<run>/proxy.jsonl    the relay proxy's request rows
#   rig/server-<run>.log       the rig's llama-server log (the draft's acceptance in the agent tables)
# R209's two Ornith runs shared one client folder, so o5oc1's card log and proxy rows hold o5oc0's too: the window and
# the arm pick each run's own. Each run's turns.csv and attempts.csv are the published ones (agent/<run>/ in R220's,
# R227's and R209's folders). Writes this folder's tables and check.txt; the table folders go to TABLES/tab_<table>.
set -euo pipefail
T=$(cd "${1:?usage: tables.sh TABLES}" && pwd)
D=docs/data/2026-11-21_fn101_joules_token
B=bench/agent_check
py() { nice -n 19 python3 -I "$@"; }
declare -A SRC=(
  [c10a]=2026-11-20_r220_flash_ratio075 [r075a]=2026-11-20_r220_flash_ratio075
  [c10b]=2026-11-20_r220_flash_ratio075 [r075b]=2026-11-20_r220_flash_ratio075
  [c600a]=2026-11-20_r227_flash_taper1400_agent [t1400a]=2026-11-20_r227_flash_taper1400_agent
  [c600b]=2026-11-20_r227_flash_taper1400_agent [t1400b]=2026-11-20_r227_flash_taper1400_agent
  [f5oc0]=2026-11-19_r209_oc_v5 [f5oc1]=2026-11-19_r209_oc_v5 [o5oc0]=2026-11-19_r209_oc_v5 [o5oc1]=2026-11-19_r209_oc_v5)
declare -A ARM=([o5oc0]==o5oc0 [o5oc1]==o5oc1)  # the proxy rows' own arm, where a log holds more than one run
RUNS=(c10a r075a c10b r075b c600a t1400a c600b t1400b f5oc0 f5oc1 o5oc0 o5oc1)

# the energy of every run, the shift that best lines its requests up with the card power, and each card's share
spec=(); for r in "${RUNS[@]}"; do spec+=("$r=$T/rig_$r/cards.csv=$T/proxy_$r/proxy.jsonl${ARM[$r]:-}"); done
py $B/energy.py "${spec[@]}" --fit --tsv "$D/energy.tsv" --cards-tsv "$D/energy_cards.tsv" > "$D/energy.md"
# the split's watts with the proxy's times moved half a second either way
for s in -0.5 0.5; do py $B/energy.py "${spec[@]}" --shift "$s" --tsv "$T/energy_shift$s.tsv" > /dev/null; done
py - "$D/energy.tsv" "$T/energy_shift-0.5.tsv" "$T/energy_shift0.5.tsv" > "$D/shift.tsv" <<'PY'
import csv, sys
t = [{r["run"]: r for r in csv.DictReader(open(p), delimiter="\t")} for p in sys.argv[1:]]
print("run\tpart\tw_shift_-0.5\tw_shift_0\tw_shift_+0.5")
for run in t[0]:
    for part in ("generating", "prompt", "idle"):
        print(f"{run}\t{part}\t{t[1][run]['w_' + part]}\t{t[0][run]['w_' + part]}\t{t[2][run]['w_' + part]}")
PY

# agent_tables.py's folders: the published rows, the server logs, each run's card log and proxy rows
table() {  # NAME REF RUN...: TABLES/tab_NAME, agent_NAME.tsv
  local name=$1 f=$T/tab_$1
  shift
  rm -rf "$f"; mkdir -p "$f/res" "$f/rig"
  dirs=()
  for r in "$@"; do
    dirs+=("docs/data/${SRC[$r]}/agent/$r")
    cp "$T/rig/server-$r.log" "$f/rig/"
    ln -s "$T/rig_$r" "$f/rig_$r"; ln -s "$T/proxy_$r" "$f/proxy_$r"
  done
  py - "$f/res" "${dirs[@]}" <<'PY'
import csv, sys
out, dirs = sys.argv[1], sys.argv[2:]
for f in ("turns.csv", "attempts.csv"):
    rows = [r for d in dirs for r in csv.DictReader(open(f"{d}/{f}"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{out}/{f}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
PY
  py $B/agent_tables.py "$f" "$@" > "$f/agent_tables.out"
  cp "$f/agent.tsv" "$D/agent_$name.tsv"
}
table r220 c10a r075a c10b r075b
table r227 c600a t1400a c600b t1400b
table r209_flash f5oc0 f5oc1
table r209_ornith o5oc0 o5oc1
# the class of each change against its control, two runs an arm (R220: the draft ratio 0.75; R227: the prompt taper
# at 1,400 tokens)
py $B/class_arms.py "$D/agent_r220.tsv" r075=r075a,r075b c10=c10a,c10b > "$D/class_r220.tsv"
py $B/class_arms.py "$D/agent_r227.tsv" t1400=t1400a,t1400b c600=c600a,c600b > "$D/class_r227.tsv"
# R209: one run an arm, so the pairs are set beside the spread of two runs of the same settings (R220's and R227's
# controls) instead of being classed
py - "$D/agent_r209_flash.tsv" "$D/agent_r209_ornith.tsv" "$D/agent_r220.tsv" "$D/agent_r227.tsv" \
  > "$D/r209_pairs.tsv" <<'PY'
import csv, sys
rd = lambda p: {r["arm"]: r for r in csv.DictReader(open(p), delimiter="\t")}
fl, orn, r220, r227 = map(rd, sys.argv[1:])
cols = ["gen_tok_s", "equal_work_min_per_task", "w_generating", "w_prompt", "w_idle", "equal_work_j_per_task",
        "joules_per_task", "joules_per_token"]
print("measure\tmodel\tstock\toverclocked\tdiff\tdiff_pct\tflash_control_pairs_pct")
for c in cols:
    ctl = []
    for t, a, b in ((r220, "c10a", "c10b"), (r227, "c600a", "c600b")):
        x, y = float(t[a][c]), float(t[b][c])
        ctl.append(f"{(y - x) / x * 100:+.2f} %")
    for model, t, s, o in (("Flash-Next", fl, "f5oc0", "f5oc1"), ("Ornith", orn, "o5oc0", "o5oc1")):
        x, y = float(t[s][c]), float(t[o][c])
        print(f"{c}\t{model}\t{t[s][c]}\t{t[o][c]}\t{y - x:+.2f}\t{(y - x) / x * 100:+.2f} %\t{' '.join(ctl)}")
PY

# the checks
py - "$D" "$T" > "$D/check.txt" <<'PY'
import csv, sys
D, T = sys.argv[1], sys.argv[2]
rd = lambda p: list(csv.DictReader(open(p), delimiter="\t"))
pub = {"r220": "docs/data/2026-11-20_r220_flash_ratio075/agent.tsv",
       "r227": "docs/data/2026-11-20_r227_flash_taper1400_agent/agent.tsv",
       "r209_flash": "docs/data/2026-11-19_r209_oc_v5/flash_agent.tsv",
       "r209_ornith": "docs/data/2026-11-19_r209_oc_v5/agent.tsv"}
first = ["arm", "passed", "turns", "prompt_tok_s", "gen_tok_s", "ttft_median_s", "ttft_p90_s", "ttft_turn1_median_s",
         "acceptance", "tokens_per_step", "minutes_all", "minutes_per_task_median"]
energy = {r["run"]: r for r in rd(f"{D}/energy.tsv")}
same = {"energy_kj": "energy_kj", "joules_per_task": "joules_per_task", "joules_per_token": "joules_per_token",
        "w_generating": "w_generating", "w_prompt": "w_prompt", "w_idle": "w_idle"}
fail = 0
for name, path in pub.items():
    new = {r["arm"]: r for r in rd(f"{D}/agent_{name}.tsv")}
    old = {r["arm"]: r for r in rd(path)}
    for arm, r in new.items():
        a = [r[c] for c in first]
        b = [old[arm][c] for c in first] if arm in old else None
        ok = a == b
        fail += not ok
        print(f"{'same' if ok else 'DIFFERS'}: {name} {arm}: the first 12 columns are the published {path.split('/')[2]}"
              f"/{path.split('/')[3]} row")
        e = [r[c] for c in same] == [energy[arm][c] for c in same.values()]
        t = r["turns"] == energy[arm]["requests"]
        fail += not (e and t)
        print(f"{'same' if e else 'DIFFERS'}: {name} {arm}: the energy columns are energy.py's; "
              f"{'same' if t else 'DIFFERS'}: the turns are its requests ({r['turns']})")
print(f"{'all checks passed' if not fail else f'{fail} checks failed'}")
sys.exit(1 if fail else 0)
PY
cat "$D/check.txt"
