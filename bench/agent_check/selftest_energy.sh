#!/usr/bin/env bash
# selftest_energy.sh OUT: the energy columns (FN101, Q1518) on fixed inputs: energy.py, report.py's table and compare
# with --cards, agent_tables.py's energy columns and class_arms.py's energy measures. Python 3 only (no sandbox, no
# server, no packages); selftest.sh runs it too. Exits non-zero if a check fails.
#
# The run: two cards logged every 0.5 s from t = 1000 to 1060 s; card 0 draws 10 W throughout, card 1 50 W at the
# samples from 1010 to 1030 s and 10 W at the others (so it ramps over 1009.5-1010 and 1030-1030.5). Arm e sends four
# requests: task t turn 1 at 1002 s (first token 1006, end 1010, 40 tokens), turn 2 at 1010 s (1012, 1030, 160), task
# u at 1034 s (1035, 1040, 50), task w at 1043 s (1044, 1048, 50). By hand:
#   window 1002-1048 s = 46 s, 1,740 J (card 0 460 J, card 1 1,280 J); 3 tasks, 300 tokens: 580 J a task, 5.80 a token;
#   generating 31 s, 1,350 J (43.5 W); reading the prompt 8 s, 240 J (30.0 W); idle 7 s (1030-1034, 1040-1043), 150 J
#   (21.4 W);
#   without task u: its span 1034-1040 s cut from the window: 40 s, 1,620 J; 2 tasks, 250 tokens: 810 J a task, 6.48 a
#   token; generating 26 s, 1,250 J (48.1 W); prompt 7 s, 220 J (31.4 W); idle 7 s, 150 J (21.4 W);
#   with the samples between 1036 and 1042.5 s missing (a 6.5 s gap): 39.5 s covered, 1,610 J.
set -euo pipefail
OUT=${1:?usage: selftest_energy.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"
PY=(python3 -I)
"${PY[@]}" - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
head = "epoch,time,took_ms," + ",".join(f"g{i}_{f}" for i in range(2) for f in
                                        ("pstate", "sm_mhz", "mem_mhz", "w", "c", "used_mib", "util", "reasons"))
def row(t, w0, w1):
    card = lambda w: f"P0,1500,4000,{w:.2f},30,1000,50,0x0"
    return f"{t:.3f},2026-01-01 00:00:00.000,30.0,{card(w0)},{card(w1)}"
ts = [1000 + k / 2 for k in range(121)]
w1 = lambda t: 50.0 if 1010 <= t <= 1030 else 10.0
plain = [head] + [row(t, 10.0, w1(t)) for t in ts]
open(f"{out}/cards.csv", "w").write("# start\n" + "\n".join(plain) + "\n# stop\n")
# the same log as a resumed run appends it: a second header and events after 1030 s
cut = plain.index(row(1030.0, 10.0, 50.0)) + 1
open(f"{out}/cards_appended.csv", "w").write("\n".join(plain[:cut]) + "\n# stop\n" + head + "\n# start\n"
                                             + "\n".join(plain[cut:]) + "\n")
gap = [head] + [row(t, 10.0, w1(t)) for t in ts if not 1036 < t < 1042.5]
open(f"{out}/cards_gap.csv", "w").write("\n".join(gap) + "\n")
reqs = [("t", 1, 1002.0, 4.0, 8.0, 40), ("t", 2, 1010.0, 2.0, 20.0, 160), ("u", 1, 1034.0, 1.0, 6.0, 50),
        ("w", 1, 1043.0, 1.0, 5.0, 50)]
def rows(arm, shift=0.0):
    return [dict(id=f"{arm}-{k}", tag=f"{arm}/{task}/a1/t1/{turn}", t_wall=t0 + shift, t_first_token=tf, t_end=te,
                 status=200, prompt_n=100, cache_n=0, prompt_ms=tf * 1000, predicted_n=gen,
                 predicted_ms=(te - tf) * 1000, finish_reason="stop", n_tool_calls=0)
            for k, (task, turn, t0, tf, te, gen) in enumerate(reqs)]
with open(f"{out}/proxy.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows("e") + [dict(id="x", tag="other", t_wall=1.0)])
# for the fit: every request draws 60 W, the gaps 20 W (one card), and the proxy's clock runs 0.7 s behind the rig's
spans = [(t0 + 0.7, t0 + 0.7 + te) for _, _, t0, _, te, _ in reqs]
fit = ["epoch,time,took_ms,g0_w"] + [f"{t:.3f},x,1,{60 if any(a <= t <= b for a, b in spans) else 20}" for t in ts]
open(f"{out}/cards_fit.csv", "w").write("\n".join(fit) + "\n")
# report.py: the attempts of arm e with their results, and an arm g without proxy rows
for arm in ("e",):
    for task, wall in (("t", 30.0), ("u", 8.0), ("w", 6.0)):
        r = dict(arm=arm, task_id=task, track="python", attempt=1, try_n=1, tag=f"{arm}/{task}/a1/t1", status="scored",
                 success=1, hidden_pass=1, tests_untouched=1, n_pass=1, n_fail=0, n_total=1, turns=2 if task == "t" else 1,
                 tool_calls=1, tool_errors=0, malformed_calls=0, unknown_tool=0, invalid_args=0, markup_in_text=0,
                 call_in_think=0, length_stops=0, stop_reasons={"stop": 2 if task == "t" else 1}, runaway="",
                 compactions=0, error_type="", wall_s=wall, usage_input=0, usage_output=0, usage_cache_read=0,
                 started=f"2026-01-01T00:00:{ {'t': 10, 'u': 20, 'w': 30}[task]}", finished="2026-01-01T00:01:00")
        import os
        os.makedirs(f"{out}/res/{arm}/{task}/a1", exist_ok=True)
        open(f"{out}/res/{arm}/{task}/a1/result.json", "w").write(json.dumps(r))
# agent_tables.py: run e with its card log and proxy rows, run f with neither; the rows as turns.csv and attempts.csv
os.makedirs(f"{out}/tab/res", exist_ok=True)
os.makedirs(f"{out}/tab/rig_e", exist_ok=True)
os.makedirs(f"{out}/tab/proxy_e", exist_ok=True)
open(f"{out}/tab/rig_e/cards.csv", "w").write(open(f"{out}/cards.csv").read())
open(f"{out}/tab/proxy_e/proxy.jsonl", "w").write(open(f"{out}/proxy.jsonl").read())
turns = [("t", 1, 400, 4000, 40, 4000), ("t", 2, 100, 2000, 160, 18000), ("u", 1, 50, 1000, 50, 5000),
         ("w", 1, 50, 1000, 50, 4000)]
with open(f"{out}/tab/res/turns.csv", "w") as f:
    f.write("arm,task_id,attempt,turn,ttft_s,t_end_s,prompt_n,cache_n,prompt_ms,predicted_n,predicted_ms,finish_reason,"
            "n_tool_calls\n")
    for arm in ("e", "f"):
        f.writelines(f"{arm},{t},1,{k},1.0,2.0,{pn},0,{pms},{gn},{gms},stop,0\n" for t, k, pn, pms, gn, gms in turns)
with open(f"{out}/tab/res/attempts.csv", "w") as f:
    f.write("task_id,file,attempt,wall_s,pass\n")
    f.writelines(f"{t},{arm},1,{w},1\n" for arm in ("e", "f") for t, w in (("t", 30), ("u", 8), ("w", 6)))
EOF
E=$HERE/energy.py
"${PY[@]}" "$E" e="$OUT/cards.csv"="$OUT/proxy.jsonl"=e --tsv "$OUT/e.tsv" --cards-tsv "$OUT/e_cards.tsv" > "$OUT/e.md"
"${PY[@]}" "$E" e="$OUT/cards.csv"="$OUT/proxy.jsonl" --skip u --tsv "$OUT/e_skip.tsv" > /dev/null
"${PY[@]}" "$E" e="$OUT/cards_appended.csv"="$OUT/proxy.jsonl"=e --tsv "$OUT/e_appended.tsv" > /dev/null
"${PY[@]}" "$E" e="$OUT/cards.csv,$OUT/cards_gap.csv"="$OUT/proxy.jsonl" --tsv "$OUT/e_two.tsv" > /dev/null
"${PY[@]}" "$E" e="$OUT/cards_gap.csv"="$OUT/proxy.jsonl" --tsv "$OUT/e_gap.tsv" > /dev/null
"${PY[@]}" "$E" e="$OUT/cards_fit.csv"="$OUT/proxy.jsonl" --fit > "$OUT/e_fit.md"
"${PY[@]}" "$E" e="$OUT/cards.csv"="$OUT/proxy.jsonl"=nosucharm > "$OUT/e_none.md"
R=$HERE/report.py
(cd "$OUT" && "${PY[@]}" "$R" compare res --log proxy.jsonl --turns-csv turns_plain.csv > compare_plain.md \
  && "${PY[@]}" "$R" compare res --log proxy.jsonl --turns-csv turns_cards.csv --cards cards.csv > compare_cards.md \
  && "${PY[@]}" "$R" table res --log proxy.jsonl --csv attempts_plain.csv > table_plain.md \
  && "${PY[@]}" "$R" table res --log proxy.jsonl --csv attempts_cards.csv --cards cards.csv > table_cards.md)
"${PY[@]}" "$HERE/agent_tables.py" "$OUT/tab" e f > /dev/null
mkdir -p "$OUT/tab_plain"; cp -r "$OUT/tab/res" "$OUT/tab_plain/"
"${PY[@]}" "$HERE/agent_tables.py" "$OUT/tab_plain" e f > /dev/null
cat > "$OUT/runs.tsv" <<'TSV'
arm	equal_work_min_per_task	gen_tok_s	prompt_tok_s	ttft_median_s	ttft_p90_s	ttft_turn1_median_s	prompt_read_per_task	reread_per_task	reread_s_per_task	ttft_max_s	energy_kj	joules_per_task	joules_per_token	w_generating	w_prompt	w_idle	equal_work_j_per_task	equal_work_j_vs_ref
a1	1.80	47	200	1.7	6	3.2	2900	0	0	12	300	25000	16.0	356	370	325	24000	+0.0 %
a2	1.82	48	210	1.8	6.5	3.3	3000	0	0	13	500	41000	15.0	357	371	326	24300	+1.2 %
b1	1.81	47	200	1.7	6	3.2	2950	0	0	12	320	26000	17.0	330	354	303	25500	+6.2 %
b2	1.83	48	210	1.8	6.5	3.3	2980	0	0	13	480	40000	16.5	331	355	304	25600	+6.7 %
n1	1.81	47	200	1.7	6	3.2	2950	0	0	12	nan	nan	nan	nan	nan	nan	nan	nan
TSV
"${PY[@]}" "$HERE/class_arms.py" "$OUT/runs.tsv" a=a1,a2 b=b1,b2 > "$OUT/class_ab.tsv"
"${PY[@]}" "$HERE/class_arms.py" "$OUT/runs.tsv" a=a1,a2 n=n1 > "$OUT/class_an.tsv"
cut -f1-11 "$OUT/runs.tsv" > "$OUT/runs_work.tsv"
"${PY[@]}" "$HERE/class_arms.py" "$OUT/runs_work.tsv" a=a1,a2 b=b1,b2 > "$OUT/class_work.tsv"
"${PY[@]}" - "$OUT" <<'EOF'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
fails = []
def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)
tsv = lambda p: list(csv.DictReader((out / p).open(), delimiter="\t"))
row = lambda p: tsv(p)[0]
want = dict(run="e", tasks="3", requests="4", generated="300", window_s="46.0", covered_s="46.0", energy_kj="1.74",
            joules_per_task="580", joules_per_token="5.80", w_mean="37.8", generating_s="31.0", w_generating="43.5",
            prompt_s="8.0", w_prompt="30.0", idle_s="7.0", w_idle="21.4")
check(row("e.tsv") == want, f"energy.py: window, energy, joules a task and a token, the watts of each part ({row('e.tsv')})")
cards = tsv("e_cards.tsv")
check([(r["card"], r["joules"], r["w_generating"], r["w_idle"]) for r in cards]
      == [("0", "460", "10.00", "10.00"), ("1", "1280", "33.55", "11.43")],
      f"energy.py: each card's joules and watts by its index ({cards})")
want_skip = dict(want, tasks="2", requests="3", generated="250", window_s="40.0", covered_s="40.0", energy_kj="1.62",
                 joules_per_task="810", joules_per_token="6.48", w_mean="40.5", generating_s="26.0",
                 w_generating="48.1", prompt_s="7.0", w_prompt="31.4", idle_s="7.0", w_idle="21.4")
check(row("e_skip.tsv") == want_skip, f"energy.py --skip: the left-out task's span is cut from the window "
      f"({row('e_skip.tsv')})")
check(row("e_appended.tsv") == want, "energy.py: an appended log (a second header, events) reads as one")
check(row("e_two.tsv") == want, "energy.py: two logs of the same samples read as one (the second's samples repeat)")
g = row("e_gap.tsv")
check(g["covered_s"] == "39.5" and g["energy_kj"] == "1.61" and g["window_s"] == "46.0",
      f"energy.py: a 6.5 s gap in the log adds nothing and is not covered ({g['covered_s']}, {g['energy_kj']})")
fit = [l for l in (out / "e_fit.md").read_text().splitlines() if l.startswith("| e |")][0].split(" | ")[-1]
d = float(fit.split()[0])
check(abs(d - 0.7) <= 0.1, f"energy.py --fit: the proxy's clock 0.7 s behind the card log is found ({fit})")
check("no requests or no card samples" in (out / "e_none.md").read_text(), "energy.py: a run without requests says so")
# report.py
plain, cards = ((out / f"compare_{k}.md").read_text().splitlines() for k in ("plain", "cards"))
ecells = " 1.7 (580) | 5.80 | 43.5 / 30.0 / 21.4 |"
arm_c = [l for l in cards if l.startswith("| e | 3 / 3")]
check(len(arm_c) == 1 and arm_c[0].endswith("|" + ecells),
      f"compare --cards: the arm's row ends with energy, joules a task, a token and the three watts ({arm_c})")
strip = [l[:-len(ecells)] if l.endswith(ecells) else l for l in cards]
strip = [l.replace(" card energy kJ (J a task) | J a generated token | W generating / reading the prompt / idle "
                   "(all cards) |", "").replace("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
                                                "|---|---|---|---|---|---|---|---|---|---|---|---|")
         for l in strip]
note = [i for i, l in enumerate(strip) if l.startswith("Card energy (FN101)")]
check(len(note) == 1 and strip[:note[0] - 1] + strip[note[0] + 1:] == [l.replace("turns_plain", "turns_cards")
                                                                       for l in plain],
      "compare --cards: every other line as without the card log, plus one line on how the energy is counted")
tp, tc = ((out / f"table_{k}.md").read_text().splitlines() for k in ("plain", "cards"))
row_c = [l for l in tc if l.startswith("| e | 3 |")]
check(len(row_c) == 1 and row_c[0].endswith("|" + ecells) and len(tc) == len(tp) + 2
      and [l for l in tp if l.startswith("| e | 3 |")][0] + ecells == row_c[0],
      f"table --cards: the energy cells after the work cells ({row_c})")
check((out / "turns_plain.csv").read_text() == (out / "turns_cards.csv").read_text()
      and (out / "attempts_plain.csv").read_text() == (out / "attempts_cards.csv").read_text(),
      "compare and table --cards: turns.csv and attempts.csv as without the card log")
# agent_tables.py
ag = {r["arm"]: r for r in tsv("tab/agent.tsv")}
head = (out / "tab/agent.tsv").read_text().splitlines()[0].split("\t")
ecols = ["energy_kj", "joules_per_task", "joules_per_token", "w_generating", "w_prompt", "w_idle",
         "equal_work_j_per_task", "equal_work_j_vs_ref"]
check(head[19:] == ecols and [ag["e"][c] for c in ecols] == ["1.74", "580", "5.80", "43.5", "30.0", "21.4", "566",
                                                             "+0.0 %"]
      and [ag["f"][c] for c in ecols] == ["nan"] * 8,
      f"agent_tables: the energy columns after the work columns; the joules at equal work (5 s outside the server at "
      f"the idle watts, 8 s of prompt, 31 s of generation, over 3 tasks: 566); nan for a run without a card log "
      f"({[ag[a][c] for a in 'ef' for c in ecols]})")
plain_t = (out / "tab_plain/agent.tsv").read_text().splitlines()
check(plain_t[0].split("\t") == head[:19] and [l.split("\t") for l in plain_t[1:]]
      == [l.split("\t")[:19] for l in (out / "tab/agent.tsv").read_text().splitlines()[1:]],
      "agent_tables: a folder without card logs gives the table without the energy columns, its other cells the same")
# class_arms.py
cl = {r["measure"]: r for r in tsv("class_ab.tsv")}
want = {"card joules a task at equal work": "gain", "card joules a task": "negligible",
        "card joules a generated token": "gain", "minutes a task at equal work": "negligible"}
check(len(cl) == 13 and all(cl[m]["class"] == c for m, c in want.items())
      and not any("W " in m or "watts" in m for m in cl),
      f"class_arms: the joules at equal work and a token beyond the spread are gains, the session's joules a task "
      f"within it negligible, the watts not classed ({[(m, cl[m]['class']) for m in want if m in cl]})")
an = {r["measure"]: r for r in tsv("class_an.tsv")}
check(all(an[m]["class"] == "n/a" and an[m]["diff_pct"] == "—" for m in want if m.startswith("card"))
      and an["minutes a task at equal work"]["class"] != "n/a",
      "class_arms: against a run without a card log the energy measures are n/a")
check([r["measure"] for r in tsv("class_work.tsv")][-1] == "longest first-token wait s"
      and len(tsv("class_work.tsv")) == 10, "class_arms: a table without the energy columns gives no energy measure")
sys.exit(1 if fails else 0)
EOF
echo "energy selftest passed; files in $OUT"
