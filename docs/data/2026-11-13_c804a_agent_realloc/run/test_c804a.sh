#!/usr/bin/env bash
# test_c804a.sh: the checks of C804's step (a) (Q1408). The agent check's report.py counts each server log's graph
# reallocations in gonogo and marks an arm in compare, and nothing else in its reports changes. CPU only, no model, no
# rig, no request to any server but the selftest's mock.
#
#   NEW=<this repository> OLD=<the repository before step (a), v4 5e46c8f> OUT=<results> \
#     R90=<R90's client folder> R72=<R72's client folder> \
#     R60=<R60's loads> R85=<R85's loads> FN84=<FN84's load> R123A=<R123a's loads> C692=<C692's work folder> \
#     nice -n 19 bash docs/data/2026-11-13_c804a_agent_realloc/run/test_c804a.sh
#
# A client folder is an agent check's collected results: res/, proxy.jsonl, rig/ with server-<arm>.log and
# meminfo.log, and the compare.md and gonogo.md its run wrote. R60 … C692 are the folders C804's own checks read
# (docs/data/2026-11-12_c804_realloc_in_checks/run/test_c804.sh): R60 (r60a/, r60ro8/), R85 (c217v/, c217i/), FN84
# (fn84_topk/), R123A (r123a_<rung>/), each with load-A/server.log, and C692 (server-<label>.log). The selftest needs
# pi's sandbox: AC_SANDBOX (default docker, the image AC_IMAGE, at --cpu-shares AC_CPU_SHARES, default 256) or bwrap
# with AC_HOME, AC_NODE_DIR, AC_PI_ROOT (bench/agent_check/README.md). STEPS (default: syntax selftest reports client
# counter). Writes OUT/summary.tsv (step, check, verdict, detail), OUT/logs.tsv (each agent server log's count),
# OUT/agent_reallocs.tsv (the counter's table of those logs) and OUT/reports/<run>/{old,new}/ (the reports).
set -uo pipefail
: "${NEW:?}" "${OLD:?}" "${OUT:?}" "${R90:?}" "${R72:?}" "${R60:?}" "${R85:?}" "${FN84:?}" "${R123A:?}" "${C692:?}"
export PYTHONDONTWRITEBYTECODE=1
export LC_ALL=C  # globs in the client container's order (gonogo lists rig/server-*.log in it)
STEPS=${STEPS:-syntax selftest reports client counter}
AC=bench/agent_check
D=$NEW/docs/data
RUNS="r90:$R90:v3,gains r72:$R72:gains,v4c2,v4c22,v4c2g"
RUNGS="base graphs kvonly overlap ngram base2"
mkdir -p "$OUT"
SUM=$OUT/summary.tsv
[ -f "$SUM" ] || printf 'step\tcheck\tverdict\tdetail\n' > "$SUM"
add() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$SUM"; echo "$1 $2: $3 ($4)"; }
ok() { [ "$1" = 0 ] && echo OK || echo FAIL; }
step() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }

if step syntax; then
  for f in $AC/client_report.sh $AC/selftest.sh $AC/release_check.sh $AC/race_rigs.sh \
           docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh \
           docs/data/2026-11-13_c804a_agent_realloc/run/test_c804a.sh; do
    bash -n "$NEW/$f" 2> "$OUT/syntax.err"; add syntax "bash -n ${f##*/}" "$(ok $?)" "$(head -c 200 "$OUT/syntax.err")"
  done
  for f in $AC/report.py scripts/realloc_counts.py; do
    python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$NEW/$f" 2> "$OUT/syntax.err"
    add syntax "parse ${f##*/}" "$(ok $?)" "$(head -c 200 "$OUT/syntax.err")"
  done
fi

if step selftest; then
  # the agent check's own end-to-end test against its mock server, with the new checks of the report's count
  AC_SANDBOX=${AC_SANDBOX:-docker} AC_CPU_SHARES=${AC_CPU_SHARES:-256} \
    bash "$NEW/$AC/selftest.sh" "$OUT/selftest" > "$OUT/selftest.txt" 2>&1
  rc=$?
  add selftest "bench/agent_check/selftest.sh" "$(ok $rc)" \
    "$(grep -c '^ok ' "$OUT/selftest.txt") checks ok, $(grep -c '^FAIL ' "$OUT/selftest.txt") failed"
  for k in "gonogo: each server log" "compare: the arm that reallocates" "compare: every other line"; do
    grep -q "^ok   $k" "$OUT/selftest.txt"; add selftest "$k" "$(ok $?)" "$(grep -m1 " $k" "$OUT/selftest.txt" | cut -c6-)"
  done
fi

# gonogo after against before: the same lines, each server-log line extended by its count and nothing else
gonogo_same() {
  python3 -I - "$1" "$2" <<'EOF'
import sys
a, b = (open(p).read().splitlines() for p in sys.argv[1:3])
bad = [(x, y) for x, y in zip(a, b)
       if x != y and not (x.startswith("- server log ") and y.startswith(x + ", graph reallocations "))]
ext = sum(1 for x, y in zip(a, b) if x != y)
print(f"{len(a)} lines, {ext} server-log lines extended" if len(a) == len(b) and not bad else f"differs: {bad[:2]}")
sys.exit(0 if len(a) == len(b) and not bad else 1)
EOF
}

if step reports; then
  # each client folder's three reports, written by client_report.sh before and after in a copy of the folder
  for run in $RUNS; do
    name=${run%%:*}; rest=${run#*:}; src=${rest%:*}; arms=${rest##*:}
    for v in old new; do
      d=$OUT/reports/$name/$v
      rm -rf "$d"; mkdir -p "$d/rig"
      cp -a "$src/res" "$src/proxy.jsonl" "$d/" && cp -a "$src"/rig/server-*.log "$src/rig/meminfo.log" "$d/rig/"
      R=$OLD; [ $v = new ] && R=$NEW
      bash "$R/$AC/client_report.sh" "$d" "$arms" > "$d/console.txt" 2>&1
      add reports "$name: client_report.sh ($v)" "$(ok $?)" "arms $arms"
    done
    o=$OUT/reports/$name/old n=$OUT/reports/$name/new
    cmp -s "$o/compare.md" "$src/compare.md"; add reports "$name: compare.md before = the run's own" "$(ok $?)" ""
    cmp -s "$o/gonogo.md" "$src/gonogo.md"; add reports "$name: gonogo.md before = the run's own" "$(ok $?)" ""
    cmp -s "$o/table.md" "$n/table.md"; add reports "$name: table.md after = before" "$(ok $?)" ""
    cmp -s "$o/res/attempts.csv" "$n/res/attempts.csv"; add reports "$name: attempts.csv after = before" "$(ok $?)" ""
    cmp -s "$o/res/turns.csv" "$n/res/turns.csv"; add reports "$name: turns.csv after = before" "$(ok $?)" ""
    diff "$o/compare.md" "$n/compare.md" > "$OUT/reports/$name/compare.diff"
    removed=$(grep -c '^<' "$OUT/reports/$name/compare.diff"); added=$(grep -c '^>' "$OUT/reports/$name/compare.diff")
    narms=$(tr ',' '\n' <<< "$arms" | wc -l)
    [ "$removed" = 0 ] && [ "$added" = $((narms + 3)) ]
    add reports "$name: compare.md after = before plus the count" "$(ok $?)" \
      "$removed lines changed, $added added: $(grep '^> - ' "$OUT/reports/$name/compare.diff" | cut -c5- | paste -sd ';' -)"
    det=$(gonogo_same "$o/gonogo.md" "$n/gonogo.md"); add reports "$name: gonogo.md after = before plus the count" "$(ok $?)" "$det"
    grep '^- server log' "$n/gonogo.md" | sed 's/^- server log \([^:]*\):.*, \(graph reallocations .*\)$/\1: \2/' \
      > "$OUT/reports/$name/gonogo_counts.txt"
    add reports "$name: gonogo's counts" OK "$(paste -sd ';' "$OUT/reports/$name/gonogo_counts.txt")"
  done
  # each agent server log: its build, whether C691's switch is set, and its count
  printf 'run\tarm\tbuild\ttop_k_rows\trequests\trequests_with_realloc\treallocs_same\treallocs_changed\n' > "$OUT/logs.tsv"
  args=()
  for run in $RUNS; do
    name=${run%%:*}; rest=${run#*:}; src=${rest%:*}
    for f in "$src"/rig/server-*.log; do
      arm=${f##*/server-}; arm=${arm%.log}
      build=$(head -n 1 "$f" | grep -o 'llama\.cpp-[^/ ]*' | head -n 1)
      topk=$(head -n 1 "$f" | grep -c 'LLAMA_SAMPLING_TOP_K_ROWS=1')
      line=$(python3 -I "$NEW/scripts/realloc_counts.py" --line "$arm=$f")
      read -r w n s dd <<< "$(sed 's/.*: \([0-9]*\) of \([0-9]*\) requests (\([0-9]*\) at the same size, \([0-9]*\) changed)/\1 \2 \3 \4/' <<< "$line")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$arm" "$build" "$topk" "$n" "$w" "$s" "$dd" >> "$OUT/logs.tsv"
      args+=("$name-$arm=$f")
    done
  done
  python3 -I "$NEW/scripts/realloc_counts.py" "${args[@]}" > "$OUT/agent_reallocs.tsv"
  add reports "the counter's table of the six agent logs" "$(ok $?)" "$(($(wc -l < "$OUT/agent_reallocs.tsv") - 1)) rows"
fi

if step client; then
  # the prep's copy (release_check.sh, race_rigs.sh, release_check_s12.sh): realloc_counts.py lands beside the bench
  # files, and report.py run from that copy counts as the checkout does; without it the report says so and runs on
  HERE=$NEW/$AC
  for c in with without; do
    rm -rf "$OUT/client/$c"; mkdir -p "$OUT/client/$c"
    if [ $c = with ]; then
      tar -C "$HERE/.." --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py \
        -C "$HERE/../../scripts" realloc_counts.py | tar -C "$OUT/client/$c" -xf -
    else
      tar -C "$HERE/.." --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py | tar -C "$OUT/client/$c" -xf -
    fi
  done
  [ -f "$OUT/client/with/realloc_counts.py" ] && cmp -s "$OUT/client/with/realloc_counts.py" "$NEW/scripts/realloc_counts.py"
  add client "the prep's tar puts realloc_counts.py beside task_results.py" "$(ok $?)" ""
  d=$OUT/reports/r72/new
  for c in with without; do
    (cd "$d" && python3 -I "$OUT/client/$c/agent_check/report.py" gonogo res --log proxy.jsonl --rtt 0.3 \
      --meminfo rig/meminfo.log --server-log rig/server-*.log > "$OUT/client/gonogo_$c.md"
     python3 -I "$OUT/client/$c/agent_check/report.py" compare res --log proxy.jsonl --arms gains,v4c2,v4c22,v4c2g \
      --turns-csv "$OUT/client/turns_$c.csv" --server-log rig/server-*.log > "$OUT/client/compare_$c.md")
    echo $? > "$OUT/client/rc_$c"
  done
  cmp -s "$OUT/client/gonogo_with.md" "$d/gonogo.md" \
    && cmp -s <(sed 1d "$OUT/client/compare_with.md") <(sed 1d "$d/compare.md")
  add client "report.py from the copy = from the checkout (gonogo, compare)" "$(ok $?)" ""
  n=$(grep -c 'graph reallocations not counted (scripts/realloc_counts.py not found)' "$OUT/client/gonogo_without.md")
  m=$(grep -c 'not counted (scripts/realloc_counts.py not found)' "$OUT/client/compare_without.md")
  [ "$(cat "$OUT/client/rc_without")" = 0 ] && [ "$n" = 4 ] && [ "$m" = 4 ]
  add client "a copy without realloc_counts.py: the reports are written and say so" "$(ok $?)" \
    "gonogo $n lines, compare $m lines"
fi

if step counter; then
  # realloc_counts.py after against before: the same output on every log C804 checked, in every mode, and the
  # published tables byte for byte; a log with two loads counts each from its own start
  T=$NEW/scripts/realloc_counts.py T0=$OLD/scripts/realloc_counts.py
  mkdir -p "$OUT/counter"
  LOGS=(a="$R60/r60a/load-A/server.log" ro8="$R60/r60ro8/load-A/server.log" r85v="$R85/c217v/load-A/server.log"
        r85i="$R85/c217i/load-A/server.log" topk="$FN84/fn84_topk/load-A/server.log")
  for r in $RUNGS; do LOGS+=("r123a_$r=$R123A/r123a_$r/load-A/server.log"); done
  for f in "$C692"/server-*.log; do n=${f##*/server-}; LOGS+=("c692_${n%.log}=$f"); done
  for mode in buckets requests line marks; do
    opt=(); [ $mode = requests ] && opt=(--requests); [ $mode = line ] && opt=(--line)
    [ $mode = marks ] && opt=(--marks ladder)
    python3 -I "$T" "${opt[@]}" "${LOGS[@]}" > "$OUT/counter/new_$mode.tsv"
    python3 -I "$T0" "${opt[@]}" "${LOGS[@]}" > "$OUT/counter/old_$mode.tsv"
    cmp -s "$OUT/counter/new_$mode.tsv" "$OUT/counter/old_$mode.tsv"
    add counter "after = before, $mode, ${#LOGS[@]} logs" "$(ok $?)" "$(wc -l < "$OUT/counter/new_$mode.tsv") lines"
  done
  python3 -I "$T" r85v="$R85/c217v/load-A/server.log" r85i="$R85/c217i/load-A/server.log" \
    a="$R60/r60a/load-A/server.log" ro8="$R60/r60ro8/load-A/server.log" > "$OUT/counter/r60.tsv"
  cmp -s "$OUT/counter/r60.tsv" "$D/2026-11-12_r60_taper_v4/reallocs.tsv"; add counter "R60's reallocs.tsv" "$(ok $?)" ""
  python3 -I "$T" topk="$FN84/fn84_topk/load-A/server.log" > "$OUT/counter/fn84.tsv"
  cmp -s "$OUT/counter/fn84.tsv" "$D/2026-11-12_fn84_graph_realloc/reallocs.tsv"; add counter "FN84's reallocs.tsv" "$(ok $?)" ""
  args=(); for r in $RUNGS; do args+=("$r=$R123A/r123a_$r/load-A/server.log"); done
  python3 -I "$T" "${args[@]}" > "$OUT/counter/r123a.tsv"
  cmp -s "$OUT/counter/r123a.tsv" "$D/2026-11-12_r123a_prompt_ladder/reallocs.tsv"; add counter "R123a's reallocs.tsv" "$(ok $?)" ""
  # two loads in one log, as when an arm is resumed: R72's gains log twice
  cat "$R72/rig/server-gains.log" "$R72/rig/server-gains.log" > "$OUT/counter/two_loads.log"
  one=$(python3 -I "$T" --line g="$R72/rig/server-gains.log")
  two=$(python3 -I "$T" --line g="$OUT/counter/two_loads.log")
  before=$(python3 -I "$T0" --line g="$OUT/counter/two_loads.log")
  read -r w n s dd <<< "$(sed 's/.*: \([0-9]*\) of \([0-9]*\) requests (\([0-9]*\) at the same size, \([0-9]*\) changed)/\1 \2 \3 \4/' <<< "$one")"
  [ "$two" = "g: graph reallocations: $((2 * w)) of $((2 * n)) requests ($((2 * s)) at the same size, $((2 * dd)) changed)" ]
  add counter "two loads in one log: each counted from its start" "$(ok $?)" "one load: ${one#g: }; two: ${two#g: }; before step (a): ${before#g: }"
fi
echo "summary: $(awk -F'\t' 'NR > 1 && $3 == "OK"' "$SUM" | wc -l) of $(($(wc -l < "$SUM") - 1)) checks OK ($SUM)"
