#!/usr/bin/env bash
# test_c804.sh: C804's (Q1408) checks, on any Linux machine with bash, python3, mawk, busybox and Docker; no GPU, no
# model, nothing timed. Every check reads server logs that earlier rows wrote; the drivers run against stand-ins (a
# server that prints a recorded log, a docker that writes a recorded line), so no llama process starts and no request
# is sent. The container steps run in IMAGE (an Ubuntu 24.04 with python3 and curl) without a network.
#   REPO=<this repository> OUT=<results> ORIG=<the drivers before C804> R60=<R60's loads> R85=<R85's loads> \
#     FN84=<FN84's load> R123A=<R123a's loads> R122=<R122's rig out/> C692=<C692's WORK> \
#     nice -n 19 bash test_c804.sh
# Env: REPO, OUT, ORIG (a folder with the six drivers as they were before C804: rig_taper.sh, rig_loads.sh,
#   deep_table.py, r123a_logs.sh, r123b_logs.sh, kld_depth.sh), R60 (folder with r60a/ and r60ro8/), R85 (c217v/ and
#   c217i/), FN84 (fn84_topk/), R123A (r123a_<rung>/ for base graphs kvonly overlap ngram base2), R122 (out/ of R122's
#   rig run: <load>/server-<load>.log), C692 (server-<label>.log), IMAGE (pq-c735-srv:24.04), DOCKER_ARGS, STEPS
#   (default: syntax tool awk marks logs kld taper loads deep).
# Output: OUT/summary.tsv (step, check, verdict, detail), each step's files in OUT/<step>/; exit 1 if a check fails.
set -uo pipefail
: "${REPO:?}" "${OUT:?}" "${ORIG:?}" "${R60:?}" "${R85:?}" "${FN84:?}" "${R123A:?}" "${R122:?}" "${C692:?}"
REPO=$(realpath "$REPO"); OUT=$(realpath -m "$OUT"); ORIG=$(realpath "$ORIG")
IMAGE=${IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-syntax tool awk marks logs kld taper loads deep}
TOOL=$REPO/scripts/realloc_counts.py
D=$REPO/docs/data
SELF=$(realpath "$0")
mkdir -p "$OUT"
step() { [[ " $STEPS " == *" $1 "* ]]; }
add() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$OUT/summary.tsv"; echo "$1 $2: $3 ($4)"; }
ok() { [ "$1" = 0 ] && echo OK || echo FAIL; }
# fn_of FILE NAME: a shell function's text, from its line `NAME() {` to the first line that is `}` alone
fn_of() { awk -v f="$2" 'index($0, f "() {") == 1 || index($0, "  " f "() {") == 1 { p = 1 } p { print } p && /^ *}$/ { exit }' "$1"; }
# the logs: name=path
LOGS=(a="$R60/r60a/load-A/server.log" ro8="$R60/r60ro8/load-A/server.log" r85v="$R85/c217v/load-A/server.log"
      r85i="$R85/c217i/load-A/server.log" topk="$FN84/fn84_topk/load-A/server.log")
RUNGS="base graphs kvonly overlap ngram base2"
for r in $RUNGS; do LOGS+=("r123a_$r=$R123A/r123a_$r/load-A/server.log"); done
for f in "$C692"/server-*.log; do n=${f##*/server-}; LOGS+=("c692_${n%.log}=$f"); done
DRIVERS=(docs/data/2026-11-04_prompt_taper/run/rig_taper.sh docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh
         docs/data/2026-11-11_c692_device_rows_ptx/run/run_c692.sh bench/draft_count/run_draft_count.sh
         docs/data/2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh)

# ---- inside the container: the drivers end to end against stand-ins --------------------------------------------------
if [ "${1:-}" = taper-inner ]; then
  # rig_taper.sh before and after C804, two loads: A prints FN84's log, B R60's load a; the stand-in server prints its
  # log and exits, so each load ends at "server did not start" and the driver's end-of-load lines run on that log
  W=$OUT/taper; mkdir -p "$W/bin" "$W/builds/b/bin"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/bin/pkill"
  printf '#!/usr/bin/env bash\nexit 1\n' > "$W/bin/curl"
  cat > "$W/builds/b/bin/llama-server" << EOF
#!/usr/bin/env bash
case " \$* " in *" --ctx-checkpoints 4 "*) cat "$FN84/fn84_topk/load-A/server.log" ;; *) cat "$R60/r60a/load-A/server.log" ;; esac
EOF
  echo "name: stand-in" > "$W/builds/b/BUILDINFO.txt"
  chmod +x "$W/bin/"* "$W/builds/b/bin/llama-server"
  for v in orig new; do
    mkdir -p "$W/$v"; touch "$W/$v/taper_replay.py" "$W/$v/turns.csv"
    [ $v = orig ] && cp "$ORIG/rig_taper.sh" "$W/$v/" || cp "$REPO/docs/data/2026-11-04_prompt_taper/run/rig_taper.sh" "$W/$v/"
    PATH=$W/bin:$PATH BUILDS=$W/builds BUILD=b MODEL=/dev/null OUT=$W/$v/out LOADS="A B" bash "$W/$v/rig_taper.sh" \
      > "$W/$v/console.txt" 2>&1
  done
  exit 0
fi
if [ "${1:-}" = loads-inner ]; then
  # rig_loads.sh before and after C804 with R122's loads (r122_loads.sh); the stand-in rig_server.sh puts the load's
  # recorded server log in its folder. Run `real`: R122's logs; run `swap`: loads a and a-2 print R60's load a instead
  W=$OUT/loads; mkdir -p "$W/bin"
  for c in pkill pgrep; do printf '#!/usr/bin/env bash\necho 0; exit 1\n' > "$W/bin/$c"; done
  printf '#!/usr/bin/env bash\necho "{}"\n' > "$W/bin/curl"
  cat > "$W/rig_server.sh" << 'EOF'
#!/usr/bin/env bash
# stand-in: ACTION=start copies the load's recorded log; ACTION=end does nothing
[ "$ACTION" = start ] || exit 0
src=$SRC/$ARM/server-$ARM.log
case " ${SWAP:-} " in *" $ARM "*) src=$SWAP_LOG ;; esac
mkdir -p "$LOGDIR" && cp "$src" "$LOGDIR/server-$ARM.log"
EOF
  printf '#!/usr/bin/env bash\nwhile [ ! -e "$STOP" ]; do sleep 0.2; done\n' > "$W/vm_sampler.sh"
  chmod +x "$W/bin/"*
  printf 'PRESET_ENV="X=1"\nPRESET_ARGS="-ngl 99"\n' > "$W/preset.env"
  echo x > "$W/table"; echo '{}' > "$W/req.json"
  for run in real swap; do
    for v in orig new; do
      d=$W/$run/$v; mkdir -p "$d"
      [ $v = orig ] && cp "$ORIG/rig_loads.sh" "$d/" || cp "$REPO/docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh" "$d/"
      sw=""; [ $run = swap ] && sw="a a-2"
      PATH=$W/bin:$PATH SRC=$R122 SWAP="$sw" SWAP_LOG=$R60/r60a/load-A/server.log BIN=/x MODEL=/x DRAFT=/x \
        TABLE=$W/table PRESET=$W/preset.env RIG_SERVER=$W/rig_server.sh VM_SAMPLER=$W/vm_sampler.sh \
        REQ_GREEDY=$W/req.json REQ_LONG=$W/req.json OUT=$d/out PAUSE=0 RIG_LOADS=$d/rig_loads.sh \
        bash "$REPO/docs/data/2026-11-12_r122_host_ladder/run/r122_loads.sh" > "$d/console.txt" 2>&1
    done
  done
  exit 0
fi
if [ "${1:-}" = deep-inner ]; then
  # deep_lane's own selftest with the new deep_table.py, then its rig run's table again with recorded server logs
  W=$OUT/deep
  bash "$REPO/bench/deep_lane/selftest.sh" "$W/self" > "$W/selftest.txt" 2>&1
  echo "selftest exit $?" >> "$W/selftest.txt"
  exit 0
fi

rm -f "$OUT/summary.tsv"
printf 'step\tcheck\tverdict\tdetail\n' > "$OUT/summary.tsv"

if step syntax; then
  for f in "${DRIVERS[@]}" docs/data/2026-11-12_r123a_prompt_ladder/run/r123a_logs.sh \
           docs/data/2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh docs/data/2026-11-11_fn64_sparse_prompt_ptx/run/kld_depth.sh \
           docs/data/2026-11-12_c804_realloc_in_checks/run/test_c804.sh; do
    bash -n "$REPO/$f" 2> "$OUT/syntax.err"; add syntax "bash -n $(basename "$f")" "$(ok $?)" "$(head -c 200 "$OUT/syntax.err")"
  done
  for f in scripts/realloc_counts.py bench/deep_lane/deep_table.py; do
    python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$REPO/$f" 2> "$OUT/syntax.err"
    add syntax "parse $(basename "$f")" "$(ok $?)" "$(head -c 200 "$OUT/syntax.err")"
  done
fi

if step tool; then
  # the new tool's default table is the published one, and R60's script's on the same logs
  mkdir -p "$OUT/tool"
  python3 -I "$TOOL" r85v="$R85/c217v/load-A/server.log" r85i="$R85/c217i/load-A/server.log" \
    a="$R60/r60a/load-A/server.log" ro8="$R60/r60ro8/load-A/server.log" > "$OUT/tool/r60.tsv"
  cmp -s "$OUT/tool/r60.tsv" "$D/2026-11-12_r60_taper_v4/reallocs.tsv"; add tool "R60's reallocs.tsv" "$(ok $?)" \
    "$(awk -F'\t' '$1 == "a" && $2 == "all" { print "a: " $7 " of " $3 " requests, " $5 " same, " $6 " changed" }' "$OUT/tool/r60.tsv")"
  python3 -I "$TOOL" topk="$FN84/fn84_topk/load-A/server.log" > "$OUT/tool/fn84.tsv"
  cmp -s "$OUT/tool/fn84.tsv" "$D/2026-11-12_fn84_graph_realloc/reallocs.tsv"; add tool "FN84's reallocs.tsv" "$(ok $?)" \
    "$(awk -F'\t' '$2 == "all" { print $7 " of " $3 " requests" }' "$OUT/tool/fn84.tsv")"
  args=(); for r in $RUNGS; do args+=("$r=$R123A/r123a_$r/load-A/server.log"); done
  python3 -I "$TOOL" "${args[@]}" > "$OUT/tool/r123a.tsv"
  cmp -s "$OUT/tool/r123a.tsv" "$D/2026-11-12_r123a_prompt_ladder/reallocs.tsv"; add tool "R123a's reallocs.tsv" "$(ok $?)" \
    "$(awk -F'\t' '$2 == "all" { w += $7; n += $3 } END { print w " of " n " requests in 6 loads" }' "$OUT/tool/r123a.tsv")"
  python3 -I "$TOOL" "${LOGS[@]}" > "$OUT/tool/all_new.tsv"
  python3 -I "$D/2026-11-12_r60_taper_v4/run/realloc_counts.py" "${LOGS[@]}" > "$OUT/tool/all_r60script.tsv"
  cmp -s "$OUT/tool/all_new.tsv" "$OUT/tool/all_r60script.tsv"; add tool "same as R60's script on ${#LOGS[@]} logs" "$(ok $?)" \
    "$(($(wc -l < "$OUT/tool/all_new.tsv") - 1)) rows"
fi

if step awk; then
  # each driver's realloc_count, with mawk and busybox awk, against the tool on every log: the summary line and the
  # per-request rows (request, prompt tokens, reallocations of each kind)
  mkdir -p "$OUT/awk"
  python3 -I "$TOOL" --line "${LOGS[@]}" > "$OUT/awk/expected_lines.txt"
  python3 -I "$TOOL" --requests "${LOGS[@]}" > "$OUT/awk/expected_requests.tsv"
  for f in "${DRIVERS[@]}"; do
    b=$(basename "$f" .sh)
    fn_of "$REPO/$f" realloc_count > "$OUT/awk/fn_$b.sh"
    for impl in mawk busybox; do
      : > "$OUT/awk/${b}_$impl.lines"; : > "$OUT/awk/${b}_$impl.requests"
      for spec in "${LOGS[@]}"; do
        name=${spec%%=*}
        bash -c ". '$OUT/awk/fn_$b.sh'; awk() { if [ $impl = busybox ]; then busybox awk \"\$@\"; else mawk \"\$@\"; fi; }
                 realloc_count '${spec#*=}' '$OUT/awk/req.tsv'" | sed "s/^/$name: /" >> "$OUT/awk/${b}_$impl.lines"
        tail -n +2 "$OUT/awk/req.tsv" | awk -F'\t' -v n="$name" '{ print n "\t" $1 "\t" $2 "\t" $4 "\t" $5 }' \
          >> "$OUT/awk/${b}_$impl.requests"
      done
      cmp -s "$OUT/awk/${b}_$impl.lines" "$OUT/awk/expected_lines.txt"; r1=$?
      tail -n +2 "$OUT/awk/expected_requests.tsv" | cut -f1,2,4,6,7 | cmp -s - "$OUT/awk/${b}_$impl.requests"; r2=$?
      add awk "$b with $impl" "$(ok $((r1 + r2)))" \
        "$(grep -c . "$OUT/awk/${b}_$impl.lines") summaries, $(grep -c . "$OUT/awk/${b}_$impl.requests") requests; $(grep -E '^(a|topk|c692_day-pre):' "$OUT/awk/${b}_$impl.lines" | tr '\n' ';')"
    done
  done
fi

if step marks; then
  mkdir -p "$OUT/marks"
  L=(); for l in base a b c d e f base-2 a-2 b-2 c-2 d-2 e-2 f-2; do L+=("$l=$R122/$l/server-$l.log"); done
  python3 -I "$TOOL" --marks ladder "${L[@]}" > "$OUT/marks/r122_ladder.tsv"
  v=$(awk -F'\t' 'NR > 1 { print $1 ":" $2 ":" $8 }' "$OUT/marks/r122_ladder.tsv" | tr '\n' ' ')
  want="base:-:- a:base,base-2:none b:a,a-2:none c:b,b-2:none d:c,c-2:none e:d,d-2:none f:e,e-2:none base-2:-:- a-2:base,base-2:none b-2:a,a-2:none c-2:b,b-2:none d-2:c,c-2:none e-2:d,d-2:none f-2:e,e-2:none "
  [ "$v" = "$want" ]; add marks "R122's ladder (14 loads)" "$(ok $?)" "$v"
  A=(); for r in $RUNGS; do A+=("$r=$R123A/r123a_$r/load-A/server.log"); done
  python3 -I "$TOOL" --marks ladder "${A[@]}" > "$OUT/marks/r123a_ladder.tsv"
  v=$(awk -F'\t' 'NR > 1 { print $1 ":" $2 ":" $8 }' "$OUT/marks/r123a_ladder.tsv" | tr '\n' ' ')
  [ "$v" = "base:-:- graphs:base,base2:none kvonly:graphs:none overlap:kvonly:none ngram:overlap:none base2:-:- " ]
  add marks "R123a's ladder (6 loads)" "$(ok $?)" "$v"
  # a switch that adds reallocations: FN84's load as the line, R60's load a as the line with the switch
  cat "$FN84/fn84_topk/load-A/server.log" "$R60/r60a/load-A/server.log" > "$OUT/marks/mixed.log"
  python3 -I "$TOOL" --marks 'ladder mixed=sw sw2=mixed' base="$FN84/fn84_topk/load-A/server.log" \
    sw="$R60/r60a/load-A/server.log" base2="$FN84/fn84_topk/load-A/server.log" mixed="$OUT/marks/mixed.log" \
    sw2="$R60/r60a/load-A/server.log" > "$OUT/marks/switch.tsv"
  v=$(awk -F'\t' 'NR > 1 { print $1 ":" $2 ":" $4 "/" $3 ":" $8 }' "$OUT/marks/switch.tsv" | tr '\n' ' ')
  [ "$v" = "base:-:0/8:- sw:base,base2:373/373:reallocates base2:-:0/8:- mixed:sw:373/381:as its line sw2:mixed:373/373:reallocates more " ]
  add marks "a switch's load against FN84's line" "$(ok $?)" "$v"
fi

if step logs; then
  # r123a_logs.sh and r123b_logs.sh before and after C804 on R123a's six loads: the same files, plus realloc_marks.tsv
  mkdir -p "$OUT/logs"
  args=(); for r in $RUNGS; do args+=("$r=$R123A/r123a_$r"); done
  for s in r123a r123b; do
    for v in orig new; do
      if [ $v = orig ]; then f=$ORIG/${s}_logs.sh; else f=$D/2026-11-12_${s}_prompt_ladder/run/${s}_logs.sh; fi
      OUT=$OUT/logs/$s-$v REPO=$REPO bash "$f" "${args[@]}" > "$OUT/logs/$s-$v.console" 2>&1
      echo "exit $?" >> "$OUT/logs/$s-$v.console"
    done
    same=0
    for x in "$OUT/logs/$s-orig"/*; do cmp -s "$x" "$OUT/logs/$s-new/$(basename "$x")" || same=1; done
    add logs "$s: the files it wrote before, unchanged" "$(ok $same)" "$(ls "$OUT/logs/$s-orig" | tr '\n' ' ')"
    v=$(awk -F'\t' 'NR > 1 { print $1 ":" $8 }' "$OUT/logs/$s-new/realloc_marks.tsv" 2> /dev/null | tr '\n' ' ')
    [ "$v" = "base:- graphs:none kvonly:none overlap:none ngram:none base2:- " ]; add logs "$s: realloc_marks.tsv" "$(ok $?)" "$v"
  done
  cmp -s "$OUT/logs/r123a-new/loads.tsv" "$D/2026-11-12_r123a_prompt_ladder/loads.tsv"
  add logs "r123a: loads.tsv as published" "$(ok $?)" "$(grep -c . "$OUT/logs/r123a-new/loads.tsv") lines"
fi

if step kld; then
  # kld_depth.sh with a stand-in docker: GGML_SCHED_DEBUG_REALLOC=1 reaches every run by default, SCHED_DEBUG_REALLOC=0
  # leaves it out, and a run that aborts on a reallocation is named in the console
  W=$OUT/kld; rm -rf "$W"; mkdir -p "$W/bin"
  cat > "$W/bin/docker" << 'EOF'
#!/usr/bin/env bash
# stand-in: logs its arguments; llama-perplexity's lines, or the scheduler's abort when the run has FAKE_REALLOC=1
case $1 in rm|container) exit 1 ;; esac
echo "docker $*" >> "$KLD_ARGS"
case " $* " in
  *" -e FAKE_REALLOC=1 "*) echo "ggml_backend_sched_alloc_splits: unexpected graph reallocation (graph size = 9000, nodes = 6944, leafs = 1200), debug_realloc = 1"; exit 134 ;;
  *--kl-divergence-base*--kl-divergence*) printf 'Mean    KLD:   0.000001\nSame top p: 100.000 %%\n' ;;
  *) for a in "$@"; do case $a in *.partial) : > "$a" ;; esac; done; echo "Final estimate: PPL = 5.0" ;;
esac
EOF
  chmod +x "$W/bin/docker"
  for v in on off; do
    sdr=""; [ $v = off ] && sdr=0
    PATH=$W/bin:$PATH KLD_ARGS=$W/args_$v.txt SCHED_DEBUG_REALLOC=$sdr BIN=/x MODEL=/x TEXT=/x OUT=$W/$v GPUS=0,1 \
      ARMS="base sw" ARM_ENV_sw=FAKE_REALLOC=1 DEPTHS=8192 \
      bash "$D/2026-11-11_fn64_sparse_prompt_ptx/run/kld_depth.sh" > "$W/console_$v.txt" 2>&1
  done
  [ "$(grep -c -- '-e GGML_SCHED_DEBUG_REALLOC=1 ' "$W/args_on.txt")" = 2 ] && ! grep -q GGML_SCHED_DEBUG_REALLOC "$W/args_off.txt"
  add kld "the switch in every run, none with SCHED_DEBUG_REALLOC=0" "$(ok $?)" \
    "on: $(grep -c -- 'GGML_SCHED_DEBUG_REALLOC=1' "$W/args_on.txt") of $(grep -c . "$W/args_on.txt") runs; off: $(grep -c GGML_SCHED_DEBUG_REALLOC "$W/args_off.txt")"
  grep -q 'graph reallocation at an unchanged size in sw 8192: unexpected graph reallocation' "$W/console_on.txt" \
    && grep -q 'FAILED sw 8192' "$W/console_on.txt" && [ -e "$W/on/failed" ]
  add kld "an aborted run named and failed" "$(ok $?)" "$(grep 'unchanged size' "$W/console_on.txt" | cut -c1-120)"
fi

DOCKER=(docker run --rm --init --network none --label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}
        -v "$REPO:$REPO:ro" -v "$OUT:$OUT" -v "$ORIG:$ORIG:ro" -v "$R60:$R60:ro" -v "$FN84:$FN84:ro" -v "$R122:$R122:ro"
        -e "REPO=$REPO" -e "OUT=$OUT" -e "ORIG=$ORIG" -e "R60=$R60" -e "R85=$R85" -e "FN84=$FN84" -e "R123A=$R123A"
        -e "R122=$R122" -e "C692=$C692" "$IMAGE")

if step taper; then
  rm -rf "$OUT/taper"; mkdir -p "$OUT/taper"
  "${DOCKER[@]}" bash "$SELF" taper-inner > "$OUT/taper/docker.txt" 2>&1
  W=$OUT/taper
  strip() { sed -E 's/^[0-9T:+-]+ //; s/up after [0-9]+ s/up after N s/; s/lowest MemAvailable [0-9]+ MiB/lowest MemAvailable N MiB/' "$1"; }
  diff <(strip "$W/orig/out/run.log") <(strip "$W/new/out/run.log") > "$W/run_log.diff"
  added=$(grep -c '^>' "$W/run_log.diff"); removed=$(grep -c '^<' "$W/run_log.diff")
  [ "$removed" = 0 ] && [ "$added" = 2 ]; add taper "rig_taper.sh: run.log as before plus one line a load" "$(ok $?)" \
    "$added added, $removed removed: $(grep '^>' "$W/run_log.diff" | cut -c3- | tr '\n' ';')"
  grep -q 'load A: graph reallocations: 0 of 8 requests (0 at the same size, 0 changed)' "$W/new/out/run.log" \
    && grep -q 'load B: graph reallocations: 373 of 373 requests (802 at the same size, 745 changed)' "$W/new/out/run.log"
  add taper "rig_taper.sh: FN84 0 of 8, R60 373 of 373" "$(ok $?)" "$(ls "$W/new/out/load-A" | tr '\n' ' ')"
  same=0; for x in "$W/orig/out/load-A"/* "$W/orig/out/load-B"/*; do
    case $x in *memavail.txt|*memory_*) continue ;; esac
    cmp -s "$x" "${x/\/orig\//\/new\/}" || same=1; done
  add taper "rig_taper.sh: every other file of a load unchanged" "$(ok $same)" "reallocs.tsv added"
fi

if step loads; then
  rm -rf "$OUT/loads"; mkdir -p "$OUT/loads"
  "${DOCKER[@]}" bash "$SELF" loads-inner > "$OUT/loads/docker.txt" 2>&1
  W=$OUT/loads
  strip() { sed -E 's/^[0-9T:+-]+ //; s/[0-9]{10}\.[0-9]{3}/T/g; s/quiet: load [0-9. ]+;/quiet: load L;/; s/MemAvailable: +[0-9]+ kB/MemAvailable: M kB/; s/\([0-9]+ bytes\)/(B bytes)/; s#/(orig|new)/#/V/#g' "$1"; }
  for run in real swap; do
    diff <(strip "$W/$run/orig/out/status.txt") <(strip "$W/$run/new/out/status.txt") > "$W/$run.diff"
    added=$(grep -c '^>' "$W/$run.diff"); removed=$(grep -c '^<' "$W/$run.diff")
    marks=$(awk -F'\t' 'NR > 1 { print $1 ":" $3 ":" $9 }' "$W/$run/new/out/realloc_marks.tsv" | tr '\n' ' ')
    if [ $run = real ]; then
      want="base:-:- a:base,base-2:none b:a,a-2:none c:b,b-2:none d:c,c-2:none e:d,d-2:none f:e,e-2:none base-2:-:- a-2:base,base-2:none b-2:a,a-2:none c-2:b,b-2:none d-2:c,c-2:none e-2:d,d-2:none f-2:e,e-2:none "
      n=15
    else
      want="base:-:- a:base,base-2:reallocates b:a,a-2:none c:b,b-2:none d:c,c-2:none e:d,d-2:none f:e,e-2:none base-2:-:- a-2:base,base-2:reallocates b-2:a,a-2:none c-2:b,b-2:none d-2:c,c-2:none e-2:d,d-2:none f-2:e,e-2:none "
      n=16
    fi
    [ "$removed" = 0 ] && [ "$added" = "$n" ]
    add loads "rig_loads.sh ($run): status.txt as before plus the count lines" "$(ok $?)" "$added added, $removed removed"
    [ "$marks" = "$want" ]; add loads "rig_loads.sh ($run): realloc_marks.tsv" "$(ok $?)" "$marks"
  done
  grep '^>' "$W/swap.diff" | grep -q 'graph reallocations: a reallocates (373 of 373 requests; its line without the switch, base,base-2: 0 of 2,0 of 2)' ; r=$?
  add loads "rig_loads.sh (swap): the line that names load a" "$(ok $r)" "$(grep '^>' "$W/swap.diff" | grep 'reallocates (' | cut -c3- | tr '\n' ';')"
fi

if step deep; then
  rm -rf "$OUT/deep"; mkdir -p "$OUT/deep"
  "${DOCKER[@]}" bash "$SELF" deep-inner > "$OUT/deep/docker.txt" 2>&1
  W=$OUT/deep
  grep -q 'selftest exit 0' "$W/selftest.txt"; add deep "deep_lane's selftest with the new deep_table.py" "$(ok $?)" \
    "$(grep -c '^ok\|^OK\|PASS' "$W/selftest.txt") ok lines; $(tail -n 2 "$W/selftest.txt" | head -n 1)"
  grep -q 'Graph reallocations: no `graph reallocs` line' "$W/self/rig/table.md"
  add deep "the mock's logs: no count, said so" "$(ok $?)" "$(grep 'Graph reallocations' "$W/self/rig/table.md" | cut -c1-100)"
  # the rig run's table again, its server logs replaced by recorded ones: fast's loads R60's load a, the rest FN84's
  T=$W/table; mkdir -p "$T"; cp "$W/self/rig/requests.csv" "$W/self/rig/loads.tsv" "$T/"
  for ld in $(awk -F'\t' 'NR > 1 { print $1 }' "$T/loads.tsv"); do
    case $ld in fast*) cp "$R60/r60a/load-A/server.log" "$T/server-$ld.log" ;; *) cp "$FN84/fn84_topk/load-A/server.log" "$T/server-$ld.log" ;; esac
  done
  python3 -I "$REPO/bench/deep_lane/deep_table.py" "$T" --out "$T/table.md" > /dev/null 2> "$T/err.txt"
  grep -q '\*\*`fast` reallocates where `base` does not.\*\*' "$T/table.md" && grep -q 'fast1: 373 of 373 requests (802 same size, 745 changed)' "$T/table.md"
  add deep "fast's loads marked against base" "$(ok $?)" "$(grep 'Graph reallocations' "$T/table.md" | cut -c1-240)"
  python3 -I "$ORIG/deep_table.py" "$T" --out "$T/table_orig.md" > /dev/null 2>&1
  diff "$T/table_orig.md" "$T/table.md" > "$T/table.diff"
  [ "$(grep -c '^>' "$T/table.diff")" = 1 ] && [ "$(grep -c '^<' "$T/table.diff")" = 0 ]
  add deep "the table as before plus one line" "$(ok $?)" "$(grep -c '^>' "$T/table.diff") added, $(grep -c '^<' "$T/table.diff") removed"
fi

fails=$(awk -F'\t' 'NR > 1 && $3 != "OK"' "$OUT/summary.tsv" | wc -l)
echo "$(($(wc -l < "$OUT/summary.tsv") - 1)) checks, $fails failed"
[ "$fails" = 0 ]
