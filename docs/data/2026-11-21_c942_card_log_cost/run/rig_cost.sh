#!/usr/bin/env bash
# rig_cost.sh (C943): what scripts/card_log.py costs the host on a P106 rig: today's sample against CARD_LOG_FAST=1
# and CARD_LOG_SLOW_EVERY=4 (C942), and whether a running log slows the model. Run it on the rig itself, in a quiet
# window (it refuses to start while a llama-server runs), from a folder holding card_log.py and this script.
#   1. idle, no model loaded: card_log.py alone for IDLE_S seconds (120) an arm, the arms today, fast and slow4 in turn,
#      REPS times (2); then today and slow4 once more with CARD_LOG_PROFILE=1 (each NVML function's calls and CPU).
#   2. busy: one llama-server load with a preset (PRESET, a release/<model>/settings.env), then ROUNDS (3) rounds of
#      the arms off (no log), today, fast and slow4 in turn, each arm REQS (4) requests of N_PREDICT (384) tokens from
#      one fixed prompt at temperature 0 while that arm's log runs; then one arm today_profile.
# Everything goes to OUT (default /tmp/c943): idle_<arm>_<rep>.csv and busy_<round>_<arm>.csv (the logs),
# resp_<round>_<arm>_<i>.json (the server's answers), host_cpu.tsv (/proc/stat for every arm), busy.tsv (each
# request's tokens and speeds), summary.md (per arm: the logger's CPU a sample from its stop line, the host's busy
# share, each request's generation tok/s by value), server.log, run.log, DONE. It stops the server and every logger
# it started on its way out. Est. 30 min with the defaults (two-thirds of it the idle phase and the load).
#   BIN=/path/to/llama-p106-v4.0 MODELS=/mnt/models PRESET=/path/to/release/flint-v4/settings.env bash rig_cost.sh
#   MODEL=/path/to/file.gguf in place of MODELS loads that file with the preset's settings.
#   SKIP_IDLE=1 or SKIP_BUSY=1 runs one phase; PORT (18943), CARD_LOG_PY (card_log.py beside this script).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/c943} PORT=${PORT:-18943} CARD_LOG_PY=${CARD_LOG_PY:-$HERE/card_log.py}
IDLE_S=${IDLE_S:-120} REPS=${REPS:-2} ROUNDS=${ROUNDS:-3} REQS=${REQS:-4} N_PREDICT=${N_PREDICT:-384}
SKIP_IDLE=${SKIP_IDLE:-0} SKIP_BUSY=${SKIP_BUSY:-0}
ARMS_IDLE="today fast slow4" ARMS_BUSY="off today fast slow4"
mkdir -p "$OUT" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/run.log"; }
die() { log "stopped: $*"; exit 1; }
arm_env() {  # the environment that makes card_log.py run arm $1, each variable set whatever the caller's are
  case $1 in
    today) echo "CARD_LOG_FAST=0 CARD_LOG_SLOW_EVERY=1 CARD_LOG_PROFILE=0" ;;
    fast) echo "CARD_LOG_FAST=1 CARD_LOG_SLOW_EVERY=1 CARD_LOG_PROFILE=0" ;;
    slow4) echo "CARD_LOG_FAST=1 CARD_LOG_SLOW_EVERY=4 CARD_LOG_PROFILE=0" ;;
    today_profile) echo "CARD_LOG_FAST=0 CARD_LOG_SLOW_EVERY=1 CARD_LOG_PROFILE=1" ;;
    slow4_profile) echo "CARD_LOG_FAST=1 CARD_LOG_SLOW_EVERY=4 CARD_LOG_PROFILE=1" ;;
  esac
}
since() { awk -v a="$(date +%s.%N)" -v b="$1" 'BEGIN { printf "%.3f", a - b }'; }  # seconds since $1
cpu_line() { awk '$1 == "cpu" { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat; }  # busy and idle jiffies
SRV="" CL=""
stop_all() {
  [ -n "$CL" ] && { kill "$CL" 2> /dev/null; sleep 1; kill -9 "$CL" 2> /dev/null; }
  [ -n "$SRV" ] && { kill "$SRV" 2> /dev/null; for _ in $(seq 20); do kill -0 "$SRV" 2> /dev/null || break; sleep 0.5; done
    kill -9 "$SRV" 2> /dev/null; }
  CL="" SRV=""
}
trap 'stop_all' EXIT
trap 'log "signal: stopping"; exit 1' INT TERM HUP

[ -r "$CARD_LOG_PY" ] || die "no card_log.py at $CARD_LOG_PY"
command -v curl > /dev/null || die "no curl"
pgrep -x llama-server > /dev/null && die "a llama-server runs already: not a quiet window"
log "rig_cost: OUT $OUT, card_log.py md5 $(md5sum < "$CARD_LOG_PY" | cut -c1-32), this script md5 $(md5sum < "$0" | cut -c1-32)"
log "settings: IDLE_S $IDLE_S REPS $REPS ROUNDS $ROUNDS REQS $REQS N_PREDICT $N_PREDICT PRESET ${PRESET:-none}"
printf 'phase\tround\tarm\tseconds\tbusy_jiffies\tidle_jiffies\n' > "$OUT/host_cpu.tsv"

if [ "$SKIP_IDLE" != 1 ]; then
  for rep in $(seq "$REPS") p; do
    arms=$ARMS_IDLE
    [ "$rep" = p ] && arms="today_profile slow4_profile"
    for arm in $arms; do
      read -r b0 i0 < <(cpu_line); s0=$(date +%s.%N)
      env $(arm_env "$arm") python3 -I "$CARD_LOG_PY" --log "$OUT/idle_${arm}_${rep}.csv" --interval 0.5 \
        --duration "$IDLE_S" > "$OUT/idle_${arm}_${rep}.out" 2>&1
      read -r b1 i1 < <(cpu_line)
      printf 'idle\t%s\t%s\t%s\t%s\t%s\n' "$rep" "$arm" "$(since "$s0")" $((b1 - b0)) $((i1 - i0)) \
        >> "$OUT/host_cpu.tsv"
      log "idle $rep $arm: $(grep -o 'card_log stop.*late by more than one interval' "$OUT/idle_${arm}_${rep}.csv" || cat "$OUT/idle_${arm}_${rep}.out")"
    done
  done
fi

if [ "$SKIP_BUSY" != 1 ]; then
  [ -n "${PRESET:-}" ] && [ -r "$PRESET" ] || die "PRESET (a release settings.env) is needed for the busy phase"
  [ -x "${BIN:-}/llama-server" ] || die "no llama-server in BIN (${BIN:-unset})"
  . "$PRESET"
  model=${MODEL:-${MODELS:-}/$PRESET_MODEL}
  [ -r "$model" ] || die "no model file at $model"
  log "busy: loading $model"
  env $PRESET_ENV "$BIN/llama-server" -m "$model" $PRESET_ARGS --host 127.0.0.1 --port "$PORT" \
    > "$OUT/server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1800); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 "$SRV" 2> /dev/null || die "the server exited while loading (server.log)"
    sleep 1
  done
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 || die "the server is not up after 30 min"
  log "busy: server up after $i s"
  BODY="{\"prompt\": \"Write a Python function that parses a CSV file of card readings (epoch, clock, power) and returns, for each card, the longest stretch of consecutive rows below a given clock, with its start time and length. Explain each step.\", \"n_predict\": $N_PREDICT, \"temperature\": 0, \"seed\": 1, \"cache_prompt\": false}"
  curl -s --max-time 600 -H 'Content-Type: application/json' -d "$BODY" "http://127.0.0.1:$PORT/completion" \
    > "$OUT/resp_warmup.json"  # the first request warms the graphs; not counted
  for round in $(seq "$ROUNDS") p; do
    arms=$ARMS_BUSY
    [ "$round" = p ] && arms=today_profile
    for arm in $arms; do
      csv=$OUT/busy_${round}_${arm}.csv
      if [ "$arm" != off ]; then
        env $(arm_env "$arm") python3 -I "$CARD_LOG_PY" --log "$csv" --interval 0.5 --parent $$ --stop "$csv.stop" \
          > "$csv.out" 2>&1 < /dev/null &
        CL=$!
        for _ in $(seq 20); do grep -qs '^[0-9]' "$csv" && break; sleep 0.5; done
        grep -qs '^[0-9]' "$csv" || log "busy $round $arm: no first sample after 10 s ($(tr '\n' ' ' < "$csv.out"))"
      fi
      read -r b0 i0 < <(cpu_line); s0=$(date +%s.%N)
      for i in $(seq "$REQS"); do
        r0=$(date +%s.%N)
        curl -s --max-time 600 -H 'Content-Type: application/json' -d "$BODY" "http://127.0.0.1:$PORT/completion" \
          > "$OUT/resp_${round}_${arm}_${i}.json"
        echo "$round $arm $i $(since "$r0")" >> "$OUT/req_wall.txt"
      done
      read -r b1 i1 < <(cpu_line)
      printf 'busy\t%s\t%s\t%s\t%s\t%s\n' "$round" "$arm" "$(since "$s0")" $((b1 - b0)) $((i1 - i0)) \
        >> "$OUT/host_cpu.tsv"
      if [ -n "$CL" ]; then
        kill "$CL" 2> /dev/null
        for _ in $(seq 6); do kill -0 "$CL" 2> /dev/null || break; sleep 0.5; done
        kill -9 "$CL" 2> /dev/null
        CL=""
      fi
      log "busy $round $arm: $(grep -o 'card_log stop.*late by more than one interval' "$csv" 2> /dev/null || echo 'no log')"
    done
  done
  stop_all
fi

python3 -I - "$OUT" << 'TABLES' > "$OUT/summary.md" 2>> "$OUT/run.log"
"""summary.md from OUT: per arm the logger's CPU a sample (its stop line), the host's busy share (/proc/stat) and,
for the busy phase, each request's generation and prompt speed by value."""
import csv, glob, json, os, re, sys
out = sys.argv[1]
stop = re.compile(r"card_log stop \([^)]*\): (\d+) samples, ([0-9.]+) s CPU \(user\+sys\), sampling ([0-9.]+) ms median")
prof = re.compile(r"card_log profile: (.*)")
def logger(path):
    if not os.path.exists(path):
        return "", ""
    text = open(path).read()
    m, p = stop.search(text), prof.search(text)
    cost = f"{float(m.group(2)) / int(m.group(1)) * 1000:.2f} ms CPU a sample ({m.group(1)} samples, {m.group(2)} s; " \
           f"sampling {m.group(3)} ms median)" if m and int(m.group(1)) else "no stop line"
    return cost, p.group(1) if p else ""
host = list(csv.DictReader(open(os.path.join(out, "host_cpu.tsv")), delimiter="\t"))
rows = []
for fn in sorted(glob.glob(os.path.join(out, "resp_*_*_*.json"))):
    m = re.fullmatch(r"resp_([^_]+)_(.+)_(\d+)\.json", os.path.basename(fn))
    if not m:
        continue
    rnd, arm, i = m.groups()
    try:
        t = json.load(open(fn)).get("timings", {})
    except ValueError:
        t = {}
    rows.append({"round": rnd, "arm": arm, "request": i, "prompt_n": t.get("prompt_n", ""),
                 "prompt_tok_s": f"{t['prompt_per_second']:.2f}" if "prompt_per_second" in t else "",
                 "predicted_n": t.get("predicted_n", ""),
                 "predicted_tok_s": f"{t['predicted_per_second']:.2f}" if "predicted_per_second" in t else ""})
with open(os.path.join(out, "busy.tsv"), "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["round"], delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print("# rig_cost.sh summary\n")
print("| Phase | Round | Arm | Logger | Host busy | Generation tok/s, each request | Prompt tok/s, each request |")
print("|---|---|---|---|---|---|---|")
for h in host:
    busy, idle = int(h["busy_jiffies"]), int(h["idle_jiffies"])
    share = f"{100 * busy / max(busy + idle, 1):.2f} % of 2 cores"
    path = os.path.join(out, f"{h['phase']}_{h['arm']}_{h['round']}.csv" if h["phase"] == "idle"
                        else f"busy_{h['round']}_{h['arm']}.csv")
    cost, _ = logger(path)
    mine = [r for r in rows if r["round"] == h["round"] and r["arm"] == h["arm"]] if h["phase"] == "busy" else []
    print(f"| {h['phase']} | {h['round']} | {h['arm']} | {cost or 'off'} | {share} | "
          f"{', '.join(r['predicted_tok_s'] for r in mine)} | {', '.join(r['prompt_tok_s'] for r in mine)} |")
for path in sorted(glob.glob(os.path.join(out, "*profile*.csv"))):
    print(f"\n{os.path.basename(path)}: {logger(path)[1] or 'no profile line'}")
TABLES
log "done: summary.md"
touch "$OUT/DONE"
