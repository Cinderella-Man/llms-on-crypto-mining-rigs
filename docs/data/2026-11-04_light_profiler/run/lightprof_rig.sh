#!/usr/bin/env bash
# C142 / Q681 on the rig: one short Flash-Next server line per profiler mode on the -lightprof build, the same greedy
# requests in each (lightprof_requests.py), so that every mode's sum per token can be checked against the unprofiled
# token time. Runs ON the rig, detached, from a folder holding this script, lightprof_requests.py, prof_summary.py and
# ram_watchdog.sh (both from scripts/), as root (the watchdog sets the server's oom_score_adj):
#   BIN=<build>/bin MODEL=<first shard> OUT=<work dir> setsid nohup bash lightprof_rig.sh > <work dir>.console 2>&1 < /dev/null &
# Env: CTX (65536), ARGS (R13's v3 line: -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio), ARMS (off graph
# layer sample=8 group: off is the unprofiled reference and comes first, group is today's mode and the slowest; add a
# second "off" at the end to see the drift between loads), PORT (8096), N_WARM N_TOK REPS (16 64 3), EVENT_US (the
# profiler's cost per timed span from C140's bench, 0 if not known yet: the summaries can be rerun), TOL_PCT (5).
# Per arm: the load time, the server's profiler and memory lines, the requests, the profile CSV, the RAM watchdog's TSV;
# then per profiled arm prof_summary.py against the off arm's ms per token. Ends with $OUT/DONE; the table is
# $OUT/check.tsv, the verdicts $OUT/check.txt.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
ARMS=${ARMS:-off graph layer sample=8 group}
PORT=${PORT:-8096}
N_WARM=${N_WARM:-16} N_TOK=${N_TOK:-64} REPS=${REPS:-3}
EVENT_US=${EVENT_US:-0}
TOL_PCT=${TOL_PCT:-5}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
printf 'arm\trep\tprompt_n\tprompt_ms\tpredicted_n\tpredicted_ms\tms_per_token\tclient_s\thash\n' > "$OUT/requests.tsv"
if command -v chrt > /dev/null; then pre="chrt -f 10"; else pre="nice -n -20"; fi
echo "== lightprof: $BIN, $MODEL, -c $CTX $ARGS, arms: $ARMS $(date -Is)"
seen=""
for arm in $ARMS; do
  tag=${arm//=/}
  case " $seen " in *" $tag "*) tag=${tag}2 ;; esac
  seen="$seen $tag"
  envs=()
  [ "$arm" = off ] || envs=("GGML_CUDA_PROFILE=$OUT/prof-$tag.csv" "GGML_CUDA_PROFILE_MODE=$arm")
  echo "== $tag ($arm) $(date -Is)"
  pkill -x llama-server; sleep 2
  env ${envs[@]+"${envs[@]}"} "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" $ARGS -np 1 \
    --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" > "$OUT/server-$tag.log" 2>&1 < /dev/null &
  srv=$!
  rm -f "/dev/shm/lightprof-wd-$tag.tsv"
  PID=$srv LOG=/dev/shm/lightprof-wd-$tag.tsv SERVER_LOG="$OUT/server-$tag.log" $pre bash "$HERE/ram_watchdog.sh" \
    > "$OUT/watchdog-$tag.out" 2>&1 < /dev/null &
  wd=$!
  t0=$(date +%s)
  for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
  echo "load $(( $(date +%s) - t0 )) s"
  grep -E 'ggml_cuda_profile|graph splits|compute buffer size|out of memory|failed|error' "$OUT/server-$tag.log" | cut -c1-200 | head -20
  echo "cards: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')"
  free -m | sed -n 2p
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    python3 -I "$HERE/lightprof_requests.py" "$PORT" "$tag" "$OUT/requests.tsv" "$N_WARM" "$N_TOK" "$REPS"
    free -m | sed -n 2p
  else
    echo "server did not start"; tail -20 "$OUT/server-$tag.log"
  fi
  kill -INT $srv 2> /dev/null
  for i in $(seq 1 30); do kill -0 $srv 2> /dev/null || break; sleep 1; done
  pkill -9 -x llama-server
  kill $wd 2> /dev/null; sleep 1; kill -9 $wd 2> /dev/null
  cp "/dev/shm/lightprof-wd-$tag.tsv" "$OUT/ram_watchdog-$tag.tsv" 2> /dev/null && rm -f "/dev/shm/lightprof-wd-$tag.tsv"
  [ -f "$OUT/prof-$tag.csv" ] && echo "profile: $(wc -l < "$OUT/prof-$tag.csv") lines, $(du -h "$OUT/prof-$tag.csv" | cut -f1)"
done

# the unprofiled ms per token: the off arm's measured requests (not the warm-up)
ms_of() { awk -F'\t' -v a="$1" '$1 == a && $2 != "warm" { s += $7; n++ } END { if (n) printf "%.3f", s / n }' "$OUT/requests.tsv"; }
hash_of() { awk -F'\t' -v a="$1" '$1 == a && $2 != "warm" { print $9 }' "$OUT/requests.tsv" | sort -u | tr '\n' ' '; }
off_ms=$(ms_of off)
printf 'arm\tserver_ms_per_token\tserver_vs_off_pct\tsum_ms\tsum_vs_off_pct\tevals_ms\tevals_corr_ms\twall_run_ms\tcoverage\tdecode_evals\tused_evals\tskipped\tsame_output_as_off\tcheck\n' > "$OUT/check.tsv"
: > "$OUT/check.txt"
for tag in $seen; do
  ms=$(ms_of "$tag")
  same=$([ "$(hash_of "$tag")" = "$(hash_of off)" ] && echo yes || echo no)
  vs=$(awk -v a="$ms" -v b="$off_ms" 'BEGIN { if (a != "" && b > 0) printf "%+.1f", 100 * (a / b - 1) }')
  if [ -f "$OUT/prof-$tag.csv" ] && [ -n "$off_ms" ]; then
    python3 -I "$HERE/prof_summary.py" "$OUT/prof-$tag.csv" --event-us "$EVENT_US" --wall-ms "$off_ms" --tol-pct "$TOL_PCT" \
      > "$OUT/summary-$tag.txt" 2>&1
    s=$(grep '^summary: ' "$OUT/summary-$tag.txt" | tail -1)
    get() { echo "$s" | tr ' ' '\n' | sed -n "s/^$1=//p"; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ms" "$vs" "$(get sum_ms)" "$(get sum_vs_ref_pct)" \
      "$(get evals_ms)" "$(get evals_corr_ms)" "$(get wall_run_ms)" "$(get coverage)" "$(get decode_evals)" "$(get used_evals)" \
      "$(get skipped)" "$same" "$(get check)" >> "$OUT/check.tsv"
  else
    printf '%s\t%s\t%s\t\t\t\t\t\t\t\t\t\t%s\t\n' "$tag" "$ms" "$vs" "$same" >> "$OUT/check.tsv"
  fi
done
{
  echo "unprofiled: ${off_ms:-?} ms per token (off arm, measured requests); event cost subtracted: $EVENT_US us per span"
  awk -F'\t' -v tol="$TOL_PCT" 'NR > 1 && $1 != "off" && $1 !~ /^off/ {
      p = ($3 != "" && ($3 < 0 ? -$3 : $3) <= tol) ? "pass" : "fail"
      printf "%-8s server %s ms/token (%s %% vs off: the profiler'"'"'s own cost %s), sum of timed spans %s ms (%s %% vs off: %s), same output: %s\n",
        $1, $2, $3, p, $4, $5, ($14 == "" ? "-" : $14), $13 }' "$OUT/check.tsv"
  echo "C142's check: the layer and graph arms' sums within $TOL_PCT % of the unprofiled token time (their 'sum' verdicts above)."
} | tee "$OUT/check.txt"
echo "== end $(date -Is)"
touch "$OUT/DONE"
