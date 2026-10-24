#!/usr/bin/env bash
# Times plain against pipelined decoding with llama-pipespec on the rig. One process per model and card set runs every
# job with a single model load (a load from a USB stick takes minutes):
#   - a warm-up;
#   - for each prompt (chat, code writing, whole-file edit, rename-and-rewrite): plain decoding (the reference, also
#     written as a transcript for bench/ngram_replay's simulator), the pipelined API without guesses (its own overhead),
#     the guesser "copy lookup >= 4, else continuation counts" (Q96; cont) at depths DEPTHS, and at depth 3 the copy
#     lookup alone (match >= 2 and >= 4) and ngram-map-k 4/8; with TABLE=<file>, also cont plus that continuation table;
#   - on the chat prompt, an oracle that replays the plain output: all guesses right (the pipeline's ceiling: one token
#     per slowest stage) and all wrong (the cost of wrong guesses) at every depth, and 30 % wrong at depth 3.
# Every pipelined job is checked against the plain job of its prompt: same tokens and bit-identical logits.
#
# Run it on the rig, from this repository:
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5 \
#       [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1'] [TS=6,6,6,6,6,6,6,5,2] bash /tmp/pipespec/rig_run.sh \
#       > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# Optional: N (tokens per job, 512), DEPTHS ('1 2 3 4 5'), PROMPTS ('chat code edit rewrite'), CTX (8192),
# BG (1: greedy sampling on the last card in both modes, as llama-server's backend sampling does, so only a token id
# crosses the link instead of 0.6 MB of logits; 0: logits to the host and sampling there), SAMPLING ('--temp 0'),
# TABLE (a continuation table from make_table.py), GPU_LOG (1: nvidia-smi clocks, power and load every second),
# EXTRA_ARGS. Output in /tmp/pipespec/out/<TAG>/: jobs.txt, run.log, results.txt (one PIPESPEC line per job),
# summary.tsv, plain.jsonl (the plain outputs), gpu.csv, host.csv (the process's CPU use each second, % of one core),
# *.trace (summarize with traces.py).
# summary.tsv's tok_s and speedup come from each job's trace: generated tokens per second from its first sampled token to
# its last. llama-pipespec's own tok_s (column tok_s_tgen) also counts the wait for its first token, which can include
# up to several seconds of the prompt still running on the cards (llama_decode returns before they finish), so it is
# low by a varying amount on the long prompts.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5}
N=${N:-512}
DEPTHS=${DEPTHS:-1 2 3 4 5}
PROMPTS=${PROMPTS:-chat code edit rewrite}
CTX=${CTX:-8192}
SAMPLING=${SAMPLING:---temp 0}
BG=${BG:-1}
TABLE=${TABLE:-}
GPU_LOG=${GPU_LOG:-1}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"
rm -f "$OUT/plain.jsonl" "$OUT/host.csv"

dmax=3
for D in $DEPTHS; do [ "$D" -gt "$dmax" ] && dmax=$D; done

category() { case $1 in rewrite) echo edit ;; *) echo "$1" ;; esac; }

jobs=$OUT/jobs.txt
{
    echo "name=warm mode=plain prompt=$HERE/prompts/chat.txt chat=1 n=64"
    for P in $PROMPTS; do
        c="prompt=$HERE/prompts/$P.txt chat=1 n=$N"
        echo "name=plain-$P mode=plain $c trace=$OUT/plain-$P.trace jsonl=$OUT/plain.jsonl cat=$(category "$P")"
        echo "name=d0-$P mode=pipe depth=0 guess=none $c trace=$OUT/d0-$P.trace"
        for D in $DEPTHS; do
            echo "name=cont-d$D-$P mode=pipe depth=$D guess=cont copy_min=4 $c trace=$OUT/cont-d$D-$P.trace"
        done
        echo "name=copy2-d3-$P mode=pipe depth=3 guess=copy copy_min=2 $c trace=$OUT/copy2-d3-$P.trace"
        echo "name=copy4-d3-$P mode=pipe depth=3 guess=copy copy_min=4 $c trace=$OUT/copy4-d3-$P.trace"
        echo "name=mapk-d3-$P mode=pipe depth=3 guess=spec spec=ngram-map-k $c trace=$OUT/mapk-d3-$P.trace"
    done
    P=$(echo "$PROMPTS" | awk '{print $1}')
    c="prompt=$HERE/prompts/$P.txt chat=1 n=$N"
    for D in $DEPTHS; do
        echo "name=oracle-d$D-m0-$P mode=pipe depth=$D guess=oracle oracle=@plain-$P miss=0 $c trace=$OUT/oracle-d$D-m0-$P.trace"
        echo "name=oracle-d$D-m1-$P mode=pipe depth=$D guess=oracle oracle=@plain-$P miss=1 $c trace=$OUT/oracle-d$D-m1-$P.trace"
    done
    echo "name=oracle-d3-m0.3-$P mode=pipe depth=3 guess=oracle oracle=@plain-$P miss=0.3 $c trace=$OUT/oracle-d3-m0.3-$P.trace"
} > "$jobs"

# the table variant runs in a second process (the table is a process setting)
jobs_table=$OUT/jobs-table.txt
: > "$jobs_table"
if [ -n "$TABLE" ]; then
    for P in $PROMPTS; do
        c="prompt=$HERE/prompts/$P.txt chat=1 n=$N"
        echo "name=plain-$P mode=plain $c" >> "$jobs_table"
        echo "name=conttab-d3-$P mode=pipe depth=3 guess=cont copy_min=4 $c trace=$OUT/conttab-d3-$P.trace" >> "$jobs_table"
    done
fi

gpu_pid=""
if [ "$GPU_LOG" = 1 ] && command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
        > "$OUT/gpu.csv" 2>/dev/null &
    gpu_pid=$!
fi

run() { # jobs file, log, extra args
    # shellcheck disable=SC2086
    env ${ENVS:-} CUDA_VISIBLE_DEVICES="$GPUS" timeout 7200 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on -c "$CTX" \
        ${TS:+-ts $TS} ${EXTRA_ARGS:-} $SAMPLING $([ "$BG" = 1 ] && echo --pipe-backend-greedy) \
        --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 8 \
        --pipe-slots $((2*(dmax + 1))) $3 --pipe-jobs "$1" > "$2" 2>&1 &
    local pid=$!
    # the process's CPU use each second (utime + stime from /proc, in % of one core)
    (
        hz=$(getconf CLK_TCK); prev=""
        while kill -0 "$pid" 2>/dev/null; do
            t=$(awk '{print $14 + $15}' "/proc/$pid/stat" 2>/dev/null) || break
            [ -n "$prev" ] && echo "$(date +%s) $(( (t - prev) * 100 / hz ))"
            prev=$t; sleep 1
        done
    ) >> "$OUT/host.csv" &
    wait "$pid"
}

echo "$(date -Is) start $TAG: $(wc -l < "$jobs") jobs, GPUs $GPUS"
run "$jobs" "$OUT/run.log" ""
echo "$(date -Is) exit $? $TAG"
grep -o 'PIPESPEC.*' "$OUT/run.log" > "$OUT/results.txt"
if [ -s "$jobs_table" ]; then
    run "$jobs_table" "$OUT/run-table.log" "--pipe-table $TABLE"
    echo "$(date -Is) table run exit $?"
    grep -o 'PIPESPEC name=conttab.*' "$OUT/run-table.log" >> "$OUT/results.txt"
fi
[ -n "$gpu_pid" ] && kill "$gpu_pid" 2>/dev/null

# per job with a trace: generated tokens per second from its first sampled token to its last
for t in "$OUT"/*.trace; do
    [ -e "$t" ] || continue
    awk -v name="$(basename "$t" .trace)" 'NR == 2 { t0 = $1 } NR > 1 { t1 = $1; n++ }
        END { if (n > 2 && t1 > t0) printf "%s %.3f\n", name, 1e6 * (n - 1) / (t1 - t0) }' "$t"
done > "$OUT/trace_rates.txt"

# summary: speed-up against the plain job of the same prompt, from the traces' rates when both jobs have a trace, else
# from llama-pipespec's tok_s (column basis), and whether tokens and logits match it
awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
FNR == NR { tr[$1] = $2; next }
{ name = f("name"); n = split(name, parts, "-"); prompt = parts[n]
  row[++nr] = $0; nm[nr] = name; pr[nr] = prompt
  rate[nr] = (name in tr) ? tr[name] : f("tok_s"); traced[nr] = (name in tr)
  if (name ~ /^plain-/) { base[prompt] = rate[nr]; btr[prompt] = traced[nr]; btg[prompt] = f("tok_s")
                          tok[prompt] = f("tokens"); lg[prompt] = f("logits") } }
END {
  print "name\tprompt\tguess\tdepth\tn_gen\ttok_s\tspeedup\tbasis\ttok_s_tgen\thits\tmisses\twasted\tavg_inflight\tsame_as_plain"
  for (i = 1; i <= nr; i++) {
    $0 = row[i]; p = pr[i]
    both = traced[i] && btr[p]
    basis = both ? "trace" : "tgen"
    sp = both ? (base[p] > 0 ? sprintf("%.3f", rate[i] / base[p]) : "-") : (btg[p] > 0 ? sprintf("%.3f", f("tok_s") / btg[p]) : "-")
    same = (f("tokens") == tok[p] && f("logits") == lg[p]) ? "yes" : "NO"
    if (nm[i] == "warm") { sp = "-"; same = "-" }
    print nm[i] "\t" p "\t" f("guess") "\t" f("depth") "\t" f("n_gen") "\t" rate[i] "\t" sp "\t" basis "\t" f("tok_s") "\t" f("hits") "\t" f("misses") "\t" f("wasted") "\t" f("avg_inflight_guesses") "\t" same
  }
}' "$OUT/trace_rates.txt" "$OUT/results.txt" > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
