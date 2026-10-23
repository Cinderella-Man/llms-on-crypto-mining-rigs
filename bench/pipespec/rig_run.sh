#!/usr/bin/env bash
# Times plain against pipelined decoding with llama-pipespec on the rig. One process per model and card set runs every
# job with a single model load (a load from a USB stick takes minutes):
#   - a warm-up;
#   - for each prompt (chat, whole-file edit, rename-and-rewrite): plain decoding (the reference), the pipelined API
#     without guesses (its own overhead), the copy lookup (match >= 2) at depths DEPTHS, and at depth 3 the copy lookup
#     with match >= 4, ngram-simple 4/16 and ngram-map-k 4/8;
#   - on the chat prompt, an oracle that replays the plain output: all guesses right at every depth (the pipeline's
#     ceiling: one token per slowest stage), and 30 % / 100 % of them wrong at depth 3 (the cost of wrong guesses).
# Every pipelined job is checked against the plain job of its prompt: same tokens and bit-identical logits.
#
# Run it on the rig, from this repository:
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5 \
#       [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1'] [TS=6,6,6,6,6,6,6,5,2] bash /tmp/pipespec/rig_run.sh \
#       > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# Optional: N (tokens per job, 512), DEPTHS ('1 2 3 4 5'), PROMPTS ('chat edit rewrite'), CTX (8192),
# BG (1: greedy sampling on the last card in both modes, as llama-server's backend sampling does, so only a token id
# crosses the link instead of 0.6 MB of logits; 0: logits to the host and sampling there), SAMPLING ('--temp 0'),
# EXTRA_ARGS. Output: /tmp/pipespec/out/<TAG>/{jobs.txt,run.log,results.txt,summary.tsv,*.trace}; summarize the traces
# with bench/pipespec/traces.py.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5}
N=${N:-512}
DEPTHS=${DEPTHS:-1 2 3 4 5}
PROMPTS=${PROMPTS:-chat edit rewrite}
CTX=${CTX:-8192}
SAMPLING=${SAMPLING:---temp 0}
BG=${BG:-1}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

dmax=3
for D in $DEPTHS; do [ "$D" -gt "$dmax" ] && dmax=$D; done

jobs=$OUT/jobs.txt
{
    echo "name=warm mode=plain prompt=$HERE/prompts/chat.txt chat=1 n=64"
    for P in $PROMPTS; do
        c="prompt=$HERE/prompts/$P.txt chat=1 n=$N"
        echo "name=plain-$P mode=plain $c trace=$OUT/plain-$P.trace"
        echo "name=d0-$P mode=pipe depth=0 guess=none $c trace=$OUT/d0-$P.trace"
        for D in $DEPTHS; do
            echo "name=copy2-d$D-$P mode=pipe depth=$D guess=copy copy_min=2 $c trace=$OUT/copy2-d$D-$P.trace"
        done
        echo "name=copy4-d3-$P mode=pipe depth=3 guess=copy copy_min=4 $c"
        echo "name=simple-d3-$P mode=pipe depth=3 guess=ngram-simple $c"
        echo "name=mapk-d3-$P mode=pipe depth=3 guess=spec spec=ngram-map-k $c"
    done
    P=$(echo "$PROMPTS" | awk '{print $1}')
    c="prompt=$HERE/prompts/$P.txt chat=1 n=$N"
    for D in $DEPTHS; do
        echo "name=oracle-d$D-m0-$P mode=pipe depth=$D guess=oracle oracle=@plain-$P miss=0 $c trace=$OUT/oracle-d$D-m0-$P.trace"
    done
    echo "name=oracle-d3-m0.3-$P mode=pipe depth=3 guess=oracle oracle=@plain-$P miss=0.3 $c trace=$OUT/oracle-d3-m0.3-$P.trace"
    echo "name=oracle-d3-m1-$P mode=pipe depth=3 guess=oracle oracle=@plain-$P miss=1 $c trace=$OUT/oracle-d3-m1-$P.trace"
} > "$jobs"

echo "$(date -Is) start $TAG: $(wc -l < "$jobs") jobs, GPUs $GPUS"
# shellcheck disable=SC2086
env ${ENVS:-} CUDA_VISIBLE_DEVICES="$GPUS" timeout 7200 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on -c "$CTX" \
    ${TS:+-ts $TS} ${EXTRA_ARGS:-} $SAMPLING $([ "$BG" = 1 ] && echo --pipe-backend-greedy) \
    --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 16 \
    --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 8 \
    --pipe-slots $((2*(dmax + 1))) --pipe-jobs "$jobs" > "$OUT/run.log" 2>&1
echo "$(date -Is) exit $? $TAG"
grep -o 'PIPESPEC.*' "$OUT/run.log" > "$OUT/results.txt"

# summary: speed-up against the plain job of the same prompt, and whether tokens and logits match it
awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
{ name = f("name"); n = split(name, parts, "-"); prompt = parts[n]
  row[NR] = $0; nm[NR] = name; pr[NR] = prompt
  if (name ~ /^plain-/) { base[prompt] = f("tok_s"); tok[prompt] = f("tokens"); lg[prompt] = f("logits") } }
END {
  print "name\tprompt\tguess\tdepth\tn_gen\ttok_s\tspeedup\thits\tmisses\twasted\tavg_inflight\tsame_as_plain"
  for (i = 1; i <= NR; i++) {
    $0 = row[i]; p = pr[i]
    sp = (base[p] > 0) ? sprintf("%.3f", f("tok_s") / base[p]) : "-"
    same = (f("tokens") == tok[p] && f("logits") == lg[p]) ? "yes" : "NO"
    if (nm[i] == "warm") { sp = "-"; same = "-" }
    print nm[i] "\t" p "\t" f("guess") "\t" f("depth") "\t" f("n_gen") "\t" f("tok_s") "\t" sp "\t" f("hits") "\t" f("misses") "\t" f("wasted") "\t" f("avg_inflight_guesses") "\t" same
  }
}' "$OUT/results.txt" > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
