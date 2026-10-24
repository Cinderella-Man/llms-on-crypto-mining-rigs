#!/usr/bin/env bash
# Times races of K samples of one request on the rig (Q93: keep the first sample that closes its reasoning). One process
# runs every job with a single model load, at temperature 1 (top-k 20, top-p 0.95), host sampling:
#   - for each seed in SEEDS: one sample alone (race=1: timed to its "</think>" the same way), then races of K = RACES
#     samples until the first "</think>"; every winner continues with the copy lookup >= 4, else continuation counts
#     (Q96), at depth 3;
#   - each race, the single sample included, runs plainly (samples round robin through llama_decode) and pipelined
#     (one token of every sample in flight); the plain run is the identity check for the pipelined one.
# summary.tsv: per job the race's tokens over all samples, the time until the race ended (t_race), the winner, the
# winner's total tokens, and whether the pipelined race matches the plain race (winner, every sample's tokens and
# logits). Compare t_race of race K with race 1 of the same seed (pipelined): that is the wall-clock gain of the race.
#
# Run it on the rig, from this repository:
#   scp -r bench/pipespec rig:/tmp/
#   ssh rig "setsid nohup env BIN=/opt/llama-builds/<build>/bin MODEL=<gguf> TAG=<name> GPUS=0,1,2,3,4,5 \
#       [ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg'] [RACES='2 3'] [SEEDS='1 2 3'] [PROMPT=short] [N=3072] [STOP='</think>'] \
#       bash /tmp/pipespec/rig_race.sh > /tmp/pipespec/<name>.out 2>&1 < /dev/null &"
# With STOP set to a text that never appears (e.g. @@never@@) every race runs until its first sample reaches N tokens:
# K real sequences, one token each in flight, which measures the staggered rate of K sequences with host sampling.
# Output: /tmp/pipespec/out/<TAG>/{jobs.txt,run.log,results.txt,summary.tsv,*.trace}.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${TAG:?set TAG}"
HERE=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5}
RACES=${RACES:-2 3}
SEEDS=${SEEDS:-1 2 3}
PROMPT=${PROMPT:-short}
N=${N:-3072}
STOP=${STOP:-</think>}
CTX=${CTX:-16384}
OUT=$HERE/out/$TAG
mkdir -p "$OUT"

kmax=1
for K in $RACES; do [ "$K" -gt "$kmax" ] && kmax=$K; done

jobs=$OUT/jobs.txt
{
    echo "name=warm mode=plain prompt=$HERE/prompts/chat.txt chat=1 n=64"
    for S in $SEEDS; do
        c="prompt=$HERE/prompts/$PROMPT.txt chat=1 n=$N race_stop=$STOP"
        # the seed is set per process; a job's samples use seed, seed + 1, ... so seeds are spread with an offset
        for K in 1 $RACES; do
            echo "name=s$S-race$K-plain mode=plain race=$K $c"
            echo "name=s$S-race$K-pipe mode=pipe race=$K depth=3 guess=cont copy_min=4 $c trace=$OUT/s$S-race$K-pipe.trace"
        done
    done
} > "$jobs"

echo "$(date -Is) start $TAG: $(wc -l < "$jobs") jobs, GPUs $GPUS"
# one process per seed, since the seed is a process setting
: > "$OUT/results.txt"
for S in $SEEDS; do
    grep -E "^name=(warm|s$S-)" "$jobs" > "$OUT/jobs-s$S.txt"
    # shellcheck disable=SC2086
    env ${ENVS:-} CUDA_VISIBLE_DEVICES="$GPUS" timeout 7200 "$BIN/llama-pipespec" -m "$MODEL" -ngl 99 -fa on -c "$CTX" \
        --temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed $((S * 100)) --pipe-slots $((2 * kmax + 8)) \
        --pipe-jobs "$OUT/jobs-s$S.txt" > "$OUT/run-s$S.log" 2>&1
    echo "$(date -Is) seed $S exit $?"
    grep -o 'PIPESPEC.*' "$OUT/run-s$S.log" >> "$OUT/results.txt"
done

awk '
function f(k,   i, a) { for (i = 1; i <= NF; i++) { split($i, a, "="); if (a[1] == k) return a[2] } return "" }
{ name = f("name"); row[NR] = $0; nm[NR] = name
  if (name ~ /-plain$/) { key = name; sub(/-plain$/, "", key); w[key] = f("winner"); rh[key] = f("race_hash"); tk[key] = f("tokens") } }
END {
  print "name\trace\tmode\tn_gen\ttok_s\trace_tokens\tt_race\trace_tok_s\twinner\tsame_as_plain"
  for (i = 1; i <= NR; i++) {
    $0 = row[i]; key = nm[i]; sub(/-(plain|pipe)$/, "", key)
    rts = (f("t_race") > 0) ? sprintf("%.2f", f("race_tokens") / f("t_race")) : "-"
    same = "-"
    if (nm[i] ~ /-pipe$/ && (key in tk)) same = (f("winner") == w[key] && f("race_hash") == rh[key] && f("tokens") == tk[key]) ? "yes" : "NO"
    print nm[i] "\t" f("race") "\t" f("mode") "\t" f("n_gen") "\t" f("tok_s") "\t" f("race_tokens") "\t" f("t_race") "\t" rts "\t" f("winner") "\t" same
  }
}' "$OUT/results.txt" > "$OUT/summary.tsv"
column -t -s "$(printf '\t')" "$OUT/summary.tsv"
