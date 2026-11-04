#!/usr/bin/env bash
# Q897's checks of llama-imatrix --span-mask on a CPU, no GPU needed: a random qwen4exp model from test-llama-archs
# (32 routed experts, 8 used, a 4,096-token test vocabulary) over span_mask.py's check text (a fixed 8-token sentence
# after random fillers), 4 chunks of 512 tokens, every binary in a container of the build image (build.sh's cpu step).
#   all      every token in class 1: class 1's matrix is the usual one bit for bit, class 0's is zero, and the usual
#            matrix is the one a run without a mask writes
#   none     every token in class 0: the other way round
#   random   classes 0-3 at random: the four matrices add up to the usual one (counts exactly, sums within 1e-4), each
#            counts its own tokens and 8 routings per token
#   chunk    class 1 = chunk 3 of 4: its counts are a 4-chunk run's minus a 3-chunk run's, and class 0 is the 3-chunk
#            run's matrix bit for bit (the right chunk, the right tokens)
#   ngram    class 1 = the last token of every repeat of the sentence (the same 8 tokens), class 2 = the token after
#            each: layer 0 reads a token and its 5 predecessors (qwen4exp's n-gram embeddings), so its projections'
#            input has no variance over class 1's rows and some over class 2's (the right tokens within a chunk); also
#            with -b 256 -ub 256 (two batches per chunk)
#   refuse   a mask with one token id changed, and a batch split over two ubatches (-b 512 -ub 256): both refused
#   analysis the random classes as call/argument spans: span_counts.py against a ranking from the usual matrix,
#            force_keep.py at 24 of 32 experts (experts with 1 % of the spans, at most 2 a layer), then scripts/gguf_prune.py with --force-ids
#            on the model: its kept ids are the ranking's 24 with the forced ones in, and `verify` passes
# Env: WORK, BIN (build.sh's cpu/bin), REPO (a checkout of this repository: span_mask.py, span_check.py and
#      scripts/gguf_reshard.py), THREADS (8), IMAGE, MOUNTS (more folders to mount, at their own paths), DOCKER_ARGS
# Writes $WORK/random, $WORK/checks/ (the check text, the dump, masks, matrices, logs), one verdict line per check in
# $WORK/checks.tsv.
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${REPO:?}"
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
R=$REPO/docs/data/2026-11-04_toolcall_experts/run
D="--rm --cpu-shares 256 --label pascal-quants=1 ${DOCKER_ARGS:-}"
mounts="-v $WORK:$WORK -v $BIN:$BIN"
for d in ${MOUNTS:-}; do mounts="$mounts -v $d:$d"; done
M=$WORK/random K=$WORK/checks
TEXT=$K/text.txt
mkdir -p "$M" "$K"
touch "$WORK/checks.tsv"

log()     { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
box()     { docker run $D $mounts "$IMAGE" "$@"; }
py()      { nice -n 19 python3 -I "$@"; }
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/checks.tsv"; }
# check <name> <span_check.py args>: its verdict line
check()   { local out; out=$(py "$R/span_check.py" "${@:2}" 2>&1); verdict "$1" "${out%% *}" "${out#* }"; }
# imx <out> <extra args>: llama-imatrix over 4 chunks of 512 tokens (unless the args say otherwise)
imx()     { local o=$1; shift
            box nice -n 19 "$BIN/llama-imatrix" -m "$MODEL" -f "$TEXT" -c 512 -b 512 --chunks 4 -t "$THREADS" \
                --no-ppl -ofreq 1000 -o "$K/$o.gguf" "$@" > "$K/$o.log" 2>&1; }

MODEL=$M/qwen4exp-moe.gguf
if [ ! -f "$MODEL" ]; then
    log "random model"
    docker run $D $mounts -e LLAMA_TEST_ARCHS_N_VOCAB=4096 -e LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$IMAGE" \
        "$BIN/test-llama-archs" -a qwen4exp -o "$M" > "$M/archs.log" 2>&1
fi
[ -f "$MODEL" ] || { verdict model FAIL "no $MODEL ($M/archs.log)"; exit 1; }

log "text, dump and masks"
py "$R/span_mask.py" text "$TEXT"
box "$BIN/llama-imatrix" -m "$MODEL" -f "$TEXT" --span-dump "$K/dump.tsv" > "$K/dump.log" 2>&1
[ -s "$K/dump.tsv" ] || { verdict dump FAIL "$K/dump.log"; exit 1; }
for kind in all none ngram random; do py "$R/span_mask.py" synth "$K/dump.tsv" --kind $kind --out "$K/mask-$kind.txt" > "$K/mask-$kind.log"; done
py "$R/span_mask.py" synth "$K/dump.tsv" --kind chunk --chunk 3 --out "$K/mask-chunk.txt" > "$K/mask-chunk.log"
verdict dump OK "$(head -n 1 "$K/dump.tsv" | tr '\t' ' '); $(cat "$K/mask-ngram.log")"

log "runs"
imx base
imx base3 --chunks 3
for kind in all none ngram random chunk; do imx "$kind" --span-mask "$K/mask-$kind.txt"; done
imx ngramb --span-mask "$K/mask-ngram.txt" -b 256 -ub 256
for o in base base3 all none ngram random chunk ngramb; do
    [ -f "$K/$o.gguf" ] || verdict "run $o" FAIL "$K/$o.log"
done

check "all: usual = no mask" same "$K/base.gguf" "$K/all.gguf"
check "all: class 1 = usual" same "$K/base.gguf" "$K/all.span1.gguf"
check "all: class 0 zero" zero "$K/all.span0.gguf"
check "none: class 0 = usual" same "$K/base.gguf" "$K/none.span0.gguf"
check "none: class 1 zero" zero "$K/none.span1.gguf"
check "random: classes add up" sum "$K/base.gguf" "$K"/random.span{0,1,2,3}.gguf
check "ngram: classes add up" sum "$K/base.gguf" "$K"/ngram.span{0,1,2}.gguf
check "chunk: class 1 = 4 - 3 chunks" diff "$K/base.gguf" "$K/base3.gguf" "$K/chunk.span1.gguf"
check "chunk: class 0 = 3 chunks" same "$K/base3.gguf" "$K/chunk.span0.gguf"
check "ngram: one window, no variance" variance "$K/ngram.span1.gguf" --layer 0 --max 1e-4
check "ngram: the next token varies" variance "$K/ngram.span2.gguf" --layer 0 --expect-above 1e-2
check "ngram -b 256: one window, no variance" variance "$K/ngramb.span1.gguf" --layer 0 --max 1e-4
check "ngram -b 256: classes add up" sum "$K/ngramb.gguf" "$K"/ngramb.span{0,1,2}.gguf

# negative controls
out=$(py "$R/span_check.py" same "$K/base.gguf" "$K/base3.gguf" 2>&1)
verdict "control: same tells 3 from 4 chunks" "$([[ $out == FAIL* ]] && echo OK || echo FAIL)" "$out"
out=$(py "$R/span_check.py" variance "$K/ngram.span2.gguf" --layer 0 --max 1e-4 2>&1)
verdict "control: variance sees mixed tokens" "$([[ $out == FAIL* ]] && echo OK || echo FAIL)" "$out"
awk 'NR == 3 { $1 = ($1 + 1) % 4096 } { print }' "$K/mask-all.txt" > "$K/mask-bad.txt"
imx bad --span-mask "$K/mask-bad.txt"; rc=$?
verdict "refuse: a changed token id" "$([ $rc -ne 0 ] && [ ! -f "$K/bad.span0.gguf" ] && echo OK || echo FAIL)" \
    "exit $rc: $(grep -m 1 -o 'token [0-9]* is .*' "$K/bad.log")"
imx split --span-mask "$K/mask-all.txt" -ub 256; rc=$?
verdict "refuse: two ubatches" "$([ $rc -ne 0 ] && echo OK || echo FAIL)" "exit $rc: $(grep -m 1 -o "'--span-mask' needs.*" "$K/split.log" | cut -c1-90)"
log "analysis"
A=$K/analysis
mkdir -p "$A"
py - "$R" "$K/base.gguf" "$A/experts.tsv" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from span_counts import routings
r, _h = routings(sys.argv[2])
with open(sys.argv[3], "w") as f:
    f.write("layer\texpert\troutings\n")
    f.writelines(f"{l}\t{e}\t{v}\n" for l in sorted(r) for e, v in enumerate(r[l]))
PY
py "$R/span_counts.py" --full "$K/base.gguf" --span "$K"/random.span{0,1,2,3}.gguf --experts "$A/experts.tsv" \
    --keep 24,16 --out "$A" > "$A/span_counts.log" 2>&1
rc=$?
verdict "analysis: span_counts" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "exit $rc: $(grep -P '^24\tcall' "$A/span_cut_summary.tsv" | tr '\t' ' ')"
py "$R/force_keep.py" --span-experts "$A/span_experts.tsv.gz" --experts "$A/experts.tsv" --keep 24 --max-per-layer 2 \
    --min-share 1 --min-routings 1 --out "$A" > "$A/force_keep.log" 2>&1
rc=$?
verdict "analysis: force_keep" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "exit $rc: $(head -n 1 "$A/force_keep.log")"
py "$REPO/scripts/gguf_prune.py" prune --keep 24 --experts "$A/experts.tsv" --force-ids "$A/force_24.tsv" \
    --out-dir "$A/pruned" "$MODEL" > "$A/prune.log" 2>&1 \
    && py "$REPO/scripts/gguf_prune.py" verify "$MODEL" --new "$A/pruned/$(basename "$MODEL")" --ids "$A/pruned/kept.tsv" \
        >> "$A/prune.log" 2>&1
rc=$?
forced_in=$(py - "$A/force_24.tsv" "$A/pruned/kept.tsv" <<'PY'
import sys
read = lambda p: {int(c[0]): set(map(int, c[1].split(","))) if len(c) > 1 and c[1] else set()
                  for c in (l.rstrip("\n").split("\t") for l in open(p)) if c[0].isdigit()}
f, k = read(sys.argv[1]), read(sys.argv[2])
print(sum(len(v) for v in f.values()), all(f[l] <= k[l] for l in f), all(len(v) == 24 for v in k.values()))
PY
)
verdict "analysis: gguf_prune --force-ids" "$([ $rc -eq 0 ] && [[ $forced_in == [1-9]*" True True" ]] && echo OK || echo FAIL)" \
    "exit $rc; forced, all kept, 24 a layer: $forced_in; $(tail -n 1 "$A/prune.log")"
log "done: $(cut -f3 "$WORK/checks.tsv" | sort | uniq -c | tr '\n' ' ')"
