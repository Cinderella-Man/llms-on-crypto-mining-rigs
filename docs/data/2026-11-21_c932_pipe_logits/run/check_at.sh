#!/usr/bin/env bash
# C932 (2026-11-21): does pipelined decoding give plain decoding's logits on the CPU at one fork commit?
# 1. builds the commit's llama-pipespec, llama-pipe-recurrent-check and test-llama-archs on the CPU with cpu_tests.sh
#    (docs/data/2026-11-05_c565_ladder4fix/run/; GGML_CUDA=OFF, shared libraries, ccache, the build under LOCK) and
#    copies the binaries and libraries to $W/bin/<commit>/;
# 2. makes each architecture of ARCHS as a random model with that build's test-llama-archs (seed 1, a 4,096-token
#    vocabulary), so each commit is checked on a model its own code writes;
# 3. runs bench/pipespec/check_identity.sh on each model (CHAT=0, greedy, N tokens, depth 1, MISSES; plain, pipe-d0, the
#    oracle and the four guessers), and with PRC=1 llama-pipe-recurrent-check (greedy, N tokens).
# The commit is good when every check gives every hash the same. Exit: 0 good, 1 bad (a check differs), 125 untestable
# (no build, no model, or a tool that failed before comparing), for git bisect run.
# Usage: W=<work dir> SRC=<llama.cpp clone holding the commit> REPO=<folder with cpu_tests.sh and pipespec/ (bench/pipespec)>
#   [STAGE=<label for verdicts.tsv>] [ARCHS='qwen4exp'] [MISSES='0'] [N=32] [PRC=0] [THREADS=6] [JOBS=8] [LOCK=<flock file>]
#   [CHECK_ARGS='-ngl 0 -t 6 -c 2048 -fa on'] [CHECK_ENV='LLAMA_GDN_FUSE=0 ...' (variables for the checks)] [IMAGE=pq-c565-cpu:24.04] bash check_at.sh <commit>
# Every container: labels pascal-quants=1 and c932=1, --cpu-shares=256, --memory=16g, no network.
# Output: $W/at/<commit>/ (gen-<arch>.log, <arch>/ check_identity's folder, <arch>.log its table, prc-<arch>.log),
#   one line per architecture in $W/verdicts.tsv (time, stage, commit, subject, arch, verdict, pipe-d0's columns).
set -uo pipefail
C=${1:?commit}
W=${W:?W=<work dir>}
SRC=${SRC:?SRC=<llama.cpp clone>}
REPO=${REPO:?REPO=<scripts folder>}
STAGE=${STAGE:-check}
ARCHS=${ARCHS:-qwen4exp}
MISSES=${MISSES:-0}
N=${N:-32}
PRC=${PRC:-0}
THREADS=${THREADS:-6}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
CHECK_ARGS=${CHECK_ARGS:--ngl 0 -t $THREADS -c 2048 -fa on}
CHECK_ENV=${CHECK_ENV:-}
SFX=${CHECK_ENV:+-$(echo "$CHECK_ENV" | tr " =" "_-")} # the checks' folders and logs with CHECK_ENV
LBL="--label pascal-quants=1 --label c932=1"
MEM="--memory=16g --memory-swap=16g"
TARGETS=llama-pipespec,llama-pipe-recurrent-check,test-llama-archs
full=$(git -C "$SRC" rev-parse "$C^{commit}") || exit 125
short=${full:0:9}
subj=$(git -C "$SRC" log -1 --format=%s "$full" | cut -c1-90)
A=$W/at/$short
BB=$W/bin/$short
mkdir -p "$A" "$W/cpu"
[ -f "$W/verdicts.tsv" ] || printf 'time\tstage\tcommit\tsubject\tarch\tverdict\tpipe-d0 tokens logits state probe\n' > "$W/verdicts.tsv"
rec() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%T)" "$STAGE" "$short" "$subj" "$1${CHECK_ENV:+ $CHECK_ENV}" "$2" "$3" | tee -a "$W/verdicts.tsv"; }
box() { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -v "$REPO:$REPO:ro" \
    -e LD_LIBRARY_PATH="$BB" "$IMAGE" "$@"; }
echo "=== $short $subj ($STAGE) $(date -Is)"

# a tree without the pipelined tools (an upstream commit inside a merge) cannot be checked
git -C "$SRC" cat-file -e "$full:examples/pipespec/pipespec.cpp" 2> /dev/null || { rec - SKIP "no examples/pipespec"; exit 125; }

if [ ! -x "$BB/llama-pipespec" ]; then
    WORK="$W/cpu" SRC="$SRC" TREES="$short=$full:$TARGETS" STEPS="tree build" JOBS="$JOBS" LOCK="${LOCK:-}" \
        DOCKER_ARGS="$LBL $MEM" IMAGE="$IMAGE" KEEP_BUILD=1 bash "$REPO/cpu_tests.sh" > "$A/build-driver.log" 2>&1
    cp "$W/cpu/$short/build.log" "$A/build.log" 2> /dev/null
    B=$W/cpu/tree/build/bin
    for t in ${TARGETS//,/ }; do [ -x "$B/$t" ] || { rec - SKIP "build: no $t ($(grep -acE 'error:' "$A/build.log") error lines)"; exit 125; }; done
    rm -rf "$BB" && mkdir -p "$BB"
    cp -a "$B"/*.so* "$BB/" && cp -a "$B"/{llama-pipespec,llama-pipe-recurrent-check,test-llama-archs} "$BB/"
fi

bad=0 err=0
for a in $ARCHS; do
    rm -rf "$A/models-$a" && mkdir -p "$A/models-$a"
    box bash -c "cd $A/models-$a && LLAMA_TEST_ARCHS_N_VOCAB=4096 $BB/test-llama-archs -a '^$a\$' -s 1 -o . > $A/gen-$a.log 2>&1"
    m=$(ls "$A/models-$a/$a"-moe.gguf "$A/models-$a/$a"*.gguf 2> /dev/null | head -n 1)
    [ -n "$m" ] || { rec "$a" ERR "no model: $(tail -n 1 "$A/gen-$a.log")"; err=1; continue; }
    md5=$(md5sum < "$m" | cut -c1-12)
    box env $CHECK_ENV BIN="$BB" MODEL="$m" CHAT=0 SPEED=0 TIMEOUT=900 PROMPT="$REPO/pipespec/prompts/rewrite.txt" OUT="$A/$a$SFX" \
        ARGS="$CHECK_ARGS" N="$N" DEPTHS=1 MISSES="$MISSES" bash "$REPO/pipespec/check_identity.sh" > "$A/$a$SFX.log" 2>&1
    rc=$?
    d0=$(awk '$1 == "pipe-d0" { print $8, $9, $10, $11 }' "$A/$a$SFX.log")
    if [ $rc = 0 ]; then v=SAME
    elif grep -q '^plain ' "$A/$a$SFX.log" && grep -q ' DIFF$' "$A/$a$SFX.log"; then v=DIFF; bad=1
    else v=ERR; err=1; fi
    rec "$a" "$v" "${d0:-none}; pipelined runs differing $(grep -c " DIFF$" "$A/$a$SFX.log") of $(grep -vc "^plain " <(grep -E " (same|DIFF)$" "$A/$a$SFX.log")); model $(basename "$m") md5 $md5"
    if [ "$PRC" = 1 ]; then
        box env $CHECK_ENV timeout 900 "$BB/llama-pipe-recurrent-check" -m "$m" -f "$REPO/pipespec/prompts/rewrite.txt" -n "$N" \
            $CHECK_ARGS > "$A/prc-$a$SFX.log" 2>&1
        prc=$?
        rec "$a" "prc rc=$prc" "$(grep -a -o "PIPE-RECURRENT-CHECK.*" "$A/prc-$a$SFX.log" | tail -n 1 | cut -c1-200)"
    fi
done
[ $bad = 1 ] && exit 1
[ $err = 1 ] && exit 125
exit 0
