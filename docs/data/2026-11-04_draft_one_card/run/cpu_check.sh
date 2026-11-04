#!/usr/bin/env bash
# Q961 on a build machine, CPU only: the MTP draft's embedding row cache (LLAMA_DRAFT_EMBD_CACHE) and its hidden rows
# kept on the device between passes (LLAMA_DRAFT_H_CARRY), fork branch pascal-p106-v3-b12076-draft1card, against the
# same build with the switches off, on the real files:
#   1. build  llama-server and test-draft-mem from $WORK/src; CPU backend, static
#   2. unit   test-draft-mem: the cache's rows hold the table's bytes for every token, get_rows over them gives the
#             table's numbers exactly (F32, Q8_0, Q3_K), the slot read longest ago gives way but never to a token of the
#             same batch, batches past the size or with ids outside the vocabulary read the table
#   3. files  the target (MODEL) and its MTP draft (DRAFT) on the CPU, one load per arm, the same requests to each:
#               nodraft      plain decoding
#               base         the MTP draft at n-max 3, no switch
#               cache        LLAMA_DRAFT_EMBD_CACHE=8192 (the rig's setting)
#               cache-small  LLAMA_DRAFT_EMBD_CACHE=16,4: rows replaced all the time; batches over 4 tokens read the table
#               carry        LLAMA_DRAFT_H_CARRY=1
#               both         the cache and the carry
#               short        LLAMA_DRAFT_SHORTLIST=32768,8192,2 (the rig's draft head)
#               short-both   the shortlist, the cache and the carry (the rig's d1on line)
#             Requests: P1 and P2 greedy (96 and 64 tokens), P1 sampled (temperature 1.0, top-k 20, top-p 0.95,
#             seed 42: verified by rejection; its draft counts show the drafts)
#             Verdicts: files-identity, cache, cache-small, carry and both return base's tokens and draft counts on all
#             three requests, short-both short's; files-log, the cache arms log the cache's counts and the carry arms the
#             carry (the draft's own log lines: the server does not print the context's); files-plain, every greedy reply
#             of a drafting arm equals nodraft's, INFO instead of OK/FAIL when base's own replies differ (then the
#             verify batch's numerics differ from single-token decodes on this build, Q971, whatever the switches do)
# Usage: WORK=<dir whose src/ holds the branch> MODEL=<target, first shard> DRAFT=<MTP draft> [JOBS=8] [THREADS=8]
#   [PORT=18093] [LOCK=<file>] [DOCKER_ARGS=...] [STEPS="build unit files"] [ARMS="nodraft base cache ..."] bash cpu_check.sh
#   STEPS  the stages to run; without build, the binaries of the last build are used; "verdicts" alone recomputes the
#          files verdicts from the last files run's outputs
#   LOCK   a file to flock around the build, when several sessions share the build machine
# Output: $WORK/cpu/{bin,unit,files}/, $WORK/logs/, $WORK/cpu/summary.tsv (check, verdict, detail), $WORK/cpu/files.tsv
# (arm, request, tokens, draft_n, draft_n_accepted, md5 of the token ids); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
PORT=${PORT:-18093}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
mkdir -p "$WORK/logs" "$C/bin" "$C/unit" "$C/files"
rm -f "$C/DONE"
printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
STEPS=${STEPS:-build unit files}
ARMS=${ARMS:-nodraft base cache cache-small carry both short short-both}
has() { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
echo "=== cpu check started $(date -Is)"

build() {
    docker run --rm $D --label pascal-quants=1 -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        git log --oneline -1
        git status --short | head -20
        cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server test-draft-mem \
            > /work/logs/cpu-build.log 2>&1 || { grep -E "error" /work/logs/cpu-build.log | head -40; exit 1; }
        grep -E "warning" /work/logs/cpu-build.log | grep -E "embd-cache|embd_cache|speculative.cpp|qwen4exp|llama-graph|llama-context" | head -20 || true
        cp build-cpu/bin/llama-server build-cpu/bin/test-draft-mem /work/cpu/bin/
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build
export WORK JOBS IMAGE D

if has build; then
    echo "=== 1. build $(date -Is)"
    locked bash -c build > "$WORK/logs/cpu-build.console" 2>&1
    rc=$?
    if [ $rc != 0 ]; then
        verdict build FAIL "rc $rc; $(tail -n 5 "$WORK/logs/cpu-build.console" | tr '\n' ' ')"
        date -Is > "$C/DONE"; exit 1
    fi
    verdict build OK "$(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | head -1 | cut -c 1-120)"
else
    verdict build SKIP "the last build's binaries: $(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | head -1 | cut -c 1-120)"
fi

if has unit; then
    echo "=== 2. unit $(date -Is)"
    docker run --rm $D --label pascal-quants=1 -v "$C/bin:/x:ro" -w /tmp "$IMAGE" /x/test-draft-mem > "$C/unit/test-draft-mem.log" 2>&1
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^embd cache|checks' "$C/unit/test-draft-mem.log" | tr '\n' ' ' | cut -c 1-400)"
fi

if ! has files && ! has verdicts; then
    cat "$C/summary.tsv"; date -Is > "$C/DONE"; exit 0
fi
if has files; then
echo "=== 3. files $(date -Is)"
: "${MODEL:?MODEL=<target, first shard>}" "${DRAFT:?DRAFT=<MTP draft>}"
MDIR=$(dirname "$(realpath "$MODEL")"); DDIR=$(dirname "$(realpath "$DRAFT")")
rm -f "$C/files/"*
docker run --rm $D --label pascal-quants=1 -v "$C:/c" -v "$MDIR:$MDIR:ro" -v "$DDIR:$DDIR:ro" \
    -e MODEL="$(realpath "$MODEL")" -e DRAFT="$(realpath "$DRAFT")" -e THREADS="$THREADS" -e PORT="$PORT" -e ARMS="$ARMS" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
    set -u
    cd /c/files
    P1="<|im_start|>user
Write a Python function \`merge_intervals(intervals)\` that merges overlapping closed intervals and returns them sorted. Add a short docstring and three asserts.<|im_end|>
<|im_start|>assistant
<think>

</think>

"
    P2="<|im_start|>user
The file src/server.c fails to compile with \"error: implicit declaration of function strndup\". Explain the cause in two sentences, then give the one-line fix.<|im_end|>
<|im_start|>assistant
<think>

</think>

"
    arm_env() {
        case $1 in
            cache)       echo "LLAMA_DRAFT_EMBD_CACHE=8192" ;;
            cache-small) echo "LLAMA_DRAFT_EMBD_CACHE=16,4" ;;
            short)       echo "LLAMA_DRAFT_SHORTLIST=32768,8192,2" ;;
            carry)       echo "LLAMA_DRAFT_H_CARRY=1" ;;
            both)        echo "LLAMA_DRAFT_EMBD_CACHE=8192 LLAMA_DRAFT_H_CARRY=1" ;;
            short-both)  echo "LLAMA_DRAFT_SHORTLIST=32768,8192,2 LLAMA_DRAFT_EMBD_CACHE=8192 LLAMA_DRAFT_H_CARRY=1" ;;
            *)           echo "" ;;
        esac
    }
    req() { # <arm> <name> <prompt> <n_predict> <temperature> <seed>
        jq -n --arg p "$3" --argjson n "$4" --argjson t "$5" --argjson s "$6" \
            "{prompt: \$p, n_predict: \$n, temperature: \$t, top_k: (if \$t == 0 then 1 else 20 end), top_p: 0.95,
              min_p: 0, seed: \$s, return_tokens: true, cache_prompt: false}" |
            curl -s --max-time 1800 -H "Content-Type: application/json" -d @- "http://127.0.0.1:$PORT/completion" > $1.$2.json
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$(jq ".tokens | length" $1.$2.json)" \
            "$(jq ".timings.draft_n // 0" $1.$2.json)" "$(jq ".timings.draft_n_accepted // 0" $1.$2.json)" \
            "$(jq -c .tokens $1.$2.json | md5sum | cut -c 1-12)" | tee -a files.tsv
    }
    printf "arm\trequest\ttokens\tdraft_n\tdraft_n_accepted\ttokens_md5\n" > files.tsv
    for arm in $ARMS; do
        a=(-m "$MODEL" -c 4096 -np 1 -t "$THREADS" -tb "$THREADS" --host 127.0.0.1 --port "$PORT" --no-webui)
        [ "$arm" != nodraft ] && a+=(-md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3)
        echo "== $arm: $(arm_env $arm) ${a[*]}"
        env $(arm_env $arm) /c/bin/llama-server "${a[@]}" > $arm.server.log 2>&1 &
        pid=$!
        up=0
        for _ in $(seq 1 1800); do
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
            kill -0 $pid 2> /dev/null || break
            sleep 1
        done
        if [ $up = 0 ]; then
            echo "$arm: the server did not come up"; tail -n 20 $arm.server.log; kill $pid 2> /dev/null; wait $pid; continue
        fi
        req $arm P1 "$P1" 96 0 1
        req $arm P2 "$P2" 64 0 1
        req $arm P1s "$P1" 64 1.0 42
        kill -TERM $pid 2> /dev/null; wait $pid
        grep -h "embedding row cache\|embd_cache\|draft head shortlist\|hidden row" $arm.server.log | sed "s/^/$arm: /" | cut -c 1-300
    done
    chown -R "$HOST_UID:$HOST_GID" /c' > "$C/files-console.txt" 2>&1
cat "$C/files-console.txt"
cp "$C/files/files.tsv" "$C/files.tsv" 2> /dev/null
fi

T=$C/files.tsv
row() { awk -F'\t' -v a="$1" -v r="$2" '$1 == a && $2 == r { print $3 "/" $4 "/" $5 "/" $6 }' "$T"; }
n_same=0; n_pairs=0; detail=""
for pair in cache:base cache-small:base carry:base both:base short-both:short; do
    x=${pair%%:*}; y=${pair##*:}
    for r in P1 P2 P1s; do
        [ -n "$(row "$x" "$r")" ] || continue
        n_pairs=$((n_pairs + 1))
        if [ "$(row "$x" "$r")" = "$(row "$y" "$r")" ]; then n_same=$((n_same + 1)); else detail="$detail $x/$r differs from $y;"; fi
    done
done
verdict files-identity "$([ $n_pairs -gt 0 ] && [ $n_same = $n_pairs ] && echo OK || echo FAIL)" \
    "$n_same/$n_pairs replies of the switch arms with the tokens and draft counts of the arm without the switches;$detail"
n_ll=0; n_g=0
for arm in $ARMS; do
    [ "$arm" = nodraft ] && continue
    for r in P1 P2; do
        [ -n "$(row "$arm" "$r")" ] || continue
        n_g=$((n_g + 1))
        [ "$(row "$arm" "$r" | cut -d/ -f4)" = "$(row nodraft "$r" | cut -d/ -f4)" ] && n_ll=$((n_ll + 1))
    done
done
n_base=0
for r in P1 P2; do [ "$(row base "$r" | cut -d/ -f4)" = "$(row nodraft "$r" | cut -d/ -f4)" ] && n_base=$((n_base + 1)); done
if [ $n_base != 2 ]; then v=INFO; else v=$([ $n_g -gt 0 ] && [ $n_ll = $n_g ] && echo OK || echo FAIL); fi
verdict files-plain $v "$n_ll/$n_g greedy replies of the drafting arms equal to plain decoding; base (no switch) $n_base/2"
n_want_c=0; n_want_h=0
for arm in $ARMS; do
    case $arm in cache|cache-small|both|short-both) n_want_c=$((n_want_c + 1)) ;; esac
    case $arm in carry|both|short-both)             n_want_h=$((n_want_h + 1)) ;; esac
done
n_cnt=$(grep -c "^[a-z-]*: .*draft embedding row cache: [0-9]* token rows read from it" "$C/files-console.txt")
n_car=$(grep -c "^[a-z-]*: .*draft hidden rows: a step.s drafting passes keep them on the device" "$C/files-console.txt")
verdict files-log "$([ "$n_cnt" = $n_want_c ] && [ "$n_car" = $n_want_h ] && echo OK || echo FAIL)" \
    "$n_cnt/$n_want_c cache arms logged the cache's counts, $n_car/$n_want_h carry arms the carry"
cat "$C/summary.tsv"
date -Is > "$C/DONE"
echo "=== cpu check ended $(date -Is)"
