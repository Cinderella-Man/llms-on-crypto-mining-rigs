#!/usr/bin/env bash
# Q687 on a build machine, CPU only: llama-server's draft length chosen per step (LLAMA_SPEC_ADAPT, fork branch
# pascal-p106-v3-b12076-adraft), without a GPU or a real model:
#   1. build  llama-server, test-spec-adapt and test-llama-archs from $WORK/src, and llama-server of the base
#             ($BASE_REF, the branch's parent) from a worktree $WORK/src-base; CPU backend, static
#   2. unit   test-spec-adapt: the rule against an exhaustive search of its objective, hand cases, the parser
#   3. smoke  a random-weight qwen4exp (Flash-Next's architecture, from test-llama-archs) as target and as its own
#             draft-simple drafter (--spec-draft-n-max 4), the same three requests to every arm:
#               nospec    the new build without a drafter (plain decoding)
#               base      the base build with the drafter
#               off       the new build, no LLAMA_SPEC_ADAPT* set
#               log       the new build with LLAMA_SPEC_ADAPT_LOG only (the rule recorded, not applied)
#               on        LLAMA_SPEC_ADAPT=1 at the defaults (Q583's cost table, widths 1/3/5/9/17, S run)
#               on-step   S step on a cheap table (T(k) = 100, 110, ..., 140 ms)
#               on-fixed  S fixed at 0.5, scale 0.5 (keeps more)
#               on8       --spec-draft-n-max 8 on a cheaper table (widths 1/3/5/9)
#               on-all    S fixed at 0.001, scale 0.001: every drafted token pays, so the drafts stay whole (the
#                         base build's steps)
#               bad       a malformed LLAMA_SPEC_ADAPT_COST: the server must refuse to start
#             Requests: R1 greedy (96 tokens), R2 temperature 1.0 (seed 42, 48 tokens; verified by rejection
#             sampling, exact in distribution, so its tokens may depend on the draft lengths), R3 greedy on a
#             longer prompt (64 tokens). The random model's context is 256 tokens and every request stays below
#             it (at the limit the speculative server stops a few tokens before plain decoding, the base build
#             alike).
#             Verdicts: off, log and on-all return the base's tokens and draft counts on all three; every greedy
#             reply of every drafting arm equals nospec's; adapt_log.py --check passes on every log; the on arms
#             choose at least three verify widths between them, and only allowed ones
# Usage: WORK=<dir whose src/ holds the branch> REPO=<this repository> [BASE_REF=a330672c1] [JOBS=6] [THREADS=6]
#   [PORT=18091] [LOCK=<file>] [DOCKER_ARGS=...] [STEPS="build unit smoke"] bash cpu_check.sh
#   STEPS  the stages to run; without build, the binaries of the last build are used
#   LOCK   a file to flock around the build, when several sessions share the build machine
# Output: $WORK/cpu/{bin,bin-base,models,unit,smoke}/, $WORK/cpu/smoke-console.txt, $WORK/logs/,
# $WORK/cpu/summary.tsv (check, verdict, detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BASE_REF=${BASE_REF:-a330672c1}
JOBS=${JOBS:-6}
THREADS=${THREADS:-6}
PORT=${PORT:-18091}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
RUN=$REPO/docs/data/2026-11-04_adaptive_draft/run
mkdir -p "$WORK/logs" "$C/bin" "$C/bin-base" "$C/models" "$C/unit" "$C/smoke"
rm -f "$C/DONE"
printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
STEPS=${STEPS:-build unit smoke}
has() { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
echo "=== cpu check started $(date -Is)"

build() {
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e BASE_REF="$BASE_REF" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        git config --global --add safe.directory /work/src-base
        cfg="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
             -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        cd /work/src
        git log --oneline -1
        cmake -B build-cpu $cfg -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server test-spec-adapt test-llama-archs \
            > /work/logs/cpu-build.log 2>&1 || { grep -E "error" /work/logs/cpu-build.log | head -40; exit 1; }
        cp build-cpu/bin/llama-server build-cpu/bin/test-spec-adapt build-cpu/bin/test-llama-archs /work/cpu/bin/
        [ -d /work/src-base ] || git worktree add -f /work/src-base "$BASE_REF" > /dev/null
        cd /work/src-base
        git checkout -q --force "$BASE_REF"
        git log --oneline -1
        cmake -B build-cpu $cfg -DLLAMA_BUILD_TESTS=OFF > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server \
            > /work/logs/cpu-build-base.log 2>&1 || { grep -E "error" /work/logs/cpu-build-base.log | head -40; exit 1; }
        cp build-cpu/bin/llama-server /work/cpu/bin-base/
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build
export WORK JOBS BASE_REF IMAGE D

if has build; then
    echo "=== 1. build $(date -Is)"
    locked bash -c build > "$WORK/logs/cpu-build.console" 2>&1
    rc=$?
    if [ $rc != 0 ]; then
        verdict build FAIL "rc $rc; $(tail -n 5 "$WORK/logs/cpu-build.console" | tr '\n' ' ')"
        date -Is > "$C/DONE"; exit 1
    fi
    verdict build OK "$(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | tr '\n' ';' | cut -c 1-200)"
else
    verdict build SKIP "the last build's binaries: $(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | tr '\n' ';' | cut -c 1-200)"
fi

if has unit; then
    echo "=== 2. unit $(date -Is)"
    docker run --rm $D -v "$C/bin:/x:ro" -w /tmp "$IMAGE" /x/test-spec-adapt > "$C/unit/test-spec-adapt.log" 2>&1
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 2 "$C/unit/test-spec-adapt.log" | tr '\n' ' ')"
fi

if ! has smoke; then
    cat "$C/summary.tsv"; date -Is > "$C/DONE"; exit 0
fi
echo "=== 3. smoke $(date -Is)"
docker run --rm $D -v "$C:/c" -v "$RUN:/r:ro" -e THREADS="$THREADS" -e PORT="$PORT" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
    set -u
    cd /c
    rm -f models/*.gguf smoke/*
    /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models > /c/models/save.log 2>&1
    M=$(ls /c/models/qwen4exp-moe.gguf 2>/dev/null || ls /c/models/qwen4exp-*.gguf | head -1)
    echo "model $M"
    # deterministic token lists below the vocabulary size (a Park-Miller generator)
    toks() { awk -v n="$1" -v s="$2" -v v="$3" "BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf \"%s%d\", (i ? \",\" : \"\"), 1 + x % (v - 1) } }"; }
    serve() { # <arm> <bin> <server args> [env...]
        local arm=$1 bin=$2 sargs=$3; shift 3
        env "$@" $bin -m "$M" $sargs -ngl 0 -t "$THREADS" -c 4096 -np 1 --no-webui --host 127.0.0.1 --port "$PORT" \
            > /c/smoke/$arm.server.log 2>&1 &
        echo $! > /c/smoke/$arm.pid
        for _ in $(seq 1 120); do
            curl -s "http://127.0.0.1:$PORT/health" | grep -q "\"ok\"" && return 0
            kill -0 "$(cat /c/smoke/$arm.pid)" 2> /dev/null || { echo "$arm: server exited"; return 1; }
            sleep 1
        done
        echo "$arm: server did not start"; tail -20 /c/smoke/$arm.server.log; return 1
    }
    stop() { kill "$(cat /c/smoke/$1.pid)" 2>/dev/null; sleep 2; }
    req() { # <arm> <name> <tokens> <n_predict> <temperature> <seed>
        printf "{\"prompt\":[%s],\"n_predict\":%s,\"temperature\":%s,\"top_k\":20,\"top_p\":0.95,\"min_p\":0,\"seed\":%s,\"return_tokens\":true,\"cache_prompt\":false}" "$3" "$4" "$5" "$6" > /tmp/req.json
        curl -s --max-time 600 -H "Content-Type: application/json" --data @/tmp/req.json "http://127.0.0.1:$PORT/completion" > /c/smoke/$1.$2.json
        echo "$1 $2 $(jq -c "{n: (.tokens | length), draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}" /c/smoke/$1.$2.json)"
    }
    DRAFT="-md $M -ngld 0 --spec-type draft-simple --spec-draft-n-max 4"
    run_arm() { # <arm> <bin> <server args> [env...]
        local arm=$1
        serve "$@" || return 1
        nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r ".data[0].meta.n_vocab")
        a=$(toks 40 3 "$nv"); b=$(toks 30 5 "$nv"); c=$(toks 20 9 "$nv")
        P1="$a,$b,$a,$c"   # a block repeated: something a drafter can copy
        P3="$P1,$b"
        req "$arm" R1 "$P1" 96 0 1
        req "$arm" R2 "$P1" 48 1.0 42
        req "$arm" R3 "$P3" 64 0 1
        grep -h "adaptive draft\|LLAMA_SPEC_ADAPT" /c/smoke/$arm.server.log | sed "s/^/$arm: /" | head -8
        stop "$arm"
    }
    L=LLAMA_SPEC_ADAPT_LOG
    run_arm nospec   /c/bin/llama-server      ""
    run_arm base     /c/bin-base/llama-server "$DRAFT"
    run_arm off      /c/bin/llama-server      "$DRAFT"
    run_arm log      /c/bin/llama-server      "$DRAFT" $L=/c/smoke/log.adapt.jsonl
    run_arm on       /c/bin/llama-server      "$DRAFT" LLAMA_SPEC_ADAPT=1 $L=/c/smoke/on.adapt.jsonl
    run_arm on-step  /c/bin/llama-server      "$DRAFT" LLAMA_SPEC_ADAPT=1 $L=/c/smoke/on-step.adapt.jsonl \
        LLAMA_SPEC_ADAPT_S=step LLAMA_SPEC_ADAPT_COST=100,110,120,130,140
    run_arm on-fixed /c/bin/llama-server      "$DRAFT" LLAMA_SPEC_ADAPT=1 $L=/c/smoke/on-fixed.adapt.jsonl \
        LLAMA_SPEC_ADAPT_S=0.5 LLAMA_SPEC_ADAPT_SCALE=0.5
    run_arm on8      /c/bin/llama-server      "${DRAFT/--spec-draft-n-max 4/--spec-draft-n-max 8}" LLAMA_SPEC_ADAPT=1 \
        $L=/c/smoke/on8.adapt.jsonl LLAMA_SPEC_ADAPT_COST=100,104,108 LLAMA_SPEC_ADAPT_DRAFT_MS=0
    run_arm on-all   /c/bin/llama-server      "$DRAFT" LLAMA_SPEC_ADAPT=1 $L=/c/smoke/on-all.adapt.jsonl \
        LLAMA_SPEC_ADAPT_S=0.001 LLAMA_SPEC_ADAPT_SCALE=0.001
    echo "--- bad"
    if serve bad /c/bin/llama-server "$DRAFT" LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_COST=136,abc; then
        echo "bad STARTED"; stop bad
    else
        echo "bad REFUSED: $(grep -m 1 "LLAMA_SPEC_ADAPT" /c/smoke/bad.server.log)"
    fi

    same() { cmp -s <(jq -c .tokens "$1") <(jq -c .tokens "$2"); }
    counts() { jq -c "[.timings.draft_n, .timings.draft_n_accepted]" "$1"; }
    echo "--- off, log and on-all against base: tokens and draft counts"
    for arm in off log on-all; do
        for r in R1 R2 R3; do
            if same /c/smoke/$arm.$r.json /c/smoke/base.$r.json && [ "$(counts /c/smoke/$arm.$r.json)" = "$(counts /c/smoke/base.$r.json)" ]; then
                echo "$arm $r identical $(counts /c/smoke/$arm.$r.json)"
            else
                echo "$arm $r DIFFER $(counts /c/smoke/$arm.$r.json) base $(counts /c/smoke/base.$r.json)"
            fi
        done
    done
    echo "--- greedy replies against plain decoding (nospec)"
    for arm in base off log on on-step on-fixed on8 on-all; do
        for r in R1 R3; do
            if same /c/smoke/$arm.$r.json /c/smoke/nospec.$r.json; then echo "$arm $r lossless"; else echo "$arm $r NOT-LOSSLESS"; fi
        done
    done
    echo "--- temperature 1.0 (rejection sampling), the on arms against nospec (may differ: exact in distribution)"
    for arm in on on-step on-fixed on8 on-all; do
        same /c/smoke/$arm.R2.json /c/smoke/nospec.R2.json && echo "$arm R2 same as nospec" || echo "$arm R2 other tokens"
    done
    echo "--- logs"
    for arm in log on on-step on-fixed on8 on-all; do
        f=/c/smoke/$arm.adapt.jsonl
        echo "$arm: $(grep -c . $f) lines, widths $(jq -r "select(.hdr == null) | .n_keep + 1" $f | sort -n | uniq -c | tr -s " " | tr "\n" ";")"
        python3 -I /r/adapt_log.py --check $f --csv /c/smoke/$arm.summary.csv > /c/smoke/$arm.check.txt; echo "$arm check rc=$?"
        tail -n 1 /c/smoke/$arm.check.txt
    done
    jq -r "select(.hdr == null) | .n_keep + 1" /c/smoke/on.adapt.jsonl /c/smoke/on-step.adapt.jsonl \
        /c/smoke/on-fixed.adapt.jsonl /c/smoke/on8.adapt.jsonl /c/smoke/on-all.adapt.jsonl | sort -nu | tr "\n" " " | sed "s/^/on widths: /"; echo
    head -n 2 /c/smoke/on.adapt.jsonl | cut -c 1-700
    chown -R "$HOST_UID:$HOST_GID" /c' > "$C/smoke-console.txt" 2>&1
cat "$C/smoke-console.txt"
S=$C/smoke-console.txt
n_ident=$(grep -c '^\(off\|log\|on-all\) R[123] identical' "$S")
n_lossless=$(grep -c '^[a-z0-9-]* R[13] lossless$' "$S")
n_check=$(grep -c '^[a-z0-9-]* check rc=0$' "$S")
n_widths=$(grep -o '^on widths: .*' "$S" | wc -w)
bad=$(grep -c '^bad REFUSED' "$S")
verdict smoke-off "$([ "$n_ident" = 9 ] && echo OK || echo FAIL)" \
    "$n_ident/9 replies of off, log and on-all with the base build's tokens and draft counts"
verdict smoke-lossless "$([ "$n_lossless" = 16 ] && echo OK || echo FAIL)" \
    "$n_lossless/16 greedy replies of the drafting arms equal to plain decoding"
verdict smoke-log "$([ "$n_check" = 6 ] && [ "$n_widths" -ge 5 ] && echo OK || echo FAIL)" \
    "$n_check/6 logs pass adapt_log.py --check; on arms' verify widths: $(grep -o '^on widths: .*' "$S" | cut -d: -f2)"
verdict smoke-bad "$([ "$bad" = 1 ] && echo OK || echo FAIL)" "a malformed cost table $([ "$bad" = 1 ] && echo stops the server || echo did not stop the server)"

cat "$C/summary.tsv"
date -Is > "$C/DONE"
echo "=== cpu check ended $(date -Is)"
