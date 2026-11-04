#!/usr/bin/env bash
# Q688 on a build machine, CPU only: the llama-server dump of speculative verify steps (LLAMA_SPEC_DUMP, fork branch
# pascal-p106-v3-b12076-specdump) and bench/tree_replay, without a GPU or a real model:
#   1. build    llama-server, test-spec-dump and test-llama-archs from $WORK/src, CPU backend only
#   2. unit     test-spec-dump: the DFlash2 lattice reduction on synthetic lattices with known answers
#   3. selftest bench/tree_replay/tree_replay.py --selftest (hand-made and calibrated synthetic dumps)
#   4. smoke    a random-weight qwen4exp (Flash-Next's architecture, from test-llama-archs) as target and as its own
#               draft-simple drafter: the same requests to a server with LLAMA_SPEC_DUMP and to one without; the
#               dump must pass tree_replay.py --check (the drafts, the target's tokens, the accepted lengths and the
#               positions agree), carry the candidates, the target's rows and the context match, and both servers
#               must return the same tokens (the dump changes nothing it records)
# Usage: WORK=<dir whose src/ holds the branch> REPO=<this repository> [JOBS=6] [THREADS=8] [PORT=18090] \
#   [LOCK=<file>] [DOCKER_ARGS=...] bash cpu_check.sh
#   LOCK   a file to flock around the build, when several sessions share the build machine
# Output: $WORK/cpu/{bin,models,unit,selftest,smoke}/, $WORK/logs/, $WORK/cpu/summary.tsv (check, verdict, detail);
# done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
PORT=${PORT:-18090}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
mkdir -p "$WORK/logs" "$C/bin" "$C/models" "$C/unit" "$C/selftest" "$C/smoke"
rm -f "$C/DONE"
printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
echo "=== cpu check started $(date -Is)"

build() {
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        git log --oneline -1
        cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
            -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server test-spec-dump test-llama-archs \
            > /work/logs/cpu-build.log 2>&1 || { grep -E "error" /work/logs/cpu-build.log | head -40; exit 1; }
        cp build-cpu/bin/llama-server build-cpu/bin/test-spec-dump build-cpu/bin/test-llama-archs /work/cpu/bin/
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build
export WORK JOBS IMAGE D

echo "=== 1. build $(date -Is)"
locked bash -c build > "$WORK/logs/cpu-build.console" 2>&1
rc=$?
if [ $rc != 0 ]; then
    verdict build FAIL "rc $rc; $(tail -n 5 "$WORK/logs/cpu-build.console" | tr '\n' ' ')"
    date -Is > "$C/DONE"; exit 1
fi
verdict build OK "$(head -n 1 "$WORK/logs/cpu-build.console")"

echo "=== 2. unit $(date -Is)"
docker run --rm $D -v "$C/bin:/x:ro" "$IMAGE" /x/test-spec-dump > "$C/unit/test-spec-dump.log" 2>&1
rc=$?
verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$C/unit/test-spec-dump.log")"

echo "=== 3. selftest $(date -Is)"
docker run --rm $D -v "$REPO/bench/tree_replay:/t:ro" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3 > /dev/null
    python3 -I /t/tree_replay.py --selftest' > "$C/selftest/selftest.log" 2>&1
rc=$?
verdict selftest "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -c '^ok' "$C/selftest/selftest.log") ok, $(grep -c '^FAIL' "$C/selftest/selftest.log") failed"

echo "=== 4. smoke $(date -Is)"
docker run --rm $D -v "$C:/c" -v "$REPO/bench/tree_replay:/t:ro" -e THREADS="$THREADS" -e PORT="$PORT" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
    set -u
    cd /c
    rm -f models/*.gguf
    /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models > /c/models/save.log 2>&1
    M=$(ls /c/models/qwen4exp-moe.gguf 2>/dev/null || ls /c/models/qwen4exp-*.gguf | head -1)
    echo "model $M"
    # deterministic token lists below the vocabulary size (a Park-Miller generator)
    toks() { awk -v n="$1" -v s="$2" -v v="$3" "BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf \"%s%d\", (i ? \",\" : \"\"), 1 + x % (v - 1) } }"; }
    serve() { # <arm> [env...]
        local arm=$1; shift
        rm -f /c/smoke/$arm.dump.jsonl
        env "$@" /c/bin/llama-server -m "$M" -md "$M" --spec-type draft-simple --spec-draft-n-max 4 -ngl 0 -ngld 0 \
            -t "$THREADS" -c 4096 -np 1 --no-webui --host 127.0.0.1 --port "$PORT" > /c/smoke/$arm.server.log 2>&1 &
        echo $! > /c/smoke/$arm.pid
        for _ in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" | grep -q "\"ok\"" && return 0; sleep 1; done
        echo "$arm: server did not start"; tail -20 /c/smoke/$arm.server.log; return 1
    }
    stop() { kill "$(cat /c/smoke/$1.pid)" 2>/dev/null; sleep 2; }
    req() { # <arm> <name> <tokens> <n_predict> <temperature> <seed>
        printf "{\"prompt\":[%s],\"n_predict\":%s,\"temperature\":%s,\"top_k\":20,\"top_p\":0.95,\"min_p\":0,\"seed\":%s,\"return_tokens\":true,\"cache_prompt\":true}" "$3" "$4" "$5" "$6" > /tmp/req.json
        curl -s --max-time 600 -H "Content-Type: application/json" --data @/tmp/req.json "http://127.0.0.1:$PORT/completion" > /c/smoke/$1.$2.json
        jq -c "{n: (.tokens | length), draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}" /c/smoke/$1.$2.json
    }
    run_arm() { # <arm> [env...]
        local arm=$1
        serve "$@" || return 1
        nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r ".data[0].meta.n_vocab")
        a=$(toks 40 3 "$nv"); b=$(toks 30 5 "$nv"); c=$(toks 20 9 "$nv")
        P1="$a,$b,$a,$c"   # a block repeated: the context match has something to find
        echo "--- $arm (n_vocab $nv)"
        req "$arm" R1 "$P1" 48 0 1
        P3="$P1,$(jq -r ".tokens | map(tostring) | join(\",\")" /c/smoke/$arm.R1.json),$b"
        req "$arm" R2 "$P1" 48 1.0 42
        req "$arm" R3 "$P3" 32 0 1
        stop "$arm"
    }
    run_arm dump LLAMA_SPEC_DUMP=/c/smoke/dump.dump.jsonl LLAMA_SPEC_DUMP_K=4 LLAMA_SPEC_DUMP_TGT_K=4 LLAMA_SPEC_DUMP_NGRAM=8
    run_arm plain
    echo "--- same tokens with and without the dump"
    for r in R1 R2 R3; do
        if cmp -s <(jq -c .tokens /c/smoke/dump.$r.json) <(jq -c .tokens /c/smoke/plain.$r.json); then echo "$r same"; else echo "$r DIFFER"; fi
    done
    echo "--- dump"
    wc -l < /c/smoke/dump.dump.jsonl | sed "s/^/lines /"
    grep -m 1 "LLAMA_SPEC_DUMP" /c/smoke/dump.server.log
    jq -s -c "{steps: length, types: (map(.type) | unique), with_cand: (map(select(.cand != null)) | length), with_tgt: (map(select((.tgt_ids | length) > 0)) | length), with_ng: (map(select(.ng_len > 0)) | length), drafted: (map(.draft | length) | add), accepted: (map(.n_acc) | add)}" /c/smoke/dump.dump.jsonl
    head -n 1 /c/smoke/dump.dump.jsonl | cut -c 1-600
    echo "--- check"
    python3 -I /t/tree_replay.py --check /c/smoke/dump.dump.jsonl; echo "check rc=$?"
    python3 -I /t/tree_replay.py /c/smoke/dump.dump.jsonl --out /c/smoke/replay --budgets 1,2,3,4 > /dev/null; echo "replay rc=$?"
    chown -R "$HOST_UID:$HOST_GID" /c' > "$C/smoke/console.txt" 2>&1
cat "$C/smoke/console.txt"
nsame=$(grep -c '^R[123] same' "$C/smoke/console.txt")
crc=$(grep -o 'check rc=[0-9]*' "$C/smoke/console.txt" | cut -d= -f2)
lines=$(grep -o '^lines [0-9]*' "$C/smoke/console.txt" | cut -d' ' -f2)
ncand=$(grep -o '"with_cand":[0-9]*' "$C/smoke/console.txt" | cut -d: -f2)
verdict smoke "$([ "$nsame" = 3 ] && [ "${crc:-1}" = 0 ] && [ "${lines:-0}" -gt 0 ] && [ "${ncand:-0}" = "${lines:-x}" ] && echo OK || echo FAIL)" \
    "$nsame/3 requests same tokens with and without the dump; $lines lines, $ncand with candidates; check rc ${crc:-none}; $(grep -m 1 '^[0-9]* steps' "$C/smoke/console.txt")"

cat "$C/summary.tsv"
date -Is > "$C/DONE"
echo "=== cpu check ended $(date -Is)"
