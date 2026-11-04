#!/usr/bin/env bash
# Q804: the full-block dump (llama-server's LLAMA_SPEC_DUMP_FULL_BLOCK, fork branch pascal-p106-v3-b12076-specdump2) and
# bench/tree_replay's per-position tables, on a build machine, CPU only, without a GPU or a real model:
#   1. build    llama-server, test-spec-dump and test-llama-archs from $WORK/src, CPU backend only
#   2. unit     test-spec-dump: the DFlash2 lattice reduction, the selector's greedy walk and the full block's record
#               on synthetic lattices with known answers
#   3. selftest bench/tree_replay/tree_replay.py --selftest (the tree tests, and synthetic full-block dumps whose
#               per-position acceptance, conditional acceptance, context and segment splits are known)
#   4. smoke    a random-weight qwen4exp (Flash-Next's architecture) as target and as its own draft-simple drafter,
#               the same requests to three servers: no dump, the dump, and the dump with LLAMA_SPEC_DUMP_FULL_BLOCK=1.
#               All three must return the same tokens, the two dumps must be line for line the same (a drafter that
#               is not DFlash2 has no second block), and the dump must pass tree_replay.py --check
# Usage: WORK=<dir whose src/ holds the branch> REPO=<this repository> [JOBS=6] [THREADS=8] [PORT=18090] \
#   [LOCK=<file>] [DOCKER_ARGS=...] [STAGES="build unit selftest smoke"] bash cpu_check.sh
#   LOCK   a file to flock around the build, when several sessions share the build machine
# Output: $WORK/cpu/{bin,models,unit,selftest,smoke}/, $WORK/logs/, $WORK/cpu/summary.tsv (check, verdict, detail);
# done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
PORT=${PORT:-18090}
STAGES=${STAGES:-build unit selftest smoke}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--label pascal-quants=1 --cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
mkdir -p "$WORK/logs" "$C/bin" "$C/models" "$C/unit" "$C/selftest" "$C/smoke"
rm -f "$C/DONE"
[ -f "$C/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
echo "=== cpu check started $(date -Is) ($STAGES)"

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

if stage build; then
    echo "=== 1. build $(date -Is)"
    locked bash -c build > "$WORK/logs/cpu-build.console" 2>&1
    rc=$?
    if [ $rc != 0 ]; then
        verdict build FAIL "rc $rc; $(tail -n 5 "$WORK/logs/cpu-build.console" | tr '\n' ' ')"
        date -Is > "$C/DONE"; exit 1
    fi
    verdict build OK "$(grep -m 1 -E '^[0-9a-f]{7,12} ' "$WORK/logs/cpu-build.console" | cut -c 1-120); $(grep -c 'warning:' "$WORK/logs/cpu-build.log") warnings, $(grep 'warning:' "$WORK/logs/cpu-build.log" | grep -c -E 'common/speculative|server-context|test-spec-dump') in the changed files"
fi

if stage unit; then
    echo "=== 2. unit $(date -Is)"
    docker run --rm $D -v "$C/bin:/x:ro" "$IMAGE" /x/test-spec-dump > "$C/unit/test-spec-dump.log" 2>&1
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$C/unit/test-spec-dump.log")"
fi

if stage selftest; then
    echo "=== 3. selftest $(date -Is)"
    docker run --rm $D -v "$REPO/bench/tree_replay:/t:ro" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq python3 > /dev/null
        python3 -I /t/tree_replay.py --selftest' > "$C/selftest/selftest.log" 2>&1
    rc=$?
    verdict selftest "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -c '^ok' "$C/selftest/selftest.log") ok, $(grep -c '^FAIL' "$C/selftest/selftest.log") failed"
fi

if stage smoke; then
    echo "=== 4. smoke $(date -Is)"
    docker run --rm $D -v "$C:/c" -v "$REPO/bench/tree_replay:/t:ro" -e THREADS="$THREADS" -e PORT="$PORT" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
        set -u
        cd /c
        rm -f models/*.gguf smoke/*.json smoke/*.jsonl smoke/*.log smoke/*.pid
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models > /c/models/save.log 2>&1
        M=$(ls /c/models/qwen4exp-moe.gguf 2>/dev/null || ls /c/models/qwen4exp-*.gguf | head -1)
        echo "model $M"
        # deterministic token lists below the vocabulary size (a Park-Miller generator)
        toks() { awk -v n="$1" -v s="$2" -v v="$3" "BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf \"%s%d\", (i ? \",\" : \"\"), 1 + x % (v - 1) } }"; }
        serve() { # <arm> [env...]
            local arm=$1; shift
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
        K="LLAMA_SPEC_DUMP_K=4 LLAMA_SPEC_DUMP_TGT_K=4 LLAMA_SPEC_DUMP_NGRAM=8"
        run_arm plain
        run_arm dump LLAMA_SPEC_DUMP=/c/smoke/dump.jsonl $K
        run_arm full LLAMA_SPEC_DUMP=/c/smoke/full.jsonl LLAMA_SPEC_DUMP_FULL_BLOCK=1 $K
        echo "--- same tokens in the three arms"
        for r in R1 R2 R3; do
            if cmp -s <(jq -c .tokens /c/smoke/plain.$r.json) <(jq -c .tokens /c/smoke/dump.$r.json) &&
               cmp -s <(jq -c .tokens /c/smoke/plain.$r.json) <(jq -c .tokens /c/smoke/full.$r.json); then echo "$r same"; else echo "$r DIFFER"; fi
        done
        echo "--- dumps"
        wc -l < /c/smoke/dump.jsonl | sed "s/^/lines /"
        grep -m 1 "LLAMA_SPEC_DUMP" /c/smoke/full.server.log
        # the two dumps hold the same lines but for the server start; neither carries a second block
        if cmp -s <(jq -c "del(.run)" /c/smoke/dump.jsonl) <(jq -c "del(.run)" /c/smoke/full.jsonl); then echo "dumps same"; else echo "dumps DIFFER"; fi
        echo "fb lines $(jq -c "select(.fb != null)" /c/smoke/full.jsonl | wc -l)"
        jq -s -c "{steps: length, types: (map(.type) | unique), with_cand: (map(select(.cand != null)) | length), with_tgt: (map(select((.tgt_ids | length) > 0)) | length), with_ng: (map(select(.ng_len > 0)) | length), with_gram: (map(select(.gram != null)) | length), drafted: (map(.draft | length) | add), accepted: (map(.n_acc) | add)}" /c/smoke/full.jsonl
        head -n 1 /c/smoke/full.jsonl | cut -c 1-600
        echo "--- check"
        python3 -I /t/tree_replay.py --check /c/smoke/full.jsonl; echo "check rc=$?"
        python3 -I /t/tree_replay.py /c/smoke/full.jsonl --out /c/smoke/replay --budgets 1,2,3,4 > /dev/null; echo "replay rc=$?"
        chown -R "$HOST_UID:$HOST_GID" /c' > "$C/smoke/console.txt" 2>&1
    cat "$C/smoke/console.txt"
    nsame=$(grep -c '^R[123] same' "$C/smoke/console.txt")
    crc=$(grep -o 'check rc=[0-9]*' "$C/smoke/console.txt" | cut -d= -f2)
    lines=$(grep -o '^lines [0-9]*' "$C/smoke/console.txt" | cut -d' ' -f2)
    ncand=$(grep -o '"with_cand":[0-9]*' "$C/smoke/console.txt" | cut -d: -f2)
    dsame=$(grep -c '^dumps same' "$C/smoke/console.txt")
    verdict smoke "$([ "$nsame" = 3 ] && [ "$dsame" = 1 ] && [ "${crc:-1}" = 0 ] && [ "${lines:-0}" -gt 0 ] && [ "${ncand:-0}" = "${lines:-x}" ] && echo OK || echo FAIL)" \
        "$nsame/3 requests same tokens in the three arms; dumps same $dsame; $lines lines, $ncand with candidates; check rc ${crc:-none}; $(grep -m 1 '^[0-9]* steps' "$C/smoke/console.txt")"
fi

cat "$C/summary.tsv"
date -Is > "$C/DONE"
echo "=== cpu check ended $(date -Is)"
