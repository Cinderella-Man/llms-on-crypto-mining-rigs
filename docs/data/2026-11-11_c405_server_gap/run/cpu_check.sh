#!/usr/bin/env bash
# C405 (Q960) on a build machine, CPU only: llama-server's step trace (LLAMA_STEP_TRACE) of a fork branch.
# Steps:
#   cpubuild  llama-server of the branch without CUDA (static, native) into $WORK/cpu/bin
#   smoke     llama-server on the CPU backend with a tiny qwen4exp model (test-llama-archs's random weights) and a
#             drafter: draft-mtp (the model's own MTP layer, at three tokens, as the release preset drafts) and
#             draft-simple (the same file as a separate draft model). Every arm answers the same three requests
#             (greedy 96 tokens, temperature 1.0 seed 42 48 tokens, greedy 64 tokens on a longer prompt):
#               base-<d>  the base build (BASE_BIN)
#               off-<d>   the branch, the variable unset
#               on-<d>    the branch, LLAMA_STEP_TRACE=<file>
#               bad-mtp   the branch, LLAMA_STEP_TRACE=<a path it cannot write>: it must serve as off does
#             Verdicts: off, on and bad give base's tokens and draft counts for every request; on writes a trace that
#             step_gap.py --check passes, whose drafted and accepted counts add up to the server's; off writes none
#   archs     test-llama-archs -b CPU -s 1 on every architecture: the base's CPU build writes each model's logits, the
#             branch's compares (as C620's off step); it also saves and reloads models, the saver's change included
# Usage: WORK=<dir> BRANCH=<fork branch> BASE_BIN=<bin/ of the base's static native CPU build> [REMOTE=fork]
#   [JOBS=8] [THREADS=4] [PORT=18405] [LOCK=<flock file>] [DOCKER_ARGS=...] [STEPS='cpubuild smoke archs']
#   bash cpu_check.sh
#   WORK   a work folder whose llama.cpp clone in $WORK/src has the fork as remote REMOTE; $WORK/ccache
#   LOCK   one build or test at a time across the sessions that share the machine: every step waits for it
# Output: $WORK/cpu/summary.tsv (check, verdict, detail), $WORK/cpu/smoke/ (replies, server logs, traces, step_gap
#   tables), $WORK/logs/; done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BASE_BIN:?BASE_BIN=<bin/ of the base CPU build>}"
BASE_BIN=$(realpath "$BASE_BIN")
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
PORT=${PORT:-18405}
STEPS=${STEPS:-cpubuild smoke archs}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
RUN=$(dirname "$(realpath "$0")")
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
mkdir -p "$WORK/logs" "$C/bin" "$C/smoke" "$C/models"
rm -f "$C/DONE"
[ -f "$C/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
echo "=== c405 cpu check $(date -Is), $BRANCH at ${TIP:0:9}, base $BASE_BIN, steps: $STEPS"

if step cpubuild; then
    echo "=== cpubuild $(date -Is)"
    git -C "$WORK/src" checkout -q --detach "$TIP"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cpu-cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target llama-server test-llama-archs \
            > /work/logs/cpu-build.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            cp /work/cpu/build/bin/llama-server /work/cpu/build/bin/test-llama-archs /work/cpu/bin/
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs /work/ccache
        exit $rc'
    rc=$?
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; commit ${TIP:0:9}; logs/cpu-build.log"
fi

if step smoke; then
    echo "=== smoke $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$C:/c" -v "$BASE_BIN:/base:ro" -v "$RUN:/r:ro" \
        -e THREADS="$THREADS" -e PORT="$PORT" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
        set -u
        cd /c
        rm -rf smoke && mkdir -p smoke
        # the plain model for draft-simple (a file with an MTP block loads only with the MTP layer requested), the
        # model with one MTP block for draft-mtp
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models > /c/models/save.log 2>&1
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models --mtp > /c/models/save-mtp.log 2>&1
        M0=/c/models/qwen4exp-moe.gguf
        M1=/c/models/qwen4exp-moe-mtp.gguf
        echo "models $M0 ($(stat -c %s "$M0") bytes), $M1 ($(stat -c %s "$M1") bytes)"
        # deterministic token lists below the vocabulary size (a Park-Miller generator)
        toks() { awk -v n="$1" -v s="$2" -v v="$3" "BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf \"%s%d\", (i ? \",\" : \"\"), 1 + x % (v - 1) } }"; }
        serve() { # <arm> <bin> <server args> [env...]
            local arm=$1 bin=$2 sargs=$3; shift 3
            env "$@" $bin -m "$MM" $sargs -ngl 0 -t "$THREADS" -c 4096 -np 1 --no-webui --host 127.0.0.1 --port "$PORT" \
                > /c/smoke/$arm.server.log 2>&1 &
            echo $! > /c/smoke/$arm.pid
            for _ in $(seq 1 120); do
                curl -s "http://127.0.0.1:$PORT/health" | grep -q "\"ok\"" && return 0
                kill -0 "$(cat /c/smoke/$arm.pid)" 2> /dev/null || { echo "$arm: server exited"; tail -5 /c/smoke/$arm.server.log; return 1; }
                sleep 1
            done
            echo "$arm: server did not start"; tail -20 /c/smoke/$arm.server.log; return 1
        }
        stop() { # SIGTERM and wait, so that the trace file is whole
            local p; p=$(cat /c/smoke/$1.pid); kill "$p" 2>/dev/null
            for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || return 0; sleep 1; done
            kill -9 "$p" 2>/dev/null; echo "$1: killed"
        }
        req() { # <arm> <name> <tokens> <n_predict> <temperature> <seed>
            printf "{\"prompt\":[%s],\"n_predict\":%s,\"temperature\":%s,\"top_k\":20,\"top_p\":0.95,\"min_p\":0,\"seed\":%s,\"return_tokens\":true,\"cache_prompt\":false}" "$3" "$4" "$5" "$6" > /tmp/req.json
            curl -s --max-time 600 -H "Content-Type: application/json" --data @/tmp/req.json "http://127.0.0.1:$PORT/completion" > /c/smoke/$1.$2.json
            echo "$1 $2 $(jq -c "{n: (.tokens | length), draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}" /c/smoke/$1.$2.json)"
        }
        MTP="--spec-type draft-mtp --spec-draft-n-max 3"
        SIM="-md $M0 -ngld 0 --spec-type draft-simple --spec-draft-n-max 3"
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
            grep -h "LLAMA_STEP_TRACE\|step_trace" /c/smoke/$arm.server.log | sed "s/^/$arm: /" | head -3
            stop "$arm"
        }
        T=LLAMA_STEP_TRACE
        for d in mtp sim; do
            if [ $d = mtp ]; then A=$MTP; MM=$M1; else A=$SIM; MM=$M0; fi
            run_arm base-$d /base/llama-server  "$A"
            run_arm off-$d  /c/bin/llama-server "$A"
            run_arm on-$d   /c/bin/llama-server "$A" $T=/c/smoke/on-$d.trace.jsonl
        done
        MM=$M1
        run_arm bad-mtp /c/bin/llama-server "$MTP" $T=/nonexistent/c405/trace.jsonl

        same()   { cmp -s <(jq -c .tokens "$1") <(jq -c .tokens "$2"); }
        counts() { jq -c "[.timings.draft_n, .timings.draft_n_accepted]" "$1"; }
        echo "--- against base: tokens and draft counts"
        for d in mtp sim; do
            for arm in off-$d on-$d $([ $d = mtp ] && echo bad-mtp); do
                for r in R1 R2 R3; do
                    f=/c/smoke/$arm.$r.json; g=/c/smoke/base-$d.$r.json
                    if [ -s "$f" ] && [ -s "$g" ] && same "$f" "$g" && [ "$(counts "$f")" = "$(counts "$g")" ]; then
                        echo "$arm $r identical $(counts "$f") n=$(jq ".tokens | length" "$f")"
                    else
                        echo "$arm $r DIFFER $(counts "$f" 2>/dev/null) base $(counts "$g" 2>/dev/null)"
                    fi
                done
            done
        done
        echo "--- traces"
        for d in mtp sim; do
            t=/c/smoke/on-$d.trace.jsonl
            sd=0; sa=0
            for r in R1 R2 R3; do
                sd=$((sd + $(jq ".timings.draft_n // 0" /c/smoke/on-$d.$r.json 2>/dev/null || echo 0)))
                sa=$((sa + $(jq ".timings.draft_n_accepted // 0" /c/smoke/on-$d.$r.json 2>/dev/null || echo 0)))
            done
            td=$(tail -n +2 "$t" 2>/dev/null | jq -s "map(.n.drf // 0) | add // 0")
            ta=$(tail -n +2 "$t" 2>/dev/null | jq -s "map(.n.acc // 0) | add // 0")
            echo "on-$d trace: $(wc -l < "$t") lines; drafted $td (server $sd), accepted $ta (server $sa)"
            python3 -I /r/step_gap.py --check --skip 3 --tsv /c/smoke/on-$d.steps.tsv "$t" > /c/smoke/on-$d.gap.md
            echo "on-$d step_gap check rc=$?: $(tail -n 1 /c/smoke/on-$d.gap.md)"
        done
        echo "--- files the off arms wrote"
        ls /c/smoke | grep -c "^off-.*trace" | sed "s/^/off traces: /"
        grep -h "cannot write LLAMA_STEP_TRACE" /c/smoke/bad-mtp.server.log | head -1 | sed "s/^/bad-mtp: /"
        chown -R "$HOST_UID:$HOST_GID" /c/smoke /c/models' > "$C/smoke-console.txt" 2>&1
    rc=$?
    S=$C/smoke-console.txt
    n_id=$(grep -c ' identical ' "$S"); n_diff=$(grep -c ' DIFFER ' "$S")
    verdict identity "$([ "$n_diff" = 0 ] && [ "$n_id" = 15 ] && echo OK || echo FAIL)" \
        "$n_id of 15 replies identical to base (off, on, bad x R1-R3 x mtp; off, on x R1-R3 x sim), $n_diff differ; rc=$rc"
    for d in mtp sim; do
        line=$(grep "^on-$d trace:" "$S")
        chk=$(grep "^on-$d step_gap check" "$S")
        td=$(sed -n 's/.*drafted \([0-9]*\) (server \([0-9]*\)).*/\1 \2/p' <<< "$line")
        ta=$(sed -n 's/.*accepted \([0-9]*\) (server \([0-9]*\)).*/\1 \2/p' <<< "$line")
        ok=FAIL
        if [[ "$chk" == *"rc=0"* ]] && [ -n "$td" ] && [ "${td% *}" = "${td#* }" ] && [ "${ta% *}" = "${ta#* }" ]; then ok=OK; fi
        verdict "trace-$d" "$ok" "$line; $chk"
    done
    nt=$(sed -n 's/^off traces: //p' "$S")
    verdict off-writes-nothing "$([ "$nt" = 0 ] && echo OK || echo FAIL)" "trace files from off arms: $nt"
    verdict bad-path "$(grep -q '^bad-mtp: .*cannot write LLAMA_STEP_TRACE' "$S" && echo OK || echo FAIL)" \
        "$(grep '^bad-mtp: .*cannot write LLAMA_STEP_TRACE' "$S" | head -1); its replies are in the identity count"
fi

if step archs; then
    echo "=== archs $(date -Is)"
    K=$C/archs
    rm -rf "$K" && mkdir -p "$K/k"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$C:/c" -v "$BASE_BIN:/base:ro" -v "$K/k:/k" "$IMAGE" bash -c '
        run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        run base /base
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run branch /c/bin
        rm -f /k/*.f32
        chown -R '"$(id -u):$(id -g)"' /k' > "$K/rc.txt" 2>&1
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep 'logits against' "$K/k/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
    nb=$(cat "$K/k/n_base.txt" 2>/dev/null || echo 0)
    nc=$(grep -c 'logits against' "$K/k/branch.log")
    grep 'logits against' "$K/k/branch.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" > "$K/differ.txt"
    bad=$(cut -d: -f1 "$K/differ.txt" | tr '\n' ' ')
    sb=$(grep -m 1 '^Summary' "$K/k/base.log") sm=$(grep -m 1 '^Summary' "$K/k/branch.log")
    verdict archs "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && { [ -z "$bad" ] && echo OK || echo REVIEW; } || echo FAIL)" \
        "base wrote $nb logits files; $nc compared, $(wc -l < "$K/differ.txt") differ: ${bad:-none}; base: $sb; branch: $sm; $(grep -o -E '(base|branch) rc=[0-9]+' "$K/rc.txt" | tr '\n' ' ')"
fi

echo "=== done $(date -Is)"
cat "$C/summary.tsv"
touch "$C/DONE"
