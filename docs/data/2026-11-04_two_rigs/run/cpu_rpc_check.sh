#!/usr/bin/env bash
# Q743 on a build machine, CPU only: Miner Llama's ggml-rpc (fork branch pascal-p106-v3-b12076-rpc) with two local
# ggml-rpc-server processes standing in for a second rig's cards. Steps (STEPS="build models tokens ops bytes"):
#   1. build   ggml-rpc-server, llama-server and test-llama-archs from $WORK/src, GGML_CUDA=OFF GGML_RPC=ON, static
#   2. models  random-weight qwen4exp files (Flash-Next's architecture) from test-llama-archs
#   3. tokens  llama-server on each file, every arm against the same requests:
#                local      everything on the local CPU
#                rpc1       every layer on one rpc-server
#                rpc2       the layers split over two rpc-servers, as two rigs' cards would be: -ts 1,2 puts block 0 on
#                           the first, block 1 and the output on the second (the random file has two blocks), so the
#                           residual crosses between two servers inside the model
#                mixed      rpc2 with block 1 on the local CPU (remote -> local -> remote boundaries)
#                hop        rpc2 with LLAMA_HOP_CAST=f16 (the fork's F16 hop at the RPC0 -> RPC1 boundary)
#                draft-l    local with the file as its own drafter (draft-simple, 4 tokens a step)
#                draft-r    the same on rpc2, the drafter on the remote devices too
#              requests: R1 greedy 64 tokens with the top 5 probabilities, R2a/R2b a prompt and then the same prompt
#              grown by 30 tokens with cache_prompt (the second reuses the cached prefix: recurrent state and
#              checkpoints restored on remote buffers), R3 sampled (temperature 1, seed 42); rpc1, rpc2 and mixed must
#              give local's tokens, draft-r draft-l's; hop, draft-l and draft-r against local are reported (F16 rounds
#              the residual; a verify batch may round differently from one-token steps), with the top-5 probabilities
#   4. ops     GGML_SCHED_DEBUG=2 on rpc2 and hop: every op the scheduler put on each remote device, each split's
#              inputs
#   5. bytes   socket byte counters (ss -ti) of the client's connections and the client's RPC commands
#              (GGML_RPC_DEBUG=3: name, bytes in and out, time) per decoded token, at two prompt lengths (decode = the
#              request with 65 new tokens minus the one with 1, over 64)
# Usage (on the build machine; WORK/src is a clone of the fork at the branch under test):
#   WORK=/path/work REPO=/path/this-repo JOBS=6 THREADS=4 LOCK=/path/build.lock DOCKER_ARGS="--label x=1" \
#     bash cpu_rpc_check.sh
# Output: $WORK/cpu/{bin,models,tokens,ops,bytes}/ and $WORK/cpu/summary.tsv (one verdict line per check).
set -uo pipefail
WORK=${WORK:?}
REPO=${REPO:?}
JOBS=${JOBS:-6}
THREADS=${THREADS:-4}
PORT=${PORT:-18191}
RPC_A=${RPC_A:-50152}
RPC_B=${RPC_B:-50153}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
RUN=$REPO/docs/data/2026-11-04_two_rigs/run
mkdir -p "$WORK/logs" "$C/bin" "$C/models" "$C/tokens" "$C/ops" "$C/bytes"
rm -f "$C/DONE"
STEPS=${STEPS:-build models tokens ops bytes}
has() { [[ " $STEPS " == *" $1 "* ]]; }
[ -f "$C/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
echo "=== cpu rpc check started $(date -Is), steps: $STEPS"

build() {
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        git log --oneline -1
        cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_RPC=ON -DGGML_NATIVE=OFF \
            -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target ggml-rpc-server llama-server test-llama-archs \
            > /work/logs/cpu-build.log 2>&1 || { grep -E "error" /work/logs/cpu-build.log | head -40; exit 1; }
        cp build-cpu/bin/ggml-rpc-server build-cpu/bin/llama-server build-cpu/bin/test-llama-archs /work/cpu/bin/
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
    verdict build OK "$(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | head -n 1 | cut -c 1-120)"
fi

# everything after the build runs in one container: the servers talk over its loopback
docker run --rm $D -v "$C:/c" -v "$RUN:/r:ro" -e THREADS="$THREADS" -e PORT="$PORT" -e RPC_A="$RPC_A" \
    -e RPC_B="$RPC_B" -e STEPS="$STEPS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq python3 iproute2 > /dev/null
    set -u
    cd /c
    has() { [[ " $STEPS " == *" $1 "* ]]; }
    verdict() { printf "%s\t%s\t%s\n" "$1" "$2" "$3" | tee -a /c/summary.tsv; }
    if has models || ! ls /c/models/*.gguf > /dev/null 2>&1; then
        rm -f /c/models/*.gguf
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -o /c/models > /c/models/save.log 2>&1
        verdict models "$(ls /c/models/*.gguf > /dev/null 2>&1 && echo OK || echo FAIL)" "$(cd /c/models && ls *.gguf | tr "\n" " ")"
    fi
    # deterministic token lists below the vocabulary size (a Park-Miller generator)
    toks() { awk -v n="$1" -v s="$2" -v v="$3" "BEGIN { x = s; for (i = 0; i < n; i++) { x = (x * 16807) % 2147483647; printf \"%s%d\", (i ? \",\" : \"\"), 1 + x % (v - 1) } }"; }
    rpc_up() { # <dir> <port>...: one CPU rpc-server per port
        local dir=$1; shift
        for p in "$@"; do
            /c/bin/ggml-rpc-server -H 127.0.0.1 -p "$p" -t "$THREADS" > "$dir/rpc-$p.log" 2>&1 &
            echo $! > "$dir/rpc-$p.pid"
        done
        sleep 2
    }
    rpc_down() { # <dir> <tag>: peak RSS of each rpc-server, then stop it
        local dir=$1 tag=$2 f
        for f in "$dir"/rpc-*.pid; do
            [ -f "$f" ] || continue
            echo "$tag $(basename "$f" .pid) $(grep -E "VmHWM|VmRSS" /proc/$(cat "$f")/status 2>/dev/null | tr -s " \t" " " | tr "\n" " ")" >> "$dir/rss.txt"
            kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
        done
        sleep 1
    }
    serve() { # <dir> <tag> <model> <server args> [env...]
        local dir=$1 tag=$2 m=$3 sargs=$4; shift 4
        env "$@" /c/bin/llama-server -m "$m" $sargs -t "$THREADS" -c 4096 -np 1 -fa on --no-repack --no-webui \
            --host 127.0.0.1 --port "$PORT" > "$dir/$tag.server.log" 2>&1 &
        echo $! > "$dir/server.pid"
        for _ in $(seq 1 180); do
            curl -s "http://127.0.0.1:$PORT/health" | grep -q "\"ok\"" && return 0
            kill -0 "$(cat "$dir/server.pid")" 2> /dev/null || { echo "$tag: server exited"; tail -n 20 "$dir/$tag.server.log"; return 1; }
            sleep 1
        done
        echo "$tag: server did not start"; tail -n 20 "$dir/$tag.server.log"; return 1
    }
    stop() { kill "$(cat "$1/server.pid")" 2>/dev/null; sleep 2; }
    req() { # <out.json> <tokens> <n_predict> <temperature> <seed> <cache_prompt>
        printf "{\"prompt\":[%s],\"n_predict\":%s,\"temperature\":%s,\"top_k\":20,\"top_p\":0.95,\"min_p\":0,\"seed\":%s,\"n_probs\":5,\"return_tokens\":true,\"cache_prompt\":%s}" "$2" "$3" "$4" "$5" "$6" > /tmp/req.json
        curl -s --max-time 900 -H "Content-Type: application/json" --data @/tmp/req.json "http://127.0.0.1:$PORT/completion" > "$1"
    }
    RPC2="--rpc 127.0.0.1:$RPC_A,127.0.0.1:$RPC_B -ngl 99 -ts 1,2"
    RPC1="--rpc 127.0.0.1:$RPC_A -ngl 99"
    arm_args() { # <arm> <model>: the server arguments, then "|", then the env
        local d="-md $2 --spec-type draft-simple --spec-draft-n-max 4"
        case $1 in
            local)   echo "-ngl 0 |" ;;
            rpc1)    echo "$RPC1 |" ;;
            rpc2)    echo "$RPC2 |" ;;
            mixed)   echo "$RPC2 -ot ^blk\\.1\\.=CPU |" ;;
            hop)     echo "$RPC2 | LLAMA_HOP_CAST=f16" ;;
            draft-l) echo "-ngl 0 $d -ngld 0 |" ;;
            draft-r) echo "$RPC2 $d -ngld 99 |" ;;
        esac
    }
    ARMS="local rpc1 rpc2 mixed hop draft-l draft-r"
    if has tokens; then
        echo "=== 3. tokens $(date -Is)"
        for m in /c/models/*.gguf; do
            name=$(basename "$m" .gguf)
            dir=/c/tokens/$name; rm -rf "$dir"; mkdir -p "$dir"
            for arm in $ARMS; do
                a=$(arm_args "$arm" "$m"); sargs=${a%%|*}; envs=${a#*|}
                rpc_up "$dir" $RPC_A $RPC_B
                if ! serve "$dir" "$arm" "$m" "$sargs" $envs; then
                    verdict "tokens/$name/$arm" FAIL "server did not start: $(grep -m 1 -iE "error|abort|assert" "$dir/$arm.server.log" | cut -c 1-160)"
                    rpc_down "$dir" "$arm"; continue
                fi
                nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r ".data[0].meta.n_vocab")
                a1=$(toks 40 3 "$nv"); b1=$(toks 30 5 "$nv"); c1=$(toks 20 9 "$nv")
                P1="$a1,$b1,$a1,$c1"
                req "$dir/$arm.R1.json"  "$P1"       64 0   1  false
                req "$dir/$arm.R2a.json" "$P1"       16 0   1  true
                req "$dir/$arm.R2b.json" "$P1,$b1"   32 0   1  true
                req "$dir/$arm.R3.json"  "$P1"       48 1.0 42 false
                grep -m 3 -E "graph splits|RPC|card hops" "$dir/$arm.server.log" | sed "s/^/$arm: /" > "$dir/$arm.notes"
                stop "$dir"
                rpc_down "$dir" "$arm"
            done
            python3 -I /r/rpc_tools.py compare "$dir" local "rpc1 rpc2 mixed" | tee "$dir/compare-local.txt"
            python3 -I /r/rpc_tools.py compare "$dir" draft-l "draft-r" | tee "$dir/compare-draft.txt"
            python3 -I /r/rpc_tools.py compare "$dir" local "hop draft-l draft-r" | tee "$dir/compare-report.txt"
            for f in compare-local compare-draft; do
                bad=$(grep -c "DIFFER\|MISSING" "$dir/$f.txt")
                verdict "tokens/$name/$f" "$([ "$bad" = 0 ] && echo OK || echo FAIL)" "$(grep -c " SAME" "$dir/$f.txt") same, $bad differ or missing"
            done
            verdict "tokens/$name/report" REPORT "$(grep -E "R1|R3" "$dir/compare-report.txt" | tr "\n" ";" | cut -c 1-300)"
        done
    fi
    if has ops; then
        echo "=== 4. ops $(date -Is)"
        for m in /c/models/*.gguf; do
            name=$(basename "$m" .gguf)
            dir=/c/ops/$name; rm -rf "$dir"; mkdir -p "$dir"
            for arm in rpc2 hop; do
                a=$(arm_args "$arm" "$m"); sargs=${a%%|*}; envs=${a#*|}
                rpc_up "$dir" $RPC_A $RPC_B
                if serve "$dir" "$arm" "$m" "$sargs -v" GGML_SCHED_DEBUG=2 $envs; then
                    nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r ".data[0].meta.n_vocab")
                    req "$dir/$arm.R.json" "$(toks 40 3 "$nv")" 2 0 1 false
                    stop "$dir"
                    python3 -I /r/rpc_tools.py ops "$dir/$arm.server.log" > "$dir/$arm.ops.tsv"
                    grep -m 3 -E "card hops|graph splits" "$dir/$arm.server.log" | sed "s/^/$arm: /" > "$dir/$arm.notes"
                    verdict "ops/$name/$arm" "$(grep -q "^RPC" "$dir/$arm.ops.tsv" && echo OK || echo FAIL)" "$(awk -F "\t" "\$1 ~ /^RPC/ {print \$2}" "$dir/$arm.ops.tsv" | sort -u | tr "\n" " " | cut -c 1-300) $(cut -c 1-120 "$dir/$arm.notes" | tr "\n" " ")"
                    gzip -f "$dir/$arm.server.log"
                else
                    verdict "ops/$name/$arm" FAIL "server did not start"
                fi
                rpc_down "$dir" "ops-$arm"
            done
        done
    fi
    if has bytes; then
        echo "=== 5. bytes $(date -Is)"
        m=$(ls /c/models/qwen4exp-moe.gguf 2>/dev/null || ls /c/models/*.gguf | head -n 1)
        dir=/c/bytes; rm -rf "$dir"/*
        snap() { # <label>: the client sockets towards each rpc-server
            for p in $RPC_A $RPC_B; do
                echo "$1 $p $(ss -tinH state established "( dport = :$p )" | tr "\n" " ")"
            done >> "$dir/ss.txt"
        }
        rpc_up "$dir" $RPC_A $RPC_B
        # the random model has a 256-token context: prompts of 32 and 160 tokens, then 1 or 65 new ones
        if serve "$dir" rpc2 "$m" "$RPC2 -v" GGML_RPC_DEBUG=3; then
            nv=$(curl -s "http://127.0.0.1:$PORT/v1/models" | jq -r ".data[0].meta.n_vocab")
            log=$dir/rpc2.server.log
            for plen in 32 160; do
                P=$(toks $plen 11 "$nv")
                req "$dir/warm.json" "$P" 4 0 1 false
                for n in 1 65; do
                    l0=$(wc -l < "$log")
                    snap "before $plen $n"; req "$dir/p$plen-n$n.json" "$P" $n 0 1 false; snap "after $plen $n"
                    sleep 1; sed -n "$((l0 + 1)),\$p" "$log" > "$dir/p$plen-n$n.rpc.log"
                done
            done
            stop "$dir"
            python3 -I /r/rpc_tools.py bytes "$dir/ss.txt" > "$dir/per_token.tsv"
            python3 -I /r/rpc_tools.py cmds "$dir" > "$dir/commands.tsv"
            gzip -f "$log" "$dir"/*.rpc.log
            verdict bytes "$([ -s "$dir/per_token.tsv" ] && echo OK || echo FAIL)" "$(tail -n +2 "$dir/per_token.tsv" | tr "\t\n" " ;" | cut -c 1-300)"
        else
            verdict bytes FAIL "server did not start"
        fi
        rpc_down "$dir" bytes
    fi
    chown -R "$HOST_UID:$HOST_GID" /c
'
cat "$C/summary.tsv"
date -Is > "$C/DONE"
