#!/usr/bin/env bash
# two_rigs.sh OUT: Qwen3.8-Flash-Next on two rigs (18 P106 cards) over llama.cpp's RPC (Q816): rig 1 runs llama-server
# with its own nine cards and reads the GGUF (and the n-gram table, from its SSD); rig 2 runs ggml-rpc-server, which
# exposes its nine cards. Both run the fork's RPC build (pascal-p106-v3-b12076-rpc, Q743). Runs only ssh here; nothing
# is timed until step agent, and no command goes to a rig during that step.
# Steps (STEPS, default "net rpcup devices ident smoke hop agent rpcdown"):
#   net      ping rig 1 -> rig 2 (small and 1,472-byte packets, 200 each, 0.2 s apart), both NICs' coalescing (ethtool -c)
#   rpcup    ggml-rpc-server on rig 2 (RPC_ARGS; with RPC_CACHE set, -c caches every tensor above 10 MB there, so the
#            second load sends hashes instead of weights), guarded by scripts/ram_watchdog.sh (PROC=ggml-rpc-server); its device
#            list and its host memory once up
#   devices  llama-server --list-devices with --rpc on rig 1: 9 CUDA + 9 RPC devices expected
#   ident    the same file on 9 and on 18 cards, greedy: MODEL_ID (default: Swift IQ3_XXS, which fits one rig) with
#            ID_ARGS9, then with --rpc and ID_ARGS18, each through smoke_dflash.sh (the agent's first turn, ~2,780
#            tokens, then smoke_greedy.json at temperature 0); the greedy replies must be the same text, and the first
#            turn's prompt and decode speed give the 18-card cost before the timed arm
#   smoke    MODEL (Swift IQ3_S) on 18 cards with SMOKE_ARGS (the agent arm's line), the same two requests
#   hop      the smoke with LLAMA_HOP_CAST=f16 (the residual crosses every card boundary and the LAN in F16, Q231)
#   agent    one agent-check arm, "two": MODEL with AGENT_ARGS through release_check_s12.sh (STEPS=run, BUDGET_MIN 40),
#            then its report against REF_ARMS from the same client folder
#   rpcdown  ggml-rpc-server and its watchdog stopped; rig 2's logs copied to OUT and removed (the cache stays)
# Environment: everything smoke_dflash.sh and release_check_s12.sh take (RIG = rig 1, BENCH, CLIENT, the ports, the
# client folders, RIG_LOGDIR, REL_BIN = the RPC build's llama-server on rig 1), plus
#   RIG2       rig 2's ssh alias
#   RIG2_ADDR  rig 2's address on the rigs' own LAN (the one rig 1 reaches it on; never a VPN address: RPC is
#              unauthenticated and its round trips are the cost)
#   RPC_PORT   (50052)   RPC_BIN  ggml-rpc-server on rig 2
#   RPC_CACHE  optional: a folder on a rig-2 drive with ~30 GB free for -c (worth it only where that drive reads faster
#              than the LAN carries, ~110 MB/s; unset, every load sends rig 2's ~28 GB again, ~4-5 min)
#   RIG2_LOGDIR (/tmp/two-rigs-rpc)  MODEL  Swift IQ3_S shard 1 on rig 1   MODEL_ID  Swift IQ3_XXS shard 1 on rig 1
#   TS18 (default the plan for Swift IQ3_S at 64k, -ub 512, whole layers: 1,780 MiB left on the fullest card)
#   TS18_ID, TS9_ID (Swift IQ3_XXS on 18 and 9 cards; defaults from the same planner)
#   REF_ARMS ("nodraft"), CTX (65536), GUARD_ENV (limits for rig 2's watchdog, e.g. "MIN_USABLE_MB=40")
set -uo pipefail
OUT=$(realpath -m "${1:?usage: two_rigs.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${RIG:?}" "${RIG2:?}" "${RIG2_ADDR:?}" "${RPC_BIN:?}" "${REL_BIN:?}" "${MODEL:?}" "${MODEL_ID:?}"
export BENCH=${BENCH:-$REPO/bench/agent_check}
STEPS=${STEPS:-net rpcup devices ident smoke hop agent rpcdown}
RPC_PORT=${RPC_PORT:-50052}
RIG2_LOGDIR=${RIG2_LOGDIR:-/tmp/two-rigs-rpc}
CTX=${CTX:-65536}
REF_ARMS=${REF_ARMS:-nodraft}
RPC="--rpc $RIG2_ADDR:$RPC_PORT"
# llama.cpp puts the RPC devices first: the first nine -ts values are rig 2's cards (RPC0-8), the last nine rig 1's
# (CUDA0-8); rig 1's last card holds the output head, so the logits never cross the LAN. From bench/file_choice/fit.py
# --cards 18 (fit18.tsv in this folder): Swift IQ3_S, 64k, F16 cache, -ub 512, whole layers, no draft
TS18=${TS18:-3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,2,2,3}
TS18_ID=${TS18_ID:-3,3,3,3,3,3,3,3,3,3,2,2,2,3,3,2,2,3}
TS9_ID=${TS9_ID:-6,6,7,5,5,5,5,5,5}
ID_CTX=${ID_CTX:-16384}
ID_ARGS9=${ID_ARGS9:-"-b 512 -ub 64 -ts $TS9_ID -fit off -lm dio"}
ID_ARGS18=${ID_ARGS18:-"-b 512 -ub 64 -ts $TS18_ID -fit off -lm dio $RPC"}
SMOKE_ARGS=${SMOKE_ARGS:-"-b 2048 -ub 512 -ts $TS18 -fit off -lm dio $RPC"}
AGENT_ARGS=${AGENT_ARGS:-$SMOKE_ARGS}
S18=$REPO/docs/data/2026-11-03_flash_next_dflash2
REL_CHECK=$REPO/docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
say "two_rigs: steps $STEPS; rig 2 at $RIG2_ADDR:$RPC_PORT; REL_BIN $REL_BIN"

if has net; then
  ssh "$RIG" "ping -c 200 -i 0.2 -s 56 $RIG2_ADDR | tail -n 2; ping -c 200 -i 0.2 -s 1472 -M do $RIG2_ADDR | tail -n 2" \
    > "$OUT/ping.txt" 2>&1
  for r in "$RIG" "$RIG2"; do
    ssh "$r" 'i=$(ip -o route get '"$RIG2_ADDR"' | grep -o "dev [^ ]*" | cut -d" " -f2); [ "$i" = lo ] && i=$(ip -o route show default | grep -o "dev [^ ]*" | head -n 1 | cut -d" " -f2); echo "$i"; ethtool -c "$i" 2>&1 | grep -iE "usecs|frames" | head -n 8'
  done > "$OUT/coalescing.txt" 2>&1
  say "net: $(grep -h "rtt\|packet loss" "$OUT/ping.txt" | tr '\n' ' ')"
fi

if has rpcup; then
  ssh "$RIG2" "mkdir -p $RIG2_LOGDIR ${RPC_CACHE:-} && cat > $RIG2_LOGDIR/ram_watchdog.sh" < "$REPO/scripts/ram_watchdog.sh"
  cache=${RPC_CACHE:+"LLAMA_CACHE=$RPC_CACHE"}
  timeout 30 ssh "$RIG2" "cd $RIG2_LOGDIR && $cache setsid nohup $RPC_BIN -H $RIG2_ADDR -p $RPC_PORT ${RPC_CACHE:+-c} \
    ${RPC_ARGS:-} > rpc-server.log 2>&1 < /dev/null & echo \$! > $RIG2_LOGDIR/rpc.pid"
  sleep 15
  timeout 30 ssh "$RIG2" "cd $RIG2_LOGDIR && PID=\$(cat rpc.pid) LOG=/dev/shm/ram_watchdog-rpc.tsv ${GUARD_ENV:-} \
    setsid nohup chrt -f 10 bash ram_watchdog.sh > watchdog.out 2>&1 < /dev/null & echo \$! > $RIG2_LOGDIR/watchdog.pid"
  sleep 2
  ssh "$RIG2" "cat $RIG2_LOGDIR/rpc-server.log; grep -E 'VmRSS|VmHWM' /proc/\$(cat $RIG2_LOGDIR/rpc.pid)/status; \
    grep -E 'MemAvailable|CmaFree' /proc/meminfo" > "$OUT/rpc_up.txt" 2>&1
  if ! ssh "$RIG2" "kill -0 \$(cat $RIG2_LOGDIR/rpc.pid)"; then
    say "rpcup: ggml-rpc-server is not running; stopping"; cat "$OUT/rpc_up.txt"; exit 1
  fi
  say "rpcup: $(grep -c 'CUDA' "$OUT/rpc_up.txt") CUDA lines; $(grep -h VmRSS "$OUT/rpc_up.txt")"
fi

if has devices; then
  ssh "$RIG" "$REL_BIN $RPC --list-devices" > "$OUT/devices.txt" 2>&1
  say "devices: $(grep -c 'CUDA[0-9]' "$OUT/devices.txt") CUDA, $(grep -c 'RPC[0-9]' "$OUT/devices.txt") RPC"
fi

smoke() {  # smoke <name> <model> <args> [env]: S18's smoke test on rig 1
  say "smoke $1: $3 ${4:-}"
  MODEL="$2" ARGS="$3" SMOKE_ENV="${4:-}" SMOKE_LOGDIR="${RIG_LOGDIR:-/tmp/agent-check}-smoke-$1" \
    REQUESTS="$S18/run/smoke_greedy.json" bash "$S18/run/smoke_dflash.sh" "$OUT/smoke-$1" > "$OUT/smoke-$1.console" 2>&1
  local rc=$?
  ssh "$RIG2" "grep -E 'VmRSS|VmHWM' /proc/\$(cat $RIG2_LOGDIR/rpc.pid)/status 2>/dev/null; \
    grep -E 'MemAvailable|CmaFree' /proc/meminfo" > "$OUT/smoke-$1/rig2_host.txt" 2>&1
  say "smoke $1: rc $rc; $(grep -h 'request long done\|did not come up\|RAM watchdog' "$OUT/smoke-$1/smoke.log" | tr '\n' ' ')"
  say "smoke $1: $(grep -o '"timings":{[^}]*}' "$OUT/smoke-$1/reply_long.json" 2>/dev/null | grep -oE '"(prompt_per_second|predicted_per_second)":[0-9.]*' | tr '\n' ' ')"
  return $rc
}
text() { grep -oE '"(reasoning_content|content)":"([^"\\]|\\.)*"' "$1" 2> /dev/null; }  # the reply's text fields

if has ident; then
  CTX=$ID_CTX smoke id9 "$MODEL_ID" "$ID_ARGS9"
  CTX=$ID_CTX smoke id18 "$MODEL_ID" "$ID_ARGS18"
  if [ "$(text "$OUT/smoke-id9/reply_1.json")" = "$(text "$OUT/smoke-id18/reply_1.json")" ] \
      && [ -n "$(text "$OUT/smoke-id9/reply_1.json")" ]; then
    say "ident: the greedy reply on 18 cards is the 9-card one"
  else
    say "ident: the greedy replies DIFFER (or one is missing): diff $OUT/smoke-id9/reply_1.json $OUT/smoke-id18/reply_1.json"
  fi
fi
has smoke && smoke iq3s "$MODEL" "$SMOKE_ARGS"
has hop && smoke iq3s-hop "$MODEL" "$SMOKE_ARGS" "LLAMA_HOP_CAST=f16"

if has agent; then
  say "agent: arm two, $AGENT_ARGS"
  env "ARGS_two=$AGENT_ARGS" "BUDGET_MIN_two=${BUDGET_MIN:-40}" STEPS=run ARMS=two MODEL="$MODEL" CTX="$CTX" \
    bash "$REL_CHECK" "$OUT/agent-two" > "$OUT/agent-two.console" 2>&1
  say "agent: rc $?; $(tail -n 1 "$OUT/agent-two/run.log" 2> /dev/null)"
  env "ARGS_two=$AGENT_ARGS" STEPS="report collect" ARMS="$REF_ARMS two" MODEL="$MODEL" CTX="$CTX" \
    bash "$REL_CHECK" "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
  say "report: rc $?; $OUT/agent-report"
fi

if has rpcdown; then
  ssh "$RIG2" "kill \$(cat $RIG2_LOGDIR/rpc.pid) 2>/dev/null; sleep 2; kill \$(cat $RIG2_LOGDIR/watchdog.pid) 2>/dev/null; \
    sleep 1; cp /dev/shm/ram_watchdog-rpc.tsv $RIG2_LOGDIR/ 2>/dev/null; rm -f /dev/shm/ram_watchdog-rpc.tsv; du -sh ${RPC_CACHE:-/dev/null}"
  ssh "$RIG2" "tar -C $RIG2_LOGDIR -cf - ." | tar -C "$OUT" -xf - && ssh "$RIG2" "rm -rf $RIG2_LOGDIR" \
    && say "rpcdown: rig 2's logs in $OUT (rpc-server.log, ram_watchdog-rpc.tsv), removed on rig 2"
fi
say "two_rigs: done"
