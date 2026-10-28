#!/usr/bin/env bash
# release_check.sh OUT: the agent check as a release validation. A coding agent fixes small bugs (the bugfix task set:
# 12 tasks, Python and JavaScript alternating, 5-10 tool calls each) against llama-server on the rig, once with stock
# llama.cpp and once with the release engine, and the report gives per task the result, minutes, turns and tool-call
# failures, and the time to first token per turn.
#
# Three machines, and the one this runs on only runs ssh:
#   - the rig runs llama-server on its loopback (rig_server.sh, with its monitors and RAM watchdog);
#   - the client (any Linux machine with Docker; CLIENT) runs pi, the logging proxy and the tasks' tests in containers
#     (client_arm.sh), labelled LABEL and at --cpu-shares CPU_SHARES;
#   - this machine copies the files over, starts and stops everything over ssh, and relays the client's requests to the
#     rig through two ssh forwards: -L RELAY_PORT -> rig 127.0.0.1:RIG_PORT and -R client 127.0.0.1:CLIENT_PORT ->
#     RELAY_PORT. Neither machine has to reach the other, and the server never listens beyond the rig's loopback.
#
# Arms (one llama-server each; the same model, template, sampling, context, checkpoints and metrics, COMMON):
#   stock  STOCK_BIN with STOCK_ARGS (default -b 2048 -ub 512)
#   rel    REL_BIN with REL_ARGS (default -b 2048 -ub 512 --pipe-depth 3) and the environment variables REL_ENV (e.g.
#          "LLAMA_SERVER_TEXT_REUSE=1"; default none)
# The context is CTX (default 65,536; pi's context window is set to match). Stock llama.cpp keeps an attention mask of
# context × micro-batch entries in pinned host memory: 264 MiB at 65,536 tokens with -ub 512, ~1 GiB at 262,144, which
# a 4 GB host cannot hold (docs/benchmarks/2026-10-22-qwen-baselines.md). A bug fix stays far below 65,536 tokens.
# One attempt per task, capped at TIME_CAP_MIN minutes (default 15) and 40 turns; BUDGET_MIN (default none) starts no
# attempt in an arm after that many minutes, so a slow arm cannot overrun its rig slot (the report pairs the tasks
# every arm scored).
#
# Environment (the machine-specific values; keep them in a private file):
#   RIG, CLIENT              ssh aliases, non-interactive
#   RIG_PORT                 llama-server's port on the rig's loopback
#   RELAY_PORT               this machine's end of the -L forward (loopback)
#   CLIENT_PORT, PROXY_PORT  the -R forward's end and the proxy, both on the client's loopback
#   CLIENT_DIR               folder on the client that receives the bench files (agent_check/ inside)
#   CLIENT_OUT               results folder on the client
#   STOCK_BIN, REL_BIN, MODEL   on the rig
#   optional: ARMS ("stock rel"), TASK_SET (bugfix), TASKS (all), CTX (65536), RIG_LOGDIR (/tmp/agent-check-rel),
#   TIME_CAP_MIN (15), BUDGET_MIN or per arm BUDGET_MIN_stock / BUDGET_MIN_rel (none),
#   CPU_SHARES (256), IMAGE (agent-check:0.87.1), LABEL (pascal-quants=1),
#   SAMPLING (the sampling every request carries, JSON, e.g. '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}';
#   default Flint's makers' settings, temperature 1.0, top-p 0.95, top-k 20, min-p 0)
#   STEPS ("prep run report collect"):
#     prep     copy the bench files to the client, build the sandbox image there (cached), validate the task set in it;
#              touches nothing on the rig
#     run      the relay, then per arm: the server on the rig, the client's attempts, the server stopped; the rig's
#              logs go to CLIENT_OUT/rig and are removed on the rig
#     report   compare.md (report.py compare), table.md, gonogo.md, on the client in a container
#     collect  CLIENT_OUT (without the workspaces) copied to OUT here
#   Replay instead of live attempts (REPLAY_LOG set): in step run each arm's server gets the recorded requests of
#   REPLAY_TRACES (comma-separated trace tags, e.g. rel/bugfix-python-list-ops/a1/t1) from REPLAY_LOG (a proxy.jsonl
#   on the client) through replay.py run: the same prompts, a fixed seed per request and the recorded reply lengths, so
#   two exact engines generate the same tokens and only their speed differs. Rows go to CLIENT_OUT/replay.jsonl
#   labelled REPLAY_LABEL (default the arm); several runs with the same CLIENT_OUT and other labels add up. Step report
#   then writes replay.md (replay.py summary: per label the cost per prompt and per generated token, and the ratios to
#   REPLAY_REF, default the first label, with 95 % intervals over traces).
# Resumable: run it again with the same CLIENT_OUT and finished attempts are skipped.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: release_check.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RIG:?}" "${CLIENT:?}" "${RIG_PORT:?}" "${RELAY_PORT:?}" "${CLIENT_PORT:?}" "${PROXY_PORT:?}" "${CLIENT_DIR:?}" \
  "${CLIENT_OUT:?}"
ARMS=${ARMS:-stock rel}; TASK_SET=${TASK_SET:-bugfix}; TASKS=${TASKS:-all}; CTX=${CTX:-65536}
TIME_CAP_MIN=${TIME_CAP_MIN:-15}; CPU_SHARES=${CPU_SHARES:-256}; IMAGE=${IMAGE:-agent-check:0.87.1}
LABEL=${LABEL:-pascal-quants=1}; RIG_LOGDIR=${RIG_LOGDIR:-/tmp/agent-check-rel}
STEPS=${STEPS:-prep run report collect}
STOCK_ARGS=${STOCK_ARGS:--b 2048 -ub 512}; REL_ARGS=${REL_ARGS:--b 2048 -ub 512 --pipe-depth 3}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
CAC="$CLIENT_DIR/agent_check"
SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3)
mkdir -p "$OUT"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
cl() { ssh "${SSH_OPTS[@]}" "$CLIENT" "$@"; }

arm_config() {  # sets SERVER_BIN, SERVER_ENV and SERVER_ARGS for an arm
  case $1 in
    stock) SERVER_BIN=${STOCK_BIN:-}; SERVER_ARGS="$COMMON $STOCK_ARGS"; SERVER_ENV="";;
    rel) SERVER_BIN=${REL_BIN:-}; SERVER_ARGS="$COMMON $REL_ARGS"; SERVER_ENV=${REL_ENV:-};;
    *) say "unknown arm $1"; return 1;;
  esac
  [ -n "$SERVER_BIN" ] && [ -n "${MODEL:-}" ] || { say "binary or MODEL for $1 not set"; return 1; }
}

relay_loop() {  # relay_loop NAME ssh-arguments…: keeps one forward up until killed
  local name=$1
  shift
  while true; do
    ssh -N "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes "$@" &
    echo $! > "$OUT/relay-$name.pid"
    wait $!
    sleep 3
  done
}

relay_start() {
  (relay_loop L -L "127.0.0.1:$RELAY_PORT:127.0.0.1:$RIG_PORT" "$RIG") >> "$OUT/relay.log" 2>&1 &
  RELAY_L=$!
  sleep 2
  (relay_loop R -R "127.0.0.1:$CLIENT_PORT:127.0.0.1:$RELAY_PORT" "$CLIENT") >> "$OUT/relay.log" 2>&1 &
  RELAY_R=$!
  sleep 3
  say "relay: client 127.0.0.1:$CLIENT_PORT -> here 127.0.0.1:$RELAY_PORT -> rig 127.0.0.1:$RIG_PORT"
}

relay_stop() {
  for x in L R; do
    v=RELAY_$x
    [ -n "${!v:-}" ] && kill "${!v}" 2> /dev/null
    [ -f "$OUT/relay-$x.pid" ] && kill "$(cat "$OUT/relay-$x.pid")" 2> /dev/null
    rm -f "$OUT/relay-$x.pid"
  done
  RELAY_L= RELAY_R=
}

server_start() {  # server_start ARM: llama-server on the rig, then /health through the whole relay from the client
  arm_config "$1" || return 1
  say "starting $1 on the rig: ${SERVER_ENV:+$SERVER_ENV }$SERVER_BIN $SERVER_ARGS"
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=start ARM=$1 BIN=$SERVER_BIN MODEL=$MODEL PORT=$RIG_PORT SERVER_ENV='$SERVER_ENV' SERVER_ARGS='$SERVER_ARGS' \
    bash -s" < "$HERE/rig_server.sh" 2>&1 | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  for _ in $(seq 40); do
    cl "curl -sf 127.0.0.1:$CLIENT_PORT/health" > /dev/null 2>&1 && { say "client reaches $1 through the relay"; return 0; }
    sleep 3
  done
  say "the client does not reach the server through the relay (relay.log)"
  return 1
}

server_stop() {
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=stop PORT=$RIG_PORT bash -s" < "$HERE/rig_server.sh" 2>&1 | tee -a "$OUT/run.log"
}

rig_end() {  # monitors stopped; the rig's logs appended to the client's CLIENT_OUT/rig (a resumed run adds to them),
  # then removed on the rig
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=end PORT=$RIG_PORT bash -s" < "$HERE/rig_server.sh" 2>&1 | tee -a "$OUT/run.log"
  if ssh "$RIG" "test -d $RIG_LOGDIR"; then
    ssh "$RIG" "tar -C $RIG_LOGDIR -cf - ." | cl "rm -rf $CLIENT_OUT/rig.new && mkdir -p $CLIENT_OUT/rig.new \
      $CLIENT_OUT/rig && tar -C $CLIENT_OUT/rig.new -xf - && for f in $CLIENT_OUT/rig.new/*; do \
      cat \"\$f\" >> $CLIENT_OUT/rig/\$(basename \"\$f\"); done && rm -rf $CLIENT_OUT/rig.new" \
      && ssh "$RIG" "rm -rf $RIG_LOGDIR" && say "rig logs in $CLIENT_OUT/rig on the client"
  fi
}

client_arm() {  # client_arm ARM: start client_arm.sh detached on the client, follow its log until it writes ARM.rc
  local arm=$1 seen=0 n rc
  local budget_var=BUDGET_MIN_$arm
  # only the setsid command goes to the background, its output in the log, so this ssh returns at once
  cl "mkdir -p $CLIENT_OUT && cd $CAC || exit 1; OUT=$CLIENT_OUT ARM=$arm UPSTREAM_PORT=$CLIENT_PORT \
    PROXY_PORT=$PROXY_PORT TASK_SET=$TASK_SET TASKS=$TASKS TIME_CAP_MIN=$TIME_CAP_MIN \
    BUDGET_MIN=${!budget_var:-${BUDGET_MIN:-}} CTX=$CTX CPU_SHARES=$CPU_SHARES IMAGE=$IMAGE LABEL=$LABEL \
    SAMPLING='${SAMPLING:-}' setsid nohup bash client_arm.sh > $CLIENT_OUT/$arm.log 2>&1 < /dev/null & sleep 1" \
    || { say "could not start the client for $arm"; return 1; }
  CLIENT_ARM=$arm
  while true; do
    sleep 15
    n=$(cl "wc -l < $CLIENT_OUT/$arm.log" 2> /dev/null) || continue
    if [ "$n" -gt "$seen" ]; then
      cl "sed -n '$((seen + 1)),${n}p' $CLIENT_OUT/$arm.log" 2> /dev/null | tee -a "$OUT/run.log" && seen=$n
    fi
    rc=$(cl "cat $CLIENT_OUT/$arm.rc 2> /dev/null || { test -f $CLIENT_OUT/$arm.pid && kill -0 \$(cat $CLIENT_OUT/$arm.pid) \
      2> /dev/null && echo running; } || echo gone" 2> /dev/null) || continue
    case $rc in
      running) ;;
      gone) say "client arm $arm ended without an exit code"; CLIENT_ARM=; return 1;;
      *) CLIENT_ARM=; say "client arm $arm done (run.py exit $rc)"; return 0;;
    esac
  done
}

client_replay() {  # client_replay ARM: replay.py run detached on the client against the arm's server, followed to its exit
  local label=${REPLAY_LABEL:-$1} rc
  cl "mkdir -p $CLIENT_OUT && cd $CAC || exit 1; rm -f $CLIENT_OUT/replay-$label.rc; setsid nohup bash -c \
    'nice -n 19 python3 -I replay.py run --log $REPLAY_LOG --traces $REPLAY_TRACES \
    --server http://127.0.0.1:$CLIENT_PORT --label $label --out $CLIENT_OUT/replay.jsonl; \
    echo \$? > $CLIENT_OUT/replay-$label.rc' > $CLIENT_OUT/replay-$label.log 2>&1 < /dev/null & \
    echo \$! > $CLIENT_OUT/replay-$label.pid; sleep 1" || { say "could not start the replay for $label"; return 1; }
  CLIENT_REPLAY=$label
  say "replay $label: $REPLAY_TRACES"
  while true; do
    sleep 15
    rc=$(cl "cat $CLIENT_OUT/replay-$label.rc 2> /dev/null || { kill -0 \$(cat $CLIENT_OUT/replay-$label.pid) \
      2> /dev/null && echo running; } || echo gone" 2> /dev/null) || continue
    case $rc in
      running) ;;
      gone) say "replay $label ended without an exit code"; CLIENT_REPLAY=; return 1;;
      *) CLIENT_REPLAY=; say "replay $label done (exit $rc): $(cl "tail -1 $CLIENT_OUT/replay-$label.log")"
         [ "$rc" = 0 ]; return;;
    esac
  done
}

client_stop() {  # stop a client arm or replay still running (this script interrupted)
  if [ -n "${CLIENT_REPLAY:-}" ]; then
    cl "kill -- -\$(cat $CLIENT_OUT/replay-$CLIENT_REPLAY.pid) 2> /dev/null" 2> /dev/null
    say "replay $CLIENT_REPLAY stopped"
  fi
  [ -n "${CLIENT_ARM:-}" ] || return 0
  cl "p=\$(cat $CLIENT_OUT/$CLIENT_ARM.pid 2> /dev/null) && kill -- -\$p 2> /dev/null; sleep 3; \
    docker ps -aq --filter name=^acrel-$CLIENT_ARM- | xargs -r docker rm -f > /dev/null" 2> /dev/null
  say "client arm $CLIENT_ARM stopped"
}

on_exit() {
  client_stop
  if [ -n "${RUNNING:-}" ]; then server_stop > /dev/null 2>&1; rig_end > /dev/null 2>&1; fi
  relay_stop
}
trap on_exit EXIT
trap 'exit 130' INT TERM

if has prep; then
  say "prep: bench files to $CLIENT:$CLIENT_DIR"
  tar -C "$HERE/.." --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py \
    | cl "mkdir -p $CLIENT_DIR && tar -C $CLIENT_DIR -xf -" || { say "copy failed"; exit 1; }
  say "prep: sandbox image $IMAGE on the client"
  cl "mkdir -p $CLIENT_OUT && cd $CAC && docker build -q --label $LABEL -t $IMAGE -f docker/Dockerfile . \
    > $CLIENT_OUT/image-build.log 2>&1" || { say "image build failed ($CLIENT_OUT/image-build.log on the client)"; exit 1; }
  say "prep: validating the $TASK_SET tasks in the image"
  cl "mkdir -p $CLIENT_OUT && cd $CAC && nice -n 19 python3 validate.py --task-set $TASK_SET --sandbox docker \
    --image $IMAGE --docker-label $LABEL --docker-prefix acrel-val --docker-cpu-shares $CPU_SHARES --jobs 2 \
    --out $CLIENT_OUT/validation.json > $CLIENT_OUT/validation.txt 2>&1; rc=\$?; tail -1 $CLIENT_OUT/validation.txt; \
    exit \$rc" | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "task validation failed (validation.txt on the client)"; exit 1; }
fi

if has run; then
  relay_start
  RUNNING=1
  for arm in $ARMS; do
    say "arm $arm"
    server_start "$arm" || { say "server $arm did not start; stopping"; exit 1; }
    if [ -n "${REPLAY_LOG:-}" ]; then
      client_replay "$arm" || { say "replay of $arm failed; stopping"; exit 1; }
    else
      client_arm "$arm" || { say "arm $arm failed; stopping"; exit 1; }
    fi
    server_stop
  done
  rig_end
  RUNNING=
  relay_stop
fi

if has report && [ -n "${REPLAY_LOG:-}" ]; then
  say "replay report on the client"
  cl "cd $CAC && nice -n 19 python3 -I replay.py summary $CLIENT_OUT/replay.jsonl --log $REPLAY_LOG \
    ${REPLAY_REF:+--reference $REPLAY_REF} > $CLIENT_OUT/replay.md && cat $CLIENT_OUT/replay.md" | tee -a "$OUT/run.log"
elif has report; then
  say "report on the client"
  cl "docker run --rm --label $LABEL --cpu-shares $CPU_SHARES -v $CLIENT_DIR:$CLIENT_DIR:ro -v $CLIENT_OUT:$CLIENT_OUT \
    $IMAGE bash $CAC/client_report.sh $CLIENT_OUT ${ARMS// /,}" | tee -a "$OUT/run.log"
fi

if has collect; then
  say "collect: $CLIENT:$CLIENT_OUT -> $OUT"
  cl "tar -C $CLIENT_OUT --exclude=./work -cf - ." | tar -C "$OUT" -xf - && say "results in $OUT"
fi
