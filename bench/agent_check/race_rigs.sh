#!/usr/bin/env bash
# race_rigs.sh OUT: the first-finish race across rigs (Q1195): one agent, the preset's llama-server on every rig of
# RIGS, loaded once, then the task set once per run of RUNS, in that order (default "one race2 one race2"):
#   one    pi -> proxy.py -> one rig's server; the k-th `one` run uses rig ((k - 1) mod N) + 1, so the rigs take turns
#   raceN  pi -> race_proxy.py -> the first N rigs' servers, a seed each; the first whole reply wins and the rest are
#          cancelled (RACE_ARGS adds race_proxy.py options)
#   raceNr the same with race_proxy.py --win-at reasoning: the first sample to end its reasoning wins (Q1271)
# A run is labelled <run>-<k>, its k-th time (one-1 race2-1 one-2 race2-2). The machines are release_check.sh's:
#   - each rig runs llama-server on its loopback (rig_server.sh, with its monitors and the RAM watchdog);
#   - the client (any Linux machine with Docker) runs pi, the proxy and the tasks' tests in containers (client_arm.sh);
#   - this machine only runs ssh: it copies the files, starts and stops everything, and relays the client's requests,
#     for rig k through -L RELAY_PORT+k-1 -> rig k's 127.0.0.1:RIG_PORT and -R client 127.0.0.1:CLIENT_PORT+k-1 ->
#     that. Neither machine has to reach another, and no server listens beyond its rig's loopback.
#
#   RIGS="<rig 1> <rig 2>" CLIENT=<alias> RIG_PORT=… RELAY_PORT=… CLIENT_PORT=… PROXY_PORT=… CLIENT_DIR=… CLIENT_OUT=… \
#     BIN=<llama-server on the rigs> MODEL=<first shard on the rigs> MODEL_DRAFT=<MTP draft on the rigs> \
#     bash bench/agent_check/race_rigs.sh OUT
#
# The server line is the preset's, the same on every rig: PRESET (default release/flash-next-v3/settings.env),
# PRESET_ENV and PRESET_ARGS (its nine-card split included) with -md MODEL_DRAFT and EXTRA_ARGS (default "--metrics
# --no-webui", as the release check has them); ARGS_<k> and ENV_<k> add to rig k's line. RIG_PORT_<k> overrides rig
# k's port (two "rigs" on one machine in a dry run).
# Optional: RUNS, RACE_ARGS, STEPS ("prep run report collect", release_check.sh's steps; run loads the servers, runs
# every run and unloads them), TASK_SET (bugfix), TASKS (all), CTX (65536, pi's window = the preset's -c),
# TIME_CAP_MIN (15), TURN_CAP (40), BUDGET_MIN (none: no attempt of a run starts after that many minutes),
# RIG_LOGDIR (/tmp/agent-check-race: rig k logs into RIG_LOGDIR-k; the logs come back to CLIENT_OUT/rig for rig 1 and
# CLIENT_OUT/rig<k> for the others), MIN_AVAIL_MB (60, the preset's release runs' floor: the host goes below 400 MiB on
# this line), WD_ENV, LOAD_TIMEOUT_S (1200), CPU_SHARES (256), IMAGE (agent-check:0.87.1), LABEL (pascal-quants=1),
# PREFIX (acrace), SAMPLING, SSH (the ssh command, default ssh; a dry run can pass a wrapper).
# Dry run without a rig: RIGS="<client> <client>", RIG_PORT_1/RIG_PORT_2 two free ports there, BIN=<CLIENT_DIR>/
# agent_check/mock_server.py, MODEL=MODEL_DRAFT=/dev/null, PRESET=<a file with PRESET_ENV='' and PRESET_ARGS='' or the
# mock's options>, ARGS_1/ARGS_2 the mocks' speeds (e.g. "--decode-ms 2", "--decode-ms 6").
# Writes on the client, copied to OUT by collect: compare.md (the runs side by side), table.md, gonogo.md (rig 1's
# memory and server log), race.md (race_proxy.py summary, the one-rig runs as its --one), <label>.log, res/, rig*/.
# Resumable: run it again with the same CLIENT_OUT and finished attempts are skipped.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: race_rigs.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RIGS:?}" "${CLIENT:?}" "${RIG_PORT:?}" "${RELAY_PORT:?}" "${CLIENT_PORT:?}" "${PROXY_PORT:?}" "${CLIENT_DIR:?}" \
  "${CLIENT_OUT:?}" "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}"
read -r -a RIGA <<< "$RIGS"
N=${#RIGA[@]}
RUNS=${RUNS:-one race2 one race2}; STEPS=${STEPS:-prep run report collect}
TASK_SET=${TASK_SET:-bugfix}; TASKS=${TASKS:-all}; CTX=${CTX:-65536}; TIME_CAP_MIN=${TIME_CAP_MIN:-15}
TURN_CAP=${TURN_CAP:-40}; CPU_SHARES=${CPU_SHARES:-256}; IMAGE=${IMAGE:-agent-check:0.87.1}
LABEL=${LABEL:-pascal-quants=1}; PREFIX=${PREFIX:-acrace}; RIG_LOGDIR=${RIG_LOGDIR:-/tmp/agent-check-race}
EXTRA_ARGS=${EXTRA_ARGS---metrics --no-webui}; SSH=${SSH:-ssh}
PRESET=${PRESET:-$HERE/../../release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
CAC="$CLIENT_DIR/agent_check"
SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3)
mkdir -p "$OUT"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
cl() { $SSH "${SSH_OPTS[@]}" "$CLIENT" "$@"; }
rig() { local k=$1; shift; $SSH "${SSH_OPTS[@]}" "${RIGA[k - 1]}" "$@"; }
rport() { local v=RIG_PORT_$1; echo "${!v:-$RIG_PORT}"; }
race_n() { local n=${1#race}; echo "${n%r}"; }  # the rigs of a race run: race2 and race2r -> 2
race_args() { if [[ $1 == *r ]]; then echo "${RACE_ARGS:-} --win-at reasoning"; else echo "${RACE_ARGS:-}"; fi; }
for r in $RUNS; do
  case $r in
    one) ;;
    race[0-9]*) [[ $r =~ ^race[0-9]+r?$ ]] && [ "$(race_n "$r")" -ge 2 ] && [ "$(race_n "$r")" -le "$N" ] \
      || { say "run $r needs 2 to $N rigs"; exit 1; };;
    *) say "unknown run $r (one, raceN or raceNr)"; exit 1;;
  esac
done

relay_loop() {  # relay_loop NAME ssh-arguments…: keeps one forward up until killed
  local name=$1
  shift
  while true; do
    $SSH -N "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes "$@" &
    echo $! > "$OUT/relay-$name.pid"
    wait $!
    sleep 3
  done
}

RELAYS=()
relay_start() {
  local k
  for k in $(seq "$N"); do
    (relay_loop "L$k" -L "127.0.0.1:$((RELAY_PORT + k - 1)):127.0.0.1:$(rport "$k")" "${RIGA[k - 1]}") \
      >> "$OUT/relay.log" 2>&1 &
    RELAYS+=($!)
    sleep 2
    (relay_loop "R$k" -R "127.0.0.1:$((CLIENT_PORT + k - 1)):127.0.0.1:$((RELAY_PORT + k - 1))" "$CLIENT") \
      >> "$OUT/relay.log" 2>&1 &
    RELAYS+=($!)
    sleep 3
    say "relay $k: client 127.0.0.1:$((CLIENT_PORT + k - 1)) -> here 127.0.0.1:$((RELAY_PORT + k - 1)) -> ${RIGA[k - 1]} 127.0.0.1:$(rport "$k")"
  done
}

relay_stop() {
  local p f
  for p in "${RELAYS[@]}"; do kill "$p" 2> /dev/null; done
  for f in "$OUT"/relay-*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2> /dev/null; rm -f "$f"; done
  RELAYS=()
}

rig_server() {  # rig_server K ACTION [extra env]: rig_server.sh on rig k
  local k=$1 action=$2
  shift 2
  rig "$k" "LOGDIR=$RIG_LOGDIR-$k ACTION=$action PORT=$(rport "$k") $* bash -s" < "$HERE/rig_server.sh"
}

servers_start() {  # every rig's server, loading in parallel, then /health through each relay from the client
  local k pids=() ok=0 ea ev
  for k in $(seq "$N"); do
    ea=ARGS_$k; ev=ENV_$k
    SARGS="$PRESET_ARGS -md $MODEL_DRAFT $EXTRA_ARGS ${!ea:-}"; SENV="$PRESET_ENV ${!ev:-}"
    say "starting rig $k (${RIGA[k - 1]}): ${SENV:+$SENV }$BIN $SARGS -m $MODEL"
    rig_server "$k" start "MIN_AVAIL_MB=${MIN_AVAIL_MB:-60} ${WD_ENV:-} LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-1200} ARM=rig$k \
      BIN=$BIN MODEL=$MODEL SERVER_ENV='$SENV' SERVER_ARGS='$SARGS'" > "$OUT/start-$k.log" 2>&1 &
    pids+=($!)
  done
  for k in $(seq "$N"); do
    wait "${pids[k - 1]}" || ok=1
    tee -a "$OUT/run.log" < "$OUT/start-$k.log"
  done
  [ "$ok" = 0 ] || return 1
  for k in $(seq "$N"); do
    for _ in $(seq 40); do
      cl "curl -sf 127.0.0.1:$((CLIENT_PORT + k - 1))/health" > /dev/null 2>&1 && continue 2
      sleep 3
    done
    say "the client does not reach rig $k through the relay (relay.log)"
    return 1
  done
  say "the client reaches every rig through the relays"
}

servers_stop() {
  local k
  for k in $(seq "$N"); do rig_server "$k" stop 2>&1 | tee -a "$OUT/run.log"; done
}

rigs_end() {  # monitors stopped; rig k's logs appended to the client's CLIENT_OUT/rig (rig 1) or rig<k>, then removed
  local k dest
  for k in $(seq "$N"); do
    rig_server "$k" end 2>&1 | tee -a "$OUT/run.log"
    dest=$CLIENT_OUT/rig$([ "$k" = 1 ] || echo "$k")
    if rig "$k" "test -d $RIG_LOGDIR-$k"; then
      rig "$k" "tar -C $RIG_LOGDIR-$k -cf - ." | cl "rm -rf $dest.new && mkdir -p $dest.new $dest && \
        tar -C $dest.new -xf - && for f in $dest.new/*; do cat \"\$f\" >> $dest/\$(basename \"\$f\"); done && \
        rm -rf $dest.new" && rig "$k" "rm -rf $RIG_LOGDIR-$k" && say "rig $k's logs in $dest on the client"
    fi
  done
}

client_run() {  # client_run LABEL PORTS RACE [RACE_ARGS]: client_arm.sh detached on the client, followed to LABEL.rc
  local label=$1 ports=$2 race=$3 rargs=${4:-} seen=0 n rc up='' ups=''
  if [ "$race" = 1 ]; then ups=$ports; else up=$ports; fi
  # only the setsid command goes to the background, its output in the log, so this ssh returns at once
  cl "mkdir -p $CLIENT_OUT && cd $CAC || exit 1; OUT=$CLIENT_OUT ARM=$label UPSTREAM_PORT=$up \
    UPSTREAM_PORTS='$ups' RACE_ARGS='$rargs' PROXY_PORT=$PROXY_PORT TASK_SET=$TASK_SET TASKS=$TASKS \
    TIME_CAP_MIN=$TIME_CAP_MIN TURN_CAP=$TURN_CAP BUDGET_MIN=${BUDGET_MIN:-} CTX=$CTX CPU_SHARES=$CPU_SHARES \
    IMAGE=$IMAGE LABEL=$LABEL PREFIX=$PREFIX SAMPLING='${SAMPLING:-}' \
    setsid nohup bash client_arm.sh > $CLIENT_OUT/$label.log 2>&1 < /dev/null & sleep 1" \
    || { say "could not start the client for $label"; return 1; }
  CLIENT_ARM=$label
  while true; do
    sleep 15
    n=$(cl "wc -l < $CLIENT_OUT/$label.log" 2> /dev/null) || continue
    if [ "$n" -gt "$seen" ]; then
      cl "sed -n '$((seen + 1)),${n}p' $CLIENT_OUT/$label.log" 2> /dev/null | tee -a "$OUT/run.log" && seen=$n
    fi
    rc=$(cl "cat $CLIENT_OUT/$label.rc 2> /dev/null || { test -f $CLIENT_OUT/$label.pid && \
      kill -0 \$(cat $CLIENT_OUT/$label.pid) 2> /dev/null && echo running; } || echo gone" 2> /dev/null) || continue
    case $rc in
      running) ;;
      gone) say "run $label ended without an exit code"; CLIENT_ARM=; return 1;;
      *) CLIENT_ARM=; say "run $label done (run.py exit $rc)"; return 0;;
    esac
  done
}

client_stop() {  # stop a run still going on the client (this script interrupted)
  [ -n "${CLIENT_ARM:-}" ] || return 0
  cl "p=\$(cat $CLIENT_OUT/$CLIENT_ARM.pid 2> /dev/null) && kill -- -\$p 2> /dev/null; sleep 3; \
    docker ps -aq --filter name=^$PREFIX-$CLIENT_ARM- | xargs -r docker rm -f > /dev/null" 2> /dev/null
  say "run $CLIENT_ARM stopped"
}

on_exit() {
  client_stop
  if [ -n "${RUNNING:-}" ]; then servers_stop > /dev/null 2>&1; rigs_end > /dev/null 2>&1; fi
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
  cl "cd $CAC && nice -n 19 python3 validate.py --task-set $TASK_SET --sandbox docker --image $IMAGE \
    --docker-label $LABEL --docker-prefix $PREFIX-val --docker-cpu-shares $CPU_SHARES --jobs 2 \
    --out $CLIENT_OUT/validation.json > $CLIENT_OUT/validation.txt 2>&1; rc=\$?; tail -1 $CLIENT_OUT/validation.txt; \
    exit \$rc" | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { say "task validation failed (validation.txt on the client)"; exit 1; }
fi

LABELS=()
declare -A seen
for r in $RUNS; do seen[$r]=$((${seen[$r]:-0} + 1)); LABELS+=("$r-${seen[$r]}"); done
if has run; then
  relay_start
  RUNNING=1
  servers_start || { say "the servers did not start; stopping"; exit 1; }
  unset seen
  declare -A seen
  for r in $RUNS; do
    seen[$r]=$((${seen[$r]:-0} + 1))
    label=$r-${seen[$r]}
    if [ "$r" = one ]; then
      k=$(((seen[one] - 1) % N + 1))
      say "run $label: rig $k alone (${RIGA[k - 1]})"
      client_run "$label" $((CLIENT_PORT + k - 1)) 0 || { say "run $label failed; stopping"; exit 1; }
    else
      ports=$(for k in $(seq "$(race_n "$r")"); do printf '%s ' $((CLIENT_PORT + k - 1)); done)
      ra=$(race_args "$r")
      say "run $label: race of rigs 1 to $(race_n "$r")${ra:+ ($ra)}"
      client_run "$label" "${ports% }" 1 "$ra" || { say "run $label failed; stopping"; exit 1; }
    fi
  done
  servers_stop
  rigs_end
  RUNNING=
  relay_stop
fi

if has report; then
  say "report on the client"
  cl "docker run --rm --label $LABEL --cpu-shares $CPU_SHARES -v $CLIENT_DIR:$CLIENT_DIR:ro -v $CLIENT_OUT:$CLIENT_OUT \
    $IMAGE bash -c 'bash $CAC/client_report.sh $CLIENT_OUT $(IFS=,; echo "${LABELS[*]}") > /dev/null; \
    test -s $CLIENT_OUT/proxy.samples.jsonl && python3 -I $CAC/race_proxy.py summary $CLIENT_OUT/proxy.samples.jsonl \
    --one $CLIENT_OUT/proxy.jsonl > $CLIENT_OUT/race.md; cat $CLIENT_OUT/compare.md $CLIENT_OUT/race.md'" \
    | tee -a "$OUT/run.log"
fi

if has collect; then
  say "collect: $CLIENT:$CLIENT_OUT -> $OUT"
  cl "tar -C $CLIENT_OUT --exclude=./work -cf - ." | tar -C "$OUT" -xf - && say "results in $OUT"
fi
