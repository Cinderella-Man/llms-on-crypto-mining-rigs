#!/usr/bin/env bash
# preset_agent.sh: the coding-agent check of a model's preset file as it stands, one arm after another, then the report.
# The server line, the context (-c) and the micro-batch (-ub) all come from the preset (PRESET, a release/*/settings.env:
# PRESET_ENV, PRESET_ARGS), so a run measures the preset a user would start and needs no hand edit when the preset
# changes. The agent's context window is the line's -c. The harness is the agent check's release check
# (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh, whose rig_server.sh and client_arm.sh live here): it
# starts "$COMMON $ARGS_<arm>" with the environment REL_ENV, where COMMON is its run line (-ngl 99 -fa on -c CTX ...,
# the sampling, --metrics --no-webui). This script checks that the preset's PRESET_ARGS begin with that run line word
# for word (without --metrics --no-webui) and hands the harness the rest, with -md DRAFT after -lm dio, so the started
# line is the preset's line plus the harness's --metrics --no-webui and -md.
#
# Arms (ARMS, default "pre"; names of a-z and 0-9):
#   pre, pre2, ...   the preset unchanged (any arm with neither ENV_<arm> nor SET_<arm>)
#   ENV_<arm>        switches for that arm, NAME=VALUE words: each replaces the variable of that name in PRESET_ENV
#                    (found exactly once) or, if the preset lacks it, is appended
#   SET_<arm>        flag values for that arm, "FLAG VALUE ..." pairs: each FLAG must stand exactly once in PRESET_ARGS
#                    with one value after it, which is replaced (e.g. SET_c64='-c 65536 -ub 512')
# Each arm's line is written to OUT/<arm>/dry_line.txt (the environment, the binary, its arguments: the form of
# R243's and R244's dry lines) before it runs. DRY=1 writes them for every arm and stops, without touching a machine.
#
# Environment:
#   OUT          a local folder
#   DRAFT        the MTP draft's path on the rig (the preset's PRESET_DRAFT under the rig's model folder)
#   BIN or NEW   llama-server on the rig, or the build's folder under /opt/llama-builds (BIN wins)
#   PRESET       (release/flash-next-v5/settings.env)
#   STEPS        ("agent report"; "prep" or "files" first to copy the bench files to the client, as R156's driver)
#   AGENT_BY     (21:00) no arm starts later: an arm of at most 40 minutes plus a 12-minute task ends by 22:00
#   REPORT_ARMS  (ARMS)
#   for a run, also: RIG, CLIENT, MODEL (the first shard on the rig), RIGDIR (a folder on the rig for the logs),
#   CLIENT_DIR, CLIENT_OUT and the harness's ports RIG_PORT, RELAY_PORT, PROXY_PORT (CLIENT_PORT unless CLIENT=local),
#   CPU_SHARES (256), PREFIX (the client's container names; default the harness's own from CLIENT_OUT)
# Run from the repository root.
set -uo pipefail
: "${OUT:?}" "${DRAFT:?}"
PRESET=${PRESET:-release/flash-next-v5/settings.env}
ARMS=${ARMS:-pre}
STEPS=${STEPS:-agent report}
AGENT_BY=${AGENT_BY:-21:00}
REPORT_ARMS=${REPORT_ARMS:-$ARMS}
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
if [ -n "${BIN:-}" ]; then SERVER_BIN=$BIN; else : "${NEW:?BIN or NEW}"; SERVER_BIN=/opt/llama-builds/$NEW/bin/llama-server; fi
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
count() { local s=$1 n=0; while [[ $s == *"$2"* ]]; do s=${s#*"$2"}; n=$((n + 1)); done; echo $n; }
mkdir -p "$OUT"
[ -f "$PRESET" ] && [ -f "$RC" ] || { say "missing $PRESET or $RC"; exit 1; }
# shellcheck disable=SC1090
. "$PRESET"  # PRESET_ENV, PRESET_ARGS
[ -n "${PRESET_ARGS:-}" ] || { say "$PRESET sets no PRESET_ARGS"; exit 1; }

common_of() {  # common_of CTX: the harness's COMMON for that context, read from the harness itself
  local CTX=$1 COMMON
  eval "$(sed -n '/^COMMON="/,/"$/p' "$RC")"
  printf '%s' "$COMMON"
}

build() {  # build ARM: sets A_ENV, A_ARGS (the harness's ARGS_<arm>), A_CTX, A_LINE; returns 1 on any mismatch
  local arm=$1 envv="ENV_$1" setv="SET_$1" e=" $PRESET_ENV " a=" $PRESET_ARGS " w name flag val n common head
  if [ -n "${!envv:-}" ]; then
    for w in ${!envv}; do
      [[ $w == *=* ]] || { say "arm $arm: '$w' in $envv is not NAME=VALUE"; return 1; }
      name=${w%%=*}
      n=$(count "$e" " $name=")
      if [ "$n" = 1 ]; then
        local old=${e#*" $name="}; old=" $name=${old%% *} "
        e=${e/"$old"/" $w "}
      elif [ "$n" = 0 ]; then
        e="$e$w "
      else
        say "arm $arm: $name stands $n times in PRESET_ENV"; return 1
      fi
    done
  fi
  if [ -n "${!setv:-}" ]; then
    set -- ${!setv}
    while [ $# -gt 0 ]; do
      flag=$1; val=${2:-}
      [ -n "$val" ] || { say "arm $arm: $setv has no value after $flag"; return 1; }
      [ "$(count "$a" " $flag ")" = 1 ] || { say "arm $arm: '$flag' does not stand exactly once in PRESET_ARGS"; return 1; }
      local old=${a#*" $flag "}; old=" $flag ${old%% *} "
      [[ ${old#" $flag "} != -* ]] || { say "arm $arm: '$flag' has no value in PRESET_ARGS"; return 1; }
      a=${a/"$old"/" $flag $val "}
      shift 2
    done
  fi
  [ "$(count "$a" " -c ")" = 1 ] || { say "arm $arm: -c does not stand exactly once in the line"; return 1; }
  A_CTX=${a#*" -c "}; A_CTX=${A_CTX%% *}
  [[ $A_CTX =~ ^[0-9]+$ ]] || { say "arm $arm: -c $A_CTX is not a number"; return 1; }
  common=$(common_of "$A_CTX")
  head=" ${common% --metrics --no-webui} "
  [ "$head" != " $common " ] || { say "the harness's COMMON does not end in --metrics --no-webui"; return 1; }
  [[ $a == "$head"* ]] || { say "arm $arm: the line does not begin with the harness's run line:$head"; return 1; }
  a=${a#"$head"}
  [ "$(count " $a" " -lm dio ")" = 1 ] || { say "arm $arm: '-lm dio' does not stand exactly once"; return 1; }
  [ "$(count " $a" " -md ")" = 0 ] || { say "arm $arm: the preset already names a draft (-md)"; return 1; }
  a=" $a"; a=${a/" -lm dio "/" -lm dio -md $DRAFT "}
  A_ARGS=${a# }; A_ARGS=${A_ARGS% }
  e=${e# }; A_ENV=${e% }
  A_LINE="$A_ENV $SERVER_BIN $common $A_ARGS"
}

for arm in $ARMS; do
  [[ $arm =~ ^[a-z0-9]+$ ]] || { say "arm name '$arm': use a-z and 0-9"; exit 1; }
  build "$arm" || exit 1
  mkdir -p "$OUT/$arm"
  printf '%s\n' "$A_LINE" > "$OUT/$arm/dry_line.txt"
  say "arm $arm: -c $A_CTX; line $(md5sum < "$OUT/$arm/dry_line.txt" | cut -c1-32), $(wc -c < "$OUT/$arm/dry_line.txt") bytes"
done
[ "${DRY:-0}" = 1 ] && { say "DRY=1: lines written, nothing started"; exit 0; }

: "${RIG:?}" "${CLIENT:?}" "${MODEL:?}" "${RIGDIR:?}"
export BENCH=$PWD/bench/agent_check RIG CLIENT MODEL TIME_CAP_MIN=12 MIN_AVAIL_MB=60 \
  CPU_SHARES=${CPU_SHARES:-256} RIG_LOGDIR=$RIGDIR/agent-logs

if has prep; then
  say "prep"
  build "${ARMS%% *}" || exit 1
  env CTX="$A_CTX" ARMS="${ARMS%% *}" "ARGS_${ARMS%% *}=$A_ARGS" STEPS=prep bash $RC "$OUT/agent-prep" \
    > "$OUT/agent-prep.console" 2>&1 || { say "prep failed"; exit 1; }
  say "prep: $(tail -n 1 "$OUT/agent-prep.console")"
elif has files; then
  : "${CLIENT_DIR:?}"
  say "files: bench files to $CLIENT:$CLIENT_DIR"
  # CLIENT=local: this machine is the client (the harness's convention), so a local shell, not ssh
  if [ "$CLIENT" = local ]; then to_client() { bash -c "$1"; }; else to_client() { ssh "$CLIENT" "$1"; }; fi
  tar -C bench --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py -C ../scripts realloc_counts.py \
    | to_client "mkdir -p $CLIENT_DIR && tar -C $CLIENT_DIR -xf -" || { say "files: copy failed"; exit 1; }
fi

if has agent; then
  for arm in $ARMS; do
    if [ "$(date +%H%M)" -gt "${AGENT_BY/:/}" ]; then
      say "AGENT_SKIPPED: arm $arm not started, past $AGENT_BY (it would not end before the rig's evening stop)"
      continue
    fi
    build "$arm" || exit 1
    say "arm $arm: $SERVER_BIN, -c $A_CTX"
    env CTX="$A_CTX" REL_BIN="$SERVER_BIN" "ARGS_$arm=$A_ARGS" "BUDGET_MIN_$arm=40" REL_ENV="$A_ENV" \
      ARMS="$arm" STEPS=run bash $RC "$OUT/agent-$arm" > "$OUT/agent-$arm.console" 2>&1
    say "arm $arm: exit $?"
  done
fi
if has report; then
  build "${REPORT_ARMS%% *}" || exit 1
  say "report: $REPORT_ARMS"
  CTX=$A_CTX ARMS="$REPORT_ARMS" STEPS="report collect" bash $RC "$OUT/agent" > "$OUT/agent-report.console" 2>&1
  say "report: exit $?"
fi

ssh "$RIG" "rmdir $RIGDIR 2> /dev/null; date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'; \
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; \
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo" > "$OUT/rig_end.txt" 2>&1
say "PRESET_AGENT_DONE:$(tr '\n' ' ' < "$OUT/rig_end.txt")"
