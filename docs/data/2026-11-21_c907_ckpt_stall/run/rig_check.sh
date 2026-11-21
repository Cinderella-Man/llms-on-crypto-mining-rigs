#!/usr/bin/env bash
# rig_check.sh: C907's rig check (QUEUES C946, registry Q1685): R183's Ornith arm again on one rig, on C907's build, once per arm,
# each arm its own load of the v5 preset (release/ornith-v5/settings.env) with one change:
#   off  the preset unchanged (LLAMA_SERVER_CKPT_NOWAIT unset)
#   on   PRESET_ENV plus LLAMA_SERVER_CKPT_NOWAIT=1
#   cms  PRESET_ARGS plus --checkpoint-min-step 7680 (C945, Q1686; the switch unset)
# In each arm: R183's prompts at TARGETS (the 8,000-token control first, as R183), greedy, seed 1546, R190's question,
# through bench/recall_depth (remote.sh on the rig, recall_depth.py here, an ssh tunnel to the rig's loopback port).
# Then, per arm, R183's calls.py over the server's log (each prompt's calls, its checkpoint counters) and a comparison
# of every reply against the first arm's (compare_arms below): the same reasoning and answer, and the prompt seconds,
# ckpt_save_ms and checkpoint counts side by side.
# Env: RIG (ssh alias), BIN (C907's bin folder on the rig), MODEL (Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the
# rig), OUT (results here), ARMS (default "off on"), TARGETS (default "8000 16000 32000 47000"), LPORT (default 18080),
# STOP_AT (a time for date -d: no arm starts after it, and a running arm's client stops there). Every call to the rig is
# under a timeout.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/../../../..
TOOL=$ROOT/bench/recall_depth
: "${RIG:?}" "${BIN:?}" "${MODEL:?}" "${OUT:?}"
ARMS=${ARMS:-off on}
TARGETS=${TARGETS:-8000 16000 32000 47000}
LPORT=${LPORT:-18080}
export RIG BIN MODEL PORT=8080 DRAFT='' CTX=49152 SEED=1546 MAX_TOKENS=8192 TIMEOUT_S=3600 ON_MISS=go
log() { echo "=== $(date '+%F %T') $*"; }
left() {
  [ -z "${STOP_AT:-}" ] && { echo 86400; return; }
  local s=$(( $(date -d "$STOP_AT" +%s) - $(date +%s) ))
  [ "$s" -lt 1 ] && s=1
  echo "$s"
}

preset() {  # preset ARM FILE: the v5 preset with the arm's one change
  cp "$ROOT/release/ornith-v5/settings.env" "$2"
  case $1 in
    off) ;;
    on)  echo 'PRESET_ENV="$PRESET_ENV LLAMA_SERVER_CKPT_NOWAIT=1"  # C907' >> "$2" ;;
    cms) echo 'PRESET_ARGS="$PRESET_ARGS --checkpoint-min-step 7680"  # C945' >> "$2" ;;
    *)   echo "unknown arm $1"; return 1 ;;
  esac
}

compare_arms() {  # one row per arm and prompt: the reply against the first arm's, the times and counters
  python3 -I - "$OUT" $ARMS <<'PY'
import glob, json, os, sys
out, arms = sys.argv[1], sys.argv[2:]
def load(arm):
    res = {}
    for f in glob.glob(os.path.join(out, arm, 'client', '*-1.json')):
        d = json.load(open(f))
        res[d['row']['name']] = d
    return res
runs = {a: load(a) for a in arms}
ref = runs[arms[0]]
print('\t'.join(['arm', 'prompt', 'prompt_tokens', 'reply_same_as_' + arms[0], 'prompt_s', 'ckpt_save_s', 'ckpt_created_n',
                 'ckpt_incall_n', 'ckpt_nowait_n', 'completion_tokens']))
for a in arms:
    for name in sorted(runs[a], key=lambda n: int(runs[a][n]['row']['prompt_tokens'])):
        d, r = runs[a][name], ref.get(name)
        msg = lambda x: x['reply']['choices'][0]['message']
        same = r is not None and msg(d).get('content') == msg(r).get('content') and \
            msg(d).get('reasoning_content') == msg(r).get('reasoning_content')
        t = d['reply'].get('timings', {})
        print('\t'.join(str(x) for x in [a, name, d['row']['prompt_tokens'], 'yes' if same else 'NO',
              round(t.get('prompt_ms', 0) / 1000, 2), round(t.get('ckpt_save_ms', 0) / 1000, 2), t.get('ckpt_created_n'),
              t.get('ckpt_incall_n'), t.get('ckpt_nowait_n', 0), d['reply'].get('usage', {}).get('completion_tokens')]))
PY
}

mkdir -p "$OUT"
for arm in $ARMS; do
  [ "$(left)" -le 1500 ] && { log "STOP_AT too close: arm $arm not started"; break; }
  A=$OUT/$arm
  mkdir -p "$A"
  preset "$arm" "$A/settings.env" || exit 1
  log "arm $arm: load"
  if PRESET=$A/settings.env timeout 120 bash "$TOOL/remote.sh" put && bash "$TOOL/remote.sh" start \
      && timeout 1300 bash "$TOOL/remote.sh" wait; then
    timeout "$(left)" ssh -N -o ExitOnForwardFailure=yes -L "$LPORT:127.0.0.1:$PORT" "$RIG" &
    tun=$!
    sleep 5
    log "arm $arm: prompts $TARGETS"
    # shellcheck disable=SC2086
    QUESTION=kinds NAME=kinds URL=http://127.0.0.1:$LPORT OUT=$A/client \
      timeout "$(left)" python3 -I "$TOOL/recall_depth.py" run $TARGETS
    log "arm $arm: client exit $?"
    kill "$tun" 2> /dev/null
  else
    log "arm $arm: load failed"
  fi
  timeout 120 bash "$TOOL/remote.sh" stop
  OUT=$A/rig timeout 600 bash "$TOOL/remote.sh" collect
  python3 -I "$ROOT/docs/data/2026-11-20_r183_ornith_recall/run/calls.py" "$A/rig/server.log" "$A/client" > "$A/prompt_calls.tsv"
  log "arm $arm: done"
done
compare_arms | tee "$OUT/arms.tsv"
log "all done"
