#!/usr/bin/env bash
# first_turn_request.sh OUT_FILE: builds the agent's first-turn request (first_turn_request.py, Q607) as a chat completion
# body in OUT_FILE, for the draft smoke tests (docs/data/2026-11-03_*/run/smoke_*.sh send it before their short requests).
# The machine this runs on only runs ssh: the task folder is rebuilt, and the body built, on BUILD_SSH's machine (any
# Linux machine with python3, the agent check's client for instance) from the polyglot tasks with the command the README
# gives for the bugfix set,
#   python3 build_bugfix.py --out <folder>/tasks/bugfix
# in a temporary folder there, removed afterwards. Nothing of the task is kept here but the request in OUT_FILE: don't
# commit it. Environment:
#   BUILD_SSH      the ssh command that reaches that machine, default "ssh $CLIENT"
#   LONG_TOKENS    the size to aim for in the server's tokens (2780: the first bugfix task's first request in the S12 and
#                  S18 runs); LONG_CPT characters per token (3.6), LONG_TEMPLATE_TOKENS what
#                  the chat template adds (250), LONG_MAX_TOKENS the reply's cap (256), LONG_TASK a
#                  task folder name (default: the first of tasks/bugfix/order.txt)
# The summary line (estimated tokens) goes to stderr. The server's own count comes back in the reply's usage.
set -uo pipefail
OUT=${1:?usage: first_turn_request.sh OUT_FILE}
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD_SSH=${BUILD_SSH:-ssh ${CLIENT:?set BUILD_SSH or CLIENT}}
args="--tokens ${LONG_TOKENS:-2780} --chars-per-token ${LONG_CPT:-3.6} --template-tokens ${LONG_TEMPLATE_TOKENS:-250} --max-tokens ${LONG_MAX_TOKENS:-256}"
[ -n "${LONG_TASK:-}" ] && args="$args --task $LONG_TASK"
mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086
tar -C "$HERE" -cf - first_turn_request.py build_bugfix.py bugfix_bugs.json tasks/polyglot \
  | $BUILD_SSH "d=\$(mktemp -d /tmp/first-turn.XXXXXX) && trap 'rm -rf \"\$d\"' EXIT && tar -xf - -C \"\$d\" \
    && python3 -I \"\$d/build_bugfix.py\" --out \"\$d/tasks/bugfix\" > /dev/null \
    && python3 -I \"\$d/first_turn_request.py\" --tasks \"\$d/tasks/bugfix\" $args" > "$OUT"
rc=${PIPESTATUS[1]}
if [ "$rc" != 0 ] || [ ! -s "$OUT" ]; then
  echo "first_turn_request.sh: building the request failed (rc $rc)" >&2
  rm -f "$OUT"
  exit 1
fi
