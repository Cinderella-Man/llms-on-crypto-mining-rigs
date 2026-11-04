#!/usr/bin/env bash
# Copy agent-check proxy logs to the analysis machine.
#
#   RIG="ssh <host>" WORK=/path/on/that/machine ./copy_in.sh name=dir-with-proxy-logs [name=dir ...]
#
# Puts each folder's proxy.jsonl and proxy.jsonl.bodies.gz in $WORK/in/<name>/ and prints their md5 sums.
set -euo pipefail
: "${RIG:?RIG is the ssh command that reaches the analysis machine}" "${WORK:?}"
for spec in "$@"; do
  name=${spec%%=*}
  dir=${spec#*=}
  $RIG "mkdir -p '$WORK/in/$name'"
  for f in proxy.jsonl proxy.jsonl.bodies.gz; do
    cat "$dir/$f" | $RIG "cat > '$WORK/in/$name/$f'"
  done
  md5sum "$dir/proxy.jsonl" "$dir/proxy.jsonl.bodies.gz"
  $RIG "cd '$WORK/in/$name' && md5sum proxy.jsonl proxy.jsonl.bodies.gz"
done
