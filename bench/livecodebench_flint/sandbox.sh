#!/bin/bash
# Run a command (evaluate.py) with no network: bubblewrap gives it its own network, PID and IPC namespaces, the
# system read-only, an empty $HOME and /tmp, the listed paths read-only and OUT_DIR writable. Every process is
# limited to MEM_GB of address space (default 8), so a runaway program cannot take the machine's memory.
#
# usage: sandbox.sh OUT_DIR READ_ONLY_PATH [READ_ONLY_PATH ...] -- COMMAND [ARG ...]
# Pass every path the command needs: this folder's parent (bench/), LCB_ROOT, the problems file, the raw outputs,
# and the Python environment (its sys.prefix and sys.base_prefix) when it lives under $HOME or /tmp.
set -euo pipefail
out=$(realpath "$1")
shift
mkdir -p "$out"
binds=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
  p=$(realpath "$1")
  binds+=(--ro-bind "$p" "$p")
  shift
done
[ "${1:-}" = "--" ] || { echo "usage: sandbox.sh OUT_DIR PATH... -- COMMAND [ARG...]" >&2; exit 2; }
shift
ulimit -v $(( ${MEM_GB:-8} * 1024 * 1024 ))
exec bwrap --unshare-all --die-with-parent --new-session \
  --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs "$HOME" \
  "${binds[@]}" --bind "$out" "$out" --chdir "$out" \
  --setenv HOME /tmp --setenv LCB_ROOT "${LCB_ROOT:-/lcb/LiveCodeBench}" \
  "$@"
