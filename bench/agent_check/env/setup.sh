#!/usr/bin/env bash
# env/setup.sh AC_HOME: create the shared test environments the sandbox mounts, in AC_HOME (~150 MB):
#   AC_HOME/node/node_modules  jest, babel-jest and Exercism's babel preset (env/package.json, locked versions)
#   AC_HOME/py                 a relocatable Python venv with pytest (env/requirements.txt), on the system python3
# Needs node/npm and uv. Run once per machine; the tasks' workspaces link to these, nothing is installed per task.
set -euo pipefail
AC_HOME=${1:?usage: env/setup.sh AC_HOME}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$AC_HOME/node"
cp "$HERE/package.json" "$HERE/package-lock.json" "$AC_HOME/node/"
(cd "$AC_HOME/node" && npm ci --no-audit --no-fund --ignore-scripts)
uv venv -q --relocatable --python "$(command -v python3)" "$AC_HOME/py"
VIRTUAL_ENV="$AC_HOME/py" uv pip install -q -r "$HERE/requirements.txt"
du -sh "$AC_HOME/node/node_modules" "$AC_HOME/py"
