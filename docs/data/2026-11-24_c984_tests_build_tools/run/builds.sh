#!/usr/bin/env bash
# C984 (Q1717): Miner Llama's check tools in every TESTS=1 tarball of scripts/build_llamacpp_docker.sh. Six builds in
# a row, each under a shared build lock, all with TESTS=1:
#   old  the v5 tip with the script before the change
#   new  the same commit with the changed script
#   off  the same with CHECK_TOOLS=0 (should be old's tarball again)
#   dup  the same with the extra targets earlier briefs passed (EXTRA_TARGETS="test-llama-archs test-cuda-keepalive")
#   v3   the v3.0 release tag (no pool tests in that tree) with the changed script
#   v2   the v2.0 release tag (only test-llama-archs of the eight) with the changed script
# Each build's tarball and folder move to $WORK/out/<arm>/, its exit code and seconds to $WORK/out/<arm>.rc, its log
# to $WORK/logs/<arm>.log. Then per arm: each binary's md5 and size, the tarball's bytes and md5, BUILDINFO's lines,
# and every binary of old against new, off and dup ($WORK/out/compare-<arm>.txt). $WORK/DONE marks the end.
# Env: WORK (the build folder; src/ and ccache/ in it are reused), OLD and NEW (the two scripts), TIP (the commit to
#      build), LOCK (a flock file other builds share), DOCKER_ARGS (e.g. a label and --cpu-shares), JOBS (default 8),
#      UI (the web page release for the tip, default b12483, what the tip's earlier builds embedded)
set -uo pipefail
: "${WORK:?}" "${OLD:?}" "${NEW:?}" "${TIP:?}" "${LOCK:?}"
JOBS=${JOBS:-8}
UI=${UI:-b12483}
DOCKER_ARGS=${DOCKER_ARGS:-}
mkdir -p "$WORK/out" "$WORK/logs"
md5sum "$OLD" "$NEW" > "$WORK/out/scripts.md5"

b() {  # arm script ref [VAR=value ...]
    local arm=$1 script=$2 ref=$3 t0 rc
    shift 3
    t0=$(date +%s)
    flock "$LOCK" env WORK="$WORK" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DOCKER_ARGS" "$@" \
        timeout 3600 bash "$script" "$ref" > "$WORK/logs/$arm.log" 2>&1
    rc=$?
    echo "rc=$rc seconds=$(( $(date +%s) - t0 )) ref=$ref $*" > "$WORK/out/$arm.rc"
    mkdir -p "$WORK/out/$arm"
    mv "$WORK"/llama.cpp-* "$WORK/out/$arm/" 2>/dev/null
    return 0
}

b old "$OLD" "$TIP" HF_UI_VERSION="$UI"
b new "$NEW" "$TIP" HF_UI_VERSION="$UI"
b off "$NEW" "$TIP" HF_UI_VERSION="$UI" CHECK_TOOLS=0
b dup "$NEW" "$TIP" HF_UI_VERSION="$UI" EXTRA_TARGETS="test-llama-archs test-cuda-keepalive"
b v3 "$NEW" miner-llama-v3.0
b v2 "$NEW" miner-llama-v2.0

: > "$WORK/out/tarballs.txt"
for arm in old new off dup v3 v2; do
    d=$(ls -d "$WORK/out/$arm"/llama.cpp-*/ 2>/dev/null | head -n 1)
    tgz=$(ls "$WORK/out/$arm"/llama.cpp-*.tar.gz 2>/dev/null | head -n 1)
    [ -n "$d" ] && [ -n "$tgz" ] || { echo "$arm no tarball" >> "$WORK/out/tarballs.txt"; continue; }
    (cd "$d/bin" && md5sum -- *) > "$WORK/out/$arm.bin.md5"
    (cd "$d/bin" && stat -c "%s %n" -- *) > "$WORK/out/$arm.bin.size"
    tar tzf "$tgz" > "$WORK/out/$arm.tar.list"
    echo "$arm $(stat -c %s "$tgz") $(md5sum < "$tgz" | cut -c1-32) $(basename "$tgz")" >> "$WORK/out/tarballs.txt"
    grep -E "^(name|commit|web page|check tools):" "$d/BUILDINFO.txt" > "$WORK/out/$arm.buildinfo"
done
for arm in new off dup; do
    [ -f "$WORK/out/$arm.bin.md5" ] && [ -f "$WORK/out/old.bin.md5" ] || continue
    join -1 2 -2 2 -a 1 -a 2 -e - -o 0,1.1,2.1 <(sort -k 2 "$WORK/out/old.bin.md5") <(sort -k 2 "$WORK/out/$arm.bin.md5") |
        awk '{ s = ($2 == "-") ? "only in " a : ($3 == "-") ? "only in old" : ($2 == $3) ? "same" : "differs"
               print $1, s }' a="$arm" > "$WORK/out/compare-$arm.txt"
done
date -Is > "$WORK/DONE"
