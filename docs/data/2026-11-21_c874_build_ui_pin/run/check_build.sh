#!/usr/bin/env bash
# C874 (Q1577, 2026-11-21): one rig build of a fork branch with scripts/build_llamacpp_docker.sh and no HF_UI_VERSION
# passed anywhere, to show that the script pins llama-server's web page by itself; then every binary's md5 against an
# earlier build of the same commit that passed HF_UI_VERSION by hand (REF_BIN). CPU only, no GPU.
# Steps: a local clone of SEED (a llama.cpp clone with upstream's b<N> tags and the commit) with the fork as remote
# "fork"; the branch's tip checked against HEAD_SHA; git describe on HEAD_SHA and on the v4.0 tag; the build (TESTS=1,
# test-llama-archs and test-cuda-keepalive added, as REF_BIN's build), under LOCK; its web page lines; md5.tsv.
# Usage: W=<work dir> SEED=<llama.cpp clone> REPO=<folder with build_llamacpp_docker.sh> REF_BIN=<reference bin folder>
#   [CCACHE=<warm ccache dir, hard-linked>] [LOCK=<flock file>] [REF=fork/pascal-p106-v5] [HEAD_SHA=<full sha>]
#   [JOBS=8] bash check_build.sh
# Every container: labels pascal-quants=1 and c874=1, --cpu-shares=256, --memory=16g. Done when $W/DONE exists (it
# holds the build's exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with upstream tags and the commit>}
REPO=${REPO:?REPO=<folder with build_llamacpp_docker.sh>}
REF_BIN=${REF_BIN:?REF_BIN=<bin folder of a reference build of the same commit>}
REF=${REF:-fork/pascal-p106-v5}
HEAD_SHA=${HEAD_SHA:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
JOBS=${JOBS:-8}
mkdir -p "$W/rig" "$W/logs"
rm -f "$W/DONE"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c874=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
echo "=== start $(date -Is): $REF ($HEAD_SHA)"

S=$W/rig/src
if [ ! -d "$S/.git" ]; then
    git clone -q --no-checkout "$SEED" "$S" || finish 1
    git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*'
    git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
fi
git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
[ -d "$W/rig/ccache" ] || { [ -n "${CCACHE:-}" ] && cp -al "$CCACHE" "$W/rig/ccache"; }
tip=$(git ls-remote "$FORK_URL" "refs/heads/${REF#fork/}" | cut -f1)
echo "$REF = $tip"
[ "$tip" = "$HEAD_SHA" ] || { echo "$REF is not $HEAD_SHA: stop"; finish 1; }
echo "b<N> tags in the clone: $(git -C "$S" tag -l 'b[0-9]*' | wc -l)"
echo "describe $HEAD_SHA: $(git -C "$S" describe --tags --abbrev=0 --match 'b[0-9]*' "$HEAD_SHA" 2>&1)"
echo "describe miner-llama-v4.0: $(git -C "$S" describe --tags --abbrev=0 --match 'b[0-9]*' miner-llama-v4.0 2>&1)"
echo "HF_UI_VERSION in this environment: ${HF_UI_VERSION:-unset}"

echo "=== build $(date -Is)"
locked env -u HF_UI_VERSION -u HF_WEBUI_VERSION WORK="$W/rig" TESTS=1 JOBS="$JOBS" \
    EXTRA_TARGETS="test-llama-archs test-cuda-keepalive" \
    DOCKER_ARGS="--label pascal-quants=1 --label c874=1 --cpu-shares=256 --memory=16g --memory-swap=16g" \
    bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
rc=$?
echo "=== build end $(date -Is) rc=$rc"
grep -a -e '^web page' -e '^-- UI:' -e 'WARNING' "$W/logs/rig.log"
T=$(ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1)
[ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no tarball"; finish 1; }
(cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz")
printf 'file\tthis build\treference\tsame\n' > "$W/md5.tsv"
for f in "$T"/bin/*; do
    b=$(basename "$f")
    a=$(md5sum < "$f" | cut -d' ' -f1)
    r=$( [ -f "$REF_BIN/$b" ] && md5sum < "$REF_BIN/$b" | cut -d' ' -f1 || echo missing)
    printf '%s\t%s\t%s\t%s\n' "$b" "$a" "$r" "$([ "$a" = "$r" ] && echo yes || echo no)"
done >> "$W/md5.tsv"
cat "$W/md5.tsv"
echo "binaries the same: $(grep -c 'yes$' "$W/md5.tsv") of $(($(wc -l < "$W/md5.tsv") - 1))"
diff <(grep -v -e '^name:' -e '^ref:' -e '^built:' -e '^web page:' "$REF_BIN/../BUILDINFO.txt") \
     <(grep -v -e '^name:' -e '^ref:' -e '^built:' -e '^web page:' "$T/BUILDINFO.txt") > "$W/buildinfo.diff" \
    && echo "BUILDINFO.txt the same apart from name, ref, built and web page" || { echo "BUILDINFO.txt differs:"; cat "$W/buildinfo.diff"; }
cp "$T/BUILDINFO.txt" "$W/logs/BUILDINFO.txt"
finish "$rc"
