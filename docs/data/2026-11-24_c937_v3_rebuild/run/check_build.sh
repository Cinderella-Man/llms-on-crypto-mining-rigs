#!/usr/bin/env bash
# C937 (Q1677, 2026-11-24): one rig build of a release tag with scripts/build_llamacpp_docker.sh as a reader would run
# it (TESTS=1, nothing else set, no HF_UI_VERSION anywhere unless UI_VERSION is given), to see whether the tag still
# builds now that upstream's newest web page archive no longer fits older trees; then every binary's md5 against the
# released build (REF_BIN). CPU only, no GPU.
# Steps: a local clone of SEED (a llama.cpp clone with upstream's b<N> tags and the fork's release tags); the tag
# checked against HEAD_SHA and against the fork on GitHub; git describe on it; the build, under LOCK; its web page
# lines; md5_run<RUN>.tsv.
# Usage: W=<work dir> SEED=<llama.cpp clone> REPO=<folder with build_llamacpp_docker.sh> REF_BIN=<released bin folder>
#   [RUN=1] [REF=miner-llama-v3.0] [HEAD_SHA=<full sha>] [UI_VERSION=b<N>] [ABBREV=<git core.abbrev, e.g. 9>]
#   [LOCK=<flock file>] [JOBS=8] bash check_build.sh
# The source clone and compiler cache in $W/rig are kept between runs. Every container: labels pascal-quants=1 and
# c937=1, --cpu-shares=256, --memory=16g. Done when $W/DONE_run<RUN> exists (it holds the build's exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with upstream tags and the release tag>}
REPO=${REPO:?REPO=<folder with build_llamacpp_docker.sh>}
REF_BIN=${REF_BIN:?REF_BIN=<bin folder of the released build>}
RUN=${RUN:-1}
REF=${REF:-miner-llama-v3.0}
HEAD_SHA=${HEAD_SHA:-1551f5fe12424d7006748c1628b0df9550f20171}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
JOBS=${JOBS:-8}
mkdir -p "$W/rig" "$W/logs"
rm -f "$W/DONE_run$RUN"
LOG=$W/logs/run$RUN.log
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c937=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE_run$RUN"; exit "$1"; }
echo "=== run $RUN start $(date -Is): $REF ($HEAD_SHA), build script md5 $(md5sum < "$REPO/build_llamacpp_docker.sh" | cut -d' ' -f1)"

S=$W/rig/src
if [ ! -d "$S/.git" ]; then
    git clone -q --no-checkout "$SEED" "$S" || finish 1
    git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
fi
got=$(git -C "$S" rev-parse "$REF^{commit}" 2>/dev/null)
remote=$(git ls-remote "$FORK_URL" "refs/tags/$REF^{}" | cut -f1)
echo "$REF here = $got, on the fork = $remote"
[ "$got" = "$HEAD_SHA" ] && [ "$remote" = "$HEAD_SHA" ] || { echo "$REF is not $HEAD_SHA: stop"; finish 1; }
echo "b<N> tags in the clone: $(git -C "$S" tag -l 'b[0-9]*' | wc -l)"
echo "describe $REF: $(git -C "$S" describe --tags --abbrev=0 --match 'b[0-9]*' "$REF" 2>&1)"
# the binaries embed the commit id as git rev-parse --short prints it, whose length grows with the clone's object count
if [ -n "${ABBREV:-}" ]; then git -C "$S" config core.abbrev "$ABBREV"; else git -C "$S" config --unset core.abbrev; fi
echo "core.abbrev: ${ABBREV:-unset}; the commit id the binaries embed: $(git -C "$S" rev-parse --short "$REF^{commit}")"
UI_ENV=()
[ -n "${UI_VERSION:-}" ] && UI_ENV=(HF_UI_VERSION="$UI_VERSION")
echo "HF_UI_VERSION passed to the build: ${UI_VERSION:-none}"

echo "=== build $(date -Is)"
locked env -u HF_UI_VERSION -u HF_WEBUI_VERSION ${UI_ENV[@]+"${UI_ENV[@]}"} WORK="$W/rig" TESTS=1 JOBS="$JOBS" \
    DOCKER_ARGS="--label pascal-quants=1 --label c937=1 --cpu-shares=256 --memory=16g --memory-swap=16g" \
    bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$LOG" 2>&1
rc=$?
echo "=== build end $(date -Is) rc=$rc"
grep -a -e '^web page' -e '^-- UI:' -e 'WARNING' -e 'UI: ' -e 'missing required' -e 'Error' -e 'error:' "$LOG" | head -n 40
T=$(ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}" 2>/dev/null | head -n 1)
[ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no tarball"; finish "$([ "$rc" = 0 ] && echo 1 || echo "$rc")"; }
(cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz")
printf 'file\tthis build\treleased v3.0\tsame\n' > "$W/md5_run$RUN.tsv"
for f in "$T"/bin/*; do
    b=$(basename "$f")
    a=$(md5sum < "$f" | cut -d' ' -f1)
    r=$( [ -f "$REF_BIN/$b" ] && md5sum < "$REF_BIN/$b" | cut -d' ' -f1 || echo missing)
    printf '%s\t%s\t%s\t%s\n' "$b" "$a" "$r" "$([ "$a" = "$r" ] && echo yes || echo no)"
done >> "$W/md5_run$RUN.tsv"
cat "$W/md5_run$RUN.tsv"
echo "binaries the same: $(grep -c 'yes$' "$W/md5_run$RUN.tsv") of $(($(wc -l < "$W/md5_run$RUN.tsv") - 1))"
diff <(grep -v -e '^name:' -e '^ref:' -e '^built:' -e '^web page:' -e '(0x' "$REF_BIN/../BUILDINFO.txt") \
     <(grep -v -e '^name:' -e '^ref:' -e '^built:' -e '^web page:' -e '(0x' "$T/BUILDINFO.txt") > "$W/buildinfo_run$RUN.diff" \
    && echo "BUILDINFO.txt the same apart from name, ref, built, web page and load addresses" \
    || { echo "BUILDINFO.txt differs:"; cat "$W/buildinfo_run$RUN.diff"; }
cp "$T/BUILDINFO.txt" "$W/logs/BUILDINFO_run$RUN.txt"
finish "$rc"
