#!/usr/bin/env bash
# C1190: one rig tarball built natively (scripts/build_llamacpp_docker.sh with NATIVE=1 and TESTS=1) at the lowest CPU
# priority, detached-safe: the log goes to $OUT/$TAG.log, then "<exit code> <seconds>" to $OUT/$TAG.exit (the done
# marker) and the tarball's md5 to $OUT/$TAG.md5.
#
#   SCRIPT=<build_llamacpp_docker.sh> WORK=<folder> OUT=<log folder> TAG=<name> HF_UI_VERSION=b<N> \
#       [CUDA_HOME=<CUDA 12.x>] [JOBS=8] bash build_native.sh <git-ref>
set -uo pipefail
: "${SCRIPT:?}" "${WORK:?}" "${OUT:?}" "${TAG:?}" "${HF_UI_VERSION:?}"
ref=${1:?git-ref}
mkdir -p "$OUT"
rm -f "$OUT/$TAG.exit" "$OUT/$TAG.md5"
start=$(date +%s)
NATIVE=1 TESTS=1 JOBS=${JOBS:-8} WORK=$WORK HF_UI_VERSION=$HF_UI_VERSION nice -n 19 bash "$SCRIPT" "$ref" > "$OUT/$TAG.log" 2>&1
rc=$?
if [ "$rc" = 0 ]; then
    name=$(sed -n 's/^name: //p' "$OUT/$TAG.log" | tail -n 1)
    (cd "$WORK" && md5sum "$name.tar.gz" && stat -c '%s %n' "$name.tar.gz") > "$OUT/$TAG.md5"
fi
echo "$rc $(( $(date +%s) - start ))" > "$OUT/$TAG.exit"
