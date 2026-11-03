#!/usr/bin/env bash
# The sm_61 SASS of every CUDA kernel in a merged build's test-backend-ops against the builds of the branches it merges
# (sass_identity.py does the comparison). CPU only: cuobjdump in the CUDA 12.9 container, Python in a slim Python
# container; no GPU.
#
# Usage: WORK=<dir> sass_identity.sh <merged build dir> <base tag>=<build dir> <tag>[:<cut-from tag>]=<build dir> ...
#   a build dir is an unpacked tarball of scripts/build_llamacpp_docker.sh (bin/test-backend-ops); the base is the tree
#   the merge starts from; a branch cut from an older tree names that tree's tag (which must be listed too, e.g. as
#   older=<dir> with no colon) after a colon
# Output: $WORK/sass/{<tag>.sass.gz,<tag>.res.gz,kernels.tsv,carried.tsv,summary.txt}
set -euo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PYIMAGE=${PYIMAGE:-python:3.12-slim-bookworm}
DA="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(dirname "$(realpath "$0")")
S=$WORK/sass
mkdir -p "$S"

dump() { # <tag> <build dir>
    docker run --rm $DA -v "$(realpath "$2"):/b:ro" -v "$S:/s" "$IMAGE" bash -euo pipefail -c "
        nice -n 19 cuobjdump -sass -arch sm_61 /b/bin/test-backend-ops | gzip -1 > /s/$1.sass.gz
        nice -n 19 cuobjdump -res-usage -arch sm_61 /b/bin/test-backend-ops | gzip -1 > /s/$1.res.gz"
    echo "dumped $1: $(zcat "$S/$1.sass.gz" | grep -c 'Function : ') functions"
}

dump merged "$1"
shift
base=""
tags=()
known=()
for a in "$@"; do
    spec=${a%%=*}
    dump "${spec%%:*}" "${a#*=}"
    if [ -z "$base" ]; then base=$spec; else case $spec in *:*) tags+=("$spec") ;; *) known+=("$spec") ;; esac; fi
done
# plain tags after the base are branches cut from the base, unless a colon spec names them as another branch's tree
for t in "${known[@]:-}"; do
    [ -n "$t" ] || continue
    printf '%s\n' "${tags[@]:-}" | grep -q ":$t\$" || tags+=("$t")
done
docker run --rm $DA -v "$S:/s" -v "$HERE:/r:ro" "$PYIMAGE" nice -n 19 python3 -I /r/sass_identity.py /s merged "$base" \
    "${tags[@]}" | tee "$S/summary.txt"
