#!/usr/bin/env bash
# C1179: one line per CUDA object of a build's ggml-cuda library: the object, the md5 of its machine code
# (cuobjdump -sass) and of its PTX (cuobjdump -ptx), "-" where it has none. Two builds of the same source folder and
# build folder (the base commit, then the branch built incrementally) compare object by object with `join`.
# An object differs here as soon as one constant in it moved: the MMQ objects' stubs for tile configurations that do not
# exist (NO_DEVICE_CODE) print their source line, so code_check.sh compares function by function.
#
#   BUILD=<build folder> CUDA=/usr/local/cuda bash code_digest.sh > digest.tsv
set -euo pipefail
: "${BUILD:?}"
export LC_ALL=C
CUDA=${CUDA:-/usr/local/cuda}
dump=$CUDA/bin/cuobjdump
dir=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir

# the identifier lines name the source file; the code is what is compared
digest() {
    local listing
    listing=$("$dump" "$1" "$3" 2> /dev/null || true)
    if [ -z "$listing" ]; then
        echo -
        return
    fi
    { "$dump" "$2" "$3" 2> /dev/null || true; } | grep -v -e 'identifier' -e '^Fatbin' | md5sum | cut -c1-32
}

find "$dir" -name '*.o' | sort | while read -r obj; do
    printf '%s\t%s\t%s\n' "${obj#"$dir"/}" "$(digest -lelf -sass "$obj")" "$(digest -lptx -ptx "$obj")"
done
