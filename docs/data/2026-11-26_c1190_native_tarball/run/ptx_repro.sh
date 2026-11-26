#!/usr/bin/env bash
# C1190: is a PTX difference between two builds the compiler's own variation? One CUDA source compiled N times at once
# on one machine with the build's own flags and no compiler cache (the same source, the same compiler, the same
# folder), then each compile's code compared with the first one's by card_compare.py, and one function's PTX (FUNC,
# after card_compare.py's renaming of hashed names) hashed in each compile and in the two builds' full listings.
#
#   BUILD=<CMake build folder> SRC=<the .cu file> FUNC=<mangled name> OUT=<folder> [N=4] [CUDA=/usr/local/cuda] \
#       [LISTING_A=<cuobjdump -ptx of build A's program>] [LISTING_B=<the same of build B>] \
#       [ROOT=<the source tree> ALIAS=<a symbolic link to make for it>] bash ptx_repro.sh
#
# With ROOT and ALIAS the compiles see the source tree under another path (ALIAS, e.g. one as long as the container's
# /work/src): SRC and the include folders are rewritten to it, and the link is removed at the end.
#
# Writes OUT/repro.tsv (per compile: the whole object against compile 1, SASS and PTX) and OUT/func.tsv (FUNC's PTX:
# its md5 and line count in each compile and listing).
set -euo pipefail
: "${BUILD:?}" "${SRC:?}" "${FUNC:?}" "${OUT:?}"
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
N=${N:-4}
CUDA=${CUDA:-/usr/local/cuda}
dir=$BUILD/ggml/src/ggml-cuda
flags=$dir/CMakeFiles/ggml-cuda.dir/flags.make
get() { sed -n "s/^$1 = //p" "$flags"; }
mkdir -p "$OUT"
includes=$(get CUDA_INCLUDES)
if [ -n "${ALIAS:-}" ]; then
    : "${ROOT:?ROOT=<the source tree> with ALIAS}"
    [ -e "$ALIAS" ] && { echo "$ALIAS exists" >&2; exit 1; }
    ln -s "$ROOT" "$ALIAS"
    trap 'rm -f "$ALIAS"' EXIT
    sed "s|$ROOT|$ALIAS|g" "$dir/CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp" > "$OUT/includes.rsp"
    includes="--options-file $OUT/includes.rsp"
    SRC=${SRC/#$ROOT/$ALIAS}
fi
for i in $(seq 1 "$N"); do
    # make hands these variables to the shell, so eval splits and unquotes them the same way
    (cd "$dir" && eval "nice -n 19 \"$CUDA/bin/nvcc\" -forward-unknown-to-host-compiler $(get CUDA_DEFINES) \
        $includes $(get CUDA_FLAGS) -x cu -c \"$SRC\" -o \"$OUT/obj$i.o\"") &
done
wait
for i in $(seq 1 "$N"); do
    "$CUDA/bin/cuobjdump" -sass "$OUT/obj$i.o" > "$OUT/obj$i.sass"
    "$CUDA/bin/cuobjdump" -ptx "$OUT/obj$i.o" > "$OUT/obj$i.ptx"
done
printf 'compile\tagainst\tsass\tptx\n' > "$OUT/repro.tsv"
for i in $(seq 2 "$N"); do
    python3 -I "$here/card_compare.py" "$OUT/obj1.sass" "$OUT/obj$i.sass" "$OUT/obj1.ptx" "$OUT/obj$i.ptx" "$OUT/cmp$i" \
        > "$OUT/cmp$i.txt"
    printf '%s\t1\t%s\t%s\n' "$i" "$(sed -n 's/^sass //p' "$OUT/cmp$i.txt")" "$(sed -n 's/^ptx //p' "$OUT/cmp$i.txt")" \
        >> "$OUT/repro.tsv"
done
python3 -I - "$here/card_compare.py" "$FUNC" "$OUT/func.tsv" "$OUT"/obj*.ptx ${LISTING_A:+"$LISTING_A"} ${LISTING_B:+"$LISTING_B"} << 'PY'
import hashlib, importlib.util, os, sys
spec = importlib.util.spec_from_file_location('cc', sys.argv[1])
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
func, out = sys.argv[2], sys.argv[3]
with open(out, 'w') as f:
    f.write('listing\tfunction found\tlines\tmd5 of its PTX (hashed names renamed)\n')
    for path in sys.argv[4:]:
        funcs = cc.ptx_functions(path, set())
        hits = [v for k, v in funcs.items() if cc.unhash(func, set()) == k[1]]
        for v in hits:
            f.write(f'{os.path.basename(path)}\tyes\t{len(v[1].splitlines())}\t{hashlib.md5(v[1].encode()).hexdigest()}\n')
        if not hits:
            f.write(f'{os.path.basename(path)}\tno\t-\t-\n')
PY
rm -f "$OUT"/obj*.o "$OUT"/obj*.sass "$OUT"/obj*.ptx
cat "$OUT/repro.tsv" "$OUT/func.tsv"
