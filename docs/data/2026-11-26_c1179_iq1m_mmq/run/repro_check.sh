#!/usr/bin/env bash
# C1179: does the compiler give the same code twice? Compiles named MMQ instance objects of build A (the branch, as
# all.sh leaves it) twice each, without the compiler cache, with the build's own command, and compares the two runs
# function by function (code_compare.py), as code_check.sh compares base and branch.
#
#   WORK=<folder> CUDA=/usr/local/cuda TYPES="q4_0 q4_1 q8_0" bash repro_check.sh
set -euo pipefail
: "${WORK:?}" "${TYPES:?}"
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA:-/usr/local/cuda}
export PATH="$CUDA/bin:$PATH"
out=${OUT:-$WORK/logs/repro}
mkdir -p "$out/run1" "$out/run2"
cd "$WORK/build-a"
for t in $TYPES; do
    obj=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-$t.cu.o
    cmd=$(ninja -t commands "$obj" | tail -1 | sed 's/^ccache //')
    for k in 1 2; do
        o=$out/$t.$k.o
        c=${cmd/-o $obj/-o $o}
        c=${c/-MF $obj.d/-MF $o.d}
        eval "nice -n 19 $c"
        cuobjdump -sass "$o" > "$out/run$k/$t.sass" 2> /dev/null || true
        cuobjdump -ptx "$o" > "$out/run$k/$t.ptx" 2> /dev/null || true
        rm -f "$o" "$o.d"
    done
done
python3 -I "$here/code_compare.py" "$out/run1" "$out/run2" "$out/compare" | tee "$out/compare_totals.txt"
