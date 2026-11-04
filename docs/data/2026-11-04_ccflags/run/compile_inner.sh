#!/usr/bin/env bash
# Inside the CUDA container (compile_arms.sh starts it): runs the jobs of /w/jobs.tsv, $JOBS at a time, biggest files first.
set -uo pipefail
cd /w
mapfile -t FLAGS < flags.txt
export FLAGS_FILE=/w/flags.txt
nvcc --version | tail -n 2 > /w/nvcc_version.txt
ptxas --version | tail -n 2 >> /w/nvcc_version.txt

one() { # <arm> <file> <arm flags>
    local arm=$1 f=$2 extra=$3
    mapfile -t FL < /w/flags.txt
    mkdir -p /w/cubin/$arm /w/sass/$arm /w/res/$arm /w/ptxas/$arm
    local t0=$SECONDS
    # shellcheck disable=SC2086
    nice -n 19 nvcc "${FL[@]}" $extra -Xptxas -v -cubin -x cu "/work/src/ggml/src/ggml-cuda/$f" \
        -o "/w/cubin/$arm/$f.cubin" 2> "/w/ptxas/$arm/$f.txt"
    local rc=$?
    if [ $rc -eq 0 ]; then
        cuobjdump -sass "/w/cubin/$arm/$f.cubin" | gzip -1 > "/w/sass/$arm/$f.sass.gz"
        cuobjdump -res-usage "/w/cubin/$arm/$f.cubin" > "/w/res/$arm/$f.res.txt"
        sed -n "s/^ *Function \(.*\):\$/\1/p" "/w/res/$arm/$f.res.txt" | cu++filt > "/w/res/$arm/$f.dem.txt"
    fi
    printf '%s\t%s\t%s\t%s\n' "$arm" "$f" "$rc" "$((SECONDS - t0))" >> /w/time.tsv
    echo "$(date +%T) $arm $f rc=$rc $((SECONDS - t0)) s"
}
export -f one
# the same files' machine code as the build itself made it (the baseline check: a cubin of the base arm must carry the
# build's code), from the objects of the build under /work/src/build
mkdir -p /w/sass/build
while read -r f; do
    o=/work/src/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/$f.o
    [ -f "$o" ] && cuobjdump -sass -arch sm_61 "$o" 2>/dev/null | gzip -1 > "/w/sass/build/$f.sass.gz"
done < /w/files.txt
: > /w/time.tsv
# biggest sources first so the long compiles overlap
order() { for f in mmvq-pascal.cu moe-persistent.cu mmvq-pascal-mtok.cu mmvf-pascal.cu fattn-tile-pascal.cu; do awk -F'\t' -v f=$f '$2==f' /w/jobs.tsv; done
          awk -F'\t' '$2!="mmvq-pascal.cu" && $2!="moe-persistent.cu" && $2!="mmvq-pascal-mtok.cu" && $2!="mmvf-pascal.cu" && $2!="fattn-tile-pascal.cu"' /w/jobs.tsv; }
order | xargs -P "${JOBS:-6}" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r a f x <<< "$1"; one "$a" "$f" "$x"' _ {}
