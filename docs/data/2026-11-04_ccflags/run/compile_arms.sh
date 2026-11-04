#!/usr/bin/env bash
# Q671 / C131: compile each Pascal kernel file of a build to an sm_61 cubin with the build's own nvcc flags, once per
# arm of arms.tsv (the baseline, -Xptxas -dlcm=ca, register caps, ...), and dump each cubin's SASS and resource usage.
# CPU only: nvcc and cuobjdump in the CUDA 12.9 container, flag extraction in a slim Python container; no GPU.
#
# Usage: WORK=<dir> SRC=<llama.cpp tree> [JOBS=6] [LOCK=<file>] [ARMS='base dlcm_ca ...'] [DOCKER_ARGS=...] compile_arms.sh
#   SRC   a llama.cpp checkout that was built with scripts/build_llamacpp_docker.sh (its build/ holds
#         compile_commands.json; the tree is mounted read-only at /work/src, where that build saw it)
#   WORK  output folder: cubin/<arm>/<file>.cubin, sass/<arm>/<file>.sass.gz, res/<arm>/<file>.res.txt,
#         ptxas/<arm>/<file>.txt (ptxas -v), time.tsv (seconds per compile), flags.txt, DONE when finished
#   LOCK  a file to flock around the compiles when several sessions share the build machine
# arms.tsv and files.tsv are read from this script's folder; an arm applies to the files its third column matches.
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<output dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp tree with build/>}")
JOBS=${JOBS:-6}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PYIMAGE=${PYIMAGE:-python:3.12-slim-bookworm}
HERE=$(dirname "$(realpath "$0")")
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/run"
cp "$HERE"/{arms.tsv,files.tsv,compile_inner.sh,extract_flags.py} "$WORK/run/"
rm -f "$WORK/DONE"
cut -f2 "$HERE/files.tsv" | tail -n +2 > "$WORK/files.txt"
# the arms to run, with the files each applies to: <arm> <flags> <file>
: > "$WORK/jobs.tsv"
while IFS=$'\t' read -r arm flags re _; do
    [ "$arm" = arm ] && continue
    case " ${ARMS:-$arm} " in *" $arm "*) ;; *) continue ;; esac
    [ "$flags" = - ] && flags=""
    while read -r f; do
        printf '%s\n' "$f" | grep -Eq "$re" && printf '%s\t%s\t%s\n' "$arm" "$f" "$flags" >> "$WORK/jobs.tsv"
    done < "$WORK/files.txt"
done < "$HERE/arms.tsv"
echo "=== $(wc -l < "$WORK/jobs.tsv") compiles, $JOBS at a time, $(date -Is)"

docker run --rm $D -v "$SRC:/work/src:ro" -v "$WORK:/w" "$PYIMAGE" nice -n 19 python3 -I /w/run/extract_flags.py \
    /work/src/build/compile_commands.json /w/flags.txt $(tr '\n' ' ' < "$WORK/files.txt") || exit 1

locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
locked docker run --rm $D -u "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="$JOBS" \
    -v "$SRC:/work/src:ro" -v "$WORK:/w" "$IMAGE" bash /w/run/compile_inner.sh
echo "=== done $(date -Is)"
touch "$WORK/DONE"
