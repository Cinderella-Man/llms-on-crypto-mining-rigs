#!/usr/bin/env bash
# C997 (Q1730): the card's own time of the TOP_K op with GGML_CUDA_TOP_K_SORT unset and set on one P106 (CUDA0 only):
# test-backend-ops perf on three cases under the fork's profiler (GGML_CUDA_PROFILE, group mode: every launch group
# timed with CUDA events), alternated pu1 ps1 pu2 ps2. Env: BIN (the build's bin folder), O (output folder).
set -u
: "${BIN:?}" "${O:?}"
cd "$O" || exit 1
F='151936,1,1,1\],k=20|248320,1,1,1\],k=20|131072,1,1,1\],k=2048,ties=0'
for a in pu1 ps1 pu2 ps2; do
  case $a in ps*) E="GGML_CUDA_TOP_K_SORT=1" ;; *) E="GGML_CUDA_TOP_K_SORT_UNSET=1" ;; esac
  env $E CUDA_VISIBLE_DEVICES=0 GGML_CUDA_PROFILE="$O/gp-$a.csv" GGML_CUDA_PROFILE_MODE=group \
    timeout 300 "$BIN/test-backend-ops" perf -o TOP_K -b CUDA0 -p "$F" > "perf-$a.txt" 2>&1
  echo "$a rc=$?" >> rc-prof.txt
done
touch done-prof
