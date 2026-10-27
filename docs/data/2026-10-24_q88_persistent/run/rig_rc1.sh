#!/bin/bash
# rc1 on the rig, before any timing: release candidate pascal-p106-v2-rc 594d20e08 (v2-next d379da309 + the compute-6.1
# rescale default, the staircase drafter, chunk streams and the continuation store, Q151's decode attention, Q67,
# Q149's K-quant kernels).
# On the quant machine's V100s the compute-6.1 defaults are off unless forced, and its check_identity runs logged at
# default verbosity, so they don't show which kernels ran. Here, on the P106, everything runs at its defaults: no
# GGML_CUDA_* or LLAMA_NO_MOE_DOWN_RESCALE variables (rc1 drops the one-token rescale on compute 6.1 by itself).
#   0. which kernels run, K2h-r16 (UD for the K-quant kernels), 6 cards, -ub 1, -c 128, one chunk:
#      a. the persistent MoE kernel: its first-use line (-lv 4), its self-check (GGML_CUDA_PERSISTENT_MOE_CHECK=1:
#         0 differed expected), no rescale warning
#      b. kernels without a log line of their own: logits with the default against the kernel turned off. The Pascal
#         matrix-vector kernels, the decode attention kernel and the K-quant kernels change the summation order, so
#         "differ" means the default runs them
#      c. the attention block (bit-identical on and off): launch groups per token, GGML_CUDA_PROFILE counts only, 9 cards,
#         35 x 4 = 140 fewer expected with the default
#   2. llama-pipespec (bench/pipespec/rig_run.sh), plain against pipelined, depths 1 and 3, 9 cards, K2h-r16 and UD,
#      -lv 4: same_as_plain for every job, the persistent kernel's first-use line in each run.log. Speeds stay in the
#      output files and are not reported until step 2 passes
#   3. llama-server --pipe-depth 3 against no guesses (bench/pipespec/check_server.sh), 9 cards, K2h-r16 and UD, -lv 4,
#      no speeds: same outputs, the first-use line in each server log
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=llama.cpp-20261024-594d20e08-pascal-p106-v2-rc [RIG=rig] [OUT=dir] rig_rc1.sh
# (or T=llama.cpp-20261025-594d20e08-pascal-p106-v2-rc-sched8). STEPS='0' etc. runs some steps.
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<rc1 tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
C6=0,1,2,3,4,5
C9=0,1,2,3,4,5,6,7,8
PPL="-f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 1 -b 128 -ub 1 -ngl 99 -fa on"
STEPS=${STEPS:-0 2 3}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
echo "# rc1 identity and kernel use on the rig, $T, $(date -Is)"
if has 0; then
  echo "== 0a. persistent MoE kernel at the defaults, K2h-r16, 6 cards, -ub 1, -lv 4 $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 GGML_CUDA_PERSISTENT_MOE_CHECK=1 $B/llama-perplexity -m $K2H $PPL -lv 4 2>&1" \
    | grep -oE "MoE layers at one token run as one persistent kernel|rescale_notice|persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed|Final estimate.*" \
    | sort | uniq -c
  echo "-- expected: the first-use line once per card, no rescale_notice, 0 differed throughout"
  echo "== 0b. logits, default against off, 6 cards, -ub 1 $(date +%T)"
  for arm in "default|$K2H|-" "GGML_CUDA_MMVQ_PASCAL=0|$K2H|GGML_CUDA_MMVQ_PASCAL=0" "GGML_CUDA_FATTN_PASCAL=0|$K2H|GGML_CUDA_FATTN_PASCAL=0" \
             "UD default|$UD|-" "UD GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K|$UD|GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K"; do
    IFS='|' read -r label m e <<< "$arm"
    [ "$e" = - ] && e=""
    f=/tmp/rc1-$(echo "$label" | tr -c 'A-Za-z0-9\n' '_').kld
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 $e $B/llama-perplexity -m $m $PPL --kl-divergence-base $f 2>&1" | grep -oE "Final estimate.*" | sed "s/^/$label: /"
  done
  ssh "$RIG" "cd /tmp && for x in GGML_CUDA_MMVQ_PASCAL_0 GGML_CUDA_FATTN_PASCAL_0; do cmp -s rc1-default.kld rc1-\$x.kld && echo \"\$x: IDENTICAL to the default (kernel not reached)\" || echo \"\$x: differs from the default (the default runs the kernel)\"; done
    cmp -s rc1-UD_default.kld rc1-UD_GGML_CUDA_MMVQ_PASCAL_SKIP_q4_K_q6_K.kld && echo 'K-quant skip: IDENTICAL to the default (kernels not reached)' || echo 'K-quant skip: differs from the default (the default runs the K-quant kernels)'
    rm -f /tmp/rc1-*.kld"
  echo "== 0c. launch groups per token, attention block default against off, K2h-r16, 9 cards (counts, not times) $(date +%T)"
  scp -q $R/scripts/cuda_profile_summary.py "$RIG":/tmp/
  for s in default 0; do
    e=""; [ "$s" = 0 ] && e="GGML_CUDA_ATTN_BLOCK=0"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 $e GGML_CUDA_PROFILE=/tmp/rc1-prof-$s.csv $B/llama-bench -m $K2H -ngl 99 -fa on -p 0 -n 32 -r 1 > /dev/null 2>&1; python3 /tmp/cuda_profile_summary.py /tmp/rc1-prof-$s.csv --tokens 24 --by op 2>&1 | head -30" \
      | sed "s/^/ATTN_BLOCK $s: /"
  done
  ssh "$RIG" "rm -f /tmp/rc1-prof-*.csv /tmp/cuda_profile_summary.py"
fi
if has 2; then
  ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    echo "== 2. llama-pipespec plain against pipelined at the defaults, $tag, 9 cards, depths 1 and 3, -lv 4 $(date +%T)"
    ssh "$RIG" "env BIN=$B MODEL=$m TAG=rc1-$tag-9c GPUS=$C9 DEPTHS='1 3' N=256 GPU_LOG=0 EXTRA_ARGS='-lv 4' bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/rc1-$tag-9c.out 2>&1
      O=/tmp/pipespec/out/rc1-$tag-9c; tail -2 /tmp/pipespec/rc1-$tag-9c.out | grep -E 'exit|error'
      awk -F'\t' 'NR > 1 { print \$1, \$14; n++; if (\$14 == \"NO\") bad++ } END { printf \"%d jobs, %d not same as plain\\n\", n, bad }' \$O/summary.tsv
      echo \"first-use lines: \$(grep -c 'run as one persistent kernel' \$O/run.log), rescale warnings: \$(grep -c rescale_notice \$O/run.log)\""
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec" | tar xzf - -C "$OUT"
  ssh "$RIG" "rm -rf /tmp/pipespec"
fi
if has 3; then
  ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_server"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    echo "== 3. llama-server --pipe-depth against no guesses at the defaults, $tag, 9 cards, -lv 4 $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 BIN=$B MODEL=$m ARGS='-ngl 99 -fa on -c 8192 -ub 512 -lv 4' DEPTHS='3' GUESSES='cont' KEEP_SPEED=0 OUT=/tmp/pipespec_server/$tag bash /tmp/pipespec/check_server.sh 2>&1 | tail -20
      for f in /tmp/pipespec_server/$tag/*.log; do echo \"\$(basename \$f): first-use lines \$(grep -c 'run as one persistent kernel' \$f), rescale warnings \$(grep -c rescale_notice \$f)\"; done"
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec_server" | tar xzf - -C "$OUT"
  ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_server"
fi
echo RC1_DONE
} | tee -a "$OUT/rig_rc1.md"
