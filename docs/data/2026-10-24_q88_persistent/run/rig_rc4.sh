#!/bin/bash
# rc4 on the rig, before any timing: release candidate pascal-p106-v2-rc4 ab4a5cd02 = rc3 7e6ec762f (rc1 594d20e08 + Q163's
# compute-6.1 prompt attention, on by default on compute 6.1, + Q111's agent prompt cache, server options off unless asked
# for) + the staircase drafter (daf310b66/c3a324ba0: off without a head file) + the fed-token guard for M-RoPE/PLE
# models (99d61a35e). rig_rc3.sh's checks, plus the drafter's identity (step d). Everything runs at its defaults: no
# GGML_CUDA_* or LLAMA_NO_MOE_DOWN_RESCALE variables (the rc drops the one-token rescale on compute 6.1 by itself).
# Written for ab4a5cd02 and run on its successors too: 2d53179aa (the allocator fix) and the v2.0 commit 37e717ff6.
#   0. which kernels run, K2h-r16 (UD for the K-quant kernels), 6 cards:
#      a. the persistent MoE kernel (-ub 1, -lv 4): its first-use line, its self-check (GGML_CUDA_PERSISTENT_MOE_CHECK=1:
#         0 differed expected), no rescale warning beyond the warm-up's two-token batch
#      b. kernels without a log line of their own (-ub 1): logits with the default against the kernel turned off. The
#         Pascal matrix-vector kernels, the decode attention kernel and the K-quant kernels change the summation order,
#         so "differ" means the default runs them
#      c. the attention block (bit-identical on and off): launch groups per token, GGML_CUDA_PROFILE counts only, 9 cards,
#         35 x 4 = 140 fewer expected with the default
#      d. Q163's prompt attention (512-token batches, -lv 4): its first-use line, and logits with the default against
#         GGML_CUDA_FATTN_TILE_PASCAL=0 ("differ": the default runs it)
#   2. llama-pipespec (bench/pipespec/rig_run.sh), plain against pipelined, depths 1 and 3, 9 cards, K2h-r16 and UD,
#      -lv 4: same_as_plain for every job, the persistent kernel's and Q163's first-use lines in each run.log. Speeds stay
#      in the output files and are not reported until step 2 passes
#   3. llama-server --pipe-depth 3 against no guesses (bench/pipespec/check_server.sh), 9 cards, K2h-r16 and UD, -lv 4,
#      no speeds: same outputs, the first-use lines in each server log
#   d. the drafter (needs HEAD): check_identity.sh with the head as the staircase guesser behind the copy lookup, K2h-r16,
#      depths 1 and 3, misses 0.5 for the oracle, -lv 4 (the drafter's load line and the first-use lines): 6 cards
#      (-ts 9,8,8,8,8,8) greedy on the last card; 9 cards with the even split (-ts 6,6,6,6,5,5,5,5,5) and with the
#      release's light-last-card split (-ts 6,6,6,6,6,6,6,5,2, the pipelined configuration the g9light head is trained
#      for), each greedy and at the makers' sampling (T 1 on the last card, each draw keyed by its position). Every
#      stair-dD run must equal plain decoding
#   4. timed, reported only once 0, 2, 3 and d pass: Q163's prompt speed on this build, K2h-r16, 9 cards,
#      llama-batched-bench at 8192 and 32768 prompt tokens, -ub 512, GGML_CUDA_FATTN_TILE_PASCAL=0 then the default, per
#      depth (rig_q163.sh step 2 on this build; ~20 min)
# Q111's agent cache has its own rig test (bench/agent_cache/rig_check.sh); here it stays off. The
# drafter's speed is measured separately (docs/drafter.md).
# Usage, from any directory, with the tarball unpacked in /opt/llama-builds on the rig (scripts/build_llamacpp_docker.sh):
#   T=llama.cpp-20261025-2d53179aa-pascal-p106-v2-rc4 HEAD=<drafter head .gguf on the rig> [RIG=rig] [OUT=dir] rig_rc4.sh
# (or the -sched8 name). STEPS='0' etc. runs some steps; without HEAD, step d is skipped with a note.
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the log (and copied-back outputs) go
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # the repository root
T=${T:?T=<rc4 tarball name without .tar.gz>}
B=/opt/llama-builds/$T/bin
UD=${UD:-/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
C6=0,1,2,3,4,5
C9=0,1,2,3,4,5,6,7,8
PPL="-f /opt/llama-builds/data/wiki.test.raw -c 128 --chunks 1 -b 128 -ub 1 -ngl 99 -fa on"
PPL512="-f /opt/llama-builds/data/wiki.test.raw -c 512 --chunks 1 -b 512 -ub 512 -ngl 99 -fa on"
STEPS=${STEPS:-0 2 3 d 4}
DEPTHS=${DEPTHS:-8192 32768}
HEAD=${HEAD:-}
MAKERS="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
{
echo "# rc4 identity and kernel use on the rig, $T, $(date -Is)"
if has 0; then
  echo "== 0a. persistent MoE kernel at the defaults, K2h-r16, 6 cards, -ub 1, -lv 4 $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 GGML_CUDA_PERSISTENT_MOE_CHECK=1 $B/llama-perplexity -m $K2H $PPL -lv 4 2>&1" \
    | grep -oE "MoE layers at one token run as one persistent kernel|rescale_notice|persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed|Final estimate.*" \
    | sort | uniq -c
  echo "-- expected: the first-use line once per card, at most one rescale_notice (the warm-up), 0 differed throughout"
  echo "== 0b. logits, default against off, 6 cards, -ub 1 $(date +%T)"
  for arm in "default|$K2H|-" "GGML_CUDA_MMVQ_PASCAL=0|$K2H|GGML_CUDA_MMVQ_PASCAL=0" "GGML_CUDA_FATTN_PASCAL=0|$K2H|GGML_CUDA_FATTN_PASCAL=0" \
             "UD default|$UD|-" "UD GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K|$UD|GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K"; do
    IFS='|' read -r label m e <<< "$arm"
    [ "$e" = - ] && e=""
    f=/tmp/rc4-$(echo "$label" | tr -c 'A-Za-z0-9\n' '_').kld
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 $e $B/llama-perplexity -m $m $PPL --kl-divergence-base $f 2>&1" | grep -oE "Final estimate.*" | sed "s/^/$label: /"
  done
  ssh "$RIG" "cd /tmp && for x in GGML_CUDA_MMVQ_PASCAL_0 GGML_CUDA_FATTN_PASCAL_0; do cmp -s rc4-default.kld rc4-\$x.kld && echo \"\$x: IDENTICAL to the default (kernel not reached)\" || echo \"\$x: differs from the default (the default runs the kernel)\"; done
    cmp -s rc4-UD_default.kld rc4-UD_GGML_CUDA_MMVQ_PASCAL_SKIP_q4_K_q6_K.kld && echo 'K-quant skip: IDENTICAL to the default (kernels not reached)' || echo 'K-quant skip: differs from the default (the default runs the K-quant kernels)'
    rm -f /tmp/rc4-*.kld"
  echo "== 0c. launch groups per token, attention block default against off, K2h-r16, 9 cards (counts, not times) $(date +%T)"
  scp -q $R/scripts/cuda_profile_summary.py "$RIG":/tmp/
  for s in default 0; do
    e=""; [ "$s" = 0 ] && e="GGML_CUDA_ATTN_BLOCK=0"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 $e GGML_CUDA_PROFILE=/tmp/rc4-prof-$s.csv $B/llama-bench -m $K2H -ngl 99 -fa on -p 0 -n 32 -r 1 > /dev/null 2>&1; python3 /tmp/cuda_profile_summary.py /tmp/rc4-prof-$s.csv --tokens 24 --by op 2>&1 | head -30" \
      | sed "s/^/ATTN_BLOCK $s: /"
  done
  ssh "$RIG" "rm -f /tmp/rc4-prof-*.csv /tmp/cuda_profile_summary.py"
  echo "== 0d. Q163 prompt attention, K2h-r16, 6 cards, 512-token batches $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 $B/llama-perplexity -m $K2H $PPL512 -lv 4 --kl-divergence-base /tmp/rc4-pp-default.kld 2>&1" \
    | grep -oE "compute-6.1 prompt attention kernel[^(]*|Final estimate.*" | sort | uniq -c | sed "s/^/default: /"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C6 GGML_CUDA_FATTN_TILE_PASCAL=0 $B/llama-perplexity -m $K2H $PPL512 --kl-divergence-base /tmp/rc4-pp-off.kld 2>&1" \
    | grep -oE "Final estimate.*" | sed "s/^/GGML_CUDA_FATTN_TILE_PASCAL=0: /"
  ssh "$RIG" "cmp -s /tmp/rc4-pp-default.kld /tmp/rc4-pp-off.kld && echo 'prompt attention: IDENTICAL to off (kernel not reached)' || echo 'prompt attention: differs from off (the default runs it)'; rm -f /tmp/rc4-pp-*.kld"
fi
if has 2; then
  ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    echo "== 2. llama-pipespec plain against pipelined at the defaults, $tag, 9 cards, depths 1 and 3, -lv 4 $(date +%T)"
    ssh "$RIG" "env BIN=$B MODEL=$m TAG=rc4-$tag-9c GPUS=$C9 DEPTHS='1 3' N=256 GPU_LOG=0 EXTRA_ARGS='-lv 4' bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/rc4-$tag-9c.out 2>&1
      O=/tmp/pipespec/out/rc4-$tag-9c; tail -2 /tmp/pipespec/rc4-$tag-9c.out | grep -E 'exit|error'
      awk -F'\t' 'NR > 1 { print \$1, \$14; n++; if (\$14 == \"NO\") bad++ } END { printf \"%d jobs, %d not same as plain\\n\", n, bad }' \$O/summary.tsv
      echo \"first-use lines: persistent \$(grep -c 'run as one persistent kernel' \$O/run.log), prompt attention \$(grep -c 'compute-6.1 prompt attention kernel' \$O/run.log); rescale warnings: \$(grep -c rescale_notice \$O/run.log)\""
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec" | tar xzf - -C "$OUT"
  ssh "$RIG" "rm -rf /tmp/pipespec"
fi
if has 3; then
  ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_server"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
  for arm in "k2h|$K2H" "ud|$UD"; do
    IFS='|' read -r tag m <<< "$arm"
    case " ${ARMS3:-k2h ud} " in *" $tag "*) ;; *) continue ;; esac   # ARMS3='ud' runs one model
    echo "== 3. llama-server --pipe-depth against no guesses at the defaults, $tag, 9 cards, -lv 4 $(date +%T)"
    ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 BIN=$B MODEL=$m ARGS='-ngl 99 -fa on -c 8192 -ub 512 -lv 4' DEPTHS='3' GUESSES='cont' KEEP_SPEED=0 OUT=/tmp/pipespec_server/$tag bash /tmp/pipespec/check_server.sh 2>&1 | tail -20
      for f in /tmp/pipespec_server/$tag/*.log; do echo \"\$(basename \$f): first-use lines persistent \$(grep -c 'run as one persistent kernel' \$f), prompt attention \$(grep -c 'compute-6.1 prompt attention kernel' \$f), rescale warnings \$(grep -c rescale_notice \$f)\"; done"
  done
  ssh "$RIG" "tar czf - -C /tmp pipespec_server" | tar xzf - -C "$OUT"
  ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_server"
fi
if has d; then
  if [ -z "$HEAD" ]; then
    echo "== d. skipped: no HEAD (a drafter head .gguf on the rig)"
  else
    ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_check"; scp -q -r $R/bench/pipespec "$RIG":/tmp/
    # 6c runs llama's own split, as the e2e does (-ts 9,8,8,8,8,8 put 6 GB on card 0 and failed to load, 2026-10-25)
    for arm in "6c-greedy|$C6||--pipe-backend-greedy|--temp 0" \
               "9c-greedy|$C9|6,6,6,6,5,5,5,5,5|--pipe-backend-greedy|--temp 0" "9c-makers|$C9|6,6,6,6,5,5,5,5,5|--pipe-backend-sample|$MAKERS" \
               "9light-greedy|$C9|6,6,6,6,6,6,6,5,2|--pipe-backend-greedy|--temp 0" "9light-makers|$C9|6,6,6,6,6,6,6,5,2|--pipe-backend-sample|$MAKERS"; do
      IFS='|' read -r tag cards ts bg samp <<< "$arm"
      echo "== d. drafter identity, $(basename "$HEAD"), K2h-r16, $tag, depths 1 and 3, -lv 4 $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards BIN=$B MODEL=$K2H ARGS='-ngl 99 -fa on -c 4096 -ub 512 ${ts:+-ts $ts} $bg -lv 4' SAMPLING='$samp' PROMPT=/tmp/pipespec/prompts/edit.txt N=128 DEPTHS='1 3' MISSES='0.5' DRAFT=$HEAD SPEED=0 OUT=/tmp/pipespec_check/$tag bash /tmp/pipespec/check_identity.sh 2>&1 | grep -E '^(run|plain|pipe|stair|cont4|copy2|oracle)|TIMEOUT|exit|only|no plain'
        L=/tmp/pipespec_check/$tag/run.log; grep -m1 -o 'staircase drafter .*' \$L
        echo \"first-use lines: persistent \$(grep -c 'run as one persistent kernel' \$L), prompt attention \$(grep -c 'compute-6.1 prompt attention kernel' \$L); stair runs: \$(grep -c 'name=stair' /tmp/pipespec_check/$tag/results.txt)\""
    done
    ssh "$RIG" "tar czf - -C /tmp pipespec_check" | tar xzf - -C "$OUT"
    ssh "$RIG" "rm -rf /tmp/pipespec /tmp/pipespec_check"
  fi
fi
if has 4; then
  for npp in $DEPTHS; do
    for s in 0 default; do
      e=""; [ "$s" = 0 ] && e="GGML_CUDA_FATTN_TILE_PASCAL=0"
      echo "== 4. Q163 prompt speed, K2h-r16, 9 cards, $npp-token prompt, prompt attention $s $(date +%T)"
      ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 $e $B/llama-batched-bench -m $K2H -ngl 99 -fa on -c $((npp + 512)) -b 2048 -ub 512 -npp $npp -ntg 32 -npl 1 -lv 4 2>&1" \
        | grep -E '^\|  *[0-9]|error|fail|out of memory|prompt attention kernel' | sed -E 's/^[0-9.]+ I //'
    done
  done
fi
echo RC4_DONE
} | tee -a "$OUT/rig_rc4.md"
