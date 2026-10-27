#!/usr/bin/env bash
# The switches that made plain decoding faster when turned off in ablation.sh, together (2026-10-25), in plain decoding
# (llama-bench) and pipelined decoding (llama-pipespec, built-in guessing), on rc4 2d53179aa with v2-next 62fa5fc3b and
# v1 eeaa59660 as references, nine cards. Run from the repository root with nothing else on the rig.
# Env: RIG (ssh alias), BIN, NEXT, V1 (bin folders on the rig), K2H, Q8 (GGUF paths on the rig), OUT, STEPS.
#   k2h-bench  Flint-Pascal-Q5 (recipe K2h-r16) llama-bench pp512 + tg128: all on; decode attention off; + input layer
#              on the host; + packed uploads off; v2-next
#   k2h-pipe   Flint-Pascal-Q5 llama-pipespec (bench/pipespec/rig_run.sh): all on against the three off, depths 1 and 3
#   q8-bench   Flint Q8_0 the same as k2h-bench (-ts 6/6/5/5/6/5/5/6/6), with v1 as its reference
#   q8-pipe    Flint Q8_0 llama-pipespec: all on against the three off
set -uo pipefail
RIG=${RIG:-rig}
STEPS=${STEPS:-k2h-bench k2h-pipe q8-bench q8-pipe}
T=${T:-llama.cpp-20261025-2d53179aa-pascal-p106-v2-rc4}
B=${BIN:-/opt/llama-builds/$T/bin}
NEXT=${NEXT:-/opt/llama-builds/llama.cpp-20261024-62fa5fc3b-pascal-p106-v2-next/bin}
V1=${V1:-/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
K2H=${K2H:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
Q8=${Q8:-/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf}
C9=0,1,2,3,4,5,6,7,8
OFF1="GGML_CUDA_FATTN_PASCAL=0"
OFF2="GGML_CUDA_FATTN_PASCAL=0 LLAMA_INPUT_ON_GPU=0"
OFF3="GGML_CUDA_FATTN_PASCAL=0 LLAMA_INPUT_ON_GPU=0 LLAMA_INPUT_UPLOAD=0"
O=${OUT:-.}
mkdir -p "$O"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
bench() { # <label> <bin> <model> <env or -> <extra args>
  local label=$1 b=$2 m=$3 e=$4; shift 4
  [ "$e" = - ] && e=""
  echo "== $label $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$C9 LLAMA_NO_MOE_DOWN_RESCALE=tg $e $b/llama-bench -m $m -ngl 99 -fa on -p 512 -n 128 -r 3 $* -o md 2>&1" \
    | grep -E '^\| .*(pp|tg)[0-9]+|error|failed|out of memory|GGML_ASSERT' | sed "s/^/$label | /"
}
pipe() { # <tag> <model> <env or -> [TS]
  local tag=$1 m=$2 e=$3 ts=${4:-}
  [ "$e" = - ] && e=""
  echo "== pipespec $tag ($e) $(date +%T)"
  ssh "$RIG" "env BIN=$B MODEL=$m TAG=$tag GPUS=$C9 ${ts:+TS=$ts} ENVS='$e' DEPTHS='1 3' N=256 GPU_LOG=0 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/$tag.out 2>&1
    tail -2 /tmp/pipespec/$tag.out | grep -E 'exit|error|Segmentation'
    awk -F'\t' 'NR > 1 { print \$1, \$6, \$14; n++; if (\$14 == \"NO\") bad++ } END { printf \"%d jobs, %d not same as plain\\n\", n, bad }' /tmp/pipespec/out/$tag/summary.tsv"
}
{
echo "# Combined off-switches, plain and pipelined, $T, $(date -Is)"
if has k2h-bench; then
  bench "K2h all on" $B $K2H -
  bench "K2h $OFF1" $B $K2H "$OFF1"
  bench "K2h $OFF2" $B $K2H "$OFF2"
  bench "K2h $OFF3" $B $K2H "$OFF3"
  bench "K2h v2-next 62fa5fc3b" $NEXT $K2H -
fi
if has k2h-pipe || has q8-pipe; then ssh "$RIG" "rm -rf /tmp/pipespec"; scp -q -r bench/pipespec "$RIG":/tmp/; fi
if has k2h-pipe; then
  pipe combo-k2h-on $K2H -
  pipe combo-k2h-off3 $K2H "$OFF3"
fi
if has q8-bench; then
  bench "Q8_0 all on" $B $Q8 - -ts 6/6/5/5/6/5/5/6/6
  bench "Q8_0 $OFF1" $B $Q8 "$OFF1" -ts 6/6/5/5/6/5/5/6/6
  bench "Q8_0 $OFF3" $B $Q8 "$OFF3" -ts 6/6/5/5/6/5/5/6/6
  bench "Q8_0 v1 eeaa59660" $V1 $Q8 - -ts 6/6/5/5/6/5/5/6/6
fi
if has q8-pipe; then
  pipe combo-q8-on $Q8 - 6,6,5,5,6,5,5,6,6
  pipe combo-q8-off3 $Q8 "$OFF3" 6,6,5,5,6,5,5,6,6
fi
if has k2h-pipe || has q8-pipe; then
  mkdir -p "$O/combo-out" && ssh "$RIG" "tar czf - -C /tmp pipespec/out" | tar xzf - -C "$O/combo-out"
  ssh "$RIG" "rm -rf /tmp/pipespec"
fi
echo "== done $(date +%T)"
} | tee -a "$O/combo.md"
