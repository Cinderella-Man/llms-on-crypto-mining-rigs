# Q126/Q127 (2026-10-24): agent 8k KLD on two V100s (an sm_70 build of the Q106 tree), against Q106's Q8_0 reference,
# with per-token dumps next to Q106's; sourced by the job scripts. Env: BIN (bin/ of the CUDA build), WORK (Q106's
# folder: agent/eval.txt, q8.agent.kld, the dumps; see docs/data/2026-10-24_q106/run/reference.sh), MODELS, UD (Unsloth's
# flint-UD-Q4_K_XL.gguf, default $MODELS/flint-UD-Q4_K_XL.gguf), GPUS.
B=$BIN; W=$WORK; L=$WORK/logs; mkdir -p $L
UD=${UD:-$MODELS/flint-UD-Q4_K_XL.gguf}
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
MOUNTS="-v $WORK:$WORK -v $MODELS:$MODELS:ro -v $(dirname "$BIN"):$(dirname "$BIN"):ro"
NR="-e LLAMA_NO_MOE_DOWN_RESCALE=1"
# one file on the 24 agent chunks; $1 name (dump-$1-agent.tsv), $2 file, $3 extra docker args (the rescale switch)
agent() {
  docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID $MOUNTS $3 \
    -e LLAMA_KLD_DUMP=$W/dump-$1-agent.tsv $IMG $B/llama-perplexity -m $2 -f $W/agent/eval.txt --parse-special \
    --no-escape -c 8192 -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence-base $W/q8.agent.kld --kl-divergence \
    > $L/$1-agent.log 2>&1
  grep -E "^Mean +KLD|^Same top p|^Mean PPL\(Q\)/PPL\(base\)" $L/$1-agent.log | sed "s/^/$1: /"
}
