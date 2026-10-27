# Q106 (2026-10-24): shared definitions for the long-context KLD jobs, sourced by the job scripts in this folder.
# Env: BIN (bin/ of a CUDA build of Miner Llama with patches/perplexity-parse-special-kld-dump.diff), WORK (holds
# agent/eval.txt and agent12/eval.txt from ../eval/, wikitext-2-raw/, and gets the reference files, dumps and logs),
# MODELS (the GGUF files), Q8 (Unsloth's Flint Q8_0, default $MODELS/flint-Q8_0.gguf), GPUS (e.g. 5,6).
B=$BIN; W=$WORK; L=$WORK/logs; mkdir -p $L
WIKI=$W/wikitext-2-raw/wiki.test.raw
Q8=${Q8:-$MODELS/flint-Q8_0.gguf}
FLAGS="-b 2048 -ub 512 -ngl 99 -fa on"
run() { docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID "$@"; }
MOUNTS="-v $WORK:$WORK -v $MODELS:$MODELS:ro -v $(dirname "$BIN"):$(dirname "$BIN"):ro"
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
# one candidate against the three references; $1 = name, $2 = file, $3 = extra docker args (rescale switch)
score() {
  for set in agent:c8192:$W/agent/eval.txt wiki8k:c8192:$WIKI wiki512:c512:$WIKI; do
    s=${set%%:*}; rest=${set#*:}; c=${rest%%:*}; f=${rest#*:}
    run $MOUNTS $3 -e LLAMA_KLD_DUMP=$W/dump-$1-$s.tsv $IMG $B/llama-perplexity -m $2 -f $f -c ${c#c} $FLAGS \
      --kl-divergence-base $W/q8.$s.kld --kl-divergence > $L/$1-$s.log 2>&1
    grep -E "^Mean +KLD|^Same top p|^Mean PPL\(Q\)/PPL\(base\)" $L/$1-$s.log | sed "s/^/$1 $s: /"
  done
}
# one candidate against one reference on one text; $1 name, $2 file, $3 extra docker args, $4 reference file in $W,
# $5 label for the dump and log, $6 text directory in $W (default agent)
score1() {
  run $MOUNTS $3 -e LLAMA_KLD_DUMP=$W/dump-$1-$5.tsv $IMG $B/llama-perplexity -m $2 -f $W/${6:-agent}/eval.txt -c 8192 $FLAGS \
    --kl-divergence-base $W/$4 --kl-divergence > $L/$1-$5.log 2>&1
  grep -E "^Mean +KLD|^Same top p|^Mean PPL\(Q\)/PPL\(base\)" $L/$1-$5.log | sed "s/^/$1 $5: /"
}
