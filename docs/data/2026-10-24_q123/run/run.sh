#!/usr/bin/env bash
# Q123 (2026-10-24): the three stop_decision.py analyses. TOOLS: bench/long_context_kld at 5143f10; DUMPS: Q106/Q107's
# per-token dumps (dump-<file>-agent.tsv, dump-<file>-bf16agent12.tsv) with agent/ and agent12/ chunks.jsonl; MODEL: a
# Flint GGUF for the vocabulary (G3 here); OUT: output folder. CPU only, llama.cpp's gguf-py on PYTHONPATH.
# Q123: log p of the actual </think> (stops) and of hesitation markers, per file, from Q106/Q107's per-token dumps
set -euo pipefail
: "${TOOLS:?}" "${DUMPS:?}" "${MODEL:?}" "${OUT:?}"
cd "$TOOLS"
W=$DUMPS; M=$MODEL
python stop_decision.py --ref ud --chunks $W/agent/chunks.jsonl --model $M --base-name Q8_0 \
  ud=$W/dump-ud-agent.tsv g3=$W/dump-g3-agent.tsv g3h=$W/dump-g3h-agent.tsv k2h=$W/dump-k2h-agent.tsv \
  k1h=$W/dump-k1h-agent.tsv g3g=$W/dump-g3g-agent.tsv k2hg=$W/dump-k2hg-agent.tsv > "$OUT/stop-agent.md"
python stop_decision.py --ref ud --chunks $W/agent12/chunks.jsonl --model $M --base-name BF16 \
  ud=$W/dump-ud-bf16agent12.tsv q8=$W/dump-q8-bf16agent12.tsv g3=$W/dump-g3-bf16agent12.tsv \
  g3g=$W/dump-g3g-bf16agent12.tsv k2h=$W/dump-k2h-bf16agent12.tsv k2hg=$W/dump-k2hg-bf16agent12.tsv > "$OUT/stop-bf16agent12.md"
echo Q123_DONE
# the 12 BF16 chunks cut from the 24-chunk dumps: separates the subset from the reference (a file's own log p does not
# depend on the reference)
cd "$TOOLS"
python stop_decision.py --ref ud --chunks $W/agent12/chunks.jsonl --model $M --base-name Q8_0 \
  --only-chunks 0,1,3,6,9,10,12,13,15,18,21,22 ud=$W/dump-ud-agent.tsv g3=$W/dump-g3-agent.tsv \
  g3g=$W/dump-g3g-agent.tsv k2h=$W/dump-k2h-agent.tsv k2hg=$W/dump-k2hg-agent.tsv > "$OUT/stop-agent-sub12.md"
echo Q123_SUB_DONE
