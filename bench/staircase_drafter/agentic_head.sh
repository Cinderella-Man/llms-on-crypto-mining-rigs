#!/usr/bin/env bash
# Flint's MTP head (the staircase drafter, method M4 in docs/drafter.md) from agent sessions, on a GPU machine with
# Docker: one export of the agentic data runs, a held-out set by module, the dump, the vocabulary, training, scoring
# and the GGUF side file. Every file a later step reads is gated with check_sources.py first.
#
#   WORK=<out> RUNS="<agentic run WORK> ..." MODEL=<flint-Q8_0.gguf> HF_DIR=<RecursiveSI/flint small files> \
#   LLAMA=<llama.cpp tree with build-cuda/bin/llama-staircase-dump> Q51_GEN=<Q51 answers> Q51=<Q51 prompts.jsonl> \
#   Q51_SEQS=<Q51's dump input, rows numbered 0, 1, ... in Q51_GEN's order> bash bench/staircase_drafter/agentic_head.sh
#
# Environment:
#   WORK        output folder: export/, ids/, seqs/, flint/, vocab.npy, heads/, scores/, gguf/, head.log
#   RUNS        the bench/agentic_data/run.sh folders whose sessions.gate.json passed (several writers: one export)
#   FEATS_DIRS  folders for the dump files, one per dump instance in turn (default WORK/feats): put them on several
#               disks, since trainers read them at random
#   DUMP_SETS   GPU sets, one llama-staircase-dump each, fed from one pool (default "5,6 0,1,2 3,4"); the first set
#               dumps the test files (Q51, held-out sessions) before it joins
#   HEADS       "name:GPU:train.py options" per head, run at once (default a 3-12-card head and a nine-card one)
#   MINUTES     training budget per head (default 240); CTX the committed positions the head attends to, as the
#               engine's --pipe-draft-ctx (default 4096); CHUNK target positions per training piece (16384)
#   GRID        the dumped layer inputs (default the 26-layer grid of M2); DCTX the dump's context (66048)
#   EVERY TAKE  held-out modules: every EVERY-th by hash (20), TAKE sessions of them scored (40)
#   STEPS       default "export split weights dump vocab train score gguf"; any subset, in order
# Every container carries the label pascal-quants=1 and --cpu-shares 256 (training and dumps run on their own GPUs);
# the script's own Python runs at nice 19. Quality and acceptance only: times on a shared machine are not speed results.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)
: "${WORK:?}" "${RUNS:?}" "${MODEL:?}" "${HF_DIR:?}" "${LLAMA:?}" "${Q51_GEN:?}" "${Q51:?}" "${Q51_SEQS:?}"
FEATS_DIRS=${FEATS_DIRS:-$WORK/feats}
DUMP_SETS=${DUMP_SETS:-5,6 0,1,2 3,4}
HEADS=${HEADS:-gen:2:--general|nine:5:--general --cards 9e,9l,9l}
MINUTES=${MINUTES:-240}; CTX=${CTX:-4096}; CHUNK=${CHUNK:-16384}; DCTX=${DCTX:-66048}
GRID=${GRID:-4,6,8,9,10,12,14,16,17,18,20,22,24,25,27,29,30,33,34,36,37,39,41,42,44,47}
EVERY=${EVERY:-20}; TAKE=${TAKE:-40}
STEPS=${STEPS:-export split weights dump vocab train score gguf}
SRC=$HERE/sources.json; CB=$BENCH/agentic_data/codebases.json; AGT=$BENCH/agent_check/tasks
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
BIN=$LLAMA/build-cuda/bin/llama-staircase-dump
PY="nice -n 19 python3 -I"
X=$WORK/export/train
mkdir -p "$WORK"/{ids,seqs,heads,scores,gguf}
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$WORK/head.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
mounts=(-v "$BENCH:$BENCH:ro" -v "$WORK:$WORK" -v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro")
for d in $FEATS_DIRS; do mkdir -p "$d"; mounts+=(-v "$d:$d"); done
for r in $RUNS; do mounts+=(-v "$r:$r:ro"); done
mounts+=(-v "$(dirname "$Q51_GEN"):$(dirname "$Q51_GEN"):ro" -v "$LLAMA:$LLAMA:ro")
# numpy and torch tools run in the PyTorch image on the CPU, from this folder: trun runs a command line (recorded
# as such by the gates), gate runs check_sources.py (the dump gate reads the feature files with numpy)
trun() { docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none "${mounts[@]}" \
           -e PYTHONPATH="$LLAMA/gguf-py" -w "$HERE" "$TORCH_IMAGE" bash -c "$1"; }
gate() { docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none "${mounts[@]}" -w "$HERE" \
           "$TORCH_IMAGE" python -I check_sources.py "$SRC" "$@" | tail -1 | tee -a "$WORK/head.log"; }
gates=(); for r in $RUNS; do gates+=("$r/sessions.gate.json"); done
held=(--q51 "$Q51" --agent-tasks "$AGT")

if has export; then
  log "one export of the runs' passed sessions: ${RUNS}"
  $PY "$BENCH/agentic_data/export.py" "${gates[@]}" --codebases "$CB" --apache "$HF_DIR/LICENSE" \
    --out "$WORK/export" | tee -a "$WORK/head.log"
  $PY "$BENCH/agentic_data/gate.py" answers "$X/sessions_seq.jsonl" --parent "${gates[@]}" --sources "$SRC" \
    --codebases "$CB" --out "$X/seq.gate.json" | tail -1 | tee -a "$WORK/head.log"
  $PY "$BENCH/agentic_data/gate.py" answers "$X/turns.jsonl" --parent "${gates[@]}" --sources "$SRC" \
    --codebases "$CB" --out "$X/answers.gate.json" | tail -1 | tee -a "$WORK/head.log"
  $PY "$BENCH/agentic_data/gate.py" answers "$X/turn_prompts.jsonl" --parent "${gates[@]}" --sources "$SRC" \
    --codebases "$CB" --out "$X/turn_prompts.gate.json" | tail -1 | tee -a "$WORK/head.log"
  $PY "$BENCH/agentic_data/gate.py" dataset "$WORK/export/hf" --parent "${gates[@]}" --sources "$SRC" \
    --codebases "$CB" "${held[@]}" --out "$WORK/export/hf.gate.json" | tail -1 | tee -a "$WORK/head.log"
fi

if has split; then
  log "held-out sessions: every ${EVERY}th module, ${TAKE} sessions scored"
  $PY "$BENCH/agentic_data/heldout.py" "$X/sessions_seq.jsonl" --every "$EVERY" --take "$TAKE" \
    --test "$WORK/ids/test.txt" --train "$WORK/ids/train.txt" --test-rows "$WORK/seqs/test-rows.jsonl" \
    | tee -a "$WORK/head.log"
  trun "python -I feats.py seqs $WORK/seqs/test.txt $WORK/seqs/test-map.jsonl $WORK/seqs/test-rows.jsonl \
--max-tok 1000000" | tee -a "$WORK/head.log"
  # Q51's ids number each category on its own (chat_0, code_0, ...): its dump input numbers the rows instead, and
  # score.py labels it by row
  cp "$Q51_SEQS" "$WORK/seqs/q51.txt"
fi

if has weights; then
  log "Flint's embedding table, output head and output norm"
  trun "python flint_weights.py $MODEL $WORK/flint" | tail -3 | tee -a "$WORK/head.log"
fi

if has dump; then
  log "dump: test files on the first set, then every training session over: $DUMP_SETS"
  names=(); feats=(); dirs=($FEATS_DIRS); k=0
  for g in $DUMP_SETS; do
    n=d${g//,/}; names+=("$n"); d=${dirs[$((k % ${#dirs[@]}))]}; k=$((k + 1))
    feats+=("$d/train-$n.bin"); touch "$WORK/seqs/train-$n.txt" "$WORK/seqs/train-$n.log"
    env=(DUMP_LAYERS=$GRID DUMP_ALLPOS=4,6,9 DUMP_THREADS=12 DUMP_GPU_TOPK=1)
    run="$BIN -m $MODEL -ngl 99 -fa on -c $DCTX -b 512 -ub 512"
    pre=""
    if [ "$k" = 1 ]; then   # the test files: Q51 and the held-out sessions, one pass each
      pre="DUMP_IN=$WORK/seqs/q51.txt DUMP_OUT=$WORK/seqs/q51.bin ${env[*]} $run >> $WORK/seqs/test-dump.log 2>&1 &&
           DUMP_IN=$WORK/seqs/test.txt DUMP_OUT=$WORK/seqs/test.bin ${env[*]} $run >> $WORK/seqs/test-dump.log 2>&1 &&"
    fi
    docker rm -f "pq-head-$n" >/dev/null 2>&1 || true
    docker run -d --name "pq-head-$n" --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$g\"" \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${mounts[@]}" "$CUDA_IMAGE" bash -c "$pre
      tail -n +1 -F $WORK/seqs/train-$n.txt | DUMP_IN=/dev/stdin DUMP_OUT=$d/train-$n.bin ${env[*]} $run \
        >> $WORK/seqs/train-$n.log 2>&1" >/dev/null
  done
  inst=()
  for n in "${names[@]}"; do
    s=$WORK/seqs/train-$n; inst+=(--inst "$n:$s.txt:$s-map.jsonl:$s.log")
  done
  docker rm -f pq-head-feed >/dev/null 2>&1 || true
  docker run -d --name pq-head-feed --label pascal-quants=1 --cpu-shares 256 "${mounts[@]}" python:3.12-slim \
    python3 -I "$HERE/feed_split.py" --gen "$X/sessions_seq.jsonl" --only "$WORK/ids/train.txt" "${inst[@]}" \
    --every 60 --low 400000 --chunk 800000 --max-tok 1000000 --stop "$WORK/seqs/FEED_STOP" \
    >/dev/null
  want=$(wc -l < "$WORK/ids/train.txt")
  while :; do   # done when every training session is in some instance's log
    sleep 300
    got=$(cat "$WORK"/seqs/train-*.log | grep -c '^sequence ' || true)
    log "dump: $got of $want training sessions"
    for n in "${names[@]}"; do
      if [ "$(docker inspect -f '{{.State.Running}}' "pq-head-$n" 2>/dev/null)" != true ]; then
        log "dump pq-head-$n stopped"; exit 1
      fi
    done
    [ "$got" -ge "$want" ] && break
  done
  touch "$WORK/seqs/FEED_STOP"
  sleep 5
  for n in "${names[@]}"; do docker rm -f "pq-head-$n" >/dev/null; done
  docker rm -f pq-head-feed >/dev/null 2>&1 || true
  gate dump "${feats[@]}" --parent "$X/seq.gate.json" --out "$WORK/feats.gate.json"
  gate dump "$WORK/seqs/test.bin" --parent "$X/seq.gate.json" --out "$WORK/seqs/test.gate.json"
  trun "python -I check_dump.py ${feats[0]} $WORK/flint --max-seqs 2" | tail -4 | tee -a "$WORK/head.log"
fi

FEATS=$(for d in $FEATS_DIRS; do ls "$d"/train-*.bin 2>/dev/null; done | tr '\n' ' ')

if has vocab; then
  log "vocabulary: the 40,000 most frequent generated ids of the training sessions"
  cmd="python -I vocab.py $WORK/vocab.npy $X/sessions_seq.jsonl --size 40000 --only $WORK/ids/train.txt \
--test $WORK/seqs/test-rows.jsonl"
  trun "$cmd" | tee -a "$WORK/head.log"
  gate derived "$WORK/vocab.npy" --kind vocab --parent "$X/seq.gate.json" --cmd "$cmd" \
    --test "$WORK/seqs/test-rows.jsonl" --out "$WORK/vocab.gate.json"
fi

if has train; then
  IFS='|' read -ra hs <<< "$HEADS"
  for h in "${hs[@]}"; do
    name=${h%%:*}; rest=${h#*:}; gpu=${rest%%:*}; opts=${rest#*:}
    H=$WORK/heads/$name
    cmd="python -I -u train.py --head A --feats $FEATS --vocab $WORK/vocab.npy --flint $WORK/flint --out $H \
--minutes $MINUTES --spans $X/sessions_seq.jsonl --ctx $CTX --chunk-targets $CHUNK --max-tok 1000000 $opts"
    log "head $name on GPU $gpu: $opts"
    mkdir -p "$H"; echo "$cmd" > "$H/cmd.txt"
    docker rm -f "pq-head-$name" >/dev/null 2>&1 || true
    docker run -d --name "pq-head-$name" --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$gpu\"" \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID --shm-size 8g --network none "${mounts[@]}" -w "$HERE" "$TORCH_IMAGE" \
      bash -c "$cmd >> $H/train.log 2>&1" >/dev/null
  done
  for h in "${hs[@]}"; do
    name=${h%%:*}
    docker wait "pq-head-$name" >/dev/null
    log "head $name: $(tail -1 "$WORK/heads/$name/train.log" | cut -c1-200)"
    docker rm "pq-head-$name" >/dev/null
    gate derived "$WORK/heads/$name/head.pt" --kind head --parent "$WORK/feats.gate.json" "$WORK/vocab.gate.json" \
      "$X/seq.gate.json" --cmd "$(cat "$WORK/heads/$name/cmd.txt")" --allow-dir "$WORK/heads/$name" "$WORK/flint" \
      --out "$WORK/heads/$name/head.gate.json"
  done
fi

if has score; then
  IFS='|' read -ra hs <<< "$HEADS"
  dirs=(); for h in "${hs[@]}"; do dirs+=("$WORK/heads/${h%%:*}"); done
  gpu=${DUMP_SETS%% *}; gpu=${gpu%%,*}
  for t in q51 test; do
    if [ $t = q51 ]; then g=$Q51_GEN; sp=(); else g=$WORK/seqs/test-rows.jsonl; sp=(--spans "$g"); fi
    log "scores on $t (GPU $gpu)"
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$gpu\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      --network none "${mounts[@]}" -w "$HERE" "$TORCH_IMAGE" python -I score.py --heads "${dirs[@]}" \
      --feats "$WORK/seqs/$t.bin" --gen "$g" --vocab "$WORK/vocab.npy" --flint "$WORK/flint" "${sp[@]}" \
      --out "$WORK/scores/$t" > "$WORK/scores/$t.md" 2>&1
    grep -E "^## head|\| g9light \||\| g9 \||\| g6 \|" "$WORK/scores/$t.md" | tee -a "$WORK/head.log" || true
  done
fi

if has gguf; then
  IFS='|' read -ra hs <<< "$HEADS"
  for h in "${hs[@]}"; do
    name=${h%%:*}
    out=$WORK/gguf/flint-mtp-$name.gguf
    cmd="python export_gguf.py --head $WORK/heads/$name --vocab $WORK/vocab.npy --out $out --type q8_0 --alpha 3"
    log "GGUF side file: $out"
    trun "$cmd" | tail -2 | tee -a "$WORK/head.log"
    gate derived "$out" --kind model --parent "$WORK/heads/$name/head.gate.json" "$WORK/vocab.gate.json" \
      --cmd "$cmd" --allow-dir "$WORK/heads/$name" --out "$out.gate.json"
  done
fi
log "done: $STEPS"
