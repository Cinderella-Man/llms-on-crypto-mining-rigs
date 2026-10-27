#!/bin/bash
# Q30 BFCL v3 check (2026-10-23): after BF16, every quant runs on the same RTX 3090 pair (GPUs 5,6), one after another.
# OUT: the harness folder (mounted at /bfcl in the client container bfcl-py); MODELS: folder holding the GGUF files;
# BIN: the main llama-server build; BIN_G64: the q41g64 build (Q4_1_64 type); WORK as in serve.sh.
cd "${OUT:?}"
until grep -q DONE logs/run-BF16.log 2>/dev/null; do sleep 10; done
bash stop_server.sh bfcl-srv-d
# the first makers run (V100 pair) and the partial mixS4 run (GPUs 0,1,2) are kept as cross-hardware checks
until grep -q DONE logs/run-makers-q4_1.log; do sleep 10; done
bash stop_server.sh bfcl-srv-a
mv raw/makers-q4_1.jsonl raw/makers-q4_1_v100.jsonl; mv logs/run-makers-q4_1.log logs/run-makers-q4_1_v100.log
mv raw/mixS4.jsonl raw/mixS4_gpu012_partial.jsonl; mv logs/run-mixS4.log logs/run-mixS4_gpu012_partial.log
MAIN=${BIN:?}
G64=${BIN_G64:?}
NR=LLAMA_NO_MOE_DOWN_RESCALE=1
run() {  # TAG MODEL BIN [ENV]
  bash serve.sh bfcl-srv-e 18085 5,6 1,1 "$2" "$3" $4 || { echo "serve failed for $1"; return 1; }
  docker exec bfcl-py bash -c "cd /bfcl && python run.py http://127.0.0.1:18085 tasks.jsonl raw/$1.jsonl 2 4096 > logs/run-$1.log 2>&1"
  bash stop_server.sh bfcl-srv-e
  echo "finished $1"
}
run makers-q4_1 "$MODELS/flint-int4-q4_1.gguf" $MAIN
run mixS4 "$MODELS/flint-mixS4.gguf" $MAIN $NR
run mixS1 "$MODELS/flint-mixS1.gguf" $MAIN $NR
run makers-fast "$MODELS/flint-makers-fast.gguf" $G64 $NR
run udmirror-v1-nr_rep "$MODELS/flint-udmirror-v1-nr.gguf" $MAIN $NR
echo "phase3 all done"
