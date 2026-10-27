#!/bin/bash
# Q30 BFCL v3 check (2026-10-23): after UD-mirror finishes on GPUs 5,6, pause mixS4 (resumable) and serve BF16 on
# 0,1,2,5,6. OUT: the harness folder (these scripts, the harness, tasks.jsonl, raw/, logs/), mounted at /bfcl in the
# client container bfcl-py; MODELS: folder holding the GGUF files; WORK, BIN as in serve.sh.
cd "${OUT:?}"
until grep -q DONE logs/run-udmirror-v1-nr.log; do sleep 10; done
docker exec bfcl-py pkill -f "raw/mixS4.jsonl" || true
sleep 2
bash stop_server.sh bfcl-srv-b
bash stop_server.sh bfcl-srv-c
bash serve.sh bfcl-srv-d 18084 0,1,2,5,6 10,12,24,24,24 "$MODELS/flint-BF16.gguf" || exit 1
docker exec -d bfcl-py bash -c "cd /bfcl && python run.py http://127.0.0.1:18084 tasks.jsonl raw/BF16.jsonl 2 4096 > logs/run-BF16.log 2>&1"
echo "phase2 started BF16"
