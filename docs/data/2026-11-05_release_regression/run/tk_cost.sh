#!/usr/bin/env bash
# The cost table for the adaptive draft length (LLAMA_SPEC_ADAPT_COST): T(k), the milliseconds of one k-token pass,
# from llama-bench's k-token prompt passes (-p 1,...,K -n 0, REPS repetitions) on the model's own split, so
# T(k) = 1000 k / (tokens per second of the k-token pass). Run from the repository root with nothing else on the rig.
# Env: RIG, MODEL, BIN (llama-bench on the rig), ARGS (llama-bench's split and batch flags, shares separated by "/"),
#   DIR (a rig folder), OUT (local folder); optional K (4), REPS (5)
# Output: OUT/tk.jsonl (llama-bench), OUT/tk.err, and the table on the rig as DIR/t_k.txt and here as OUT/t_k.txt
# ("T(1),T(2),...", ms)
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BIN:?}" "${ARGS:?}" "${DIR:?}" "${OUT:?}"
K=${K:-4}; REPS=${REPS:-5}
mkdir -p "$OUT"
ps=$(seq -s, 1 "$K")
ssh "$RIG" "mkdir -p $DIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
ssh "$RIG" "LOG=/dev/shm/rw_tk.tsv PROC=llama-bench MIN_AVAIL_MB=200 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
    cd $DIR && timeout 1800 $BIN -m $MODEL -ngl 99 -fa 1 $ARGS -p $ps -n 0 -r $REPS -o jsonl > tk.jsonl 2> tk.err; echo \"llama-bench rc=\$?\"
    kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_tk.tsv $DIR/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh
    python3 -I -c 'import json,sys
r = sorted((d[\"n_prompt\"], d[\"avg_ts\"]) for d in map(json.loads, open(\"tk.jsonl\")) if d[\"n_gen\"] == 0)
print(\",\".join(\"%.1f\" % (1000.0 * k / ts) for k, ts in r))' > t_k.txt; cat t_k.txt"
for f in tk.jsonl tk.err t_k.txt rw_tk.tsv; do ssh "$RIG" "cat $DIR/$f" > "$OUT/$f"; done
