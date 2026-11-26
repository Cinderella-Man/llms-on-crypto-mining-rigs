#!/usr/bin/env bash
# C839 (Q1473): Flash-Next's attention call (head 256, 2 KV heads of 12 query heads, 2,051 kept cells a token, F16 and
# Q8_0 caches in the cache layout) timed alone on one P106-090 card with the engine build's own test-backend-ops perf
# mode, at 1 and 4 query tokens over 4,096 to 131,072 cells:
#   dense  GGML_CUDA_FATTN_SPARSE_PASCAL=0: the stock kernels over every cell
#   token  =2 (any cache size) with _NSM=5: the per-token sparse kernel, each token its own list
#   union  =2 _UNION=1 _NSM=5: the union kernel, one list for the 4 tokens (4-token calls only)
# then the split count of each sparse kernel (_SPLITS 1 to 32) at 4,096 and 8,192 cells, F16. Each time is the op as the
# engine runs it: the mask compaction and the kernel. REPEATS rounds of every arm, one process an arm, in this order.
#
# STEP=start    write runs.sh to $RIG:$WORK and start it detached; `done` when every run has ended
# STEP=status   which runs have ended
# STEP=stop     stop the runs (runs.sh, test-backend-ops and the clock sampler)
# STEP=collect  copy the outputs to $OUT
# STEP=clean    remove $WORK from the rig (only when no test-backend-ops runs)
# Env: RIG (ssh alias), WORK (folder on the rig), BIN (the engine build's bin folder on the rig), CARD (default 0),
# REPEATS (default 3), OUT (collect's folder).
set -euo pipefail
RIG=${RIG:?RIG=<ssh alias of the rig>}
WORK=${WORK:?WORK=<folder on the rig>}
CARD=${CARD:-0}
REPEATS=${REPEATS:-3}
STEP=${STEP:?STEP=start|status|stop|collect|clean}

case $STEP in
start)
    BIN=${BIN:?BIN=<the build bin folder on the rig>}
    timeout 60 ssh "$RIG" "test -x $BIN/test-backend-ops && mkdir -p $WORK && ! test -e $WORK/started" ||
        { echo "no test-backend-ops in BIN, or already started"; exit 1; }
    timeout 60 ssh "$RIG" "cat > $WORK/runs.sh" <<EOF
#!/usr/bin/env bash
cd $WORK
trap 'kill \$(jobs -p) 2> /dev/null' EXIT
TBO=$BIN/test-backend-ops
F='nr23=\\[12,1\\],kv=[0-9]+,nb=[14],.*permute=\\[0,2,1,3\\].*n_kv_max=2051'
F4='nr23=\\[12,1\\],kv=[0-9]+,nb=4,.*permute=\\[0,2,1,3\\].*n_kv_max=2051'
SW='nr23=\\[12,1\\],kv=(4096|8192),nb=[14],.*type_K=f16,.*permute=\\[0,2,1,3\\].*n_kv_max=2051'
SW4='nr23=\\[12,1\\],kv=(4096|8192),nb=4,.*type_K=f16,.*permute=\\[0,2,1,3\\].*n_kv_max=2051'
SP='GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5'
run() { # name, filter, variables
    local name=\$1 re=\$2
    shift 2
    nvidia-smi -i $CARD --query-gpu=timestamp,clocks.sm,clocks.mem,pstate --format=csv,noheader -lms 1000 > \$name.clocks 2>&1 &
    local smi=\$!
    local t0=\$(date +%T)
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD "\$@" timeout 300 \$TBO perf -b CUDA0 -o FLASH_ATTN_EXT -p "\$re" --output sql > \$name.sql 2> \$name.err
    echo "\$name rc=\$? \$t0 \$(date +%T) \$*" >> runs.log
    kill \$smi
    wait \$smi 2> /dev/null
}
for r in \$(seq 1 $REPEATS); do
    run dense-r\$r "\$F"  GGML_CUDA_FATTN_SPARSE_PASCAL=0
    run token-r\$r "\$F"  \$SP
    run union-r\$r "\$F4" \$SP GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1
    for S in 1 2 3 4 5 6 8 10 16 20 32; do
        run sweep-token-S\$S-r\$r "\$SW"  \$SP GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS=\$S
        run sweep-union-S\$S-r\$r "\$SW4" \$SP GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1 GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS=\$S
    done
done
date +%T > done
EOF
    # an ssh call that starts a detached job may wait for it: the timeout ends the call, not the job
    timeout 30 ssh "$RIG" "date +%T > $WORK/started; cd $WORK && setsid nohup bash runs.sh > runs.out 2>&1 < /dev/null & sleep 2; echo started" || true
    timeout 60 ssh "$RIG" "ps -eo comm | grep -x test-backend-op || echo 'test-backend-ops not running yet'"
    ;;
status)
    timeout 60 ssh "$RIG" "cd $WORK && cat started; wc -l < runs.log 2> /dev/null; tail -n 3 runs.log 2> /dev/null; cat done 2> /dev/null || true"
    ;;
stop)
    timeout 60 ssh "$RIG" "p=\$(ps -eo pid,args | grep '[b]ash runs.sh' | awk '{print \$1}'); [ -n \"\$p\" ] && kill \$p; pkill -x test-backend-op; sleep 2; p=\$(ps -eo pid,args | grep '[n]vidia-smi -i $CARD --query-gpu=timestamp' | awk '{print \$1}'); [ -n \"\$p\" ] && kill \$p; ps -eo comm | grep -x test-backend-op || echo stopped"
    ;;
collect)
    OUT=${OUT:?OUT=<local folder>}
    mkdir -p "$OUT"
    timeout 300 ssh "$RIG" "cd $WORK && tar czf - started done runs.sh runs.log runs.out *.sql *.err *.clocks" | tar xzf - -C "$OUT"
    ls "$OUT" | wc -l
    ;;
clean)
    timeout 60 ssh "$RIG" "if ps -eo comm | grep -qx test-backend-op; then echo 'test-backend-ops still runs'; exit 1; fi; rm -rf $WORK; echo removed"
    ;;
*)
    echo "STEP=start|status|stop|collect|clean" >&2
    exit 2
    ;;
esac
