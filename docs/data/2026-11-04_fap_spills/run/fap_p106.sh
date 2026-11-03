#!/usr/bin/env bash
# The Pascal decode attention kernel alone on one P106-090: test-backend-ops perf of Flint's one-token decode attention
# per layer call (4 KV heads of 8 query heads, head size 128, F16 cache viewed as in the KV cache, mask and sinks) at
# 256 to 131,072 keys, the 1,024 / 8,192 / 32,768 the comparison is about among them, for four arms:
#   before-stock, before-fap: the -ladder build (98bbd6a2f) with GGML_CUDA_FATTN_PASCAL=0 (the stock kernels) and =1
#   after-stock,  after-fap:  the -fap build, the same two settings
# The stock arms are the same code in both builds (a control for run-to-run spread); the -ladder build has no 1,024 case
# (768 instead). Run it on the rig with nothing else running. Output: $OUT/<arm>.<pass>.txt and $OUT/perf.tsv (arm, pass,
# keys, microseconds per call, GB/s of K and V read: 2,048 bytes per key), plus $OUT/done.
# BEFORE, AFTER: unpacked tarball trees (bin/ inside); CARD: the card (default 0); REPS: passes per arm (default 2,
# alternating the arms so that drift shows).
# Usage: BEFORE=<dir> AFTER=<dir> OUT=<dir> [CARD=0] [REPS=2] fap_p106.sh
set -u
BEFORE=${BEFORE:?BEFORE=<tree of the -ladder build>}
AFTER=${AFTER:?AFTER=<tree of the -fap build>}
OUT=${OUT:?OUT=<output dir>}
CARD=${CARD:-0}
REPS=${REPS:-2}
mkdir -p "$OUT"
rm -f "$OUT/done"
F='hsk=128,hsv=128,nh=4,nr23=\[8,1\],kv=[0-9]+,nb=1,mask=1,sinks=1,.*permute=\[0,2,1,3\]'
{
    echo "start $(date -Is)"
    nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu --format=csv,noheader -i "$CARD"
} > "$OUT/console.txt"
printf "arm\tpass\tkeys\tus_per_call\tkv_GB_per_s\n" > "$OUT/perf.tsv"
for r in $(seq 1 "$REPS"); do
    for arm in before-stock before-fap after-stock after-fap; do
        case $arm in before-*) B=$BEFORE ;; *) B=$AFTER ;; esac
        case $arm in *-fap) m=1 ;; *) m=0 ;; esac
        CUDA_VISIBLE_DEVICES=$CARD GGML_CUDA_FATTN_PASCAL=$m "$B/bin/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT \
            -p "$F" > "$OUT/$arm.$r.txt" 2>&1
        echo "$arm rep $r: exit $? $(date +%T)" >> "$OUT/console.txt"
        # FLASH_ATTN_EXT(hsk=128,...,kv=8192,...):  N runs -  X us/run -  Y GFLOP/run -  Z GFLOPS
        sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/$arm.$r.txt" | grep -E '^ *FLASH_ATTN_EXT\(' | \
            sed -E 's/.*kv=([0-9]+),.* ([0-9.]+) us\/run.*/\1 \2/' | \
            awk -v a="$arm" -v r="$r" '{ printf "%s\t%s\t%s\t%s\t%.1f\n", a, r, $1, $2, $1*2048/($2*1e3) }' >> "$OUT/perf.tsv"
    done
done
echo "end $(date -Is)" >> "$OUT/console.txt"
touch "$OUT/done"
