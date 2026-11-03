#!/usr/bin/env bash
# qsa_gpu.sh: Q230's checks and kernel timings on one GPU: a P106-090 natively, or a newer card through the build's
# compute_61 PTX (CUDA_FORCE_PTX_JIT=1; NSM=5 then gives the P106-090's split counts). Parts, in PARTS's order:
#   check  test-backend-ops FLASH_ATTN_EXT against the CPU backend under GGML_CUDA_FATTN_SPARSE_PASCAL=1 (the path below
#          Turing, from 1.5 x n_kv_max cells), =2 with _NQ=128 (any size, and the 64- and 67-token cases too) and =0
#          (the stock kernels on the same cases)
#   fault  =2 with each of the three _FAULT codes (testing the tests): every run must fail cases
#   fasp   bench/fattn_sparse's fasp_check --gpu: the kernel itself against FP64 and ggml's CPU kernel on 20 cases, with
#          the host model beside it (lists built on the host, as the compaction would)
#   perf   test-backend-ops perf on Flash-Next's QSA call (head 256, 2 x 12 heads, 2,051 cells, cache layout) at 1 and
#          4 tokens over 4k-128k cells, F16 and Q8_0 caches: =2 (the path forced at every size, to find where it starts
#          to pay: mode 1 takes a call from 1.5 x 2,051 cells per query token) against =0; perf.tsv pairs them
#   bench  fasp_check --bench: the kernel alone, us per call and K/V GB/s
# Env: BIN (bin/ of the build: test-backend-ops), FASP (fasp_check), OUT, DEV (CUDA0), PARTS (default "check fault fasp"),
#      NSM (unset: the device's own), FILTER (test-backend-ops -p for check and fault; default the n_kv_max cases)
# Output: OUT/console.txt (one verdict line per run), OUT/<run>.txt (each run's output), OUT/perf.tsv, OUT/done
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
DEV=${DEV:-CUDA0}
PARTS=${PARTS:-check fault fasp}
FILTER=${FILTER:-n_kv_max=[1-9]}
PERF_FILTER='hsk=256,hsv=256,nh=2,nr23=\[12,1\],kv=[0-9]+,nb=[14],.*permute=\[0,2,1,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=2051'
mkdir -p "$OUT"
rm -f "$OUT/done"
has() { [[ " $PARTS " == *" $1 "* ]]; }
NSM_ENV=()
[ -n "${NSM:-}" ] && NSM_ENV=(GGML_CUDA_FATTN_SPARSE_PASCAL_NSM="$NSM")
log() { echo "$*" | tee -a "$OUT/console.txt"; }
log "== qsa_gpu $(date -Is) dev $DEV parts '$PARTS' nsm ${NSM:-device} $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2> /dev/null)"

# tbo NAME want ENV...: test-backend-ops eval; want ok (all pass) or fail (some fail)
tbo() {
    local name=$1 want=$2
    shift 2
    env "$@" "${NSM_ENV[@]}" "$BIN/test-backend-ops" -o FLASH_ATTN_EXT -b "$DEV" -p "$FILTER" > "$OUT/$name.txt" 2>&1
    local rc=$?
    local passed fails
    passed=$(grep -E "tests passed" "$OUT/$name.txt" | tail -1 | sed 's/^ *//')
    fails=$(grep -c "FAIL" "$OUT/$name.txt")
    local verdict=WRONG
    { [ "$want" = ok ] && [ "$rc" = 0 ] && [ "$fails" = 0 ]; } && verdict=ok
    { [ "$want" = fail ] && [ "$fails" -gt 0 ]; } && verdict=ok
    log "$verdict  $name: rc=$rc, '$passed', $fails FAIL lines ($*)"
}

if has check; then
    tbo sparse1 ok GGML_CUDA_FATTN_SPARSE_PASCAL=1
    tbo sparse2 ok GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128
    tbo sparse0 ok GGML_CUDA_FATTN_SPARSE_PASCAL=0
fi

if has fault; then
    for f in 1 2 3; do
        tbo fault$f fail GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT=$f
    done
fi

if has fasp; then
    : "${FASP:?FASP=<fasp_check>}"
    "$FASP" --gpu --nsm "${NSM:-5}" > "$OUT/fasp_gpu.txt" 2>&1
    rc=$?
    log "$([ $rc = 0 ] && echo ok || echo WRONG)  fasp_gpu: rc=$rc, $(grep "cases pass" "$OUT/fasp_gpu.txt")"
fi

if has perf; then
    for v in 0 2; do
        env GGML_CUDA_FATTN_SPARSE_PASCAL=$v "${NSM_ENV[@]}" "$BIN/test-backend-ops" perf -o FLASH_ATTN_EXT -b "$DEV" \
            -p "$PERF_FILTER" > "$OUT/perf$v.txt" 2>&1
        log "perf$v: rc=$?, $(grep -c "us/run" "$OUT/perf$v.txt") cases"
    done
    # case -> us/run of each run: kv, tokens, K/V type, stock us, sparse us, speed-up
    {
        printf "kv\ttokens\ttype\tstock_us\tsparse_us\tspeedup\n"
        for v in 0 2; do
            sed -nE 's/.*FLASH_ATTN_EXT\(([^)]*)\).* ([0-9.]+) us\/run.*/\1 \2/p' "$OUT/perf$v.txt" |
                sed -nE "s/.*kv=([0-9]+),nb=([0-9]+),.*type_K=([a-z0-9_]+),.* ([0-9.]+)\$/$v \1 \2 \3 \4/p"
        done | awk '{ k = $2 "\t" $3 "\t" $4; if ($1 == 0) s[k] = $5; else p[k] = $5; if (!(k in seen)) { seen[k] = 1; o[n++] = k } }
                END { for (i = 0; i < n; i++) { k = o[i]; printf "%s\t%s\t%s\t%s\n", k, s[k], p[k], (s[k] && p[k]) ? sprintf("%.2f", s[k] / p[k]) : "" } }'
    } > "$OUT/perf.tsv"
    log "perf.tsv:"
    column -t "$OUT/perf.tsv" | tee -a "$OUT/console.txt"
fi

if has bench; then
    : "${FASP:?FASP=<fasp_check>}"
    "$FASP" --bench --nsm "${NSM:-5}" > "$OUT/fasp_bench.txt" 2>&1
    log "fasp_bench: rc=$?"
    sed -n '/^bench:/,$p' "$OUT/fasp_bench.txt" | tee -a "$OUT/console.txt"
fi

log "== end $(date -Is)"
touch "$OUT/done"
