#!/usr/bin/env bash
# C1111 (Q1844): the intermittent wrong result of
#   MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[8,1],nr=[4,1],per=[0,2,1,3])
# seen once (ERR 0.4995) in land-v5-batch3's MUL_MAT replay with --cpu-check 25 on an RTX 3090, two cases after its
# F16 sibling's live check. Each step runs test-backend-ops from a rig build (compute 6.1 PTX, run through the
# driver's JIT) in its own container, with the fork's matrix-vector kernels forced at a P106's shapes (ON).
#
# The case's neighbourhood (NB): every MUL_MAT case of type f16, bf16 or f32 with m and k 1056/1057 against 128/129,
# n = 1 (the permuted [0,2,1,3] cases and their k_v views, every bs and nr), 384 cases, in the suite's own order.
#
# Steps (the first argument; several may be given):
#   paths     per build: the neighbourhood replayed with --cpu-check CPU_CHECK and GGML_CUDA_GRAPH_DEBUG=1 (which
#             CUDA graphs are captured and replayed, node by node), and the BF16 case and its F16 sibling each alone
#             under Nsight Systems (NSYS) for the kernels they run
#   sanitize  per build, sanitizer tool (TOOLS) and mode (SMODES): compute-sanitizer --tool <tool> on the
#             neighbourhood; modes: cc25 (--replay REC --cpu-check 25, the failing run's way), cc1 (--cpu-check 1: every
#             case live first, then replayed on the same inputs), live (inputs drawn, the reference on this processor),
#             full25 (the whole MUL_MAT replay, FULL_FILTER, with --cpu-check 25: the failing run's command)
#   load      LOAD_SHARES on LOAD_GPU, live with ON, again and again until W/STOP exists (start it in the background)
#   repeat    N_REPEAT runs of the neighbourhood replayed with --cpu-check CPU_CHECK on GPU, one process each (REPEAT_ENV
#             added, e.g. GGML_CUDA_DISABLE_GRAPHS=1 with a build tag that says so)
#   full      N_FULL runs of the whole MUL_MAT replay (filter FULL_FILTER) with --cpu-check CPU_CHECK on GPU, one
#             process each: the failing run's command
#   siblings  per build: F16 cases each followed by their BF16 sibling, with GGML_CUDA_GRAPH_DEBUG=1, --cpu-check 25
#             and 1: does a BF16 case ever run the graph captured for its F16 sibling
#   again     one sibling set (AGAIN_SET) N_AGAIN times per build with --cpu-check AGAIN_CC, AGAIN_ENV added (the arm
#             AGAIN_TAG, e.g. GGML_CUDA_DISABLE_GRAPHS=1 as "nographs"): how often a BF16 case fails, with and without
#             CUDA graphs
#   kernels   the sibling set once per build under Nsight Systems with each graph's kernels traced, with and without
#             CUDA graphs: which kernels ran (a BF16 case that ran its F16 sibling's graph shows F16 kernels instead)
#   stop      touch W/STOP and stop the load containers
# Tables: W/paths.tsv, W/sanitize.tsv, W/repeat.tsv, W/full.tsv, W/load.tsv, W/siblings.tsv; logs in W/logs/ and
# per step.
#
# Env: W (outputs), BUILDS ("tag=bin-folder ..."), REC (the kept record's mul_mat folder), GPU (0), LOAD_GPU (1),
#      MOUNTS (folders mounted read-only at the same paths: the builds, the record), DOCKER_ARGS (labels, --cpu-shares),
#      NAME (container name prefix, c1111), IMG, ON, CPU_CHECK (25), NB, TOOLS, SMODES, N_REPEAT (200), N_FULL (20),
#      FULL_FILTER, LOAD_SHARES ("MUL_MAT FLASH_ATTN_EXT"), NSYS (the nsys binary; its folder is mounted), TBO_TIMEOUT
#      (seconds a test-backend-ops run may take, 1800), SAN_TIMEOUT (5400).
set -uo pipefail
: "${W:?}" "${BUILDS:?}" "${REC:?}"
GPU=${GPU:-0}
LOAD_GPU=${LOAD_GPU:-1}
NAME=${NAME:-c1111}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
ON=${ON:-"GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"}
CPU_CHECK=${CPU_CHECK:-25}
NB=${NB:-'^type_a=(f16|bf16|f32),type_b=f32,(m=105[67],n=1,k=12[89]|m=12[89],n=1,k=105[67]),'}
CASE_BF16='^type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'
CASE_F16='^type_a=f16,type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'
TOOLS=${TOOLS:-memcheck racecheck initcheck synccheck}
SMODES=${SMODES:-cc25 cc1}
N_REPEAT=${N_REPEAT:-200}
N_FULL=${N_FULL:-20}
FULL_FILTER=${FULL_FILTER:-'^(?!.*dup=1)'}
LOAD_SHARES=${LOAD_SHARES:-MUL_MAT FLASH_ATTN_EXT}
TBO_TIMEOUT=${TBO_TIMEOUT:-1800}
SAN_TIMEOUT=${SAN_TIMEOUT:-5400}
# compute-sanitizer reports a CUDA call's error return as an error by default; the backend expects one
# (cudaGraphExecUpdate refusing a changed graph, after which it instantiates the graph anew), so the second pass
# passes SAN_OPTS='--report-api-errors no' and SAN_TAG=-noapi, which leaves the memory checks alone in the count
SAN_OPTS=${SAN_OPTS:-}
SAN_TAG=${SAN_TAG:-}
PRINT_LIMIT=${PRINT_LIMIT:-200} # reports printed per sanitizer run (0: all of them)
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$W:$W")
[ -n "${NSYS:-}" ] && VOL+=(-v "$(dirname "$(dirname "$NSYS")"):$(dirname "$(dirname "$NSYS")"):ro")
mkdir -p "$W/logs" "$W/jit"
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" -e "CUDA_CACHE_PATH=$W/jit" -e CUDA_CACHE_MAXSIZE=4294967296)

strip() { sed 's/\x1b\[[0-9;]*m//g' "$@"; }

# run <gpu> <container name> <log> <timeout> <command prefix> <extra env> <bin> <test-backend-ops arguments>
run() {
    local gpu=$1 cname=$2 log=$3 to=$4 pre=$5 extra=$6 bin=$7
    shift 7
    "${DR[@]}" --name "$cname" --gpus "\"device=$gpu\"" "$IMG" \
        bash -c "env $ON $extra timeout $to $pre $bin/test-backend-ops test -b CUDA0 $* > $log 2>&1; echo rc=\$? >> $log" \
        > /dev/null 2>&1
}

# summary fields of a test-backend-ops log: rc, tests passed, the failing cases (;-joined), the cpu-check line
summ() {
    local f=$1 rc pass fails cc
    rc=$(grep -a -o 'rc=[0-9]*' "$f" | tail -n 1 | cut -d= -f2)
    pass=$(strip "$f" | grep -a -oE '[0-9]+/[0-9]+ tests passed' | tail -n 1 | cut -d' ' -f1)
    fails=$(strip "$f" | grep -a -E '^\[?.*: FAIL$' | grep -a -oE 'MUL_MAT\(type_a=[a-z0-9_]+[^)]*per=\[[0-9,]+\],k_v=[0-9]+' | sed 's/,type_b=f32//' | paste -sd';')
    cc=$(strip "$f" | grep -a -oE 'replay cpu-check: .*' | tail -n 1)
    printf '%s\t%s\t%s\t%s' "${rc:--}" "${pass:--}" "${fails:-none}" "${cc:--}"
}

builds() { for b in $BUILDS; do echo "${b%%=*} ${b#*=}"; done; }

step_paths() {
    mkdir -p "$W/paths"
    [ -f "$W/paths.tsv" ] || printf 'build\trun\trc\tpassed\tfailing\tcpu_check\tgraphs\n' > "$W/paths.tsv"
    builds | while read -r tag bin; do
        local f=$W/paths/$tag-graph.log
        run "$GPU" "$NAME-paths-$tag" "$f" "$TBO_TIMEOUT" "" "GGML_CUDA_GRAPH_DEBUG=1" "$bin" \
            -o MUL_MAT -p "'$NB'" --replay "$REC" --cpu-check "$CPU_CHECK"
        printf '%s\tgraph-debug\t%s\t%s\n' "$tag" "$(summ "$f")" \
            "$(strip "$f" | grep -a -c 'warmup complete') warm-ups complete, $(strip "$f" | grep -a -c 'reused') reused" >> "$W/paths.tsv"
        if [ -n "${NSYS:-}" ]; then
            for c in bf16 f16; do
                local cs=$CASE_BF16
                [ "$c" = f16 ] && cs=$CASE_F16
                f=$W/paths/$tag-nsys-$c.log
                run "$GPU" "$NAME-nsys-$tag-$c" "$f" "$TBO_TIMEOUT" \
                    "$NSYS profile -t cuda,cublas --sample=none --cpuctxsw=none --stats=true --force-overwrite true -o $W/paths/$tag-$c" \
                    "" "$bin" -o MUL_MAT -p "'$cs'" --replay "$REC"
                printf '%s\tnsys-%s\t%s\t-\n' "$tag" "$c" "$(summ "$f")" >> "$W/paths.tsv"
            done
        fi
    done
}

step_sanitize() {
    mkdir -p "$W/san"
    [ -f "$W/sanitize.tsv" ] || printf 'build\ttool\tmode\trc\tpassed\tfailing\tcpu_check\tsanitizer\tseconds\n' > "$W/sanitize.tsv"
    builds | while read -r tag bin; do
        for tool in $TOOLS; do
            for mode in $SMODES; do
                local args="-o MUL_MAT -p '$NB'" opt="" f t0
                case $mode in
                    cc25) args+=" --replay $REC --cpu-check 25" ;;
                    cc1)  args+=" --replay $REC --cpu-check 1" ;;
                    full25) args="-o MUL_MAT -p '$FULL_FILTER' --replay $REC --replay-allow-missing --cpu-check 25" ;;
                esac
                [ "$tool" = racecheck ] && opt="--racecheck-report all"
                f=$W/san/$tag-$tool-$mode$SAN_TAG.log
                t0=$(date +%s)
                run "$GPU" "$NAME-san-$tag-$tool-$mode" "$f" "$SAN_TIMEOUT" \
                    "/usr/local/cuda/bin/compute-sanitizer --tool $tool $opt $SAN_OPTS --print-limit $PRINT_LIMIT --error-exitcode 99" "" "$bin" "$args"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$tool$SAN_TAG" "$mode" "$(summ "$f")" \
                    "$(strip "$f" | grep -a -E '^========= (ERROR|RACECHECK) SUMMARY' | tail -n 1 | sed 's/^========= //')" \
                    "$(( $(date +%s) - t0 ))" >> "$W/sanitize.tsv"
                echo "$(date +%H:%M:%S) sanitize $tag $tool $mode: $(tail -n 1 "$W/sanitize.tsv" | cut -f4-9)"
            done
        done
    done
}

step_load() {
    [ -f "$W/load.tsv" ] || printf 'share\titeration\tstart\trc\tpassed\n' > "$W/load.tsv"
    local i=0 tag bin
    read -r tag bin < <(builds | head -n 1)
    while [ ! -e "$W/STOP" ]; do
        i=$((i + 1))
        for sh in $LOAD_SHARES; do
            (
                local f=$W/logs/load-$sh-$i.log s
                s=$(date +%H:%M:%S)
                run "$LOAD_GPU" "$NAME-load-$sh" "$f" "$TBO_TIMEOUT" "" "" "$bin" -o "$sh" -p "'$FULL_FILTER'"
                printf '%s\t%s\t%s\t%s\t%s\n' "$sh" "$i" "$s" "$(grep -a -o 'rc=[0-9]*' "$f" | tail -n 1 | cut -d= -f2)" \
                    "$(strip "$f" | grep -a -oE '[0-9]+/[0-9]+ tests passed' | tail -n 1 | cut -d' ' -f1)" >> "$W/load.tsv"
                rm -f "$f"   # a passing load log is not needed; the table keeps its counts
            ) &
        done
        wait
    done
}

step_repeat() {
    [ -f "$W/repeat.tsv" ] || printf 'build\trun\tstart\trc\tpassed\tfailing\tcpu_check\n' > "$W/repeat.tsv"
    builds | while read -r tag bin; do
        for i in $(seq 1 "$N_REPEAT"); do
            local f=$W/logs/repeat-$tag-$i.log s
            s=$(date +%H:%M:%S)
            run "$GPU" "$NAME-rep-$tag" "$f" "$TBO_TIMEOUT" "" "${REPEAT_ENV:-}" "$bin" -o MUL_MAT -p "'$NB'" --replay "$REC" --cpu-check "$CPU_CHECK"
            printf '%s\t%s\t%s\t%s\n' "$tag" "$i" "$s" "$(summ "$f")" >> "$W/repeat.tsv"
            # keep a failing run's log whole; a passing one only as its table row
            if [ "$(tail -n 1 "$W/repeat.tsv" | cut -f6)" = none ]; then rm -f "$f"; fi
        done
        echo "$(date +%H:%M:%S) repeat $tag: $(awk -F'\t' -v t="$tag" '$1==t {n++; if ($6=="none") ok++} END {print ok+0 " of " n+0 " without a failing case"}' "$W/repeat.tsv")"
    done
}

step_full() {
    [ -f "$W/full.tsv" ] || printf 'build\trun\tstart\tseconds\trc\tpassed\tfailing\tcpu_check\n' > "$W/full.tsv"
    builds | while read -r tag bin; do
        for i in $(seq 1 "$N_FULL"); do
            local f=$W/logs/full-$tag-$i.log s t0
            s=$(date +%H:%M:%S)
            t0=$(date +%s)
            run "$GPU" "$NAME-full-$tag" "$f" "$TBO_TIMEOUT" "" "" "$bin" -o MUL_MAT -p "'$FULL_FILTER'" --replay "$REC" \
                --replay-allow-missing --cpu-check "$CPU_CHECK"
            printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$i" "$s" "$(( $(date +%s) - t0 ))" "$(summ "$f")" >> "$W/full.tsv"
            if [ "$(tail -n 1 "$W/full.tsv" | cut -f7)" = none ]; then gzip -f "$f"; fi
        done
        echo "$(date +%H:%M:%S) full $tag: $(awk -F'\t' -v t="$tag" '$1==t {n++; if ($7=="none") ok++} END {print ok+0 " of " n+0 " without a failing case"}' "$W/full.tsv")"
    done
}

step_siblings() {
    # F16 cases each followed by their BF16 sibling (same shapes and strides, so the CUDA graph cache's property
    # check, which leaves the sources' types out, cannot tell them apart), with GGML_CUDA_GRAPH_DEBUG=1: the
    # failing case's pair, and the [0,2,1,3] cases of bs [1,1], [4,1] and [8,1] (m 1056/1057, k 128/129)
    mkdir -p "$W/siblings"
    [ -f "$W/siblings.tsv" ] || printf 'build\tset\tmode\tcases\trc\tpassed\tfailing\tcpu_check\twarmups_complete\twarmups_reset\n' > "$W/siblings.tsv"
    local pair='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'
    local sib='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    local sib4='^type_a=(f16|bf16),type_b=f32,m=105[67],n=1,k=12[89],bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    builds | while read -r tag bin; do
        for set in pair sib sib4; do
            for cc in 25 1; do
                local f=$W/siblings/$tag-$set-cc$cc.log flt
                flt=${!set}
                run "$GPU" "$NAME-sib-$tag-$set-$cc" "$f" "$TBO_TIMEOUT" "" "GGML_CUDA_GRAPH_DEBUG=1" "$bin" \
                    -o MUL_MAT -p "'$flt'" --replay "$REC" --cpu-check "$cc"
                printf '%s\t%s\tcc%s\t%s\t%s\t%s\t%s\n' "$tag" "$set" "$cc" \
                    "$(strip "$f" | grep -a -c -E 'MUL_MAT\(.*: (OK|FAIL)$')" "$(summ "$f")" \
                    "$(strip "$f" | grep -a -c 'warmup complete')" "$(strip "$f" | grep -a -c 'warmup reset')" >> "$W/siblings.tsv"
            done
        done
    done
}

step_again() {
    # one sibling set (AGAIN_SET: pair, sib or sib4) with --cpu-check AGAIN_CC, N_AGAIN runs per build, one process
    # each, with AGAIN_ENV added (e.g. GGML_CUDA_DISABLE_GRAPHS=1) and the graph log on; AGAIN_TAG names the arm
    local set=${AGAIN_SET:-sib} cc=${AGAIN_CC:-1} n=${N_AGAIN:-20} extra=${AGAIN_ENV:-} arm=${AGAIN_TAG:-graphs}
    local pair='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'
    local sib='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    local sib4='^type_a=(f16|bf16),type_b=f32,m=105[67],n=1,k=12[89],bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    local flt=${!set}
    mkdir -p "$W/again"
    [ -f "$W/again.tsv" ] || printf 'build\tarm\tset\tmode\trun\trc\tpassed\tfailing\tcpu_check\twarmups_complete\twarmups_reset\terrors\n' > "$W/again.tsv"
    builds | while read -r tag bin; do
        for i in $(seq 1 "$n"); do
            local f=$W/again/$tag-$arm-$set-cc$cc-$i.log
            run "$GPU" "$NAME-again-$tag" "$f" "$TBO_TIMEOUT" "" "GGML_CUDA_GRAPH_DEBUG=1 $extra" "$bin" \
                -o MUL_MAT -p "'$flt'" --replay "$REC" --cpu-check "$cc"
            printf '%s\t%s\t%s\tcc%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$arm" "$set" "$cc" "$i" "$(summ "$f")" \
                "$(strip "$f" | grep -a -c 'warmup complete')" "$(strip "$f" | grep -a -c 'warmup reset')" \
                "$(strip "$f" | grep -a -oE 'ERR = [0-9.]+' | sort -u | cut -d' ' -f3 | paste -sd';')" >> "$W/again.tsv"
            if [ "$(tail -n 1 "$W/again.tsv" | cut -f8)" = none ]; then rm -f "$f"; fi
        done
        echo "$(date +%H:%M:%S) again $tag $arm $set cc$cc: $(awk -F'\t' -v t="$tag" -v a="$arm" -v s="$set" '$1==t && $2==a && $3==s {n++; if ($8=="none") ok++} END {print ok+0 " of " n+0 " without a failing case"}' "$W/again.tsv")"
    done
}

step_kernels() {
    # the sibling set AGAIN_SET with --cpu-check AGAIN_CC once per build under Nsight Systems (NSYS) with each CUDA
    # graph's kernels traced, with CUDA graphs and with GGML_CUDA_DISABLE_GRAPHS=1: which kernels the cases ran
    local set=${AGAIN_SET:-sib} cc=${AGAIN_CC:-1}
    local sib='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    local pair='^type_a=(f16|bf16),type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'
    local sib4='^type_a=(f16|bf16),type_b=f32,m=105[67],n=1,k=12[89],bs=\[[148],1\],nr=\[[14],1\],per=\[0,2,1,3\],'
    local flt=${!set}
    : "${NSYS:?NSYS=<nsys binary>}"
    mkdir -p "$W/kernels"
    [ -f "$W/kernels_runs.tsv" ] || printf 'build\tarm\tset\tmode\trc\tpassed\tfailing\tcpu_check\n' > "$W/kernels_runs.tsv"
    builds | while read -r tag bin; do
        for arm in graphs nographs; do
            local extra="" f=$W/kernels/$tag-$arm-$set-cc$cc.log
            [ "$arm" = nographs ] && extra=GGML_CUDA_DISABLE_GRAPHS=1
            run "$GPU" "$NAME-kern-$tag-$arm" "$f" "$TBO_TIMEOUT" \
                "$NSYS profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --stats=true --force-overwrite true -o $W/kernels/$tag-$arm-$set-cc$cc" \
                "$extra" "$bin" -o MUL_MAT -p "'$flt'" --replay "$REC" --cpu-check "$cc"
            printf '%s\t%s\t%s\tcc%s\t%s\n' "$tag" "$arm" "$set" "$cc" "$(summ "$f")" >> "$W/kernels_runs.tsv"
        done
    done
}

step_stop() {
    touch "$W/STOP"
    for sh in $LOAD_SHARES; do docker stop "$NAME-load-$sh" > /dev/null 2>&1; done
}

for s in "$@"; do
    case $s in
        paths|sanitize|load|repeat|full|siblings|again|kernels|stop) "step_$s" ;;
        *) echo "unknown step $s" >&2; exit 2 ;;
    esac
done
