#!/bin/bash
# R232 (a), Q1706: pull request 9's attention switches at Ornith-1.5-122B's shape on one P106-090.
#
# Arms (each a fresh process, so each switch is read once at its first use):
#   base   nothing set
#   min1   GGML_CUDA_FATTN_TILE_PASCAL_MIN=1    the Pascal tile kernel from 1 query token (default 3 for Q8_0, 9 for F16)
#   dec4   GGML_CUDA_FATTN_DEC_PASCAL=4         the Q8_0 decode kernel at head 256 for up to 4 tokens
#   both   the two together                     (what Ornith's agent check with -ctk q8_0 -ctv q8_0 uses)
#
# Steps (STEPS, default "test perf"):
#   test   correctness, one run per arm: the suite's FLASH_ATTN_EXT cases at head 256 (-p 'hsk=256,hsv=256': trimmed to
#          the model's head size, so the rig's 2-core CPU computes only those references) and the Ornith cases of
#          $CASES (make_cases.py), each against the CPU
#   perf   kernel times, test-backend-ops perf on the Ornith cases of $PCASES, REPEATS rounds in ABBA order
#          (base min1 dec4 both, then both dec4 min1 base, ...)
#
# Environment: BIN (the build's bin/), OUT (results), CASES (the correctness case file), PCASES (the timing case file,
#   default CASES; perf repeats each case for ~32 GB of traffic, so a 512-token prompt at depth 4,096 takes ~70 s a
#   run and is left to the correctness file), DEV (CUDA device, default 0),
#   TMO (seconds per run, default 3600). Writes OUT/<run>.log, .rc, .smi (clocks before and after), progress.log,
#   and OUT/DONE at the end.
set -u
: "${BIN:?set BIN to the bin directory of the build}" "${OUT:?set OUT}" "${CASES:?set CASES to the case file}"
DEV=${DEV:-0}
STEPS=${STEPS:-"test perf"}
REPEATS=${REPEATS:-2}
TMO=${TMO:-3600}
PCASES=${PCASES:-$CASES}
ARMS=${ARMS:-"base min1 dec4 both"}
mkdir -p "$OUT"
echo $$ > "$OUT/pid"
rm -f "$OUT/DONE"

arm_env() {
    case $1 in
        base) echo "" ;;
        min1) echo "GGML_CUDA_FATTN_TILE_PASCAL_MIN=1" ;;
        dec4) echo "GGML_CUDA_FATTN_DEC_PASCAL=4" ;;
        both) echo "GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

smi() {
    nvidia-smi -i "$DEV" --query-gpu=index,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,memory.used \
        --format=csv,noheader
}

run() {   # run <name> <arm> <test-backend-ops args...>
    local name=$1 arm=$2
    shift 2
    local e
    e=$(arm_env "$arm")
    echo "$(date +%T) start $name ($arm: ${e:-nothing set})" >> "$OUT/progress.log"
    { echo "before $(date +%T) $(smi)"; } > "$OUT/$name.smi"
    local t0=$(date +%s)
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $e timeout "$TMO" "$BIN/test-backend-ops" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo $rc > "$OUT/$name.rc"
    { echo "after $(date +%T) $(smi)"; } >> "$OUT/$name.smi"
    echo "$(date +%T) end $name rc $rc ($(( $(date +%s) - t0 )) s)" >> "$OUT/progress.log"
}

echo "$(date +%T) R232a start: BIN=$BIN DEV=$DEV STEPS=$STEPS REPEATS=$REPEATS ARMS=$ARMS" >> "$OUT/progress.log"
md5sum "$BIN/test-backend-ops" "$CASES" "$PCASES" > "$OUT/md5.txt"

for step in $STEPS; do
    case $step in
        test)
            for arm in $ARMS; do
                run "test-suite256-$arm" "$arm" test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256'
                run "test-ornith-$arm" "$arm" test -b CUDA0 -o FLASH_ATTN_EXT --test-file "$CASES"
            done
            ;;
        perf)
            fwd="$ARMS"
            rev=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')
            for r in $(seq 1 "$REPEATS"); do
                if [ $((r % 2)) -eq 1 ]; then order=$fwd; else order=$rev; fi
                for arm in $order; do
                    run "perf-ornith-$arm-r$r" "$arm" perf -b CUDA0 -o FLASH_ATTN_EXT --test-file "$PCASES"
                done
            done
            ;;
        *) echo "unknown step $step" >> "$OUT/progress.log" ;;
    esac
done

echo "$(date +%T) R232a end" >> "$OUT/progress.log"
date +%T > "$OUT/DONE"
