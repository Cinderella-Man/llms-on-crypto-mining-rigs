#!/bin/bash
# C1037, Q1770: pull request 9's two prompt-kernel switches for a Q8_0 cache, GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 and
# GGML_CUDA_FATTN_TILE_PASCAL_KDP=1, at Ornith-1.5-122B's attention shape on one P106-090.
# R232a's driver (docs/data/2026-11-24_r232a_ornith_q8_kernels/run/r232a.sh) with its arms replaced:
#
# Arms (each a fresh process, so each switch is read once at its first use):
#   base    nothing set: the F16 cache as Ornith's v5 preset ran it before 2026-11-24 (timed on F16 cases only)
#   q8      GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4: the v5 preset's Q8_0-cache switches
#   lds     q8 + GGML_CUDA_FATTN_TILE_PASCAL_LDS=1   Q8_0 loads decoded as they are stored to shared memory
#   kdp     q8 + GGML_CUDA_FATTN_TILE_PASCAL_KDP=1   at head 256 the Q8_0 keys scored in int8 with dp4a
#   ldskdp  q8 + both                                (at head 256 KDP's kernel already decodes at store: the same
#                                                     kernel as kdp, so the pair is a same-kernel floor)
#
# Steps (STEPS, default "test perf"):
#   test   correctness, one run per arm of TEST_ARMS: the suite's FLASH_ATTN_EXT cases at head 256 (-p 'hsk=256,hsv=256')
#          and the Ornith cases of $CASES, each against the CPU
#   perf   kernel times, test-backend-ops perf, REPEATS rounds in ABBA order over ARMS; the base arm reads $PCASES_F16,
#          every other arm $PCASES_Q8
#
# Environment: BIN (the build's bin/), OUT (results), CASES (correctness case file), PCASES_Q8 and PCASES_F16 (timing
#   case files), DEV (CUDA device, default 0), TMO (seconds per run, default 3600). Writes OUT/<run>.log, .rc, .smi
#   (clocks before and after), progress.log, md5.txt and OUT/DONE at the end.
set -u
: "${BIN:?set BIN to the bin directory of the build}" "${OUT:?set OUT}" "${CASES:?set CASES to the case file}"
: "${PCASES_Q8:?set PCASES_Q8}" "${PCASES_F16:?set PCASES_F16}"
DEV=${DEV:-0}
STEPS=${STEPS:-"test perf"}
REPEATS=${REPEATS:-2}
TMO=${TMO:-3600}
ARMS=${ARMS:-"base q8 lds kdp ldskdp"}
TEST_ARMS=${TEST_ARMS:-"q8 lds kdp ldskdp"}
mkdir -p "$OUT"
echo $$ > "$OUT/pid"
rm -f "$OUT/DONE"

Q8="GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
arm_env() {
    case $1 in
        base)   echo "" ;;
        q8)     echo "$Q8" ;;
        lds)    echo "$Q8 GGML_CUDA_FATTN_TILE_PASCAL_LDS=1" ;;
        kdp)    echo "$Q8 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1" ;;
        ldskdp) echo "$Q8 GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1" ;;
        *) echo "unknown arm $1" >&2; exit 2 ;;
    esac
}

arm_pcases() {
    if [ "$1" = base ]; then echo "$PCASES_F16"; else echo "$PCASES_Q8"; fi
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

echo "$(date +%T) C1037 start: BIN=$BIN DEV=$DEV STEPS=$STEPS REPEATS=$REPEATS ARMS=$ARMS TEST_ARMS=$TEST_ARMS" >> "$OUT/progress.log"
md5sum "$BIN/test-backend-ops" "$CASES" "$PCASES_Q8" "$PCASES_F16" > "$OUT/md5.txt"

for step in $STEPS; do
    case $step in
        test)
            for arm in $TEST_ARMS; do
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
                    run "perf-ornith-$arm-r$r" "$arm" perf -b CUDA0 -o FLASH_ATTN_EXT --test-file "$(arm_pcases "$arm")"
                done
            done
            ;;
        *) echo "unknown step $step" >> "$OUT/progress.log" ;;
    esac
done

echo "$(date +%T) C1037 end" >> "$OUT/progress.log"
date +%T > "$OUT/DONE"
