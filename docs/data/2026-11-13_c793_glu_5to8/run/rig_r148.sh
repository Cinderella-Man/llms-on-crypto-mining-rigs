#!/usr/bin/env bash
# rig_r148.sh: R148 (Q1454), C793 (Q1388) on a P106 rig. GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 (fork branch
# pascal-p106-v4-b12246-c793 or a build merged from it) takes Qwen3.8-Flash-Next's shared-expert pairs with an IQ4_XS,
# Q4_K or Q5_K product at 5-8 tokens as one launch; at 1 they run as two read-once products (C729's
# GGML_CUDA_MMVQ_PASCAL_RO8=1) and a SWIGLU. Checked natively and timed. Every arm has the preset's k-column switches, the
# read-once kernels (_DMT=1 _DMT_R=1 _Q5K_MT=1) and _RO8=1. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# C757's rig_c757.sh (docs/data/2026-11-11_c757_shexp_fused) with this row's arms. Steps (STEPS, default
# "replay bitid perf table"):
#   replay  the recorded set glumt's C793 runs (RUNS, default the glumt8-* runs: on, without the prefetch, one warp pair a
#           block, without the 2-4-token read-once kernels, without _RO8, the value 1, off, and the fault) through
#           scripts/replay_checks.sh on card DEV; passing: every run ends "ok" (the fault run: its cases fail, as expected)
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid glu: every fused 2-8-token launch against the two products and the
#           SWIGLU run apart (at the same tokens and token by token), bit for bit, at GLU_MT=3, =4, =3 with _NP=1, =1 and
#           off; then the fault (_FAULT=1), the harness's own control: comparisons differ, exit 1
#   perf    test-backend-ops perf -o MUL_MAT_GLU at the shared expert's shape (2560 -> 640), every pair of PERF_TYPES (default
#           Q2_0, Q3_K, Q4_K, Q5_K, IQ4_XS), PERF_N tokens (default 5-8), the arms of PERF_ARMS (default "off on")
#           alternated over PASSES (default 2) passes; the Q2_0 and Q3_K pairs run the same launch in both arms (the
#           spread)
#   table   perf/table.tsv: per pass, pair and token count off's and on's us/run; and per pass and token count the est.
#           ms a verify pass saves: the sum over the file's pairs of its layers times (off - on) (the 22 layers of the
#           11 pairs with an IQ4_XS, Q4_K or Q5_K product; Q2_0 and Q3_K pairs as the control)
# Env: BIN (bin/ of the tarball), OUT, DEV (card, default 0; CUDA_DEVICE_ORDER=PCI_BUS_ID); replay: SET (the glumt
#   folder), REPLAY (scripts/replay_checks.sh); bitid: BITID (the multitoken_bitid binary built against the same tree);
#   PASSES, PERF_ARMS ("off on nopf np1" for all), PERF_N (a regex digit class), PERF_TYPES
# Usage: BIN=... OUT=... SET=... REPLAY=... BITID=... setsid -f nohup bash rig_r148.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-replay bitid perf table}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
PERF_ARMS=${PERF_ARMS:-off on}
PERF_N=${PERF_N:-[5-8]}
PERF_TYPES=${PERF_TYPES:-q2_0|q3_K|q4_K|q5_K|iq4_xs}
RUNS=${RUNS:-glumt8-on glumt8-nopf glumt8-np1 glumt8-ro8only glumt8-ro8-off glumt8-m1 glumt8-off glumt8-fault}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"            # the Flash-Next preset's k-column switches
RO="${P}_DMT=1 ${P}_DMT_R=1 ${P}_Q5K_MT=1 ${P}_RO8=1"  # the read-once kernels the launch repeats, 2-4 and 5-8 tokens
M="${P}_GLU_MT"
arm_env() { # <arm>: the switches of a perf arm
    case $1 in
        off)  echo "$KP $RO $M=1" ;;
        on)   echo "$KP $RO $M=3" ;;
        nopf) echo "$KP $RO $M=4" ;;
        np1)  echo "$KP $RO $M=3 ${M}_NP=1" ;;
        *)    return 1 ;;
    esac
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step replay; then
    : "${SET:?}" "${REPLAY:?}"
    log "replay start ($RUNS)"
    SET="$SET" BIN="$BIN" OUT="$OUT/replay" DEV="$DEV" RUNS="$RUNS" bash "$REPLAY" > "$OUT/replay-console.txt" 2>&1
    log "replay exit $?: $(tail -n 1 "$OUT/replay/console.txt" 2>/dev/null)"
fi

if step bitid; then
    : "${BITID:?}"
    mkdir -p "$OUT/bitid"
    for a in "on:$KP $RO $M=3" "nopf:$KP $RO $M=4" "np1:$KP $RO $M=3 ${M}_NP=1" "m1:$KP $RO $M=1" "off:$KP $RO" \
             "fault:$KP $RO $M=3 ${M}_FAULT=1"; do
        n=${a%%:*}
        log "bitid $n start (${a#*:})"
        # shellcheck disable=SC2086
        env CUDA_VISIBLE_DEVICES="$DEV" ${a#*:} "$BITID" glu > "$OUT/bitid/$n.txt" 2>&1
        log "bitid $n exit $?: $(tail -n 1 "$OUT/bitid/$n.txt")"
    done
fi

if step perf; then
    mkdir -p "$OUT/perf"
    PP="type_gate=($PERF_TYPES),type_up=($PERF_TYPES),m=640,n=$PERF_N,k=2560,"
    for pass in $(seq 1 "$PASSES"); do
        for arm in $PERF_ARMS; do
            e=$(arm_env "$arm") || { log "unknown arm $arm"; continue; }
            quiet
            log "perf $arm pass $pass start ($e)"
            # shellcheck disable=SC2086
            env CUDA_VISIBLE_DEVICES="$DEV" GGML_TEST_RAW_QUANT=1 $e "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT_GLU -p "$PP" \
                > "$OUT/perf/$arm-$pass.txt" 2>&1
            log "perf $arm pass $pass exit $?: $(grep -c 'us/run' "$OUT/perf/$arm-$pass.txt") lines"
        done
    done
    # one line per arm, pass, type pair and token count: arm, pass, gate, up, m, k, n, us/run
    for f in "$OUT"/perf/*-*.txt; do
        b=$(basename "$f" .txt)
        grep 'us/run' "$f" | sed -E "s/^ *MUL_MAT_GLU\(type_gate=([^,]+),type_up=([^,]+),m=([0-9]+),n=([0-9]+),k=([0-9]+),.*\): *[0-9]+ runs - +([0-9.]+) us\/run.*/${b%-*}\t${b##*-}\t\1\t\2\t\3\t\5\t\4\t\6/"
    done | sort > "$OUT/perf/us_run.tsv"
fi

if step table; then
    # the file's (Swift GSQ-RCO Q2_0) shared-expert pairs that the launch takes, by layers (C757's shapes.py)
    awk -F'\t' 'BEGIN { OFS = "\t"
            split("q2_0+q3_K:8 q3_K+q3_K:6 q3_K+q2_0:3 q2_0+q4_K:3 q2_0+iq4_xs:3 iq4_xs+q3_K:3 q4_K+q4_K:2 q4_K+q3_K:2 " \
                  "q4_K+q2_0:2 q3_K+iq4_xs:2 iq4_xs+q2_0:2 q5_K+q3_K:1 q3_K+q5_K:1 iq4_xs+q4_K:1", a, " ")
            for (i in a) { split(a[i], b, ":"); L[b[1]] = b[2] } }
        $1 == "off" || $1 == "on" { us[$1, $2, $3 "+" $4, $7] = $8; seen[$2, $3 "+" $4, $7] = 1 }
        END {
            print "pass", "pair", "layers", "tokens", "off_us", "on_us", "off_minus_on_us", "new_pair"
            for (k in seen) {
                split(k, s, SUBSEP)
                if (!((("off", s[1], s[2], s[3]) in us) && (("on", s[1], s[2], s[3]) in us))) continue
                o = us["off", s[1], s[2], s[3]]; n = us["on", s[1], s[2], s[3]]
                nw = s[2] ~ /(q4_K|q5_K|iq4_xs)/
                print s[1], s[2], (s[2] in L ? L[s[2]] : 0), s[3], o, n, sprintf("%.2f", o - n), nw
                if (s[2] in L) { ms[s[1], s[3], nw] += L[s[2]] * (o - n) / 1000; ps[s[1] SUBSEP s[3]] = 1 }
            }
            for (k in ps) {
                split(k, s, SUBSEP)
                printf "est. ms a verify pass, pass %s, %s tokens: %.3f saved by the 22 new-pair layers; %.3f by the 17 Q2_0/Q3_K layers (the same launch in both arms: the spread)\n", \
                    s[1], s[2], ms[s[1], s[2], 1], ms[s[1], s[2], 0] > "/dev/stderr"
            }
        }' "$OUT/perf/us_run.tsv" 2> "$OUT/perf/est_ms.txt" | sort -t$'\t' -k1,1n -k2,2 -k4,4n > "$OUT/perf/table.tsv"
    sort -o "$OUT/perf/est_ms.txt" "$OUT/perf/est_ms.txt"
    log "table: $(wc -l < "$OUT/perf/table.tsv") lines; $(tr '\n' ' ' < "$OUT/perf/est_ms.txt")"
fi

quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
