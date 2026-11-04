#!/usr/bin/env bash
# C372 (Q931): is a replayed verdict a live verdict? On one newer NVIDIA card through the rig build's compute_61 PTX
# (the fork's P106 kernel paths forced on with a P106's launch shapes, as each branch's gpu_checks.sh MODE=ptx runs
# them). Run inside nvidia/cuda:12.9.2-devel-ubuntu24.04 with one GPU.
#   same    the same inputs judged both ways: a live run that also records (--record on the card: its verdict and
#           each node's error go into the record), then the replay of that record under the same switches; every
#           case's verdict and every node's error (printed to 9 digits) must agree. Filters: the k-column products
#           (KCOL=2, every product on the fork's kernels), the Q2_0 and grid-type experts, the sparse attention
#   faults  the negative controls: the sparse attention's fault codes 2 and 3 and FAP's 2 and 3 corrupt the card's
#           result. Live + record under each fault, then the replay under the same fault: the same cases fail; the
#           replay of the same record without the fault: every case passes
#   sets    the rig sets recorded on the CPU alone (scripts/record_checks.sh), every run of their runs.tsv through
#           scripts/replay_checks.sh: each run as expected (the faults fail, the rest pass), the counts against the
#           live PTX runs of the same filters (fresh inputs) beside them
#   corrupt one byte of a copy of one case flipped: the replay fails that case on its md5
#   cross   a CPU-recorded set replayed by another build of the same source (BIN86, sm_86 code): the same verdicts
# Env: BIN (bin/ of the rig tarball), BIN86 (optional), W (scratch folder), SETS (the folder with the recorded suites),
#   REPO (this repository: scripts/replay_checks.sh), DEV (card: index or UUID, default 0), PARTS (default "same faults
#   corrupt sets cross", in this order), SUITES (the sets part: default "kshape ladder3-rest"), SAME (the same
#   part's filters: default "kcol-all mtok mtok-iq qsa-sparse2"), LIVE (the sets part's live runs beside the replays:
#   default "kcol-all mtok-iq"), MEMCHECK (0: the sets part leaves out the memcheck runs)
# Output: W/exact.txt (one line per check: PASS or FAIL and the counts), W/<part>/ (logs, replay logs)
set -uo pipefail
BIN=${BIN:?}; W=${W:?}; REPO=${REPO:?}
DEV=${DEV:-0}
PARTS=${PARTS:-same faults corrupt sets cross}
mkdir -p "$W"
R=$W/exact.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV CUDA_CACHE_PATH=$W/nvcache CUDA_CACHE_MAXSIZE=4294967296
PTX="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
     GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
     GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
T="$BIN/test-backend-ops -b CUDA0"
part() { [[ " $PARTS " == *" $1 "* ]]; }
wait_set() { # <set folder>: its manifest.tsv is written when its record is complete (up to 90 min)
    local i
    for i in $(seq 1 540); do [ -f "$1/manifest.tsv" ] && return 0; sleep 10; done
    return 1
}
res() { echo "$1 $2: $3" | tee -a "$R"; }
EXPERT_OPS=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs|q4_0|q5_0|iq4_nl|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s|iq1_m),(type_b=f32,)?m=[0-9]+,n=[1-8],'
IQ='iq(2_xxs|2_xs|2_s|3_xxs|3_s|1_m)'
QSA='n_kv_max=[1-9]'
FAP_OWN='hsk=128,hsv=128,nh=(4|8),nr23=\[8,1\],kv=[0-9]+,nb=1,'
echo "== $(date -Is) gpu_exact DEV=$DEV $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2> /dev/null) parts: $PARTS" | tee -a "$R"

# compare <record dir> <replay log>: the record's live lines against the replay's, case by case; prints the cases,
# those with the same verdict, those with the same errors to 9 digits, the live and the replayed FAIL counts
compare() {
    awk -F'\t' 'FNR == 1 { f++ }
        f == 1 { live[$1] = $2 "\t" $3 }
        f == 2 { rep[$1] = $3 "\t" $4 }
        END { n = sv = se = lf = rf = 0
              for (i in live) { n++; split(live[i], a, "\t"); split(rep[i], b, "\t")
                  if (a[1] == b[1]) sv++; if (live[i] == rep[i]) se++; if (a[1] == "FAIL") lf++; if (b[1] == "FAIL") rf++ }
              print n, sv, se, lf, rf }' \
        <(for m in "$1"/cases/*.meta; do awk -F'\t' -v i="$(basename "$m" .meta)" '$1 == "live" { print i "\t" $3 "\t" $4 }' "$m"; done) \
        "$2"
}
same_both() { # <tag> <switches> <-o> <-p> [expect: pass|fail]: live + record, then replay; verdicts and errors must agree
    local tag=$1 envs=$2 ops=$3 p=$4 expect=${5:-pass} d=$W/same/$1 line
    mkdir -p "$d"
    rm -rf "$d/rec"
    # shellcheck disable=SC2086
    env $PTX $envs $T -o "$ops" -p "$p" --record "$d/rec" > "$d/live.log" 2>&1
    # shellcheck disable=SC2086
    env $PTX $envs $T -o "$ops" -p "$p" --replay "$d/rec" --replay-log "$d/replay.tsv" > "$d/replay.log" 2>&1
    local n sv se lf rf v=FAIL
    read -r n sv se lf rf <<< "$(compare "$d/rec" "$d/replay.tsv")"
    [ "${n:-0}" -gt 0 ] && [ "$sv" = "$n" ] && [ "$se" = "$n" ] && v=PASS
    { [ "$expect" = fail ] && [ "${lf:-0}" -eq 0 ]; } && v=FAIL
    res "same/$tag" $v "${envs:-defaults}: $n cases, $sv with the same verdict, $se with the same errors to 9 digits; FAIL live $lf, replayed $rf"
}

if part same; then
    sw() { [[ " ${SAME:-kcol-all mtok mtok-iq qsa-sparse2} " == *" $1 "* ]]; }
    sw kcol-all && same_both kcol-all "GGML_CUDA_MMVQ_PASCAL_KCOL=2" MUL_MAT "$KT"
    sw mtok && same_both mtok "" $EXPERT_OPS 'type_a=q2_0|type=q2_0'
    sw mtok-iq && same_both mtok-iq "" $EXPERT_OPS "type_a=$IQ|type=$IQ"
    sw qsa-sparse2 && same_both qsa-sparse2 "GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128" FLASH_ATTN_EXT "$QSA"
fi

if part faults; then
    for f in "qsa-fault2:GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT=2:GGML_CUDA_FATTN_SPARSE_PASCAL=2:$QSA" \
             "qsa-fault3:GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT=3:GGML_CUDA_FATTN_SPARSE_PASCAL=2:$QSA" \
             "fap-fault2:GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_PASCAL_FAULT=2:GGML_CUDA_FATTN_PASCAL=2:$FAP_OWN" \
             "fap-fault3:GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_PASCAL_FAULT=3:GGML_CUDA_FATTN_PASCAL=2:$FAP_OWN"; do
        IFS=: read -r tag on off p <<< "$f"
        same_both "$tag" "$on" FLASH_ATTN_EXT "$p" fail
        d=$W/same/$tag
        # the same record replayed without the fault: every case passes
        # shellcheck disable=SC2086
        env $PTX $off $T -o FLASH_ATTN_EXT -p "$p" --replay "$d/rec" > "$d/replay-nofault.log" 2>&1
        rc=$?
        s=$(grep -a 'tests passed' "$d/replay-nofault.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')
        [ $rc = 0 ] && [ "$(grep -a -c FAIL "$d/replay-nofault.log")" = 0 ] && v=PASS || v=FAIL
        res "faults/$tag-off" $v "the fault's record replayed with $off: exit $rc, $s"
    done
fi

if part corrupt; then
    : "${SETS:?the corrupt part needs SETS}"
    d=$W/corrupt; rm -rf "$d"; mkdir -p "$d/cases"
    s=$SETS/kshape
    if wait_set "$s"; then
        i=$(awk -F'\t' '$2 ~ /^MUL_MAT\(type_a=q4_0,type_b=f32,m=16,/ { print $1; exit }' "$s/manifest.tsv")
        [ -n "$i" ] || i=$(head -n 1 "$s/manifest.tsv" | cut -f1)
        cp "$s/cases/$i".* "$d/cases/"
        name=$(awk -F'\t' '$1 == "name" { print $2 }' "$d/cases/$i.meta")
        b=$(od -An -tu1 -j 200 -N1 "$d/cases/$i.in" | tr -d ' ')
        printf "$(printf '\\%03o' $((255 - b)))" | dd of="$d/cases/$i.in" bs=1 seek=200 conv=notrunc 2> /dev/null
        # shellcheck disable=SC2086
        env $PTX $T -o "$name" --replay "$d" > "$d/replay.log" 2>&1
        rc=$?
        mm=$(grep -a -c 'record mismatch: .*md5' "$d/replay.log")
        [ $rc = 1 ] && [ "$mm" = 1 ] && v=PASS || v=FAIL
        res corrupt $v "case $i ($name), one byte of its .in flipped: exit $rc, $mm md5 mismatch line"
    else
        res corrupt FAIL "no complete record of $s"
    fi
fi

if part sets; then
    : "${SETS:?the sets part needs SETS}"
    mkdir -p "$W/sets"
    for s in ${SUITES:-kshape ladder3-rest}; do
        s=$SETS/$s; [ -f "$s/runs.tsv" ] || continue
        wait_set "$s" || { res "sets/$(basename "$s")" FAIL "no complete record"; continue; }
        name=$(basename "$s")
        # MEMCHECK=0 leaves out the runs under compute-sanitizer
        runs=$(awk -F'\t' -v m="${MEMCHECK:-1}" 'm == 1 || $6 != "memcheck" { print $1 }' "$s/runs.tsv" | tr '\n' ' ')
        RUNS="$runs" ENV_ALL="$PTX" SET=$s BIN=$BIN OUT=$W/sets/$name DEV="$DEV" bash "$REPO/scripts/replay_checks.sh" > "$W/sets/$name.out" 2>&1
        ok=$(grep -a -c ' expect [a-z]* *ok$' "$W/sets/$name/console.txt")
        wrong=$(grep -a -c ' expect [a-z]* *WRONG$' "$W/sets/$name/console.txt")
        [ "$wrong" = 0 ] && [ "$ok" -gt 0 ] && v=PASS || v=FAIL
        res "sets/$name" $v "$ok runs as expected, $wrong WRONG (console: sets/$name/console.txt)"
    done
    # the live PTX runs of two filters beside them (fresh inputs): the same pass counts
    while IFS=$'\t' read -r tag envs ops p set; do
        [ "$envs" = - ] && envs=""
        [[ " ${LIVE:-kcol-all mtok-iq} " == *" $tag "* ]] || continue
        # shellcheck disable=SC2086
        env $PTX $envs $T -o "$ops" -p "$p" > "$W/sets/live-$tag.log" 2>&1
        live=$(grep -a 'tests passed' "$W/sets/live-$tag.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')
        rep=$(grep -a 'tests passed' "$W/sets/$set/$tag.log" 2> /dev/null | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')
        [ -n "$live" ] && [ "${live%% *}" = "${rep%% *}" ] && v=PASS || v=FAIL
        res "sets/live-$tag" $v "live (fresh inputs): $live; replay of the $set set: $rep"
    done < <(printf '%s\t%s\t%s\t%s\t%s\n' kcol-all GGML_CUDA_MMVQ_PASCAL_KCOL=2 MUL_MAT "$KT" kshape \
                 mtok-iq - $EXPERT_OPS "type_a=$IQ|type=$IQ" kshape)
fi

if part cross && [ -n "${BIN86:-}" ]; then
    : "${SETS:?the cross part needs SETS}"
    while IFS=$'\t' read -r tag envs ops p; do
        [ "$envs" = - ] && envs=""
        # sm_86 code: the fork's kernels forced on with a P106's shapes, no PTX
        # shellcheck disable=SC2086
        env $PTX $envs "$BIN86/test-backend-ops" -b CUDA0 -o "$ops" -p "$p" --replay "$SETS/kshape" > "$W/cross-$tag.log" 2>&1
        rc=$?
        s=$(grep -a 'tests passed' "$W/cross-$tag.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//')
        [ $rc = 0 ] && [ "$(grep -a -c FAIL "$W/cross-$tag.log")" = 0 ] && v=PASS || v=FAIL
        res "cross/$tag" $v "the CPU-recorded kshape set replayed by the sm_86 build: exit $rc, $s"
    done < <(printf '%s\t%s\t%s\t%s\n' kcol-all GGML_CUDA_MMVQ_PASCAL_KCOL=2 MUL_MAT "$KT" \
                 mtok - $EXPERT_OPS 'type_a=q2_0|type=q2_0')
fi
echo "done $(date -Is)" >> "$R"
