#!/usr/bin/env bash
# Records test-backend-ops checks for replay on a mining rig (C372, Q931). test-backend-ops builds each case's inputs,
# quantizes them and computes the reference on the CPU before it runs the GPU; on the rig's 2-core Celeron that CPU
# work is nearly all of a check's time (hours for the fork's kernel suites). Here a build machine's CPU does it once:
# each case's tensors and the CPU backend's results go into a folder, and scripts/replay_checks.sh on the rig restores
# them, runs only the GPU side and compares with the same thresholds and output lines (test-backend-ops --record and
# --replay, the fork's replay branch).
#
# The record is made with the rig build's own test-backend-ops (scripts/build_llamacpp_docker.sh TESTS=1: CPU code
# without AVX, SSE4.2 at most), so the references are what the rig's CPU itself would compute, bit for bit; it runs in
# a container without GPUs, so only the CPU backend is present.
#
# Steps (STEPS, default "record tar"):
#   build   the rig tarball of BRANCH with test-backend-ops into WORK (scripts/build_llamacpp_docker.sh, TESTS=1)
#   record  every filter of SUITE into OUT/SUITE/ (test-backend-ops -b CPU --record), the suite's runs into
#           OUT/SUITE/runs.tsv (what replay_checks.sh runs), the cases into OUT/SUITE/manifest.tsv
#   tar     OUT/SUITE.tar (not compressed: the blobs are random data) and OUT/SUITE.tar.md5
# Suites (SUITE; each filter is the union of the cases its runs select, every run's switches replay one record):
#   kshape        the k-column products, the large-activation cases, the Q2_0 experts and the grid types' experts
#                 (docs/data/2026-11-04_kshape/run/gpu_checks.sh, its 18 test-backend-ops runs)
#   ladder3-rest  the grid types and IQ1_M in the k-column and multi-token kernels (docs/data/2026-11-04_iq_kcol),
#                 C2's remaining one-token runs (docs/data/2026-11-04_iq2xs_kernel), the sparse decode attention with
#                 its three faults (docs/data/2026-11-04_qsa_sparse_attn), and both scripts' memcheck cases
#   fap           FLASH_ATTN_EXT, every case, FAP on and its faults (docs/data/2026-11-04_ladder3_merge's fap part)
#   flashnext-moe the experts at Flash-Next's shapes (512 experts, 10 used; MUL_MAT_ID, the fused gate/up, the shared expert), every type
#   n2p           N2P, N3P, N2P-128 and N3P-128 at every op and token count, the regression of Q2_0, Q6_K and the grid
#                 types, GET_ROWS of every type (docs/data/2026-11-04_n2p_flash/run/gpu_checks.sh, its nine native runs).
#                 Its runs-ptx.tsv is the run list for a newer card through the compute_61 PTX: 1-8 tokens, no MMQ
# Env: OUT (folder for the records), SUITE, BIN (bin/ of an unpacked rig tarball; the build step sets it), WORK and
#   REPO and BRANCH (build step), JOBS (test-backend-ops -j workers, default 4), SAMPLE (record about one case in N),
#   MAX_CASE_MB, BUDGET_GB (caps: cases over them are left out and replay reports them), DRY=1 (sizes only),
#   EXTRA_P (a -p regex every filter must also match, e.g. a "model shapes only" set), LINK_FROM (a complete set whose
#   cases this suite selects are hardlinked in instead of recorded again), IMAGE, DOCKER_ARGS
# Usage: OUT=<dir> SUITE=kshape BIN=<rig tarball>/bin bash scripts/record_checks.sh
set -uo pipefail
OUT=${OUT:?OUT=<folder for the records>}
SUITE=${SUITE:?SUITE=kshape|ladder3-rest|fap|flashnext-moe|n2p}
STEPS=${STEPS:-record tar}
JOBS=${JOBS:-4}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
S=$OUT/$SUITE
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/$SUITE.log"; }

# filters: "<-o ops><TAB><-p params>" per line; runs: "<run><TAB><switches><TAB><-o><TAB><-p><TAB><expect><TAB><wrap>"
# and, for a run with cases that are known to fail, "<TAB><known>" (expect: pass = every case passes, fail = some case
# fails, the run being a fault; wrap: memcheck = under compute-sanitizer; known: a regex on the case string of cases
# that may fail until a named fix: replay_checks.sh counts their FAIL lines apart). The switches and filters are those
# of the native runs of each branch's gpu_checks.sh.
EXPERT_OPS=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
P=GGML_CUDA_MMVQ_PASCAL
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs|q4_0|q5_0|iq4_nl|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s|iq1_m),(type_b=f32,)?m=[0-9]+,n=[1-8],'
IQ='iq(2_xxs|2_xs|2_s|3_xxs|3_s|1_m)'
IQL=iq2_xxs,iq2_xs,iq3_s,iq2_s,iq3_xxs,iq1_m
QSA='n_kv_max=[1-9]'
FAP_OWN='hsk=128,hsv=128,nh=(4|8),nr23=\[8,1\],kv=[0-9]+,nb=1,'
FN_MOE='n_mats=512,n_used=10,|^type=[^,]+,n=[0-9]+$'
N4='(n2p|n3p|n2p_128|n3p_128)'
HW='(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s)'
# Q1181: the one-token expert product of N2P-128 and N3P-128 at 2,561 rows fails on some runs, on the fork's kernels
# and on llama.cpp's (no model has that row count)
N2P_KNOWN='MUL_MAT_ID\(type_a=n[23]p_128,type_b=f32,n_mats=32,n_used=10,b=0,m=2561,n=1,k=640,'
suite_filters() {
    case $1 in
        kshape)
            printf '%s\t%s\n' MUL_MAT "$KT" MUL_MAT 'big=' $EXPERT_OPS 'type_a=q2_0|type=q2_0' $EXPERT_OPS "type_a=$IQ|type=$IQ" ;;
        ladder3-rest)
            printf '%s\t%s\n' "MUL_MAT,$EXPERT_OPS" "=$IQ," FLASH_ATTN_EXT "$QSA" ;;
        fap)
            printf '%s\t%s\n' FLASH_ATTN_EXT '.' ;;
        flashnext-moe)
            printf '%s\t%s\n' $EXPERT_OPS "$FN_MOE" ;;
        n2p)
            printf '%s\t%s\n' "MUL_MAT,$EXPERT_OPS,GET_ROWS" "type(_a)?=$N4,.*n=[0-9]+,|type=$N4," \
                "MUL_MAT,$EXPERT_OPS" "type(_a)?=$HW,.*n=[0-9]+,|type=$HW," GET_ROWS '.' ;;
        *) return 1 ;;
    esac
}
suite_runs() {
    local K2="${P}_KCOL=2" r
    case $1 in
        kshape)
            for r in "kcol:" "kcol-table0:${P}_KCOL_TABLE=0" "kcol-all:$K2" "kcol-old:$K2 ${P}_KCOL_SMEM=49152" \
                     "kcol-s32k:$K2 ${P}_KCOL_SMEM=32768" "kcol-r1:$K2 ${P}_KCOL_R=1" "kcol-r2:$K2 ${P}_KCOL_R=2" \
                     "kcol-itt1:$K2 ${P}_KCOL_ITT=1" "kcol-off:${P}_KCOL=0"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$KT"
            done
            printf '%s\t%s\t%s\t%s\tpass\t\n' stock "" MUL_MAT 'big=' \
                stock-skip "${P}_KCOL=0 ${P}_SKIP=q4_0,q4_1,q5_0" MUL_MAT 'big='
            for r in "mtok:" "mtok-direct:${P}_MTOK=1,0,0,0,0" "mtok-gdir:${P}_MTOK=1,0,0,1" "mtok-cap1:${P}_MTOK=1,1" \
                     "mtok-rb1:${P}_MTOK=1,0,1" "mtok-off:${P}_MTOK=0"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" $EXPERT_OPS 'type_a=q2_0|type=q2_0'
            done
            printf '%s\t%s\t%s\t%s\tpass\t\n' mtok-iq "" $EXPERT_OPS "type_a=$IQ|type=$IQ" ;;
        ladder3-rest)
            # C163 (docs/data/2026-11-04_iq_kcol/run/gpu_checks.sh): one run per switch setting over the six types,
            # every shape (the native runs left out the large ones only for the Celeron's sake)
            printf '%s\t%s\t%s\t%s\tpass\t\n' \
                iqkcol-on "" "MUL_MAT,$EXPERT_OPS" "=$IQ," \
                iqkcol-kcol_r2 "${P}_KCOL_R=2" MUL_MAT "=$IQ," \
                iqkcol-cap1 "${P}_MTOK=1,1" $EXPERT_OPS "=$IQ," \
                iqkcol-rb1 "${P}_MTOK=1,0,1" $EXPERT_OPS "=$IQ," \
                iqkcol-off "${P}_KCOL_SKIP=$IQL ${P}_MTOK_SKIP=$IQL" "MUL_MAT,$EXPERT_OPS" "=$IQ," \
                iqkcol-skip "${P}_SKIP=$IQL" "MUL_MAT,$EXPERT_OPS" "=$IQ,"
            # C2 (docs/data/2026-11-04_iq2xs_kernel/run/gpu_checks.sh): 1 to 4 tokens (the fusion test names them m)
            local T5='(iq2_xxs|iq2_xs|iq3_s|iq2_s|iq3_xxs)'
            printf '%s\t%s\t%s\t%s\tpass\t\n' \
                iq2xs-on-fusion "" MUL_MAT_VEC_FUSION "=iq3_xxs,.*m=[1-4]," \
                iq2xs-on-shared "" MUL_MAT_ID_SHARED "=iq3_xxs,.*n=[1-4]" \
                iq2xs-skip-mm "${P}_SKIP=iq2_xxs,iq2_xs,iq3_s" MUL_MAT "=$T5,.*n=[1-4]," \
                iq2xs-skip-mmid "${P}_SKIP=iq2_xxs,iq2_xs,iq3_s" MUL_MAT_ID "=$T5,.*n=[1-4]," \
                iq2xs-skip-fusion "${P}_SKIP=iq2_xxs,iq2_xs,iq3_s" MUL_MAT_VEC_FUSION "=$T5,.*m=[1-4]," \
                iq2xs-skip-shared "${P}_SKIP=iq2_xxs,iq2_xs,iq3_s" MUL_MAT_ID_SHARED "=$T5,.*n=[1-4]"
            # C6 (docs/data/2026-11-04_qsa_sparse_attn/run/qsa_gpu.sh): faults 2 and 3 must fail cases
            local Q=GGML_CUDA_FATTN_SPARSE_PASCAL
            printf '%s\t%s\t%s\t%s\t%s\t\n' \
                qsa-sparse1 "$Q=1" FLASH_ATTN_EXT "$QSA" pass \
                qsa-sparse2 "$Q=2 ${Q}_NQ=128" FLASH_ATTN_EXT "$QSA" pass \
                qsa-sparse0 "$Q=0" FLASH_ATTN_EXT "$QSA" pass \
                qsa-fault1 "$Q=2 ${Q}_FAULT=1" FLASH_ATTN_EXT "$QSA" pass \
                qsa-fault2 "$Q=2 ${Q}_FAULT=2" FLASH_ATTN_EXT "$QSA" fail \
                qsa-fault3 "$Q=2 ${Q}_FAULT=3" FLASH_ATTN_EXT "$QSA" fail
            # both scripts' memcheck cases, which ran through the PTX only so far
            local t c k
            for t in iq2_xxs iq2_xs iq3_s; do
                k=0
                for c in "MUL_MAT_ID|=$t,.*m=640,n=1,k=2560" "MUL_MAT_ID|=$t,.*m=2560,n=4,k=1024" \
                         "MUL_MAT_VEC_FUSION|=$t,.*m=2,n=640,k=2560,use_id=1" "MUL_MAT|=$t,.*m=641,n=1,k=2560" \
                         "MUL_MAT|=$t,.*m=2560,n=3,k=6144"; do
                    k=$((k + 1))
                    printf '%s\t%s\t%s\t%s\tpass\tmemcheck\n' "memcheck-iq2xs-$t-$k" "" "${c%%|*}" "${c#*|}"
                done
            done
            for t in iq2_xxs iq2_xs iq3_s iq2_s iq3_xxs iq1_m; do
                k=0
                for c in "MUL_MAT_ID|=$t,.*m=640,n=8,k=2560" "MUL_MAT_ID|=$t,.*m=1000,n=5,k=1024" \
                         "MUL_MAT_VEC_FUSION|=$t,.*m=7,n=641,k=2560" "MUL_MAT|=$t,.*m=641,n=6,k=2560" \
                         "MUL_MAT|=$t,.*m=2560,n=8,k=6144"; do
                    k=$((k + 1))
                    printf '%s\t%s\t%s\t%s\tpass\tmemcheck\n' "memcheck-iqkcol-$t-$k" "" "${c%%|*}" "${c#*|}"
                done
            done ;;
        fap)
            local F=GGML_CUDA_FATTN_PASCAL
            printf '%s\t%s\t%s\t%s\t%s\t\n' \
                fap-on "$F=1" FLASH_ATTN_EXT '.' pass \
                fap-fault1 "$F=1 ${F}_FAULT=1" FLASH_ATTN_EXT "$FAP_OWN" pass \
                fap-fault2 "$F=1 ${F}_FAULT=2" FLASH_ATTN_EXT "$FAP_OWN" fail \
                fap-fault3 "$F=1 ${F}_FAULT=3" FLASH_ATTN_EXT "$FAP_OWN" fail \
                fap-off "$F=0" FLASH_ATTN_EXT "$FAP_OWN" pass ;;
        flashnext-moe)
            printf '%s\t%s\t%s\t%s\tpass\t\n' fnmoe "" $EXPERT_OPS "$FN_MOE" fnmoe-mtok0 "${P}_MTOK=0" $EXPERT_OPS "$FN_MOE" ;;
        n2p)
            suite_runs_n2p 'n=[0-9]+,' "type(_a)?=" ;;
    esac
}
# docs/data/2026-11-04_n2p_flash/run/gpu_checks.sh's runs: <the token counts> <the products' type field>; natively
# every token count and the prompt batches, through the PTX 1-8 tokens of the products (MMQ takes the host's tile
# configuration there, not Pascal's)
suite_runs_n2p() {
    local NT=$1 TA=$2 r
    local NP="${TA}$N4,.*$NT|type=$N4," RH="${TA}$HW,.*$NT|type=$HW,"
    for r in "np:" "np-mtok-off:${P}_MTOK=0" "np-cap1:${P}_MTOK=1,1" "np-rb1:${P}_MTOK=1,0,1" \
             "np-notile:${P}_MTOK=1,0,0,0,0" "np-skip:${P}_SKIP=n2p,n3p,n2p_128,n3p_128"; do
        printf '%s\t%s\t%s\t%s\tpass\t\t%s\n' "${r%%:*}" "${r#*:}" "MUL_MAT,$EXPERT_OPS,GET_ROWS" "$NP" "$N2P_KNOWN"
    done
    [ "$NT" = 'n=[1-8],' ] || printf '%s\t%s\t%s\t%s\tpass\t\t\n' np-mmq "" MUL_MAT,MUL_MAT_ID "type_a=$N4,.*n=(9|33|128),"
    printf '%s\t%s\t%s\t%s\tpass\t\t\n' reg-half "" "MUL_MAT,$EXPERT_OPS" "$RH" reg-rows "" GET_ROWS '.'
}
# the suite's runs for a newer card through the compute_61 PTX, when they differ from the native ones
suite_runs_ptx() {
    case $1 in
        n2p) suite_runs_n2p 'n=[1-8],' "type_a=" ;;
    esac
}
suite_filters "$SUITE" > /dev/null || { echo "unknown SUITE $SUITE" >&2; exit 1; }

if step build; then
    : "${WORK:?the build step needs WORK}" "${REPO:?and REPO}" "${BRANCH:?and BRANCH}"
    log "build fork/$BRANCH"
    TESTS=1 WORK=$WORK DOCKER_ARGS="--cpu-shares=256 ${DOCKER_ARGS:-}" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$OUT/build.log" 2>&1 || { log "build FAILED: $OUT/build.log"; exit 1; }
    T=$(ls -td "$WORK"/llama.cpp-*-"${BRANCH##*/}" | grep -v '\.tar\.gz$' | head -n 1)
    BIN=$T/bin
    log "build done: $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1)"
fi

if step record; then
    BIN=$(realpath "${BIN:?BIN=<bin/ of the rig tarball>}")
    [ -x "$BIN/test-backend-ops" ] || { echo "no test-backend-ops in $BIN" >&2; exit 1; }
    mkdir -p "$S"
    # empty fields as "-": read with a tab IFS would merge two tabs
    dash() { awk -F'\t' -v OFS='\t' '{ for (i = 1; i <= NF; i++) if ($i == "") $i = "-"; print }'; }
    suite_runs "$SUITE" | dash > "$S/runs.tsv"
    suite_runs_ptx "$SUITE" | dash > "$S/runs-ptx.tsv"
    [ -s "$S/runs-ptx.tsv" ] || rm -f "$S/runs-ptx.tsv"
    {
        echo "suite: $SUITE"
        echo "recorded: $(date -Is)"
        grep -E '^(name|commit)' "$BIN/../BUILDINFO.txt" 2>/dev/null
        echo "recording CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
        echo "test-backend-ops md5: $(md5sum < "$BIN/test-backend-ops" | cut -d' ' -f1)"
        echo "filters (-o, -p${EXTRA_P:+, and every case also matches $EXTRA_P}):"
        suite_filters "$SUITE" | sed 's/^/  /'
        cut -f7 "$S/runs.tsv" | grep -v -x -e - -e '' | sort -u | sed 's/^/known to fail (a regex on the case string): /'
    } > "$S/record.txt"
    A=(-j "$JOBS")
    [ -n "${SAMPLE:-}" ] && A+=(--sample "$SAMPLE")
    [ -n "${MAX_CASE_MB:-}" ] && A+=(--record-max-case-mb "$MAX_CASE_MB")
    [ -n "${BUDGET_GB:-}" ] && A+=(--record-budget-gb "$BUDGET_GB")
    [ "${DRY:-0}" = 1 ] && A+=(--record-dry)
    # the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
    # (CUDA then finds no device; only the CPU backend is present)
    tbo() {
        # shellcheck disable=SC2086
        docker run --rm --label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-} -v "$BIN:$BIN:ro" -v "$S:$S" "$IMAGE" sh -c \
            's=/usr/local/cuda/lib64/stubs/libcuda.so
             if [ -e $s ]; then
                 mkdir -p /tmp/stub && ln -sf $s /tmp/stub/libcuda.so.1
                 export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
             fi
             exec nice -n 19 "$@"' sh "$BIN/test-backend-ops" "$@"
    }
    flt() { # <params>: the -p filter with EXTRA_P
        if [ -n "${EXTRA_P:-}" ]; then echo "(?=.*($EXTRA_P))(?:.*)($1)"; else echo "$1"; fi
    }
    if [ -n "${LINK_FROM:-}" ] && [ "${DRY:-0}" != 1 ]; then
        # cases a complete set already holds (the same case string, so the same id): hardlinked in and kept, not drawn
        # and quantized again
        [ -f "$LINK_FROM/manifest.tsv" ] || { echo "no manifest.tsv in $LINK_FROM" >&2; exit 1; }
        mkdir -p "$S/cases"
        while IFS=$'\t' read -r ops params; do
            tbo -b CPU -o "$ops" -p "$(flt "$params")" --record-dry 2> /dev/null \
                | sed -n 's/^  \(.*\): in [0-9]* B, ref [0-9]* B$/\1/p'
        done < <(suite_filters "$SUITE") > "$S/selected.txt"
        awk -F'\t' 'NR == FNR { sel[$0] = 1; next } ($2 in sel) { print $1 }' "$S/selected.txt" "$LINK_FROM/manifest.tsv" \
            > "$S/linked.txt"
        while read -r id; do
            for e in in ref meta; do ln -f "$LINK_FROM/cases/$id.$e" "$S/cases/$id.$e"; done
        done < "$S/linked.txt"
        log "linked $(wc -l < "$S/linked.txt") of $(wc -l < "$S/selected.txt") selected cases from $LINK_FROM"
    fi
    n=0
    while IFS=$'\t' read -r ops params; do
        n=$((n + 1))
        p=$(flt "$params")
        log "record filter $n: -o $ops -p '$p'"
        tbo -b CPU -o "$ops" -p "$p" --record "$S" "${A[@]}" > "$S/record-$n.log" 2>&1
        log "filter $n exit $?: $(grep -a '  record:' "$S/record-$n.log" | sed 's/^ *//')"
    done < <(suite_filters "$SUITE")
    if [ "${DRY:-0}" != 1 ]; then
        # one line per case: id, case string, input bytes and md5, reference bytes and md5
        for m in "$S"/cases/*.meta; do
            awk -F'\t' -v i="$(basename "$m" .meta)" '$1 == "name" { n = $2 } $1 == "in" { ib = $2; im = $3 }
                $1 == "ref" { rb = $2; rm = $3 } END { print i "\t" n "\t" ib "\t" im "\t" rb "\t" rm }' "$m"
        done | sort -n > "$S/manifest.tsv"
        log "manifest: $(wc -l < "$S/manifest.tsv") cases, $(du -sb "$S/cases" | cut -f1) bytes in cases/"
    fi
fi

if step tar && [ "${DRY:-0}" != 1 ]; then
    tar cf "$OUT/$SUITE.tar" -C "$OUT" "$SUITE"
    md5sum < "$OUT/$SUITE.tar" | cut -d' ' -f1 > "$OUT/$SUITE.tar.md5"
    log "tar: $OUT/$SUITE.tar $(stat -c %s "$OUT/$SUITE.tar") bytes, md5 $(cat "$OUT/$SUITE.tar.md5")"
fi
