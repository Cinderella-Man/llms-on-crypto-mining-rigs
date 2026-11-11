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
#   runs    only the run lists (runs.tsv, runs-ptx.tsv) of an OUT/SUITE/ recorded earlier, from this script's suite: for
#           a suite whose runs changed after the record (the cases must be the same); then "tar" packs it again
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
#   qsa-prompt    the sparse attention's cases (FLASH_ATTN_EXT with n_kv_max) under its prompt rule
#                 (GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT, docs/data/2026-11-11_c705_sparse_prompt_rule): mode 1 with the
#                 rule off and on, its two thresholds and its query-token cap, mode 2, the faults. Its runs-ptx.tsv
#                 holds the mode-2 runs, as mode 1 leaves a newer card on its own kernels
#   q5kmt         Q5_K's products of 1-4 tokens (the output layer of Qwen3.8-Flash-Next among them) with the 2-4-token
#                 kernel that decodes each weight once (GGML_CUDA_MMVQ_PASCAL_Q5K_MT, C690) at each setting, under the
#                 Flash-Next preset's k-column switches, and without it (docs/data/2026-11-11_c690_head_four_rows)
#   indexer-rows  LIGHTNING_INDEXER (LID below): the k-pool indexer reading the pooled keys in place beside its scores
#                 over the gathered copy (bit for bit), and the fault that scores the wrong rows
#                 (docs/data/2026-11-11_c704_indexer_gather)
#   range-mask    RANGE_MASK, every case: the QSA pool mask built on the card from each token's range of visible pools
#                 (exact), and the fault that makes one more column visible after each range
#                 (docs/data/2026-11-11_c719_pool_mask_on_cards)
#   dmt           IQ4_XS's and Q4_K's products of 1-5 tokens (Qwen3.8-Flash-Next's attention and recurrent projections
#                 among them) with the 2-4-token kernel that decodes each weight once (GGML_CUDA_MMVQ_PASCAL_DMT, C722) at
#                 each setting, under the Flash-Next preset's k-column switches, and without it
#                 (docs/data/2026-11-11_c722_short_batch_products)
#   dmtkt         dmt's cases (with K = 6144 at 333 and 4097 rows and a view of wider rows) with the kernel's K tiles
#                 (GGML_CUDA_MMVQ_PASCAL_DMT_KT, C728) at each setting: where they help (=1) and wherever they fit (=2),
#                 two row blocks a round, without the prefetch, fewer warps, R = 2 (no tiles), without the preset, the
#                 kernel without them, and the fault (each iteration takes the next tile's activations), which must fail
#                 cases (docs/data/2026-11-11_c728_readonce_k6144)
#   top-k-rows    TOP_K at Flash-Next's 248,320 columns over one to eight rows against the CPU, and TOP_K_ROWS: the
#                 top-k of several rows in one op against each row's own, on the card (the backend samplers' one top-k
#                 over the output rows, LLAMA_SAMPLING_TOP_K_ROWS, C691; docs/data/2026-11-11_c691_sampling_launches)
#   qsa-union     the sparse attention's cases (FLASH_ATTN_EXT with n_kv_max) with union lists
#                 (GGML_CUDA_FATTN_SPARSE_PASCAL_UNION, docs/data/2026-11-11_c103_union_gather): mode 1 with and without
#                 them and their threshold, mode 2 through the union kernel at three split counts, the four faults. Its
#                 runs-ptx.tsv holds the mode-2 runs, as mode 1 leaves a newer card on its own kernels
#   v4night       the seven suites of the v4 night's branches in one set, as merged in pascal-p106-v4-b12246-night
#                 (docs/data/2026-11-11_v4_night_merge): NIGHT below; each filter recorded once, every suite's runs
#                 (runs-ptx.tsv: a suite's PTX runs where it has them, else its native runs)
#   v4day         the seven kernel suites of the v4 day's branches in one set, as merged in pascal-p106-v4-b12246-day
#                 (docs/data/2026-11-11_v4_day_merge): DAY below, each filter recorded once, every suite's runs, as v4night (on
#                 that build _DMT=1 and _Q5K_MT=1 run R = 1, C758 and C766)
#   bf16x         BF16's 2-D products of 1-8 tokens (Qwen3.8-Flash-Next's hyper-connections, indexer and recurrent gates
#                 among them) under the launch rule and with GGML_CUDA_MMVF_PASCAL_EXACT=1 (C750: the 3-4-token products
#                 in the one-token order) at three rules and as for 10 SMs, with =2, and the fault that leaves a quarter
#                 of each row out of the 32-lane kernels of that order (docs/data/2026-11-11_c750_bf16_exact). Its
#                 runs-ptx.tsv runs them on a newer card with the P106-090's launch shapes
#   f32x          F32's 2-D products of 1-8 tokens (Qwen3.8-Flash-Next's shared-expert gate and router among them) with
#                 cuBLAS at 4-8 tokens (off) and with GGML_CUDA_MMVF_F32_MT=1 and =2 (C754: mul_mat_vec_f, each token's
#                 one-token sums), and the fault that adds only the first half of each row of the products that switch
#                 takes (docs/data/2026-11-11_c754_f32_mmvf). Its runs-ptx.tsv runs them on a newer card at =2
#   q3kmt         Q3_K's products of 1-8 tokens (Qwen3.8-Flash-Next's Q3_K attention and recurrent projections and the
#                 shared expert's Q3_K products among them) with the 2-8-token kernel in the k-column kernel's order
#                 (GGML_CUDA_MMVQ_PASCAL_Q3K_MT, C726) at each setting, under the Flash-Next preset's k-column switches,
#                 without it, and its fault (docs/data/2026-11-11_c726_q3k_multitoken)
#   glumt         MUL_MAT_GLU: a gate and an up product of their own types and their SWIGLU at 1-8 tokens (Qwen3.8-Flash-
#                 Next's shared-expert type pairs at 2560 -> 640, and every pair of Q2_0, Q3_K, Q4_K, Q5_K and IQ4_XS at
#                 other shapes) with the one launch of GGML_CUDA_MMVQ_PASCAL_GLU_MT (C757) at each setting, under the
#                 preset's k-column switches with the read-once kernels it repeats (_DMT=1 _DMT_R=1, _Q5K_MT=1), without
#                 some of them, off, and its fault (docs/data/2026-11-11_c757_shexp_fused)
#   q5k1r         Q5_K's one-token products (q5kmt's one-token cases and Qwen3.8-Flash-Next's other Q5_K shapes) with
#                 C690's kernel at one token (GGML_CUDA_MMVQ_PASCAL_Q5K_1R, C767) at each setting, under the preset's
#                 k-column switches, without them, with the 2-4-token switch too, off, Q5_K left out, and its fault (each
#                 row's last iteration left out), which must fail cases (docs/data/2026-11-11_c767_q5k_one_row)
#   ro8           IQ4_XS's, Q4_K's and Q5_K's products of 5-8 tokens (Qwen3.8-Flash-Next's attention and recurrent
#                 projections and its Q5_K products among them) with the read-once kernels of C722 and C690 at 5-8 tokens
#                 (GGML_CUDA_MMVQ_PASCAL_RO8, C729) at each setting, under the preset's k-column switches, with the 2-4-
#                 token switches too, without the preset, off, types left out, and its fault (every token stages the first
#                 token's activations), which must fail cases (docs/data/2026-11-11_c729_readonce_5to8)
#   q6kmt         Q6_K's products of 1-5 tokens (Qwen3.8-Flash-Next's Q6_K attn_output and ssm_out, 6144 -> 2560, among them)
#                 with C722's 2-4-token kernel (GGML_CUDA_MMVQ_PASCAL_Q6K_MT, C727) at each setting, under the preset's
#                 k-column switches, without them, beside _DMT=1, off, Q6_K left out, and its fault (every token stages the
#                 first token's activations), which must fail cases, and the fault without the switch, which must not
#                 (docs/data/2026-11-11_c727_q6k_readonce)
#   v4kern        the three kernel suites of the v4 kernel merge in one set, as merged in pascal-p106-v4 (C796,
#                 docs/data/2026-11-11_v4_kernel_merge): KERN below, each filter recorded once, every suite's runs, and
#                 runs with GGML_CUDA_MMVQ_PASCAL_Q6K_MT, _DMT_KT and _RO8 on together over all their cases, at several
#                 settings and with each one's fault (which must fail cases)
# Env: OUT (folder for the records), SUITE, BIN (bin/ of an unpacked rig tarball; the build step sets it), WORK and
#   REPO and BRANCH (build step), JOBS (test-backend-ops -j workers, default 4), SAMPLE (record about one case in N),
#   MAX_CASE_MB, BUDGET_GB (caps: cases over them are left out and replay reports them), DRY=1 (sizes only),
#   EXTRA_P (a -p regex every filter must also match, e.g. a "model shapes only" set), LINK_FROM (complete sets, separated by
#   spaces, whose cases this suite selects are hardlinked in instead of recorded again; the first set that holds a case
#   gives it), IMAGE, DOCKER_ARGS
# Usage: OUT=<dir> SUITE=kshape BIN=<rig tarball>/bin bash scripts/record_checks.sh
set -uo pipefail
OUT=${OUT:?OUT=<folder for the records>}
SUITE=${SUITE:?SUITE=kshape|ladder3-rest|fap|flashnext-moe|n2p|qsa-prompt|q5kmt|indexer-rows|range-mask|dmt|dmtkt|top-k-rows|qsa-union|v4night|bf16x|f32x|q3kmt|glumt|q5k1r|v4day|ro8|q6kmt|v4kern}
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
# that may fail until a named fix: replay_checks.sh counts their FAIL lines apart; no suite names one today). The
# switches and filters are those of the native runs of each branch's gpu_checks.sh.
EXPERT_OPS=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
P=GGML_CUDA_MMVQ_PASCAL
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs|q4_0|q5_0|iq4_nl|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s|iq1_m),(type_b=f32,)?m=[0-9]+,n=[1-8],'
IQ='iq(2_xxs|2_xs|2_s|3_xxs|3_s|1_m)'
IQL=iq2_xxs,iq2_xs,iq3_s,iq2_s,iq3_xxs,iq1_m
QSA='n_kv_max=[1-9]'
FAP_OWN='hsk=128,hsv=128,nh=(4|8),nr23=\[8,1\],kv=[0-9]+,nb=1,'
FN_MOE='n_mats=512,n_used=10,|^type=[^,]+,n=[0-9]+$'
N4='(n2p|n3p|n2p_128|n3p_128)'
# the indexer: every in-place case, every four-head case (Flash-Next's), and the one- and 32-token cases of 32 and 64
# heads (the 512-token ones of those hold 7.6 GB, and their kernel is the one-token cases')
LID='n_cells=|nh=4,|nb=(1|32),'
HW='(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s)'
Q5MT='type_a=q5_K,(type_b=f32,)?m=[0-9]+,n=[1-4],'
Q51='type_a=q5_K,(type_b=f32,)?m=[0-9]+,n=1,'
RO8='type_a=(iq4_xs|q4_K|q5_K),(type_b=f32,)?m=[0-9]+,n=[5-8],'
Q6MT='type_a=q6_K,(type_b=f32,)?m=[0-9]+,n=[1-5],'
DMT='type_a=(iq4_xs|q4_K),(type_b=f32,)?m=[0-9]+,n=[1-5],'
Q3MT='type_a=q3_K,(type_b=f32,)?m=[0-9]+,n=[1-8],'
# C757's gate/up/SWIGLU cases (MUL_MAT_GLU), and those its kernel takes under every run but the ones that turn its
# kernels off: 2-8 tokens of Q2_0 and Q3_K pairs at the shared expert's shape
GLUMT='type_gate=[^,]+,type_up=[^,]+,m=[0-9]+,n=[1-8],'
GLUMT4='type_gate=(q2_0|q3_K),type_up=(q2_0|q3_K),m=640,n=[2-8],'
# the top-k cases at Flash-Next's vocabulary
TKV='ne=\[248320,'
# BF16's 2-D products of 1-8 tokens, and those GGML_CUDA_MMVF_PASCAL_EXACT's 32-lane kernel takes on 5 SMs (3-4 tokens,
# 128 lanes at one token)
BFX='type_a=bf16,type_b=f32,m=[0-9]+,n=[1-8],k=[0-9]+,bs=\[1,1\],nr=\[1,1\],'
BFX4='type_a=bf16,type_b=f32,m=(320|333),n=[34],k=(10240|4104|4392|4736),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,'
# F32's 2-D products of 1-8 tokens, and those of C754's shapes that GGML_CUDA_MMVF_F32_MT takes (4-8 tokens, rows of an
# even width at an even stride)
F32X='type_a=f32,type_b=f32,m=[0-9]+,n=[1-8],k=[0-9]+,bs=\[1,1\],nr=\[1,1\],'
F32X4='type_a=f32,type_b=f32,m=(1|512|48|33|129|7|256),n=[4-8],k=(2560|64|100|300|6144|320),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=(0|336),'
NIGHT='qsa-prompt qsa-union q5kmt dmt indexer-rows range-mask top-k-rows'
DAY='dmt q5kmt q3kmt bf16x f32x glumt q5k1r'
KERN='q6kmt dmtkt ro8'
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
        qsa-prompt)
            printf '%s\t%s\n' FLASH_ATTN_EXT "$QSA" ;;
        q5kmt)
            printf '%s\t%s\n' MUL_MAT "$Q5MT" ;;
        q5k1r)
            printf '%s\t%s\n' MUL_MAT "$Q51" ;;
        ro8)
            printf '%s\t%s\n' MUL_MAT "$RO8" ;;
        q6kmt)
            printf '%s\t%s\n' MUL_MAT "$Q6MT" ;;
        indexer-rows)
            printf '%s\t%s\n' LIGHTNING_INDEXER "$LID" ;;
        range-mask)
            printf '%s\t%s\n' RANGE_MASK '.' ;;
        dmt|dmtkt)
            printf '%s\t%s\n' MUL_MAT "$DMT" ;;
        q3kmt)
            printf '%s\t%s\n' MUL_MAT "$Q3MT" ;;
        glumt)
            printf '%s\t%s\n' MUL_MAT_GLU "$GLUMT" ;;
        top-k-rows)
            printf '%s\t%s\n' TOP_K "$TKV" TOP_K_ROWS '.' ;;
        qsa-union)
            printf '%s\t%s\n' FLASH_ATTN_EXT "$QSA" ;;
        bf16x)
            printf '%s\t%s\n' MUL_MAT "$BFX" ;;
        f32x)
            printf '%s\t%s\n' MUL_MAT "$F32X" ;;
        v4night)
            local s
            for s in $NIGHT; do suite_filters "$s"; done | awk '!seen[$0]++' ;;
        v4day)
            local s
            for s in $DAY; do suite_filters "$s"; done | awk '!seen[$0]++' ;;
        v4kern)
            local s
            for s in $KERN; do suite_filters "$s"; done | awk '!seen[$0]++' ;;
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
        qsa-prompt)
            suite_runs_qsa_prompt 1 ;;
        q5kmt)
            # the Flash-Next preset's k-column switches (release/flash-next-v3/settings.env), then the kernel's settings,
            # each run naming its rows per lane group and its prefetch, so that it runs the same kernel on builds before
            # and after C766 changed the defaults (R = 2 with the prefetch before, R = 1 without it since; `defaults`
            # is the build's own)
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_Q5K_MT"
            for r in "q5kmt-on:$KP $M=1 ${M}_R=2 ${M}_PF=1" "q5kmt-nopf:$KP $M=2 ${M}_R=2" \
                     "q5kmt-nw2:$KP $M=1 ${M}_R=2 ${M}_PF=1 ${M}_NW=2" "q5kmt-nw1:$KP $M=1 ${M}_R=2 ${M}_PF=1 ${M}_NW=1" \
                     "q5kmt-r1:$KP $M=1 ${M}_R=1 ${M}_PF=1" "q5kmt-r1-nopf:$KP $M=2 ${M}_R=1" \
                     "q5kmt-r1-nw1:$KP $M=1 ${M}_R=1 ${M}_PF=1 ${M}_NW=1" "q5kmt-defaults:$M=1" "q5kmt-off:$KP" \
                     "q5kmt-skip:$KP $M=1 ${P}_SKIP=q5_K"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$Q5MT"
            done ;;
        q5k1r)
            # C767: the preset's k-column switches, then the one-token switch at each setting of the kernel (R = 1 and 2,
            # with and without the prefetch, one warp a block, the output layer alone), with the 2-4-token switch too,
            # without the preset, off, Q5_K left to llama.cpp's kernels (so the new path is too), and the fault (each
            # row's last iteration left out of the one-token products), which must fail cases
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_Q5K_MT" O="${P}_Q5K_1R"
            for r in "q5k1r-on:$KP $O=1" "q5k1r-r2:$KP $O=1 ${M}_R=2" "q5k1r-pf:$KP $O=1 ${M}_PF=1" \
                     "q5k1r-r2-pf:$KP $O=1 ${M}_R=2 ${M}_PF=1" "q5k1r-nw1:$KP $O=1 ${M}_NW=1" \
                     "q5k1r-head:$KP $O=1 ${O}_ROWS=100000" "q5k1r-mt:$KP $M=1 $O=1" "q5k1r-defaults:$O=1" \
                     "q5k1r-off:$KP" "q5k1r-skip:$KP $O=1 ${P}_SKIP=q5_K"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$Q51"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' q5k1r-fault "$KP $O=1 ${O}_FAULT=1" MUL_MAT "$Q51" ;;
        ro8)
            # C729: the preset's k-column switches, then 5-8 tokens on the read-once kernels at each setting (with and
            # without the prefetch, Q5_K's prefetch, one and two warps a block, the 2-4-token kernels' R = 2, which 5-8
            # tokens do not take), with the 2-4-token switches too, without the preset, off, Q4_K or every type left to
            # the other kernels, and the fault (every token stages the first token's activations), which must fail cases
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_RO8" D="${P}_DMT" Q="${P}_Q5K_MT"
            for r in "ro8-on:$KP $M=1" "ro8-nopf:$KP $M=2" "ro8-pf:$KP $M=1 ${Q}_PF=1" \
                     "ro8-nw1:$KP $M=1 ${D}_NW=1 ${Q}_NW=1" "ro8-nw2:$KP $M=1 ${D}_NW=2 ${Q}_NW=2" \
                     "ro8-r2:$KP $M=1 ${D}_R=2 ${Q}_R=2" "ro8-mt:$KP $D=1 $Q=1 $M=1" "ro8-mt-nopf:$KP $D=2 $Q=2 $M=2" \
                     "ro8-defaults:$M=1" "ro8-off:$KP" "ro8-skip-q4k:$KP $M=1 ${D}_SKIP=q4_K" \
                     "ro8-skip:$KP $M=1 ${P}_SKIP=iq4_xs,q4_K,q5_K"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$RO8"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' ro8-fault "$KP $M=1 ${M}_FAULT=1" MUL_MAT "$RO8" ;;
        q6kmt)
            # C727, as dmt: the preset's k-column switches, then the kernel's settings (R = 1 by default, forced to 2;
            # without the prefetch; one warp a block allowed; at most 2 warps), beside _DMT=1, without the preset, off,
            # Q6_K left out of the kernel and of the Pascal kernels, the fault (every token stages the first token's
            # activations), which must fail cases, and the fault without the switch, which must not
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_Q6K_MT" D="${P}_DMT"
            for r in "q6kmt-on:$KP $M=1" "q6kmt-nopf:$KP $M=2" "q6kmt-r2:$KP $M=1 ${D}_R=2" \
                     "q6kmt-r2-nopf:$KP $M=2 ${D}_R=2" "q6kmt-nw1:$KP $M=1 ${D}_NW=1" "q6kmt-nw2:$KP $M=1 ${D}_NW=2" \
                     "q6kmt-dmt:$KP $M=1 $D=1" "q6kmt-defaults:$M=1" "q6kmt-off:$KP" \
                     "q6kmt-skip:$KP $M=1 ${D}_SKIP=q6_K" "q6kmt-pskip:$KP $M=1 ${P}_SKIP=q6_K" \
                     "q6kmt-fault-off:$KP ${M}_FAULT=1"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$Q6MT"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' q6kmt-fault "$KP $M=1 ${M}_FAULT=1" MUL_MAT "$Q6MT" ;;
        indexer-rows)
            # C704: the LID cases, then the in-place reads with GGML_CUDA_LID_ROWS_FAULT=1, which scores rows i mod n_cells
            # instead of the listed ones and so must fail every case (C749; before it, rows 0 to n-1 and lists up to n_cells)
            printf '%s\t%s\t%s\t%s\t%s\t\n' \
                lid "" LIGHTNING_INDEXER "$LID" pass \
                lid-rows-fault GGML_CUDA_LID_ROWS_FAULT=1 LIGHTNING_INDEXER 'n_cells=' fail ;;
        range-mask)
            # C719: every RANGE_MASK case, then GGML_CUDA_RANGE_MASK_FAULT=1, which makes one more column visible after
            # each range and so must fail cases
            printf '%s\t%s\t%s\t%s\t%s\t\n' \
                range-mask "" RANGE_MASK . pass \
                range-mask-fault GGML_CUDA_RANGE_MASK_FAULT=1 RANGE_MASK . fail ;;
        dmt)
            # C722, as q5kmt: the preset's k-column switches, then the kernel's settings (the default, forced to R = 1
            # and 2, C722's choice per product; without the prefetch; fewer warps), without the preset, off, each type
            # left out, the one-token kernels off for both types (so the new kernel is too). Since C758 (-c758) the
            # default is R = 1 and _R=0 is C722's choice; before it _R=0 was the default, so there dmt-rule = dmt-on.
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_DMT"
            for r in "dmt-on:$KP $M=1" "dmt-nopf:$KP $M=2" "dmt-r1:$KP $M=1 ${M}_R=1" "dmt-r2:$KP $M=1 ${M}_R=2" \
                     "dmt-r1-nopf:$KP $M=2 ${M}_R=1" "dmt-r2-nopf:$KP $M=2 ${M}_R=2" "dmt-rule:$KP $M=1 ${M}_R=0" \
                     "dmt-nw1:$KP $M=1 ${M}_NW=1" "dmt-nw2:$KP $M=1 ${M}_NW=2" \
                     "dmt-defaults:$M=1" "dmt-off:$KP" "dmt-skip-q4k:$KP $M=1 ${M}_SKIP=q4_K" \
                     "dmt-skip-iq4xs:$KP $M=1 ${M}_SKIP=iq4_xs" "dmt-skip:$KP $M=1 ${P}_SKIP=iq4_xs,q4_K"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$DMT"
            done ;;
        dmtkt)
            # C728, as dmt: the preset's k-column switches and the read-once kernel, then its K tiles at each setting
            # (=1 where they give a block more warps, =2 wherever they fit, two row blocks a round, without the prefetch,
            # 4 and 1 warps a block), with R = 2 (no tiles), without the preset, the kernel without tiles, and the fault
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_DMT"
            for r in "dmtkt-kt1:$KP $M=1 ${M}_KT=1" "dmtkt-kt2:$KP $M=1 ${M}_KT=2" "dmtkt-kt1-m2:$KP $M=1 ${M}_KT=1 ${M}_KT_M=2" \
                     "dmtkt-kt2-m2:$KP $M=1 ${M}_KT=2 ${M}_KT_M=2" "dmtkt-kt1-nopf:$KP $M=2 ${M}_KT=1" \
                     "dmtkt-kt2-nopf:$KP $M=2 ${M}_KT=2" "dmtkt-kt2-nw4:$KP $M=1 ${M}_KT=2 ${M}_KT_NW=4" \
                     "dmtkt-kt2-nw1:$KP $M=1 ${M}_KT=2 ${M}_KT_NW=1" "dmtkt-kt1-r2:$KP $M=1 ${M}_R=2 ${M}_KT=1" \
                     "dmtkt-defaults:$M=1 ${M}_KT=1" "dmtkt-on:$KP $M=1"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$DMT"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' dmtkt-fault "$KP $M=1 ${M}_KT=1 ${M}_KT_FAULT=1" MUL_MAT "$DMT" ;;
        q3kmt)
            # C726, as dmt: the preset's k-column switches, then the kernel's settings (R = 1 by default, forced to 2;
            # without the prefetch; fewer warps; no slot padding), without the preset, off, the k-column kernel off or
            # without Q3_K (the new kernel follows it), and the fault (each token takes the next token's activations),
            # which must fail cases
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" M="${P}_Q3K_MT"
            for r in "q3kmt-on:$KP $M=1" "q3kmt-nopf:$KP $M=2" "q3kmt-r2:$KP $M=1 ${M}_R=2" \
                     "q3kmt-r2-nopf:$KP $M=2 ${M}_R=2" "q3kmt-nw1:$KP $M=1 ${M}_NW=1" "q3kmt-nw2:$KP $M=1 ${M}_NW=2" \
                     "q3kmt-pad0:$KP $M=1 ${M}_PAD=0" "q3kmt-defaults:$M=1" "q3kmt-off:$KP" \
                     "q3kmt-kcol-skip:$KP $M=1 ${P}_KCOL_SKIP=q3_K" "q3kmt-kcol-off:$M=1 ${P}_KCOL=0"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$Q3MT"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' q3kmt-fault "$KP $M=1 ${M}_FAULT=1" MUL_MAT "$Q3MT" ;;
        glumt)
            # C757: the preset's k-column switches with the read-once kernels the one launch repeats (C722's at R = 1,
            # C690's), then its settings (without the prefetch; 1 and 4 warp pairs a block), without the preset, without
            # C722's and C690's kernels (only Q2_0 and Q3_K pairs taken), the k-column kernel off (no pair taken), the
            # switch off, and the fault (each token takes the next token's activations), which must fail cases
            local KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" RO="${P}_DMT=1 ${P}_DMT_R=1 ${P}_Q5K_MT=1" M="${P}_GLU_MT"
            for r in "glumt-on:$KP $RO $M=1" "glumt-nopf:$KP $RO $M=2" "glumt-np1:$KP $RO $M=1 ${M}_NP=1" \
                     "glumt-np4:$KP $RO $M=1 ${M}_NP=4" "glumt-defaults:$RO $M=1" "glumt-ro-off:$KP $M=1" \
                     "glumt-kcol-off:$RO $M=1 ${P}_KCOL=0" "glumt-off:$KP $RO"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT_GLU "$GLUMT"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' glumt-fault "$KP $RO $M=1 ${M}_FAULT=1" MUL_MAT_GLU "$GLUMT4" ;;
        top-k-rows)
            # C691: no CUDA switch (the change is in the graph); the op at the vocabulary's width, and its rows
            printf '%s\t%s\t%s\t%s\tpass\t\n' tk "" TOP_K "$TKV" tk-rows "" TOP_K_ROWS '.' ;;
        qsa-union)
            suite_runs_qsa_union 1 ;;
        bf16x)
            suite_runs_bf16x "" ;;
        f32x)
            suite_runs_f32x 1 ;;
        v4night)
            local s
            for s in $NIGHT; do suite_runs "$s"; done ;;
        v4day)
            local s
            for s in $DAY; do suite_runs "$s"; done ;;
        v4kern)
            # C796: the three suites' runs, then the three switches on together over every case of the three (each
            # product on the kernel of the branch that owns it), at the preset's settings, without the prefetch, the tiles
            # wherever they fit and two row blocks a round, R = 2 at 2-4 tokens, without the preset; then each branch's
            # fault with all three on, which must fail cases
            local s KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0" ALL="$DMT|$Q6MT|$RO8" r
            local ON="${P}_DMT=1 ${P}_DMT_KT=1 ${P}_Q6K_MT=1 ${P}_RO8=1 ${P}_Q5K_MT=1"
            for s in $KERN; do suite_runs "$s"; done
            for r in "kern-all:$KP $ON" "kern-all-nopf:$KP ${P}_DMT=2 ${P}_DMT_KT=1 ${P}_Q6K_MT=2 ${P}_RO8=2 ${P}_Q5K_MT=2" \
                     "kern-all-kt2:$KP ${P}_DMT=1 ${P}_DMT_KT=2 ${P}_Q6K_MT=1 ${P}_RO8=1 ${P}_Q5K_MT=1" \
                     "kern-all-m2:$KP $ON ${P}_DMT_KT_M=2" "kern-all-r2:$KP $ON ${P}_DMT_R=2" "kern-all-defaults:$ON"; do
                printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" MUL_MAT "$ALL"
            done
            printf '%s\t%s\t%s\t%s\tfail\t\n' kern-q6k-fault "$KP $ON ${P}_Q6K_MT_FAULT=1" MUL_MAT "$ALL" \
                kern-kt-fault "$KP $ON ${P}_DMT_KT_FAULT=1" MUL_MAT "$ALL" \
                kern-ro8-fault "$KP $ON ${P}_RO8_FAULT=1" MUL_MAT "$ALL" ;;
    esac
}
# C754's runs: <1 natively (the switch on compute 6.1) | 2 through the PTX on a newer card (the switch on any NVIDIA
# GPU)>. cuBLAS at 4-8 tokens; the switch, which sends those to mul_mat_vec_f; the fault in the products it takes must
# fail them, and without the switch it reaches nothing
suite_runs_f32x() {
    local F=GGML_CUDA_MMVF_F32_MT M=$1
    printf '%s\t%s\t%s\t%s\t%s\t\n' f32x-off "" MUL_MAT "$F32X" pass
    [ "$M" = 1 ] && printf '%s\t%s\t%s\t%s\t%s\t\n' \
        f32x-on "$F=1" MUL_MAT "$F32X" pass \
        f32x-fault "$F=1 ${F}_FAULT=1" MUL_MAT "$F32X4" fail
    printf '%s\t%s\t%s\t%s\t%s\t\n' \
        f32x-on2 "$F=2" MUL_MAT "$F32X" pass \
        f32x-fault2 "$F=2 ${F}_FAULT=1" MUL_MAT "$F32X4" fail \
        f32x-fault-off "${F}_FAULT=1" MUL_MAT "$F32X4" pass
}
# C750's runs: <the switches every run starts with ("" natively; on a newer card the kernel on and the P106-090's 5 SMs)>.
# The launch rule alone; the switch at rules 2 (the default), 1 and 0, as for 10 SMs (the one-token shapes where 3-4
# tokens take 128 lanes) and at 2 (its other 32-lane kernel); the fault in the products the 32-lane kernels take must
# fail them, and without the switch it reaches nothing
suite_runs_bf16x() {
    local F=GGML_CUDA_MMVF_PASCAL B=$1
    local E="${F}_EXACT=1" E2="${F}_EXACT=2"
    # shellcheck disable=SC2086 # echo joins the switches with single spaces
    printf '%s\t%s\t%s\t%s\t%s\t\n' \
        bf16x-off "$(echo $B)" MUL_MAT "$BFX" pass \
        bf16x-on "$(echo $B $E)" MUL_MAT "$BFX" pass \
        bf16x-on-rule1 "$(echo $B ${F}_RULE=1 $E)" MUL_MAT "$BFX" pass \
        bf16x-on-rule0 "$(echo $B ${F}_RULE=0 $E)" MUL_MAT "$BFX" pass \
        bf16x-on-nsm10 "$(echo ${B/${F}_NSM=5/} ${F}_NSM=10 $E)" MUL_MAT "$BFX" pass \
        bf16x-on2 "$(echo $B $E2)" MUL_MAT "$BFX" pass \
        bf16x-fault "$(echo $B $E ${F}_EXACT_FAULT=1)" MUL_MAT "$BFX4" fail \
        bf16x-fault2 "$(echo $B $E2 ${F}_EXACT_FAULT=1)" MUL_MAT "$BFX4" fail \
        bf16x-fault-off "$(echo $B ${F}_EXACT_FAULT=1)" MUL_MAT "$BFX4" pass
}
# the prompt rule's runs:<1 natively (mode 1 on a P106) | 2 through the PTX on a newer card (mode 2 only)>; which
# kernel each prompt case takes in each run is fasp_check --rule's last table (bench/fattn_sparse)
suite_runs_qsa_prompt() {
    local Q=GGML_CUDA_FATTN_SPARSE_PASCAL
    local M2="$Q=2 ${Q}_PROMPT=1 ${Q}_PROMPT_NQ=512"
    printf '%s\t%s\t%s\t%s\tpass\t\n' qp-stock "$Q=0" FLASH_ATTN_EXT "$QSA"
    [ "$1" = 1 ] && printf '%s\t%s\t%s\t%s\tpass\t\n' \
        qp-sparse1 "$Q=1" FLASH_ATTN_EXT "$QSA" \
        qp-prompt "$Q=1 ${Q}_PROMPT=1" FLASH_ATTN_EXT "$QSA" \
        qp-r100 "$Q=1 ${Q}_MIN_RATIO=1.0 ${Q}_PROMPT=1" FLASH_ATTN_EXT "$QSA" \
        qp-pratio "$Q=1 ${Q}_PROMPT=1 ${Q}_PROMPT_MIN_RATIO=2.5" FLASH_ATTN_EXT "$QSA" \
        qp-pnq512 "$Q=1 ${Q}_PROMPT=1 ${Q}_PROMPT_NQ=512" FLASH_ATTN_EXT "$QSA"
    printf '%s\t%s\t%s\t%s\t%s\t\n' \
        qp-m2 "$M2" FLASH_ATTN_EXT "$QSA" pass \
        qp-m2-off "$Q=2" FLASH_ATTN_EXT "$QSA" pass \
        qp-fault1 "$M2 ${Q}_FAULT=1" FLASH_ATTN_EXT "$QSA" pass \
        qp-fault2 "$M2 ${Q}_FAULT=2" FLASH_ATTN_EXT "$QSA" fail \
        qp-fault3 "$M2 ${Q}_FAULT=3" FLASH_ATTN_EXT "$QSA" fail
}
# the union lists' runs: <1 natively (mode 1 on a P106) | 2 through the PTX on a newer card (mode 2 only)>; which
# kernel each case takes in each run is fasp_check --rule's last table (bench/fattn_sparse). The m2 runs send every
# case of 2 to 512 query tokens through the union kernel (prompt cases with _PROMPT_NQ=512), at the splits the switch
# picks, at one split and at 32; fault 4 scores each token with the next token's mask row
suite_runs_qsa_union() {
    local Q=GGML_CUDA_FATTN_SPARSE_PASCAL
    local M2="$Q=2 ${Q}_PROMPT=1 ${Q}_PROMPT_NQ=512"
    local U2="$M2 ${Q}_UNION=1"
    printf '%s\t%s\t%s\t%s\tpass\t\n' qu-stock "$Q=0" FLASH_ATTN_EXT "$QSA"
    [ "$1" = 1 ] && printf '%s\t%s\t%s\t%s\tpass\t\n' \
        qu-sparse1 "$Q=1" FLASH_ATTN_EXT "$QSA" \
        qu-union "$Q=1 ${Q}_UNION=1" FLASH_ATTN_EXT "$QSA" \
        qu-uprompt "$Q=1 ${Q}_UNION=1 ${Q}_PROMPT=1" FLASH_ATTN_EXT "$QSA" \
        qu-ur05 "$Q=1 ${Q}_UNION=1 ${Q}_UNION_MIN_RATIO=0.5 ${Q}_PROMPT=1 ${Q}_PROMPT_NQ=512" FLASH_ATTN_EXT "$QSA"
    printf '%s\t%s\t%s\t%s\t%s\t\n' \
        qu-m2 "$M2" FLASH_ATTN_EXT "$QSA" pass \
        qu-m2-union "$U2" FLASH_ATTN_EXT "$QSA" pass \
        qu-m2-union-s1 "$U2 ${Q}_SPLITS=1" FLASH_ATTN_EXT "$QSA" pass \
        qu-m2-union-s32 "$U2 ${Q}_SPLITS=32" FLASH_ATTN_EXT "$QSA" pass \
        qu-fault1 "$U2 ${Q}_FAULT=1" FLASH_ATTN_EXT "$QSA" pass \
        qu-fault2 "$U2 ${Q}_FAULT=2" FLASH_ATTN_EXT "$QSA" fail \
        qu-fault3 "$U2 ${Q}_FAULT=3" FLASH_ATTN_EXT "$QSA" fail \
        qu-fault4 "$U2 ${Q}_FAULT=4" FLASH_ATTN_EXT "$QSA" fail
}
# docs/data/2026-11-04_n2p_flash/run/gpu_checks.sh's runs: <the token counts> <the products' type field>; natively
# every token count and the prompt batches, through the PTX 1-8 tokens of the products (MMQ takes the host's tile
# configuration there, not Pascal's)
suite_runs_n2p() {
    local NT=$1 TA=$2 r
    local NP="${TA}$N4,.*$NT|type=$N4," RH="${TA}$HW,.*$NT|type=$HW,"
    for r in "np:" "np-mtok-off:${P}_MTOK=0" "np-cap1:${P}_MTOK=1,1" "np-rb1:${P}_MTOK=1,0,1" \
             "np-notile:${P}_MTOK=1,0,0,0,0" "np-skip:${P}_SKIP=n2p,n3p,n2p_128,n3p_128"; do
        printf '%s\t%s\t%s\t%s\tpass\t\n' "${r%%:*}" "${r#*:}" "MUL_MAT,$EXPERT_OPS,GET_ROWS" "$NP"
    done
    [ "$NT" = 'n=[1-8],' ] || printf '%s\t%s\t%s\t%s\tpass\t\n' np-mmq "" MUL_MAT,MUL_MAT_ID "type_a=$N4,.*n=(9|33|128),"
    printf '%s\t%s\t%s\t%s\tpass\t\n' reg-half "" "MUL_MAT,$EXPERT_OPS" "$RH" reg-rows "" GET_ROWS '.'
}
# the suite's runs for a newer card through the compute_61 PTX, when they differ from the native ones
suite_runs_ptx() {
    case $1 in
        n2p) suite_runs_n2p 'n=[1-8],' "type_a=" ;;
        qsa-prompt) suite_runs_qsa_prompt 2 ;;
        qsa-union) suite_runs_qsa_union 2 ;;
        bf16x) suite_runs_bf16x "GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVF_PASCAL_NSM=5" ;;
        f32x) suite_runs_f32x 2 ;;
        v4night)
            local s
            for s in $NIGHT; do
                if [ -n "$(suite_runs_ptx "$s")" ]; then suite_runs_ptx "$s"; else suite_runs "$s"; fi
            done ;;
        v4day)
            local s
            for s in $DAY; do
                if [ -n "$(suite_runs_ptx "$s")" ]; then suite_runs_ptx "$s"; else suite_runs "$s"; fi
            done ;;
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

write_runs() {
    # empty fields as "-": read with a tab IFS would merge two tabs
    dash() { awk -F'\t' -v OFS='\t' '{ for (i = 1; i <= NF; i++) if ($i == "") $i = "-"; print }'; }
    suite_runs "$SUITE" | dash > "$S/runs.tsv"
    suite_runs_ptx "$SUITE" | dash > "$S/runs-ptx.tsv"
    [ -s "$S/runs-ptx.tsv" ] || rm -f "$S/runs-ptx.tsv"
}

if step runs && ! step record; then
    [ -f "$S/manifest.tsv" ] || { echo "no recorded set in $S" >&2; exit 1; }
    write_runs
    log "runs: $(wc -l < "$S/runs.tsv") runs written into $S/runs.tsv"
fi

if step record; then
    BIN=$(realpath "${BIN:?BIN=<bin/ of the rig tarball>}")
    [ -x "$BIN/test-backend-ops" ] || { echo "no test-backend-ops in $BIN" >&2; exit 1; }
    mkdir -p "$S"
    write_runs
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
        for L in $LINK_FROM; do
            [ -f "$L/manifest.tsv" ] || { echo "no manifest.tsv in $L" >&2; exit 1; }
        done
        mkdir -p "$S/cases"
        while IFS=$'\t' read -r ops params; do
            tbo -b CPU -o "$ops" -p "$(flt "$params")" --record-dry 2> /dev/null \
                | sed -n 's/^  \(.*\): in [0-9]* B, ref [0-9]* B$/\1/p'
        done < <(suite_filters "$SUITE") > "$S/selected.txt"
        : > "$S/linked.txt"
        for L in $LINK_FROM; do
            awk -F'\t' 'NR == FNR { sel[$0] = 1; next } ($2 in sel) { print $1 }' "$S/selected.txt" "$L/manifest.tsv" > "$S/link1.txt"
            n=0
            while read -r id; do
                [ -e "$S/cases/$id.meta" ] && continue
                for e in in ref meta; do ln -f "$L/cases/$id.$e" "$S/cases/$id.$e"; done
                echo "$id" >> "$S/linked.txt"
                n=$((n + 1))
            done < "$S/link1.txt"
            rm -f "$S/link1.txt"
            log "linked $n of $(wc -l < "$S/selected.txt") selected cases from $L"
        done
        log "linked $(wc -l < "$S/linked.txt") of $(wc -l < "$S/selected.txt") selected cases in all"
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
