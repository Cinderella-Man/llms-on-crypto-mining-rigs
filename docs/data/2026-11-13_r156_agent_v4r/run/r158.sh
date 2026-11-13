#!/usr/bin/env bash
# R158 (Q1476) on one rig: a second v3.0 run of the agent check, arm v3b = R90's v3 arm exactly (REF, the v3.0 release
# build, with the preset as shipped: Swift Q2_0, the MTP draft, rig_t1.sh's server line), as the same-day baseline for
# v4r's two runs. This is R156's driver below with the arm v3b added beside v3; set REF to the v3.0 build folder and
# AGENT_ORDER=v3b. Its results go into R72's client folder like R156's (REPORT_ARMS names every arm there that the
# tables read, plus v3b). R156's header follows; its arms stay as they were.
# R156 (Q1474) on one rig: Flash-Next's final v4 settings v4r on the release candidate v4-rc2 (pascal-p106-v4
# eb82916d3, upstream b12360), on the agent check. v4r is v4c3 (R142a, R142b) without C812's LLAMA_PP_MULTI_SPLIT=1
# (R141: a loss), with LLAMA_SCHED_PER_SIZE_RESERVE=2 in place of =1 (R143), with C755/C828's
# GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0 (R150, R155); the draft's micro-batch stays v4c3's 512 (V4R_DRAFT_UB, the
# flag upstream #32078 needs for the MTP draft; 512 reads prompts faster than 128, R53); the union kernel
# stays off (R88: a loss). This is R142's driver (docs/data/2026-11-13_r142_agent_v4c3/run/r142.sh) with the arms v4r
# and v4r2 added; one arm a run (R156a, R156b), its results written into R72's client folder (CLIENT_OUT) so that the
# report puts it beside R72's, R142's and R154's arms (REPORT_ARMS).
# R142's header follows; its arms stay as they were.
# R142 (Q1446) on one rig: v4's candidate line v4c3 on the agent check, on C825's tarball of pascal-p106-v4's tip
# 65ac9a47d. v4c3 is v4c2 (R72b, R72d) with six switches whose P106 checks passed added; the build, the model (Swift
# Q2_0), the draft, -b 4096 and the rest of the command are v4c2's. This is the R72 driver (docs/data/
# 2026-11-12_r72_agent_v4c2/run/r72d.sh) with the arm v4c3 and the step files added; one arm a run (R142a, R142b), its
# results written into R72's client folder (CLIENT_OUT) so that the report puts it beside R72's arms (REPORT_ARMS).
# Flash-Next's v3 preset (release/flash-next-v3/settings.env) with Swift Q2_0 and the MTP draft. Run from the repository
# root on a machine that only runs ssh. The steps run in this order, whatever STEPS lists:
#   prep    the agent check's bench files and sandbox image on the client, the bugfix tasks validated
#   files   only prep's first part: the bench files copied to the client (CLIENT_DIR), with scripts/realloc_counts.py, so
#           that a client prepared before C804 step (a) reports each arm's graph reallocations; the sandbox image and
#           the validated tasks stay as they are
#   agent   the agent check (bench/agent_check, 12 bugfix tasks, a 12-minute cap a task, a 40-minute budget an arm), one
#           arm per build in AGENT_ORDER, each on the server line of v3.0's arm t1swift
#           (docs/data/2026-11-05_release_flashnext/run/rig_t1.sh) with the arm's switches added; it starts only when
#           the clock is not past AGENT_BY, so that the arms end before the rig's evening stop
#   report  the agent check's report of REPORT_ARMS; the client's folder collected to OUT/agent
# Arms:
#   v3      REF with the preset as shipped
#   gains   NEW with the preset, GAINS_ENV and GAINS_ARGS: R70's candidate (the sparse Pascal attention, probabilistic
#           draft verification, four checkpoints on the cards, the micro-batch taper, the draft's micro-batch of 512) and
#           the six switches proved on 2026-11-11 (C722, C690, C691, C704, C100, C217), as R90 ran it
#   v4c2    NEW with the preset, GAINS_ENV, V4C2_ENV and V4C2_ARGS: the gains line with R60's taper
#           (--ubatch-taper short=600:64 in place of on), C691's LLAMA_SAMPLING_TOP_K_ROWS=1 kept (C802 fixes its node
#           count, R127), and the switches 2026-11-12's rig rows kept:
#             C720 and C719's k-pool staging and pool mask on the cards  LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1
#             R122's eight host switches                LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1
#                                                       LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1
#                                                       LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1
#             R123a's three                             GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1
#             R120b's three verify-pass kernels         GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1
#                                                       GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1
#             FN66's stop on a rejected draft           LLAMA_SPEC_REJECT_STOP=1
#             R123b's three                             LLAMA_DRAFT_EMBD_DEV=layer (the draft's token table on its layer's
#                                                       card, with -otd token_embd=CUDA4 taken out of the line),
#                                                       GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1, and -b 4096 in place of
#                                                       -b 2048
#   v4c22   v4c2's line exactly, a second run of it (R72d)
#   v4c3    v4c2's line with V4C3_ENV added, each switch passed on a P106 rig on 2026-11-12 or 2026-11-13:
#             C803's reserve per graph size (R130)   LLAMA_SCHED_PER_SIZE_RESERVE=1
#             C807's f32 matrix-vector kernel (R132) GGML_CUDA_MMVF_F32_MT=1
#             C812's prompt split (R135)             LLAMA_PP_MULTI_SPLIT=1
#             C811 with C817's fix (R140)            GGML_CUDA_KEEPALIVE_EAGER=1 (the timeout stays at its default)
#             C813's lazy hidden row (R139)          LLAMA_MTP_HROW_LAZY=1
#             C805's exact k-columns (R126)          GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1
#   v4c32   v4c3's line exactly, a second run of it (R142b)
#   v4r     NEW (v4-rc2) with v4c3's line and these differences, each from a rig row of 2026-11-13:
#             R141: C812's prompt split out          LLAMA_PP_MULTI_SPLIT=1 removed
#             R143: the shared scheduler re-planned  LLAMA_SCHED_PER_SIZE_RESERVE=2 in place of =1
#             R150, R155: C755's skip at 1-4 tokens  GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0 added
#             the draft's micro-batch               --spec-draft-n-ubatch V4R_DRAFT_UB (512, v4c3's)
#   v4r2    v4r's line exactly, a second run of it (R156b)
# Env: RIG, CLIENT (ssh aliases), MODEL (Swift Q2_0's first shard on the rig), DRAFT (the MTP draft on the rig), NEW, REF
#   (build folders under /opt/llama-builds; REF is unused unless AGENT_ORDER names v3), RIGDIR (a folder on the rig for
#   the agent check's logs), OUT (a local folder), STEPS ("prep agent report"), AGENT_ORDER ("v4r"),
#   V4R_DRAFT_UB (v4r's --spec-draft-n-ubatch), REPORT_ARMS
#   (AGENT_ORDER), AGENT_BY (21:00: one arm of at most 40 minutes plus a 12-minute task and the load ends by 22:00),
#   and release_check_s12.sh's RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT CPU_SHARES (256).
set -uo pipefail
: "${RIG:?}" "${CLIENT:?}" "${MODEL:?}" "${DRAFT:?}" "${NEW:?}" "${REF:?}" "${OUT:?}"
STEPS=${STEPS:-prep agent report}
AGENT_ORDER=${AGENT_ORDER:-v4r}
REPORT_ARMS=${REPORT_ARMS:-$AGENT_ORDER}
AGENT_BY=${AGENT_BY:-21:00}
GAINS_ENV=${GAINS_ENV:-GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1}
GAINS_ARGS=${GAINS_ARGS:---spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on}
V4C2_ENV=${V4C2_ENV:-LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 LLAMA_SPEC_REJECT_STOP=1 LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1}
V4C2_TAPER=${V4C2_TAPER:-short=600:64}
V4C3_ENV=${V4C3_ENV:-LLAMA_SCHED_PER_SIZE_RESERVE=1 GGML_CUDA_MMVF_F32_MT=1 LLAMA_PP_MULTI_SPLIT=1 GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_MTP_HROW_LAZY=1 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1}
V4R_ENV=${V4R_ENV:-LLAMA_SCHED_PER_SIZE_RESERVE=2 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_MTP_HROW_LAZY=1 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0}
V4R_DRAFT_UB=${V4R_DRAFT_UB:-512}
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
# shellcheck disable=SC1091
. release/flash-next-v3/settings.env  # PRESET_ENV, PRESET_ARGS
mkdir -p "$OUT"

# rig_t1.sh's MT: the preset's server line after its common part (release_check_s12.sh's COMMON adds that)
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
MT_GAINS="${MT/--spec-draft-n-ubatch 128/--spec-draft-n-ubatch 512} $GAINS_ARGS"
[[ $MT_GAINS == *"--spec-draft-n-ubatch 512"* ]] || { say "the draft's micro-batch was not replaced"; exit 1; }
# v4c2: the gains line with R60's taper, -b 4096 and without the draft's token table on CUDA4 (LLAMA_DRAFT_EMBD_DEV)
MT_V4C2=${MT_GAINS/--ubatch-taper on/--ubatch-taper $V4C2_TAPER}
MT_V4C2=${MT_V4C2/-b 2048 /-b 4096 }
MT_V4C2=${MT_V4C2/ -otd token_embd=CUDA4/}
[[ $MT_V4C2 == *"--ubatch-taper $V4C2_TAPER"* && $MT_V4C2 == "-b 4096 "* && $MT_V4C2 != *token_embd* ]] \
  || { say "v4c2's line was not built: $MT_V4C2"; exit 1; }
# v4r: v4c2's line with the draft's micro-batch V4R_DRAFT_UB (512: the line unchanged)
MT_V4R=${MT_V4C2/--spec-draft-n-ubatch 512 /--spec-draft-n-ubatch $V4R_DRAFT_UB }
[[ $MT_V4R == *"--spec-draft-n-ubatch $V4R_DRAFT_UB "* ]] || { say "v4r's line was not built: $MT_V4R"; exit 1; }
export BENCH=$PWD/bench/agent_check RIG CLIENT MODEL CTX=65536 TIME_CAP_MIN=12 MIN_AVAIL_MB=60 \
  CPU_SHARES=${CPU_SHARES:-256} RIG_LOGDIR=${RIGDIR:-/tmp/r156}/agent-logs
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh

if has prep; then
  say "prep"
  ARMS=gains STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1 || { say "prep failed"; exit 1; }
  say "prep: $(tail -n 1 "$OUT/agent-prep.console")"
elif has files; then
  : "${CLIENT_DIR:?}"
  say "files: bench files to $CLIENT:$CLIENT_DIR"
  tar -C bench --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py -C ../scripts realloc_counts.py \
    | ssh "$CLIENT" "mkdir -p $CLIENT_DIR && tar -C $CLIENT_DIR -xf -" || { say "files: copy failed"; exit 1; }
fi

if has agent; then
  if [ "$(date +%H%M)" -gt "${AGENT_BY/:/}" ]; then
    say "AGENT_SKIPPED: past $AGENT_BY, the arms would not end before the rig's evening stop"
  else
    for a in $AGENT_ORDER; do
      case $a in
        v4c2|v4c22) b=$NEW; e="$PRESET_ENV $GAINS_ENV $V4C2_ENV"; args=$MT_V4C2 ;;
        v4c3|v4c32) b=$NEW; e="$PRESET_ENV $GAINS_ENV $V4C2_ENV $V4C3_ENV"; args=$MT_V4C2 ;;
        v4r|v4r2) b=$NEW; e="$PRESET_ENV $GAINS_ENV $V4C2_ENV $V4R_ENV"; args=$MT_V4R ;;
        gains) b=$NEW; e="$PRESET_ENV $GAINS_ENV"; args=$MT_GAINS ;;
        v3|v3b) b=$REF; e=$PRESET_ENV; args=$MT ;;
        *) say "unknown arm $a"; exit 1 ;;
      esac
      say "arm $a: $b"
      env REL_BIN="/opt/llama-builds/$b/bin/llama-server" "ARGS_$a=$args" "BUDGET_MIN_$a=40" REL_ENV="$e" \
        ARMS="$a" STEPS=run bash $RC "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
      say "arm $a: exit $?"
    done
    if has report; then
      say "report: $REPORT_ARMS"
      ARMS="$REPORT_ARMS" STEPS="report collect" bash $RC "$OUT/agent" > "$OUT/agent-report.console" 2>&1
      say "report: exit $?"
    fi
  fi
fi

ssh "$RIG" "rmdir ${RIGDIR:-/tmp/r156} 2> /dev/null; date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'; \
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; \
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo" > "$OUT/rig_end.txt" 2>&1
say "R158_DONE:$(tr '\n' ' ' < "$OUT/rig_end.txt")"
