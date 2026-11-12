#!/usr/bin/env bash
# R72b (Q1261) on one rig: the agent check of v4's candidate line `v4c2` on the C802 build (pascal-p106-v4 8a55934f3,
# which sits on the kernel merge d097d6d88). This is run/r72a.sh with the arm v4c2 added: the same prep, agent arm and
# report, the report over REPORT_ARMS, so that R72's arms written into the same client folder (CLIENT_OUT) are reported
# side by side. Flash-Next's v3 preset (release/flash-next-v3/settings.env) with Swift Q2_0 and the MTP draft. Run from
# the repository root on a machine that only runs ssh. The steps run in this order, whatever STEPS lists:
#   prep    the agent check's bench files and sandbox image on the client, the bugfix tasks validated
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
# Env: RIG, CLIENT (ssh aliases), MODEL (Swift Q2_0's first shard on the rig), DRAFT (the MTP draft on the rig), NEW, REF
#   (build folders under /opt/llama-builds; REF is unused unless AGENT_ORDER names v3), RIGDIR (a folder on the rig for
#   the agent check's logs), OUT (a local folder), STEPS ("prep agent report"), AGENT_ORDER ("v4c2"), REPORT_ARMS
#   (AGENT_ORDER), AGENT_BY (21:00: one arm of at most 40 minutes plus a 12-minute task and the load ends by 22:00),
#   and release_check_s12.sh's RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT CPU_SHARES (256).
set -uo pipefail
: "${RIG:?}" "${CLIENT:?}" "${MODEL:?}" "${DRAFT:?}" "${NEW:?}" "${REF:?}" "${OUT:?}"
STEPS=${STEPS:-prep agent report}
AGENT_ORDER=${AGENT_ORDER:-v4c2}
REPORT_ARMS=${REPORT_ARMS:-$AGENT_ORDER}
AGENT_BY=${AGENT_BY:-21:00}
GAINS_ENV=${GAINS_ENV:-GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1}
GAINS_ARGS=${GAINS_ARGS:---spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on}
V4C2_ENV=${V4C2_ENV:-LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 LLAMA_SPEC_REJECT_STOP=1 LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1}
V4C2_TAPER=${V4C2_TAPER:-short=600:64}
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
export BENCH=$PWD/bench/agent_check RIG CLIENT MODEL CTX=65536 TIME_CAP_MIN=12 MIN_AVAIL_MB=60 \
  CPU_SHARES=${CPU_SHARES:-256} RIG_LOGDIR=${RIGDIR:-/tmp/r72b}/agent-logs
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh

if has prep; then
  say "prep"
  ARMS=gains STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1 || { say "prep failed"; exit 1; }
  say "prep: $(tail -n 1 "$OUT/agent-prep.console")"
fi

if has agent; then
  if [ "$(date +%H%M)" -gt "${AGENT_BY/:/}" ]; then
    say "AGENT_SKIPPED: past $AGENT_BY, the arms would not end before the rig's evening stop"
  else
    for a in $AGENT_ORDER; do
      case $a in
        v4c2) b=$NEW; e="$PRESET_ENV $GAINS_ENV $V4C2_ENV"; args=$MT_V4C2 ;;
        gains) b=$NEW; e="$PRESET_ENV $GAINS_ENV"; args=$MT_GAINS ;;
        v3) b=$REF; e=$PRESET_ENV; args=$MT ;;
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

ssh "$RIG" "rmdir ${RIGDIR:-/tmp/r72b} 2> /dev/null; date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'; \
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; \
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo" > "$OUT/rig_end.txt" 2>&1
say "R72B_DONE: $(tr '\n' ' ' < "$OUT/rig_end.txt")"
