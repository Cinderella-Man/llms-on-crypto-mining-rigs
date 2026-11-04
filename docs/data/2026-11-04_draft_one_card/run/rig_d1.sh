#!/usr/bin/env bash
# rig_d1.sh OUT: Q961 on a P106 rig. The MTP draft's passes on one card: its token rows from a row cache on its
# layer's card (LLAMA_DRAFT_EMBD_CACHE) and its hidden rows kept there between a step's passes (LLAMA_DRAFT_H_CARRY),
# fork branch pascal-p106-v3-b12076-draft1card, against the same build without them, on S35's line: pascal-mix's
# pmix-q2-fast, the k-column setting, the Q8_0 hop cast, the Gyro MTP draft at n-max 3 with S15's
# -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host and the shortlist head. Arms:
#   d1off    S35's line on this build, no new switch (the control)
#   d1cache  + LLAMA_DRAFT_EMBD_CACHE=8192 (21 MiB on card 8; every draft pass and catch-up on card 8 alone)
#   d1carry  + LLAMA_DRAFT_H_CARRY=1 (the drafting passes' 40 KB hidden rows stay on card 8)
#   d1on     both
#   d1share  + LLAMA_DRAFT_SHARE=embd,head (S31: the draft reads the target's host embeddings and its head; the pass's
#            row comes from the host, other numbers: the embeddings are Q3_K-class, not the draft's Q8_0)
# Steps (STEPS, default all):
#   tarball  the build from the build machine to the rig through this machine (BUILD_SSH, BUILD_DIR), md5 against MD5
#   profile  per arm one load through docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh (the RAM watchdog,
#            floor 60) with the fork's host profile GGML_CUDA_PROFILE_HOST, untimed: the agent's first-turn request
#            (~2,780 tokens) and S12's smoke request; then run/draft_passes.py (each draft pass: the crossing, the GPU
#            time, the host's wait) and the C338 split (docs/data/2026-11-04_verify_host/run/host_steps.py) on the build
#            machine, never on the rig's CPU
#   prep     the agent check's client folder (release_check_s12.sh STEPS=prep)
#   arms     the agent check (12 tasks, 40 minutes a budget) per arm in AGENT_ARMS (default d1off d1on)
#   report   the agent check's report over AGENT_ARMS
# The profile decides: the step's median time and the crossings' sum over ~200+ steps per arm; the agent check shows
# the speed over a session and that the acceptance is unchanged (d1on drafts the same tokens as d1off).
# Environment (machine-specific values; keep them in a private file): RIG, RIG_LOGDIR (a folder on the rig), BENCH
# (bench/agent_check), BUILDS (/opt/llama-builds on the rig), T (the build's folder name), MD5 (its tarball's md5),
# BUILD_SSH (ssh command of the build machine), BUILD_DIR (the tarball's folder there), WORK (a folder there for the
# split), MODEL (pmix-q2-fast's first shard on the rig), DRAFT (the MTP draft on the rig), RIG_PORT, RELAY_PORT,
# CLIENT, CLIENT_PORT, PROXY_PORT, CLIENT_DIR, CLIENT_OUT (release_check_s12.sh's), optional PROFILE_ARMS (d1off
# d1cache d1carry d1on d1share), AGENT_ARMS (d1off d1on), KENV (the k-column setting, default C350's SMEM=32768 TABLE=0).
# Run from the repository root; prints D1_STEP <step> done and D1_DONE.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: rig_d1.sh OUT}")
: "${RIG:?}" "${RIG_LOGDIR:?}" "${BENCH:?}" "${BUILDS:?}" "${T:?}" "${MODEL:?}" "${DRAFT:?}" "${RIG_PORT:?}"
mkdir -p "$OUT"
STEPS=${STEPS:-tarball profile prep arms report}
has() { [[ " $STEPS " == *" $1 "* ]]; }
D=docs/data/2026-11-04_draft_one_card/run
KENV=${KENV:-"GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"}
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
BASE="$KENV $HOP $SL"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
export REL_BIN=$BUILDS/$T/bin/llama-server MIN_AVAIL_MB=60 MODEL DRAFT
arm_env() {
  case $1 in
    d1off)   echo "$BASE" ;;
    d1cache) echo "$BASE LLAMA_DRAFT_EMBD_CACHE=8192" ;;
    d1carry) echo "$BASE LLAMA_DRAFT_H_CARRY=1" ;;
    d1on)    echo "$BASE LLAMA_DRAFT_EMBD_CACHE=8192 LLAMA_DRAFT_H_CARRY=1" ;;
    d1share) echo "$BASE LLAMA_DRAFT_SHARE=embd,head" ;;
    *)       echo "unknown arm $1" >&2; return 1 ;;
  esac
}

if has tarball; then
  : "${BUILD_SSH:?}" "${BUILD_DIR:?}" "${MD5:?}"
  if ssh "$RIG" "[ -x $BUILDS/$T/bin/llama-server ]"; then
    echo "tarball: $T already on the rig"
  else
    $BUILD_SSH "cat $BUILD_DIR/$T.tar.gz" | ssh "$RIG" "cat > $BUILDS/$T.tar.gz && cd $BUILDS && md5sum $T.tar.gz" | tee "$OUT/tarball_md5.txt"
    grep -q "^$MD5 " "$OUT/tarball_md5.txt" || { echo "tarball: md5 differs from $MD5; stopping"; ssh "$RIG" "rm -f $BUILDS/$T.tar.gz"; exit 1; }
    ssh "$RIG" "cd $BUILDS && tar xzf $T.tar.gz && rm $T.tar.gz && ls $T/bin/llama-server"
  fi
  echo "D1_STEP tarball done $(date -Is)"
fi

if has profile; then
  for a in ${PROFILE_ARMS:-d1off d1cache d1carry d1on d1share}; do
    e=$(arm_env "$a") || exit 1
    { date -Is; ssh "$RIG" 'ps -eo comm | grep -x -E "llama-server|llama-bench" || echo "no llama process"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; grep -E "MemAvailable|SwapFree" /proc/meminfo'; } > "$OUT/quiet_profile_$a.txt" 2>&1
    LD=$RIG_LOGDIR/profile-$a
    REQUESTS="$D/greedy_request.json docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json" ARGS="$MT" \
      SMOKE_ENV="$e GGML_CUDA_PROFILE_HOST=$LD/host_profile.csv" SMOKE_LOGDIR=$LD \
      bash docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh "$OUT/profile-$a" > "$OUT/profile-$a.console" 2>&1
    echo "profile $a: exit $? $(date -Is)"; tail -n 3 "$OUT/profile-$a.console" | cut -c 1-200
    grep -h 'embedding row cache\|hidden row\|reads the target\|draft head shortlist' "$OUT/profile-$a/"server*.log 2>/dev/null | cut -c 1-200 | head -6
    for r in reply_long.json reply_1.json reply_2.json; do
      f=$OUT/profile-$a/$r; [ -s "$f" ] || continue
      echo "  $r: $(grep -o '"draft_n":[0-9]*\|"draft_n_accepted":[0-9]*\|"predicted_n":[0-9]*\|"predicted_per_second":[0-9.]*' "$f" | tr '\n' ' ')"
    done
    ssh "$RIG" "rm -rf $LD"
    if [ -s "$OUT/profile-$a/host_profile.csv" ] && [ -n "${BUILD_SSH:-}" ] && [ -n "${WORK:-}" ]; then
      $BUILD_SSH "mkdir -p $WORK/rig/$a $WORK/run"
      $BUILD_SSH "cat > $WORK/run/draft_passes.py" < $D/draft_passes.py
      $BUILD_SSH "cat > $WORK/run/host_steps.py" < docs/data/2026-11-04_verify_host/run/host_steps.py
      $BUILD_SSH "cat > $WORK/rig/$a/host_profile.csv" < "$OUT/profile-$a/host_profile.csv"
      $BUILD_SSH "cd $WORK && nice -n 19 python3 -I run/draft_passes.py rig/$a/host_profile.csv" > "$OUT/passes-$a.md"
      $BUILD_SSH "cd $WORK && nice -n 19 python3 -I run/host_steps.py rig/$a/host_profile.csv" > "$OUT/steps-$a.md"
      echo "  split: $(grep -E '^\| (step_ms|lead_sum_ms|idle_sum_drafting_ms) ' "$OUT/passes-$a.md" | tr -s ' ' | tr '\n' ' ')"
    fi
  done
  # the same drafts on the cards: the greedy request's text and draft counts of each switch arm against d1off's, on
  # the build machine
  if [ -n "${BUILD_SSH:-}" ] && [ -n "${WORK:-}" ] && [ -s "$OUT/profile-d1off/reply_1.json" ]; then
    $BUILD_SSH "mkdir -p $WORK/run && cat > $WORK/run/identity.py" < $D/identity.py
    files=""
    for a in d1off d1cache d1carry d1on; do
      [ -s "$OUT/profile-$a/reply_1.json" ] || continue
      $BUILD_SSH "mkdir -p $WORK/rig/$a && cat > $WORK/rig/$a/greedy_reply.json" < "$OUT/profile-$a/reply_1.json"
      files="$files rig/$a/greedy_reply.json"
    done
    $BUILD_SSH "cd $WORK && python3 -I run/identity.py $files" | tee "$OUT/identity.txt"
  fi
  echo "D1_STEP profile done $(date -Is)"
fi

RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
if has prep; then
  ARMS=d1off STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
  echo "prep: exit $? $(date -Is)"; tail -n 3 "$OUT/agent-prep.console" | cut -c 1-160
  echo "D1_STEP prep done $(date -Is)"
fi
if has arms; then
  for a in ${AGENT_ARMS:-d1off d1on}; do
    e=$(arm_env "$a") || exit 1
    { date -Is; ssh "$RIG" 'ps -eo comm | grep -x -E "llama-server|llama-bench" || echo "no llama process"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; grep -E "MemAvailable|SwapFree|Shmem:" /proc/meminfo; cat /proc/pressure/memory'; } > "$OUT/quiet_check_$a.txt" 2>&1
    env "ARGS_$a=$MT" "BUDGET_MIN_$a=40" REL_ENV="$e" ARMS=$a STEPS=run bash $RC "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
    echo "arm $a: release_check exit $? $(date -Is)"; tail -n 2 "$OUT/agent-$a/run.log" | cut -c 1-160
    ssh "$RIG" "ps -eo comm | grep -x llama-server && echo 'SERVER STILL UP' || echo 'no llama-server'"
  done
  echo "D1_STEP arms done $(date -Is)"
fi
if has report; then
  ARMS="${AGENT_ARMS:-d1off d1on}" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
  echo "report: exit $?"; sed -n '/^### Per arm/,/^### Time/p' "$OUT/agent-report.console" | head -8
fi
date -Is; echo D1_DONE
