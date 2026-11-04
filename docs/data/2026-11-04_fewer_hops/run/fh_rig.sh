#!/usr/bin/env bash
# fh_rig.sh (Q964): fewer card boundaries on one rig of nine P106-090 cards. pmix-q2-fast on Miner Llama's -kshape build
# at S35's agent line. Each card boundary is a host round trip per pass over PCIe 1.0 x1; plan.sh found that the file
# holds its 48 layers on 8 cards with whole layers (7 hops a pass instead of 8): the fullest card keeps 416 MiB with no
# draft, and 379 MiB with the MTP draft's whole pass (its layer, head, cache and token embeddings, 2,781 MiB) on the
# freed card. Those figures are split_plan.py's sums from the headers, within 2 MiB a card of llama.cpp's projection on
# S34's and S35's nine-card splits. Steps (STEPS, default "probe load tk prep arms report"):
#   probe   scripts/fit_probe.sh, llama.cpp's projection with no weights loaded (seconds a line): nd8 and own8 (the arms'
#           lines), nd9u256 and nd7u256 (-ub 256: its compute buffers, and the 7-card split it may allow, est. 317 MiB
#           free); with PROBE_FILES ("label=<first shard> ...", e.g. header_copies.sh's copies unpacked on the rig) nd8
#           and own8 for each of those files too
#   load    one real load per line (nd8, own8) at 64k with the arm's switches and the RAM watchdog: the server up, one
#           short request (with the draft), nvidia-smi's used and free MiB per card, the server's buffer lines
#   tk      T(k) on real text, k = 1-8 (docs/data/2026-11-04_kreal_p106/run/kreal.sh, its arm s32kt0 = KENV, with HOP in
#           the environment): S35's nine-card split, then the 8-card plan (its -dev goes in through kreal.sh's NOPIPE)
#   prep, arms, report   the agent check (release_check_s12.sh) in S35's client folder, so the report pairs every arm
#           with S35's s35stack (12/12, 18.8 tok/s):
#             fh8own  s35stack exactly (KENV HOP SL, the MTP line) with the main model on 8 cards (-dev DEV8 -ts TS8)
#                     and the draft on the freed card (-devd FREE -otd token_embd=FREE instead of -otd token_embd=CUDA4):
#                     7 hops a pass, and each draft pass on one card (Q961)
#             fh8nd   no draft, 8 cards (the freed card idle), KENV HOP
#             fh9nd   no draft, S34's nine-card split, KENV HOP (fh8nd's control)
#             s35ctl  s35stack exactly (the control when BUILD or the rig differ from S35's)
#           ARMS_FH (default fh8own) picks them, in order; 40 minutes each.
# Run from the repository root (it runs only ssh here). ENVFILE is S12's machine file
# (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in, with DRAFT), with RIG, RIG_PORT, RELAY_PORT,
# CLIENT_PORT, PROXY_PORT, CLIENT_DIR, CLIENT_OUT and RIG_LOGDIR set after its other values (S35's ports and client
# folder, to pair with its arm); MODEL the file's
# first shard on the rig; BUILD the build's folder under BUILDS (default /opt/llama-builds); TEXT the agent text on the
# rig (kreal_p106's: the first 20,000 bytes of the agent text, md5 312fd205dec2fba4879d4bf421f1e80e); RDIR a work folder
# on the rig; OUT a local folder. The plan: DEV8 (CUDA0,...,CUDA7), TS8 (6,6,6,6,6,6,6,7), FREE (CUDA8), TS7
# (7,7,7,7,7,7,7), TS9 (S35's 6,6,6,6,5,6,5,6,3), TSND9 (S34's 6,5,5,6,5,5,6,5,6); another file or card count takes its
# values from plan.sh. Nothing runs on the rig's CPU beyond llama-server, its requests and the watchdog.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
set -a; . "$ENVFILE"; set +a
: "${RIG:?ENVFILE sets RIG, the ssh alias of the rig}"; BUILDS=${BUILDS:-/opt/llama-builds}; RDIR=${RDIR:-/tmp/fh}; LOAD_PORT=${LOAD_PORT:-8097}
export RIG MIN_AVAIL_MB=60 MODEL=$M REL_BIN=$BUILDS/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-$RDIR/agent-logs}
STEPS=${STEPS:-probe load tk prep arms report}
has() { [[ " $STEPS " == *" $1 "* ]]; }
DEV8=${DEV8:-CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,CUDA7}; TS8=${TS8:-6,6,6,6,6,6,6,7}; FREE=${FREE:-CUDA8}
TS7=${TS7:-7,7,7,7,7,7,7}; TS9=${TS9:-6,6,6,6,5,6,5,6,3}; TSND9=${TSND9:-6,5,5,6,5,5,6,5,6}
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
DR="-md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3"
MT="-b 2048 -ub 512 -ts $TS9 -fit off -lm dio $DR -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
OWN8="-b 2048 -ub 512 -dev $DEV8 -ts $TS8 -fit off -lm dio $DR -devd $FREE -otd token_embd=$FREE --spec-draft-n-ubatch 128 --no-host"
ND8="-b 2048 -ub 512 -dev $DEV8 -ts $TS8 -fit off -lm dio"
ND9="-b 2048 -ub 512 -ts $TSND9 -fit off -lm dio"
COMMON="-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
nofit() { echo "$1" | sed 's/ -fit off -lm dio//'; }
mkdir -p "$OUT"

if has probe; then
  RIG=$RIG MODEL=$M BUILDS=$BUILDS OUT=$OUT/fit_probe.md KEEP=400 CONFIGS="nd8|$T|65536|$(nofit "$ND8");own8|$T|65536|\
$(nofit "$OWN8");nd9u256|$T|65536|$(nofit "${ND9/-ub 512/-ub 256}");nd7u256|$T|65536|-b 2048 -ub 256 -dev ${DEV8%,*} -ts $TS7" \
    bash scripts/fit_probe.sh > /dev/null
  for spec in ${PROBE_FILES:-}; do
    RIG=$RIG MODEL=${spec#*=} BUILDS=$BUILDS OUT=$OUT/fit_probe_${spec%%=*}.md KEEP=400 \
      CONFIGS="nd8|$T|65536|$(nofit "$ND8");own8|$T|65536|$(nofit "$OWN8")" bash scripts/fit_probe.sh > /dev/null
  done
  grep -hE 'common_params_fit_impl:   - CUDA|projected to use|^== ' "$OUT"/fit_probe*.md | cut -c1-150
  echo "FH_STEP probe done $(date -Is)"
fi

if has load; then
  ssh "$RIG" "mkdir -p $RDIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
  load_line() {  # <label> <env> <args>: one real load, one request, the memory per card
    timeout 1800 ssh "$RIG" "pkill -x llama-server; sleep 2; cd $RDIR
      LOG=/dev/shm/rw_load.tsv PROC=llama-server MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw_load.pid
      setsid nohup env $2 $BUILDS/$T/bin/llama-server -m $M $COMMON $3 --host 127.0.0.1 --port $LOAD_PORT > load-$1.log 2>&1 < /dev/null &
      for i in \$(seq 1 300); do curl -sf http://127.0.0.1:$LOAD_PORT/health > /dev/null && break; sleep 3; done
      curl -sf -m 300 http://127.0.0.1:$LOAD_PORT/completion -d '{\"prompt\": \"def fibonacci(n):\", \"n_predict\": 64}' | head -c 400; echo
      echo '== nvidia-smi index, used, free'; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
      pkill -INT -x llama-server; sleep 8; pkill -9 -x llama-server; kill \$(cat /dev/shm/rw_load.pid)
      grep -E 'KILL' /dev/shm/rw_load.tsv; rm -f /dev/shm/rw_load.*
      echo '== server'; grep -E 'buffer size|failed|error|out of memory|draft' load-$1.log | cut -c1-200 | head -150" > "$OUT/load-$1.txt" 2>&1
    echo "load $1: $(grep -A9 '== nvidia-smi' "$OUT/load-$1.txt" | tail -n 9 | tr '\n' ' ')"
  }
  load_line nd8 "$KENV $HOP" "$ND8"
  load_line own8 "$KENV $HOP $SL" "$OWN8"
  ssh "$RIG" "rm -f /dev/shm/ram_watchdog.sh"
  echo "FH_STEP load done $(date -Is)"
fi

if has tk; then
  : "${TEXT:?the agent text on the rig}"
  K=docs/data/2026-11-04_kreal_p106/run
  ssh "$RIG" "mkdir -p $RDIR/kreal" && tar -cf - -C $K kreal.sh kreal_req.py | ssh "$RIG" "tar -xf - -C $RDIR/kreal"
  for sp in 9 8; do
    if [ $sp = 9 ]; then ts=$TS9; dev=""; else ts=$TS8; dev="-dev $DEV8"; fi
    timeout 5400 ssh "$RIG" "cd $RDIR/kreal && rm -rf out-$sp
      setsid nohup env $HOP BIN=$BUILDS/$T/bin MODEL=$M TEXT=$TEXT OUT=$RDIR/kreal/out-$sp TS=$ts ARMS=s32kt0 \
        NOPIPE='-ot token_embd.weight=CPU $dev' bash kreal.sh > kreal-$sp.console 2>&1 < /dev/null & p=\$!
      while [ ! -e out-$sp/DONE ] && kill -0 \$p 2> /dev/null; do sleep 30; done; cat out-$sp/status.txt" > "$OUT/tk-$sp.status" 2>&1
    ssh "$RIG" "cat $RDIR/kreal/out-$sp/requests.tsv" > "$OUT/tk-$sp.tsv"
    echo "tk $sp cards: $(grep -c . "$OUT/tk-$sp.tsv") rows; $(grep -E 'ms each' "$OUT/tk-$sp.status" | sed 's/.*k=\([0-9]\).* \([0-9.]*\) ms each/k\1 \2/' | tr '\n' ' ')"
  done
  echo "FH_STEP tk done $(date -Is)"
fi

run_arm() {  # <arm> <REL_ENV> <args>
  { date -Is; ssh "$RIG" 'ps -eo comm | grep -x -E "llama-server|llama-bench" || echo "no llama process"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; grep -E "MemAvailable|SwapFree|Shmem:" /proc/meminfo; cat /proc/pressure/memory'; } > "$OUT/quiet_check_$1.txt" 2>&1
  env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
  echo "arm $1: release_check exit $? $(date -Is)"; tail -n 2 "$OUT/agent-$1/run.log" | cut -c 1-160
  ssh "$RIG" "ps -eo comm | grep -x llama-server && echo 'SERVER STILL UP' || echo 'no llama-server'"
}
ARMS_FH=${ARMS_FH:-fh8own}
if has prep; then
  ARMS=${ARMS_FH%% *} STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
  echo "prep: exit $? $(date -Is)"; tail -n 3 "$OUT/agent-prep.console" | cut -c 1-160
  echo "FH_STEP prep done $(date -Is)"
fi
if has arms; then
  for a in $ARMS_FH; do
    case $a in
      fh8own) run_arm fh8own "$KENV $HOP $SL" "$OWN8" ;;
      fh8nd)  run_arm fh8nd  "$KENV $HOP"     "$ND8" ;;
      fh9nd)  run_arm fh9nd  "$KENV $HOP"     "$ND9" ;;
      s35ctl) run_arm s35ctl "$KENV $HOP $SL" "$MT" ;;
      *) echo "unknown arm $a" ;;
    esac
  done
  echo "FH_STEP arms done $(date -Is)"
fi
if has report; then
  ARMS="s35stack $ARMS_FH" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
  echo "report: exit $?"; sed -n '/^### Per arm/,/^Tasks scored/p' "$OUT/agent-report.console" | head -14
fi
date -Is; echo FH_DONE
