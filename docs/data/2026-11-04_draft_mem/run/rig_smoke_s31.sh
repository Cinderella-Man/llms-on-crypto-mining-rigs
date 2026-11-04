#!/usr/bin/env bash
# S31 (Q564, Q635, Q592, Q580) on a rig: the -draftmem build's switches on the MTP draft line, one smoke test per arm,
# on the kho=off boot. The six arms planned in this folder's README, with two changes: each smoke is
# smoke_dflash.sh (the agent's first-turn request, ~2,780 tokens, then S12's smoke request: S33's mtp2k line was stopped
# 18 s into that first prompt, which S12's short request alone never reaches), through rig_server.sh's RAM watchdog
# (Q559; it covers mem_guard.sh's swap-out limit), and two arms first that ask the owner's question (2026-11-04: do
# the shared embeddings and head let the draft load and run without S15's workarounds?): sharenw and stagenw are the
# plain draft line (no -otd, the default draft micro-batch, pinned host buffers) with the switches.
# Run from the repository root (it runs only ssh here). ARMS picks arms; MIN_AVAIL_MB (default 60) and WD_ENV go to
# the watchdog. Prints ARM_DONE <arm> <exit> after each, ALL_DONE at the end.
: "${ENVFILE:?the machine file: S12 values (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example) with DRAFT}" "${OUT:?a local folder}"
T=${BUILD:-llama.cpp-20261103-1e0caf230-pascal-p106-v3-b12076-draftmem}
FLOOR=${MIN_AVAIL_MB:-60}; WDE=${WD_ENV:-}   # read before ENVFILE, which may set MIN_AVAIL_MB itself
set -a; . "$ENVFILE"; set +a; O=$OUT
export REL_BIN=/opt/llama-builds/$T/bin/llama-server
export MIN_AVAIL_MB=$FLOOR WD_ENV=$WDE
PUB=.
SMOKE=$PUB/docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh
export REQUESTS=$PUB/docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json
# the plain draft line (S12's), and S15's mtp2 line without its -otd (the arms add it where the draft keeps its embeddings)
P="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3"
B="$P --spec-draft-n-ubatch 128 --no-host"
SL=32768,8192,2
mkdir -p $O
arm() { # <arm> <SERVER_ENV> <args>
    local out=$O/smoke-$1
    mkdir -p "$out"
    { date -Is; ssh "$RIG" 'ps -eo comm | grep -x llama-server || echo "no llama-server"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; cat /proc/loadavg; grep -E "MemAvailable|CmaFree|SwapFree|Shmem:" /proc/meminfo; cat /proc/pressure/memory'; } > "$out/quiet_check.txt" 2>&1
    SMOKE_ENV="$2" ARGS="$3" SMOKE_LOGDIR=/mnt/models2/s31/smoke-$1 bash "$SMOKE" "$out" > "$out.console" 2>&1
    local rc=$?
    ssh "$RIG" 'ps -eo comm | grep -x llama-server && echo "SERVER STILL UP" || echo "no llama-server"'
    echo "ARM_DONE $1 exit $rc $(date -Is) | $(grep -h -E 'KILL' $out/ram_watchdog-smoke.tsv 2>/dev/null | head -n 1 | cut -c 1-120)"
}
for a in ${ARMS:-sharenw stagenw base embd share stage short short-own}; do
    case $a in
        sharenw)   arm sharenw   "LLAMA_DRAFT_SHARE=embd,head"                                        "$P" ;;
        stagenw)   arm stagenw   "LLAMA_DRAFT_SHARE=embd,head LLAMA_DIO_STAGING=2x16"                 "$P" ;;
        base)      arm base      ""                                                                   "$B -otd token_embd=CUDA4" ;;
        embd)      arm embd      "LLAMA_DRAFT_SHARE=embd"                                             "$B" ;;
        share)     arm share     "LLAMA_DRAFT_SHARE=embd,head"                                        "$B" ;;
        stage)     arm stage     "LLAMA_DRAFT_SHARE=embd,head LLAMA_DIO_STAGING=2x16"                 "$B" ;;
        short)     arm short     "LLAMA_DRAFT_SHARE=embd,head LLAMA_DIO_STAGING=2x16 LLAMA_DRAFT_SHORTLIST=$SL" "$B" ;;
        short-own) arm short-own "LLAMA_DRAFT_SHORTLIST=$SL"                                          "$B -otd token_embd=CUDA4" ;;
    esac
done
ssh "$RIG" 'rm -rf /mnt/models2/s31; ls /dev/shm'
echo ALL_DONE
