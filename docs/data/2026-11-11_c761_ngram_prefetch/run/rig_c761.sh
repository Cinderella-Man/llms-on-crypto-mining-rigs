#!/usr/bin/env bash
# rig_c761.sh: C761's rig row (R106, Q1351), ON the rig, detached, the rig idle; ends with OUT/DONE. R81's driver (R42's
# r42_run.sh with R42_LOADS=warm4096: one server load of the Flash-Next preset with the MTP draft, the replay cut to one
# task, then two fresh 4,097-token prompts, each with the micro-batch taper off and on) once per arm of ARMS, every load
# under LLAMA_UBATCH_TRACE=1 (the per-input split of each micro-batch's inputs and one ngram-prefetch line per call):
#   off  the n-gram rows issued in set_input, just before card 0 (the inline path, counted by the trace)
#   on   LLAMA_NGRAM_PREFETCH=1: the rows of a call's later micro-batches issued by the helper thread, at most
#        LLAMA_NGRAM_PREFETCH_MB (default 48) MiB ahead (export it to change the cap)
# Before each load the n-gram table's file leaves the page cache (posix_fadvise DONTNEED on TABLE: that file's clean
# pages only, nothing else is dropped), so that both loads read their fresh prompts' rows from the disk. vm_sampler.sh
# logs MemAvailable, swap and paging every 0.5 s through each load.
# Usage: R42=<folder with r42_run.sh> R43=<R43's folder: r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv,
#   settings.env> BUILDS=<builds> BUILD=<this branch's build folder> MODEL=<first shard> DRAFT=<MTP draft>
#   TABLE=<the GGUF shard that holds the n-gram table> OUT=<dir> [ARMS='off on']
#   setsid nohup bash rig_c761.sh > <dir>.out 2>&1 < /dev/null &
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${R42:?}" "${R43:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${TABLE:?}" "${OUT:?}"
ARMS=${ARMS:-off on}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
md5sum "$0" "$here/vm_sampler.sh" "$R42/r42_run.sh" "$BUILDS/$BUILD/bin/llama-server" > "$OUT/md5.txt" 2>&1
grep -E '^(name|commit)' "$BUILDS/$BUILD/BUILDINFO.txt" >> "$OUT/md5.txt" 2>/dev/null
for arm in $ARMS; do
    case $arm in
        off) envs=(LLAMA_UBATCH_TRACE=1) ;;
        on)  envs=(LLAMA_UBATCH_TRACE=1 LLAMA_NGRAM_PREFETCH=1) ;;
        *) log "unknown arm $arm"; continue ;;
    esac
    d=$OUT/$arm
    mkdir -p "$d"
    # the previous load's server must be gone
    for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ] && break; sleep 2; done
    python3 -I -c 'import os, sys
fd = os.open(os.path.realpath(sys.argv[1]), os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)' "$TABLE"
    log "$arm: the table's pages dropped (rc $?); $(grep -E '^(MemAvailable|Cached|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
    rm -f "$d/vm.stop"
    OUT="$d/vm.tsv" STOP="$d/vm.stop" bash "$here/vm_sampler.sh" &
    vm=$!
    log "$arm: start, ${envs[*]}${LLAMA_NGRAM_PREFETCH_MB:+ LLAMA_NGRAM_PREFETCH_MB=$LLAMA_NGRAM_PREFETCH_MB}"
    env "${envs[@]}" R43="$R43" BUILDS="$BUILDS" BUILD="$BUILD" MODEL="$MODEL" DRAFT="$DRAFT" OUT="$d/r42" \
        R42_LOADS=warm4096 bash "$R42/r42_run.sh" > "$d/r42_run.out" 2>&1 < /dev/null
    log "$arm: exit $?"
    touch "$d/vm.stop"
    wait "$vm"
done
log "done; llama processes $(ps -eo comm | grep -c '^llama-')"
touch "$OUT/DONE"
