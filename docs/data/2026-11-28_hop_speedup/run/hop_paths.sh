#!/usr/bin/env bash
# The standalone hop measurements of 2026-11-28 on an idle rig (no model loaded), in one go, ~15 minutes:
#   1. vram_write (bench/hop_paths) on cards CARDS_VRAM: the host writing a hop's bytes into the card's video memory
#      through its BAR1 window, against the card's own copies to and from pinned host memory (Q1941)
#   2. memops_check (bench/hop_paths) on each pair in PAIRS: stream memory operations on a P106, and their signal from
#      one card's stream to another's against an event (Q752)
#   3. megakernel_p106 --hop (bench/megakernel_p106, C140's hop microbenchmark) on each pair in PAIRS at HOP_SIZES: the
#      driver's peer copy, host-staged copies, the one-shot mailbox, the manual staged hop in HOP_CHUNKS chunks (what
#      the engine's GGML_CUDA_HOP_STAGED does) and the streamed mailbox in HOP_MBOX chunks (GGML_CUDA_HOP_MAILBOX)
# Run from a machine that runs only ssh and tar. Env: RIG (ssh alias), TOOLS (local folder with vram_write and
# memops_check, as bench/hop_paths/build.sh leaves them), BENCH_BIN (local megakernel_p106), OUT (local folder),
# RDIR (folder on the rig, default /var/tmp/hop-paths, removed at the end), CARDS_VRAM (default "0 4 5": behind the
# first switch, alone on a chipset port, behind the second switch), PAIRS (default "0,1 3,4 4,5 3,5": inside one
# switch, a switch to the lone card and back, across the switches), HOP_SIZES (default one token's hop in Q8_0, F16
# and F32, a four-token Q8_0 verify pass and its F32 size, a 64-token and a 256-token Q8_0 prompt micro-batch),
# HOP_REPS (100), HOP_CHUNKS (1,2,4,8), HOP_MBOX (1,4,8,16). Card numbers are in PCI bus order (CUDA_DEVICE_ORDER).
set -uo pipefail
: "${RIG:?}" "${TOOLS:?}" "${BENCH_BIN:?}" "${OUT:?}"
RDIR=${RDIR:-/var/tmp/hop-paths}
CARDS_VRAM=${CARDS_VRAM:-0 4 5}
PAIRS=${PAIRS:-0,1 3,4 4,5 3,5}
HOP_SIZES=${HOP_SIZES:-10880,20480,40960,43520,163840,696320,2785280}
HOP_REPS=${HOP_REPS:-100}
HOP_CHUNKS=${HOP_CHUNKS:-1,2,4,8}
HOP_MBOX=${HOP_MBOX:-1,4,8,16}
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/console.txt"; }

log "idle check"
timeout 60 ssh "$RIG" "ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' && echo NOT_IDLE || echo idle
  nvidia-smi --query-gpu=index,pci.bus_id,memory.used,clocks.sm,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  ls /usr/lib/x86_64-linux-gnu/libvulkan.so.1 2>&1; grep MemAvailable /proc/meminfo" > "$OUT/start_state.txt" 2>&1
cat "$OUT/start_state.txt"
grep -q NOT_IDLE "$OUT/start_state.txt" && { log "the rig is not idle: stop"; exit 1; }

log "copy the three programs"
timeout 120 ssh "$RIG" "rm -rf $RDIR && mkdir -p $RDIR/out" || exit 1
tar -C "$TOOLS" -cf - vram_write memops_check | timeout 120 ssh "$RIG" "tar -C $RDIR -xf -" || exit 1
timeout 120 ssh "$RIG" "cat > $RDIR/megakernel_p106 && chmod +x $RDIR/megakernel_p106" < "$BENCH_BIN" || exit 1
timeout 60 ssh "$RIG" "cd $RDIR && md5sum vram_write memops_check megakernel_p106" | tee "$OUT/md5_rig.txt"

for c in $CARDS_VRAM; do
  log "1. vram_write card $c"
  timeout 300 ssh "$RIG" "cd $RDIR && ./vram_write $c 300 20 > out/vram_$c.csv 2> out/vram_$c.err; echo exit \$?; cat out/vram_$c.err | tail -3"
done
for p in $PAIRS; do
  a=${p%,*} b=${p#*,}
  log "2. memops_check $a $b"
  timeout 300 ssh "$RIG" "cd $RDIR && CUDA_DEVICE_ORDER=PCI_BUS_ID ./memops_check $a $b 1000 > out/memops_${a}_${b}.txt 2>&1; echo exit \$?; tail -2 out/memops_${a}_${b}.txt"
done
for p in $PAIRS; do
  log "3. megakernel_p106 --hop $p"
  timeout 1200 ssh "$RIG" "cd $RDIR && CUDA_DEVICE_ORDER=PCI_BUS_ID ./megakernel_p106 --hop $p --hop-only --hop-sizes $HOP_SIZES \
    --hop-reps $HOP_REPS --hop-chunks $HOP_CHUNKS --hop-mbox $HOP_MBOX --csv out/hop_${p/,/_}.csv > out/hop_${p/,/_}.txt 2>&1; echo exit \$?"
done

log "collect, clean up"
timeout 300 ssh "$RIG" "tar -C $RDIR -cf - out" | tar -C "$OUT" -xf - || exit 1
timeout 60 ssh "$RIG" "rm -rf $RDIR; ps -eo comm | grep -x -E 'vram_write|memops_check|megakernel_p106' || echo 'no tool process'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; printf 'Xid lines: '; dmesg | grep -c 'NVRM: Xid'" > "$OUT/end_state.txt" 2>&1
cat "$OUT/end_state.txt"
log "done"
