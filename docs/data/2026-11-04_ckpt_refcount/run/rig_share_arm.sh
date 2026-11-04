#!/usr/bin/env bash
# C154 (Q704) on the rig: the agent check of the checkpoint line with --ctx-checkpoints-share, and the host memory at
# its low point. Runs agent_check.sh (docs/data/2026-11-03_flash_next_pipelined_rig/run/) with these arms on REL_BIN
# (the pascal-p106-v3-b12076-ckptrc build):
#   ckvs   LINE --ctx-checkpoints-share     (the change)
#   ckvr   LINE                              (the same build without it; ARMS="ckvr ckvs" runs it first)
# LINE is the checkpoint line under test, e.g. the ckv arm of the checkpoints-on-the-cards check
# (docs/data/2026-11-04_ckpt_cards/): "<the release line> --ctx-checkpoints-vram 4". The rig's lines run --cache-ram 0,
# where nothing copies a checkpoint, so ckvs is expected to give ckvr's (or that check's ckv arm's) host memory; the
# arm measures it, together with the time the saves take without the zero-fill of each new state buffer.
# Then, per arm, from mem_guard's log (one line a second: epoch avail cma avail-minus-cma swap-used swapout psi-some
# psi-full server-pid): the low point of MemAvailable and of MemAvailable minus CmaFree while the server ran, the most
# swap in use, and the seconds below MIN_LOW_MIB (default 150) of MemAvailable minus CmaFree.
# Usage (on the machine that drives the check; it only runs ssh): agent_check.sh's environment (RIG, CLIENT,
#   CLIENT_OUT, ports, REL_BIN, MODEL, ...; e.g. docs/data/2026-11-03_flash_next_mtp/run/s12.env.example) plus
#   LINE='<server arguments>' [ARMS=ckvs] [BUDGET_MIN_ckvs=40] [STEPS='prep arms report'] bash rig_share_arm.sh OUT
# Output: agent_check.sh's files in OUT, and OUT/host_memory.tsv
set -uo pipefail
OUT=$(realpath -m "${1:?usage: rig_share_arm.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
AC=${AC:-$HERE/../../2026-11-03_flash_next_pipelined_rig/run/agent_check.sh}
: "${LINE:?LINE=<the checkpoint line under test>}"
export ARMS=${ARMS:-ckvs}
export ARGS_ckvr="$LINE"
export ARGS_ckvs="$LINE --ctx-checkpoints-share"
export BUDGET_MIN_ckvr=${BUDGET_MIN_ckvr:-40} BUDGET_MIN_ckvs=${BUDGET_MIN_ckvs:-40}
MIN_LOW_MIB=${MIN_LOW_MIB:-150}
mkdir -p "$OUT"

bash "$AC" "$OUT" || exit $?

printf 'arm\tsamples\tmin_avail_MiB\tmin_avail_minus_cma_MiB\tmax_swap_MiB\ts_below_%s_MiB\n' "$MIN_LOW_MIB" > "$OUT/host_memory.tsv"
for arm in $ARMS; do
    f=$OUT/mem_guard-$arm.log
    [ -f "$f" ] || { echo "$arm: no mem_guard log"; continue; }
    awk -v a="$arm" -v low="$MIN_LOW_MIB" '
        /^#/ || NF < 9 || $9 == "-" { next }
        { n++; if (!s || $2 < ma) ma = $2; if (!s || $4 < mc) mc = $4; if ($5 > sw) sw = $5; if ($4 < low) b++; s = 1 }
        END { printf "%s\t%d\t%s\t%s\t%s\t%d\n", a, n, ma, mc, sw, b }' "$f" >> "$OUT/host_memory.tsv"
done
column -t "$OUT/host_memory.tsv"
