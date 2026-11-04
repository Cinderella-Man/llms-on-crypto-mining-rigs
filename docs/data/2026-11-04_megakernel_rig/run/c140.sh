#!/usr/bin/env bash
# C140 on one rig: megakernel_p106's smoke, checks and stage timings on card 0 with C140's own
# bundle (binary md5 b041450c…), then the card-to-card hop on four pairs (two behind one switch, switch to root port and
# back, two switches) with the card-hops bundle's superset binary (md5 f47c545d…: --hop-chunks 1,2,4 and the manual
# staged hop). Detached on the rig; ends with $W/ALLDONE.
set -uo pipefail
W=${W:?W=<work folder holding mk/ (C140 bundle) and hops/ (card-hops bundle)>}
cd "$W/mk" && md5sum -c MD5SUMS > "$W/mk-md5check.txt" 2>&1 && OUT=$W/mk/out HOP= ./run_rig.sh > "$W/mk-console.log" 2>&1
echo "run_rig exit $?" >> "$W/status.txt"
cd "$W/hops" && md5sum -c MD5SUMS > "$W/hops-md5check.txt" 2>&1 && OUT=$W/hops/out LINES=0 ./rig_hops.sh > "$W/hops-console.log" 2>&1
echo "rig_hops exit $?" >> "$W/status.txt"
touch "$W/ALLDONE"
