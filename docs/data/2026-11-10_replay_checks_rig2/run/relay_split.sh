#!/usr/bin/env bash
# A recorded replay suite, build machine -> rig over the LAN (nc), onto three disks of the rig when no single disk
# has room for it. One pass of the suite's tar: the rig takes the stream's md5 on the way (compared with the tar's
# own) and three tar readers unpack one member list each (split_lists.sh, recv_split.sh). The cases on disks 2 and 3
# are then symlinked into disk 1's cases/ folder, so the suite reads as one folder; every replayed case checks its own
# blobs' md5 again. Run from a third machine that reaches both over ssh; it copies only the lists and the scripts.
#
# Env: CLIENT (ssh command of the build machine), RIG (ssh command of the rig), RIG_ADDR (the rig's LAN address),
#   TAR (the suite's tar on the build machine, with <tar>.md5 beside it), WORK (a scratch folder on the build machine),
#   D1 D2 D3 (a folder on each of the rig's three disks; D1 gets the suite's folder), C1 C2 (bytes for D1 and D2, the
#   rest goes to D3), PORT (default 19571), SCRIPT (scripts/replay_checks.sh, copied into D1)
# Usage: CLIENT='ssh build' RIG='ssh rig' RIG_ADDR=<address> TAR=<sets>/ladder3-rest.tar WORK=<dir> \
#        D1=<dir> D2=<dir> D3=<dir> C1=5600000000 C2=5200000000 bash relay_split.sh
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
Q=${CLIENT:?CLIENT=<ssh command of the build machine>}
R=${RIG:?RIG=<ssh command of the rig>}
A=${RIG_ADDR:?RIG_ADDR=<the rig's LAN address>}; PORT=${PORT:-19571}
TAR=${TAR:?TAR=<the suite's tar on the build machine>}
W=${WORK:?WORK=<scratch folder on the build machine>}
D1=${D1:?}; D2=${D2:?}; D3=${D3:?}
SET=$(basename "$TAR" .tar)
SCRIPT=${SCRIPT:-$H/../../../../scripts/replay_checks.sh}

date -Is; echo "== lists on the build machine"
$Q "mkdir -p $W && cat > $W/split_lists.sh" < "$H/split_lists.sh" || exit 1
$Q "bash $W/split_lists.sh $TAR $W ${C1:-5600000000} ${C2:-5200000000}" || exit 1
SMD5=$($Q "cut -d' ' -f1 $TAR.md5") || exit 1
SSIZE=$($Q "stat -c %s $TAR") || exit 1

echo "== lists and scripts to the rig"
$R "mkdir -p $D1/lists $D2 $D3" || exit 1
for i in 1 2 3; do $Q "cat $W/list$i.txt" | $R "cat > $D1/lists/list$i.txt" || exit 1; done
$R "cat > $D1/recv_split.sh" < "$H/recv_split.sh" || exit 1
$R "cat > $D1/replay_checks.sh" < "$SCRIPT" || exit 1
$R "wc -l $D1/lists/list[123].txt; df -h $D1 $D2 $D3 | cat"

date -Is; echo "== $SET.tar ($SSIZE B, md5 $SMD5)"
# an ssh call that starts a detached job waits for it, hence the timeout
timeout 20 $R "setsid nohup bash $D1/recv_split.sh $A $PORT $D1/lists $D1 $D2 $D3 > $D1/recv.log 2>&1 < /dev/null &"
sleep 3
timeout 3600 $Q "nc -N $A $PORT < $TAR" || { echo "SEND FAILED $SET"; exit 1; }
for _ in $(seq 1 120); do [ -n "$($R "cat $D1/recv.done 2>/dev/null")" ] && break; sleep 5; done
got=$($R "cat $D1/recv.md5" | awk '{print $1}')
rc=$($R "cat $D1/recv.1.rc $D1/recv.2.rc $D1/recv.3.rc" | tr '\n' ' ')
$R "cat $D1/recv.log"
if [ "$got" = "$SMD5" ] && [ "$rc" = "0 0 0 " ]; then echo "MD5 OK $SET $got, untar rc $rc"; else echo "MD5 MISMATCH or untar failed: $SET got=$got want=$SMD5 rc=$rc"; exit 1; fi

echo "== join the three parts"
$R "set -e; cd $D1/$SET/cases; for d in $D2 $D3; do find \$d/$SET/cases -type f -print0 | xargs -0 -r ln -s -t .; done
    echo \"entries: \$(ls | wc -l), of them links: \$(find . -maxdepth 1 -type l | wc -l), dangling: \$(find -L . -maxdepth 1 -type l | wc -l)\"
    echo \"meta \$(ls | grep -c '\\.meta\$') in \$(ls | grep -c '\\.in\$') ref \$(ls | grep -c '\\.ref\$')\"
    du -shL $D1/$SET; du -sh $D1/$SET $D2 $D3; df -h $D1 $D2 $D3 | cat"
date -Is; echo RELAY_DONE "$SET" "$D1/$SET"
