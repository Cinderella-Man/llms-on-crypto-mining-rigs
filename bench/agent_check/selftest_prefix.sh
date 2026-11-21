#!/usr/bin/env bash
# selftest_prefix.sh OUT: agent checks sharing one client at once (C896, Q1619), against mock_server.py, with the
# sandbox image in Docker as on a real client; no model, light (about 5 minutes). Needs Docker, curl and the image
# (AC_IMAGE, default agent-check:0.87.1, docker/Dockerfile); AC_CPU_SHARES (256) and AC_LABEL (pascal-quants=1) go on
# every container. Uses ports AC_TEST_PORT to AC_TEST_PORT+40 on 127.0.0.1 (default 18270). AC_CLIENT_BENCH, a client
# folder prepared earlier (agent_check/, task_results.py, paired_tasks.py, realloc_counts.py, as a release check's prep
# leaves it), runs part 2 with that folder's client_arm.sh instead of this tree's: the drivers name the containers.
# Checks, and exits non-zero if any fails:
# Part 1, client_arm.sh as the drivers start it (OUT/c):
# - one run alone (arm `same`, 2 tasks): exit 0, every attempt scored, no infrastructure try, the files as before
#   (res/, proxy.jsonl, same.rc, proxy-same.out, no same.pid), same.prefix holding acrel-<folder>-<CRC-32>-same, the
#   proxy and the attempts' containers named with it, and none left at the end;
# - three runs at once: A (3 tasks, a slower mock) and B (1 task) with the same arm and no PREFIX, D with
#   PREFIX=acpfx; then, while A and D run, C with A's PROXY_PORT and E with D's PREFIX and arm. A, B and D finish with
#   every attempt scored and no infrastructure try, A still runs when B ends and its proxy is still up then, the three
#   prefixes differ, and C and E are refused before they start anything (a message naming the cause, no rc file, no
#   container) with A's and D's proxies still up after them;
# Part 2, the drivers against a fake rig (OUT/d): an `ssh` stand-in runs a host's commands here, from $HOME; a
# "fakerig*" host's with a PATH in which chrt starts the RAM watchdog at normal priority and the card log does not start
# (no card is read), the mock standing in for llama-server and the watchdog's limits off. Its -N -L/-R forwards listen
# here and fail as ssh does with ExitOnForwardFailure when their port is taken (ssh's message, exit 255).
# - release_check_s12.sh with CLIENT=local (as on the laptop): A2 (6 tasks, a slower mock) and B2 (1 task) at once,
#   both arm v4r, and release_check.sh R2 (its client through the stand-in and an -R forward, arm rel, 1 task): all
#   three finish with every attempt scored, no infrastructure try and container names of their own, each named in its
#   driver's log ("client <arm>: containers acrel-<CLIENT_OUT's folder>-<CRC-32>-<arm>-*") and, with this tree's
#   client, in <arm>.prefix; A2 still runs when B2 ends; A2's report and collect steps write compare.md and res/;
# - while A2 runs: C2 with A2's RELAY_PORT stops before it starts anything (no rig log folder, no client); D2, whose
#   RELAY_PORT is bound but not listening (free to the port check, taken for the relay), stops when its relay cannot
#   take the port, before its client starts; E2, stopped with TERM as soon as its client's proxy runs, and F2,
#   stopped with TERM once an attempt runs, each remove their own containers and stop their rig server and relay;
#   A2's proxy is up after each of them;
# - at the end no container of any run is left and no port of the drivers is still in use.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: selftest_prefix.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
S12=${S12:-$HERE/../../docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh}
P=${AC_TEST_PORT:-18270}
IMAGE=${AC_IMAGE:-agent-check:0.87.1}; LABEL=${AC_LABEL:-pascal-quants=1}; CPU=${AC_CPU_SHARES:-256}
export IMAGE LABEL CPU_SHARES=$CPU
NAMES='^(acrel-st-|acpfx-)'  # every container this test's runs name
FAILS=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
t() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }  # t DESCRIPTION CONDITION
port_busy() { timeout 3 bash -c 'exec 3<> "/dev/tcp/127.0.0.1/$0"' "$1" 2> /dev/null; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2> /dev/null)" = true ]; }
ours() { docker ps -a --format '{{.Names}}' | grep -E "${1:-$NAMES}"; }  # containers of ours, by name
wait_for() {  # wait_for SECONDS CONDITION
  local n=0
  until eval "$2"; do n=$((n + 1)); [ "$n" -gt $(($1 * 2)) ] && return 1; sleep 0.5; done
}
scored() {  # scored RESULTS-FOLDER ARM N: N attempts, every one scored, no infrastructure try
  python3 -I - "$1/res/$2" "$3" << 'EOF'
import json, re, sys
from pathlib import Path
root, n = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(p.read_text()) for p in root.glob("*/*/result.json") if re.fullmatch(r"a\d+", p.parent.name)]
infra = [p for p in root.glob("*/*") if p.name.endswith("-infra")]
print(f"     {root}: {len(rows)} attempts, statuses {sorted({r.get('status') for r in rows})}, "
      f"passed {sum(r.get('success', 0) for r in rows)}, infrastructure tries {len(infra)}")
sys.exit(0 if len(rows) == n and all(r.get("status") == "scored" for r in rows) and not infra else 1)
EOF
}

for i in $(seq 0 40); do
  port_busy $((P + i)) && { echo "port $((P + i)) is in use; set AC_TEST_PORT to a free range of 41"; exit 2; }
done
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "no image $IMAGE (docker/Dockerfile)"; exit 2; }
ours > /dev/null && { echo "containers of an earlier run of this test are left: $(ours | tr '\n' ' ')"; exit 2; }
PIDS=()
cleanup() {
  kill "${PIDS[@]}" 2> /dev/null
  ours | xargs -r docker rm -f > /dev/null 2>&1
}
trap cleanup EXIT
rm -rf "$OUT"; mkdir -p "$OUT/c" "$OUT/d"
OUTP=$(cd "$OUT" && pwd -P)  # the physical path, as client_arm.sh labels its proxy with it
(while true; do echo "$(date +%T) $(ours | tr '\n' ' ')"; sleep 0.5; done) > "$OUT/containers.log" 2>&1 &
PIDS+=($!)

echo "== part 1: client_arm.sh"
mock() { python3 "$HERE/mock_server.py" --port "$@" > "$OUT/c/mock-$1.log" 2>&1 & PIDS+=($!); }
mock $P
mock $((P + 1)) --prefill-ms 3
mock $((P + 2))
mock $((P + 3)) --prefill-ms 2
wait_for 20 'port_busy $P && port_busy $((P + 1)) && port_busy $((P + 2)) && port_busy $((P + 3))' \
  || { echo "the mocks did not start"; exit 1; }
carm() {  # carm NAME UPSTREAM PROXY TASKS [VAR=value …]: client_arm.sh in the background, as the drivers start it
  local name=$1 up=$2 px=$3 tasks=$4
  shift 4
  mkdir -p "$OUT/c/$name"
  env OUT="$OUT/c/$name" ARM=same UPSTREAM_PORT="$up" PROXY_PORT="$px" TASK_SET=bugfix TASKS="$tasks" TIME_CAP_MIN=5 \
    "$@" setsid bash "$HERE/client_arm.sh" > "$OUT/c/$name.log" 2>&1 < /dev/null &
  PID=$!
}

carm st-one $P $((P + 10)) 2
wait $PID; rc=$?
D=$OUT/c/st-one
pfx=$(cat "$D/same.prefix" 2> /dev/null)
t "one run alone: exit 0 and same.rc 0" '[ $rc = 0 ] && [ "$(cat $D/same.rc)" = 0 ]'
t "one run alone: prefix $pfx is acrel-st-one-<CRC-32>-same" '[[ $pfx =~ ^acrel-st-one-[0-9a-f]{8}-same$ ]]'
t "one run alone: every attempt scored" 'scored $D same 2'
t "one run alone: the files as before" \
  '[ -s $D/proxy.jsonl ] && [ -f $D/proxy-same.out ] && [ -d $D/res/same ] && [ ! -e $D/same.pid ]'
t "one run alone: its proxy and attempt containers carried the prefix" \
  'grep -q " $pfx-proxy " $OUT/containers.log && grep -Eq " $pfx-[0-9]+-s0 " $OUT/containers.log'
t "one run alone: no container left" '! ours "^$pfx-" > /dev/null'

carm st-a $((P + 1)) $((P + 11)) 3; PA=$PID
carm st-b $((P + 2)) $((P + 12)) 1; PB=$PID
carm st-d $((P + 3)) $((P + 13)) 2 PREFIX=acpfx; PD=$PID
A=$OUT/c/st-a B=$OUT/c/st-b
wait_for 30 'running "$(cat $A/same.prefix 2> /dev/null)-proxy" && running acpfx-same-proxy' \
  || bad "A's and D's proxies did not come up"
pa=$(cat "$A/same.prefix")
carm st-c $((P + 2)) $((P + 11)) 1
wait $PID; rc=$?
t "C with A's PROXY_PORT: refused (exit 1, the port named, no rc file, no container)" \
  '[ $rc = 1 ] && grep -q "PROXY_PORT $((P + 11)) is in use" $OUT/c/st-c.log && [ ! -e $OUT/c/st-c/same.rc ] \
   && ! ours "^acrel-st-c-" > /dev/null'
carm st-e $((P + 3)) $((P + 14)) 1 PREFIX=acpfx
wait $PID; rc=$?
t "E with D's PREFIX and arm: refused (exit 1, the other run named, no rc file, no container of its own)" \
  '[ $rc = 1 ] && grep -q "acpfx-same-proxy runs for another run ($OUTP/c/st-d)" $OUT/c/st-e.log \
   && [ ! -e $OUT/c/st-e/same.rc ] && [ "$(ours ^acpfx- | grep -c proxy)" = 1 ]'
t "A's and D's proxies still up after C and E" 'running $pa-proxy && running acpfx-same-proxy'
wait $PB; rc=$?
t "A still runs when B ends, its proxy up" 'kill -0 $PA 2> /dev/null && running $pa-proxy'
pb=$(cat "$B/same.prefix" 2> /dev/null)
t "B: exit 0, every attempt scored" '[ $rc = 0 ] && scored $B same 1'
wait $PA; rc=$?
t "A: exit 0, every attempt scored" '[ $rc = 0 ] && scored $A same 3'
wait $PD; rc=$?
t "D: exit 0, every attempt scored, its PREFIX in same.prefix" \
  '[ $rc = 0 ] && scored $OUT/c/st-d same 2 && [ "$(cat $OUT/c/st-d/same.prefix)" = acpfx-same ]'
t "A, B and D had prefixes of their own ($pa, $pb, acpfx-same)" \
  '[ "$pa" != "$pb" ] && [[ $pb =~ ^acrel-st-b-[0-9a-f]{8}-same$ ]] && [ "$pa" != acpfx-same ]'
t "A's and B's proxies ran at the same time" 'grep " $pa-proxy " $OUT/containers.log | grep -q " $pb-proxy "'
t "part 1: no container left" '! ours > /dev/null'

echo "== part 2: the drivers against a fake rig"
FB=$OUT/d/fakebin RB=$OUT/d/rigbin
mkdir -p "$FB" "$RB" "$OUT/d/client"
cat > "$FB/ssh" << EOF
#!/usr/bin/env bash
# ssh stand-in: ssh [options] HOST [COMMAND]; -N -L/-R LH:LP:RH:RP listens on LH:LP here and forwards to RH:RP, failing
# as ssh does with ExitOnForwardFailure when LP is taken; a COMMAND runs here with bash from HOME, a fakerig* host's
# with the fake rig's tools first in PATH
kind=""; fwd=""; nflag=0
while [ \$# -gt 0 ]; do
  case \$1 in
    -N) nflag=1; shift;;
    -L|-R) kind=\$1; fwd=\$2; shift 2;;
    -o|-c|-p|-i|-F) shift 2;;
    -*) shift;;
    *) break;;
  esac
done
host=\$1; shift
if [ \$nflag = 1 ]; then
  exec python3 -I -c '
import asyncio, sys
kind, (lh, lp, rh, rp) = sys.argv[1], sys.argv[2].split(":")
async def pipe(r, w):
    try:
        while (d := await r.read(65536)):
            w.write(d); await w.drain()
    finally:
        w.close()
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection(rh, int(rp))
    except OSError:
        cw.close(); return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw), return_exceptions=True)
async def main():
    try:
        srv = await asyncio.start_server(handle, lh, int(lp))
    except OSError:
        if kind == "-L":
            print(f"bind [{lh}]:{lp}: Address already in use", file=sys.stderr)
            print(f"channel_setup_fwd_listener_tcpip: cannot listen to port: {lp}", file=sys.stderr)
            print("Could not request local forwarding.", file=sys.stderr)
        else:
            print(f"Error: remote port forwarding failed for listen port {lp}", file=sys.stderr)
        sys.exit(255)
    async with srv:
        await srv.serve_forever()
asyncio.run(main())' "\$kind" "\$fwd"
fi
cd ~ || exit 1
case \$host in fakerig*) export PATH="$RB:\$PATH";; esac
exec bash -c "\$*"
EOF
cat > "$FB/fake-llama-server" << EOF
#!/usr/bin/env bash
# stands in for llama-server: the mock on the --port it is given, with the --decode-ms and --prefill-ms given
port=""; mo=()
while [ \$# -gt 0 ]; do
  case \$1 in --port) port=\$2;; --decode-ms|--prefill-ms) mo+=("\$1" "\$2");; esac
  shift
done
echo "fake llama-server: srv  load_model: loading"
exec python3 "$HERE/mock_server.py" --port "\$port" "\${mo[@]}"
EOF
cat > "$RB/chrt" << 'EOF'
#!/usr/bin/env bash
# no real-time priority on the fake rig: chrt [options] PRIORITY COMMAND… runs COMMAND as it is
while [ "${1#-}" != "$1" ]; do shift; done
shift
exec "$@"
EOF
cat > "$RB/python3" << EOF
#!/usr/bin/env bash
# the fake rig reads no card: the card log does not start, everything else is python3
case "\$*" in *card_log-*) echo "card log: off on the fake rig" >&2; exit 2;; esac
exec $(command -v python3) "\$@"
EOF
chmod +x "$FB"/* "$RB"/*
if [ -n "${AC_CLIENT_BENCH:-}" ]; then
  cp -a "$AC_CLIENT_BENCH/." "$OUT/d/client/"
  echo "   part 2's client folder: a copy of $AC_CLIENT_BENCH"
else
  tar -C "$HERE/.." --exclude=__pycache__ -cf - agent_check task_results.py paired_tasks.py \
    -C "$HERE/../../scripts" realloc_counts.py | tar -C "$OUT/d/client" -xf -
fi
pfx_of() {  # pfx_of NAME ARM: the container name prefix the driver of run NAME gave its client (its run.log), with ARM
  sed -n "s/.* client $2: containers \(.*\)-\*\$/\1/p" "$OUT/d/drv-$1/run.log" 2> /dev/null | tail -n 1
}
DRV=(BENCH="$HERE" CLIENT_DIR="$OUT/d/client" REL_BIN="$FB/fake-llama-server" MODEL=/nonexistent/model.gguf
  TASK_SET=bugfix TIME_CAP_MIN=5 MIN_AVAIL_MB=off WD_ENV="MIN_USABLE_MB=off PSI_FULL_PCT=off PSI_SOME_PCT=off \
KILL_SWAPOUT_MIB=off" PATH="$FB:$PATH")
drv() {  # drv NAME RIG_PORT RELAY_PORT PROXY_PORT TASKS MOCK-OPTIONS [VAR=value …]: release_check_s12.sh, CLIENT=local
  local name=$1 rp=$2 lp=$3 px=$4 tasks=$5 mo=$6
  shift 6
  env "${DRV[@]}" CLIENT=local RIG="fakerig-$name" RIG_PORT="$rp" RELAY_PORT="$lp" PROXY_PORT="$px" \
    CLIENT_OUT="$OUT/d/out/st-$name" RIG_LOGDIR="$OUT/d/rig-$name" ARMS=v4r "ARGS_v4r=$mo" \
    TASKS="$tasks" STEPS="run report collect" "$@" bash "$S12" "$OUT/d/drv-$name" > "$OUT/d/drv-$name.txt" 2>&1 &
  PID=$!
}
CO=$OUT/d/out
drv a2 $((P + 20)) $((P + 21)) $((P + 22)) 6 "--prefill-ms 5"; PA2=$PID
drv b2 $((P + 23)) $((P + 24)) $((P + 25)) 1 "--decode-ms 2"; PB2=$PID
env "${DRV[@]}" CLIENT=fakeclient RIG=fakerig-r2 RIG_PORT=$((P + 34)) RELAY_PORT=$((P + 35)) CLIENT_PORT=$((P + 36)) \
  PROXY_PORT=$((P + 37)) CLIENT_OUT="$CO/st-r2" RIG_LOGDIR="$OUT/d/rig-r2" ARMS=rel REL_ARGS="--decode-ms 2" TASKS=1 \
  STEPS="run report collect" bash "$HERE/release_check.sh" "$OUT/d/drv-r2" > "$OUT/d/drv-r2.txt" 2>&1 &
PR2=$!
wait_for 60 'pa2=$(pfx_of a2 v4r) && [ -n "$pa2" ] && running $pa2-proxy' || bad "A2's client did not come up"

drv c2 $((P + 26)) $((P + 21)) $((P + 27)) 1 "--decode-ms 2"
wait $PID; rc=$?
t "C2 with A2's RELAY_PORT: stopped before it started anything" \
  '[ $rc = 1 ] && grep -q "nothing started: RELAY_PORT $((P + 21)) is in use here" $OUT/d/drv-c2/run.log \
   && [ ! -e $OUT/d/rig-c2 ] && [ ! -e $CO/st-c2/v4r.log ] && ! port_busy $((P + 26))'
t "A2's proxy up after C2" 'running $pa2-proxy'

python3 -I -c 'import socket, sys, time
s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); time.sleep(600)' $((P + 29)) &
BLOCK=$!
PIDS+=($BLOCK)
sleep 1
drv d2 $((P + 28)) $((P + 29)) $((P + 30)) 1 "--decode-ms 2"
wait $PID; rc=$?
kill $BLOCK 2> /dev/null
t "D2 whose relay cannot take its port: stopped before its client started, its rig server stopped" \
  '[ $rc = 1 ] && grep -q "the relay could not take its port (bind \[127.0.0.1\]:$((P + 29)): Address already in use)" \
   $OUT/d/drv-d2/run.log && [ ! -e $CO/st-d2/v4r.log ] && ! port_busy $((P + 28))'
t "A2's proxy up after D2" 'running $pa2-proxy'

drv e2 $((P + 31)) $((P + 32)) $((P + 33)) 2 "--prefill-ms 2"; PE2=$PID
wait_for 60 'pe2=$(pfx_of e2 v4r) && [ -n "$pe2" ] && running $pe2-proxy' || bad "E2's client did not come up"
kill -TERM $PE2
wait_for 40 '! kill -0 $PE2 2> /dev/null' || bad "E2's driver did not stop within 40 s of TERM"
t "E2 stopped as its client started: its containers removed, its rig server and relay stopped ($pe2)" \
  'grep -q "client arm v4r stopped" $OUT/d/drv-e2/run.log && ! ours "^$pe2-" > /dev/null && ! port_busy $((P + 31)) \
   && ! port_busy $((P + 32)) && ! port_busy $((P + 33))'
t "A2's proxy up after E2's stop" 'running $pa2-proxy'

drv f2 $((P + 38)) $((P + 39)) $((P + 40)) 4 "--prefill-ms 3"; PF2=$PID
wait_for 90 'pf2=$(pfx_of f2 v4r) && [ -n "$pf2" ] && ours "^$pf2-[0-9]+-s0$" > /dev/null' \
  || bad "F2's first attempt did not start"
kill -TERM $PF2
wait_for 40 '! kill -0 $PF2 2> /dev/null' || bad "F2's driver did not stop within 40 s of TERM"
t "F2 stopped mid-attempt: its containers removed, its rig server and relay stopped ($pf2)" \
  'grep -q "client arm v4r stopped" $OUT/d/drv-f2/run.log && ! ours "^$pf2-" > /dev/null && ! port_busy $((P + 38)) \
   && ! port_busy $((P + 39)) && ! port_busy $((P + 40))'
t "A2's proxy up after F2's stop" 'running $pa2-proxy'

wait $PB2; rc=$?
t "A2 still runs when B2 ends" 'kill -0 $PA2 2> /dev/null && running $pa2-proxy'
pb2=$(pfx_of b2 v4r)
t "B2: exit 0, every attempt scored" '[ $rc = 0 ] && scored $CO/st-b2 v4r 1'
wait $PR2; rc=$?
pr2=$(pfx_of r2 rel)
t "R2 (release_check.sh, -R relay): exit 0, every attempt scored" '[ $rc = 0 ] && scored $CO/st-r2 rel 1'
wait $PA2; rc=$?
t "A2: exit 0, every attempt scored" '[ $rc = 0 ] && scored $CO/st-a2 v4r 6'
t "A2: report and collect wrote compare.md and res/ into its OUT" \
  '[ -s $OUT/d/drv-a2/compare.md ] && [ -d $OUT/d/drv-a2/res/v4r ]'
t "A2, B2, E2 and R2 had prefixes of their own ($pa2, $pb2, $pe2, $pr2)" \
  '[[ $pa2 =~ ^acrel-st-a2-[0-9a-f]{8}-v4r$ ]] && [[ $pb2 =~ ^acrel-st-b2-[0-9a-f]{8}-v4r$ ]] \
   && [[ $pe2 =~ ^acrel-st-e2-[0-9a-f]{8}-v4r$ ]] && [[ $pr2 =~ ^acrel-st-r2-[0-9a-f]{8}-rel$ ]]'
[ -n "${AC_CLIENT_BENCH:-}" ] || t "this tree's client wrote the same names into <arm>.prefix" \
  '[ "$(cat $CO/st-a2/v4r.prefix)" = "$pa2" ] && [ "$(cat $CO/st-r2/rel.prefix)" = "$pr2" ]'
t "A2's and B2's proxies ran at the same time" 'grep " $pa2-proxy " $OUT/containers.log | grep -q " $pb2-proxy "'
t "part 2: no container left" '! ours > /dev/null'
busy=()
for i in $(seq 20 40); do port_busy $((P + i)) && busy+=($((P + i))); done
t "part 2: no port of the drivers still in use (${busy[*]:-none})" '[ ${#busy[@]} = 0 ]'

echo "selftest_prefix: $FAILS failed"
[ "$FAILS" = 0 ]
