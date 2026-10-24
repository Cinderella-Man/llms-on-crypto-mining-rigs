#!/usr/bin/env bash
# selftest_pilot.sh OUT: a dry run of pilot.sh (or full_run.sh, with FULL=1) on the laptop against a fake rig, to
# check the scripts' plumbing before rig time: an `ssh` stand-in runs the rig-side commands locally (and serves -N -L
# with a small TCP forwarder), the "llama-server" binary starts mock_server.py on its --port, nvidia-smi is a stub.
# Needs AC_HOME, AC_NODE_DIR and AC_PI_ROOT as for selftest.sh. Uses ports AC_TEST_PORT … +2 (default 18060).
set -euo pipefail
OUT=$(realpath -m "${1:?usage: selftest_pilot.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
P=${AC_TEST_PORT:-18060}
rm -rf "$OUT"
mkdir -p "$OUT/fakebin"
cat > "$OUT/fakebin/ssh" <<'EOF'
#!/usr/bin/env bash
# fake ssh: ssh [options] HOST [COMMAND]; -N -L lport-spec forwards locally, a COMMAND runs here with bash
fwd=""; nflag=0
while [ $# -gt 0 ]; do
  case $1 in
    -N) nflag=1; shift;;
    -L) fwd=$2; shift 2;;
    -o|-c|-p|-i|-F) shift 2;;
    -*) shift;;
    *) break;;
  esac
done
host=$1; shift
if [ $nflag = 1 ]; then
  IFS=: read -r lh lp rh rp <<< "$fwd"
  exec python3 -c '
import asyncio, sys
lh, lp, rh, rp = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
async def pipe(r, w):
    try:
        while (d := await r.read(65536)):
            w.write(d); await w.drain()
    finally:
        w.close()
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection(rh, rp)
    except OSError:
        cw.close(); return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw), return_exceptions=True)
async def main():
    srv = await asyncio.start_server(handle, lh, lp)
    async with srv:
        await srv.serve_forever()
asyncio.run(main())' "$lh" "$lp" "$rh" "$rp"
fi
exec bash -c "$*"
EOF
cat > "$OUT/fakebin/fake-llama-server" <<EOF
#!/usr/bin/env bash
# stands in for llama-server: starts the mock on the --port it is given
port=""
while [ \$# -gt 0 ]; do [ "\$1" = --port ] && port=\$2; shift; done
echo "fake llama-server: srv  load_model: loading; created context checkpoint 1 of 4"
exec python3 "$HERE/mock_server.py" --port "\$port" --fail-every 4
EOF
cat > "$OUT/fakebin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
while true; do echo "$(date '+%Y/%m/%d %H:%M:%S'), 0, 40, 30.00 W, 1500 MHz, 1000 MiB, 0 %"; sleep 10; done
EOF
chmod +x "$OUT/fakebin/"*
export PATH="$OUT/fakebin:$PATH"
export RIG=fakerig RIG_PORT=$P TUNNEL_PORT=$((P + 1)) PROXY_PORT=$((P + 2)) RIG_LOGDIR=$OUT/riglogs
export V1_BIN=$OUT/fakebin/fake-llama-server V2_BIN=$OUT/fakebin/fake-llama-server
export STOCK_BIN=$OUT/fakebin/fake-llama-server MODEL=/nonexistent/model.gguf MODEL_K2H=/nonexistent/k2h.gguf
if [ "${FULL:-0}" = 1 ]; then
  "$HERE/full_run.sh" "$OUT/run" "${STAGES:-live replay}"
else
  "$HERE/pilot.sh" "$OUT/run"
fi
echo "--- dry run finished; outputs in $OUT/run"
ls "$OUT/run" "$OUT/run/rig"
[ ! -d "$OUT/riglogs" ] && echo "rig log folder removed after copying"
