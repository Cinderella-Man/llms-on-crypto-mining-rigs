#!/usr/bin/env bash
# Stock llama.cpp's fastest line before its agent arm: llama-bench on the model's split over micro-batches UBS (128,
# 256, 512) and flash attention off and on, a 512-token prompt and 64 generated tokens at depths 0 and DEPTH (4,096),
# one load. The pick minimizes the time of a typical agent turn at depth DEPTH, PROMPT new prompt tokens plus GEN
# generated (600 each: the agent check's Flint arms average ~3,000 of each per task over ~5 turns). Run from the
# repository root with nothing else on the rig.
# Env: RIG, MODEL, BIN (stock llama-bench on the rig), TS (the split, shares separated by "/"), DIR (rig folder), OUT
#   (local folder); optional UBS, DEPTH, PROMPT, GEN, EXTRA (more llama-bench flags, e.g. "-fit off")
# Output: OUT/tune.jsonl, OUT/tune.err, OUT/tune.md (the grid: pp and tg tokens/s, the modelled turn seconds) and
#   OUT/pick.env (UB=..., FA=on|off)
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BIN:?}" "${TS:?}" "${DIR:?}" "${OUT:?}"
UBS=${UBS:-128,256,512}; DEPTH=${DEPTH:-4096}; PROMPT=${PROMPT:-600}; GEN=${GEN:-600}
mkdir -p "$OUT"
ssh "$RIG" "mkdir -p $DIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
ssh "$RIG" "LOG=/dev/shm/rw_tune.tsv PROC=llama-bench MIN_AVAIL_MB=200 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
    cd $DIR && timeout 1500 $BIN -m $MODEL -ngl 99 -ts $TS -b 2048 -ub $UBS -fa 0,1 -p 512 -n 64 -d 0,$DEPTH -r 1 ${EXTRA:-} \
        -o jsonl > tune.jsonl 2> tune.err; echo \"llama-bench rc=\$?\"
    kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_tune.tsv $DIR/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh
    python3 -I -c 'import json, sys
rows = [json.loads(l) for l in open(\"tune.jsonl\") if l.strip()]
g = {}
for r in rows:
    k = (r[\"n_ubatch\"], r.get(\"flash_attn\") in (True, 1, \"1\", \"on\", \"true\"), r[\"n_depth\"])
    g.setdefault(k, {})[\"pp\" if r[\"n_gen\"] == 0 else \"tg\"] = r[\"avg_ts\"]
print(\"| -ub | flash attention | depth | prompt tok/s | generation tok/s | modelled turn s |\")
print(\"|---:|---|---:|---:|---:|---:|\")
best = None
for (ub, fa, d), v in sorted(g.items()):
    t = $PROMPT / v.get(\"pp\", 1e-9) + $GEN / v.get(\"tg\", 1e-9)
    print(\"| %d | %s | %d | %.1f | %.2f | %.1f |\" % (ub, \"on\" if fa else \"off\", d, v.get(\"pp\", 0), v.get(\"tg\", 0), t))
    if d == $DEPTH and (best is None or t < best[0]):
        best = (t, ub, fa)
print()
print(\"pick: -ub %d, flash attention %s (%.1f s a modelled turn at depth $DEPTH)\" % (best[1], \"on\" if best[2] else \"off\", best[0]))
open(\"pick.env\", \"w\").write(\"UB=%d\\nFA=%s\\n\" % (best[1], \"on\" if best[2] else \"off\"))' > tune.md; tail -n 1 tune.md"
for f in tune.jsonl tune.err tune.md pick.env rw_tune.tsv; do ssh "$RIG" "cat $DIR/$f" > "$OUT/$f"; done
ssh "$RIG" "rm -rf $DIR"
