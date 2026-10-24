# release_matrix: stock llama.cpp, v1 and v2, at stock clocks and with the memory clock tuned

The release's speed table in one run on the rig: every engine on every model file, on all nine cards, first at stock
clocks and then with a memory clock offset on every card. The hardware-tuning result is reported on its own, so the
two blocks never mix: every engine runs at stock clocks before any offset is set, the tuned block compares the engines
under the same offsets, and the tuning gain is each engine against itself.

## What it measures

Per block (stock clocks, tuned), model file and engine:

| Measurement | How |
|---|---|
| Generation | `llama-bench -p 512 -n 128 -r 3`: 128 tokens right after a 512-token prompt in the same process, so the cards are at their working clocks (idle cards sit at a lower clock, [idle cards](../../docs/research/notes/idle_cards.md)) |
| Served | llama-server with the chat command line of the [model pages](../../models/flint/README.md#command-lines) at the full 262,144-token context, one warm-up request, then one request per prompt and seed (up to 1,000 tokens, the makers' sampling: T 1.0, top-p 0.95, top-k 20, min-p 0); the median of the server's own speeds, reasoning tokens included |
| Served, pipelined | the same for v2 with `--pipe-depth 3` |
| Exactness under the offset | the first request on each server is a greedy one; its hash with the offset must equal the stock-clock hash. The engines are deterministic, so with stable memory the output is bit-identical. A changed hash ends the tuned block (`STRICT=1`) |
| Clocks and errors | the cards' clocks, power and temperature every second per block, the highest memory clock under load, and kernel Xid lines |

The defaults compare stock llama.cpp (`d77256999`), the v1 engine (`eeaa59660`) and the v2 build given as `BIN_v2`,
on Unsloth's UD-Q4_K_XL and on K2h-r16. Stock and v1 run without environment switches; v2 runs with
`LLAMA_NO_MOE_DOWN_RESCALE=tg`.

## Files

| File | What it does |
|---|---|
| `run_matrix.sh` | The run, on the rig as root (the tuned block sets NVML offsets) |
| `client.py` | The served requests: one greedy request with its hash, or chat requests with the server's timings |
| `make_table.py` | Parses llama-bench and the requests, and writes `tables.md`: one table per block, the tuning gain, prompt speed and the method, ready to paste into the README |
| `prompts/` | The two chat prompts |
| `test/dryrun.sh` | Runs `run_matrix.sh` against stub programs on any Linux machine (no GPU, no root): a clean run, an output that changes under the offset, a failing llama-bench, an interrupt |

## Running it

The tuned block needs a memory offset that has passed the test of [bench/mem_offset](../mem_offset/README.md) (one step
below its BEST, per the kit's README) and the unpacked kit on the rig. Nothing else may run on the cards.

```bash
scp -r bench/release_matrix rig:/tmp/matrix-kit
ssh rig "cd /tmp/matrix-kit && BIN_v2=/opt/llama-builds/<v2 build>/bin MEM_OFFSET=<MT/s> KIT=/tmp/oc/<kit> setsid nohup ./run_matrix.sh > console.log 2>&1 < /dev/null &"
ssh rig "tail -f /tmp/matrix-kit/console.log"        # ends with the tables
scp -r rig:/tmp/release_matrix/<TAG> ./ && ssh rig "rm -rf /tmp/matrix-kit /tmp/release_matrix"
```

Results go to `/tmp/release_matrix/<TAG>/` on the rig: `tables.md`, `results.tsv` (one line per measurement),
`meta.tsv`, `matrix.log`, `smi-<block>.csv`, `offsets.log` and the raw outputs under `raw/`.

- **Without an offset** it runs the stock-clock block only. `BLOCKS=tuned OUT=<the earlier folder>` adds the tuned block
  later; its exactness check reads the stock-clock hashes from that folder.
- **Per-card offsets:** `MEM_OFFSETS=600,600,400,…` (one per card of `GPUS`). A card without a units file in the kit gets
  the kit's `memoc_steps.sh units` (~1 min) before its offset is set.
- **Shorter runs:** `MODELS=UD-Q4_K_XL=/path` for one file, `SERVE="v2 v2+pipe"` for fewer served configurations,
  `ROUNDS=2` for two llama-bench runs per engine in alternating order.
- **Time (est.):** ~25 min per model and block with the defaults (six model loads of ~2.5 min each, the measurements in
  between), so ~1 h 40 min for two models in both blocks.
- **Safety:** every offset goes back to 0 when the tuned block ends, and whenever the script ends, fails or is
  interrupted (`pkill -TERM -f run_matrix.sh` stops it at once). A memory watchdog stops a server before the 4 GB host
  runs out of memory. A card that hangs needs a reboot, which clears every offset, so run the tuned block with someone
  near the rig's console.

Every setting is an environment variable; the header of `run_matrix.sh` lists them with their defaults.
