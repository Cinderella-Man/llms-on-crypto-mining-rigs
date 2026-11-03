# Q226: Qwen3.8-Flash-Next with an MTP draft on nine P106 cards: the draft does not fit the 4 GB host (2026-11-03)

The agent check (`bench/agent_check`: a coding agent fixing 12 small bugs) of Swift-1.5 GSQ-RCO Q2_0 on Miner Llama
v3's base, without and with agentionai's Gyro MTP draft (`--spec-type draft-mtp`), on a second rig identical to the
first. The arm without the draft repeats Q215's v3 line ([Q215 data](../2026-11-03_flash_next_baseline/README.md)) on
the other rig. The draft line is Q213's ([fit probe](../2026-11-03_flash_next_fit/README.md)), projected to leave
519 MiB on the fullest card. The registry row is Q226 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**The draft line does not load: the host runs out of RAM while the draft loads.** A smoke test before the timed arm (one
short request) was to show that the draft loads and drafts. llama-server loaded the main model in 211 s, as without the
draft, then the draft's 1.9 GB of weights onto card 8 (2,813 → 4,737 MiB). Then the host ran out of memory:

- Within 10 s swap grew from 287 to 991 MiB (61 MiB/s written to swap), and the two Celeron cores spent the next
  8 minutes at 100 % system time ([draft_load.csv](draft_load.csv)).
- The kernel's OOM killer first killed a desktop process (`wireplumber`, at 578 s) and then llama-server (at 723 s).
  The rig came back by itself: no reboot, nothing left on the cards.
- No token was drafted, so the draft's acceptance rate is not measured, and the timed draft arm was not run.
- **Why:** Q213 projected 788 MiB of host buffers for this line against 379 MiB without the draft, plus the draft's
  644 MiB of token embeddings, which stay on the host. Once the main model had loaded, 842 MiB was available
  ([host_memory.csv](host_memory.csv)). Even the line without the draft leaves the host at its limit: 195 MiB at its
  lowest during the agent check, with 1,043 MiB in swap.
- **MemAvailable over-counted what was left.** In the OOM report, 967 MiB of the 1,054 MiB still free sat in
  pageblocks marked CMA, which unmovable kernel allocations and long-term pinned buffers cannot use. The kernel
  reports `CmaTotal: 0 kB` and reserved no CMA at boot, yet `CmaFree` reads 411 MiB with the rig idle. The 1,049 MiB
  of shared memory in the report was freed when llama-server died. So MemAvailable read ~949 MiB while the host was
  already out of memory, and the agent check's RAM watchdog (stop the server below 150 MiB) never fired. Its 0.5 s loop
  and the 2 s MemAvailable logger then went without CPU for 8 minutes.

**Without the draft, the second rig reproduces the first rig's v3 arm** (the same build, line and 40-minute budget;
[agent/nodraft/](agent/nodraft/compare.md)):

| Rig, arm | tasks passed (of those started) | minutes, all tasks | minutes per task (median) | first token per turn, s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| second rig, no draft (this run) | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| first rig, v3 (Q215) | 9 / 9 | 44.2 | 3.0 | 7.65, 19.80 | 61 | 7.8 | 47 / 0 / 0 | 1 |

- **Identical generation speed (7.8 tok/s over the session)**, prompts within 3 %, and the same nine tasks started
  before the 40-minute budget ran out (the last three were not started on either rig). Per-task minutes differ with
  the sampled text (temperature 1.0): the second rig took 8 turns on the affine cipher against 6, and finished the
  variable-length quantity in 6.9 minutes where the first rig hit the 12-minute cap.
- **The agent check's own verdict is NO-GO on memory** (lowest MemAvailable 195 MB, swap 32 → 1,043 MB) and on first
  token against prefill (p90 1.72 s over the server's prefill: the 0.63 s checkpoint restore and the per-request fixed
  cost, as in Q215); infrastructure, tool calls (0 malformed of 47) and the cache (96 % median hit from turn 2) are GO.
- **VRAM** ([vram.csv](vram.csv)): the fullest card (card 6) holds 5,323 MiB once loaded and 5,373 MiB at most, against
  5,325–5,333 MiB on the first rig. The draft line had loaded 4,625–5,509 MiB on cards 0–7 when it stopped; card 4
  still had ~1.4 GB free.

**What would let the draft in** (Q558): the draft's 644 MiB of token embeddings on card 4 instead of the host
(`-otd`), fewer context checkpoints (each of the four holds 113 MiB in host RAM), a smaller micro-batch, or a second
4 GB SO-DIMM (the slot is free; Q86). The watchdog needs a measure the CMA pages cannot fool (Q559), and the CMA
accounting on this kernel needs its own look (Q560).

## Files

| File | What |
|---|---|
| `agent/nodraft/compare.md` | per task: result, minutes, turns, tool calls (all / failed / malformed); per arm: tasks passed, minutes, minutes per task, time to first token per turn (median, p90), prompt and generation tok/s, tool calls, runaways; time to first token and new prompt tokens per turn |
| `agent/nodraft/table.md` | attempts, pass@1, minutes per task (median, p90), mean seconds of prefill / decode / fixed cost / other, turns, prompt tokens processed per task, cache hit, TTFT p50, malformed calls, runaways |
| `agent/nodraft/gonogo.md` | the harness's checks (infrastructure, proxy rows, first token against prefill, tool calls, cache, memory), the server log's counters and the overall verdict |
| `agent/nodraft/validation.txt` | the 12 tasks validated in the sandbox image before the run |
| `agent/nodraft/attempts.csv` | one row per attempt (`bench/agent_check/report.py`'s input): result, tokens, turns, wall, prefill, decode and fixed seconds, cache, tool-call counts, test counts |
| `agent/nodraft/turns.csv` | one row per request: time to first token, end time, new and cached prompt tokens, prompt ms, generated tokens, decode ms, finish reason, tool calls |
| `host_memory.csv` | per line (no draft, draft): MemAvailable and swap used at the start, once loaded (the draft line: main model loaded, draft loading), at their lowest and highest, and the last sample |
| `draft_load.csv` | the draft line's vmstat rows every 10 s from the server's start: run queue, swap used, free, swap in and out, disk reads, CPU split, and the latest MemAvailable sample with its age (the logger stalled from 220 s to 723 s) |
| `vram.csv` | MiB used per card, once loaded and at the most, for both lines |
| `run/release_check_s12.sh` | `bench/agent_check/release_check.sh` with two changes: it runs from outside the bench folder (`BENCH`), and it takes any arm name A with `ARGS_A` set (REL_BIN with those settings), so two checks can share one client: the client names an arm's containers `acrel-<arm>-*` |
| `run/smoke_mtp.sh`, `run/smoke_request.json` | the draft line's smoke test: the server through `bench/agent_check/rig_server.sh` (same monitors and watchdog), VRAM once loaded, one short coding request, the logs back |
| `run/s12_tables.sh` | the three memory tables from the rig's monitor logs (run where the logs are) |
| `run/s12.env.example` | the machine values both scripts read, paths as placeholders |

## How it was made

On 2026-11-03, on the second of the owner's identical rigs (nine P106-090 cards, Celeron 3865U, one 4 GB SO-DIMM,
Ubuntu 24.04 with kernel 7.0.0-42 and driver 580.178.04, the desktop session running). Nothing else ran on the rig
during each arm; that was checked before each one (no GPU processes, 5 MiB per card, no other jobs). Miner Llama v3's
base `37e717ff6` (the v2.0 commit, the tarball rebuilt on 2026-11-02 that the first rig also runs). The model: Swift-1.5
GSQ-RCO Q2_0 as three shards (shards 1–2 on a USB 3 stick, the n-gram table alone in shard 3 on the internal SSD;
[Q435](../2026-11-03_ssd_room/README.md)), from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`. The draft: `mtp-Qwen3.8-Flash-Next-draft.gguf` (2,697,439,648 bytes,
SHA-256 `008183784ac5aa2cdcd3d7777e4e6febb0645ae4b8d1da1c065b2149c1d6391b`) from
[agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)
`273edc1d2ffae5241fe3cab69568d682dad4c718`. The client (pi, the proxy and the tasks' tests in Docker) ran on the 7-GPU
quant machine at normal CPU priority, with the relay through a third machine, as in Q215.

Both lines share release_check.sh's settings: `-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0
--jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics`, one attempt per task,
12 minutes and 40 turns per attempt, no attempt started after 40 minutes, the watchdog at 150 MiB.

From the repository root, with the values of `run/s12.env.example` filled in:

```bash
D=docs/data/2026-11-03_flash_next_mtp
set -a; . s12.env; set +a
# prep: the bench files to the client, the sandbox image, the 12 tasks validated (12 of 12 valid)
CPU_SHARES=256 STEPS=prep ARMS=nodraft bash $D/run/release_check_s12.sh out/agent-prep
# the arm without the draft
STEPS=run ARMS=nodraft bash $D/run/release_check_s12.sh out/agent-nodraft
# the draft line's smoke test (here it ends in the OOM kill; the script stops the server and fetches the logs)
SMOKE_LOGDIR=/mnt/models2/s12/smoke REQUEST=$D/run/smoke_request.json bash $D/run/smoke_mtp.sh out/smoke
# the timed draft arm, had the smoke test passed, then both arms' reports:
#   STEPS=run ARMS=mtp bash $D/run/release_check_s12.sh out/agent-mtp
#   STEPS="report collect" ARMS="nodraft mtp" bash $D/run/release_check_s12.sh out/agent
STEPS="report collect" ARMS=nodraft bash $D/run/release_check_s12.sh out/agent
# the memory tables, on the client (the smoke test's logs copied there first)
ssh "$CLIENT" "NODRAFT=$CLIENT_OUT/rig DRAFT=<smoke logs> OUT=<tables folder> nice -n 19 bash -s" < $D/run/s12_tables.sh
# the kernel's view of the OOM
ssh "$RIG" 'journalctl -k --since "<start>" -o cat | grep -E "Mem-Info|free_cma|shmem|Killed process"; grep Cma /proc/meminfo'
```

The files' weights are under the Swift Open License 1.0 and qwen-community-1.0 (the Swift files) and
qwen-community-1.0 (the draft); the tables here hold timings, token counts and memory figures, and the agent check's
tasks are rebuilt by `bench/agent_check/build_bugfix.py`.
