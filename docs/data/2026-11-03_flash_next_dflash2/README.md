# Q561: Qwen3.8-Flash-Next with Inco AI's DFlash2 drafter on nine P106 cards and a 4 GB host (2026-11-03)

The agent check (`bench/agent_check`: a coding agent fixing 12 small bugs) of Swift-1.5 GSQ-RCO Q2_0 on Miner Llama
v3's base with the DFlash fix (`pascal-p106-v3-dflash`, [Q566](../2026-11-03_dflash_fix/README.md)) and Inco AI's
DFlash2 drafter (`--spec-type draft-dflash`), on the second rig. It is compared with the arms without a draft and with
the Gyro MTP draft from [Q226 and Q558](../2026-11-03_flash_next_mtp/README.md), run on the same rig. A DFlash2 drafter
fills a block of positions in one pass from five of the target's layer inputs. It uses the target's token embeddings
and output head, so it needs far less memory than the MTP draft. [Q583](../2026-11-03_verify_cost/README.md) measured
what the main model's k-token verify costs on this rig: 136 ms plus 50–59 ms per extra token. The registry row is
Q561 in [quants/experiments.md](../../../quants/experiments.md).

## Result: DFlash2 generates faster than the MTP draft, and its shorter batch slows the prompts

| Arm (second rig, desktop stopped for the draft arms) | passed | minutes, all tasks | minutes per task (median) | first token per turn, s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | draft acceptance |
|---|---|---|---|---|---|---|---|---|---|
| no draft (Q226, `nodraft`) | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 | — |
| Gyro MTP, 3 drafted (Q558, `mtp2`) | 10 / 11 | 43.7 | 2.7 | 7.85, 21.09 | 55 | 8.8 | 61 / 1 / 0 | 1 | 0.701, 3.10 tokens per step |
| **DFlash2, 2 drafted (`dflash2b`)** | **11 / 11** | **41.0** | **2.9** | **8.38, 26.46** | **47** | **9.5** | 53 / 2 / 0 | 0 | **0.778, 2.56 tokens per step** |

The DFlash2 line: `-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host -md <drafter> --spec-type
draft-dflash --spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99`, with the desktop stopped
([agent/dflash2b/compare.md](agent/dflash2b/compare.md)).

- **On the nine tasks all three arms ran: 34.8 against 43.0 minutes without a draft (1.24×; 1.27× per task, geometric
  mean, 0.61–2.40×)** and 29.0 for the MTP draft (1.48×). All nine passed on every arm. In the 40-minute budget DFlash2
  started 11 tasks and passed all 11. The MTP arm also started 11, and its pig-latin attempt hit the 12-minute cap
  (DFlash2: 3.8 minutes).
- **Generation +22 %, prompts −21 %** ([common_tasks.csv](common_tasks.csv)): 9.45 against 7.77 tok/s (MTP 8.95), and
  prompts at 47.1 against 59.4 (MTP 56.3). With `-b 512` the main model processes a prompt one 512-token micro-batch
  at a time through the nine cards, where `-b 2048` keeps four in flight. `-b 2048` does not fit the host with the
  drafter (below). The second turn's first token (the task's files, ~1,500 new tokens) took 25.3 s against 18.4 and
  20.4 s (median).
- **At equal tokens DFlash2 edges out the MTP draft: 1.15–1.16× against 1.11–1.15×.** At temperature 1.0 each run
  samples different text. On the nine tasks DFlash2 generated 14,009 tokens, against 15,521 without a draft and 10,930
  for MTP. At the arm without a draft's 15,521 tokens and its own decode rate, DFlash2's nine tasks would take
  2,227–2,249 s against 2,582. The range comes from keeping either its own seconds outside decoding or those of the arm
  without a draft. The MTP draft's would take 2,254–2,320 s.
- **Acceptance over the session: 0.778** (10,006 of 12,856 drafted tokens in 60 requests, 0.62–0.94 per request,
  median 0.825; [agent/dflash2b/draft_acceptance.csv](agent/dflash2b/draft_acceptance.csv)), so 2.56 of at most 3
  tokens per step. A step took ~270 ms over the session: 1,483 s of decode for 14,009 tokens on the nine tasks. The MTP
  draft's step takes ~347 ms for 3.10 tokens.
- **The host held, with less to spare than the MTP draft's arm:** MemAvailable minus CmaFree was 126 MiB (median;
  lowest 95) through the session, against 182 (lowest 155) for `mtp2`. Swap held 537 MiB on average (at most 590). No
  swap-out burst or memory pressure above 10 % came after the drafter's load ([session_memory.csv](session_memory.csv)).

**What would make it pay more:** prompts at `-b 2048`'s speed would take ~94 s off the nine tasks (575 s of prefill at
56.3 tok/s instead of 47.1), est. 1.20× at equal tokens. That needs ~100 MiB more host memory than this session had:
the main model's 260 MiB of token embeddings on a card (Q563) would free it. Small-batch kernels that read each weight
once (Q591) lower the ~87 ms a drafted position costs (below), the cost that keeps three drafted tokens from paying.

## Smoke tests: three drafted tokens or two

Each smoke test is one load and two requests with the line above, at `-b 2048` and `--spec-draft-n-max` 3, 2 and 4.
The requests are Q226's coding request (`run/smoke_request.json`, temperature 1.0, up to 768 tokens) and the same
prompt greedy (`run/smoke_greedy.json`, temperature 0, up to 384 tokens). A reference load of the same build without
the drafter (`-b 512`) answered both. All ran under `run/mem_guard.sh` ([smoke.csv](smoke.csv),
[smoke_memory.csv](smoke_memory.csv)).

| `--spec-draft-n-max` | request | tokens | tok/s | acceptance | tokens per step | ms per step |
|---|---|---:|---:|---:|---:|---:|
| 3 | coding, temperature 1.0 (first after the load) | 423 | 8.28 | 0.701 | 3.11 | 375 |
| 3 | greedy | 384 | 8.15 | 0.610 | 2.84 | 348 |
| 2 | coding, temperature 1.0 (first after the load) | 209 | 7.02 | 0.643 | 2.30 | 326 |
| 2 | greedy | 203 | 10.06 | 0.812 | 2.64 | 261 |
| 4 | — | | | | | the rig's watchdog stopped the server (MemAvailable 109 MiB) in the first request's prompt |
| no drafter | coding, temperature 1.0 (first after the load) | 225 | 8.03 | — | 1 | 124 |
| no drafter | greedy | 178 | 8.31 | — | 1 | 120 |

- **It loads in 215–222 s and drafts:** `draft-dflash` with `block_size=8` and `n_extract=5` (five tapped layers), the
  drafter on card 8 with the output head, and no assert on the first request.
- **A drafted position costs ~87 ms a step:** 348 ms with three against 261 ms with two, on the warm greedy request.
  The first request after a load also pays its warm-up. Against 120 ms a token without the drafter, three drafted
  tokens pay only above 2.9 tokens per step, two above 2.2. **The timed arm uses two.** For three to beat two, a step
  would have to yield a third more tokens (348 / 261). The third position would then need to be accepted ~0.85 times
  per step, more often than the second, which is at most ~0.8 here.
- **DFlash2's acceptance matches the MTP draft's, and so does its step:** 3.11 tokens per step with three drafted on
  the coding request, against the MTP draft's 3.07–3.22 on the same request
  ([Q558](../2026-11-03_flash_next_mtp/README.md)). Its step of one draft pass and a 4-token verify (348–375 ms) costs
  about what the MTP draft's three draft passes and the same verify cost (~347 ms). Q583's estimate of a 15–25 ms
  draft pass and a ~337 ms step leaves ~10–40 ms unexplained, which a profile of the DFlash2 step would locate.
- **Greedy text:** the three lines (no drafter, two and three drafted) agree for the first 202 characters of the greedy
  reply. There the line without the drafter ends its reasoning and the two drafting lines continue it, each
  differently: a near-tie that verify batches of different widths, which sum in a different order, tip either way.

## Host memory: `-b 2048` does not hold a session, `-b 512` does

"Usable" is MemAvailable minus CmaFree. This kernel counts 970 MiB of CMA-marked pages in MemAvailable that CUDA's
pinned buffers cannot use ([Q558](../2026-11-03_flash_next_mtp/README.md)). The agent check's watchdog stops
llama-server below 150 MiB of MemAvailable.

| line | ended | lowest usable (MiB) | most swap-out in 10 s (MiB) | usable once loaded / after request 1 / after request 2 (MiB) |
|---|---|---:|---:|---|
| smoke, `-b 2048`, n-max 3 | completed | 124 | 461 | 431 / 239 / 226 |
| smoke, `-b 2048`, n-max 2 | completed | 126 | 391 | 453 / 198 / 274 |
| smoke, `-b 2048`, n-max 4 | watchdog (MemAvailable 109 MiB) in the first 79-token prompt | 93 | 0 | 336 / — / — |
| smoke, no drafter, `-b 512` | completed | 128 | 0 | 441 / 167 / 146 |
| timed arm `dflash2`, `-b 2048`, n-max 2 | watchdog (MemAvailable 140 MiB) at the end of the first task's 2,776-token prompt, 61 s into the arm | 82 (load), 102 (session) | 352 | 805 before the first request, 119 at the stop |
| timed arm `dflash2b`, `-b 512`, n-max 2 | completed, 11 tasks | 72 (load), 95 (session) | 454 | session median 126 |

- **The drafter's load is the tightest moment:** usable memory falls to 72–126 MiB for a few seconds and 350–460 MiB
  go out to swap in 10 s (the guard stops at 600). Then it recovers to 430–850 MiB. The drafter loads after the main
  model, like the MTP draft, but it brings no embeddings of its own.
- **A long prompt is what `-b 2048` cannot hold.** The smoke tests' 79-token prompts left 200–270 MiB usable. The
  first timed arm's first prompt (2,776 tokens, at 68–79 tok/s) took usable memory from 805 to 213 MiB. The step from
  that prompt to generation then took MemAvailable under the watchdog's floor. The drafter's layer-input buffer holds
  5 taps × 2,560 × `-b` floats of pinned memory: 100 MiB at `-b 2048`, 25 MiB at `-b 512`. The run at `-b 512` held
  11 tasks, with prompts of up to 2,780 new tokens.
- **Four drafted tokens crossed the watchdog's floor in the first 79-token prompt.** Once loaded, that line had
  336 MiB usable with 38 MiB in swap, against 431 MiB with 545 MiB in swap at three.

## Files

| File | What |
|---|---|
| `agent/dflash2b/compare.md` | the three arms: per task the result, minutes, turns and tool calls (all / failed / malformed); per arm tasks passed, minutes, minutes per task, time to first token per turn (median, p90), prompt and generation tok/s, tool calls, runaways; the first token and new prompt tokens per turn |
| `agent/dflash2b/table.md` | per arm: attempts, pass@1, minutes per task (median, p90), mean seconds of prefill / decode / fixed cost / other, turns, prompt tokens processed per task, cache hit, TTFT p50, malformed calls, runaways |
| `agent/dflash2b/gonogo.md` | the harness's checks and the server logs' counters (its memory verdict reads the rig's logs of all arms at once) |
| `agent/dflash2b/attempts.csv`, `agent/dflash2b/turns.csv` | one row per attempt and one per request, for every arm in the client folder (`bench/agent_check/report.py`'s input) |
| `agent/dflash2b/draft_acceptance.csv` | per drafting request of `dflash2b`: the server's acceptance, accepted and drafted tokens, mean accepted length |
| `common_tasks.csv` | the nine tasks all three arms ran, per arm: turns, generated tokens, decode seconds and tok/s, new prompt tokens, prefill seconds and tok/s, fixed and other seconds, wall seconds, and the wall time at the reference's generated tokens (low: the arm's own other seconds; high: the reference's) with the speed-ups |
| `smoke.csv` | per smoke test and request: generated tokens, decode seconds, tok/s, drafted and accepted tokens, acceptance, steps (from the server's mean accepted length), tokens and ms per step |
| `smoke_memory.csv` | per smoke test: how it ended, the lowest usable memory, the most swap-out in 10 s, the highest memory pressure, and usable memory, Shmem and swap once loaded and after each request |
| `session_memory.csv` | both timed arms, load and session apart: usable memory (median, lowest), swap (mean, highest), the most swap-out in 10 s, the highest memory pressure, kills by the guard |
| `run/smoke_dflash.sh` | a draft line's smoke test: the server through `bench/agent_check/rig_server.sh` (same monitors and its RAM watchdog, which took over the guard's job), VRAM and host memory once loaded, after the agent's first-turn request (~2,780 tokens, built from the task files; `LONG_REQUEST=off` skips it) and after each request in `REQUESTS`, the logs back with `ram_watchdog-smoke.tsv`; the tables above come from the version before 2026-11-04 (`git log -- docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh`), which started `mem_guard.sh` itself and sent no first-turn request ([Q559, Q607](../2026-11-04_ram_watchdog/README.md)) |
| `run/smoke_request.json`, `run/smoke_greedy.json` | the two smoke requests |
| `run/s18_tables.py` | every table here from the smoke folders, the client's `attempts.csv`, the arm's server log and the guard's log (run where the files are) |
| `run/s18.env.example` | the machine values the scripts read, with the arms' settings, paths as placeholders |

The arms run with [../2026-11-03_flash_next_mtp/run/release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh)
and guard the host with [../2026-11-03_flash_next_mtp/run/mem_guard.sh](../2026-11-03_flash_next_mtp/run/mem_guard.sh).

## How it was made

On 2026-11-03, 19:52–21:15, on the second of the owner's identical rigs (nine P106-090 cards, Celeron 3865U, one 4 GB
SO-DIMM, Ubuntu 24.04 with kernel 7.0.0-42 and driver 580.178.04), headless (`multi-user.target`). Before each load
the rig was checked quiet (no llama process, every card at 0 MiB). The engine: Miner Llama's `pascal-p106-v3-dflash`
at `2deed575e`, which is v3's base `37e717ff6` plus upstream #30927 and a test case ([Q566](../2026-11-03_dflash_fix/README.md)).
The model: Swift-1.5 GSQ-RCO Q2_0, as in Q226
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`). The drafter: `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` (652,101,024 bytes,
SHA-256 `541f4d6ee230747a34d4a2c6890e6e08cc8b19c461226dc1829701a544e46e8a`, checked on the rig) from
[incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF)
`4f5a39be8f0a9e9a573ac9824251830a8f4096d9`, on the same USB 3 stick as the model's first two shards. The client (pi,
the proxy and the tasks' tests in Docker) ran on the 7-GPU quant machine, with the relay through a third machine, as
in Q226. Another agent check's containers ran on the client during the arms.

All lines share release_check.sh's settings: `-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0
--jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics`, one attempt per task,
12 minutes and 40 turns per attempt, no attempt started after 40 minutes, the watchdog at 150 MiB. The arms without a
draft and with the MTP draft are Q226's and Q558's runs in the same client folder; only `dflash2b` ran here.

From the repository root, with the values of `run/s18.env.example` filled in:

```bash
D=docs/data/2026-11-03_flash_next_dflash2
M=docs/data/2026-11-03_flash_next_mtp
set -a; . s18.env; set +a
R="$D/run/smoke_request.json $D/run/smoke_greedy.json"
# smoke tests: three, two and four drafted tokens at -b 2048, then the same build without the drafter
ARGS="$ARGS_dflash2n3" SMOKE_LOGDIR=/mnt/models2/s18/smoke1 REQUESTS="$R" bash $D/run/smoke_dflash.sh out/smoke1-n3
for n in 2 4; do
  ARGS="${ARGS_dflash2n3/--spec-draft-n-max 3/--spec-draft-n-max $n}" SMOKE_LOGDIR=/mnt/models2/s18/smoke-n$n \
    REQUESTS="$R" bash $D/run/smoke_dflash.sh out/smoke-n$n
done
ARGS="-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host" SMOKE_LOGDIR=/mnt/models2/s18/smoke-nodraft \
  REQUESTS="$R" bash $D/run/smoke_dflash.sh out/smoke-nodraft
# a timed arm: the guard on the rig first, then the arm (dflash2 at -b 2048 was stopped by the watchdog; dflash2b)
ssh "$RIG" 'cat > /dev/shm/mem_guard.sh' < $M/run/mem_guard.sh
ssh "$RIG" 'LOG=/dev/shm/mem_guard_s18.log setsid nohup chrt -f 10 bash /dev/shm/mem_guard.sh > /dev/null 2>&1 < /dev/null &'
RIG_LOGDIR=/mnt/models2/s18/agent-logs-b STEPS=run ARMS=dflash2b bash $M/run/release_check_s12.sh out/agent-dflash2b
ssh "$RIG" 'kill $(cat /dev/shm/mem_guard_s18.log.pid); cat /dev/shm/mem_guard_s18.log' > out/mem_guard_timed_dflash2b.log
# the report against Q226's arms, then the tables (on the client, where the files are)
STEPS="report collect" ARMS="nodraft mtp2 dflash2b" bash $M/run/release_check_s12.sh out/agent-dflash2b-report
python3 -I $D/run/s18_tables.py --out tables --smoke 3=out/smoke1-n3 --smoke 2=out/smoke-n2 --smoke 4=out/smoke-n4 \
  --smoke 0=out/smoke-nodraft --attempts "$CLIENT_OUT/res/attempts.csv" --server-log "$CLIENT_OUT/rig/server-dflash2b.log" \
  --guard out/mem_guard_timed_dflash2b.log --ref nodraft --arms mtp2,dflash2b
```

The Swift files' weights are under the Swift Open License 1.0 and qwen-community-1.0, and the drafter's under
Apache-2.0. The tables here hold timings, token counts and memory figures; the agent check's tasks are rebuilt by
`bench/agent_check/build_bugfix.py`.
