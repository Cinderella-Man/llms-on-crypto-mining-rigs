# Q226 and Q558: Qwen3.8-Flash-Next with an MTP draft on nine P106 cards and a 4 GB host (2026-11-03)

The agent check (`bench/agent_check`: a coding agent fixing 12 small bugs) of Swift-1.5 GSQ-RCO Q2_0 on Miner Llama
v3's base, without and with agentionai's Gyro MTP draft (`--spec-type draft-mtp`), on a second rig identical to the
first. The arm without the draft repeats Q215's v3 line ([Q215 data](../2026-11-03_flash_next_baseline/README.md)) on
the other rig. The draft line starts from Q213's ([fit probe](../2026-11-03_flash_next_fit/README.md)), projected to
leave 519 MiB on the fullest card. As published, that line runs the 4 GB host out of memory while the draft loads
(Q226's first attempt, below). Q558 found the changes that make it load and draft, then ran the timed draft arms. The
registry rows are Q226 and Q558 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**The Gyro MTP draft runs a full agent check on the 4 GB host** with four changes: its token embeddings on card 4,
its own micro-batch, the main model's token embeddings out of pinned memory, and the desktop stopped (Q558's step 5,
arm `mtp2`, [agent/mtp2/compare.md](agent/mtp2/compare.md)). The draft adds ~15 % to generation speed:

| Arm | tasks passed (of those started) | minutes, all tasks | minutes per task (median) | first token per turn, s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | draft acceptance |
|---|---|---|---|---|---|---|---|---|---|
| no draft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 | — |
| MTP draft (`mtp2`) | 10 / 11 | 43.7 | 2.7 | 7.85, 21.09 | 55 | 8.8 | 61 / 1 / 0 | 1 | 0.701 |

The line: `-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <draft> --spec-type draft-mtp
--spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`, with the desktop stopped.

- **On the nine tasks both arms ran: 29.0 against 43.0 minutes (1.48×), 1.44× per task** (geometric mean, 0.94–2.28×),
  all nine passed on both. The draft arm started 11 tasks in the 40-minute budget against 9; the eleventh (pig latin)
  hit the 12-minute cap after 14 turns.
- **Most of that gain comes from the sampled text, not the draft** ([common_tasks.csv](common_tasks.csv)). On those nine
  tasks the draft arm generated 30 % fewer tokens (10,930 against 15,521) in 46 turns against 51. At temperature 1.0
  each run samples different text, and a draft does not change what the model samples. The draft's own effect:
  generation at 8.95 against 7.77 tok/s (+15 %) and prompts at 56.3 against 59.4 tok/s (−5 %). The fixed cost per
  request was also lower, 6 against 75 s in all, because the arm without the draft, with the desktop running, read its
  checkpoints back from swap. Had the draft arm generated the same 15,521 tokens, its nine tasks would have taken
  ~2,250–2,320 s against 2,582: **1.11–1.15× from the draft.**
- **Acceptance 0.701 over the session** (11,221 of 15,999 drafted tokens in 64 requests;
  [agent/mtp2/draft_acceptance.csv](agent/mtp2/draft_acceptance.csv)).
- **The host held:** MemAvailable minus CmaFree 182 MiB (median, lowest 155), swap 917 MiB on average (at most 979),
  memory pressure at most 5 % ([session_memory.csv](session_memory.csv)). The check's own memory verdict (NO-GO: lowest
  MemAvailable 195 MB, swap up to 1,475 MB) reads the rig's log of all three arms at once; those two figures are the
  other arms'. The fullest card (card 3) held 5,549 MiB at most ([vram.csv](vram.csv)).

**What the draft buys on v3's base:** its acceptance is high, but a verification step costs nearly three tokens'
time on this rig.

- **Acceptance 0.69–0.74** on the smoke request (0.741 and 0.690 in two loads) and **0.714 over the 12 drafting
  requests** of the first timed arm (2,550 of 3,573 drafted tokens), with 3.07–3.22 tokens per verification step
  (`--spec-draft-n-max 3`, so at most 4).
- **A step costs ~350 ms unloaded:** on the smoke request, 42.0 s for 391 tokens in 121 steps (347 ms a step) and
  40.5 s for 348 tokens in 113 steps (359 ms), where a token without the draft costs ~127 ms (7.9 tok/s over the
  47 requests of the arm without the draft, 6.4–9.9 per request). A step (three draft passes, then the main model on
  four tokens) therefore costs ~2.8 tokens' time and yields ~3.1: **9.3 and 8.6 tok/s on the smoke request against 7.9
  without the draft, 1.1–1.2×.** The four-token pass reads up to four times the routed experts and the BF16
  hyper-connection projections for each token (Q215's profile: 28 and 43 of the 136 ms of a token); Q583 splits the
  step's cost into its parts. Over `mtp2`'s session a step took
  ~347 ms as well (1,222 s of decode for 10,930 tokens at 3.10 tokens a step).
- **Prompts are slower with the draft:** the draft also reads every prompt token. In `mtp2`, prompts ran at 56.3
  against 59.4 tok/s (−5 %). In the first timed arm, swapping, they ran at 34.8 against 59.2 tok/s, and the first
  token took 19.5 against 8.4 s per turn (median), until its OOM.

## Fitting the draft into the 4 GB host (Q558)

One change at a time, each a load-and-request smoke test ([fit_steps.csv](fit_steps.csv)), with
[run/mem_guard.sh](run/mem_guard.sh) watching the host (below):

| Step | Line | Desktop | Result | Lowest MemAvailable minus CmaFree, main load / draft load (MiB) | Most swap-out in 10 s at the draft load (MiB) | Once loaded: MemAvailable minus CmaFree, swap (MiB) |
|---|---|---|---|---|---|---|
| 0 (Q226) | Q213's draft line | running | host out of memory as the draft loads; OOM kill after 8 minutes | — | 704 | — |
| 1 | step 0 + `-otd token_embd=CUDA4` | running | stopped by the guard 5 s into the draft load | 481 / 96 | 654 | — |
| 3 | step 1, desktop stopped | stopped | loads (236 s) and drafts; the timed arm ends in an OOM kill after 19 minutes | 557 / 140 | 557 | 310, 675 |
| 5 | step 3 + `--no-host --spec-draft-n-ubatch 128` | stopped | loads (229 s) and drafts, no swap during the load; the timed arm completes | 835 / 278 | 0 | 516, 42 |

- **The projection** ([fit_projection_draft.csv](fit_projection_draft.csv), from llama-server's fit step on the real
  files): `-otd token_embd=CUDA4` puts the draft's 644 MiB of Q8_0 token embeddings on card 4 and shrinks its host
  compute buffer from 409 to 101 MiB, since the row lookup runs on a card. The draft's host share falls from 1,053 to
  101 MiB. `--spec-draft-n-ubatch 128` (the draft context's micro-batch; the main model keeps `-ub 512`) takes it to
  25 MiB. Card 4 then holds 5,212 MiB projected (5,269–5,275 measured), card 8 4,908–5,101 (4,973–5,169).
- **Pinned memory is what runs out, and this kernel keeps a quarter of the RAM from it.** CUDA's pinned host buffers
  (`cudaMallocHost`: shared anonymous mappings, counted as Shmem) can't use pageblocks marked CMA, and this kernel
  marks 485 of them, 970 MiB of the 3.9 GB, without a CMA area (`CmaTotal: 0 kB`; [pageblocks.csv](pageblocks.csv),
  Q560). MemAvailable counts those pages. When pinned memory is short, the kernel swaps to find pages outside them and
  CmaFree climbs: 0 → 734 MiB in step 1's last 7 s.
- **The loader's staging and the main model's embeddings are pinned too.** With `-lm dio` the loader
  (`load_all_data`) stages each buffer's tensors through 4 pinned buffers of 64 MiB, and the draft loads after the main
  model's context already holds its pinned buffers: its 379 MiB compute buffer and, without `--no-host`, its 260 MiB of
  Q3_K token embeddings in a pinned host buffer. Shmem reaches ~660 MiB during the main model's load. `--no-host` puts
  those embeddings in ordinary memory, which can use the CMA pages; with the draft's smaller micro-batch, Shmem once
  loaded falls from 839 to 503 MiB, the 260 + 76 MiB expected.
- **Stopping the desktop gave ~76 MiB of usable memory at the main model's lowest point** (557 against 481 MiB), far
  less than its +280 MB of MemAvailable (Q83), most of which is CMA pages.
- **Steps 2 and 4 were planned and not run:** step 2, `--ctx-checkpoints 2`, allocates its checkpoints during
  requests, so it cannot change a load failure; step 4, `-ub 256`, and step 2 both change the main model's line. Step
  5's two options free the same kind of memory and leave the main model's computation as in the arm without the
  draft.

**The first timed arm (step 3's line) ended in the kernel's OOM killer after 19.4 minutes** (one task passed: the
affine cipher in 8.1 minutes, 8.2 without the draft; [agent/mtp/compare.md](agent/mtp/compare.md)). Through the
session the host had 154 MiB outside the CMA pages (median; lowest 117), 1.36 GB in swap on average (most 1.47 GB) and
memory pressure up to 59 % (full avg10, above 20 % in 15 % of the samples); checkpoint restores took 2.8 s against
0.63 s in Q215, their pages read back from swap. A kernel allocation from CUDA's event thread then failed with
1,178 MiB free, 955 MiB of it CMA-marked. llama-server, at `oom_score_adj` 1000, was the only process killed
([session_memory.csv](session_memory.csv)).

**Guarding the rig:** `run/mem_guard.sh` runs on the rig at real-time priority, gives each llama-server
`oom_score_adj` 1000 and kills it on more than 600 MiB of swap-out in 10 s or memory pressure above 60 % for 15 s. The
defaults sit between the arm without the draft (at most 397 MiB of swap in 10 s, right after loading) and Q226's
failed load (704 MiB, then 8 minutes of reclaim). It stopped step 1 in 5 s, with the rig responsive throughout.

## The first attempt: the draft line runs the host out of memory (Q226)

llama-server loaded the main model in 211 s, as without the draft, then the draft's 1.9 GB of weights onto card 8
(2,813 → 4,737 MiB). Then the host ran out of memory:

- Within 10 s swap grew from 287 to 991 MiB (61 MiB/s written to swap), and the two Celeron cores spent the next
  8 minutes at 100 % system time ([draft_load.csv](draft_load.csv)).
- The kernel's OOM killer first killed a desktop process (`wireplumber`, at 578 s) and then llama-server (at 723 s).
  The rig came back by itself: no reboot, nothing left on the cards.
- Q213 projected 788 MiB of host buffers for this line against 379 MiB without the draft, plus the draft's 644 MiB of
  token embeddings on the host. Once the main model had loaded, 842 MiB was available ([host_memory.csv](host_memory.csv)).
- **MemAvailable over-counted what was left.** In the OOM report, 967 MiB of the 1,054 MiB still free sat in
  CMA-marked pageblocks. The 1,049 MiB of shared memory in the report (the pinned buffers) was freed when llama-server
  died. MemAvailable read ~949 MiB while the host was already out of memory, and the agent check's RAM watchdog (stop
  the server below 150 MiB) never fired. Its 0.5 s loop and the 2 s MemAvailable logger then went without CPU for
  8 minutes.

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
  5,325–5,333 MiB on the first rig.

## Files

| File | What |
|---|---|
| `agent/nodraft/compare.md`, `agent/mtp/compare.md`, `agent/mtp2/compare.md` | per task: result, minutes, turns, tool calls (all / failed / malformed); per arm: tasks passed, minutes, minutes per task, time to first token per turn (median, p90), prompt and generation tok/s, tool calls, runaways; time to first token and new prompt tokens per turn. `mtp/` and `mtp2/` compare each draft arm with `nodraft` |
| `agent/*/table.md` | attempts, pass@1, minutes per task (median, p90), mean seconds of prefill / decode / fixed cost / other, turns, prompt tokens processed per task, cache hit, TTFT p50, malformed calls, runaways |
| `agent/*/gonogo.md` | the harness's checks (infrastructure, proxy rows, first token against prefill, tool calls, cache, memory), the server log's counters and the overall verdict |
| `agent/nodraft/validation.txt` | the 12 tasks validated in the sandbox image before the run |
| `agent/*/attempts.csv` | one row per attempt (`bench/agent_check/report.py`'s input): result, tokens, turns, wall, prefill, decode and fixed seconds, cache, tool-call counts, test counts |
| `agent/*/turns.csv` | one row per request: time to first token, end time, new and cached prompt tokens, prompt ms, generated tokens, decode ms, finish reason, tool calls |
| `agent/*/draft_acceptance.csv` | the draft arms: per request, the server's drafted and accepted tokens and accepted length |
| `common_tasks.csv` | the nine tasks both arms ran, per arm: turns, generated tokens, decode seconds and tok/s, new prompt tokens, prefill seconds and tok/s, fixed and other seconds, wall seconds (sums of `agent/mtp2/attempts.csv`) |
| `fit_steps.csv` | Q558's steps: line, desktop, result, load seconds; lowest MemAvailable minus CmaFree during the main model's and the draft's load; swap and swap-out at the draft load; MemAvailable, CmaFree and swap once loaded and after the smoke request; the smoke request's acceptance, prompt and generation tok/s |
| `fit_projection_draft.csv` | llama-server's projection for the draft line, with `-otd` and with the draft's micro-batch: the draft's MiB on the host and cards 4 and 8 (weights, context, compute) |
| `session_memory.csv` | the timed draft arms: load and session host memory from the guard (MemAvailable minus CmaFree, swap, swap-out, memory pressure) and how each ended |
| `pageblocks.csv` | `/proc/pagetypeinfo`'s pageblocks (2 MiB) per migrate type and zone on the idle rig, with the desktop running and stopped |
| `host_memory.csv` | Q226's first attempt, per line (no draft, draft): MemAvailable and swap used at the start, once loaded (the draft line: main model loaded, draft loading), at their lowest and highest, and the last sample |
| `draft_load.csv` | Q226's first draft line: vmstat rows every 10 s from the server's start: run queue, swap used, free, swap in and out, disk reads, CPU split, and the latest MemAvailable sample with its age (the logger stalled from 220 s to 723 s) |
| `vram.csv` | MiB used per card, once loaded and at the most, for each line |
| `run/release_check_s12.sh` | `bench/agent_check/release_check.sh` with two changes: it runs from outside the bench folder (`BENCH`), and it takes any arm name A with `ARGS_A` set (REL_BIN with those settings), so two checks can share one client: the client names an arm's containers `acrel-<arm>-*` |
| `run/smoke_mtp.sh`, `run/smoke_request.json` | a draft line's smoke test: the server through `bench/agent_check/rig_server.sh` (same monitors and watchdog), VRAM, `/proc/meminfo` and the server's `/proc/<pid>/status` once loaded and after the agent's first-turn request (~2,780 tokens, built from the task files; `LONG_REQUEST=off` skips it) and one short coding request, the logs back with `ram_watchdog-mtp-smoke.tsv`; the runs above used the version before 2026-11-04 (`git log -- docs/data/2026-11-03_flash_next_mtp/run/smoke_mtp.sh`), without the first-turn request ([Q559, Q607](../2026-11-04_ram_watchdog/README.md)) |
| `run/mem_guard.sh` | the host-memory guard (above); its job (and a floor on MemAvailable minus CmaFree) is now `scripts/ram_watchdog.sh`, started by `rig_server.sh` ([Q559](../2026-11-04_ram_watchdog/README.md)) |
| `run/s12_tables.sh` | Q226's first-attempt memory tables from the rig's monitor logs (run where the logs are) |
| `run/s12.env.example` | the machine values both scripts read, paths as placeholders |

## How it was made

On 2026-11-03, on the second of the owner's identical rigs (nine P106-090 cards, Celeron 3865U, one 4 GB SO-DIMM,
Ubuntu 24.04 with kernel 7.0.0-42 and driver 580.178.04). Nothing else ran on the rig during each arm; that was
checked before each one (no GPU processes, nothing on the cards, no other jobs). Miner Llama v3's base `37e717ff6` (the
v2.0 commit, the tarball rebuilt on 2026-11-02 that the first rig also runs). The model: Swift-1.5 GSQ-RCO Q2_0 as
three shards (shards 1–2 on a USB 3 stick, the n-gram table alone in shard 3 on the internal SSD;
[Q435](../2026-11-03_ssd_room/README.md)), from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`. The draft: `mtp-Qwen3.8-Flash-Next-draft.gguf` (2,697,439,648 bytes,
SHA-256 `008183784ac5aa2cdcd3d7777e4e6febb0645ae4b8d1da1c065b2149c1d6391b`) from
[agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)
`273edc1d2ffae5241fe3cab69568d682dad4c718`. The client (pi, the proxy and the tasks' tests in Docker) ran on the 7-GPU
quant machine at normal CPU priority, with the relay through a third machine, as in Q215; during the draft arms other
checks' and scoring jobs' containers also ran there.

All lines share release_check.sh's settings: `-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0
--jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics`, one attempt per task,
12 minutes and 40 turns per attempt, no attempt started after 40 minutes, the watchdog at 150 MiB. The arm without the
draft ran with the desktop running; Q558's steps 3 and 5 and both draft arms with it stopped
(`systemctl isolate multi-user.target`, back with `systemctl isolate graphical.target`; the default boot target
unchanged). Q83 measured no speed difference between the two.

From the repository root, with the values of `run/s12.env.example` filled in:

```bash
D=docs/data/2026-11-03_flash_next_mtp
set -a; . s12.env; set +a
# prep: the bench files to the client, the sandbox image, the 12 tasks validated (12 of 12 valid)
CPU_SHARES=256 STEPS=prep ARMS=nodraft bash $D/run/release_check_s12.sh out/agent-prep
# the arm without the draft
STEPS=run ARMS=nodraft bash $D/run/release_check_s12.sh out/agent-nodraft
# Q226's draft line, smoke test (it ends in the OOM kill; the script stops the server and fetches the logs)
SMOKE_LOGDIR=/mnt/models2/s12/smoke REQUEST=$D/run/smoke_request.json bash $D/run/smoke_mtp.sh out/smoke

# Q558: the guard on the rig, before each load
ssh "$RIG" 'mkdir -p /mnt/models2/s15 && cat > /mnt/models2/s15/mem_guard.sh' < $D/run/mem_guard.sh
ssh "$RIG" 'cd /mnt/models2/s15; LOG=/dev/shm/mem_guard.log setsid nohup chrt -f 10 bash mem_guard.sh > /dev/null 2>&1 < /dev/null &'
# the projection for the draft line, with -otd and with the draft's micro-batch (seconds each, no weights loaded)
B="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3"
V=llama.cpp-20261102-37e717ff6-pascal-p106-v3
BUILDS=/opt/llama-builds OUT=out/fit_probe.md KEEP=200 \
  CONFIGS="A|$V|65536|$B;B|$V|65536|$B -otd token_embd=CUDA4;C|$V|65536|$B -otd token_embd=CUDA4 --spec-draft-n-ubatch 128" \
  bash scripts/fit_probe.sh
# step 1; step 3 (the same with the desktop stopped); step 5
ARGS_mtp="$ARGS_mtp -otd token_embd=CUDA4" SMOKE_LOGDIR=/mnt/models2/s15/smoke-1 \
  REQUEST=$D/run/smoke_request.json bash $D/run/smoke_mtp.sh out/smoke-1
ssh "$RIG" 'systemctl isolate multi-user.target'
ARGS_mtp="$ARGS_mtp -otd token_embd=CUDA4" SMOKE_LOGDIR=/mnt/models2/s15/smoke-3 \
  REQUEST=$D/run/smoke_request.json bash $D/run/smoke_mtp.sh out/smoke-3
ARGS_mtp="$ARGS_mtp -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host" SMOKE_LOGDIR=/mnt/models2/s15/smoke-5 \
  REQUEST=$D/run/smoke_request.json bash $D/run/smoke_mtp.sh out/smoke-5
# the timed draft arms: step 3's line (mtp), step 5's (mtp2)
ARGS_mtp="$ARGS_mtp -otd token_embd=CUDA4" STEPS=run ARMS=mtp bash $D/run/release_check_s12.sh out/agent-mtp
ARGS_mtp2="$ARGS_mtp -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host" BUDGET_MIN_mtp2=40 STEPS=run ARMS=mtp2 \
  bash $D/run/release_check_s12.sh out/agent-mtp2
ssh "$RIG" 'systemctl isolate graphical.target'
# the reports (each draft arm against nodraft) and the draft's acceptance per request from the server log
STEPS="report collect" ARMS="nodraft mtp" bash $D/run/release_check_s12.sh out/agent-mtp-report
STEPS="report collect" ARMS="nodraft mtp2" bash $D/run/release_check_s12.sh out/agent-mtp2-report
grep 'draft acceptance' out/agent-mtp2-report/rig/server-mtp2.log
# the kernel's view of an OOM, and the pageblock types
ssh "$RIG" 'journalctl -k --since "<start>" -o cat | grep -E "Mem-Info|free_cma|shmem|Killed process"; grep Cma /proc/meminfo'
ssh "$RIG" 'sed -n "/Number of blocks type/,\$p" /proc/pagetypeinfo'
```

The files' weights are under the Swift Open License 1.0 and qwen-community-1.0 (the Swift files) and
qwen-community-1.0 (the draft); the tables here hold timings, token counts and memory figures, and the agent check's
tasks are rebuilt by `bench/agent_check/build_bugfix.py`.
