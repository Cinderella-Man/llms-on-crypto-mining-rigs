# The MTP draft on the agent check after the `kho=off` boot (S33, 2026-11-04)

What this measured: whether Flash-Next's MTP draft loads and runs on a 4 GB rig once the kernel no longer marks free
memory as CMA ([Q692](../../../quants/experiments.md): `kho=off` on the kernel line, CMA 0 since the morning's boot).
Swift-1.5 GSQ-RCO Q2_0 with agentionai's Gyro MTP draft (`--spec-draft-n-max 3`) on Miner Llama v3's base, the second
rig (nine P106-090, 4 GB of host RAM, headless), in two arms of the agent check:
- `mtpk`: [S12's draft line](../2026-11-03_flash_next_mtp/README.md) without the three workarounds that let S15 run it
  (the draft's token embeddings stay on the host, the default draft micro-batch, pinned host buffers);
- `mtp2k`: S15's `mtp2` line, with `-otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`.

The run was stopped on the owner's word after nine minutes, in favour of the draft-memory switches
([Q564, Q635](../../../quants/experiments.md)). After their smoke tests, S15's line ran the whole agent check
as `mtp2d` on the switches' build (every switch off) with the watchdog's floor at 60 MiB:
[the redo arm](#the-redo-arm-mtp2d) below. Registry rows: Q692 and Q558.

## The redo arm, `mtp2d`

S15's `mtp2` line, unchanged, on the fork branch `pascal-p106-v3-b12076-draftmem` (`1e0caf230`: the speed ladder on
upstream b12076 with its Pascal kernels, plus the draft-memory switches, all off), the whole bugfix task set with one
attempt each and a 40-minute budget, in the same client folder as the earlier arms. The RAM watchdog's `MemAvailable`
floor was 60 MiB; its other limits as before.

| Arm | Engine | Tasks passed | Minutes, all tasks | Minutes per task (median) | First token per turn, s (median, p90) | Prompt tok/s | Generation tok/s | Acceptance |
|---|---|---|---|---|---|---|---|---|
| no draft (S12) | v3's base | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | — |
| `mtp2` (S15) | v3's base | 10 / 11 | 43.7 | 2.7 | 7.85, 21.09 | 55 | 8.8 | 0.701 |
| DFlash2 `dflash2b` (S18) | C63's build of v3's base | 11 / 11 | 41.0 | 2.9 | 8.38, 26.46 | 47 | 9.5 | 0.778 |
| **`mtp2d`** | `-draftmem`, switches off | **11 / 12** | 47.4 | 2.9 | 9.17, 21.05 | 56 | **12.2** | **0.723** |

- **On the nine tasks every arm scored:** 29.2 against 43.0 minutes without a draft (1.48× faster; per task 1.40×,
  1.05–2.22×). It generated 11,611 tokens against 15,521. At the no-draft arm's tokens it would take 1.25–1.39× less
  time, against 1.11–1.15× for `mtp2` and 1.15–1.16× for DFlash2 ([mtp2d_common_tasks.csv](mtp2d_common_tasks.csv)).
- **The engine differs:** the three earlier arms ran on v3's base, `mtp2d` on the ladder build, whose Pascal kernels are
  faster without a draft too. These numbers rank the line on today's engine; they do not split the gain between the
  engine and the draft. A no-draft arm on the same build does that.
- **Host memory over the session** ([mtp2d_session_memory.csv](mtp2d_session_memory.csv)): usable median 476 MiB, lowest
  129; swap 404 MiB on average, 500 at most, at most 97 MiB swapped out in 10 s; memory pressure (full) at most 0.8 %; no
  stop.
- **The one task that did not pass** (`bugfix-javascript-variable-length-quantity`) ran out of time in its second turn:
  one reply reasoned for 7,448 tokens at ~11 tok/s until the client's 12-minute cap cancelled it.
- Per request acceptance: [mtp2d_draft_acceptance.csv](mtp2d_draft_acceptance.csv) (60 requests, 9,782 of 13,527 drafted
  tokens accepted); the report's per-task and per-arm tables: [mtp2d_report.md](mtp2d_report.md).

## The first run (stopped)

| Arm | Load | Usable host memory | Where it ended |
|---|---|---|---|
| `mtpk` (no workarounds) | main model 209 s, then the draft | lowest 1,250 MiB through the main model's load (S15's same line, headless, before `kho=off`: 557) | **did not load.** The draft's load took ~1.35 GB in 4 s: 1,481 → 1,050 → 815 → 166 → 133 MiB, 143 MiB swapped out in that window, memory pressure 0.18 %. The RAM watchdog stopped it at MemAvailable 133 MiB, below its 150 MiB floor |
| `mtp2k` (S15's line) | 227 s, as S15's 230 | lowest 1,514 MiB through the main model's load; 716 → 437 MiB from the end of the load into the first prompt (S15's smoke once loaded: 516) | stopped by the watchdog at MemAvailable 136 MiB, 18 s into the first task's first request (~2,770 tokens, 56 % read): two steps of −189 and −111 MiB while it read the prompt, no swap, pressure 0 % |

- **`kho=off` adds memory at idle (3,290 MiB usable against ~2,520 before) but not where the draft line runs short.**
  Through the main model's load the host keeps ~700 MiB more (1,250 against 557 MiB at the lowest point, the same line). Once the draft is loaded, usable memory sits where S15's run
  had it. The CMA-marked pages were already used by ordinary page cache and anonymous memory. What they refused was
  pinned buffers and the kernel's own allocations, which is what S15's first timed arm ran out of.
- **With CMA 0, `MemAvailable` is the usable memory,** so the agent check's 150 MiB `MemAvailable` floor is now a floor on
  truly usable memory. S15's and S18's finished arms ran lower (S18's DFlash2 arm: median 126 MiB, lowest 95), so the
  floor would stop them too. Later draft arms on these rigs run with `MIN_AVAIL_MB=60`. The watchdog's other limits stay:
  usable under 20 MiB for 2 s, memory pressure over 40 % for 10 s, more than 600 MiB swapped out in 10 s.
- Both arms ran with `MIN_AVAIL_MB=150`.

## Files

| File | What it is |
|---|---|
| [memory_mtpk.csv](memory_mtpk.csv) | `mtpk`'s smoke: the watchdog's samples (every 10 s, then each of the last 15 s): seconds from the server's start, usable MiB (MemAvailable minus CmaFree, here CmaFree is 0), swap used, MiB swapped out in the last 10 s, memory pressure (some, full avg10). The main model loads from 0 to 209 s, then the draft |
| [memory_mtp2k.csv](memory_mtp2k.csv) | the same for `mtp2k`'s timed arm: the main model to 208 s, the draft to 224 s, the first request from 230 s, the stop at 251 s |
| [run/s33_arms.sh](run/s33_arms.sh) | the run: the smoke of `mtpk`, the timed arms, the report |
| [mtp2d_report.md](mtp2d_report.md) | the agent check's report for `mtp2d` against the earlier arms: per arm and per task |
| [mtp2d_common_tasks.csv](mtp2d_common_tasks.csv) | the nine tasks every arm scored: per arm turns, generated tokens, decode, new prompt tokens, prefill, fixed, other and wall seconds, and the time at the no-draft arm's tokens |
| [mtp2d_session_memory.csv](mtp2d_session_memory.csv) | `mtp2d`'s host memory over the session from the watchdog's samples |
| [mtp2d_draft_acceptance.csv](mtp2d_draft_acceptance.csv) | `mtp2d`'s acceptance per request, from the server's log |
| [run/s33.env.example](run/s33.env.example) | the two arms' lines on top of S12's machine file |

## Commands

From the repository root, with S12's machine file filled in
([run/s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example)) and this folder's `s33.env.example`
appended to it:

```bash
ENVFILE=s33.env OUT=/tmp/s33 bash docs/data/2026-11-04_mtp_redo/run/s33_arms.sh

# the redo arm, mtp2d: S15's line on the -draftmem build, floor 60, then the report against the earlier arms
set -a; . s33.env; set +a
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
REL_BIN=/opt/llama-builds/llama.cpp-20261103-1e0caf230-pascal-p106-v3-b12076-draftmem/bin/llama-server REL_ENV= \
  ARGS_mtp2d="$ARGS_mtp2" BUDGET_MIN_mtp2d=40 MIN_AVAIL_MB=60 ARMS=mtp2d STEPS=run bash $RC /tmp/s33/agent-mtp2d
ARMS="nodraft mtp2 dflash2b mtp2d" STEPS="report collect" bash $RC /tmp/s33/agent-report-d
# the tables, on the client, in its results folder (CLIENT_OUT)
python3 -I docs/data/2026-11-03_flash_next_dflash2/run/s18_tables.py --out t --attempts res/attempts.csv \
  --server-log rig/server-mtp2d.log --guard rig/ram_watchdog-mtp2d.tsv --ref nodraft --arms mtp2,dflash2b,mtp2d
```

It uses the smoke test of [2026-11-03_flash_next_dflash2](../2026-11-03_flash_next_dflash2/README.md)
(`run/smoke_dflash.sh`: the agent's first-turn request, then S12's smoke request) and the agent check of
[2026-11-03_flash_next_mtp](../2026-11-03_flash_next_mtp/README.md) (`run/release_check_s12.sh`), both of which start
llama-server through [bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) with its RAM watchdog
([2026-11-04_ram_watchdog](../2026-11-04_ram_watchdog/README.md)).

## Provenance

Swift-1.5 GSQ-RCO Q2_0 and the Gyro MTP draft as in [2026-11-03_flash_next_mtp](../2026-11-03_flash_next_mtp/README.md);
the engine is Miner Llama v3's base (`37e717ff6`). The tables summarize the watchdog's one-second samples; the server logs
and the watchdog's full logs are not published.
