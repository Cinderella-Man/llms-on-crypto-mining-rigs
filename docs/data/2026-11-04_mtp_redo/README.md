# The MTP draft on the agent check after the `kho=off` boot (S33, 2026-11-04)

What this measured: whether Flash-Next's MTP draft loads and runs on a 4 GB rig once the kernel no longer marks free
memory as CMA ([Q692](../../../quants/experiments.md): `kho=off` on the kernel line, CMA 0 since the morning's boot).
Swift-1.5 GSQ-RCO Q2_0 with agentionai's Gyro MTP draft (`--spec-draft-n-max 3`) on Miner Llama v3's base, the second
rig (nine P106-090, 4 GB of host RAM, headless), in two arms of the agent check:
- `mtpk`: [S12's draft line](../2026-11-03_flash_next_mtp/README.md) without the three workarounds that let S15 run it
  (the draft's token embeddings stay on the host, the default draft micro-batch, pinned host buffers);
- `mtp2k`: S15's `mtp2` line, with `-otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`.

The run was stopped on the owner's word after nine minutes, in favour of the draft-memory switches
([Q564, Q635](../../../quants/experiments.md)), so it has no speed result. Registry rows: Q692 and Q558.

## Result

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
| [run/s33.env.example](run/s33.env.example) | the two arms' lines on top of S12's machine file |

## Commands

From the repository root, with S12's machine file filled in
([run/s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example)) and this folder's `s33.env.example`
appended to it:

```bash
ENVFILE=s33.env OUT=/tmp/s33 bash docs/data/2026-11-04_mtp_redo/run/s33_arms.sh
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
