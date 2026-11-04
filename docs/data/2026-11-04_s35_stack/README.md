# The whole stack on the agent check (S35, 2026-11-04)

What this measured: the coding-agent check ([bench/agent_check](../../../bench/agent_check)) of Qwen3.8-Flash-Next on
the first 4 GB rig of nine P106-090 with every change of the day that passed its gate, one arm. Registry rows Q899 and
Q825 in [quants/experiments.md](../../../quants/experiments.md). Against S34's arms on the second rig
([2026-11-04_stacked_gains](../2026-11-04_stacked_gains/README.md)), this arm changes the engine and adds the hop cast:

- the file **pmix-q2-fast** (Swift-1.5 GSQ-RCO Q2_0 with a Pascal type mix, [2026-11-04_pascal_mix](../2026-11-04_pascal_mix/README.md)), as S34;
- Miner Llama's **`-kshape`** build (`39c3ab5f1`: the routed down projection on the multi-token kernel, Q825, and the
  k-column shapes, Q823, on `-ladder3`) with the k-column kernels' 32 KB budget and the dispatch table off
  (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`, the fastest setting on the rig:
  [2026-11-04_kshape](../2026-11-04_kshape/README.md));
- S33's MTP draft line with the draft head's shortlist (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`), S34's best arm;
- the residual in Q8_0 across the eight card boundaries through the staged hop (`LLAMA_HOP_CAST=q8_0
  GGML_CUDA_HOP_STAGED=1`), whose KLD gate passed (C200k: no measurable change against the F32 hops).

The bugfix task set, one attempt each, a 40-minute budget; the host booted with `kho=off`, the RAM watchdog's
`MemAvailable` floor at 60 MiB.

## Result

| Arm | Passed | Minutes, all tasks | Minutes per task (median) | First token per turn, s (median, p90) | Prompt tok/s | Generation tok/s | Acceptance |
|---|---|---|---|---|---|---|---|
| **`s35stack`** (this rig) | **12 / 12** | 34.8 | **2.2** | **7.42, 17.34** | **68** | **18.8** | 0.708 |
| `s34mtps` (S34, the second rig) | 12 / 12 | 32.4 | 2.3 | 8.35, 19.99 | 58 | 16.0 | 0.705 |
| `s34nd` (S34, no draft) | 11 / 12 | 47.3 | 2.4 | 7.84, 17.83 | 63 | 14.3 | — |
| S12's no-draft arm (v3's base, Swift Q2_0) | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | — |

- **Generation 18.8 tok/s over the session, +17.5 % on S34's best arm** with the same file, draft line and acceptance
  (0.708 against 0.705): the engine (Q825) and the hop cast. Prompts run 17 % faster (68 against 58 tok/s) and the median
  first token comes 0.9 s sooner. Against S12's no-draft arm on v3's base the session's generation is 2.4× faster.
- **The hop cast alone, without a draft** (`llama-bench -p 0 -n 128`, three repetitions, [bench.tsv](bench.tsv)):
  14.98 ± 0.20 tok/s without it, **17.06 ± 0.04 with it** (−8.1 ms a token): about three times what the bytes alone save
  ([C140](../2026-11-04_megakernel_rig/README.md): 0.455 → 0.127 ms a boundary), so the staged hop also saves host-side
  time between the cards.
- Tool calls 62, failed 1, malformed 0; no runaway. Minutes for all tasks depend on what each attempt generated, so the
  per-token columns compare the arms.
- **Host memory** (the watchdog's 2,744 samples): usable median 749 MiB, lowest 87 MiB; swap used mean 631 MiB, highest
  1,091 MiB; at most 107 MiB swapped out in 10 s; memory pressure (full) at most 1.1 %; the watchdog never fired. The
  fit: every card keeps 823 MiB or more at 64k context (llama.cpp's own projection).

## Files

| File | What it is |
|---|---|
| [report.md](report.md) | the agent check's report: per task (result, minutes, turns, tool calls), per arm, time to first token per turn |
| [bench.tsv](bench.tsv) | `llama-bench -p 0 -n 128` on the no-draft line, without and with the hop cast |
| [run/s35_arms.sh](run/s35_arms.sh) | the run: fit, bench, prep, the arm, the report |

## Commands

From the repository root, with S12's machine file filled in (`docs/data/2026-11-03_flash_next_mtp/run/s12.env.example`,
with `DRAFT` the MTP draft) and this run's own ports and client folder after it, so it can run beside another rig's:

```bash
ENVFILE=<machine file> RIG_PORT=8192 RELAY_PORT=8199 CLIENT_PORT=18493 PROXY_PORT=18494 \
  CLIENT_DIR=<client folder> CLIENT_OUT=<client results folder> \
  MODEL=<pmix-q2-fast first shard on the rig> BUILD=llama.cpp-20261104-39c3ab5f1-pascal-p106-v3-b12076-kshape OUT=<local folder> \
  bash docs/data/2026-11-04_s35_stack/run/s35_arms.sh
```

The server line: `-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <draft> --spec-type draft-mtp
--spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`, with S12's `-c 65536 -np 1` and the
release line's other settings, in the environment `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768
GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2`.

## Provenance

Run on 2026-11-04, 19:16–20:13 (the arm 19:27–20:13). The tasks are the agent check's bugfix set (the task IDs in
[report.md](report.md); `bench/agent_check` rebuilds them). The report holds no prompts or transcripts.
