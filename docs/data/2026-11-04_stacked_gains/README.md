# The day's gains stacked on the agent check (S34, 2026-11-04)

What this measured: the coding-agent check ([bench/agent_check](../../../bench/agent_check)) of Qwen3.8-Flash-Next on
one 4 GB rig of nine P106-090, with the day's changes together. Registry row Q899 in
[quants/experiments.md](../../../quants/experiments.md):
- the file **pmix-q2-fast**: Swift-1.5 GSQ-RCO Q2_0 with a Pascal type mix
  ([2026-11-04_pascal_mix](../2026-11-04_pascal_mix/README.md));
- Miner Llama's **`-ladder3`** build (`7bc1552f4`: the speed ladder on upstream b12076 with the Pascal kernels);
- the k-column kernels' shared memory at 32 KiB (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`,
  [2026-11-04_kcurve_p106](../2026-11-04_kcurve_p106/README.md)).

Three arms, the bugfix task set with one attempt each and a 40-minute budget, in the client folder of the earlier
arms. The host booted with `kho=off` (no CMA-marked memory), with the RAM watchdog's `MemAvailable` floor at 60 MiB and
its other limits as before.

| Arm | Line | Passed | Minutes, all tasks | Minutes per task (median) | First token per turn, s (median, p90) | Prompt tok/s | Generation tok/s | Acceptance |
|---|---|---|---|---|---|---|---|---|
| no draft (S12) | v3's base, Swift Q2_0 | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | — |
| `mtp2d` (S33) | the MTP line on the `-draftmem` build, Swift Q2_0 | 11 / 12 | 47.4 | 2.9 | 9.17, 21.05 | 56 | 12.2 | 0.723 |
| **`s34nd`** | no draft | 11 / 12 | 47.3 | 2.4 | 7.84, 17.83 | 63 | **14.3** | — |
| **`s34mtp`** | `mtp2d`'s MTP line | 11 / 12 | 43.0 | 2.6 | 8.06, 20.56 | 57 | **14.6** | 0.697 |
| **`s34mtps`** | the same + `LLAMA_DRAFT_SHORTLIST=32768,8192,2` | **12 / 12** | **32.4** | 2.3 | 8.35, 19.99 | 58 | **16.0** | 0.705 |

The no-draft line is S12's (`-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio`). The MTP line is
`-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <Gyro draft> --spec-type draft-mtp --spec-draft-n-max 3
-otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`.

- **On the nine tasks every arm scored, at the no-draft arm's tokens:** `s34nd` 1.57–1.69× faster than S12's no-draft arm,
  `s34mtp` 1.58–1.64×, `s34mtps` 1.65–1.67×, against `mtp2d`'s 1.25–1.39× ([common_tasks.csv](common_tasks.csv)).
- **On this engine and file the MTP draft no longer pays by itself.** Without a draft a token takes ~68 ms
  (`llama-bench -p 0 -n 128`: 14.65 ± 0.08 tok/s, [bench.tsv](bench.tsv)), so a step of three drafted tokens and a
  four-token verify costs about what its ~2.1 accepted tokens save. The shortlist makes the draft's head cheaper, and that
  is the 10 % of `s34mtps` over `s34mtp`.
- **Host memory** ([session_memory_*.csv](session_memory_s34nd.csv)): without a draft the usable memory's median was
  490 MiB and its lowest 369, with 35 MiB of swap. With the draft the medians were 867 / 764 MiB and the lows 168 / 152,
  swap rose to 953 / 855 MiB, at most 76 / 105 MiB was swapped out in 10 s, and memory pressure (full) stayed at or
  below 1.3 %. The watchdog never fired.
- **The fit** ([fit_probe.txt](fit_probe.txt), llama.cpp's own projection): the fullest card keeps 1,013 MiB on the
  no-draft line and 823 MiB on the MTP line.
- The task that did not pass on `s34nd` and `s34mtp`, `bugfix-javascript-variable-length-quantity`, ran out of time
  each time (the runaway column); on `s34mtps` it passed.

## Files

| File | What it is |
|---|---|
| [report.md](report.md) | the agent check's report: per task and per arm, against the earlier no-draft and `mtp2d` arms |
| [common_tasks.csv](common_tasks.csv) | the nine tasks every arm scored: per arm turns, generated tokens, decode, new prompt tokens, prefill, fixed, other and wall seconds, and the time at the no-draft arm's tokens |
| [session_memory_s34nd.csv](session_memory_s34nd.csv), [session_memory_s34mtp.csv](session_memory_s34mtp.csv), [session_memory_s34mtps.csv](session_memory_s34mtps.csv) | each arm's host memory over the session from the watchdog's samples: usable (median, lowest), swap (mean, highest), most swap-out in 10 s, highest memory pressure, kills |
| [draft_acceptance_s34mtp.csv](draft_acceptance_s34mtp.csv), [draft_acceptance_s34mtps.csv](draft_acceptance_s34mtps.csv) | per request: acceptance, accepted and drafted tokens, mean accepted length |
| [bench.tsv](bench.tsv) | `llama-bench -p 0 -n 128` on the no-draft line, three repetitions |
| [fit_probe.txt](fit_probe.txt) | llama.cpp's projected memory per card for the two lines |
| [run/s34_arms.sh](run/s34_arms.sh) | the run: fit, bench, the three arms, the report |

## Commands

From the repository root, with S12's machine file filled in
([s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example), with `DRAFT`):

```bash
ENVFILE=s12.env MODEL=<pmix-q2-fast shard 1 on the rig> BUILD=llama.cpp-20261104-7bc1552f4-pascal-p106-v3-b12076-ladder3 \
  OUT=/tmp/s34 bash docs/data/2026-11-04_stacked_gains/run/s34_arms.sh
# the tables, on the client in its results folder
python3 -I docs/data/2026-11-03_flash_next_dflash2/run/s18_tables.py --out t --attempts res/attempts.csv \
  --server-log rig/server-s34mtps.log --guard rig/ram_watchdog-s34mtps.tsv --ref nodraft --arms mtp2d,s34nd,s34mtp,s34mtps
```

## Provenance

The file is pascal-mix's (its README has its tensors' types and md5s); the Gyro MTP draft and the agent check's task set
as in [2026-11-03_flash_next_mtp](../2026-11-03_flash_next_mtp/README.md). The tables summarize the client's attempt
records and the watchdog's one-second samples; server logs are not published.
