# C1173: Flash-Next's better-quality file on the rewritten grid kernels, on the agent check (2026-11-26)

Registry row [Q1906](../../../quants/experiments.md), from C1146 ([Q1871](../2026-11-26_c1146_iq_kernels_rig1/README.md)).
`gmix-q2` is a mix of Qwen3.8-Flash-Next built from `pmix-q2-fast` (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF) with some
expert layers in 2-bit grid formats (IQ2_XXS, IQ2_S, IQ2_XS, IQ1_M) and some dense layers in 3-bit ones. It is closer
to the full model (KLD on agent text 0.236062 against 0.279228, [FN75](../2026-11-04_fmt_mixes/README.md)), but its
grid formats were slow on Pascal: on v4's engine it took 2.61 minutes a task on the agent check against 2.44 for the
maker's Swift Q2_0 ([R72c](../2026-11-12_r72_agent_v4c2/README.md)). C1146 timed the rewritten multi-token kernels of
those four formats (`GGML_CUDA_MTOK_IQ_FAST=2`, hand-written PTX): `gmix-q2`'s decode step became 11.66 and 11.78 ms
shorter, and its gap to `pmix-q2-fast` on the same build fell from +16.34 and +16.90 ms to +4.68 and +5.12 ms a step,
while its long prompts still read 34.3 % slower. Release settings are chosen on the coding-agent check, so this folder
runs that check with both files on the same build.

**`gmix-q2` is a loss on the agent check: a task takes 8.7 % longer at equal work, and two-thirds of the gap is
prompt reading.** Four runs, alternated, on one rig, engine and boot:
- **Time:** a task took 2.50 and 2.49 minutes at equal work with `gmix-q2`, against 2.30 and 2.29 with `pmix-q2-fast`
  (+0.20 minutes, +8.7 %; each file's two runs 0.01 apart). Without the one task that ran to its cap, 1.65 and 1.64
  against 1.43 and 1.41 (+15.8 %). Every time measure is a loss beyond the spread of each file's two runs
  ([class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv)).
- **Where the time goes** ([shares.tsv](shares.tsv), [class_shares.tsv](class_shares.tsv)): a task at equal work is R90's
  time outside the server (1.113 minutes in every run), R90's prompt tokens at the run's prompt speed, and R90's
  generated tokens at the run's generation speed. The prompt part took 0.477 and 0.457 minutes against 0.330 and 0.331
  (+41.3 %), the decode part 0.914 and 0.921 against 0.861 and 0.843 (+7.7 %). Of the 0.20-minute gap, 0.137 (68 %)
  is prompts and 0.066 (32 %) decoding. The prompt share of a task is 19.1 % and 18.4 % with `gmix-q2` against 14.3 %
  and 14.5 %, the decode share 36.5 % and 37.0 % against 37.4 % and 36.9 %; without the capped task, where R90's
  outside time no longer dominates, prompts 31.5 % and 30.3 % against 25.1 % and 25.2 %, decoding 59.9 % and 61.0 %
  against 65.0 % and 64.8 %.
- **Prompts:** 100.8 and 105.3 tok/s against 145.7 and 145.3 (−29.2 %). The 12 first turns, which read the same 10,746
  prompt tokens in every run, took 110.88 and 111.49 s against 80.02 and 78.50 s (+40.3 %;
  [same_prompts.md](same_prompts.md)). By the turn's new tokens ([prompt_buckets.tsv](prompt_buckets.tsv)), ms a prompt
  token is 37.1 % higher under 600 tokens, 52.2 % at 600 to 1,023, 47.8 % at 1,024 to 1,999 and 30.0 % at 2,000 and
  more: the calls the taper reads in micro-batches of 64 (up to 1,400 tokens) lose the most.
- **Decoding:** 27.95 and 27.74 tok/s against 29.67 and 30.29 (−7.1 %); a decode step over all requests 116.29 and
  116.68 ms against 110.47 and 109.90 (+6.30 ms, +5.7 %; [steps.md](steps.md)); the draft's acceptance 0.747 and 0.744
  against 0.757 and 0.774.
- **Tasks passed:** 12/12 and 12/12 with `gmix-q2`, 12/12 and 11/12 with `pmix-q2-fast`. The one failure,
  JavaScript `variable-length-quantity` in `pmixb`, was a single reply of about 21,550 tokens that the 12-minute cap
  cut (30 of 31 tests passing); R90's reference run ended on that task the same way. One failure in 48 tasks cannot
  show `gmix-q2`'s better KLD.

**What it means for the file.** With the rewritten kernels the decode gap is about 6 ms a step; what remains is mostly
prompt reading, where `gmix-q2` is 29 % slower and the kernels of C1146 do not reach (they serve 2 to 8 tokens).
If `gmix-q2` read prompts at `pmix-q2-fast`'s speed, a task would take est. 2.36 minutes at equal work against 2.30
(est. +2.9 %; without the capped task est. 1.50 against 1.43, est. +5.0 %), the decode part alone. By the project's
rule a loss is parked with its numbers: `pmix-q2-fast` stays the Pascal file and `GGML_CUDA_MTOK_IQ_FAST` stays out of
the preset. Whether `gmix-q2`'s quality (agent KLD 0.236062 against 0.279228) is worth the time is the owner's call;
the prompt question is registry row Q1872 (why its long prompts read a third slower) and the new row Q1912 (the
micro-batch of 64 under the taper, where it loses the most).

## What runs

Four runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig, one engine and one boot,
alternated so that each file has one run in each half:

| # | Run | File | `GGML_CUDA_MTOK_IQ_FAST` | Launched | Server up after | Tasks | Driver |
|---|---|---|---|---|---|---|---|
| 1 | `gmixa` | `gmix-q2` | `2` | 11:24:50 | 345 s | 11:30:44–11:48:32 | exit 0 at 11:48:52 |
| 2 | `pmixa` | `pmix-q2-fast` | unset | 11:49:19 | 307 s | 11:54:36–12:11:32 | exit 0 at 12:11:43 |
| 3 | `gmixb` | `gmix-q2` | `2` | 12:11:57 | 343 s | 12:17:50–12:39:36 | exit 0 at 12:39:43 |
| 4 | `pmixb` | `pmix-q2-fast` | unset | 12:39:58 | 306 s | 12:45:16–13:11:43 | exit 0 at 13:11:53 |

Every run: Flash-Next with the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090
cards, Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env): the
sparse ratio 0.75, `--ubatch-taper short=1400:64`), the engine `pascal-p106-v5-iq-mtok-fast` `883f21f2c` (the PTX
kernels of the four grid formats on `pascal-p106-v5` `2d839da1d`, not yet landed there; `llama-server` md5
`dae09aa55e6fe7e875a0d57bbaf0b796`), the build C1146 timed. The two files:

- `gmix-q2`: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-0000{1,2,3}-of-00003.gguf` (shards 1 and 2 md5
  `35f5b9f67c343f55969f91cfd5835c98` and `e36684c6ee4724b06a24322cf54efca8`, the n-gram table in the third shard),
  with `GGML_CUDA_MTOK_IQ_FAST=2`;
- `pmix-q2-fast`: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-0000{1,2,3}-of-00003.gguf`, the same table.

The rig is the first one at stock clocks (no offsets on this boot, power limit 75 W on every card), the same rig and
boot as C1146. The laptop that drives the rig is the agent's client.

**How the preset is started.** [run/gmix_agent.sh](run/gmix_agent.sh) starts it as R229's preset arm and C912 did
([2026-11-25_r229_flash_shortlist_agent](../2026-11-25_r229_flash_shortlist_agent/README.md)): R227's
[run/flash_taper.sh](../2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh) arm `t1400a`, that is C854's launcher
([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh)), R156's driver
([r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh)), the harness's arm `v4r`, then the report. For `gmix-q2` the
script adds `GGML_CUDA_MTOK_IQ_FAST=2` at the end of the driver's `V4R_ENV`, so the started command is the preset's with
that one switch after its last; the file reaches the rig as `MODEL` (the server's `-m`, which the started command does
not print). Checked for every run:

- The `pmix-q2-fast` runs started a 1,718-byte command, md5 `0c6ee59b8c18b92e076e0ab4ce95ee49`: R229's preset
  command (`s2a`, C912's `f5cand`) byte for byte, the build's folder name aside.
- The `gmix-q2` runs started a 1,743-byte command, md5 `3a5034aa214925338b094fc43355598f`: the same with
  ` GGML_CUDA_MTOK_IQ_FAST=2` after the preset's last switch.
- Each started command equals its dry line ([run/dry_line.sh](run/dry_line.sh)) and passes
  [run/preset_check.sh](run/preset_check.sh) against `settings.env` (40 switches, plus `GGML_CUDA_MTOK_IQ_FAST=2` for
  `gmix-q2`; `PRESET_ARGS` plus the harness's `--metrics --no-webui` and the draft). Each server's `switches:` line
  read 41 set for `gmix-q2` and 40 for `pmix-q2-fast`, and its `load_model` line named the file of its run
  ([models.tsv](models.tsv)). No server logged a PTX load warning.
- Before and after every run: no llama process, 0 MiB on every card, stock clocks (power limit 75 W), 0 Xid; the rig's
  log folder of each run removed by the driver.

## The class

By the release check's rule ([bench/agent_check/class_arms.py](../../../bench/agent_check/class_arms.py): a gain or a
loss only where the difference of the means is beyond the larger of the two files' spreads), `gmix-q2` against
`pmix-q2-fast`, each run by value ([class.tsv](class.tsv)):

| Measure | `gmixa` | `gmixb` | `pmixa` | `pmixb` | Difference | Class |
|---|---:|---:|---:|---:|---:|---|
| passed | 12/12 | 12/12 | 12/12 | 11/12 | | |
| minutes a task at equal work | 2.50 | 2.49 | 2.30 | 2.29 | +0.20 (+8.7 %) | loss |
| the same without the capped task (11 tasks) | 1.65 | 1.64 | 1.43 | 1.41 | +0.225 (+15.8 %) | loss |
| prompt part of a task at equal work, min | 0.477 | 0.457 | 0.330 | 0.331 | +0.137 (+41.3 %) | loss |
| decode part of a task at equal work, min | 0.914 | 0.921 | 0.861 | 0.843 | +0.066 (+7.7 %) | loss |
| prompt share of a task at equal work | 19.1 % | 18.4 % | 14.3 % | 14.5 % | | |
| decode share of a task at equal work | 36.5 % | 37.0 % | 37.4 % | 36.9 % | | |
| prompt tok/s | 100.8 | 105.3 | 145.7 | 145.3 | −42.45 (−29.2 %) | loss |
| generation tok/s | 27.95 | 27.74 | 29.67 | 30.29 | −2.135 (−7.1 %) | loss |
| ms a decode step, all requests | 116.29 | 116.68 | 110.47 | 109.90 | +6.30 (+5.7 %) | loss |
| first token median s | 3.65 | 3.16 | 2.88 | 2.48 | +0.725 (+27.1 %) | loss |
| first token p90 s | 10.89 | 11.86 | 8.14 | 8.18 | +3.215 (+39.4 %) | loss |
| first token on a task's first turn, median s | 8.06 | 7.96 | 5.04 | 5.06 | +2.96 (+58.6 %) | loss |
| the draft's acceptance | 0.747 | 0.744 | 0.757 | 0.774 | | |
| turns | 67 | 73 | 65 | 60 | | |
| minutes, all 12 tasks (wall) | 17.6 | 21.6 | 16.8 | 26.3 | | |
| prompt tokens read a task | 2,842 | 3,256 | 3,058 | 2,754 | | |
| tokens generated a task | 1,601 | 2,064 | 1,790 | 1,538 | | |

The minutes for all 12 tasks follow what each run generated (`pmixb` holds the capped task's 12 minutes); a task at
equal work prices R90's work at each run's speeds, so it compares the files on the same work. The capped reply's
request was cut before the server sent its timings, so its 706 s count as time outside the server and its tokens are in
no decode figure of `pmixb` ([capped.tsv](capped.tsv); [agent_nocap.tsv](agent_nocap.tsv) and
[shares_nocap.tsv](shares_nocap.tsv) leave that task out of every run).

**Against C1146's step.** On the repeated request C1146 measured `gmix-q2`'s step 4.68 ms (depth 0) and 5.12 ms (depth
8,192) longer than `pmix-q2-fast`'s on this build. On the agent's requests the step is 7.01 ms longer at contexts under
4,000 tokens (113.73 and 113.16 against 106.69 and 106.18) and 6.30 ms at 4,000 to 8,000 (117.13 and 116.81 against
110.79 and 110.56), over replies that differ between the runs.

**For reference only** (another engine, preset, rig and control file): on v4's candidate line on 2026-11-12, R72c's
`gmix-q2` took 2.61 minutes a task at equal work against 2.44 for the maker's Swift Q2_0 (+7.0 %), prompts 105.6
against 136.0 tok/s ([2026-11-12_r72_agent_v4c2](../2026-11-12_r72_agent_v4c2/README.md)).

## No regressions

- No server error, no tool call malformed (0 in every run), no graph reallocation ([reallocs.tsv](reallocs.tsv)).
  Every token sampled under the tool-call grammar in all four runs ([grammar.tsv](grammar.tsv)). One reply was read
  again, in `pmixa` (Python `variable-length-quantity` turn 5, 1,380 tokens, 8.40 s of prompt;
  [rereads_turns.tsv](rereads_turns.tsv)); none in the other runs. Text reuse kept the cached tokens on 16, 11, 14 and
  4 requests ([text_reuse_points.tsv](text_reuse_points.tsv)).
- Host memory ([host.tsv](host.tsv)): over the tasks the lowest MemAvailable was 221, 265, 246 and 292 MiB, and swap
  used at most 20 MiB in every run. Each report's memory check reads NO-GO on its lowest MemAvailable, as in R229's,
  C912's and R227's runs; the RAM watchdog (60 MiB) stopped nothing. `pmixb`'s report also reads NO-GO on its proxy
  rows: the capped task's cut request left two rows for one reply. Card 8, the fullest, peaked at 5,690, 5,690, 5,696
  and 5,694 MiB.
- Busy SM clocks ([clocks.tsv](clocks.tsv)): the mean of the nine cards' medians 1,885.3, 1,882.7, 1,884.0 and
  1,882.7 MHz (stock clocks).
- The slot's time before each request ([slot_gap.tsv](slot_gap.tsv)): median 0.70, 0.71, 0.72 and 0.70 ms.

## Files

| File | What it holds |
|---|---|
| [class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv) | the class of every time measure (`class_arms.py`), `gmix-q2` against `pmix-q2-fast`; all tasks, and without the capped one |
| [shares.tsv](shares.tsv), [shares_nocap.tsv](shares_nocap.tsv) | a task at equal work in its three parts (R90's time outside the server, the prompt part, the decode part) with the prompt and decode shares, and each run's own session split the same way |
| [class_shares.tsv](class_shares.tsv), [class_shares_nocap.tsv](class_shares_nocap.tsv) | the class of those parts and of the decode step, by the same rule |
| [agent.tsv](agent.tsv), [agent_nocap.tsv](agent_nocap.tsv) | the agent check's figures of each run, with R90's v3 run that prices "a task at equal work" |
| [models.tsv](models.tsv) | the model file each server loaded |
| [steps.md](steps.md) | ms a decode step, acceptance and tokens a step by context |
| [same_prompts.md](same_prompts.md) | the requests every run read on the same prompt and cached prefix |
| [work.tsv](work.tsv), [work.md](work.md) | the work of each run: prompt tokens read, tokens generated, reply re-reads, the longest first-token wait |
| [prompt_tasks.tsv](prompt_tasks.tsv), [prompt_buckets.tsv](prompt_buckets.tsv) | prompt seconds a task, and by the turn's new tokens in the taper's buckets |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run and the later turns' median and p90 |
| [capped.tsv](capped.tsv) | the tasks that ran to a cap (one, in `pmixb`) |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv), [text_reuse_points.tsv](text_reuse_points.tsv) | reply re-reads and the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request |
| [clocks.tsv](clocks.tsv), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv), [grammar.tsv](grammar.tsv), [taper_lines.tsv](taper_lines.tsv) | busy SM clocks, host memory and card peaks, graph reallocations, tokens sampled under the grammar, the taper each server read |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report of each run |
| [run/](run/) | the scripts below |

## Commands

From the repository root on the machine that drives the rig and is the agent's client, one run at a time, in the order
`gmixa pmixa gmixb pmixb`, each with its own `OUT`, rig folder, client folder and container prefix; only `ARM` and those
names change between the runs:

```bash
export ARM=gmixa PREFIX=acr-c1173-gmixa RIG=<rig alias> CLIENT=local \
  MODEL_G=<models>/flash_next/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf \
  MODEL_P=<models>/flash_next/swift_q2_0_pmix_fast/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261126-883f21f2c-pascal-p106-v5-iq-mtok-fast REF=<any build folder, unused by this arm> \
  RIGDIR=<a folder on the rig>/c1173-gmixa OUT=<run folder>/gmixa/out RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 \
  CLIENT_DIR=<client>/bench CLIENT_OUT=<client>/c1173-gmixa
unset GAINS_ENV V4R_ENV V4C2_TAPER MODEL
bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/gmix_agent.sh
```

Before the runs, the command each run will start and the file it will load, and the check against the preset
(`GGML_CUDA_MTOK_IQ_FAST=2` as the extra switch for the `gmix-q2` runs, nothing for the others):

```bash
ARM=gmixa WORK=<an empty scratch folder> NEW=llama.cpp-20261126-883f21f2c-pascal-p106-v5-iq-mtok-fast \
  MODEL_G=<as above> MODEL_P=<as above> DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/dry_line.sh > dry_gmixa.txt
head -n 1 dry_gmixa.txt > line_gmixa.txt
bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/preset_check.sh line_gmixa.txt GGML_CUDA_MTOK_IQ_FAST=2
```

The tables and the class, with Python 3's standard library only, from the runs' collected logs (the inputs are listed
at the top of [run/tables.sh](run/tables.sh)); each client's `res/` files go into `agent/<run>/` first:

```bash
for r in gmixa pmixa gmixb pmixb; do
  python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client>/c1173-$r/res \
    docs/data/2026-11-26_c1173_gmix_agent_rig1/agent/$r v4r $r
done
bash docs/data/2026-11-26_c1173_gmix_agent_rig1/run/tables.sh <tables folder> gmixa pmixa gmixb pmixb
```

| Script | What it does |
|---|---|
| [run/gmix_agent.sh](run/gmix_agent.sh) | the driver: the v5 preset with the run's file, and `GGML_CUDA_MTOK_IQ_FAST=2` for `gmix-q2`, through C854's launcher and R156's driver |
| [run/dry_line.sh](run/dry_line.sh) | prints the server command a run will start and the file it will load, without loading anything (R229's, with this driver) |
| [run/preset_check.sh](run/preset_check.sh) | checks a started or dry command against `settings.env`, with switches after the preset's as an argument (C912's, with that argument) |
| [run/tables.sh](run/tables.sh) | every table of this folder and the class (R229's without its shortlist counts, plus `models.tsv`, `shares.tsv` and `class_shares.tsv`) |

## Provenance and licence

The files are mixes of UkisAI's Swift-1.5 GSQ-RCO Q2_0 quantization of Qwen3.8-Flash-Next (`pmix-q2-fast` and
`gmix-q2`, built by [the fmt-mixes folder](../2026-11-04_fmt_mixes/README.md)'s scripts), with the MTP draft as in
C912. The agent check's tasks are its own 12 bug-fix tasks ([bench/agent_check](../../../bench/agent_check/README.md));
the tables hold task IDs and measurements, not task text. The tables and scripts here are this project's, under the
repository's licence.
