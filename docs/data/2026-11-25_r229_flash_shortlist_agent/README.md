# R229: Flash-Next's draft with one full-head round after a shortlist miss, on the agent check (2026-11-25)

Registry row [Q1642](../../../quants/experiments.md), from R175 ([Q1507](../2026-11-20_r175_flash_shortlist/README.md)).
Qwen3.8-Flash-Next's MTP draft reads a shortlist of the vocabulary as its output head. When the main model's latest
token is outside that list, the next drafting rounds read the full head, and the token joins the list. The v5 preset
runs two such rounds after a miss (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`). On the deep lane R175 ran one (`…,1`): a step
1.31 ms shorter (−1.2 %) where the reply met tokens new to the list, the same everywhere else. Release settings are
chosen on the coding-agent check, so this folder runs that check with the preset as it is against the same preset with
`…,1`, nothing else different.

**One full-head round is negligible on the agent check, and it removes about a third of the draft's full-head rounds.**
Four runs, alternated, on one rig, engine and boot, all 12 of 12 tasks:
- **Time:** a task took 2.26 and 2.25 minutes at equal work with one round, against 2.23 and 2.25 with the preset's two.
  The model generated 30.05 and 30.82 tok/s against 31.07 and 30.43. Every time measure is inside the spread of the
  two runs of each arm, and none is slower beyond it ([class.tsv](class.tsv)).
- **The draft:** one round drafted 186 and 220 rounds with the full head, one a miss; two rounds drafted 289 and 302,
  1.99 and 1.96 a miss. That is 31.3 % fewer full-head rounds a run, a gain on the work by the same rule
  ([class_draft.tsv](class_draft.tsv)). The draft's acceptance did not change beyond the spread: 0.740 and 0.778
  against 0.772 and 0.755.
- **Why the minutes do not move:** at R175's est. 8 ms a full-head round, 92.5 fewer rounds a run is est. 0.7 s a run,
  est. 0.06 s a task, about a twentieth of the 0.02-minute (1.2 s) difference between the preset's own two runs.

By the project's rule (negligible and not measurably slower) `…,1` can join Flash-Next's v5 preset, as the sparse ratio
0.75 (R220) and the 1,400-token taper (R227) did with the same class; `release/flash-next-v5/settings.env` keeps `…,2`
until the owner's word. Until then it is listed with the negligible gains, which are tested as a group (C534).

## What runs

Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and
JavaScript alternating, a 12-minute cap a task, a 40-minute budget): the maker's Swift GSQ-RCO Q2_0
(`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram table in the third shard) with the MTP draft
(`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090 cards, Flash-Next's v5 preset as it is
([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env): the sparse ratio 0.75,
`--ubatch-taper short=1400:64`). The engine is `pascal-p106-v5-cand-b12674` `eee615cdb` (upstream b12674 merged,
`llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`), the one C912 checked
([2026-11-24_c912_flash_agent](../2026-11-24_c912_flash_agent/README.md)). The rig is the second one, overclocked with
its soaked clock profile (R237: memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core
+256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8, 75 W, the clock kit's watchdog running), the same
profile for all four runs, on one boot. The machine that drives the rig is the agent's client.

| # | Run | Setting | Launched | Server up after | Tasks | Driver |
|---|---|---|---|---|---|---|
| 1 | `s2a` | `LLAMA_DRAFT_SHORTLIST=32768,8192,2` (the preset) | 17:55:58 | 226 s | 17:59:58–18:18:16 | exit 0 at 18:18:36 |
| 2 | `s1a` | `LLAMA_DRAFT_SHORTLIST=32768,8192,1` | 18:18:52 | 225 s | 18:22:49–18:42:56 | exit 0 at 18:43:14 |
| 3 | `s2b` | `…,2` (the preset) | 18:43:32 | 226 s | 18:47:32–19:04:23 | exit 0 at 19:04:39 |
| 4 | `s1b` | `…,1` | 19:04:57 | 226 s | 19:08:57–19:34:28 | exit 0 at 19:34:49 |

**How the preset is started.** [run/flash_shortlist.sh](run/flash_shortlist.sh) starts it as C912 did: R227's
[run/flash_taper.sh](../2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh) arm `t1400a`, that is C854's launcher
([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh)), R156's driver
([r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh)), the harness's arm `v4r`, then the report. R156's driver takes
the shortlist from `release/flash-next-v3/settings.env`'s `PRESET_ENV`, which no environment variable overrides, so
the script writes a copy of `r156.sh` with one line added right after the line that reads that file
(`PRESET_ENV=${PRESET_ENV/LLAMA_DRAFT_SHORTLIST=32768,8192,2/LLAMA_DRAFT_SHORTLIST=32768,8192,<rounds>}`) and a copy of
`flash_agent.sh` that runs it. Both arms take that path; for the preset's arm the added line changes nothing.
- The preset's two runs started a 1,717-byte command, md5 `e88afbaa6d856ae489378b407852abc3`: C912's candidate run
  (`f5cand`) byte for byte.
- The one-round runs started a 1,717-byte command, md5 `a0fb778ec535e4b91cffda11a88a0b11`, which differs from it in
  `LLAMA_DRAFT_SHORTLIST=32768,8192,1` alone.
- Each started command equals the one [run/dry_line.sh](run/dry_line.sh) printed before the runs, and
  [run/preset_check.sh](run/preset_check.sh) finds each equal to `settings.env` (the 40 switches byte for byte, with
  the shortlist's rounds 2 or 1; each flag with its value).
- Each server logged the rounds it read at its load: "full head for 2 drafts after a target token outside it" on `s2a`
  and `s2b`, "for 1 drafts" on `s1a` and `s1b`. All four read `short=1400:64` ([taper_lines.tsv](taper_lines.tsv)).

## The four runs

| | **`s2a`** (two rounds) | **`s1a`** (one round) | **`s2b`** (two rounds) | **`s1b`** (one round) |
|---|---:|---:|---:|---:|
| tasks passed; turns | 12/12; 62 | 12/12; 66 | 12/12; 68 | 12/12; 67 |
| minutes a task at equal work | 2.23 | 2.26 | 2.25 | 2.25 |
| minutes for the 12 tasks; median minutes a task | 18.1; 1.21 | 19.9; 1.14 | 16.7; 1.24 | 25.4; 1.33 |
| generation tok/s; tokens generated | 31.07; 25,570 | 30.05; 27,587 | 30.43; 22,062 | 30.82; 38,682 |
| prompt tok/s; prompt tokens read; prompt s | 162.8; 38,188; 234.55 | 162.9; 40,255; 247.15 | 161.9; 39,794; 245.77 | 156.3; 37,280; 238.46 |
| first token: median, p90, first turn's median (s) | 2.52, 7.92, 4.60 | 2.19, 7.88, 4.78 | 2.16, 8.20, 4.84 | 2.14, 8.10, 4.75 |
| the longest first-token wait (s) | 27.71 | 27.36 | 27.56 | 27.35 |
| the draft's acceptance; tokens a step | 0.772; 3.32 | 0.740; 3.22 | 0.755; 3.26 | 0.778; 3.33 |
| ms a decode step: all requests; at 4,000 to 8,000 tokens | 107.01; 106.99 | 107.45; 107.13 | 107.42; 107.57 | 108.41; 106.82 |
| target tokens outside the list (misses) | 145 | 186 | 154 | 220 |
| drafting rounds with the full head; a miss | 289; 1.99 | 186; 1.00 | 302; 1.96 | 220; 1.00 |
| drafting rounds with the shortlist | 7,406 | 8,362 | 6,444 | 11,366 |
| misses a 1,000 generated tokens | 5.67 | 6.74 | 6.98 | 5.69 |
| the 12 first turns (the same prompt and cached prefix in every run), s | 75.76 | 74.92 | 75.32 | 76.15 |

- "A task at equal work" prices R90's work (R90's v3 run, as in C854's, R208's, R220's, R227's and C912's tables) at
  each run's own speeds, so runs that did different work compare on the same work ([agent.tsv](agent.tsv)). The
  minutes for the 12 tasks follow what the agent did: `s1b`'s JavaScript `variable-length-quantity` took 673 s in 9
  turns (passed), against 173, 189 and 111 s in `s2a`, `s1a` and `s2b`.
- The session's generation follows the draft's acceptance, which follows the text the model sampled: `s1a` and `s1b`
  are the arm's lowest and highest acceptance and generation of the four.
- The decode step over all requests follows the contexts each run reached: only `s1b` reached 16,000 tokens (7
  requests at 114.34 ms a step, [steps.md](steps.md)). On the contexts all four share most, 4,000 to 8,000 tokens,
  the steps were 107.13 and 106.82 ms with one round against 106.99 and 107.57 with two.
- The misses are the text's, not the setting's: 6.74 and 5.69 a 1,000 generated tokens with one round against 5.67 and
  6.98 with two. The token joins the list at once either way, so one round changes only how long the draft stays on the
  full head after a miss.
- The prompts are not the setting's work (the shortlist acts only on drafting): the 12 first turns read the same 10,746
  tokens on the same cached prefix in every run, in 75.76, 74.92, 75.32 and 76.15 s (`s2a`, `s1a`, `s2b`, `s1b`), in
  no order of the arms ([same_prompts.md](same_prompts.md)).

## The class

[class.tsv](class.tsv) is [bench/agent_check/class_arms.py](../../../bench/agent_check/class_arms.py) on
[agent.tsv](agent.tsv), the one-round arm (`s1a`, `s1b`) against the preset (`s2a`, `s2b`): a gain or a loss only where
the difference of the means is beyond the larger of the two arms' spreads (each arm's highest run less its lowest).
[class_draft.tsv](class_draft.tsv) applies the same rule to the draft's own measures (made by
[run/tables.sh](run/tables.sh)). No task ran to a cap in any run ([capped.tsv](capped.tsv)), so there is no table
without capped tasks.

| Measure | One round | Two rounds (preset) | Difference of the means | Spreads (one; two) | Class |
|---|---|---|---:|---|---|
| minutes a task at equal work | 2.26, 2.25 | 2.23, 2.25 | +0.015 (+0.7 %) | 0.01; 0.02 | negligible |
| generation tok/s | 30.05, 30.82 | 31.07, 30.43 | −0.315 (−1.0 %) | 0.77; 0.64 | negligible |
| prompt tok/s | 162.9, 156.3 | 162.8, 161.9 | −2.75 (−1.7 %) | 6.6; 0.9 | negligible |
| first token median s | 2.19, 2.14 | 2.52, 2.16 | −0.175 (−7.5 %) | 0.05; 0.36 | negligible |
| first token p90 s | 7.88, 8.10 | 7.92, 8.20 | −0.07 (−0.9 %) | 0.22; 0.28 | negligible |
| first token on a task's first turn, median s | 4.78, 4.75 | 4.60, 4.84 | +0.045 (+1.0 %) | 0.03; 0.24 | negligible |
| the draft's acceptance | 0.740, 0.778 | 0.772, 0.755 | −0.0045 (−0.6 %) | 0.038; 0.017 | negligible |
| tokens a step | 3.22, 3.33 | 3.32, 3.26 | −0.015 (−0.5 %) | 0.11; 0.06 | negligible |
| ms a decode step, all requests | 107.45, 108.41 | 107.01, 107.42 | +0.715 (+0.7 %) | 0.96; 0.41 | negligible |
| full-head drafting rounds a run | 186, 220 | 289, 302 | −92.5 (−31.3 %) | 34; 13 | gain (work) |
| full-head drafting rounds a miss | 1.00, 1.00 | 1.99, 1.96 | −0.98 (−49.4 %) | 0; 0.03 | gain (work) |
| misses a 1,000 generated tokens | 6.74, 5.69 | 5.67, 6.98 | −0.11 (−1.7 %) | 1.06; 1.31 | negligible |

## No regressions

- No server error, no tool call malformed (0 in every run), no reply read again ([rereads_runs.tsv](rereads_runs.tsv)),
  no graph reallocation ([reallocs.tsv](reallocs.tsv)). Every token sampled under the tool-call grammar in all four
  runs (the candidate engine's count, as in C912; [grammar.tsv](grammar.tsv)).
- Host memory ([host.tsv](host.tsv)): over the tasks the lowest MemAvailable was 320, 313, 319 and 201 MiB, and swap
  grew from 55, 55, 54 and 54 MiB to at most 563, 827, 543 and 54 MiB. `s1b` swapped nothing out; its MemAvailable
  fell to 201 MiB at its last task's end and to 155 MiB while the server stopped (`host.tsv` takes the whole log), far
  above the RAM watchdog's 60 MiB, which stopped nothing. The other one-round run, `s1a`, swapped the most, so the
  setting is not the cause. Each report's memory check reads NO-GO on its lowest MemAvailable, as in every run of C912
  and R227. Card 8, the fullest, peaked at 5,694, 5,700, 5,694 and 5,698 MiB.
- Busy SM clocks ([clocks.tsv](clocks.tsv)): the mean of the nine cards' medians 2,117.1, 2,115.8, 2,114.3 and
  2,117.1 MHz.
- The clock kit's watchdog ran through all four runs without a trip, and every card kept its offsets; the boot logged
  no GPU error (0 Xid) and no step-down.

## Files

| File | What it holds |
|---|---|
| [class.tsv](class.tsv) | the class of every time measure (`class_arms.py`), one round against the preset |
| [class_draft.tsv](class_draft.tsv) | the class of the draft's own measures, by the same rule |
| [shortlist.tsv](shortlist.tsv) | each server's shortlist counts: the full-head rounds after a miss it read at its load, its drafting rounds with the shortlist and with the full head, the misses and the ids in the list at its end |
| [agent.tsv](agent.tsv) | the agent check's figures of each run, with R90's v3 run that prices "a task at equal work" |
| [steps.md](steps.md) | ms a decode step, acceptance and tokens a step by context |
| [same_prompts.md](same_prompts.md) | the requests every run read on the same prompt and cached prefix |
| [work.tsv](work.tsv), [work.md](work.md) | the work of each run: prompt tokens read, tokens generated, reply re-reads, the longest first-token wait |
| [prompt_tasks.tsv](prompt_tasks.tsv), [prompt_buckets.tsv](prompt_buckets.tsv) | prompt seconds a task, and by the turn's new tokens in the taper's buckets |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run and the later turns' median and p90 |
| [capped.tsv](capped.tsv) | the tasks that ran to a cap (none) |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv), [text_reuse_points.tsv](text_reuse_points.tsv) | reply re-reads (none) and the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request |
| [clocks.tsv](clocks.tsv), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv), [grammar.tsv](grammar.tsv), [taper_lines.tsv](taper_lines.tsv) | busy SM clocks, host memory and card peaks, graph reallocations, tokens sampled under the grammar, the taper each server read |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report of each run |
| [run/](run/) | the scripts below |

## Commands

From the repository root on the machine that drives the rig and is the agent's client, one run at a time, in the order
`s2a s1a s2b s1b`, each with its own `OUT`, rig folder, client folder and container prefix; only `ARM` and those names
change between the runs:

```bash
export ARM=s2a PREFIX=acrel-r229-s2a RIG=<rig alias> CLIENT=local \
  MODEL=<models>/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 REF=<any build folder, unused by this arm> \
  RIGDIR=<a folder on the rig>/r229-s2a OUT=<run folder>/s2a/out RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client>/bench CLIENT_OUT=<client>/r229-s2a
unset GAINS_ENV V4R_ENV V4C2_TAPER
bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/flash_shortlist.sh
```

Before the runs, the command each arm will start, and its check against the preset (rounds 2 for `s2a` and `s2b`, 1
for `s1a` and `s1b`):

```bash
ARM=s1a WORK=<an empty scratch folder> NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/dry_line.sh > dry_line_s1a.txt
bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/preset_check.sh dry_line_s1a.txt 1
```

The tables and the class, with Python 3's standard library only, from the runs' collected logs (the inputs are listed
at the top of [run/tables.sh](run/tables.sh)); each client's `res/` files go into `agent/<run>/` first:

```bash
for r in s2a s1a s2b s1b; do
  python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client>/r229-$r/res \
    docs/data/2026-11-25_r229_flash_shortlist_agent/agent/$r v4r $r
done
bash docs/data/2026-11-25_r229_flash_shortlist_agent/run/tables.sh <tables folder> s2a s1a s2b s1b
```

| Script | What it does |
|---|---|
| [run/flash_shortlist.sh](run/flash_shortlist.sh) | the driver: the v5 preset with the shortlist's rounds of the arm (2 or 1), through C854's launcher and R156's driver |
| [run/dry_line.sh](run/dry_line.sh) | prints the server command an arm will start, without loading anything (R227's, with this driver) |
| [run/preset_check.sh](run/preset_check.sh) | checks a started or dry command against `settings.env`, with the shortlist's rounds as an argument (C912's, with that argument) |
| [run/tables.sh](run/tables.sh) | every table of this folder and the class (C912's, without its context rows, plus `shortlist.tsv`, `class.tsv` by `class_arms.py`, and `class_draft.tsv`) |

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 quantization of Qwen3.8-Flash-Next with its MTP draft, as in C912. The
agent check's tasks are its own 12 bug-fix tasks ([bench/agent_check](../../../bench/agent_check/README.md)); the
tables hold task IDs and measurements, not task text. The tables and scripts here are this project's, under the
repository's licence.
