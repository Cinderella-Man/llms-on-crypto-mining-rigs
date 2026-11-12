# `gmix-q2` against `pmix-q2-fast` on a P106: the fit and the step pair (R115, Q1382, 2026-11-12)

What this is: the speed of the v3.1 file pick `gmix-q2` on nine P106-090 cards
([the pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)). It is `pmix-q2-fast` with five 8-layer-group
switches, at the same bytes on the cards, and it scores better on all three texts: agent KLD 0.236062 against
0.279228, wiki 0.414086 against 0.426376, code 0.095812 against 0.108328
([its KLD](../2026-11-04_fmt_mixes/README.md#gmix-q2-measured-on-the-three-texts-q1381-2026-11-11)). Its speed cost was
est. +5.91 ms a token, priced on v3.0's one-token kernels. This run measures it on v4's line, in two parts:

1. **The fit:** llama-server's own projection per card for both files, from header-only copies, at 65,536 tokens,
   beside the plan ([pick_fit_64k_cards.txt](../2026-11-04_fmt_mixes/pick_fit_64k_cards.txt)).
2. **The step pair:** [bench/deep_lane](../../../bench/deep_lane/README.md) at 0 and 8,192 tokens of context, both
   files on one server line, loads `warm pmix1 gmix1 pmix2 gmix2`: ms a step, tokens a step, prompt tok/s and memory per
   card, every load by value.

The agent check with each file is a later run (Q1261's stacked check). Registry [Q1382](../../../quants/experiments.md).

## Result

**On v4's line `gmix-q2` is a loss in decode and in prompts.**
- **Decode:** on the repeated request, over both depths, a step takes 125.83 and 125.86 ms against 107.80 and 107.77
  (+18.06 ms, +16.75 %).
- **Prompts:** an 8,128-token prompt reads at 122.5 and 127.5 tok/s against 152.2 and 150.6 (−17.43 %).
- **Against the estimate:** the step grows by about three times the est. +5.91 ms a token. The estimate priced one-token
  decoding on v3.0's kernels; a step here is a four-token verify pass plus the draft's passes.
- **The fit:** as planned. The fullest card is card 3, with 24 MiB more in use than with `pmix-q2-fast`.

### The step, ms

ms a step = the request's generation time over its steps, a step being generated tokens minus accepted draft tokens.
One request per cell. Both loads of a file wrote the same reply at every depth and pass. The two files wrote different
replies (different weights), so their steps yield different tokens; the step's time is the comparison that does not
depend on the reply: a step is always a four-token verify pass (three draft tokens).

| Depth | Request | `pmix1` | `pmix2` | `gmix1` | `gmix2` | Spread | `gmix` − `pmix` | % | Class |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | pass 1 | 102.34 | 106.02 | 125.98 | 125.87 | 3.67 | +21.74 | +20.87 | loss |
| 0 | pass 2 (repeated) | 99.37 | 99.60 | 118.71 | 118.63 | 0.23 | +19.19 | +19.28 | loss |
| 0 | sampler pass | 123.71 | 127.75 | 142.25 | 139.56 | 4.04 | +15.18 | +12.07 | loss |
| 8,192 | pass 1 | 150.56 | 146.98 | 170.84 | 170.81 | 3.58 | +22.05 | +14.82 | loss |
| 8,192 | pass 2 (repeated) | 117.26 | 116.94 | 134.59 | 134.75 | 0.31 | +17.57 | +15.01 | loss |
| 8,192 | sampler pass | 123.50 | 124.07 | 142.93 | 142.92 | 0.57 | +19.14 | +15.47 | loss |
| both | pass 2 (repeated) | 107.80 | 107.77 | 125.83 | 125.86 | 0.03 | +18.06 | +16.75 | loss |

- **The class** is deep_table.py's: the spread is the larger of the two files' differences between their two loads,
  and a difference past it is a gain or a loss ([classes.tsv](classes.tsv) has every metric).
- **The sampler pass** is one request a depth with the server line's own sampler (temperature 1.0, a fixed seed), the
  one request where probabilistic verification acts.
- **Pass 1 at 8,192** is the first request after the prime; its step is longer in both files (R80 saw the same).

### Tokens a step and decode

| Depth | Request | `pmix` tokens a step (share accepted) | `gmix` tokens a step (share accepted) | `pmix` tok/s, two loads | `gmix` tok/s, two loads |
|---:|---|---|---|---|---|
| 0 | greedy, both passes | 2.768 (0.6000) | 2.226 (0.4147) | 27.44, 26.91 | 18.18, 18.19 |
| 0 | sampler pass | 2.560 (0.5270) | 2.876 (0.6398) | 20.69, 20.04 | 20.22, 20.61 |
| 8,192 | greedy, both passes | 3.084 (0.7033) | 2.798 (0.6127) | 23.00, 23.34 | 18.31, 18.30 |
| 8,192 | sampler pass | 2.876 (0.6350) | 2.813 (0.6157) | 23.29, 23.18 | 19.68, 19.68 |

Each value is one reply per file, the same on both loads, so the tokens a step differ with the reply, not with the
loads: `gmix` yields fewer at three of the four cells and more on the sampler pass at depth 0. Four replies do not
measure the draft's yield on a file. The agent check's many turns do. Greedy decode over both depths: 25.02 and 25.00
tok/s against 18.24 and 18.25 (−27.05 %), the longer step and the replies' yield together.

### Prompts

| Depth | Read | `pmix1` | `pmix2` | `gmix1` | `gmix2` | `gmix` − `pmix` |
|---:|---|---:|---:|---:|---:|---:|
| 8,192 | 8,128 tokens on an empty cache (the prime), tok/s | 152.2 | 150.6 | 122.5 | 127.5 | −26.4 tok/s, −17.43 %, loss |
| 8,192 | the same prime, s | 53.42 | 53.96 | 66.33 | 63.76 | +11.36 s |
| 0 | 64 tokens on an empty cache, tok/s | 54.4 | 53.0 | 51.3 | 51.2 | −2.5 tok/s, −4.58 %, loss |

The prime's seconds are from [requests.csv](requests.csv). No load reallocated the
graph: 0 reallocations of either kind in the 8 requests of each timed load, on this line without
`LLAMA_SAMPLING_TOP_K_ROWS=1` ([FN84](../2026-11-12_fn84_graph_realloc/README.md)).

### Memory per card, MiB

| Line | Source | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Fullest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no draft, `-ts 6,5,5,6,5,5,6,5,6` | plan, `pmix-q2-fast` | 4,943 | 4,243 | 4,427 | 5,001 | 4,271 | 4,432 | 5,009 | 4,276 | 4,868 | card 6 |
| | projected, `pmix-q2-fast` | 4,944 | 4,243 | 4,426 | 5,002 | 4,271 | 4,431 | 5,010 | 4,276 | 4,867 | card 6, 5,010 |
| | plan, `gmix-q2` | 4,918 | 4,029 | 4,396 | 5,033 | 4,471 | 4,445 | 5,009 | 4,276 | 4,868 | card 3 |
| | projected, `gmix-q2` | 4,919 | 4,029 | 4,396 | 5,033 | 4,471 | 4,444 | 5,010 | 4,276 | 4,867 | card 3, 5,033 |
| the preset's (MTP draft, `-ts 6,6,6,6,5,6,5,6,3`) | plan, `pmix-q2-fast` | 4,990 | 5,179 | 5,047 | 5,200 | 4,312 + 645 | 5,057 | 4,461 | 5,064 | 4,632 | card 3 |
| | projected, `pmix-q2-fast` | 4,992 | 5,178 | 5,047 | 5,200 | 4,957 | 5,058 | 4,461 | 5,064 | 4,632 | card 3, 5,200 |
| | plan, `gmix-q2` | 4,965 | 4,929 | 5,070 | 5,227 | 4,524 + 645 | 5,045 | 4,461 | 5,064 | 4,632 | card 3 |
| | projected, `gmix-q2` | 4,967 | 4,929 | 5,070 | 5,227 | 5,170 | 5,045 | 4,461 | 5,064 | 4,632 | card 3, 5,227 |
| R115's line (the preset + [line.env](run/line.env)) | projected, `pmix-q2-fast` | 4,992 | 5,178 | 5,047 | 5,200 | 4,961 | 5,058 | 4,461 | 5,064 | 4,769 | card 3, 5,200 |
| | projected, `gmix-q2` | 4,967 | 4,929 | 5,070 | 5,227 | 5,174 | 5,045 | 4,461 | 5,064 | 4,769 | card 3, 5,227 |
| | after loading, `pmix1` and `pmix2` | 5,053 | 5,101 | 4,971 | 5,121 | 4,887 | 4,981 | 4,383 | 4,989 | 4,781 | card 3, 5,121 |
| | after loading, `gmix1` and `gmix2` | 5,029 | 4,851 | 4,995 | 5,149 | 5,099 | 4,969 | 4,383 | 4,989 | 4,781 | card 3, 5,149 |
| | highest in the load, `pmix1` and `pmix2` | 5,147 | 5,189 | 5,063 | 5,209 | 4,971 | 5,075 | 4,457 | 5,083 | 4,873 | card 3, 5,209 |
| | highest in the load, `gmix1` and `gmix2` | 5,121 | 4,979 | 5,129 | 5,233 | 5,183 | 5,059 | 4,455 | 5,079 | 4,871 | card 3, 5,233 |

- **The projection is llama-server's `--fit` step** (`scripts/fit_probe.sh`): the main model's and the draft's model,
  context and compute MiB per card, summed, read from the files' metadata with no weights loaded. On card 4 it includes
  the draft's token embeddings (645 MiB, `-otd token_embd=CUDA4`); the plan lists them apart. The projection matches the
  plan to 2 MiB on every card of both files and both lines.
- **The fullest card keeps** 841 MiB of the 6,068 a P106-090 offers a process in the projection (796 against the
  6,023 free when the probe started, the plan's figure), 919 after loading and 835 at the highest, against
  `pmix-q2-fast`'s 868, 947 and 859. Card 4 holds 212 MiB more with `gmix-q2` and card 1 250 MiB less after loading (210 at the highest), as planned.
- **R115's line adds 137 MiB on card 8** to the preset's projection (the draft's micro-batch of 512) and 4 MiB on card
  4, the same for both files.
- **After loading and the highest in the load** are nvidia-smi's MiB in use (with the CUDA context), right after the
  server came up and the most in any 10-second sample of the load ([card_fit.tsv](card_fit.tsv),
  [card_max.tsv](card_max.tsv)). Both loads of a file read the same on every card.
- **Host memory:** the lowest MemAvailable over a load was 379 and 377 MiB with `pmix-q2-fast` and 328 and 330 with
  `gmix-q2`; swap in use at most 22 MiB; the RAM watchdog (floor 60 MiB) never acted ([host_memory.tsv](host_memory.tsv)).
- **Loading took** 306 and 306 s for `pmix-q2-fast` and 345 and 345 s for `gmix-q2` (the warm-up, `pmix-q2-fast`,
  307 s), from the same USB 3 disk.

### What it means for an agent task

On R90's agent check a task was 52 % generation and 33 % prompt reading
([R90](../2026-11-11_r90_v4_vs_v3/README.md); [the third review](../../research/notes/flash_next_review3_2026-11.md)), so 1 % on the step is est. 0.5 % of a task and 1 %
on prompt time est. 0.3 %. A step 16.75 % longer and prompts read at 0.8257 times the speed (21.1 % more time) make
est. +8.4 % and +6.3 %: **est. +14.7 % on an agent task**, before any change in the draft's yield. The KLD gain is bought
at that price on v4's kernels. Where the 18 ms go is not measured here: the switched tensors are routed gate/up in
IQ2_XS, IQ2_XXS, IQ2_S and IQ1_M (groups 0, 1 and 3) and the dense tensors of groups 1 and 2 from `pmix-iq3-fast`,
in place of the start's Q2_0 and dense types. A per-part profile of the four-token verify pass with each file, and the
same pair with `GGML_CUDA_MMVQ_PASCAL_MT=1` (the 2–4-token read-once kernels for IQ2_S and IQ3_XXS,
[docs/switches.md](../../switches.md)), would name the products ([Q1409](../../../quants/experiments.md)).

## How it ran

- **Build:** `pascal-p106-v4` `d097d6d88` (the v4 kernel merge's tarball), `llama-server` version 0.9.0-dev build
  12661. Nine P106-090 cards, the rig's 4 GB host, nothing else running (quiet checks before every load: no llama
  process, 0 MiB on every card).
- **Files:** `pmix-q2-fast` (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-0000{1,2,3}-of-00003.gguf`) and `gmix-q2`
  (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-0000{1,2,3}-of-00003.gguf`; shards 1–2 md5
  `35f5b9f67c343f55969f91cfd5835c98` and `e36684c6ee4724b06a24322cf54efca8`), shard 3 of both the same n-gram table;
  the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf`.
- **The line:** the Flash-Next v3 preset ([settings.env](../../../release/flash-next-v3/settings.env)) with
  [run/line.env](run/line.env): R90's `gains` line ([r90.sh](../2026-11-11_r90_v4_vs_v3/run/r90.sh), `GAINS_ENV` and
  `GAINS_ARGS`, the draft's micro-batch 512) with `--ubatch-taper short=600:64` in place of `on`
  ([R60](../2026-11-12_r60_taper_v4/README.md)) and without `LLAMA_SAMPLING_TOP_K_ROWS=1`
  ([FN84](../2026-11-12_fn84_graph_realloc/README.md)). [loads.tsv](loads.tsv) holds every load's line as it ran.
- **One file per arm:** the deep lane's `MODEL` is `pmix-q2-fast`; the `gmix` arm's arguments end in
  `-m <gmix-q2's first shard>`. llama-server takes the last `-m` given and warns that it was given twice; each server
  log's `load_model` line names the file it read (`gmix-q2` in `gmix1` and `gmix2`, `pmix-q2-fast` in the others). A
  dry run checked first that the two arms' lines differ only in that `-m`. `loads.tsv`'s `model` column is the deep
  lane's `MODEL` for every load; its `arm_args` column has the `-m`.
- **Text:** the release check's recorded agent requests (`TAG='^t1swift/'`), as in
  [R80](../2026-11-11_r80_c720_p106/README.md): prompts of exactly 64 and 8,192 tokens, 256 tokens out at temperature
  0, two passes per depth, then the sampler pass (`SAMPLER_PASS=1`).
- **Times:** fit probe 12:12–12:13; the lane 12:14–12:51 (a timed load 7 min 22 s to 8 min 32 s).

## Files

| File | What it holds |
|---|---|
| [step_pair.md](step_pair.md), [per_load.tsv](per_load.tsv), [classes.tsv](classes.tsv) | per depth and load: decode, the step on each pass, the draft's share and tokens a step, the sampler pass, prompts; each metric's class (`v4c_table.py`) |
| [table.md](table.md), [lane_classes.tsv](lane_classes.tsv) | the deep lane's own tables and classes (`deep_table.py`) |
| [card_fit.tsv](card_fit.tsv) | MiB per card: the six projections (three lines, two files), after loading and the highest in each load |
| [card_max.tsv](card_max.tsv) | the highest MiB per card in each load |
| [host_memory.tsv](host_memory.tsv) | the host's lowest MemAvailable and swap per load and per request |
| [reallocs.tsv](reallocs.tsv) | graph reallocations and prompt seconds per load by prompt size |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran; per request its counts, times and reply hashes |
| [run/line.env](run/line.env) | the line's variables and flags on top of the preset |
| [run/r115_fit.sh](run/r115_fit.sh) | the fit probe: three lines per file through `scripts/fit_probe.sh` |
| [run/r115_lane.sh](run/r115_lane.sh) | the step pair through `bench/deep_lane/remote.sh` (dry run with its check, start, wait, collect) |

## Commands

```bash
# from a machine that reaches the rig over ssh, at the repository root (RIG: its ssh alias; paths are the rig's)
# 1. the fit: header-only copies of both files' shards, unpacked on the rig with tar -xSzf into HEADERS/<file>/
RIG=$RIG HEADERS=$HEADERS DRAFT=$DRAFT BUILDS=/opt/llama-builds BUILD=$BUILD OUT=fit_probe.md \
  bash docs/data/2026-11-12_r115_gmix_step/run/r115_fit.sh
# 2. the step pair (MODEL: pmix-q2-fast's shard 1, GMIX: gmix-q2's shard 1, TEXT_LOCAL: proxy.jsonl.bodies.gz here)
export RIG WORK=/tmp/deep-lane-r115 BIN=$BUILD_DIR/bin/llama-server MODEL GMIX DRAFT TEXT_LOCAL OUT
D=docs/data/2026-11-12_r115_gmix_step/run
bash $D/r115_lane.sh dry && bash $D/r115_lane.sh start
bash $D/r115_lane.sh wait && bash $D/r115_lane.sh collect

# on a build machine, CPU only (OUT: the collected folder; RES: the tables)
R=docs/data
N="nice -n 19 python3 -I"
$N $R/2026-11-10_v4_candidate/run/v4c_table.py $OUT --control pmix --arm gmix --md $RES/step_pair.md \
  --tsv $RES/per_load.tsv --classes $RES/classes.tsv
$N $R/2026-11-10_deep_lane/run/card_fit.py --probe fit_probe.md $OUT > $RES/card_fit.tsv
$N $R/2026-11-11_r90_v4_vs_v3/run/card_max.py $OUT > $RES/card_max.tsv
$N $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
$N $R/2026-11-12_r60_taper_v4/run/realloc_counts.py pmix1=$OUT/server-pmix1.log gmix1=$OUT/server-gmix1.log \
  pmix2=$OUT/server-pmix2.log gmix2=$OUT/server-gmix2.log > $RES/reallocs.tsv
grep -a "load_model: loading model" $OUT/server-*.log   # the file each load read
```

The published `table.md`, `step_pair.md` and `loads.tsv` give `gmix-q2`'s first shard by its file name; on the rig its
`-m` held the folder too. Licence: the tables are this repository's; the model files are UkisAI's Swift-1.5 GSQ-RCO
Q2_0 recomposed ([the pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)).
