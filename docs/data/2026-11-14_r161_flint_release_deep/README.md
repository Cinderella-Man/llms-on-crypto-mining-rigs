# Flint's release deep lane: v4 against v3.0 at depth, and the candidate stack (R161, Q1480; 2026-11-14)

What this measured: how fast Miner Llama v4's release candidate decodes and reads prompts deep into a context on
Flint, with the v4 preset, against the v3.0 release, on the same file, line and rig; and, as a first screen, the same
preset with every untried candidate switch of [release/flint-v4](../../../release/flint-v4/README.md) stacked. Three
arms on nine P106-090 cards, Flint Pascal Q8 GGUF (`p065`, 40.4 GB) at the full 262,144-token context:

| Arm | Build | Settings |
|---|---|---|
| `v3` (control of the release table) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag `miner-llama-v3.0`) | v3.0's Flint preset as shipped ([release/flint-v3/settings.env](../../../release/flint-v3/settings.env)), no switch |
| `v4` (control of the stack's screen) | `v4-rc2`, `pascal-p106-v4` `eb82916d3` (upstream b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is `v4-rc2`'s) | the v4 preset ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env)): v3.0's line and three switches |
| `v4s` (the candidate stack) | `v4-rc2` | `v4` plus `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1` and `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` |

`v4s` holds every candidate of release/flint-v4's list but `LLAMA_SCHED_PER_SIZE_RESERVE=2`, which acts only when the
scheduler re-plans; no load of this run re-planned (below). Registry row Q1480 in
[quants/experiments.md](../../../quants/experiments.md).

## Result

**v4 is a little faster than v3.0 on Flint at every depth, and writes the same replies.** Decode over all six depths
runs 40.25 and 40.36 tok/s in `v4`'s two loads against v3.0's 40.02 and 40.04 (+0.7 %, a gain), a gain at every depth
from 4,096 tokens and negligible at depth 0. All five prompt chunks of a load (53,200 tokens) read at 318.9 and
321.3 tok/s against 309.9 and 311.0 (+3.1 %, a gain): 166.8 and 165.6 s against 171.7 and 171.1 s. At 49,152 tokens
`v4` decodes 23.44 and 23.44 tok/s against 23.29 and 23.31 (+0.6 %) and reads 218.0 and 219.2 tok/s against 212.7 and
213.6 (+2.6 %). One loss: the 64-token prompt at depth 0 reads at 84.5 and 84.0 tok/s against 85.1 and 85.2 (−1.1 %,
11.83 and 11.90 ms a token against 11.74 and 11.73).

**The candidate stack `v4s` loses to `v4` from 4,096 tokens on, so it is split.** It gains only where the context is
short: at depth 0 it decodes 51.37 and 52.23 tok/s against `v4`'s 49.78 and 50.79 (+3.0 %) and reads the 64-token
prompt at 111.1 and 110.2 tok/s against 84.5 and 84.0 (+31.3 %), and it reads the 4,096-token prompt at 539.5 and
542.0 against 503.3 and 515.7 (+6.1 %). From 4,096 tokens on it decodes slower at every depth, −1.2 %, −1.1 %, −1.4 %,
−2.0 % and −3.1 % at 4,096, 8,192, 16,384, 32,768 and 49,152 tokens, and from 16,384 on it reads prompts slower,
−5.2 %, −20.6 % and −17.3 %. Over all depths: decode 39.67 and 39.79 tok/s against 40.25 and 40.36 (−1.4 %, a loss),
the five prompt chunks 271.7 and 273.5 tok/s against 318.9 and 321.3 (−14.8 %, a loss: 195.8 and 194.5 s against
166.8 and 165.6 s). Which candidate costs the deep prompts and decode is not known from this run: the stack is split
next (registry row Q1486).

**The release table's deep-lane line:** *Deep into a context (Flint Pascal Q8 GGUF, nine P106 cards, two loads each),
v4 decodes at 40.25 and 40.36 tok/s against v3.0's 40.02 and 40.04 (+0.7 %) and reads prompts at 318.9 and 321.3
tok/s against 309.9 and 311.0 (+3.1 %) over six depths from 0 to 49,152 tokens; at 49,152 tokens it decodes 23.44 and
23.44 against 23.29 and 23.31 (+0.6 %) and reads 218.0 and 219.2 against 212.7 and 213.6 (+2.6 %).*

Every value is one load. Run on one rig on 2026-11-14, 10:16:22–11:56:51: a warm-up load of `v3`, then
`v31 v41 s1 v32 v42 s2`, every load through, the RAM watchdog silent.

### Per depth

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. Prompts: the
prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on an empty cache; the
chunk read and the tokens cached are the same in every arm). `v3`'s loads are listed `v31`, `v32`; `v4`'s `v41`,
`v42`; `v4s`'s `s1`, `s2`. The release class is `v4` against `v3`; the stack's class is `v4s` against `v4`.

| Depth (tokens) | Decode tok/s, `v3` | `v4` | Class | `v4s` | Class against `v4` |
|---:|---|---|---|---|---|
| 0 | 49.94, 49.96 | 49.78, 50.79 | negligible (+0.7 %) | 51.37, 52.23 | gain (+3.0 %) |
| 4,096 | 76.51, 76.54 | 77.31, 77.33 | gain (+1.0 %) | 76.42, 76.40 | loss (−1.2 %) |
| 8,192 | 48.37, 48.37 | 48.77, 48.62 | gain (+0.7 %) | 48.10, 48.18 | loss (−1.1 %) |
| 16,384 | 66.56, 66.74 | 66.91, 67.20 | gain (+0.6 %) | 66.10, 66.13 | loss (−1.4 %) |
| 32,768 | 26.18, 26.16 | 26.35, 26.36 | gain (+0.7 %) | 25.81, 25.83 | loss (−2.0 %) |
| 49,152 | 23.29, 23.31 | 23.44, 23.44 | gain (+0.6 %) | 22.71, 22.74 | loss (−3.1 %) |
| all | 40.02, 40.04 | 40.25, 40.36 | gain (+0.7 %) | 39.67, 39.79 | loss (−1.4 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v3` | `v4` | Class | `v4s` | Class against `v4` |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 85.1, 85.2 | 84.5, 84.0 | loss (−1.1 %) | 111.1, 110.2 | gain (+31.3 %) |
| 4,096 | 4,032 on 0 | 477.5, 478.2 | 503.3, 515.7 | gain (+6.6 %) | 539.5, 542.0 | gain (+6.1 %) |
| 8,192 | 4,100 on 4,028 | 442.6, 446.7 | 443.6, 476.3 | negligible (+3.4 %) | 457.4, 511.4 | negligible (+5.3 %) |
| 16,384 | 12,292 on 4,028 | 472.6, 475.4 | 488.5, 490.8 | gain (+3.3 %) | 462.1, 466.7 | loss (−5.2 %) |
| 32,768 | 16,388 on 16,316 | 321.9, 322.3 | 332.7, 332.2 | gain (+3.2 %) | 263.8, 264.0 | loss (−20.6 %) |
| 49,152 | 16,388 on 32,700 | 212.7, 213.6 | 218.0, 219.2 | gain (+2.6 %) | 180.8, 180.7 | loss (−17.3 %) |
| all primes | 53,200 | 309.9, 311.0 | 318.9, 321.3 | gain (+3.1 %) | 271.7, 273.5 | loss (−14.8 %) |

Decode depends on the reply as well as on the depth: Flint writes another reply at each depth (the same one in every
arm), and its pipelined speculation (`--pipe-depth 4`) pays off more on some replies than on others, so 4,096 tokens
decode faster than depth 0 and 8,192 slower than 16,384 here; between arms at one depth the reply is the same. The
sampled request (the line's own sampler, fixed seed) at 49,152 tokens runs 22.57 and 22.56
tok/s for `v4` against 22.39 and 22.40 for `v3` and 21.88 and 21.90 for `v4s` ([table.md](table.md), "The sampler
pass").

### The replies, the host and the cards

- **Every arm writes the same greedy reply on every pass and load at every depth**: v4's three switches and the stack
  change speed only.
- **No target graph reallocated** on any request of any load, so `LLAMA_SCHED_PER_SIZE_RESERVE=2`, which acts only
  after a re-plan, stays out of the stack. Every timed request gave 256 tokens, and every load read the same prompt
  ids.
- **The host:** the lowest MemAvailable was 959 and 943 MiB for `v3`, 942 and 954 for `v4`, 1,136 and 1,119 for `v4s`
  (which keeps its four checkpoints on the cards); the RAM watchdog did not act ([prompt_depth.md](prompt_depth.md)).
- **The cards after loading** held the same memory in every load, 48,613 MiB over the nine (cards 1 and 4 5,923 MiB).

## The stack's fit load

Before the lane, one load of `v4s` alone (no warm-up) read one prompt at the deepest depth, 49,152 tokens, and wrote
its 256 tokens twice and once more with the sampler, so that the forced CUDA graphs and the checkpoints took their
memory; a monitor logged every card's memory every 2 s and the host's MemAvailable every second
([fit/](fit/README.md)). The stack fits, and nothing was dropped:

- **The cards:** the highest memory of each card over the load was 5,427, 5,971, 5,121, 5,633, 5,971, 5,121, 5,587,
  5,341 and 4,873 MiB (cards 0 to 8, of 6,144). After loading they held 5,383, 5,923, 5,081, 5,591, 5,923, 5,081,
  5,545, 5,293 and 4,793 MiB, the 5,923 MiB of cards 1 and 4 that the v4 preset holds there too; the prompt and the
  replies added 44, 48, 40, 42, 48, 40, 42, 48 and 80 MiB, and cards 1 and 4 kept 173 MiB free at their highest.
- **The host:** the lowest MemAvailable was 1,099 MiB (usable memory, MemAvailable less CmaFree, the same: CmaFree was
  0), against the 400 MiB the fit asked for; MemAvailable after loading 1,397 MiB. The RAM watchdog did not act.
- **The load:** the server was up after 587 s; the prime read 49,028 tokens on 60 cached at 271.8 tok/s; decode at
  49,152 tokens ran 22.32 tok/s on pass 1 and 23.24 on pass 2, the sampled request 29.28. No graph reallocated.
- **One candidate does nothing on Flint:** the server turns `--ctx-checkpoints-incall` off at load, with the warning
  `--ctx-checkpoints-incall: not with sliding-window layers - off`. Flint's attention is mostly a sliding window, so
  its checkpoints are taken between calls as before. It stays in `v4s`'s line, as the stack was defined, and has no
  effect there. `--ctx-checkpoints-vram 4` acts: the server keeps each slot's newest four checkpoints on the cards, a
  restored checkpoint of the 60-token tail taking 4.102 MiB on the cards.

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R157 ran it
  ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)): two builds in one run (`BIN` is
  `v4-rc2`, `ARM_BIN_v3` the v3.0 build), the sampler pass on, the recorded agent text of the release check's
  `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768 and 49,152 tokens.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v3` at depth 0, then `v31 v41 s1 v32 v42 s2`, each a fresh server start
  after a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog with its default
  limits. Nothing else ran on the rig.
- **The lines:** `v3`'s is release/flint-v3's `PRESET_ARGS` with no variable; `v4`'s is the same line with
  release/flint-v4's `PRESET_ENV` (the run script reads it from the file and stops if that file's line differs from
  v3.0's); `v4s`'s is `v4`'s with the stack's four variables and three flags. A dry run printed the three lines first;
  `loads.tsv` holds them as they ran.
- **Classes** (the lane's rule), per depth and over all depths, on a load's speed (both decode passes' tokens over
  their time): the spread is the larger of the two arms' differences between their two loads; an arm is a gain when
  its mean is better than the control's by more than the spread, a loss when worse by more than it, else negligible.
  The release table classes `v4` against `v3`; the stack's screen classes `v4s` against `v4`.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
RIG=<rig alias> WORK=<fit folder on the rig> \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server REF_BIN=<v3.0 tree on the rig>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  FIT=1 bash docs/data/2026-11-14_r161_flint_release_deep/run/r161.sh
# then FIT=1 STEP=wait, then FIT=1 STEP=collect OUT=<folder>; the lane: the same without FIT=1 and with its own
# WORK, DRY=1 first, then STEP=wait, then STEP=collect OUT=<folder>
```

The tables, built on the rig's own CPU after the run (any machine with Python 3 builds the same):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v3 --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control v4 --out table_v4.md --tsv classes_v4.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v3 --out prompt_depth.md \
  --tsv prompt_depth
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4 --out prompt_depth_v4.md \
  --tsv prompt_depth_v4
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | `deep_table.py`'s tables, control `v3`: decode by depth and over all depths, the step, prompts, the sampler pass, the checks |
| [table_v4.md](table_v4.md), [classes_v4.tsv](classes_v4.tsv) | the same with control `v4`: the stack's screen |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | `prompt_depth.py`'s tables, control `v3`: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [prompt_depth_v4.md](prompt_depth_v4.md) and its three `prompt_depth_v4_*.tsv` | the same with control `v4` |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran; per request its counts, times and reply md5s |
| [fit/](fit/README.md) | the stack's fit load: each card's memory every 2 s, the host's every second, the load's own files |
| [run/r161.sh](run/r161.sh) | the run: the fit load (`FIT=1`) and the lane; start, wait, status, collect |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)); the prompts are the deep
lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and `requests.csv` hold
times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
