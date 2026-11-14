# Ornith's release deep lane: v4 against v3.0 at depth, with the candidate stack (R162, Q1481; 2026-11-14)

What this measured: how fast Miner Llama v4's release candidate decodes and reads prompts deep into a context with
Ornith-1.5-122B, against the v3.0 release, and whether the candidate switches of
[release/ornith-v4](../../../release/ornith-v4/README.md) ("Candidates"), stacked in one arm, help or hurt there.
Three arms on nine P106-090 cards, Ornith-1.5-122B Pascal N2P GGUF (`valid1`: N2P and N3P experts, a Q5_0 output
layer), a 49,152-token context, no draft:

| Arm | Build | Settings |
|---|---|---|
| `v3` (control of the release table) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag `miner-llama-v3.0`) | v3.0's preset as shipped ([release/ornith-v3/settings.env](../../../release/ornith-v3/settings.env)): the keep-alive `GGML_CUDA_KEEPALIVE=decode` |
| `v4` (control of the stack) | `v4-rc2`, `pascal-p106-v4` `eb82916d3` (upstream b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is `v4-rc2`'s) | the v4 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)): v3.0's line, the keep-alive and three switches, `GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1` |
| `v4s` (the candidate stack) | `v4-rc2` | `v4` plus every candidate but `LLAMA_SCHED_PER_SIZE_RESERVE=2` (no load re-planned its graphs): `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`, `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` and `-b 4096` in place of `-b 2048` |

Registry row Q1481 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**v4 alone is level with v3.0 on Ornith.** Decode over all six depths runs 12.92 and 12.92 tok/s in `v4`'s two loads
against v3.0's 12.88 and 12.88 (+0.3 %): a gain at every depth by the lane's rule (+0.3, +0.4, +0.6, +0.4, +0.2 and +0.1 %), since the
loads of each arm differ by at most 0.02 tok/s. Prompts read 145.2 and 145.0 tok/s over all five primes (54,736
tokens) against 146.8 and 146.7 (−1.1 %, a loss): a gain at 4,096 tokens (+1.1 %), a loss at every depth from 8,192
(−0.9, −0.9, −2.1 and −0.9 %). `v4` writes the same greedy reply as v3.0 at every depth.

**The candidate stack reads prompts 15.1 % faster than `v4` and decodes level with it.** `v4s` reads all five primes
at 167.2 and 166.9 tok/s against `v4`'s 145.2 and 145.0 (+15.1 %, a gain at every depth): 327.3 and 328.0 s against
377.0 and 377.4 s. The 64-token prompt at depth 0 reads at 40.6 and 40.6 tok/s against 24.6 and 24.6 (+65.1 %).
Decode over all depths runs 12.94 and 12.93 tok/s against 12.92 and 12.92 (+0.1 %, a gain): a gain at 0 and 4,096
tokens (+0.3 %, +0.4 %), negligible at 8,192, 32,768 and 48,640, and a loss at 16,384 (13.37 and 13.37 against 13.39
and 13.38, −0.1 %, beyond a spread of 0.01). With the checkpoints on the cards the host keeps much more memory free:
its lowest MemAvailable is 1,238 and 1,254 MiB in `v4s`'s loads against 651 and 655 in `v4`'s.

**`v4s` writes another greedy reply than `v4` and v3.0 at 8,192, 16,384 and 32,768 tokens,** the same reply in both
of its loads; at 0, 4,096 and 48,640 tokens all three arms write the same reply. One of the stacked candidates changes
the numbers at the last bits, so a near-tie in the logits goes the other way. Ornith decodes one token a step, so the
reply does not change a step's time; which candidate does it is registry row Q1487.

**The release table's deep-lane line:** *Deep into a context (Ornith-1.5-122B Pascal N2P GGUF, nine P106 cards, two
loads each), v4 decodes at 12.92 and 12.92 tok/s against v3.0's 12.88 and 12.88 (+0.3 %) over six depths from 0 to
48,640 tokens, and reads prompts at 145.2 and 145.0 tok/s against 146.8 and 146.7 (−1.1 %); at 48,640 tokens it
decodes 8.63 and 8.62 against 8.61 and 8.61 and reads 119.7 and 119.2 against 120.7 and 120.5.*

**The stack's line:** *v4 plus the staged hop, forced prompt graphs, four checkpoints on the cards and inside a
prompt's call, the micro-batch taper and `-b 4096` reads prompts at 167.2 and 166.9 tok/s against v4's 145.2 and 145.0
(+15.1 %, a gain at every depth) and decodes at 12.94 and 12.93 against 12.92 and 12.92 (+0.1 %), on another greedy
reply at three of six depths.*

Every value is one load. Run on one rig on 2026-11-14, 10:16:46–12:50:47: a warm-up load of `v3`, then
`v31 v41 s1 v32 v42 s2`, after a fit load of `v4s` alone (09:59:46–10:15:31). No load was stopped.

### Per depth: v4 against v3.0

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. Prompts: the
prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on an empty cache; every
arm read the same chunk on the same cached tokens). `v3`'s loads are `v31`, `v32`; `v4`'s `v41`, `v42`.

| Depth (tokens) | Decode tok/s, `v3` | `v4` | Class | Prompt tok/s, `v3` | `v4` | Class |
|---:|---|---|---|---|---|---|
| 0 | 18.34, 18.32 | 18.38, 18.39 | gain (+0.3 %) | 24.6, 24.3 | 24.6, 24.6 | negligible (+0.5 %) |
| 4,096 | 16.82, 16.80 | 16.87, 16.86 | gain (+0.4 %) | 132.4, 132.2 | 133.8, 133.9 | gain (+1.1 %) |
| 8,192 | 15.46, 15.45 | 15.54, 15.54 | gain (+0.6 %) | 144.8, 144.5 | 143.2, 143.6 | loss (−0.9 %) |
| 16,384 | 13.34, 13.33 | 13.39, 13.38 | gain (+0.4 %) | 193.8, 193.4 | 191.9, 191.7 | loss (−0.9 %) |
| 32,768 | 10.43, 10.42 | 10.46, 10.44 | gain (+0.2 %) | 155.5, 155.5 | 152.3, 152.3 | loss (−2.1 %) |
| 48,640 | 8.61, 8.61 | 8.63, 8.62 | gain (+0.1 %) | 120.7, 120.5 | 119.7, 119.2 | loss (−0.9 %) |
| all | 12.88, 12.88 | 12.92, 12.92 | gain (+0.3 %) | 146.8, 146.7 | 145.2, 145.0 | loss (−1.1 %) |

### Per depth: the stack against v4

`v4s`'s loads are `s1`, `s2`; classes against `v4` ([table_v4.md](table_v4.md), [prompt_depth_v4.md](prompt_depth_v4.md)).

| Depth (tokens) | Decode tok/s, `v4` | `v4s` | Class | Prompt tok/s, `v4` | `v4s` | Class |
|---:|---|---|---|---|---|---|
| 0 | 18.38, 18.39 | 18.45, 18.43 | gain (+0.3 %) | 24.6, 24.6 | 40.6, 40.6 | gain (+65.1 %) |
| 4,096 | 16.87, 16.86 | 16.94, 16.93 | gain (+0.4 %) | 133.8, 133.9 | 169.4, 168.8 | gain (+26.4 %) |
| 8,192 | 15.54, 15.54 | 15.54, 15.57 | negligible (+0.1 %) | 143.2, 143.6 | 182.3, 182.2 | gain (+27.1 %) |
| 16,384 | 13.39, 13.38 | 13.37, 13.37 | loss (−0.1 %) | 191.9, 191.7 | 219.8, 219.4 | gain (+14.5 %) |
| 32,768 | 10.46, 10.44 | 10.44, 10.43 | negligible (−0.1 %) | 152.3, 152.3 | 168.8, 168.6 | gain (+10.8 %) |
| 48,640 | 8.63, 8.62 | 8.65, 8.63 | negligible (+0.2 %) | 119.7, 119.2 | 136.7, 136.4 | gain (+14.3 %) |
| all | 12.92, 12.92 | 12.94, 12.93 | gain (+0.1 %) | 145.2, 145.0 | 167.2, 166.9 | gain (+15.1 %) |

Against v3.0 the stack reads all primes 13.9 % faster (167.2 and 166.9 against 146.8 and 146.7) and decodes 0.4 %
faster (12.94 and 12.93 against 12.88 and 12.88; [table.md](table.md), [prompt_depth.md](prompt_depth.md)).

### The sampler pass, the step and the checks

- **The sampled request** (the server line's sampler: `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`, a fixed seed)
  runs 18.11 and 18.14 tok/s for `v3`, 18.22 and 18.21 for `v4` and 18.21 and 18.20 for `v4s` at depth 0, and 8.60
  and 8.59, 8.61 and 8.60, 8.64 and 8.64 at 48,640 ([table.md](table.md), "The sampler pass").
- **A step** is one token (no draft): at 48,640 tokens 116.1 and 116.2 ms in `v3`, 116.0 and 116.1 in `v4`, 115.7
  and 116.0 in `v4s`.
- No graph reallocated on any request, every timed request gave 256 tokens, and every load read the same prompt ids.

### The host and the cards

- **The fit load** ([fit/](fit/memory.tsv)): `v4s` alone, one prompt of 48,640 tokens (48,516 read at 182.5 tok/s),
  256 tokens written twice greedy and once sampled, with each card's memory and the host's MemAvailable logged every
  second. The highest card memory was 5,773 MiB of 6,144 on cards 1 and 3 (5,635 after loading); the lowest
  MemAvailable 1,227 MiB, during the sampled request; the limit was 400 MB. The stack fitted whole, so nothing was
  dropped. The started command holds one `-b`, 4096: the arm's value replaced the preset's 2048 in place.
- **After loading** all three arms hold the same memory on every card, 49,139 MiB over the nine; the checkpoints and
  the forced graphs take theirs while prompts are read.
- **The lowest MemAvailable** of each load: `v31` 672 MiB, `v32` 661, `v41` 651, `v42` 655, `s1` 1,238, `s2` 1,254
  ([prompt_depth.md](prompt_depth.md)). v3.0 and `v4` keep the four checkpoints in host memory, `v4s` on the cards.
  The RAM watchdog never acted.

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R157 ran it
  ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)): builds per arm in one run (`BIN` is
  `v4-rc2`, `ARM_BIN_v3` the v3.0 build), the sampler pass on, the recorded agent text of the release check's `t1swift`
  sessions, depths 0, 4,096, 8,192, 16,384, 32,768 and 48,640 tokens (the deepest whose 256 written tokens fit the
  49,152-token context).
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (pass 1, the first request at the new depth, and pass 2, the same
  request again on the same cache), then 256 tokens with the server line's own sampler and a fixed seed. End of text is
  ignored.
- **Loads:** one untimed warm-up load of `v3` at depth 0, then `v31 v41 s1 v32 v42 s2`, each a fresh server start
  after a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog with its default
  limits. Nothing else ran on the rig.
- **The lines:** the lane reads [release/ornith-v3/settings.env](../../../release/ornith-v3/settings.env) as its
  preset; `v4`'s three switches and `v4s`'s switches and flags are the arms' own. A dry run printed all three lines
  first: `v3`'s is release/ornith-v3's, `v4`'s release/ornith-v4's, `v4s`'s `v4`'s plus the stack. `loads.tsv` holds
  them as they ran.
- **The fit load** came first, with the same lane on `v4s` alone at 48,640 tokens and `run/fitmon.sh` beside it. The
  rule: the stack fits when no card runs out of memory and MemAvailable stays at or above 400 MB; a host running short
  would have dropped `-b 4096` first, a card running short `GGML_CUDA_GRAPHS_FORCE=1` first, then the checkpoints on
  the cards.
- **Classes** per depth and over all depths (the lane's rule): the spread is the larger of the two arms' differences
  between their two loads; an arm is a gain when its mean is better than the control's by more than the spread, a loss
  when worse by more than it, else negligible. `v4` is classed against `v3`, `v4s` against `v4`.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
RIG=<rig alias> WORK=<run folder on the rig> \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server REF_BIN=<v3.0 tree on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> STEP=fit bash docs/data/2026-11-14_r162_ornith_release_deep/run/r162.sh
# then STEP=wait WORK=<WORK>-fit, stop fitmon.sh on the rig, STEP=collect WORK=<WORK>-fit OUT=<folder>
# the run: the same with DRY=1, then STEP=start, STEP=wait, STEP=tables, STEP=collect OUT=<folder>
```

`STEP=tables` builds the tables on the rig, before `collect` removes the run folder:

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v3 --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control v4 --out table_v4.md --tsv classes_v4.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v3 --out prompt_depth.md --tsv prompt_depth
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4 --out prompt_depth_v4.md --tsv prompt_depth_v4
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | `deep_table.py`'s tables, control `v3`: decode, prompts, the step, the sampler pass, the checks |
| [table_v4.md](table_v4.md), [classes_v4.tsv](classes_v4.tsv) | the same, control `v4` (the stack's classes) |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | `prompt_depth.py`'s tables, control `v3`: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [prompt_depth_v4.md](prompt_depth_v4.md) and its three `.tsv` | the same, control `v4` |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran; per request its counts, times and reply md5s |
| [fit/memory.tsv](fit/memory.tsv) | the fit load: each card's memory after loading and at its highest, the host's MemAvailable |
| [fit/table.md](fit/table.md), [fit/loads.tsv](fit/loads.tsv), [fit/requests.csv](fit/requests.csv) | the fit load's lane output |
| [run/r162.sh](run/r162.sh), [run/fitmon.sh](run/fitmon.sh) | the run (fit, start, wait, status, tables, collect); the per-second memory log of the fit load |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: Ornith-1.5-122B as Ornith-1.5-122B Pascal N2P GGUF
(`valid1`), described with its licence in [release/ornith-v3](../../../release/ornith-v3/README.md); the prompts are the
deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
