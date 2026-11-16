# Flint's candidate stack split at depth (R165, Q1486; 2026-11-16)

What this measures: which of Flint's candidate switches costs the deep prompts and decode that the candidate stack
lost in R161's deep lane ([2026-11-14_r161_flint_release_deep](../2026-11-14_r161_flint_release_deep/README.md)).
There, the v4 preset with every untried candidate of [release/flint-v4](../../../release/flint-v4/README.md) stacked
decoded −1.4 % and read prompts −14.8 % slower than the plain preset over six depths, −20.6 % on the prompt at
32,768 tokens, and gained only at depth 0. This run times the candidates one an arm against the plain preset, on nine
P106-090 cards, Flint Pascal Q8 GGUF (`p065`, 40.4 GB) at the full 262,144-token context, in two runs:

| Run | Arm | Settings | State |
|---|---|---|---|
| R165a, R165b | `v4` (the control) | the v4 preset ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env)) as it is: v3.0's line and three switches | |
| R165a | `v4gf` | `v4` plus `GGML_CUDA_GRAPHS_FORCE=1` | done 2026-11-16 |
| R165a | `v4hs` | `v4` plus `GGML_CUDA_HOP_STAGED=1` | done 2026-11-16 |
| R165b | `v4ka` | `v4` plus `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` | not yet run |
| R165b | `v4ck` | `v4` plus `--ctx-checkpoints-vram 4` | not yet run |
| R165b | `v4tp` | `v4` plus `--ubatch-taper short=600:64` | not yet run |

`--ctx-checkpoints-incall`, the stack's seventh switch, gets no arm: the server turns it off on Flint at load
(`not with sliding-window layers`, R161). `LLAMA_SCHED_PER_SIZE_RESERVE=2` was never in the stack: it acts only when the
scheduler re-plans, and no load of R161 re-planned. Every arm runs `v4-rc2` (`pascal-p106-v4` `eb82916d3`, upstream
b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is
`v4-rc2`'s). Registry row Q1486 in [quants/experiments.md](../../../quants/experiments.md).

## R165a: forced graphs and staged hops

**Neither of the two is what cost the stack its deep prompts.** The stack lost most at 32,768 and 49,152 tokens:
prompts −20.6 % and −17.3 %, decode −2.0 % and −3.1 % (R161). At those depths both candidates read prompts within
the plain preset's spread:
- at 32,768: forced graphs 334.7 and 334.5 tok/s, staged hops 335.1 and 334.5, against `v4`'s 338.6 and 332.6;
- at 49,152: 218.8 and 217.9, and 217.9 and 218.3, against 220.0 and 218.2.

There, forced graphs decode as fast as `v4`, and staged hops decode faster. The stack's loss lies in R165b's three
candidates: the keep-alive, the checkpoints on the cards, or the taper.

**`v4gf` (`GGML_CUDA_GRAPHS_FORCE=1`) is slower at no depth.**
- Decode is negligible at every depth. Over all depths it is a gain: 40.50 and 40.55 tok/s against 40.42 and 40.40
  (+0.3 %, beyond the spreads 0.04 and 0.02).
- Prompts are a gain on the 16,384 chunk: 516.4 and 514.5 tok/s against 493.8 and 483.1 (+5.5 %). They are negligible
  at every other depth.
- All five prompt chunks (53,200 tokens) read at 325.7 and 321.9 tok/s against 324.2 and 320.3 (+0.5 %, negligible).

**`v4hs` (`GGML_CUDA_HOP_STAGED=1`) decodes faster, and reads one prompt chunk slower.**
- Decode is a gain at every depth but 8,192: +5.3 % at 0, +1.7 % at 4,096, +1.2 % at 16,384, +0.9 % at 32,768 and
  +0.8 % at 49,152. At 8,192 it is negligible (+0.5 %).
- Over all depths decode runs 40.99 and 41.03 tok/s against 40.42 and 40.40 (+1.5 %, a gain).
- Prompts are a gain on the 64-token prompt at depth 0: 88.7 and 89.4 tok/s against 86.4 and 87.1 (+2.6 %).
- They are a loss on the 8,192 chunk: 446.1 and 437.4 tok/s against 478.2 and 479.2 (−7.7 %).
- They are negligible at 4,096, 16,384, 32,768 and 49,152.
- All five prompt chunks read at 321.4 and 318.3 tok/s against 324.2 and 320.3 (−0.7 %, negligible: the spreads are
  3.1 and 3.8).

Every value is one load. Run on one rig on 2026-11-16, 10:30:04–12:09:07: a warm-up load of `v4`, then
`v41 gf1 hs1 v42 gf2 hs2`, every load through. The RAM watchdog did not act.

### Per depth

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. Prompts: the
prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on an empty cache; the
chunk read and the tokens cached are the same in every arm). `v4`'s loads are listed `v41`, `v42`; `v4gf`'s `gf1`,
`gf2`; `v4hs`'s `hs1`, `hs2`. Each class is against `v4`.

| Depth (tokens) | Decode tok/s, `v4` | `v4gf` | Class | `v4hs` | Class |
|---:|---|---|---|---|---|
| 0 | 50.55, 49.04 | 50.54, 50.55 | negligible (+1.5 %) | 52.31, 52.52 | gain (+5.3 %) |
| 4,096 | 77.34, 77.45 | 77.40, 77.44 | negligible (+0.0 %) | 78.65, 78.74 | gain (+1.7 %) |
| 8,192 | 48.88, 49.46 | 49.26, 49.42 | negligible (+0.3 %) | 49.47, 49.35 | negligible (+0.5 %) |
| 16,384 | 67.28, 67.48 | 67.48, 67.48 | negligible (+0.2 %) | 68.08, 68.26 | gain (+1.2 %) |
| 32,768 | 26.39, 26.50 | 26.46, 26.50 | negligible (+0.1 %) | 26.68, 26.70 | gain (+0.9 %) |
| 49,152 | 23.52, 23.57 | 23.51, 23.53 | negligible (−0.1 %) | 23.73, 23.74 | gain (+0.8 %) |
| all | 40.42, 40.40 | 40.50, 40.55 | gain (+0.3 %) | 40.99, 41.03 | gain (+1.5 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v4` | `v4gf` | Class | `v4hs` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 86.4, 87.1 | 85.5, 86.4 | negligible (−0.9 %) | 88.7, 89.4 | gain (+2.6 %) |
| 4,096 | 4,032 on 0 | 519.0, 523.4 | 541.9, 464.8 | negligible (−3.4 %) | 532.6, 447.6 | negligible (−6.0 %) |
| 8,192 | 4,100 on 4,028 | 478.2, 479.2 | 498.5, 483.6 | negligible (+2.6 %) | 446.1, 437.4 | loss (−7.7 %) |
| 16,384 | 12,292 on 4,028 | 493.8, 483.1 | 516.4, 514.5 | gain (+5.5 %) | 498.5, 498.4 | negligible (+2.1 %) |
| 32,768 | 16,388 on 16,316 | 338.6, 332.6 | 334.7, 334.5 | negligible (−0.3 %) | 335.1, 334.5 | negligible (−0.2 %) |
| 49,152 | 16,388 on 32,700 | 220.0, 218.2 | 218.8, 217.9 | negligible (−0.4 %) | 217.9, 218.3 | negligible (−0.5 %) |
| all primes | 53,200 | 324.2, 320.3 | 325.7, 321.9 | negligible (+0.5 %) | 321.4, 318.3 | negligible (−0.7 %) |

At 4,096 tokens each candidate's second load read the chunk slower than its first: 464.8 against 541.9 tok/s for
forced graphs, 447.6 against 532.6 for staged hops. That gives spreads of 77.0 and 84.9 tok/s, and both classes
there are negligible. The sampled request (the line's own sampler, fixed seed) at 49,152 tokens runs 22.64 and 22.67
tok/s for `v4`, 22.67 and 22.67 for `v4gf` and 22.85 and 22.85 for `v4hs` ([table.md](table.md), "The sampler pass").

### The replies, the host and the cards

- **Every arm writes the same greedy reply on every pass and load at every depth.** Both candidates change speed only.
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable** was 958 and 960 MiB for `v4`, 952 and 949 for `v4gf`, and 937 and 931 for
  `v4hs`. MemAvailable after loading was 1,440 and 1,438, 1,436 and 1,436, and 1,414 and 1,417 MiB: staged hops left 26 and
  21 MiB less than `v4`'s loads, the pinned host ring they keep. Swap in use peaked at 293 or 294 MiB in every load. The RAM watchdog did not
  act ([prompt_depth.md](prompt_depth.md)).
- **The cards after loading** held the same memory in every load: 48,613 MiB over the nine, cards 1 and 4 at 5,923 MiB.
- **Each server printed the switches it was given:** `switches: 3 set` for `v4` (release/flint-v4's three), `4 set`
  for `v4gf` and `v4hs`, the fourth being the candidate.
- **The control beside R161's:** this run's `v4` decoded 40.42 and 40.40 tok/s over all depths (R161's `v4`: 40.25
  and 40.36). It read all primes at 324.2 and 320.3 tok/s (R161: 318.9 and 321.3). These are different runs, so they
  get no class.

## R165b: the keep-alive, the checkpoints on the cards and the taper

Not yet run. Its command is R165a's with its own arms and loads (below, "Commands").

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R161 ran it, on one build: the sampler
  pass on, the recorded agent text of the release check's `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768
  and 49,152 tokens.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads (R165a):** one untimed warm-up load of `v4` at depth 0, then `v41 gf1 hs1 v42 gf2 hs2`, each a fresh server
  start after a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog with its
  default limits. Nothing else ran on the rig.
- **The lines:** the lane reads `v4`'s from release/flint-v4/settings.env (its `PRESET_ENV` and `PRESET_ARGS`); each
  candidate arm adds its own variables (`ARM_ENV_<arm>`) or flags (`ARM_ARGS_<arm>`) and nothing else. The run script
  refuses a control with switches of its own, a candidate with none, and a candidate variable the preset already sets.
  A dry run printed the lines first: each candidate's line is `v4`'s with one word more. `loads.tsv` holds them as
  they ran.
- **Classes** (the lane's rule), per depth and over all depths, on a load's speed (both decode passes' tokens over
  their time; for prompts, the prime's chunk over its time): the spread is the larger of the two arms' differences
  between their two loads; an arm is a gain when its mean is better than the control's by more than the spread, a
  loss when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
# R165a (DRY=1 first: prints each arm's line and starts nothing)
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4-rc2 tree on the rig>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  ARMS="v4 v4gf v4hs" ARM_ENV_v4gf=GGML_CUDA_GRAPHS_FORCE=1 ARM_ENV_v4hs=GGML_CUDA_HOP_STAGED=1 \
  LOADS="warm:v4 v41:v4 gf1:v4gf hs1:v4hs v42:v4 gf2:v4gf hs2:v4hs" \
  bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=wait bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=collect OUT=<folder> \
  bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
# R165b: the same with
#   ARMS="v4 v4ka v4ck v4tp" ARM_ENV_v4ka="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1"
#   ARM_ARGS_v4ck="--ctx-checkpoints-vram 4" ARM_ARGS_v4tp="--ubatch-taper short=600:64"
#   LOADS="warm:v4 v41:v4 ka1:v4ka ck1:v4ck tp1:v4tp v42:v4 ka2:v4ka ck2:v4ck tp2:v4tp"
```

The tables (the driver writes `table.md` and `classes.tsv` at the end of the run; any machine with Python 3 builds the
same):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v4 --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4 --out prompt_depth.md \
  --tsv prompt_depth
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | R165a, `deep_table.py`'s tables, control `v4`: decode by depth and over all depths, the step, prompts, the sampler pass, the checks |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | R165a, `prompt_depth.py`'s tables, control `v4`: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | R165a, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [run/r165.sh](run/r165.sh) | the run, both parts: start, wait, status, collect; the arms from the environment |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2`. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)); the prompts are the deep
lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and `requests.csv` hold
times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
