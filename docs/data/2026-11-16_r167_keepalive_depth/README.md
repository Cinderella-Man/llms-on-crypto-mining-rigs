# The keep-alive at depth: Flash-Next, Ornith and Flint (R167, Q1492; 2026-11-16)

What this measures: whether the keep-alive slows a model deep into a context. The keep-alive is the pair of variables
`GGML_CUDA_KEEPALIVE=decode` (a one-thread block per card holds the boost clock while a split decodes) and
`GGML_CUDA_KEEPALIVE_EAGER=1` (that block ends before a kernel's first launch on a card, so the launch does not wait
for it). On Flint, R165b found that the two together read prompts 16.5 % slower over all depths and decoded 3.0 %
slower ([2026-11-16_r165_flint_stack_split](../2026-11-16_r165_flint_stack_split/README.md)). The v4 presets of
Flash-Next and Ornith ship both. Each part of R167 times one model's v4 preset on the deep lane against the same
preset with the variables changed, two loads an arm, on one build (`v4-rc2`) and one rig of nine P106-090 cards.

| Part | Model | Arms | State |
|---|---|---|---|
| R167b | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF with its MTP draft ([release/flash-next-v4](../../../release/flash-next-v4/README.md)) | `v4r` the preset as it is (control); `v4nk` the preset less both keep-alive variables; `v4ne` the preset less `GGML_CUDA_KEEPALIVE_EAGER=1` alone | done, below |
| R167a | Ornith-1.5-122B Pascal N2P GGUF ([release/ornith-v4](../../../release/ornith-v4/README.md)) | the same three arms on Ornith's v4 preset | not yet run |
| R167c | Flint Pascal Q8 GGUF ([release/flint-v4](../../../release/flint-v4/README.md)) | the v4 preset (which sets neither variable) against it plus `GGML_CUDA_KEEPALIVE=decode` alone | not yet run |

Registry row Q1492 in [quants/experiments.md](../../../quants/experiments.md).

## R167b: Flash-Next

**On Flash-Next the keep-alive does not cost what it costs on Flint.** Without it (`v4nk`) prompts read a little
faster and decode is slower:
- All five prompt chunks of a load (55,248 tokens) read at 219.4 and 218.2 tok/s against `v4r`'s 212.6 and 215.3
  (+2.3 %, a gain). The gain is on the three deepest chunks: +3.0 % at 16,384, +3.2 % at 32,768 and +1.4 % at 49,152
  tokens. On Flint the keep-alive cost −16.5 %.
- Decode over all six depths runs 25.72 and 25.86 tok/s against 26.15 and 26.11 (−1.3 %, a loss). Without the
  keep-alive decode gains at 4,096 (+6.1 %) and 8,192 (+3.7 %) and loses at 32,768 (−1.0 %) and 49,152 (−9.9 %:
  21.64 and 21.96 against 24.17 and 24.21).

**The eager stop is not the part that costs.** Without it (`v4ne`) prompts are within the spread at every depth (all
chunks 214.4 and 216.2 tok/s, +0.6 %, negligible). Decode over all depths runs 25.90 and 25.88 tok/s (−0.9 %, a
loss). Nearly all of that loss is one request: the first one after the 4,096-token prompt, where each step took 137.0
and 136.6 ms against 114.0 and 113.9 (about 1.8 s more over its 79 steps). That is the kind of wait for the
keep-alive block that the eager stop was made to remove (C811, C817). On the same request run again, and on every
other greedy request, `v4ne`'s step was within 3.6 % of `v4r`'s.

In short, for Flash-Next's preset: without the keep-alive, prompts read 2.3 % faster at depth, but decode runs 1.3 %
slower over all depths and 9.9 % slower at 49,152 tokens. Without the eager stop alone, prompts read no faster and
decode runs 0.9 % slower over all depths.

Every arm wrote the same greedy reply on every pass and load at every depth, with the same draft counts, so the speed
differences are the step's time alone. Every value is one load. Run on one rig on 2026-11-16, 15:49:53–17:02:38: a
warm-up load of `v4r`, then `r1 n1 e1 r2 n2 e2`, every load through. The RAM watchdog did not act.

### Per depth (R167b)

`v4r`'s loads are listed `r1`, `r2`; `v4nk`'s `n1`, `n2`; `v4ne`'s `e1`, `e2`. Each class is against `v4r`. Decode
is a load's speed: both greedy passes' tokens over their time.

| Depth (tokens) | Decode tok/s, `v4r` | `v4nk` | Class | `v4ne` | Class |
|---:|---|---|---|---|---|
| 0 | 23.57, 23.46 | 23.37, 23.78 | negligible (+0.2 %) | 23.70, 23.72 | gain (+0.8 %) |
| 4,096 | 28.97, 28.99 | 30.64, 30.84 | gain (+6.1 %) | 26.28, 26.29 | loss (−9.3 %) |
| 8,192 | 30.77, 30.52 | 31.24, 32.33 | gain (+3.7 %) | 31.23, 31.24 | gain (+1.9 %) |
| 16,384 | 30.84, 30.88 | 30.55, 28.98 | negligible (−3.5 %) | 31.19, 31.20 | gain (+1.1 %) |
| 32,768 | 21.63, 21.61 | 21.35, 21.47 | loss (−1.0 %) | 21.72, 21.50 | negligible (−0.1 %) |
| 49,152 | 24.17, 24.21 | 21.64, 21.96 | loss (−9.9 %) | 24.20, 24.38 | negligible (+0.4 %) |
| all | 26.15, 26.11 | 25.72, 25.86 | loss (−1.3 %) | 25.90, 25.88 | loss (−0.9 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v4r` | `v4nk` | Class | `v4ne` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 53.5, 51.7 | 47.2, 48.5 | loss (−9.0 %) | 53.3, 53.9 | negligible (+1.9 %) |
| 4,096 | 4,032 on 0 | 198.0, 197.3 | 198.1, 200.1 | negligible (+0.7 %) | 198.8, 197.6 | negligible (+0.3 %) |
| 8,192 | 4,612 on 3,516 | 207.9, 220.0 | 221.9, 218.9 | negligible (+3.0 %) | 221.6, 223.4 | negligible (+4.0 %) |
| 16,384 | 12,804 on 3,516 | 275.7, 281.1 | 290.7, 282.6 | gain (+3.0 %) | 284.7, 279.5 | negligible (+1.3 %) |
| 32,768 | 16,900 on 15,804 | 221.3, 224.6 | 230.1, 230.0 | gain (+3.2 %) | 225.7, 225.4 | negligible (+1.2 %) |
| 49,152 | 16,900 on 32,188 | 178.7, 179.0 | 181.5, 181.4 | gain (+1.4 %) | 174.7, 180.4 | negligible (−0.7 %) |
| all chunks | 55,248 | 212.6, 215.3 | 219.4, 218.2 | gain (+2.3 %) | 214.4, 216.2 | negligible (+0.6 %) |

**The step, where decode moves** ([r167b_table.md](r167b_table.md), "The step on pass 1" and "on pass 2"):
- `v4nk` takes less time a step on the first request at 4,096 and 8,192 tokens (105.1 and 105.0 ms against 114.0 and
  113.9; 112.4 and 110.0 against 118.5 and 120.5) and more at 49,152 (150.6 and 149.5 against 141.0 and 140.7 on pass
  1, 160.7 and 157.2 against 137.7 and 137.5 on pass 2, +15.5 %).
- `v4ne`'s one large difference is pass 1 at 4,096 (137.0 and 136.6 ms against 114.0 and 113.9, +20.0 %); its pass 2
  there takes 109.6 and 110.0 ms against 109.7 and 109.7. Classed on pass 2 alone ([r167b_table_pass2.md](r167b_table_pass2.md)),
  `v4ne`'s decode over all depths is 26.60 and 26.63 tok/s against 26.65 and 26.66 (−0.2 %).
- The sampled request (the server line's own sampler, a fixed seed) at 49,152 tokens runs 23.53 and 23.59 tok/s for
  `v4r`, 20.20 and 18.90 for `v4nk`, 23.82 and 22.11 for `v4ne`.

### The replies, the host and the cards (R167b)

- **Every arm writes the same greedy reply on every pass and load at every depth**, and accepts the same number of
  draft tokens on it (for example 180 of 225 at 49,152 tokens).
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable:** `v4r` 247 and 268 MiB, `v4nk` 307 and 288, `v4ne` 241 and 275. MemAvailable
  after loading was 866 and 840 MiB (`v4r`), 862 and 874 (`v4nk`), 866 and 854 (`v4ne`)
  ([r167b_prompt_depth.md](r167b_prompt_depth.md), "The host's memory per load").
- **Swap:** both `v4nk` loads and one `v4ne` load moved memory to swap during the prompts: the most in use was 891 and
  910 MiB for `v4nk` (+819 and +726 MiB during the prompts) and 47 and 672 for `v4ne` (+0 and +528), against 45 and
  52 for `v4r` (+14 and +5). The most written in 10 s was 125 and 137 MiB (`v4nk`), well under the watchdog's 600. This
  run does not show why: one `v4ne` load swapped and the other did not.
- **The cards after loading** held the same memory in every load: 43,983 MiB over the nine.
- **Each server printed the switches it was given:** `switches: 39 set` for `v4r`, the preset's 39 variables;
  `37 set` for `v4nk` and `38 set` for `v4ne`, each `v4r`'s list less exactly the variables the arm drops.
- **The control beside R157's `v4r`** ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)), the
  same preset and build on the same rig: decode over all depths 26.15 and 26.11 tok/s here against 26.15 and 26.14
  there, all chunks 212.6 and 215.3 against 218.5 and 218.8. These are different runs, so they get no class.

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R157 ran it on Flash-Next
  ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)), on one build: the sampler pass on, the
  recorded agent text of the release check's `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768 and 49,152
  tokens, Swift Q2_0 with its MTP draft.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v4r` at depth 0, then `r1 n1 e1 r2 n2 e2`, each a fresh server start after
  a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog (stop below 60 MiB
  available). Nothing else ran on the rig.
- **The lines:** the lane reads the preset file (its `PRESET_ENV` and `PRESET_ARGS`). The run script takes per arm
  the variables to drop (`ARM_DROP_<arm>`) and to add (`ARM_ADD_<arm>`). It refuses a control with changes of its own,
  an arm with none, a drop of a variable the preset does not set and an add of one it sets. The lane's `ENV_DROP` is
  the same for every arm, so the script drops the union of the arms' drops and gives each arm back, with the preset's
  values, the ones it keeps; those stand at the end of the line. A dry run printed the three lines first: `v4r`'s
  environment is the preset's as a set, `v4nk`'s is `v4r`'s less the two keep-alive variables, `v4ne`'s less the eager
  stop, and the binary, model and flags are the same in all three. [r167b_loads.tsv](r167b_loads.tsv) holds the lines
  as they ran.
- **Classes** (the lane's rule), per depth and over all depths: the spread is the larger of the two arms' differences
  between their two loads; an arm is a gain when its mean is better than the control's by more than the spread, a loss
  when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
# R167b (DRY=1 first: prints each arm's line and starts nothing)
RIG=<rig alias> WORK=<run folder on the rig> PRESET=release/flash-next-v4/settings.env \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> WD_ENV=MIN_AVAIL_MB=60 \
  ARMS="v4r v4nk v4ne" ARM_DROP_v4nk="GGML_CUDA_KEEPALIVE GGML_CUDA_KEEPALIVE_EAGER" \
  ARM_DROP_v4ne=GGML_CUDA_KEEPALIVE_EAGER LOADS="warm:v4r r1:v4r n1:v4nk e1:v4ne r2:v4r n2:v4nk e2:v4ne" \
  bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=wait bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=collect OUT=<folder> \
  bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
```

The tables (the driver writes `table.md` and `classes.tsv` at the end of the run; here they were built on the rig
after its run, with the same result):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v4r --out r167b_table.md --tsv r167b_classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control v4r --class-on pass2 --out r167b_table_pass2.md \
  --tsv r167b_classes_pass2.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4r --out r167b_prompt_depth.md \
  --tsv r167b_prompt_depth
```

## Files

| File | Content |
|---|---|
| [r167b_table.md](r167b_table.md), [r167b_classes.tsv](r167b_classes.tsv) | R167b, `deep_table.py`'s tables, control `v4r`, a load's speed: decode by depth and over all depths, the step on each pass, prompts, the draft, the sampler pass, the checks |
| [r167b_table_pass2.md](r167b_table_pass2.md), [r167b_classes_pass2.tsv](r167b_classes_pass2.tsv) | the same, decode classed on pass 2 alone |
| [r167b_prompt_depth.md](r167b_prompt_depth.md), [r167b_prompt_depth_prompt.tsv](r167b_prompt_depth_prompt.tsv), [r167b_prompt_depth_generation.tsv](r167b_prompt_depth_generation.tsv), [r167b_prompt_depth_memory.tsv](r167b_prompt_depth_memory.tsv) | R167b, `prompt_depth.py`'s tables: prompt reading per depth and over all chunks, the host's memory and each card's per load, generation as a check |
| [r167b_loads.tsv](r167b_loads.tsv), [r167b_requests.csv](r167b_requests.csv) | R167b, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [run/r167.sh](run/r167.sh) | the run, every part: start, wait, status, collect; the preset and each arm's drops and adds from the environment |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the rig build's `llama-server`
reports `eb312b7bc`, whose tree is `v4-rc2`'s). Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), with its MTP draft file; the prompts are
the deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
