# Flint's released v4 preset at depth: v4 against v3.0 (R168, Q1495; 2026-11-16)

What this measured: how fast Flint's v4 preset as released ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env))
decodes and reads prompts deep into a context, against the v3.0 release, on the same file, rig and lane. Two arms on
nine P106-090 cards, Flint Pascal Q8 GGUF (`p065`, 40.4 GB) at the full 262,144-token context:

| Arm | Build | Settings |
|---|---|---|
| `v3` (control) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag `miner-llama-v3.0`) | v3.0's Flint preset as shipped ([release/flint-v3/settings.env](../../../release/flint-v3/settings.env)), no switch |
| `v4` | `v4-rc2`, `pascal-p106-v4` `eb82916d3` (upstream b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is `v4-rc2`'s) | the v4 preset as released: v3.0's line, the three host switches (`LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1`), forced prompt graphs (`GGML_CUDA_GRAPHS_FORCE=1`), the scheduler's per-size reserve (`LLAMA_SCHED_PER_SIZE_RESERVE=2`), and `--ctx-checkpoints-vram 4 --ubatch-taper short=600:64` |

Each switch the preset adds to the three host switches was no loss at depth alone
([R165](../2026-11-16_r165_flint_stack_split/README.md)); this run times them together. The earlier release table,
[R161](../2026-11-14_r161_flint_release_deep/README.md), timed the three host switches alone against v3.0. Registry row
Q1495 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**The released v4 preset is faster than v3.0 on Flint at every depth from 4,096 tokens, and writes the same replies.**
- **Decode** over all six depths: 40.58 and 40.57 tok/s in `v4`'s two loads, against v3.0's 40.24 and 40.00 (+1.1 %,
  a gain). It is a gain at every depth from 4,096 tokens and negligible at depth 0.
- **Prompts:** all five prompt chunks of a load (53,200 tokens) read at 329.9 and 334.1 tok/s against 312.7 and 312.0
  (+6.3 %, a gain), 161.3 and 159.2 s against 170.1 and 170.5 s. Every chunk is a gain, the largest at 8,192 tokens
  (+20.8 %).
- **At 49,152 tokens:** decode 23.57 and 23.58 tok/s against 23.43 and 23.43 (+0.6 %), prompts 221.2 and 222.0 against
  213.3 and 214.1 (+3.7 %).

**Beside R161, which timed the three host switches alone** (the same lane, file and builds, 2026-11-14): there `v4`
decoded 40.25 and 40.36 tok/s against v3.0's 40.02 and 40.04 (+0.7 %) and read all primes at 318.9 and 321.3 against
309.9 and 311.0 (+3.1 %). Against its own run's v3.0, the released preset doubles the prompt gain at depth (+6.3 %
against +3.1 %) and decodes a little more (+1.1 % against +0.7 %). The two runs are different loads on different days,
so they are set side by side and not classed against each other.

Every value is one load. Run on one rig on 2026-11-16, 18:58:12–20:07:43: a warm-up load of `v3`, then
`v31 v41 v32 v42`, every load through, the RAM watchdog silent.

### Per depth

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. Prompts: the
prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on an empty cache; the
chunk read and the tokens cached are the same in both arms). `v3`'s loads are listed `v31`, `v32`; `v4`'s `v41`,
`v42`.

| Depth (tokens) | Decode tok/s, `v3` | `v4` | Class |
|---:|---|---|---|
| 0 | 50.68, 48.26 | 50.75, 50.34 | negligible (+2.2 %) |
| 4,096 | 76.66, 76.63 | 77.67, 77.61 | gain (+1.3 %) |
| 8,192 | 48.44, 48.58 | 49.11, 49.43 | gain (+1.6 %) |
| 16,384 | 66.88, 66.96 | 67.61, 67.61 | gain (+1.0 %) |
| 32,768 | 26.28, 26.28 | 26.52, 26.51 | gain (+0.9 %) |
| 49,152 | 23.43, 23.43 | 23.57, 23.58 | gain (+0.6 %) |
| all | 40.24, 40.00 | 40.58, 40.57 | gain (+1.1 %) |

At depth 0, `v32`'s first pass ran at 43.05 tok/s against 47.10 in `v31`, which gives v3.0 a spread of 2.42 tok/s
there, so `v4`'s +2.2 % is within it.

| Depth (tokens) | Read, on cached | Prompt tok/s, `v3` | `v4` | Class |
|---:|---|---|---|---|
| 0 | 64 on 0 | 86.4, 83.8 | 91.1, 88.3 | gain (+5.4 %) |
| 4,096 | 4,032 on 0 | 479.7, 479.9 | 549.3, 566.8 | gain (+16.3 %) |
| 8,192 | 4,100 on 4,028 | 450.0, 432.1 | 524.7, 540.5 | gain (+20.8 %) |
| 16,384 | 12,292 on 4,028 | 477.2, 475.7 | 520.4, 539.0 | gain (+11.2 %) |
| 32,768 | 16,388 on 16,316 | 327.6, 326.0 | 338.6, 342.2 | gain (+4.2 %) |
| 49,152 | 16,388 on 32,700 | 213.3, 214.1 | 221.2, 222.0 | gain (+3.7 %) |
| all primes | 53,200 | 312.7, 312.0 | 329.9, 334.1 | gain (+6.3 %) |

Decode depends on the reply as well as on the depth: Flint writes another reply at each depth (the same one in both
arms), and its pipelined speculation (`--pipe-depth 4`) pays off more on some replies than on others, so 4,096 tokens
decode faster than depth 0 and 8,192 slower than 16,384; between arms at one depth the reply is the same. The sampled
request (the line's own sampler, fixed seed) at 49,152 tokens runs 22.71 and 22.70 tok/s for `v4` against 22.54 and
22.50 for `v3` ([table.md](table.md), "The sampler pass").

### The replies, the host and the cards

- **Both arms write the same greedy reply on every pass and load at every depth.** Every timed request gave 256
  tokens, and every load read the same prompt ids.
- **No target graph reallocated** on any of the 24 requests of any load.
- **The servers' switches:** `v4`'s two servers printed `switches: 5 set` (the five variables of its
  `PRESET_ENV`); v3.0's build prints no such line, and its environment held none.
- **Checkpoint restores:** `v4` restored each 52.5 MiB checkpoint from the cards in 2.1, 2.1, 2.2 and 2.1 ms (both
  loads the same), and the 4.1 MiB one of the 60-token tail in 1.2 and 1.1 ms (`v41`) and 1.2 and 1.2 ms (`v42`); v3.0 read them from host memory in
  291.9, 292.0, 294.8 and 291.9 ms (`v31`) and 291.9, 291.9, 292.0 and 291.9 ms (`v32`), and the tail's in 23.2 and
  23.2 ms (`v31`) and 23.3 and 23.2 ms (`v32`).
- **The host's lowest MemAvailable:** `v31` 940 MiB, `v32` 952, `v41` 1,149, `v42` 1,130: the preset keeps about
  190 MiB more free at its lowest, its four checkpoints living on the cards. MemAvailable after loading was 1,435,
  1,432, 1,420 and 1,427 MiB. Swap in use peaked at 293, 294, 294 and 295 MiB, and nothing was swapped while a
  prime ran. The RAM watchdog did not act ([prompt_depth.md](prompt_depth.md)).
- **The cards after loading** held the same memory in every load: 48,613 MiB over the nine, cards 1 and 4 at 5,923
  MiB. **Over a load** they peaked at 5,953 MiB on cards 1 and 4 in both `v3` loads and at 5,987 MiB in both `v4`
  loads (157 MiB free of 6,144); the nine cards' peaks summed to 48,907 MiB for `v3` and 49,165 for `v4`
  ([card_peaks.tsv](card_peaks.tsv)).

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R161 ran it: two builds in one run
  (`BIN` is `v4-rc2`, `ARM_BIN_v3` the v3.0 build), the sampler pass on, the recorded agent text of the release
  check's `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768 and 49,152 tokens.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v3` at depth 0, then `v31 v41 v32 v42`, each a fresh server start after a
  quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog with its default limits.
  Nothing else ran on the rig.
- **The lines:** each arm's is its own settings file's, word for word. The lane takes one preset file (`v3`'s) and
  changes it per arm, so [run/r168.sh](run/r168.sh) reads both files, gives `v4` its file's `PRESET_ENV` and the flags
  its `PRESET_ARGS` adds after v3.0's line, and stops on any other difference. A dry run printed both lines first;
  `loads.tsv` holds them as they ran.
- **Classes** (the lane's rule), per depth and over all depths, on a load's speed (both decode passes' tokens over
  their time; a prime's tokens over its time): the spread is the larger of the two arms' differences between their two
  loads; `v4` is a gain when its mean is better than `v3`'s by more than the spread, a loss when worse by more than it,
  else negligible.
- **The cards' peaks** per load: the highest memory of each card in the lane's GPU monitor (every card every 10 s)
  between a load's start and end ([run/card_peaks.sh](run/card_peaks.sh)).

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
RIG=<rig alias> WORK=<run folder on the rig> \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server REF_BIN=<v3.0 tree on the rig>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  DRY=1 bash docs/data/2026-11-16_r168_flint_f4t_deep/run/r168.sh
# then the same without DRY=1, then STEP=wait, then STEP=collect OUT=<folder>
```

The tables, built on the rig's own CPU after the run (any machine with Python 3 builds the same):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v3 --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v3 --out prompt_depth.md \
  --tsv prompt_depth
bash docs/data/2026-11-16_r168_flint_f4t_deep/run/card_peaks.sh <folder> > card_peaks.tsv
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | `deep_table.py`'s tables, control `v3`: decode by depth and over all depths, the step, prompts, the sampler pass, the checks (replies, reallocations) |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | `prompt_depth.py`'s tables, control `v3`: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [card_peaks.tsv](card_peaks.tsv) | each card's highest memory per load |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran; per request its counts, times and reply md5s |
| [run/r168.sh](run/r168.sh), [run/card_peaks.sh](run/card_peaks.sh) | the run (start, wait, status, collect) and the cards' peaks |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)); the prompts are the deep
lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and `requests.csv` hold
times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
