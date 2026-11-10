# The v4 preset candidate against the v3 release preset, decode and prompts at depth (R70, Q1254; 2026-11-10)

What this measured: what five switches that each timed as a gain on their own do together, in generation and in
prompt reading, at context depths 0, 8,192, 16,384, 32,768 and 49,152 tokens. The line is the Miner Llama v3 release
preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift
Q2_0, the MTP draft of three tokens, a 65,536-token context) on nine P106-090 cards, build
`pascal-p106-v3-b12076-ladder5` `1551f5fe1`, timed with [bench/deep_lane](../../../bench/deep_lane/README.md). Registry
row Q1254 in [quants/experiments.md](../../../quants/experiments.md).

| Arm | What it is |
|---|---|
| `rel` (control) | the release preset as shipped |
| `v4c` | the preset plus the five switches below |

| Switch | Its own result |
|---|---|
| `GGML_CUDA_FATTN_SPARSE_PASCAL=1` with `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` (the P106-090's SM count) | the sparse Pascal attention: a served step of 180.77 against 301.90 ms at 49,152 tokens ([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md), Q230) |
| `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` | its threshold for verify passes: −7.05 ms a step at 8,192 tokens (same page, Q1212) |
| `--spec-draft-sampling probabilistic --spec-draft-p-min 0` | probabilistic verification: +2.97 % generation on the release session's requests ([2026-11-10_prob_verify_rig](../2026-11-10_prob_verify_rig/README.md), Q731) |
| `--ctx-checkpoints-vram 4` and `--ubatch-taper on` | the newest four checkpoints kept on the cards, and the prompt taper: −29.75 % prompt time on the agent check's prompts ([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md), Q1155) |
| `--spec-draft-n-ubatch 512` (the preset: 128) | the draft's prompt micro-batch: +3.7 % prompt reading ([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md), Q1215) |

Three of the five were timed only on prompts, and the five never together. These are single-request diagnostics at
a known depth; they do not replace the agent check, which decides what goes into a preset.

## Result

**The five switches together are a gain in decode from 16,384 tokens on and in prompt reading from 16,384 tokens on,
and at no depth a loss in either.** Over all five depths the greedy decode runs 20.66 and 20.10 tok/s in the
candidate's two loads against the preset's 14.78 and 15.85 (+33.08 %), and the four prompt chunks read at 101.4 and
101.9 tok/s against 88.7 and 88.7 (+14.55 %). At 49,152 tokens decode is 19.08 and 18.34 tok/s against 10.40 and
10.70 (+77.34 %), a step on the repeated request 182.21 and 181.66 ms against 303.51 and 302.47. At depth 0 decode is
negligible (26.63 and 26.64 against 26.91 and 25.29), and at 8,192 negligible in tok/s, where the preset's two loads
wrote different replies (15.37 and 23.66), and a gain in the step's time (143.85 and 144.48 ms against 153.86 and
152.76). A load's 19 timed requests took 711.0 and 711.8 s against 873.7 and 864.7 s (−18.2 %). Nothing measured
got slower. Worse: the draft's acceptance on the one sampled request at depth 0, the host's lowest available memory
in one load (106 MiB), and up to 180 MiB more on a card (all below). The agent check of the same two lines decides the preset (registry row Q1261).

Every value is one load. Run on rig 1 on 2026-11-10 from 20:03:36 to 21:23:24 (80 minutes): a warm-up load and
four timed loads, each started after a quiet check; the preset's loads took 19.8 and 19.7 minutes, the candidate's
17.1 and 17.1.

### Decode and prompts by depth

Decode: 256 greedy tokens twice per depth (pass 1, the first request at a new depth; pass 2, the same request again),
a load's speed being both passes' tokens over their time. The step: generation time over steps, on pass 2. Prompts:
the prime's chunk, read on top of the previous depth's checkpoint (at depth 0, the 64-token tail on an empty cache).

| Depth (tokens) | Decode tok/s, `rel` | Decode tok/s, `v4c` | Class | Step on pass 2, ms, `rel` | `v4c` | Class | Prompt tok/s, `rel` | `v4c` | Class |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 26.91, 25.29 | 26.63, 26.64 | negligible (+2.06 %) | 123.31, 123.74 | 123.68, 123.70 | negligible (+0.13 %) | 32.9, 31.2 | 48.1, 48.3 | gain (+50.37 %) |
| 8,192 | 15.37, 23.66 | 22.16, 20.62 | negligible (+9.59 %) | 153.86, 152.76 | 143.85, 144.48 | gain (−5.96 %) | 128.5, 130.1 | 127.9, 130.0 | negligible (−0.29 %) |
| 16,384 | 17.75, 19.27 | 21.32, 23.99 | gain (+22.38 %) | 183.08, 183.34 | 150.66, 151.67 | gain (−17.49 %) | 109.6, 110.0 | 123.9, 128.6 | gain (+15.01 %) |
| 32,768 | 11.96, 11.32 | 16.66, 15.03 | gain (+36.13 %) | 245.10, 244.01 | 165.51, 166.02 | gain (−32.22 %) | 90.8, 90.5 | 103.6, 103.7 | gain (+14.36 %) |
| 49,152 | 10.40, 10.70 | 19.08, 18.34 | gain (+77.34 %) | 303.51, 302.47 | 182.21, 181.66 | gain (−39.95 %) | 69.9, 69.8 | 83.5, 83.1 | gain (+19.17 %) |
| all | 14.78, 15.85 | 20.66, 20.10 | gain (+33.08 %) | 203.33, 204.89 | 153.62, 154.36 | gain (−24.56 %) | 88.7, 88.7 | 101.4, 101.9 | gain (+14.55 %) |

Ms a token (decode, per load): 37.16 and 39.54 against 37.55 and 37.54 at depth 0; 65.05 and 42.26 against 45.13 and
48.50 at 8,192; 56.33 and 51.88 against 46.90 and 41.68 at 16,384; 83.60 and 88.32 against 60.01 and 66.51 at
32,768; 96.11 and 93.48 against 52.40 and 54.53 at 49,152. The prompt row "all" is the four primes (50,124 tokens).

### The draft, greedy and sampled

| Depth (tokens) | Greedy share accepted, `rel` | `v4c` | Greedy tokens a step, `rel` | `v4c` | Sampler pass tok/s, `rel` | `v4c` | Class | Sampler share accepted, `rel` | `v4c` | Class |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7996, 0.7996 | 0.7996, 0.7996 | 3.346, 3.346 | 3.346, 3.346 | 17.04, 15.22 | 15.22, 15.27 | negligible (−5.49 %) | 0.4506, 0.4506 | 0.3943, 0.3943 | loss (−12.50 %) |
| 8,192 | 0.6779, 0.9643 | 0.8555, 0.7851 | 2.994, 3.821 | 3.531, 3.325 | 17.34, 17.83 | 19.89, 18.87 | gain (+10.23 %) | 0.6693, 0.6326 | 0.6840, 0.7149 | gain (+7.45 %) |
| 16,384 | 0.8635, 0.8638 | 0.8265, 0.9029 | 3.531, 3.556 | 3.413, 3.657 | 16.49, 18.98 | 24.11, 21.93 | gain (+29.81 %) | 0.7033, 0.8472 | 0.9118, 0.8036 | negligible (+10.63 %) |
| 32,768 | 0.6686, 0.6104 | 0.6173, 0.5245 | 2.960, 2.798 | 2.798, 2.547 | 11.27, 10.11 | 15.12, 16.22 | gain (+46.63 %) | 0.6074, 0.5168 | 0.5448, 0.6029 | negligible (+2.10 %) |
| 49,152 | 0.7358, 0.7701 | 0.8472, 0.8036 | 3.180, 3.261 | 3.507, 3.368 | 9.65, 9.78 | 17.84, 18.56 | gain (+87.37 %) | 0.6627, 0.6786 | 0.7807, 0.8227 | gain (+19.54 %) |
| all | 0.7442, 0.7891 | 0.7817, 0.7464 | 3.188, 3.320 | 3.295, 3.200 | 13.54, 13.31 | 17.88, 17.88 | gain (+33.22 %) | 0.6093, 0.6064 | 0.6286, 0.6391 | gain (+4.28 %) |

- **Probabilistic verification acts where it should.** In the greedy passes the candidate's share accepted is
  negligible against the preset's over all depths (−0.34 %), as the method says it must be; in the sampler pass it is
  a gain (0.6286 and 0.6391 against 0.6093 and 0.6064, +4.28 %; tokens a step 2.851 and 2.883 against 2.795 and
  2.789, +2.69 %), the size R38 measured on the release session's 66 requests (+5.36 % share, +4.04 % tokens a step).
  Its step on the sampled request at 49,152 is 183.96 and 183.90 ms against 304.92 and 308.03.
- **The sampled request at depth 0 accepted less:** 138 of 350 drafted tokens in both candidate loads against 146 of
  324 in both preset loads (2.169 tokens a step against 2.327); each arm wrote one reply in its two loads, so the
  spread is 0 and the rule calls it a loss. Its tok/s is negligible (15.22 and 15.27 against 17.04 and 15.22). It is
  one request with one seed: the draw went worse for this reply, which R38's 66 requests do not show on average.

### The host and the cards

| Load | Lowest MemAvailable, MiB | Most swap in use, MiB | Highest psi full, % | Fullest card at its highest, MiB | Card 8 at its highest, MiB |
|---|---:|---:|---:|---|---:|
| `rel1` | 172 | 997 | 4.50 | card 3, 5,155 of 6,144 | 4,627 |
| `rel2` | 395 | 802 | 21.93 | card 3, 5,155 | 4,627 |
| `v4c1` | 106 | 800 | 2.65 | card 3, 5,211 | 4,807 |
| `v4c2` | 211 | 1,034 | 5.77 | card 3, 5,211 | 4,807 |

- The RAM watchdog (floor 60 MiB) never acted, and no server log holds an error line. The candidate's lowest point
  is lower in one load (106 MiB) and the preset's spread is wide (172 and 395), so the host's margin is not settled
  by two loads an arm.
- On the cards the candidate holds 40 to 180 MiB more at its highest: card 8 +180 (the draft's micro-batch of 512,
  +146 there after loading, as in R53), cards 0 to 7 +40 to +64, 470 MiB together (est. the four checkpoints on the
  cards, 112.57 MiB each by the server's lines).
- **The checkpoints on the cards:** each load restored 13 checkpoints. With the preset's host store a restore took
  622.2 ms (the median of 26; one 650.7); on the cards 2.3 and 2.4 ms. The depth-0 prompt's 1.95 s against 1.33 s
  (the 64-token tail on an empty cache) is est. the checkpoint the preset copies to the host there (R43: 641.5 ms a
  copy, median).
- **The fresh 8,128-token chunk** (the prime at 8,192, the only one read on an empty cache) is the one prompt that
  does not gain: 127.9 and 130.0 tok/s against 128.5 and 130.1. Est.: the draft's micro-batch of 512 gains there
  (R53: +5.2 %, negligible by its spread) and the taper loses on a fresh long prompt while checkpoints are on (R43,
  on 2,049 tokens: registry row Q1235), and the two cancel. The three chunks read on a checkpoint gain +15.01, +14.36
  and +19.17 %.

### Checks

- Every load started on a rig checked quiet (no llama process, 0 MiB on every card); every request wrote its 256
  tokens; the prompts' ids are the same on every load and pass.
- The greedy reply at depth 0 is the same on every pass and load of both arms. From 8,192 tokens on each load writes
  its own reply in both arms (the preset's `rel1` two at 32,768), as on every earlier deep-lane run (registry row
  Q1175), so no reply is compared between the arms there.
- The step on pass 1 at depth 0: 127.54 and 127.49 ms against 125.38 and 140.58, negligible. The preset's load right
  after the warm-up was the fastest, as in R50 and R39a (registry row Q1253).
- The primes read the same chunks in both arms: 8,128 on 0 cached, 8,708 on 7,612, 16,388 on 16,316 and 16,900 on
  32,188, each past the first continued from a checkpoint.
- Both server lines are in `loads.tsv`; the taper's line in the candidate's server log reads
  `micro-batch taper of prompt calls: head=256:64,tail=512:128,short=600:64 (n_ubatch 512)`.

## Method

- **Loads:** one untimed warm-up load of `rel`, then `rel1 v4c1 rel2 v4c2`, each a fresh server start, every one
  after a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog (stop below 60 MiB
  available). Nothing else ran on the rig during the loads.
- **The lines:** `v4c` is `rel` with the variables added and the flags set through the driver's `ARM_ENV_v4c` and
  `ARM_ARGS_v4c`, which replace a flag the preset holds instead of adding it twice: `--spec-draft-n-ubatch 128`
  became 512, and `--ctx-checkpoints 4` stays once, with `--ctx-checkpoints-vram 4` beside it (all four checkpoints
  on the cards, R43's line). A dry run (`DRY=1`) printed both lines before the run; `loads.tsv` holds them as they ran.
- **The requests** of a load, depths ascending, on the release check's recorded `t1swift` sessions (12): the prime
  (the prompt without its 64-token tail, read on top of the previous depth's checkpoint: the prompt reading), then
  256 tokens at temperature 0 twice (decode, pass 1 and pass 2), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed (`SAMPLER_PASS=1`, the sampler pass). End of
  text is ignored.
- **Why the sampler pass:** probabilistic verification accepts a draft token with the probability the main model
  gives it over the draft's. At temperature 0 both distributions are all on one token, so the rule accepts exactly
  the tokens greedy verification accepts, and the server takes the rejection path only for a request whose
  temperature is above 0 (`use_spec_rejection()` in the fork's `tools/server/server-context.cpp`). In the greedy
  passes the switch therefore does not act; the sampler pass is where it does.
- **Classes,** per depth and over all depths (the driver's rule): the spread is the larger of the two arms'
  differences between their two loads; `v4c` is a gain when its mean is better than `rel`'s by more than the
  spread, a loss when worse by more than it, else negligible.

## Commands

From a machine that only runs ssh, at the repository's root. `RIG` is the rig's ssh alias; `BUILD` and `MODELS` are
folders on the rig (the release build and the model files), `RECORDED` the release check's recording here, `OUT`
the results folder here.

```bash
RIG=rig BIN=$BUILD/bin/llama-server \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=$RECORDED/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 \
  ARMS="rel v4c" \
  ARM_ENV_v4c="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5" \
  ARM_ARGS_v4c="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512" \
  bash bench/deep_lane/remote.sh start    # with DRY=1 in front: both server lines printed, nothing started
RIG=rig bash bench/deep_lane/remote.sh wait
RIG=rig OUT=$OUT bash bench/deep_lane/remote.sh collect
```

The driver writes `table.md` and `classes.tsv` itself ([deep_table.py](../../../bench/deep_lane/deep_table.py)).
The other tables, on any machine with Python 3.9 or newer:

```bash
python3 -I docs/data/2026-11-10_v4_candidate/run/v4c_table.py $OUT --md summary.md --tsv per_load.tsv --classes v4c_classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py $OUT > host_memory.tsv
```

The cards' highest memory per load is read from the driver's `gpu.csv` (one line per card every 10 s) between the
loads' start times in `loads.tsv`; the checkpoint restores from the server logs' `restored context checkpoint` lines.

## Files

| File | Content |
|---|---|
| [summary.md](summary.md) | per depth and load every number above, then every metric's class per depth and over all depths, and the host's memory per load ([run/v4c_table.py](run/v4c_table.py)) |
| [v4c_classes.tsv](v4c_classes.tsv) | the classes as a table: metric, depth, each load's value, spread, difference, percent, class |
| [per_load.tsv](per_load.tsv) | every metric per depth and load, one line each, and the host's memory per load |
| [table.md](table.md), [classes.tsv](classes.tsv) | the driver's own tables: generation, the step on pass 1 and on pass 2, prompt reading, the draft's counters, the sampler pass, the checks |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and per request: lowest MemAvailable and usable memory, swap in use and its change, pressure |
| [requests.csv](requests.csv) | one line per request of every load (the warm-up's included): tokens read, cached and generated, times, the draft's counters, reply hashes |
| [loads.tsv](loads.tsv) | one line per load: arm, timed or warm-up, start, seconds to load, the server line as it ran |
| [run/v4c_table.py](run/v4c_table.py) | the summary's script, standard library only |

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The files here hold token counts, times, the server's counters and hashes;
no prompt or reply text. The scripts are under this repository's licence.
