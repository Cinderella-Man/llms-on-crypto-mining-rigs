# Miner Llama {{VERSION}}: llama.cpp for P106 mining rigs and other Pascal GPUs

A llama.cpp build for NVIDIA GPUs of compute capability 6.1: the P106 and P104 mining cards, the GTX 10 series and the
Tesla P4/P40. Built from commit `{{COMMIT}}` of this fork, on upstream llama.cpp b12360, for Linux on x86-64 without
AVX, so it runs on mining rigs' Celeron hosts. The tag `miner-llama-{{VERSION}}` is `eb82916d3` on the branch
`pascal-p106-v4`: the merge of that commit, with the same source tree. The binaries are the build the release tests
below ran on. This release is tuned and measured on three models on nine P106-090 cards; other compute-6.1 cards run
the same code but were not measured. Measurements, settings and the scripts behind every number:
[llms-on-crypto-mining-rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

## Speed

**Qwen3.8-Flash-Next, a 125B-parameter model, at 27.61 and 26.67 tok/s for a coding agent on nine 6 GB mining cards
and a 4 GB host, against v3.0's 20.66 and 18.92.** A task's work takes 2.37 and 2.43 minutes against 3.08 and 3.19.
With 49,152 tokens of conversation behind it, {{VERSION}} reads prompts at 180.7 and 180.8 tok/s against 70.3 and
70.6, and generates 24.26 and 24.14 tok/s against 10.66 and 10.98. Flint and Ornith-1.5-122B generate as fast as on
v3.0 and read prompts faster.

A coding agent ([pi](https://github.com/earendil-works/pi)) fixing twelve small bugs, Python and JavaScript, real test
runs, the context growing every turn, on nine P106-090 cards (one PCIe 1.0 lane each, a 2-core Celeron with 4 GB of
RAM). Each engine runs the model's own preset; every value is one run, and two values are two runs:

| Model, file | Engine | Tasks passed | Minutes a task, equal work | Generation tok/s | Prompt tok/s | First token of a turn, median s |
|---|---|---|---|---|---|---|
| **Qwen3.8-Flash-Next**, UkisAI's Swift-1.5 GSQ-RCO Q2_0, 65,536-token context | v3.0 | 11 of 12, 11 of 12 | 3.08, 3.19 | 20.66, 18.92 | 65.5, 66.3 | 6.52, 6.51 |
| | **{{VERSION}}** | **12 of 12, 12 of 12** | **2.37, 2.43** | **27.61, 26.67** | **144.6, 133.9** | **2.48, 2.42** |
| **Ornith-1.5-122B**, Ornith-1.5-122B Pascal N2P GGUF (`valid1`, 48.2 GB), 49,152-token context | v3.0 | 12 of 12, 12 of 12 | 2.52, 2.54 | 16.22, 16.49 | 61.1, 58.4 | 7.26, 8.80 |
| | **{{VERSION}}** | **12 of 12, 12 of 12** | **2.11, 2.13** | 16.52, 16.13 | **103.0, 105.9** | **6.69, 6.06** |
| **Flint**, Flint Pascal Q8 GGUF (`p065`, 40.4 GB), 262,144-token context | v3.0 | 11 of 12, 12 of 12 | 1.81, 1.82 | 48.55, 48.02 | 179.2, 181.3 | 2.65, 2.61 |
| | **{{VERSION}}** | 11 of 12, 12 of 12 | 1.72, 1.81 | 49.32, 46.99 | **219.7, 204.1** | **1.74, 1.82** |
| **Flint**, Flint Pascal Q5 GGUF (29.4 GB), 262,144-token context | v3.0 | 12 of 12 | 2.55 | 51.77 | 169.1 | 2.81 |
| | **{{VERSION}}** | 12 of 12 | 2.45 | 52.01 | **195.6** | **1.82** |

"Equal work" is one reference run's prompt tokens, generated tokens and time outside the server, at each run's own
speeds, so it compares engines on the same work. The reference differs per file, so compare within a file only. A run
that misses a task stopped it at the agent's 12-minute cap; the agent sends no seed, so each run draws its own reply
lengths.

- **Flash-Next** ([preset and tables](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v4/README.md),
  [data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-13_r156_agent_v4r/README.md)).
  Each v3.0 run lost one task to the cap, a different one each time. On the ten tasks neither v3.0 run ran away on,
  v3.0 takes 2.24 and 2.21 minutes a task and {{VERSION}} 1.52 and 1.52. One turn in ten waits longer for its first
  token than 8.65 and 9.00 s, against 16.17 and 16.63 on v3.0.
- **Ornith-1.5-122B** ([preset and tables](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v4/README.md),
  [data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-14_r160_ornith_release_agent/README.md)).
  A task takes 16.2 % less time at equal work and prompts are read 74.8 % faster. A task's first answer comes after
  8.43 and 8.45 s against 16.34 and 16.52. The host's lowest free memory is 1,225 and 1,228 MiB against 731 and 799.
- **Flint** ([preset and tables](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v4/README.md),
  [data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-14_r159_flint_release_agent/README.md)).
  Prompts are read 17.6 % faster and the first token comes 32.3 % sooner; minutes a task and generation are level
  with v3.0. A checkpoint is restored from the cards in 2 ms, where v3.0 reads it from host memory in 292 ms. Flint
  Pascal Q5 GGUF ran once on each engine the same night
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_r170_flint_q5/README.md)).
  Unsloth's Q8_0 and UD-Q5_K_XL run on the same lines and were not measured on {{VERSION}}.

Deep in a context ([the deep lane](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/bench/deep_lane/README.md):
one prompt at each of six depths from 0 to 49,152 tokens, 256 tokens generated twice, two loads per engine, each value
one load):

| Model | Measure | v3.0 | {{VERSION}} | Change |
|---|---|---|---|---:|
| Flash-Next | generation over six depths, tok/s | 16.41, 16.61 | 26.15, 26.14 | +58.4 % |
| | prompts, five chunks to 55,248 tokens, tok/s | 91.2, 91.4 | 218.5, 218.8 | +139.4 % |
| | generation at 49,152 tokens, tok/s | 10.66, 10.98 | 24.26, 24.14 | +123.6 % |
| | prompts at 49,152 tokens, tok/s | 70.3, 70.6 | 180.7, 180.8 | +156.5 % |
| Flint (Flint Pascal Q8 GGUF) | generation over six depths, tok/s | 40.24, 40.00 | 40.58, 40.57 | +1.1 % |
| | prompts, five chunks to 53,200 tokens, tok/s | 312.7, 312.0 | 329.9, 334.1 | +6.3 % |
| Ornith-1.5-122B, depths to 48,640 tokens | generation over six depths, tok/s | 12.88, 12.88 | 12.97, 12.98 | |
| | prompts, five chunks to 54,736 tokens, tok/s | 146.8, 146.7 | 159.9, 159.8 | |

Flash-Next's and Flint's two engines ran in one session each
([Flash-Next](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-13_r157_release_deep/README.md),
[Flint](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_r168_flint_f4t_deep/README.md)).
Ornith's v3.0 loads are from 2026-11-14 and its {{VERSION}} loads from 2026-11-16, so no change is given
([v3.0](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-14_r162_ornith_release_deep/README.md),
[{{VERSION}}](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_r167_keepalive_depth/README.md)).

## What is new since v3

Almost all of the work was done on Flash-Next. Every change is behind a switch that is off by default, as in v3.0, and
each model's preset names the switches its own tests ran with. The list of switches, with each one's default, what it
does and which preset sets it:
[docs/switches.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/switches.md).

**Long conversations** (Flash-Next). On v3.0 a speculative step at 49,152 tokens of context took 2.45 times as long
as at the start. Flash-Next's attention keeps only the cache cells its index picks, and the engine read them all.
- `GGML_CUDA_FATTN_SPARSE_PASCAL=1` reads only the kept cells, from a cache as large as the kept cells on
  (`_MIN_RATIO=1.0`), with launch shapes for 5-SM cards.
- The index is read in place, its tables stay on the cards and only changed rows are sent, and its mask is built on
  the cards (`LLAMA_QSA_INDEXER_ROWS`, `LLAMA_QSA_KPOOL_DEV`, `LLAMA_QSA_KPOOL_STAGE`, `LLAMA_QSA_POOL_MASK_DEV`).
- A repeated step at 49,152 tokens takes 137.42 and 137.52 ms against v3.0's 301.39 and 299.47.

**The pass that verifies drafted tokens** (Flash-Next). A draft of three tokens is checked in one pass over four.
- Kernels that read each weight once for all the tokens of a 2–4-token pass, with each token's sums the one-token
  kernel's: IQ4_XS and Q4_K (`GGML_CUDA_MMVQ_PASCAL_DMT`), Q5_K (`_Q5K_MT`), Q3_K at 2–8 tokens (`_Q3K_MT`), Q6_K
  (`_Q6K_MT`), the shared expert's gate, up and SWIGLU as one launch (`_GLU_MT`), and its Q4_0/Q5_0 downs
  (`_KCOL_SKIP=q4_0,q5_0`).
- One top-k over the output rows in the card's sampling chain (`LLAMA_SAMPLING_TOP_K_ROWS=1`).
- The Q3_K kernel alone takes 2.498 ms off a served verify pass of four tokens
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-12_r120b_kern_speed/README.md)).

**Between two steps** (Flash-Next; two of the switches are in Flint's preset too, one in Ornith's). Eight host-side switches shorten the
server's work between the draft's last pass and the next verify pass: the n-gram table's rows are asked for as soon
as a token is drafted, the verify pass is prepared while the draft still runs, the draft keeps its graph, and a cache
clean-up visits only the cells it can change. Together a step takes 111.94 ms against 119.15
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-12_r122_host_ladder/README.md)).

**Drafts accepted by probability** (Flash-Next): `--spec-draft-sampling probabilistic --spec-draft-p-min 0` accepts a
drafted token by rejection sampling, which keeps the reply's distribution and accepts more tokens than an exact
match; `LLAMA_SPEC_REJECT_STOP=1` stops at an accepted end of turn.

**Prompts** (all three models, each in its own combination):
- Context checkpoints kept on the cards (`--ctx-checkpoints-vram 4`) and taken inside a prompt's call
  (`--ctx-checkpoints-incall`), so a checkpoint no longer stops the nine-card pipeline or crosses the host.
- Prompt micro-batches captured as CUDA graphs (`GGML_CUDA_GRAPHS_FORCE=1`), and prompt calls of 4,096 tokens.
- The micro-batch taper (`--ubatch-taper short=600:64`): a prompt call of at most 600 tokens is read in micro-batches
  of 64, which the nine cards work on as a pipeline. Every agent turn ends with such a call.
- For Flash-Next's draft: a catch-up pass that writes only its caches and overlaps the main model's next call, and
  its token table on its own card.
- After a re-plan, the scheduler keeps the plan for the largest batch (`LLAMA_SCHED_PER_SIZE_RESERVE=2`).

**The keep-alive's eager stop** (`GGML_CUDA_KEEPALIVE_EAGER=1`). Under CUDA's lazy loading a kernel's first launch on
a card waited about 1 s for that card's keep-alive block to time out, at the first call of every new graph shape. With
the switch the block is ended first.

**The log names the switches.** llama-server logs one line at load with every `LLAMA_` and `GGML_` variable it was
started with, and marks any this build does not read.

**Upstream base:** llama.cpp b12360 (v3.0: b12076). What upstream changed by default on the way is listed in
[docs/switches.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/switches.md#what-upstream-changed-by-default-on-the-v4-line)
and in [the merge's data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-13_c831_upstream/README.md).

**Also in the build, off by default.** Changes that were measured as a loss on nine P106 cards, or not measured to
the end, stay behind their switches: drafts of four tokens and a draft length chosen while drafting, one shared
attention read for a verify pass's four tokens, the prompt split across the cards, the tiled kernels for wide rows,
the read-ahead of n-gram rows for prompts. Each one's numbers are in its row of the switch list.

## One preset per model

| Model | Preset | What it adds to v3.0's preset | Left out, and why |
|---|---|---|---|
| Qwen3.8-Flash-Next | [flash-next-v4](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v4/README.md) | 33 switches: everything above | longer drafts, the shared attention read, the prompt split: losses on the rig |
| Ornith-1.5-122B | [ornith-v4](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v4/README.md) | three host-side switches; staged hops, prompt graphs, calls of 4,096 tokens, checkpoints on the cards and inside a call | the taper is opt-in: it changes Ornith's numbers (below) |
| Flint | [flint-v4](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v4/README.md) | three host-side switches; prompt graphs, the scheduler's kept plan, checkpoints on the cards, the taper | the keep-alive: deep in a context it reads prompts 16.5 % slower on Flint |

Flint and Ornith decode one token a step, so the kernels for 2–8-token passes never run for them and their
generation speed is v3.0's.

## What changes the numbers, and what does not

- **Flint: the same tokens as v3.0.** With the same seed, {{VERSION}} with Flint's preset writes v3.0's tokens on 26
  recorded agent requests, 16,290 tokens, the same at every position
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_c843_flint_sampling/README.md)).
  At temperature 0 it writes v3.0's reply at every depth of the deep lane.
- **Ornith: the same replies at temperature 0** as v3.0 at every depth of the deep lane, with the preset as shipped.
- **Flash-Next: another text at temperature 0 than v3.0's.** The new kernels and upstream's changes add the same
  numbers in another order than v3.0 did, and this model turns the last bits into another text. Every kept token is
  still one the model picks or accepts itself.
- **The micro-batch taper is another order of additions**, and how far that moves a model differs. Against the same
  line without it, on 24 chunks of 512 tokens of agent sessions, a repeat of the base as the floor:

  | Model | Mean KL divergence with the taper | The base repeated | Same top token | In the preset |
  |---|---:|---:|---:|---|
  | Flint | 0.000000 | 0.000000 | 100 % | yes |
  | Flash-Next | 0.012663 | 0.000000 | 96.863 % | yes |
  | Ornith-1.5-122B | 0.047074 | 0.000062 | 93.889 % | no, opt-in |

  On Ornith that is close to the distance between the file and its makers' Q8_0 on agent text (0.051815), with the
  same perplexity (4.0750 against 4.0730); on Flash-Next it is 4.4 % of the file's own distance (0.285104)
  ([Ornith](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_fn93c_ornith_taper_kld/README.md),
  [Flash-Next](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_fn95_flashnext_taper_kld/README.md),
  [Flint](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_fn96_flint_taper_kld/README.md)).
  With the taper Ornith's first token of a turn comes after 3.05 and 3.27 s (median) instead of 6.69 and 6.06.

## Known issues

- **Flash-Next at the very start of a conversation, at temperature 0.** On the deep lane's 64-token prompt
  {{VERSION}} generates 23.50 and 23.49 tok/s against v3.0's 26.71 and 26.77. Its step is shorter (98.24 and 98.27 ms
  against 123.00 and 122.98), but it writes another greedy reply there and 0.465 of the draft is accepted on it,
  against 0.823 on v3.0's. With the makers' sampling the same request runs 24.05 and 24.07 against 16.50 and 16.56.
- **Don't add the keep-alive to Flint's line.** `GGML_CUDA_KEEPALIVE=decode` reads Flint's prompts 16.5 % slower over
  five chunks to 53,200 tokens and decodes 3.0 % slower
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_r165_flint_stack_split/README.md)).
  Flash-Next's and Ornith's presets keep it: without it they decode slower at depth.
- **`--ctx-checkpoints-incall` needs `--ctx-checkpoints-vram`,** and the server turns it off on models with
  sliding-window layers (Flint).
- **A hang on newer cards.** On three RTX 3090 cards, with the compute 6.1 code forced on, llama-server stopped
  twice inside a prompt call with its main thread spinning in the CUDA driver. It has not happened on a P106
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-16_fn93_ornith_greedy_split/README.md)).
- **Keep the host quiet.** With Flash-Next's preset the host's lowest free memory in the two agent runs was 312 and
  419 MiB of its 4 GB.
- **v3.0's known issues** were not tested again and are assumed to stand: `GGML_CUDA_KEEPALIVE=idle`, `--pipe-depth`
  on a `qwen4exp` model, and Q4_1_64 files written by v2.0
  ([v3.0's notes](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/notes-v3.0.md#known-issues)).

## Install

Requirements: Linux with glibc 2.38 or newer (tested on Ubuntu 24.04) and NVIDIA's proprietary driver 580
(`nvidia-driver-580`, not `-open`). No CUDA toolkit: the second asset carries the CUDA runtime libraries.

```bash
sha256sum -c --ignore-missing SHA256SUMS
tar xzf {{BIN_ASSET}} && tar xzf {{CUDART_ASSET}}
{{PREFIX}}-{{VERSION}}/llama-server --list-devices
```

Both archives unpack into `{{PREFIX}}-{{VERSION}}/`. `scripts/p106_check.sh` in the repository checks the driver, the
cards, host memory and the model's read speed. Each model's preset, with the files to download:
- [Qwen3.8-Flash-Next](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v4/README.md)
- [Flint](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v4/README.md)
- [Ornith-1.5-122B](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v4/README.md)

From a clone of the repository, one command starts a preset (Flint here; `MODELS` is the folder with the model file):

```bash
. llms-on-crypto-mining-rigs/release/flint-v4/settings.env && env $PRESET_ENV {{PREFIX}}-{{VERSION}}/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `{{BIN_ASSET}}` | {{BIN_MB}} MB | `{{BIN_SHA256}}` |
| `{{CUDART_ASSET}}` | {{CUDART_MB}} MB | `{{CUDART_SHA256}}` |

`BUILDINFO.txt` in the first archive records the commit, the build image, and that the binaries contain no AVX
instructions and need glibc 2.38 at most. A third archive carries `test-backend-ops`, llama.cpp's operator tests, for
anyone who wants to check the kernels on their own cards.

## Licences

The binaries are llama.cpp and ggml with this fork's changes, all under the MIT licence (`LICENSE`,
`THIRD_PARTY_NOTICES.md`). The CUDA runtime libraries in the second archive are NVIDIA's, redistributed unmodified under
the NVIDIA CUDA Toolkit End User License Agreement (`NVIDIA_CUDA_RUNTIME_NOTICE.md`). Model files are not part of this
release and keep their own licences.
