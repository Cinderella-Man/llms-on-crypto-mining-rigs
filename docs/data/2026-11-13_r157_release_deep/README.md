# Flash-Next's release deep lane: v4's final settings against v3.0 at depth (R157, Q1475; 2026-11-13)

What this measured: how fast Miner Llama v4's release candidate decodes and reads prompts deep into a context, with
Flash-Next's final v4 settings, against the v3.0 release, on the same model, files and rig. Two arms on nine
P106-090 cards, Qwen3.8-Flash-Next as Swift Q2_0 with its MTP draft of three tokens, a 65,536-token context:

| Arm | Build | Settings |
|---|---|---|
| `v3` (control) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag `miner-llama-v3.0`) | v3.0's release preset as shipped ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)) |
| `v4r` | `v4-rc2`, `pascal-p106-v4` `eb82916d3` (upstream b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is `v4-rc2`'s) | v4's final Flash-Next settings, as the agent check runs them ([2026-11-13_r156_agent_v4r](../2026-11-13_r156_agent_v4r/README.md), `run/r156.sh`'s `v4r`) |

Registry row Q1475 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**v4 reads prompts much faster than v3.0 at every depth, and decodes much faster at every depth with context.**
All five prompt chunks of a load (55,248 tokens) read at 218.5 and 218.8 tok/s in `v4r`'s two loads against v3.0's
91.2 and 91.4 (+139.4 %, a gain): 252.8 and 252.6 s against 605.6 and 604.3 s. Decode over all six depths runs 26.15
and 26.14 tok/s against 16.41 and 16.61 (+58.4 %, a gain). The deeper the context, the larger the gap: at 49,152
tokens `v4r` reads 180.7 and 180.8 tok/s against 70.3 and 70.6 (+156.5 %), and decodes 24.26 and 24.14 tok/s against
10.66 and 10.98 (+123.6 %), its repeated step taking 137.42 and 137.52 ms against 301.39 and 299.47 (−54.2 %).

**One loss: greedy decode at depth 0,** 23.50 and 23.49 tok/s against 26.71 and 26.77 (−12.1 %). The step is a gain
there too (98.24 and 98.27 ms against 123.00 and 122.98, −20.1 %). The loss comes from the reply: on the 64-token
prompt `v4r` writes another greedy reply than v3.0, and accepts 148 of 318 draft tokens on it (0.465) against v3.0's
181 of 220 (0.823). The sampled request at depth 0 (the server's own sampler, as an agent uses it) runs 24.05 and 24.07
tok/s against 16.50 and 16.56.

**The release table's deep-lane line:** *Deep into a context (Flash-Next, Swift Q2_0 with its MTP draft, nine P106
cards, two loads each), v4 reads prompts at 218.5 and 218.8 tok/s against v3.0's 91.2 and 91.4 (+139.4 %) and decodes
at 26.15 and 26.14 tok/s against 16.41 and 16.61 (+58.4 %) over six depths from 0 to 49,152 tokens; at 49,152 tokens
it reads 180.7 and 180.8 tok/s against 70.3 and 70.6 (+156.5 %) and decodes 24.26 and 24.14 against 10.66 and 10.98
(+123.6 %).*

Every value is one load. Run on one rig on 2026-11-13, 19:04:50–20:16:29: a warm-up load of `v3`, then
`v31 v4r1 v32 v4r2`; the RAM watchdog stopped `v31`'s first start, which ran again last (below).

### Per depth

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. The step: one
pass of the main model (its own token plus the draft tokens it accepts), on pass 2, the same request repeated on the
same cache. Prompts: the prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on
an empty cache; the chunk read and the tokens cached are the same for both arms). `v3`'s loads are listed `v31`, `v32`;
`v4r`'s `v4r1`, `v4r2`. Classes against `v3`.

| Depth (tokens) | Decode tok/s, `v3` | `v4r` | Class | Step on pass 2, ms, `v3` | `v4r` | Class | Prompt tok/s, `v3` | `v4r` | Class |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 26.71, 26.77 | 23.50, 23.49 | loss (−12.1 %) | 123.00, 122.98 | 98.24, 98.27 | gain (−20.1 %) | 31.9, 32.2 | 51.8, 52.4 | gain (+62.7 %) |
| 4,096 | 18.81, 18.73 | 28.84, 29.01 | gain (+54.1 %) | 136.81, 137.45 | 109.56, 109.49 | gain (−20.1 %) | 123.7, 123.6 | 198.6, 195.9 | gain (+59.6 %) |
| 8,192 | 24.89, 22.20 | 30.81, 30.80 | gain (+30.8 %) | 152.40, 151.51 | 114.14, 114.01 | gain (−24.9 %) | 119.1, 119.0 | 219.9, 223.0 | gain (+86.0 %) |
| 16,384 | 18.85, 18.60 | 30.89, 30.88 | gain (+64.9 %) | 181.43, 182.51 | 118.11, 118.20 | gain (−35.1 %) | 119.5, 118.5 | 289.3, 290.7 | gain (+143.7 %) |
| 32,768 | 11.35, 12.39 | 21.66, 21.63 | gain (+82.4 %) | 242.15, 241.91 | 127.49, 127.48 | gain (−47.3 %) | 90.4, 91.1 | 229.1, 229.1 | gain (+152.4 %) |
| 49,152 | 10.66, 10.98 | 24.26, 24.14 | gain (+123.6 %) | 301.39, 299.47 | 137.42, 137.52 | gain (−54.2 %) | 70.3, 70.6 | 180.7, 180.8 | gain (+156.5 %) |
| all | 16.41, 16.61 | 26.15, 26.14 | gain (+58.4 %) | | | | 91.2, 91.4 | 218.5, 218.8 | gain (+139.4 %) |

The step on pass 1 (the first request at a new depth) is a gain at every depth as well, from 102.55 and 102.59 ms
against 127.50 and 127.00 at depth 0 (−19.4 %) to 140.30 and 141.52 against 306.32 and 330.46 at 49,152 (−55.7 %);
decode on pass 2 alone is a loss at depth 0 (−13.1 %) and a gain at every other depth, +51.2 % over all
([table.md](table.md)). The prompt row "all" is the five primes (55,248 tokens).

### The draft and the replies

- **Each `v4r` load writes the same greedy reply as its other load at every depth, and accepts the same draft
  tokens.** `v3`'s two loads write one reply at depth 0 and their own replies from 4,096 tokens on, as on every earlier
  deep-lane run of v3.0 (R90, registry row Q1175). At depth 0 `v4r`'s reply differs from v3.0's (the known effect of
  C722's and C690's switches: their 2–4-token sums are one-token decoding's, so a near-tie in the logits flips, R90).
- **The sampled request** (the server line's sampler, fixed seed) runs 24.05 and 24.07 tok/s for `v4r` against 16.50
  and 16.56 at depth 0, and 23.72 and 23.73 against 11.01 and 10.37 at 49,152 ([table.md](table.md), "The sampler
  pass").

### The host and the cards

- **The RAM watchdog stopped `v31`'s first start** after the prime at 4,096 tokens: MemAvailable 55 MiB, below its
  60 MiB floor. The driver set it aside and ran it again after the others, and that run went through. The lowest
  MemAvailable of the counted loads: `v31` 88 MiB, `v32` 76, `v4r1` 339, `v4r2` 289 ([prompt_depth.md](prompt_depth.md)).
  v3.0 keeps its four checkpoints in host memory, and `v4r` keeps them on the cards, as R90 found.
- **On the cards after loading** `v4r` holds 43,983 MiB over the nine against `v3`'s 44,019: card 4 4,175 against
  4,875 and card 8 5,545 against 4,565 (`v4r` does not place the draft's token table on card 4).
- No target graph reallocated on any request, every timed request gave 256 tokens, and every load read the same
  prompt ids.

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R90 ran it
  ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md)): two builds in one run (`BIN` is `v4-rc2`,
  `ARM_BIN_v3` the v3.0 build), the sampler pass on, the recorded agent text of the release check's `t1swift`
  sessions; with R88's six depths ([2026-11-13_r88_union_deep](../2026-11-13_r88_union_deep/README.md)): 0, 4,096,
  8,192, 16,384, 32,768 and 49,152 tokens.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v3` at depth 0, then `v31 v4r1 v32 v4r2`, each a fresh server start after a
  quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog (stop below 60 MiB
  available). Nothing else ran on the rig.
- **`v4r`'s line:** the preset with R156's environment (`GAINS_ENV`, `V4C2_ENV`, `V4R_ENV`) and its server line
  `MT_V4R`'s differences: `-b 4096`, no `-otd token_embd=CUDA4` (`LLAMA_DRAFT_EMBD_DEV=layer` places the draft's token
  table), `--spec-draft-n-ubatch 512`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0`,
  `--ctx-checkpoints-vram 4`, `--ctx-checkpoints-incall` and `--ubatch-taper short=600:64`. The lane's `ARGS_DROP` is
  one for all arms, so it drops `-otd` everywhere and `v3`'s `ARM_ARGS` puts the preset's `-otd token_embd=CUDA4`
  back: `v3`'s line holds exactly the preset's flags. A dry run printed both lines first; `loads.tsv` holds them as
  they ran.
- **Classes** against `v3`, per depth and over all depths (the lane's rule): the spread is the larger of the two
  arms' differences between their two loads; `v4r` is a gain when its mean is better than `v3`'s by more than the
  spread, a loss when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
RIG=<rig alias> WORK=<run folder on the rig> \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server REF_BIN=<v3.0 tree on the rig>/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 bash docs/data/2026-11-13_r157_release_deep/run/r157.sh
# then the same without DRY=1, then STEP=wait, then STEP=collect OUT=<folder>
```

The tables, on any machine with Python 3 (here a build machine's CPU at `nice -n 19`):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v3 --class-on pass2 --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control v3 --out table_load.md --tsv classes_load.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v3 --out prompt_depth.md \
  --tsv prompt_depth
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | `deep_table.py`'s tables, control `v3`, the load's speed and the class on pass 2: decode, the step on each pass, prompts, the draft, the sampler pass, the checks |
| [table_load.md](table_load.md), [classes_load.tsv](classes_load.tsv) | the same with a load's speed both passes' tokens over their time (the decode column above) |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | `prompt_depth.py`'s tables: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran (`v31` listed twice: its first start was stopped); per request its counts, times and reply md5s |
| [run/r157.sh](run/r157.sh) | the run: start, wait, status, collect |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), with its MTP draft file; the prompts are
the deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
