# Ornith's greedy split on newer cards: can an RTX 3090 stand for the P106? (FN93 part 1, Q1487; 2026-11-16)

What this checked: on nine P106-090 cards, Ornith's candidate stack (`v4s`) writes another greedy reply than the v4
preset (`v4`) at 8,192, 16,384 and 32,768 tokens of context
([2026-11-14_r162_ornith_release_deep](../2026-11-14_r162_ornith_release_deep/README.md), "The sampler pass, the step
and the checks"). Every candidate in the stack is meant to leave the numbers unchanged, so one of them does not. Before
screening the candidates one by one, this run asks whether a machine with newer cards can do that screen: does it write
the rig's replies, and do `v4` and `v4s` part there as they do on the rig? A correctness run: no speed is read.

Registry row Q1487 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**A 3090 cannot stand for the P106's greedy replies.** With the same build, the same model weights, the same prompt
ids at every depth and the fork's Pascal kernels forced on with the P106's launch shapes, three RTX 3090 cards write
another greedy reply than the rig at five of six depths. Each arm's reply leaves the rig's reply of the same arm at
token 7 (no context before the 64-token tail), token 6 (4,096), tokens 2 and 3 (8,192: `v4`, `v4s`), token 4 for `v4`
and after its first 48 tokens for `v4s` (16,384), and tokens 8 and 17 (32,768). Only at 48,640 tokens, where the reply
starts with a confident answer, is it the rig's byte for byte. Small differences in the numbers turn a near-tie early
in the reply, so the screen of the candidates has to run on a P106 (registry Q1487: FN93b, on a rig).

**The stack changes the numbers on the 3090s too.** `v4s` writes another greedy reply than `v4` at 8,192 tokens (from
token 2) and 16,384 (from token 4), and the same reply at 0, 4,096, 32,768 and 48,640. On the rig they part at 8,192,
16,384 and 32,768. The sampled replies (a fixed seed) part at 8,192, 16,384 and 32,768 on the 3090s. So the change does
not come from the P106's own kernels alone: some candidate changes results on any card.

**Each reply repeats itself:** greedy pass 2 is pass 1's reply in every load, and the second load of `v4` wrote the
first load's reply at every depth and pass it reached (0 to 32,768 tokens, greedy and sampled).

**The second load of `v4` hung twice in a prompt call** (registry Q1489). llama-server's main thread spun at full load
inside the CUDA driver's `cuMemcpyDtoHAsync` while all three cards sat idle. On its first start this happened after
2,048 of the 4,032 tokens that the 4,096-token depth's prime reads, and nothing moved for about 25 minutes. On its
second start it happened after 8,192 of the 16,388 tokens that the 48,640-token depth's prime reads, and nothing moved
for about 10 minutes. Both starts were then stopped.
The first loads of `v4` and `v4s` had run through. Another llama-server, on two other cards of the same machine, had
started before both hangs. The run ended there: the second load of `v4s` did not run.

### The replies

Greedy pass 1 of the first load of each arm. "same" or the first token that differs, counted from 1, against this
machine's `v4` and `v4s` and against the rig's (R162's first loads `v41` and `s1`). The rig kept each reply's md5 and
its first 160 characters, so a reply that differs from the rig's only after those reads "after char 160".

| Depth (tokens) | `v4` vs `v4s` here | `v4` vs `v4s` on the rig | `v4` here vs rig `v4` | `v4s` here vs rig `v4s` |
|---:|---|---|---|---|
| 0 | same | same | token 7 | token 7 |
| 4,096 | same | same | token 6 | token 6 |
| 8,192 | token 2 | character 45 | token 2 | token 3 |
| 16,384 | token 4 | token 6 | token 4 | after char 160 (48 tokens) |
| 32,768 | same | token 8 | token 8 | token 17 |
| 48,640 | same | same | same | same |

[replies.md](replies.md) has the same comparison as the script writes it, and [replies.tsv](replies.tsv) has every
reply: each load, the greedy passes 1 and 2 and the sampler pass, against this machine's `v4` and `v4s`, against the
rig's `v4` and `v4s`, and against the rig's reply of the same arm. The tables hold token positions and md5s, no text.

## Method

- **The build:** `v4-rc2` (`pascal-p106-v4` `eb82916d3`, upstream b12360), the rig tarball of
  `pascal-p106-v4-b12360-c828` (`eb312b7bc`, the same tree), whose `llama-server` (md5
  `e7afd26db01f4c671549ef12c504ada3`) is the binary R162 ran on the rig. It holds sm_61 code and compute 6.1 PTX; on
  the RTX 3090 (compute 8.6) the driver compiles the PTX, kept in a cache between loads.
- **The fork's kernels forced on, with the P106's shapes:** each Pascal kernel family runs on compute 6.1 only by
  default, and its mode 2 runs it on any NVIDIA card. Both arms carried `GGML_CUDA_MMVQ_PASCAL=2
  GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2` and the
  launch shapes of a 5-SM card with 96 KB of shared memory per SM (`GGML_CUDA_MMVQ_PASCAL_NSM=5
  GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVF_PASCAL_NSM=5`), as the fork's
  kernel checks through the PTX do ([2026-11-04_kshape](../2026-11-04_kshape/README.md)). llama.cpp's own kernels
  still choose some launch shapes from the card's SM count (the Gated DeltaNet kernels' warps, the MMQ tile, the
  softmax and top-k grids), and some paths from the card's compute capability. Those follow the 3090.
- **The model:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`). The copy read here differs from the rig's file only in
  one header key, the imatrix file's name (48,196,722,528 against 48,196,722,592 bytes). The last 48,000,000,000 bytes
  of both files have the same md5, `896386a598d2230f32de60f641bb9d97`; the rig's file is md5
  `7985df46ec7bb3867e482eaf91aaef39`.
- **The cards:** three RTX 3090 24 GB, `-ts 1,1,1` in place of the rig's nine-card split. In a layer split each layer
  runs whole on one card, and the cards hand activations to each other unchanged.
- **The arms:** R162's lines, with only the split, the PTX cache and the forcing above added (`loads.tsv` holds them as
  they ran). `v4` is `release/ornith-v3/settings.env` plus the three v4 switches, which is
  [release/ornith-v4](../../../release/ornith-v4/settings.env). `v4s` is `v4` plus `GGML_CUDA_HOP_STAGED=1
  GGML_CUDA_GRAPHS_FORCE=1 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64`, with
  `-b 4096` in place of `-b 2048`.
- **The requests:** R162's deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) with the same recorded
  text, tag (`^t1swift/`) and 64-token tail, so every depth's prompt has R162's md5. Per depth, in order (0, 4,096,
  8,192, 16,384, 32,768, 48,640): the prime, 256 greedy tokens twice and 256 sampled tokens with a fixed seed, each
  depth on top of the previous one's checkpoint as on the rig. Quiet checks off (`QUIET=0`, every request marked
  `valid=0`), no warm-up load, every reply kept whole (`KEEP_REPLY=1`: its token ids and each id's text) so that two
  replies compare token by token.
- **The loads:** `v41` (`v4`), `s1` (`v4s`), then `v42` (`v4`), which hung twice; `s2` did not run.
- **The comparison:** [run/replies.py](run/replies.py). Between two replies of this run it compares token ids. Against
  the rig it compares the reply's md5 with R162's. Where they differ, it finds the first character that differs from
  the rig's first 160 and reports the token of this reply that holds it.

## Commands

On the machine with the cards, from the repository root. The machine's folders go in the environment:

```bash
BIN=<v4-rc2 rig build>/bin/llama-server MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT=<the release check's proxy.jsonl.bodies.gz> OUT=<results>/step1 MOUNTS="<the folders of BIN, MODEL and OUT>" \
  GPUS=0,1,2 STEP=1 bash docs/data/2026-11-16_fn93_ornith_greedy_split/run/fn93.sh
OUT=<results>/step1 STEP=wait bash docs/data/2026-11-16_fn93_ornith_greedy_split/run/fn93.sh
python3 -I docs/data/2026-11-16_fn93_ornith_greedy_split/run/replies.py --run <results>/step1 \
  --rig <R162's collected run folder> --out replies
```

`DRY=1` prints both arms' server lines first. The script also defines one arm per candidate (`STEP=2`). That screen did
not run on these cards, since they cannot stand for the P106's replies.

## Files

| File | Content |
|---|---|
| [replies.md](replies.md), [replies.tsv](replies.tsv) | the comparison: greedy pass 1 per arm and depth; every reply of every load against both arms here and on the rig |
| [requests.csv](requests.csv) | per request its counts, the prompt's and the reply's md5s (the times are left out: no speed is read on this machine) |
| [loads.tsv](loads.tsv) | per load start its line as it ran (`<builds>` and `<jit>` stand for the machine's folders) |
| [run/fn93.sh](run/fn93.sh) | the run: both arms, the forcing, the container, `KEEP_REPLY=1` |
| [run/replies.py](run/replies.py) | the comparison tables |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2`. Model: Ornith-1.5-122B as Ornith-1.5-122B Pascal N2P GGUF
(`valid1`), described with its licence in [release/ornith-v3](../../../release/ornith-v3/README.md). The prompts are
the deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables hold counts,
token positions and md5s, no text. Run on three RTX 3090 cards; the rig's replies are R162's, on a Manli M-P106L9-N6G
(nine P106-090).
