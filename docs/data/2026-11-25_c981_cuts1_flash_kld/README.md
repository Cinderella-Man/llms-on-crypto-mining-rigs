# Does the taper's `cuts=1` change Flash-Next's next-token numbers? (C981, Q1714; 2026-11-25)

What this measured: how far Qwen3.8-Flash-Next, as UkisAI's Swift-1.5 GSQ-RCO Q2_0 (the file of Flash-Next's v5
preset, all three shards), moves when its prompt calls are read with the taper setting `short=1400:64,cuts=1` instead
of the preset's `short=1400:64`, and how both compare with the taper off. `cuts=1`
([C905](../2026-11-21_c905_taper_cuts/README.md)) plans each stretch between a call's in-call checkpoint cuts on its
own, so no micro-batch of 4 to 6 tokens is left beside a cut; on the 14 call sizes of the recorded agent sessions where
that changes the plan, each call reads in one micro-batch fewer. On nine P106s it was negligible by the fixed rule and
slower in no size bucket ([C930](../2026-11-24_c930_taper_cuts_rig/README.md)); on three RTX 3090s it changed every
such call's logits, and 64-token greedy replies parted at near-ties
([C956](../2026-11-23_c956_taper_cuts_gpu/README.md)). FN95's measure of the taper itself
([2026-11-16_fn95_flashnext_taper_kld](../2026-11-16_fn95_flashnext_taper_kld/README.md)) used `llama-perplexity`,
whose calls have no in-call cuts, so it cannot reach `cuts=1`. Here every call is read the way llama-server reads a
prompt with `--ctx-checkpoints-incall`. Correctness only: no speed is read. Registry row Q1714 in
[quants/experiments.md](../../../quants/experiments.md).

## Result

**`cuts=1` passes the rule.** Over the 60,656 positions it can change (112 fresh calls of the 14 sizes, from each
call's first changed micro-batch to its end, and the 16 tokens decoded after each), its mean KL divergence against
`short=1400:64` is **0.004828** (± 0.000078 over positions, ± 0.000330 over calls), a sixth of the 0.028 limit and
1.7 % of the file's own distance from Swift's Q8_0. The top token stays the same at 98.061 % of those positions, and
the text's perplexity moves by −0.006 % ± 0.052 %.

| Pair (fresh calls, the positions the pair can change) | Positions | Mean KLD | ± over calls | 99th percentile | Same top token | Perplexity change |
|---|---:|---:|---:|---:|---:|---:|
| `cuts=1` against `short=1400:64` (the rule) | 60,656 | **0.004828** | 0.000330 | 0.068119 | 98.061 % | −0.006 % ± 0.052 % |
| `short=1400:64` against the taper off | 109,256 | 0.006698 | 0.000372 | 0.092096 | 97.766 % | −0.035 % ± 0.046 % |
| `cuts=1` against the taper off | 108,720 | 0.006428 | 0.000341 | 0.088651 | 97.805 % | −0.038 % ± 0.045 % |
| `short=1400:64` against itself (the repeat) | 110,168 | 0 | 0 | 0 | 100 % | 0 |

- **It is a change of summation order, of the size any change of pieces makes.** It is smaller than what the preset's
  taper already does to the same calls against the taper off (0.006698), and `cuts=1` sits no farther from the taper
  off than the preset's taper does (0.006428). For scale: FN95's taper on a P106 (micro-batches of 64 against one of
  512, on 512-token calls) moved by 0.012663, and Swift's Q8_0 read at `-ub 256` against `-ub 512` on a build
  machine's cards by 0.0291 (FN2's floor); those are other texts and calls, so they give the size, not a pairing.
- **Every call size is far under the limit:** from 0.002619 (1,359 tokens) to 0.008382 (583 tokens)
  ([fresh.by_size.tsv](fresh.by_size.tsv)). The 16 tokens decoded after each call, which read the cache the call left,
  move by 0.004643 (98.326 % the same top token); the call's last position, whose distribution the server draws the
  first reply token from, by 0.004172 (111 of 112 the same top token). 99 % of positions stay under 0.068. At the two
  positions above 1 the preset's taper is the odd one out, not `cuts=1`: at the largest, 1.56 (a 599-token call),
  `short=1400:64` gives the text's next token a probability of 0.65, the taper off and `cuts=1` 0.994 and 0.997.
- **Everything that should not move did not, to the bit:** every position before a call's first changed micro-batch
  (37,440 of 37,440 the same logits), both control sizes, whose plan `cuts=1` leaves alone (12,072 of 12,072), and
  the repeat (110,168 of 110,168). So this build on these cards reads a call the same way every time, and every
  difference in the table is the change of pieces.
- **Every call ran the planned micro-batches:** in all 384 calls with the taper on, `short=1400:64` (both of its
  arms) ran the micro-batches the rig logged in R225's replay and `cuts=1` the ones C905's plan check computed, size
  for size ([fresh.checks.txt](fresh.checks.txt)); for example 577 tokens: `61 5 62 64x6 61 4` and `61 64x8 4`. The
  taper off logs no plan; it runs the plain split, each stretch between the cuts in pieces of up to 512 (`61 512 4`),
  and where that split's first piece is the taper's (569 and 577 tokens) the first stretch's logits are the same bits
  in both (912 of 912).

### On a cached context (the deep pass)

The same 14 sizes, 2 calls each, each call read after a 4,096-token prompt call of the text before it
([deep.summary.tsv](deep.summary.tsv), [deep.by_size.tsv](deep.by_size.tsv)). The 4,096-token call read the same in
every arm (its last logits the same bits in 112 of 112 arm calls), so each measured call sits on the same cache.

| Pair (deep calls, the positions the pair can change) | Positions | Mean KLD | ± over calls | 99th percentile | Same top token | Perplexity change |
|---|---:|---:|---:|---:|---:|---:|
| `cuts=1` against `short=1400:64` | 15,164 | **0.005246** | 0.001356 | 0.060405 | 98.463 % | +0.066 % ± 0.116 % |
| `short=1400:64` against the taper off | 24,296 | 0.005733 | 0.000968 | 0.066342 | 98.415 % | −0.067 % ± 0.089 % |
| `cuts=1` against the taper off | 24,162 | 0.005487 | 0.000891 | 0.063998 | 98.328 % | −0.026 % ± 0.081 % |
| `short=1400:64` against itself | 24,524 | 0 | 0 | 0 | 100 % | 0 |

The prompt positions alone are where the fresh calls are (0.004819 against 0.004833). The decoded positions average
more here, 0.019261 ± 0.014016 over 448, from a single position in one 815-token call: KLD 5.79, where the taper off
and `cuts=1` both put token 653 on top (log-probabilities of the text's token −4.90 and −6.35) and `short=1400:64`
alone puts the text's token, 353, there (−0.056). Such single positions come with any change of pieces on this model,
and fewer with `cuts=1` than with the preset's own taper: positions above 1 in the fresh and deep passes, `cuts=1`
against `short=1400:64` 2 and 6, `short=1400:64` against the taper off 6 and 14, `cuts=1` against the taper off 4
and 14.

**What it means:** the quality condition for `--ubatch-taper short=1400:64,cuts=1` in Flash-Next's preset holds, as
the speed condition did on the rig (C930). Taking it into the preset is a separate decision; this run does not change
the preset.

## The rule, fixed before the run

`cuts=1` passes if its mean KL divergence against `short=1400:64` is at most **0.028**, a tenth of Swift GSQ-RCO
Q2_0's own distance from Swift's Q8_0 on this agent text (0.285104, [FN2](../2026-11-03_flash_next_quality/README.md);
FN95's rule). The mean is taken over the positions `cuts=1` can change: in each of the 14 call sizes whose micro-batches
it changes, every position from the call's first micro-batch that differs between the two plans to the call's end,
plus the 16 tokens decoded after the call, over the fresh calls (8 per size).

Reported beside it, not judged: the same mean for `short=1400:64` against the taper off (what the preset's taper
already moves), the share of positions with the same top token, the log-probability of the text's next token
(perplexity), the repeat of `short=1400:64` (the floor), and the two control sizes whose plan `cuts=1` leaves as it is
(expected: the same bits).

## Method

- **The engine** is `pascal-p106-v5` at `2d839da1d`, the landing of 2026-11-25 and the base of every P106 row since
  (rig tarball md5 `7d774f05c29460a6993b19d38796047c`). `cuts=1` landed there on 2026-11-24 (merge `7ad7608b7`), and
  the preset that would take it runs on this branch. C905's own branch (`1720f26d3`) sits three landings lower, on
  `3dfad876f`; its taper code is the same (the batch code differs since only by an unrelated per-token field from an
  upstream merge), but the kernels around it are not the preset's any more. C930's rig verdict was on `af00bb13b`, the
  landing that brought `cuts=1` in.
- **The tool.** [run/kld_cuts.cpp](run/kld_cuts.cpp), a small program on llama.cpp's public calls, built by
  [run/build.sh](run/build.sh) against that build's own build tree (its static libraries and its `llama-perplexity`'s
  compile and link lines, in the image that built it; the tree's `llama-perplexity` has the tarball's md5, so the
  tool runs the tarball's engine code; built twice, byte-identical). Each measured call is a stretch of N tokens of the
  text, read as llama-server reads a prompt call with in-call checkpoints: one call of N tokens, cut inside at N − 516
  and N − 4 (`llama_set_ubatch_cuts`), a checkpoint filled at each cut (`llama_state_seq_store_at`, read after the
  call), the server's hints (head on a fresh call, tail since the call has an output), and the context's `n_rs_seq` at
  3, what llama-server sets for the preset's MTP draft of three tokens (the rule that keeps a stretch's last four tokens
  together). Then the next 16 tokens of the text are decoded one a call, as decoding reads them, on the cache the call
  left. Every position's logits are kept, the one difference from the server, which keeps only the last: so the last
  layer's feed-forward computes every row of a micro-batch, not just the output row. The cache and the recurrent state
  a call leaves are the same either way, and every arm reads the same way.
- **The arms**, each on a cleared memory, in this order per call: the taper off, `short=1400:64` (the preset),
  `short=1400:64,cuts=1`, `short=1400:64` again (the repeat). Per position the tool writes each arm's log-probability
  of the text's next token and its top token, and per pair (reference : test) the KL divergence of the test's
  distribution from the reference's, computed in double over the full vocabulary of 248,320 tokens from the raw logits,
  whether the top tokens agree, and whether the two logit rows are the same bits. Per call and arm it writes the
  micro-batches the context logged.
- **The calls.** Fresh pass: the 14 sizes where `cuts=1` changes the plan under `short=1400:64` (569, 577, 583, 599,
  610, 775, 803, 815, 896, 936, 1,086, 1,130, 1,300 and 1,359 tokens: every such size of R225's replay of the recorded
  agent sessions) and two control sizes it leaves alone (449 and 1,028), 8 calls each, the sizes in turn over
  consecutive stretches of the text from its first token: 128 calls, 110,296 tokens. Deep pass: the 14 sizes again, 2
  calls each, each call read after a 4,096-token prompt call of the text before it, as an agent's next turn sits on a
  cached context (that prefix call is longer than 1,400 tokens, so no taper rule touches it and it reads the same in
  every arm), from the token after the fresh pass's last.
- **The text** is [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md)'s agent text, 40 Flint-written agent
  sessions held out from training (`agent-heldout.txt`, md5 `f25e8e0ba6688053522b8f1b486aedf5`, 1,483,853 tokens in Flash-Next's vocabulary), the
  text FN2's yardstick was scored on. It is not published here, only the numbers.
- **The line**: [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)'s `PRESET_ENV`
  with the fork's compute-6.1 card paths forced as a P106 takes them
  ([scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh), `pascal_force_preset`), and of `PRESET_ARGS` those
  that act on a load or on the arithmetic: `-ngl 99 -fa on -c 65536 -b 4096 -ub 512 -fit off -lm dio --no-host`, with
  `-ts 1,1` for two cards. The model: 18,353 and 17,377 MiB on the two cards, the n-gram table (shard 3) in mapped
  host memory. [run/sysinfo.cpp](run/sysinfo.cpp), linked the same way and run with the run's environment, lists
  `MMVQ_PASCAL = 1` and `MMVF_PASCAL = 1` (with the preset's environment alone it lists neither): the fork's
  matrix-vector kernels, which the 4- to 6-token pieces run, compute here as on a P106.
- **The cards**: two of a build machine's RTX 3090s, through the rig build's compute 6.1 PTX. llama.cpp's own kernels
  there choose their splits by the card's 82 SMs, so the low bits are this machine's, not a P106's; the size of a
  change in pieces is what carries over (FN95 measured the taper on the P106 itself).
- **The run**, on 2026-11-25/26: the fresh pass 23:40:33–00:05:08 (one load, 128 calls), the deep pass
  00:05:09–00:15:41 (one load, 28 calls); both exit 0, the cards' CUDA check passed first, no container left.

## Commands

On the build machine, the build's work folder from `scripts/build_llamacpp_docker.sh` (`fork/pascal-p106-v5`
at `2d839da1d`) as `TREE`, this folder's `run/` and a copy of the repository in `REPO`:

```bash
TREE=<the build's work folder> OUT=$WORK/tool LOCK=<build lock file> \
  DOCKER_ARGS='<labels> --cpu-shares=256' bash run/build.sh
WORK=<work dir> TOOL=$WORK/tool/kld-cuts REPO=<repository copy> GPUS=0,1 \
  MODEL=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf, shards 2 and 3 beside it> \
  TEXT=<agent-heldout.txt> MOUNTS='<folders holding the model, the text and the tool>' \
  DOCKER_ARGS='<labels> --cpu-shares=256' PREFLIGHT=<CUDA check> bash run/run.sh
python3 -I run/summarize.py $WORK/fresh docs/data/2026-11-21_c905_taper_cuts/plans.tsv fresh
python3 -I run/summarize.py $WORK/deep docs/data/2026-11-21_c905_taper_cuts/plans.tsv deep
```

`run.sh` runs the passes in `PASSES` (default `fresh deep`), one load each, and writes per pass `kld.pos.tsv.gz` (per
position), `kld.calls.tsv` (per call and arm), `command.txt` and `run.log`; `summarize.py` writes the tables below.

## Files

| File | Content |
|---|---|
| [fresh.summary.tsv](fresh.summary.tsv), [deep.summary.tsv](deep.summary.tsv) | per pair and row set: rows, calls, mean KLD with its standard error over rows and over calls, 99th percentile and maximum, same top token, rows the same bits, perplexity of each arm and the change |
| [fresh.by_size.tsv](fresh.by_size.tsv), [deep.by_size.tsv](deep.by_size.tsv) | the same per pair and call size, over the changed rows |
| [fresh.checks.txt](fresh.checks.txt), [deep.checks.txt](deep.checks.txt) | each size's logged micro-batches per arm against C905's (the rig's before, the CPU plan after); the rows expected the same bits; the prefix calls; the checkpoint fills |
| [run/kld_cuts.cpp](run/kld_cuts.cpp) | the tool |
| [run/sysinfo.cpp](run/sysinfo.cpp) | the backend's feature line under an environment, without a model |
| [run/build.sh](run/build.sh) | both built against a rig build's own build tree |
| [run/run.sh](run/run.sh) | the build machine's driver: the preset's line with the paths forced, one load per pass |
| [run/summarize.py](run/summarize.py) | the tables from the tool's output |

The model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table in a third shard
([release/flash-next-v5](../../../release/flash-next-v5/settings.env)).
