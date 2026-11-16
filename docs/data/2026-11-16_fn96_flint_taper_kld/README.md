# Does the micro-batch taper change Flint's numbers? (FN96, Q1494; 2026-11-16)

What this measured: how far Flint, as Flint Pascal Q8 GGUF (`p065`), moves on nine P106-090 cards when a 512-token
prompt call is read in micro-batches of 64 tokens instead of Flint's usual 128. That is what the micro-batch taper
`--ubatch-taper short=600:64` does to a short call. Flint's tuned v4 settings `f4t` carry it: they read prompts
19.1 % faster than the plain v4 preset and give the first token 34.2 % sooner
([2026-11-14_r159_flint_release_agent](../2026-11-14_r159_flint_release_agent/README.md)). On Ornith the same taper
moved the next-token distributions by a mean KLD of 0.047
([2026-11-16_fn93c_ornith_taper_kld](../2026-11-16_fn93c_ornith_taper_kld/README.md)), and on Flash-Next by 0.0127
([2026-11-16_fn95_flashnext_taper_kld](../2026-11-16_fn95_flashnext_taper_kld/README.md)). This run is FN95's method
on Flint. Build `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the rig build reports `eb312b7bc`, branch
`pascal-p106-v4-b12360-c828`). Correctness only: no speed is read. Registry row Q1494 in
[quants/experiments.md](../../../quants/experiments.md).

## The rule, fixed before the run

The taper stays in `f4t` if all three hold; otherwise it leaves `f4t`:

- its mean KLD against Flint's micro-batch of 128 is at most **0.0066**, a tenth of the file's own distance from the
  16-bit model (0.0656, [2026-10-27_flint_near_q8](../2026-10-27_flint_near_q8/README.md));
- its perplexity is within **±0.1 %** of the base's;
- the repeat of the base's settings is exact.

## Result

**The taper stays in `f4t` under the rule. On Flint it changes nothing that can be measured:** the taper's numbers are
the repeat's to every printed digit.

On the same 24 chunks of 512 tokens of recorded agent sessions, each arm was scored against the base, Flint's
micro-batch of 128:

| | the taper against the base | the base repeated | Flint Pascal Q8 GGUF against the 16-bit model |
|---|---:|---:|---|
| mean KLD | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | **0.0656 ± 0.0029** (wikitext-2, 80 × 512 tokens); agent sessions against Q8_0 0.2292 |
| median KLD | 0.000000 | 0.000000 | |
| 99th percentile KLD | 0.000038 | 0.000038 | 0.964 |
| maximum KLD | 0.000055 | 0.000055 | |
| same top token | 100.000 ± 0.000 % | 100.000 ± 0.000 % | 91.775 ± 0.192 % |
| perplexity of the text | 19.741385 ± 1.175565 | 19.741385 ± 1.175565 | |
| against the base's own 19.7414 | **0.000 %** | 0.000 % | +4.1 % (the tool's ratio, wikitext-2) |
| the tool's paired error of the log ratio | ±1.12 % (0.011247) | ±1.12 % (0.011247) | |

| Condition | Limit | Measured | |
|---|---|---|---|
| mean KLD | ≤ 0.0066 | 0.000000 ± 0.000000 (99th percentile 0.000038) | met |
| perplexity against the base's own | within ±0.1 % | 0.000 % (19.741385 against 19.7414; equal to the repeat's 19.741385) | met |
| the repeat | exact | every chunk's running perplexity equal to the base's to the digit, 100 % same top token, no change in any token's probability | met |

- **The taper did act.** Its load logged the switch once ("micro-batch taper short=600:64 (n_ubatch 128)") and the
  plan "512 tokens, head 1, tail 1, 8 ubatches: 64x8" on all 24 chunks: each call ran as eight micro-batches of 64
  instead of the base's four of 128. `base` and `rep` logged no plan ([checks.txt](checks.txt)).
- **The taper's results are the repeat's.** Its statistics file is byte for byte the repeat's
  ([taper_stats.txt](taper_stats.txt), [rep_stats.txt](rep_stats.txt)). Every chunk's line matches too: perplexity,
  log ratio, KLD, Δp and same top token. All 24 running perplexities equal the base's ([ppl_by_chunk.tsv](ppl_by_chunk.tsv)).
  The KLD values that remain (from −0.000066 to 0.000055) are the base file's rounding, which stores each
  log-probability in 16 bits. They are the same in both arms, and no token's probability moved by more than 0.006 %.
- **The perplexity change is 0.000 %, so it is within two paired errors** (±2.25 %), and within ±0.1 %. The paired
  error the tool reports, ±1.12 %, does not come from the taper. Its log ratio compares each token with the stored
  base, and the stored base also cuts every log-probability off at 16 below the top token's. At Flint's perplexity of
  19.7 on this text many correct tokens fall below that line, so the tool reads the base back as 17.956487, not its
  own 19.7414. That moves the tool's ratio (1.099401) and widens its error. Both arms carry the same offset and the
  same error to the digit, so the taper's own difference from the repeat is zero at every chunk.
- **Why Flint differs from Ornith and Flash-Next, most likely:** every one of Flint's 49 layers is attention (full or
  sliding-window), with Q8_0 and Q5_0 weights, F16 routers and no recurrent state. Each token's arithmetic there does
  not depend on how many tokens share a micro-batch, so 64 or 128 gives the same bits. The two other models moved;
  this run does not test why.

This run does not change the preset.

## Method

`llama-perplexity` from the `v4-rc2` tree, one load per arm, in this order: `base` (writes the next-token
distributions, `--kl-divergence-base`), `taper`, then `rep` (both scored against those distributions,
`--kl-divergence`). Each load ran under [scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh), and none was
stopped.

- **The line** is `f4t`'s as R159's agent runs started it: [release/flint-v4/settings.env](../../../release/flint-v4/settings.env)'s
  environment as it is, plus `f4t`'s two variables `GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2`. Of its
  arguments it takes those that `llama-perplexity` takes and that act on the load or the arithmetic:
  `-ngl 99 -fa on -ub 128 -ts 6,6,5,5,6,5,5,6,6`.
  - `-c 512` replaces `-c 262144`, and `-b 512` stays, so each chunk is one call of 512 tokens of one sequence. The
    taper acts only on one-sequence calls of at most 600 tokens.
  - Server-only, so `llama-perplexity` does not take them: `-np 1`, `--ctx-checkpoints 4`, `--cache-ram 0`,
    `--pipe-depth 4`, `--alias`, `--jinja`, `--reasoning-format`, `--metrics`, `--no-webui`, and `f4t`'s
    `--ctx-checkpoints-vram 4` and `--ubatch-taper` (set through the environment instead, below).
  - Taken but with nothing to act on, since there is no sampling: `-bs` and `--temp`, `--top-p`, `--top-k`, `--min-p`.
  - Of the environment, `LLAMA_SERVER_YIELD` and `LLAMA_SAMPLING_TOP_K_ROWS` act in the server and its sampling,
    so here they have nothing to act on. `LLAMA_KV_SEQ_RM_BOUND` acts only when a sequence's positions are removed.
    `GGML_CUDA_GRAPHS_FORCE` acts in the backend as in the server.
  - `LLAMA_SCHED_PER_SIZE_RESERVE=2` logged that it needs per-size schedulers and was ignored, in all three loads.
- **The taper.** `--ubatch-taper` is a server flag, but the library reads the same spec from `LLAMA_UBATCH_TAPER` in
  every tool, so the `taper` arm sets `LLAMA_UBATCH_TAPER=short=600:64`. `-lv 4` shows the library's plan for each
  call ([checks.txt](checks.txt)).
- **The text** is FN93c's and FN95's: the release agent check's recorded requests (`proxy.jsonl.bodies.gz`, tag
  `t1swift/`) as plain text. That is the last request of each of its 12 sessions, every message as its role and
  content, with no chat template ([run/agent_text.py](run/agent_text.py)): 207,984 characters, the same md5 as
  FN93c's and FN95's. The first 24 × 512 of Flint's tokens were scored, the second half of each chunk (6,120 tokens).
  Flint's tokenizer splits the text differently from Flash-Next's, so the chunks hold other parts of the same text.
  The text is not published, only the numbers.
- **The base file** is 1,837,860,692 bytes (24 × 255 tokens × 150,144 log-probabilities × 2 bytes). It was removed
  with the run folder.
- **The yardstick** is the file's own distance from Flint's 16-bit weights on wikitext-2, 80 chunks of 512 tokens,
  on a build machine's cards ([2026-10-27_flint_near_q8](../2026-10-27_flint_near_q8/README.md), `kld.tsv`). It is
  another text and another machine, so it sets the size of a change and is not paired with this run.
- **The tables** were made from the logs by `fn96.sh`'s `tables` step, a `grep` and `awk` of the tool's own output.
  The same step reproduces FN95's published tables from FN95's logs.

The run was on 2026-11-16 on one rig: `base` 18:09:27–18:20:31, `taper` 18:20:31–18:31:22, `rep` 18:31:22–18:42:24.
The loads took 576.6, 576.6 and 575.7 s, reading the 40.4 GB file from a USB disk. The host's lowest MemAvailable was
1,348, 1,459 and 1,467 MiB, and swap peaked at 280 MiB.

## Commands

From the repository's root, on a machine that runs only ssh (paths on the rig go in the environment):

```bash
export RIG=<rig alias> WORK=<new folder on the rig, 1.9 GB free> BIN=<v4-rc2's llama-perplexity on the rig> \
  MODEL=<Flint Pascal Q8 GGUF on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz> OUT=<folder here>
F=docs/data/2026-11-16_fn96_flint_taper_kld/run/fn96.sh
STEP=dry bash $F       # each arm's command
STEP=start bash $F     # copies the text and scripts, starts run/driver.sh detached on the rig
STEP=wait bash $F      # returns when the three loads have ended
STEP=collect bash $F   # the logs to OUT, OUT/kld.tsv, the run folder on the rig removed, then the tables step
STEP=tables bash $F    # from OUT's logs only: ppl_by_chunk.tsv, checks.txt, the stats files, verdict.tsv and .txt
```

## Files

| File | Content |
|---|---|
| [verdict.tsv](verdict.tsv), [verdict.txt](verdict.txt) | per scored arm: mean KLD and its error, 99th percentile, same top token, perplexity against the base's own and the repeat's, the tool's paired error of the log ratio, whether the change is within two of it, chunks equal to the base, the largest Δp; and the rule's three conditions |
| [kld.tsv](kld.tsv) | per arm: chunks, mean, median, 99th percentile and maximum KLD, same top token, perplexity, the ratio to the stored base, the count of taper plans logged |
| [ppl_by_chunk.tsv](ppl_by_chunk.tsv) | each arm's running perplexity after each chunk, and each scored arm's against the base's |
| [taper_stats.txt](taper_stats.txt), [rep_stats.txt](rep_stats.txt) | `llama-perplexity`'s statistics for each scored arm: perplexity, KLD and token-probability percentiles |
| [checks.txt](checks.txt) | what each load logged about its line: chunks, batch and micro-batch, vocabulary, load time, the taper's switch and plans, the ignored switch |
| [run/fn96.sh](run/fn96.sh) | the run from a machine with ssh: dry, start, wait, collect, tables; FN95's `fn95.sh` with Flint's preset, `f4t`'s two variables and the tables step |
| [run/driver.sh](run/driver.sh) | the rig side: the text, then one `llama-perplexity` load per arm under the RAM watchdog |
| [run/agent_text.py](run/agent_text.py) | recorded agent requests as plain text |

The model file is my Flint Pascal Q8 GGUF (`p065`) ([release/flint-v4](../../../release/flint-v4/README.md)). The
recorded agent sessions are the agent check's own runs on the Exercism tasks
([bench/agent_check](../../../bench/agent_check/README.md)).
