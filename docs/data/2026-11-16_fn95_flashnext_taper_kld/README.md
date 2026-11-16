# Does the micro-batch taper change Flash-Next's numbers? (FN95, Q1493; 2026-11-16)

What this measured: how far Qwen3.8-Flash-Next, as UkisAI's Swift GSQ-RCO Q2_0 (Swift-1.5, all three shards: the
weights and the n-gram table), moves on nine P106-090 cards when a 512-token prompt call is read in micro-batches of
64 tokens instead of one of 512. That is what the micro-batch taper `--ubatch-taper short=600:64` does to a short
call, and Flash-Next's v4 preset ([release/flash-next-v4](../../../release/flash-next-v4/settings.env)) carries it. On
Ornith the same taper moved the next-token distributions about as much as the file's own quantization does
([2026-11-16_fn93c_ornith_taper_kld](../2026-11-16_fn93c_ornith_taper_kld/README.md)); this run is FN93c's method on
Flash-Next. Build `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the rig build reports `eb312b7bc`, branch
`pascal-p106-v4-b12360-c828`). Correctness only: no speed is read. Registry row Q1493 in
[quants/experiments.md](../../../quants/experiments.md).

## The rule, fixed before the run

The taper stays in Flash-Next's preset if all three hold; otherwise it becomes opt-in, as on Ornith, and Flash-Next's
release numbers are measured again without it:

- its mean KLD against the preset's 512-token micro-batch is at most **0.028**, a tenth of the file's own distance from
  Swift's Q8_0 on agent text (0.285, [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md));
- its perplexity is within **±0.1 %** of the base's;
- the repeat of the base's settings is exact.

## Result

**The taper becomes opt-in under the rule.** Its KLD is low (0.0127, under the 0.028 limit) and the repeat is exact,
but perplexity moved by −0.31 %, which is outside ±0.1 %.

On the same 24 chunks of 512 tokens of recorded agent sessions, each arm was scored against the base, the v4 preset's
micro-batch of 512:

| | the taper against the base | the base repeated | Swift GSQ-RCO Q2_0 against Swift's Q8_0 |
|---|---:|---:|---|
| mean KLD | 0.012663 ± 0.000764 | −0.000000 ± 0.000000 | agent sessions **0.285104 ± 0.005573** (12 × 8,192 tokens); code 0.112447; wikitext-2 0.428258 |
| median KLD | 0.000250 | −0.000000 | |
| 99th percentile KLD | 0.183646 | 0.000034 | |
| maximum KLD | 1.865768 | 0.000062 | |
| same top token | 96.863 ± 0.223 % | 100.000 ± 0.000 % | agent 90.714 %; code 95.980 %; wiki 77.941 % |
| perplexity of the text | 2.378004 ± 0.065432 | 2.385354 ± 0.065786 | |
| against the base's own 2.3854 | **−0.31 %** | 0.00 % | agent −5.25 %, code +5.65 %, wiki +22.55 % (the tool's ratio) |

| Condition | Limit | Measured | |
|---|---|---|---|
| mean KLD | ≤ 0.028 | 0.012663 ± 0.000764 | met |
| perplexity against the base's own | within ±0.1 % | −0.31 % (2.378004 against 2.3854) | **not met** |
| the repeat | exact | every chunk's running perplexity equal to the base's to the digit, 100 % same top token, no change in any token's probability | met |

- **The repeat is exact.** All 24 running perplexities equal the base's, and the final one, 2.385354, is the base's
  2.3854. No token's probability changed (Δp 0.000 %, at most 0.006 %), and the top token was the same 100 % of the
  time. Its KLD values (from −0.000069 to 0.000062) are the rounding of the base file, which keeps each
  log-probability in 16 bits. So on the P106 this build is deterministic, and the run-to-run floor is zero.
- **The taper's KLD is small and spread over the text.** Its mean, 0.0127, is 4.4 % of the file's own 0.285 and 27 %
  of the taper's 0.047 on Ornith. 3.1 % of tokens get another top token, and the 99th percentile is 0.18. The running
  mean reads 0.0435 after the first chunk and stays between 0.0104 and 0.0142 from the sixth on
  ([taper_stats.txt](taper_stats.txt); per-chunk lines in the raw output).
  - For scale: on a build machine's cards, Swift's Q8_0 read at `-ub 256` against its own bases already moved by
    0.0291 on agent text (FN2's floor). That is another text in other chunks, so it sets the size of such a change
    and is not paired with this run.
- **Perplexity went down with the taper, by 0.31 %.** The text became slightly more likely, not less. The taper's
  running perplexity is lower than the base's at all 24 chunks (from −0.07 % to −1.43 %, [ppl_by_chunk.tsv](ppl_by_chunk.tsv)).
  - The paired error of the tool's log ratio is ±0.31 % (−0.000444 ± 0.003065 against the stored base), so the move
    is about one error from zero. It is a change of the size the rule excludes, but not a measured cost.
  - The tool's own ratio against the stored base (0.999556 for the taper, 1.002646 for the repeat) is offset by the
    base file's 16-bit rounding, which moves the perplexity read back from it to 2.379060. So the comparison is
    against the base's own 2.3854.
  - The file's own distance shows that a perplexity change has no fixed sign for quality: on agent text the Q2_0 reads
    5.25 % lower than its Q8_0, on code 5.65 % higher. That is why the rule is two-sided.

This run does not change the preset.

## Method

`llama-perplexity` from the `v4-rc2` tree, one load per arm, in this order: `base` (writes the next-token
distributions, `--kl-divergence-base`), `taper`, then `rep` (both scored against those distributions,
`--kl-divergence`). Each load ran under [scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh), and none was
stopped.

- **The line.** [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)'s environment as it
  is, and of its arguments those that `llama-perplexity` takes and that act on the load or the arithmetic:
  `-ngl 99 -fa on -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host`.
  - `-c 512` and `-b 512` replace `-c 65536 -b 4096`, so each chunk is one call of 512 tokens of one sequence. The
    taper acts only on one-sequence calls of at most 600 tokens.
  - Not loaded: the MTP draft (`-md`). `llama-perplexity` scores the main model alone.
  - Server-only, so `llama-perplexity` does not take them: `-np 1`, `--ctx-checkpoints 4`, `--cache-ram 0`,
    `--ctx-checkpoints-vram 4`, `--ctx-checkpoints-incall`, `--jinja`, `--reasoning-format`, the `--spec-*` flags,
    and `--ubatch-taper` (set through the environment instead, below).
  - Taken but with nothing to act on, since there is no sampling: `-bs` and `--temp`, `--top-p`, `--top-k`, `--min-p`.
  - Of the environment, the switches of the draft, the verify step and the server have nothing to act on in a
    one-model scoring run: `LLAMA_DRAFT_*`, `LLAMA_MTP_*`, `LLAMA_VERIFY_PREPARE`, `LLAMA_SPEC_REJECT_STOP`,
    `LLAMA_SERVER_YIELD` and `LLAMA_SAMPLING_TOP_K_ROWS`.
  - The few-token matrix kernels (`GGML_CUDA_MMVQ_PASCAL_*`, `GGML_CUDA_MMVF_F32_MT`) act only at 2–8 tokens, which a
    512- or 64-token micro-batch never is.
  - `LLAMA_SCHED_PER_SIZE_RESERVE=2` logged that it needs per-size schedulers and was ignored.
  - The hop, keep-alive, CUDA-graph, sparse-attention, QSA and cache switches act in the library and the backend as
    they do in the server.
- **The taper.** `--ubatch-taper` is a server flag, but the library reads the same spec from `LLAMA_UBATCH_TAPER` in
  every tool, so the `taper` arm sets `LLAMA_UBATCH_TAPER=short=600:64`. `-lv 4` shows the library's plan for each
  call. The `taper` load logged the switch once and "512 tokens, head 1, tail 1, 8 ubatches: 64x8" for all 24 chunks;
  `base` and `rep` logged no plan ([checks.txt](checks.txt)).
- **The text** is FN93c's: the release agent check's recorded requests (`proxy.jsonl.bodies.gz`, tag `t1swift/`) as
  plain text. That is the last request of each of its 12 sessions, every message as its role and content, with no
  chat template ([run/agent_text.py](run/agent_text.py)): 207,984 characters, the same md5 as FN93c's. The first
  24 × 512 tokens were scored, the second half of each chunk (6,120 tokens), the same count as FN93c's. The text is
  not published, only the numbers.
- **The base file** is 3,039,534,932 bytes (24 × 255 tokens × 248,320 log-probabilities × 2 bytes). It was removed
  with the run folder.
- **The yardstick** is FN2's: Swift GSQ-RCO Q2_0 against Swift-1.5's Q8_0 reference on a build machine's cards
  ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md), `kld.tsv`). Its agent text is
  held-out agent sessions in 12 chunks of 8,192 tokens.

The run was on 2026-11-16 on one rig: `base` 17:30:45–17:38:44, `taper` 17:38:44–17:45:21, `rep` 17:45:21–17:53:08.
The loads took 287, 286 and 285 s. The host's lowest MemAvailable was 373, 473 and 505 MiB; swap peaked at 571 MiB
while the base file was written, and at 32 MiB otherwise.

## Commands

From the repository's root, on a machine that runs only ssh (paths on the rig go in the environment):

```bash
export RIG=<rig alias> WORK=<new folder on the rig, 3.1 GB free> BIN=<v4-rc2's llama-perplexity on the rig> \
  MODEL=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf on the rig, shards 2 and 3 beside it> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> OUT=<folder here>
F=docs/data/2026-11-16_fn95_flashnext_taper_kld/run/fn95.sh
STEP=dry bash $F       # each arm's command
STEP=start bash $F     # copies the text and scripts, starts run/driver.sh detached on the rig
STEP=wait bash $F      # returns when the three loads have ended
STEP=collect bash $F   # the logs to OUT, OUT/kld.tsv, the run folder on the rig removed
```

## Files

| File | Content |
|---|---|
| [kld.tsv](kld.tsv) | per arm: chunks, mean, median, 99th percentile and maximum KLD, same top token, perplexity, the ratio to the stored base, the count of taper plans logged |
| [ppl_by_chunk.tsv](ppl_by_chunk.tsv) | each arm's running perplexity after each chunk, and the taper's against the base's |
| [taper_stats.txt](taper_stats.txt), [rep_stats.txt](rep_stats.txt) | `llama-perplexity`'s statistics for each scored arm: perplexity, KLD and token-probability percentiles |
| [checks.txt](checks.txt) | what each load logged about its line: chunks, batch and micro-batch, vocabulary, load time, the taper's switch and plans, the ignored switch |
| [run/fn95.sh](run/fn95.sh) | the run from a machine with ssh: dry, start, wait, collect (and the table); FN93c's `fn93c.sh` with the preset as a variable and `-lm`, `--no-host` among the flags taken |
| [run/driver.sh](run/driver.sh) | the rig side: the text, then one `llama-perplexity` load per arm under the RAM watchdog |
| [run/agent_text.py](run/agent_text.py) | recorded agent requests as plain text |

The model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table in a third shard
([release/flash-next-v4](../../../release/flash-next-v4/README.md)). The recorded agent sessions are the agent check's
own runs on the Exercism tasks ([bench/agent_check](../../../bench/agent_check/README.md)).
