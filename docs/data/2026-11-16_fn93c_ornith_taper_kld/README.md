# Does the micro-batch taper cost Ornith quality? (FN93c, Q1487; 2026-11-16)

What this measured: how far Ornith-1.5-122B Pascal N2P GGUF (`valid1`: N2P and N3P experts, a Q5_0 output layer)
moves on nine P106-090 cards when a 512-token prompt call is read in micro-batches of 64 tokens instead of one of 512.
That is what the micro-batch taper `--ubatch-taper short=600:64` does to a short call, and FN93b
([2026-11-16_fn93b_ornith_greedy_rig](../2026-11-16_fn93b_ornith_greedy_rig/README.md)) found it is the candidate
that changes Ornith's greedy reply deep into a context. The taper's help says "results unchanged"; on Ornith it is not.
This run asks whether the change is within noise or costs quality. Build `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the
rig build reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`). Correctness only: no speed is read. Registry row
Q1487 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**Not within noise, and no measured cost: the taper is another arithmetic, about as large a change as the file's own
quantization on agent text, and it leaves perplexity where it was.** Whether it moves Ornith closer to or further from
the model is not settled by this run (below: what settles it).

On the same 24 chunks of 512 tokens of recorded agent sessions, each arm scored against the base, the v4 preset's
micro-batch of 512:

| | the taper against the base | the base repeated | `valid1` against the maker's Q8_0 |
|---|---:|---:|---|
| mean KLD | 0.047074 ± 0.002651 | 0.000062 ± 0.000062 | agent sessions 0.051815 ± 0.001033; code 0.061450 ± 0.002583; wikitext-2 0.176172 ± 0.002776 (0.175182 ± 0.002745 on the P106) |
| median KLD | 0.001752 | 0.000000 | agent 0.000861; code 0.000175; wiki 0.062778 |
| 99th percentile KLD | 0.711884 | 0.000033 | agent 0.771879; code 1.489304; wiki 1.832636 (1.795180 on the P106) |
| maximum KLD | 9.250742 | 0.379645 | agent 9.410448; code 11.203243; wiki 8.978916 |
| same top token | 93.889 ± 0.306 % | 100.000 ± 0.000 % | agent 95.037 %; code 96.691 %; wiki 84.142 % (84.270 % on the P106) |
| perplexity of the text | 4.075020 ± 0.149838 | 4.072976 ± 0.148994 | |
| against the base's own 4.0730 | +0.05 % | 0.00 % | agent +0.66 %, code +2.37 %, wiki +5.51 % (+5.41 % on the P106) |

- **The repeat is exact:** 23 of 24 chunks at a KLD of 0.00000, one token in chunk 17 at 0.38 with the same top token
  and no change in the probability of the text's next token, and the base's own perplexity, 4.0730, to the digit. The
  build is deterministic on the P106, so the run-to-run floor is zero.
- **The taper is far above that floor:** a mean KLD of 0.047, 6.1 % of tokens with another top token, a 99th
  percentile of 0.71. The change is spread over the whole text: its running mean is 0.22 after the first chunk and
  between 0.043 and 0.051 from the 12th chunk on ([taper_stats.txt](taper_stats.txt), per-chunk lines in the raw
  output).
- **The size of that change:** about as large as the file's whole distance from the maker's Q8_0 on held-out agent
  sessions (mean 0.052, 99th percentile 0.77), three quarters of it on code with the same 512-token chunks (0.061),
  and a quarter of it on wikitext-2 (0.176). Those rows are other texts (agent sessions in 8,192-token chunks, the agent
  check's code and wikitext-2 in 512-token chunks), so they set the scale, not a pair. Two builds that sum in another
  order differed by 0.0217 on the P106 on wikitext-2 and stayed equally far from the Q8_0
  ([2026-11-05_release_regression](../2026-11-05_release_regression/README.md)); the taper is twice that change.
- **No perplexity cost:** 4.075020 against the base's 4.0730, +0.05 %. The tool's own ratio against the stored base,
  1.005007 ± 0.005979, carries +0.45 % that the repeat shows too (1.004503 ± 0.001411): the base file keeps each
  token's log-probabilities rounded to 16 bits, which moves the perplexity read back from it to 4.054717. Within the
  taper's ±0.6 % either way.

**What settles it:** score both arms, the 512-token micro-batch and the taper, against Ornith's Q8_0 reference on the
P106 and pair them by chunk, as the 2026-11-05 build check did: d = KLD(taper) − KLD(base), a cost if d − 2 se > 0.
The reference for wikitext-2 in 512-token chunks exists (80 chunks, written on a build machine; 10.1 GB); with `-b 512`
every chunk is a 512-token call, which the taper reads in 64s. `llama-perplexity` scores every chunk of a base file and
ignores `--chunks`, so the whole file goes onto the rig (or a copy whose header and token list are cut to fewer chunks).

## Method

`llama-perplexity` from the `v4-rc2` tree, one load an arm, in this order: `base` (writes the next-token distributions,
`--kl-divergence-base`), `taper`, `rep` (both scored against them, `--kl-divergence`). Each load under
[scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh); none was stopped.

- **The line:** [release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)'s environment as it is, and
  of its arguments those `llama-perplexity` takes: `-ngl 99 -fa on -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off`.
  - `-c 512` and `-b 512` in place of `-c 49152 -b 2048`: each chunk is one call of 512 tokens of one sequence, the
    size of the short call the taper splits in the deep lane. With `-b 2048` the tool reads four chunks in one call as
    four sequences, and the taper acts only on calls of one sequence.
  - Not taken by `llama-perplexity`, or with nothing to act on there: `-np 1` (the tool sets its own), `-bs` and the
    sampling flags (no sampling), `--ctx-checkpoints 4 --cache-ram 0` (the server's cache), `--jinja` and
    `--reasoning-format` (no chat). Of the environment, `GGML_CUDA_KEEPALIVE=decode` and `_EAGER=1` act in the CUDA
    backend; `LLAMA_SERVER_YIELD=2` and `LLAMA_SAMPLING_TOP_K_ROWS=1` have nothing to act on.
- **The taper:** `--ubatch-taper` is a server flag, and the library reads the same spec from `LLAMA_UBATCH_TAPER`, so
  the `taper` arm sets `LLAMA_UBATCH_TAPER=short=600:64`. A call of at most 600 tokens that starts with nothing in
  flight and is waited for is read in micro-batches of 64: every chunk here. `-lv 4` shows the library's plan of each
  call: the `taper` load logged the switch once and "512 tokens, head 1, tail 1, 8 ubatches: 64x8" for all 24 chunks,
  `base` and `rep` no plan ([checks.txt](checks.txt)).
- **The text:** the release agent check's recorded requests (`proxy.jsonl.bodies.gz`, tag `t1swift/`, the text of
  R162 and FN93b), as plain text: the last request of each of its 12 sessions, every message as its role and content
  (the assistant's tool calls as name and arguments), no chat template ([run/agent_text.py](run/agent_text.py)).
  207,984 characters; the first 24 × 512 tokens were scored, the second half of each chunk (6,120 tokens). The text is
  not published, only the numbers.
- **The base:** 3,039,534,932 bytes (24 × 255 tokens × 248,320 log-probabilities × 2 bytes), removed with the run
  folder.
- **The yardstick:** `valid1` against the maker's Q8_0 from [2026-10-28_ornith_valid1](../2026-10-28_ornith_valid1/README.md)
  (`kld.tsv`, on a build machine's cards) and on the P106 from
  [2026-11-05_release_regression](../2026-11-05_release_regression/README.md) (`kld/ornith_vs_q8_ref_v3-b12076-ladder5.txt`,
  v3.0's build).

Run on 2026-11-16 on one rig: `base` 13:02:28–13:15:21, `taper` 13:15:21–13:26:45, `rep` 13:26:45–13:39:33; each
load 537 s; the host's lowest MemAvailable 1,109, 1,279 and 1,277 MiB.

## Commands

From the repository's root on a machine that runs only ssh (paths on the rig in the environment):

```bash
export RIG=<rig alias> WORK=<new folder on the rig, 3.1 GB free> BIN=<v4-rc2's llama-perplexity on the rig> \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz> OUT=<folder here>
F=docs/data/2026-11-16_fn93c_ornith_taper_kld/run/fn93c.sh
STEP=dry bash $F       # each arm's command
STEP=start bash $F     # copies the text and scripts, starts run/driver.sh detached on the rig
STEP=wait bash $F      # returns when the three loads have ended
STEP=collect bash $F   # the logs to OUT, OUT/kld.tsv, the run folder on the rig removed
```

## Files

| File | Content |
|---|---|
| [kld.tsv](kld.tsv) | per arm: chunks, mean, median, 99th percentile and maximum KLD, same top token, perplexity, the ratio to the stored base, the count of taper plans logged |
| [taper_stats.txt](taper_stats.txt), [rep_stats.txt](rep_stats.txt) | `llama-perplexity`'s statistics of each scored arm: perplexity, KLD and token-probability percentiles |
| [checks.txt](checks.txt) | what each load logged about its line: chunks, batch and micro-batch, vocabulary, load time, the taper's switch and plans |
| [run/fn93c.sh](run/fn93c.sh) | the run from a machine with ssh: dry, start, wait, collect (and the table) |
| [run/driver.sh](run/driver.sh) | the rig side: the text, then one `llama-perplexity` load an arm under the RAM watchdog |
| [run/agent_text.py](run/agent_text.py) | recorded agent requests as plain text |

The model file is Ornith-1.5-122B Pascal N2P GGUF ([models/ornith-1.5-122b](../../../models/ornith-1.5-122b/README.md));
the recorded agent sessions are the agent check's own runs on the Exercism tasks ([bench/agent_check](../../../bench/agent_check/README.md)).
