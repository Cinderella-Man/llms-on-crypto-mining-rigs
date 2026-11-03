# Q433: how far Flash-Next's n-gram table can go below 4.5 bits (2026-11-03)

Qwen3.8-Flash-Next's n-gram table (`per_layer_token_embd`) is its largest tensor: 320,001,536 rows of 160 weights in
16 hashed heads (8 of 2-token and 8 of 3-token sequences), 28.8 GB in the IQ4_NL format (4.5 bits per weight) of every
published GGUF. A token gathers 16 of its rows on the host, so any row format decodes in time
([sub_1_bit_quants.md](../../research/notes/sub_1_bit_quants.md), section 4.1), and a smaller table would need less of
the rig's SSD ([Q435](../2026-11-03_ssd_room/README.md)) and fewer bytes per random read. This folder measures how much
quality each smaller row format costs. The registry row is Q433 in [quants/experiments.md](../../../quants/experiments.md).

## Result: the smallest table within noise is the shipped one, IQ4_NL at 28.8 GB

Mean KL divergence against the unmodified file ± standard error / same top token. "Error" is the written table's
relative squared error against the decoded IQ4_NL table (sum of squared differences over sum of squares). The
unmodified file's perplexity is 4.002 on wikitext, 1.665 on the agent sessions and 1.199 on code.

| Variant | Bits per weight | Table | Written as | Error | wiki | agent | code |
|---|---:|---:|---|---:|---|---|---|
| `self`: the file against its own logits | 4.5 | 28.8 GB | | 0 | 0 / 100 % | 0.0017 ± 0.0001 / 99.62 % | 0 / 100 % |
| `ub256`: the unmodified file at `-ub 256` | 4.5 | 28.8 GB | | 0 | 0.0158 ± 0.0003 / 95.39 % | 0.0255 ± 0.0012 / 97.38 % | 0.0040 ± 0.0002 / 99.25 % |
| `iq4nl`: the table re-encoded as is | 4.5 | 28.8 GB | IQ4_NL | 0.00003 | 0.0159 ± 0.0003 / 95.67 % | 0.0227 ± 0.0010 / 97.32 % | 0.0039 ± 0.0002 / 99.29 % |
| `iq4nl-q8`: the table copied to Q8_0 | 4.5 | 28.8 GB | Q8_0 | < 10⁻⁶ | 0.0157 ± 0.0003 / 95.51 % | 0.0253 ± 0.0011 / 97.35 % | 0.0041 ± 0.0002 / 99.26 % |
| **`q4-q8`** | 4.1 | 26.2 GB | Q8_0 | 0.0078 | **0.0275 ± 0.0006 / 94.40 %** | 0.0287 ± 0.0012 / 97.06 % | **0.0081 ± 0.0004 / 99.01 %** |
| `q3-q8` | 3.1 | 19.8 GB | Q8_0 | 0.032 | 0.0556 ± 0.0012 / 92.46 % | 0.0433 ± 0.0015 / 96.39 % | 0.0182 ± 0.0009 / 98.49 % |
| `q2` | 2.1 | 13.4 GB | IQ4_NL | 0.118 | 0.140 ± 0.003 / 88.44 % | 0.063 ± 0.002 / 95.62 % | 0.054 ± 0.002 / 97.02 % |
| `q1` (sign bits) | 1.1 | 7.0 GB | IQ4_NL | 0.365 | 0.335 ± 0.006 / 82.13 % | 0.112 ± 0.003 / 94.07 % | 0.160 ± 0.005 / 94.35 % |
| `bf64` | 0.5 | 3.2 GB | IQ4_NL | 0.709 | 0.550 ± 0.008 / 75.93 % | 0.178 ± 0.004 / 92.53 % | 0.295 ± 0.008 / 91.85 % |
| `bf32` | 0.3 | 1.9 GB | IQ4_NL | 0.835 | 0.601 ± 0.009 / 74.27 % | 0.183 ± 0.004 / 92.19 % | 0.324 ± 0.008 / 91.22 % |
| `zero`: no table | 0 | 0 | IQ4_NL | 1 | 0.649 ± 0.009 / 73.03 % | 0.234 ± 0.005 / 91.22 % | 0.311 ± 0.008 / 91.12 % |

- **Any change at all costs a floor of 0.016 / 0.023–0.025 / 0.004, and "within noise" means within that floor.** The
  file scored against its own logits gives exactly 0 on wikitext and code (0.0017 on the agent sessions, whose passes
  hold 16 micro-batches: a small run-to-run difference). But three changes that cannot matter in themselves, the
  table re-encoded to IQ4_NL (0.7 % of its blocks change, relative error 2.8 × 10⁻⁵), the table copied to Q8_0, and
  the unmodified file at `-ub 256` (the same weights in another summation order), all land on the same scores. As
  Q21 found for Flint, rounding-level differences flip some of the 10 experts chosen out of 512 and grow from there.
- **The 4-bit codebook is outside noise.** At 26.2 GB (−2.6 GB) and a relative error of 0.0078, `q4` scores +0.012 on
  wikitext (about 18 standard errors of the difference from the floor), double the floor on code (+0.004, 9 standard
  errors) and +0.003 to +0.006 on the agent sessions (2–4). It passes Q433's looser gate (≤ ~0.03 on agent sessions
  and code: 0.029 and 0.008), but would need a decoder in the engine (Q434) to save 2.6 GB of 28.8.
- **3.1 bits fails the gate** on the agent sessions (0.043) and costs 3.5 times the floor on prose; **2.1 bits** costs
  0.140 with 88.4 % same top token on prose, 0.063 and 0.054 on agent sessions and code. The sign bits and the binary
  bases come close to removing the table: `bf32` on code scores 0.324, no table 0.311.
- **Above the floor the cost follows the error.** On wikitext the score above the floor is 1.5 times the relative
  error at 4.1 bits, 1.2 at 3.1, 1.05 at 2.1, 0.87 at 1.1, and flattens toward the bound (no table: 0.63 at an error of
  1). Agent sessions and code lose 0.2–0.8 times the error. So any smaller table loses quality in proportion to its
  error, and the census shows no structure a better format could use to cut the error at the same size.
- **For the rig:** the table stays 28.8 GB in IQ4_NL, on the internal SSD (Q435). Q434's compressed table is parked:
  the only size that comes near noise saves 2.6 GB, and a table small enough for the cards (≤ ~7 GB) costs 0.11–0.34.

### The census: the table looks like Gaussian noise of one size

`census.tsv` (per head) and `census_hist.tsv` (the row-norm histogram per head):

| | Bigram heads 0–7 | Trigram heads 8–15 |
|---|---|---|
| Rows | 8 × ~20.0 M | 8 × ~20.0 M |
| Rows that are zero | 0 | 0 |
| Row norm: p1 / p10 / median / p90 / p99 | 0.068 / 0.079 / 0.093 / 0.109 / 0.126 | 0.083 / 0.090 / 0.098 / 0.107 / 0.116 |
| Coefficient of variation of the norm (pure Gaussian rows: 0.056) | 0.13 | 0.07 |
| Energy in the rows below the 10th / 50th percentile of norm | 5.8–6.0 % / 39–40 % | 7.3–7.6 % / 42–43 % |
| Kurtosis of the weights in RMS units (Gaussian: 3) | 2.95 | 2.96 |
| Mean largest \|weight\| of a row, in RMS units (160 Gaussian draws: ~2.9) | 2.89 | 2.89 |
| Energy in the column means | 1.5–2.0 % | 3.3–3.5 % |
| Column RMS, smallest to largest | 0.0074–0.0081 | 0.0076–0.0092 |
| Energy in the top 64 / 32 of 160 principal directions, rows scaled to unit norm (isotropic: 40 / 20 %) | 43.4–43.7 / 23.1–23.5 % | 46.0–46.1 / 25.9–26.1 % |

- **No row is empty or small.** Every hash slot is reached and trained: the smallest row norm is 30 % of the median,
  and the rows below the 10th percentile hold 6–8 % of the energy. There is nothing to drop or store at fewer bits by
  norm.
- **Inside a row the weights are Gaussian**, with equal columns and no dominant direction, so a row's information is
  spread over all 160 values. For such data the best possible code at R bits per weight leaves a relative error of
  2⁻²ᴿ, and a per-weight codebook comes close to that (q2: 0.115 against the scalar optimum 0.1175); the binary bases
  have no low-rank structure to use.
- The trigram heads' row norms spread almost as little as pure Gaussian rows' would (coefficient of variation 0.07
  against 0.056); the bigram heads' spread twice as much (0.13).

## How it was made

**The file.** Swift-1.5 GSQ-RCO IQ3_XXS ([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
revision `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`), in the three-shard layout of
[Q435](../2026-11-03_ssd_room/README.md) that gives the table a shard of its own (`scripts/gguf_reshard.py`, the bytes
unchanged). Only shard 3, the table, is rewritten; the variants link shards 1 and 2 of the unmodified file. The
reference for every KL divergence is the unmodified file itself, so a score measures what the table's change alone
costs.

**The census** (`run/table_bits.cpp`, `census`) decodes every row and records per head the row norms (quantiles from a
histogram of 0.002-wide bins of log10 of the norm), how much of the energy (sum of squared norms) sits in the small
rows, the kurtosis of the weights scaled to unit RMS per row, the mean largest |weight| in RMS units, the share of the
energy in the column means, and the spread of the column RMS.

**The round trips** (`run/table_bits.cpp`, `roundtrip`). Each row is decoded from IQ4_NL, passed through a format, and
encoded back: to IQ4_NL with ggml's own rules (`quantize_row_iq4_nl_impl`, no importance matrix), or to Q8_0 for the
3- and 4-bit formats, where an IQ4_NL re-encode would add about as much error as the format itself. The model then
reads the format's values through an unchanged engine. Formats, with the size of their own row layout:

| Name | Row layout | Bits per weight | Table |
|---|---|---:|---:|
| `iq4nl` (control) | the decoded row re-encoded to IQ4_NL as is | 4.5 | 28.8 GB |
| `q4`, `q3`, `q2` | 4, 3 or 2 bits per weight on a per-head codebook of 16, 8 or 4 levels, times a per-head column scale, times an fp16 row scale | 4.1, 3.1, 2.1 | 26.2, 19.8, 13.4 GB |
| `q1` | one bit per weight (two per-head levels: the sign bits), column and row scales as above | 1.1 | 7.0 GB |
| `bf64`, `bf32` | 64 or 32 sign bits on a per-head real basis of 64 or 32 vectors of 160, times an fp16 row scale | 0.5, 0.3 | 3.2, 1.9 GB |
| `zero` (bound) | every row zero: the model without its table | 0 | 0 |

- The per-head parameters (codebook levels and 160 column scales, or the 160 × r basis) cost at most 0.66 MB in all
  (`bf64`'s 16 bases in FP32) and
  are fitted on 262,144 evenly spaced rows of each head by alternating least squares: 6 rounds of row fits, level
  updates and column-scale updates for the codebooks; for the bases, a start from the top r principal directions of
  the rows scaled to unit norm, then 8 rounds of row fits and a least-squares basis update. Then every row is fitted on
  its own: for a codebook a search over 17 scales and Lloyd steps, for a basis the signs of the least-squares
  coefficients and of Bᵀw as two starts, each improved by single sign flips, the better one kept.
- The binary factorization here keeps the 160-wide factor real-valued rather than binary, as in NanoQuant's format
  (section 3.2 of the note): it is shared by 20 million rows, so binarizing it saves nothing.
- `-q8` after a name means the result was written back as Q8_0 (a 54.4 GB shard; the header's type field changed).

**The scores** (`run/kld.sh`): llama-perplexity's KL divergence against the unmodified file on the three texts of
[2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (wikitext-2 test, 80 × 512 tokens; held-out agent sessions,
12 × 8,192; the agent check's Exercism code, 80 × 512), the flags of that folder's `kld_texts.sh` (`-ub 512 -fa on`,
all layers on three RTX 3090s), built for compute 8.6 from the Miner Llama v3 base commit
`37e717ff65f641a2b365eebeef33fc1e1292e707`, which knows Flash-Next (`run/build.sh`). Two references measure the
measurement: `self`, the unmodified file scored against its own logits, and `ub256`, the unmodified file with
`-ub 256` (the same weights in another summation order).

## Files

| File | What |
|---|---|
| `kld.tsv` | per variant and text: written type, bits per weight and table size of the format, its relative error (format alone, as written), mean KLD ± SE, 99th and 99.9th percentile KLD, same top token ± SE, RMS change of the top token's probability, perplexity of the variant and of the unmodified file (`run/table.py`) |
| `roundtrip.tsv` | per variant and head: rows, relative error of the format, of the re-encode, and of the written table, share of IQ4_NL blocks written byte-identical |
| `census.tsv` | per head (and all): rows, zero rows, mean, RMS, smallest and largest row norm, norm percentiles, energy below the 10th and 50th percentile, kurtosis, mean largest \|weight\| in RMS units, energy in the column means, smallest and largest column RMS |
| `census_hist.tsv` | per head: the histogram of log10 of the row norm (0.002-wide bins: lower edge, rows, share of the head's energy) |
| `heads.tsv` | the 16 heads' n-gram length, first row and row count, from the file's `qwen4exp.ple.head_offsets` and `head_vocab_sizes` |
| `pca_energy.tsv` | per head: the share of the energy (rows scaled to unit norm) in the top 64 and top 32 principal directions |
| `fit_params.txt` | the fitted per-head parameters in brief: the codebook levels and the column scales' range of `q4`–`q1`, the basis vectors' mean squared norm of `bf64` and `bf32` |
| `run/build.sh` | the two programs in Docker: llama-perplexity for compute 8.6, and table_bits |
| `run/table_info.py` | where the table sits in the shards (byte offset, type field) and the heads' row ranges |
| `run/table_bits.cpp` | the census and the round trips (formats above, ggml's IQ4_NL and Q8_0 encoders) |
| `run/tb.sh` | table_bits in Docker |
| `run/variants.sh` | one variant folder per format: the rewritten table shard and links to the other two |
| `run/kld.sh` | the base logits, then each variant's scores on the three texts |
| `run/all.sh` | variants.sh and kld.sh as a pipeline |
| `run/table.py` | the logs to `kld.tsv` |

**Checks of the tools.** The IQ4_NL encoder is ggml's `quantize_row_iq4_nl_impl` without an importance matrix: on
the decoded table it reproduces 99.31 % of the file's blocks byte for byte, and the faster index search it uses
(counting midpoints instead of a binary search) wrote a shard identical byte for byte to the binary search's. The
Q8_0 control (the decoded table written as Q8_0) has a relative error below 5 × 10⁻⁷. The table_bits binaries of the
run differed from the published source only in the census histogram's label format and the added Q8_0 output; the
round-trip code of the IQ4_NL formats was the same throughout. `kld.sh` gained the `args` file for the `ub256` run;
its other runs used the same flags.

## Commands

On the quality machine (Docker, CUDA 12.9), with `W` a work folder on a fast disk (the variants and their shards are
read hard while they are scored), `M` the folder of the three-shard file, `K` a folder for the base logits (44 GB),
`T` the folder with the agent and code texts of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) and `L` a
llama.cpp checkout at `37e717ff6`:

```bash
D=docs/data/2026-11-03_table_bits/run
S=$M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
WORK=$W SRC=$L JOBS=8 DOCKER_ARGS="--label <label> --cpu-shares 256" bash $D/build.sh
python3 -I $D/table_info.py $S-0000{1,2,3}-of-00003.gguf > $W/table_info.txt
BIN=$W/bin/table_bits MOUNTS="$M $W" HIST=census_hist.tsv bash $D/tb.sh census $W/table_info.txt $W/row_norms_f16.bin \
  > census.tsv
export MODEL=$S-00001-of-00003.gguf TB=$W/bin/table_bits INFO=$W/table_info.txt WORK=$W THREADS=20 \
  BIN=$W/bin/llama-perplexity OUT=$K GPUS=2,3,4 WIKI=wikitext-2-raw/wiki.test.raw AGENT=$T/agent-heldout.txt \
  CODE=$T/code-exercism.txt MOUNTS="$M $W $K $T" DOCKER_ARGS="--label <label> --cpu-shares 256 --name kld" \
  RT_DOCKER_ARGS="--label <label> --cpu-shares 256 --name rt"
bash $D/kld.sh $MODEL                              # the base logits of the unmodified file
bash $D/all.sh iq4nl q2 zero q1 bf64 bf32          # round trips on the CPU while the GPUs score the previous one
for v in self ub256; do mkdir -p $W/variants/$v; ln -s $S-0000?-of-00003.gguf $W/variants/$v/; touch $W/variants/$v/done; done
echo "-ub 256" > $W/variants/ub256/args
RT_DOCKER_ARGS="--label <label> --cpu-shares 256 --name rt2" bash $D/all.sh self iq4nl-q8 q4-q8 q3-q8 ub256
python3 -I $D/table.py $K $W/variants self ub256 iq4nl iq4nl-q8 q4-q8 q3-q8 q2 q1 bf64 bf32 zero > kld.tsv
```

The second `all.sh` ran while the first was still scoring, with `WAIT_PID` set to the first one's process so that the
two never shared the GPUs.

The model's weights are under the Swift Open License 1.0 and qwen-community-1.0; the tables here hold only statistics
of the table's weights and scores, no weights.
