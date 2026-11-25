# Q1765: a rounding yardstick on Ornith-1.5-122B that changes only the attention kernel (2026-11-25)

[C994](../2026-11-24_c994_attn_fault_selfref/README.md) set the form a check of a new attention kernel takes on
Ornith-1.5-122B Pascal N2P GGUF (`valid1`): the new kernel's output scored against the old kernel's on the same Q8_0
cache, stretch by stretch, beside a yardstick for what summing in another order does. Its yardstick, the Q8_0 cache
read in micro-batches of 128, reorders every sum in the model and is itself as large as an est. 4.25 % attention error,
so the check sees errors only from est. about 4.4 %. This folder measures a yardstick that changes only the attention:
the same cache and micro-batch with the fork's compute 6.1 prompt attention kernel switched off, so the build's
generic tile kernel takes the prompt passes ([Q1765](../../../quants/experiments.md)). It is a correctness check on
three RTX 3090s through the rig build's compute 6.1 PTX; no time on these cards means anything for a P106.

**Result: switching off only the fork's attention kernel moves the output 0.79 times as far as C994's yardstick does
(KLD 0.006319 against 0.007963), so the new-kernel check sees errors from est. 3.78 % (the error as large as the
yardstick) instead of est. 4.25 %. That is a modest gain, not a large one: changing the attention kernel alone already
moves Ornith's output nearly as much as reordering every sum in the model.**
- **The yardstick:** smaller than C994's on 11 of the 12 stretches in KLD (per stretch −0.001640 ± 0.000741, t −2.21;
  as a ratio −13.3 ± 6.0 %) and on all 12 in top-token disagreement (−0.341 ± 0.092 points, t −3.69; 1.720 % of tokens
  against 2.061 %). Its KLD follows C994's stretch by stretch (correlation 0.98).
- **The 5 % error against it is seen in every form:** +0.004710 ± 0.000901 KLD per stretch (t 5.23), divided by each
  stretch's yardstick +0.764 ± 0.070 (t 10.95); top-token disagreement +0.529 ± 0.074 points (t 7.12), divided
  +0.369 ± 0.077 (t 4.77); above the yardstick on all 12 stretches in both. Against C994's yardstick its top-token
  disagreement was not seen (t 2.49).
- **The smallest error seen, est.:** 4.04 % in KLD above the yardstick and 4.02 % divided by it, 4.25 % and 4.40 % in
  top-token disagreement (C994: 4.58 %, 4.43 %, 5.09 % and 4.97 %). The error as large as the yardstick is 3.78 % in
  KLD and 3.82 % in top-token disagreement (C994: 4.25 % and 4.58 %).
- **The run is C994's to the last digit:** `q8f1` repeats C994's `q8f1` on every stretch and in the mean (KLD 0.011030,
  same top 97.751 %), both arms read C994's reference perplexity (1.577859), and only `q8g`'s log lacks the fork's
  prompt kernel line.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's, C989's and C994's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C994's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -ub 512 -ctk q8_0 -ctv q8_0`; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's other Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before the pass |
| Time | 2026-11-25 11:58–12:13, 3 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |

### The arms (one pass, s)

| Arm | Prompt attention kernel | Fault | What it is |
|---|---|---|---|
| `q8` | the fork's compute 6.1 kernel (`GGML_CUDA_FATTN_TILE_PASCAL=2`) | | writes the reference: C994's `q8` |
| `q8g` | the build's generic tile kernel (`GGML_CUDA_FATTN_TILE_PASCAL=0`) | | the attention-only yardstick |
| `q8f1` | the fork's compute 6.1 kernel | `=1` | C994's 5 % error (every output of the kernel × 1.05), again |

**What `q8g` changes.** The build holds only compute 6.1 code, so no tensor-core attention kernel exists in it, and a
prompt pass of 512 tokens goes to the tile kernel. With `GGML_CUDA_FATTN_TILE_PASCAL=0` the fork's kernel declines it
and the build's generic tile kernel (`ggml_cuda_flash_attn_ext_tile`, the one upstream llama.cpp runs on Pascal) takes
it. That kernel reads F16 K/V, so the Q8_0 cache is first converted to F16 copies in the compute buffer, rounding
each value to F16's 11 bits. Everything outside the attention (the matrix products, the recurrent layers, the
micro-batch) is the same as in `q8`. The generic kernel logs nothing, so the log shows which kernel ran by what is
missing: the fork's kernel logs one line the first time it takes a pass (C994's logs: `compute-6.1 prompt attention
kernel: head 256, q8_0 K/V`), and `q8g`'s log has none. Its compute buffers differ too (two cards' larger, the
third's smaller), since the generic kernel needs the F16 copies and the fork's kernel buffers of its own.

## The check and its rule (fixed before the run)

For each stretch i, an arm's KLD against `q8`, its top-token disagreement (100 − same top %) and its ln PPL ratio,
read from `llama-perplexity`'s running means as in C994: G_i, H_i for `q8g`; F_i, T_i for `q8f1`; U_i, V_i for C994's
`q8u` (the Q8_0 cache at `-ub 128`, scored against the same reference, whose identity this run checks). Mean over the
12 stretches, standard error SE = sd/√12, t = mean/SE.

- **The answer to the row:** `q8g`'s KLD over all tokens against C994's 0.007963, and per stretch G_i − U_i (two-sided)
  and G_i/U_i − 1.
- **The new-kernel check with the new yardstick:** F_i − G_i (KLD, top-token disagreement, ln PPL ratio) and
  (F_i − G_i)/G_i (KLD, top-token disagreement). "Seen" is a mean at least 3 SE above zero in KLD and top-token
  disagreement, |t| ≥ 3 in ln PPL ratio, as in C994.
- **The smallest error seen, est.:** (a) the error whose KLD equals the yardstick's, 0.05 × √(yardstick KLD / 0.011030)
  over all tokens (C994's yardstick: 4.25 %), and per stretch 0.05 × √(mean G / mean F); for top-token disagreement
  0.05 × mean H / mean T. (b) As in C994, F_i scaled by (ε/0.05)² and T_i by ε/0.05 in each form against `q8g`: the
  smallest ε at which t reaches 3 (C994 against its yardstick: 4.58 % and 4.43 % in KLD, 5.09 % and 4.97 % in
  top-token disagreement). Both are estimates from one size of error and are marked so.
- **Validity:**
  - **The same reference as C994:** `q8g`'s and `q8f1`'s reference perplexity (read from `q8`'s file) equals C994's
    1.577859, and `q8f1` repeats C994's `q8f1` on every stretch (KLD to 0.00001, same top to 0.001 %) and in the mean
    KLD (0.011030). Then C994's U_i and V_i are on the same reference and can be set beside G_i.
  - **The kernels:** `q8` and `q8f1` log the fork's prompt kernel at head 256 with Q8_0 K/V; `q8g` logs no such line.

**How it came out** ([verdict.tsv](verdict.tsv)): every validity check held. Both arms read C994's reference
perplexity (1.577859), `q8f1` matched C994's `q8f1` on every stretch and in the mean, `q8` and `q8f1` logged the fork's
prompt kernel at head 256 with Q8_0 K/V, and `q8g` logged no such line. The 5 % error is seen against the new
yardstick in KLD and top-token disagreement, as a difference and divided per stretch; it is seen in no perplexity form.

## The result

### Each arm over the 12 stretches

Every arm against the Q8_0 cache's own run with the fork's kernel (`q8`), 49,140 scored tokens ([kld.tsv](kld.tsv));
C994's yardstick beside them. The ± in this table treats neighbouring tokens as independent; the paired table below
has the errors that hold.

| Arm | Mean KLD | Same top token | ln PPL ratio | 99th percentile KLD | Median KLD |
|---|---|---|---|---|---|
| `q8g`: the generic attention kernel | 0.006319 ± 0.000209 | 98.280 ± 0.059 % | −0.000281 ± 0.000606 | 0.080242 | 0.000159 |
| `q8f1`: 5 % error | 0.011030 ± 0.000373 | 97.751 ± 0.067 % | +0.000377 ± 0.000827 | 0.139895 | 0.000268 |
| C994's `q8u`: the whole model's sums reordered (`-ub 128`) | 0.007963 ± 0.000276 | 97.939 ± 0.064 % | +0.000428 ± 0.000703 | 0.094858 | 0.000185 |

### The forms, per stretch

Mean over the 12 stretches ± the standard error across stretches, and t = mean / SE; "seen" is t ≥ 3 in KLD and
top-token disagreement, |t| ≥ 3 in ln PPL ratio ([paired.tsv](paired.tsv)):

| Form | Pair | KLD | t | Top token differs (points) | t | ln PPL ratio | t |
|---|---|---|---:|---|---:|---|---:|
| alone | `q8g` | 0.006320 ± 0.001102 | 5.73 | 1.720 ± 0.231 | 7.43 | −0.00028 ± 0.00089 | −0.32 |
| yardstick against C994's | `q8g − q8u` | −0.001640 ± 0.000741 | −2.21 | −0.341 ± 0.092 | −3.69 | −0.00071 ± 0.00115 | −0.62 |
| yardstick against C994's | `q8g / q8u − 1` | −0.133 ± 0.060 | −2.22 | −0.165 ± 0.035 | −4.71 | | |
| against the new yardstick | `q8f1 − q8g` | **+0.004710 ± 0.000901** | **5.23** | **+0.529 ± 0.074** | **7.12** | +0.00066 ± 0.00162 | 0.41 |
| divided by the new yardstick | `(q8f1 − q8g) / q8g` | **+0.764 ± 0.070** | **10.95** | **+0.369 ± 0.077** | **4.77** | | |
| against C994's yardstick | `q8f1 − q8u` | +0.003070 ± 0.000469 | 6.54 | +0.188 ± 0.076 | 2.49 | −0.00005 ± 0.00216 | −0.02 |
| divided by C994's yardstick | `(q8f1 − q8u) / q8u` | +0.504 ± 0.075 | 6.73 | +0.121 ± 0.038 | 3.21 | | |

Bold: seen by the rule. The last two rows are C994's, recomputed from its tables by this folder's script.

**The smallest error seen, est.** ([verdict.tsv](verdict.tsv)): with `q8f1`'s KLD scaled by (ε/0.05)² and its
top-token disagreement by ε/0.05, the smallest ε at which t reaches 3, and the ε whose KLD or top-token disagreement
equals the yardstick's:

| Form | KLD, new yardstick | KLD, C994's | Top token differs, new yardstick | Top token differs, C994's |
|---|---|---|---|---|
| against the yardstick | 4.04 % | 4.58 % | 4.25 % | 5.09 % |
| divided by the yardstick | 4.02 % | 4.43 % | 4.40 % | 4.97 % |
| the error as large as the yardstick | 3.78 % | 4.25 % | 3.82 % | 4.58 % |

Over all tokens, 0.05 × √(0.006319 / 0.011030) is 3.78 % as well.

### Stretch by stretch

Each stretch's own values ([chunks.tsv](chunks.tsv), C994's `q8u` from [its chunks.tsv](../2026-11-24_c994_attn_fault_selfref/chunks.tsv)):

| Stretch | `q8g` KLD | `q8u` KLD | `q8g` / `q8u` | `q8f1 − q8g` KLD | ÷ `q8g` | `q8g` top differs | `q8u` top differs | `q8f1 − q8g` top differs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01006 | 0.01267 | 0.794 | +0.00528 | +0.525 | 1.905 | 2.686 | +0.488 |
| 2 | 0.00334 | 0.00359 | 0.930 | +0.00204 | +0.611 | 1.001 | 1.100 | +0.342 |
| 3 | 0.00745 | 0.00879 | 0.848 | +0.00399 | +0.536 | 2.467 | 2.637 | +0.367 |
| 4 | 0.01567 | 0.02495 | 0.628 | +0.01253 | +0.800 | 2.907 | 3.933 | +0.927 |
| 5 | 0.00493 | 0.00605 | 0.815 | +0.00341 | +0.692 | 1.660 | 2.149 | +0.316 |
| 6 | 0.00463 | 0.00557 | 0.831 | +0.00335 | +0.724 | 1.562 | 1.979 | +0.806 |
| 7 | 0.00663 | 0.00810 | 0.819 | +0.00447 | +0.674 | 2.589 | 2.708 | +0.562 |
| 8 | 0.00361 | 0.00244 | 1.480 | +0.00141 | +0.391 | 0.637 | 0.976 | +0.632 |
| 9 | 0.00434 | 0.00569 | 0.763 | +0.00492 | +1.134 | 1.166 | 1.686 | +0.933 |
| 10 | 0.00444 | 0.00575 | 0.772 | +0.00480 | +1.081 | 1.326 | 1.366 | +0.237 |
| 11 | 0.00167 | 0.00209 | 0.799 | +0.00176 | +1.054 | 0.677 | 0.758 | +0.176 |
| 12 | 0.00907 | 0.00983 | 0.923 | +0.00856 | +0.944 | 2.743 | 2.754 | +0.562 |

### What the tables say

- **The attention kernel alone is most of the yardstick.** The generic kernel on the same cache and micro-batch moves
  the output 0.79 times as far as reordering every sum in the model, and follows it stretch by stretch (correlation
  0.98). Two causes cannot be told apart here: the generic kernel also rounds every cached value from Q8_0 to F16,
  which is more than a new order of sums, and an attention layer's small change may grow through the layers after it to
  about the size any change reaches. The F16 cache, which both kernels read without a conversion, separates
  them ([Q1838](../../../quants/experiments.md)); a switch that changes only the order of the fork kernel's own sums
  would be the closest yardstick to a correct new kernel ([Q1837](../../../quants/experiments.md)).
- **With this yardstick, read the form divided per stretch.** The 5 % error's excess grows (0.004710 against
  0.003070) but so does its spread, because the yardstick is now well below the error's size and cancels less of the
  stretches' sensitivity; dividing by each stretch's yardstick removes it (t 10.95 against 5.23).
- **Top-token disagreement becomes a usable second measure:** the generic kernel tips fewer near-tied tokens than
  the whole model's reordering did (1.720 % against 2.061 %), so the 5 % error's excess is seen at t 7.12.
- **The square law keeps the floor close to the yardstick:** an error's KLD falls as ε², so even at t 10.95 for a 5 %
  error the smallest error seen is est. 4.02 %, about where the error's KLD meets the yardstick's (3.78 %). A
  yardstick well below the error moves the floor more than more stretches do.
- **Perplexity is still no kernel-error test:** the generic kernel moves it by −0.00028 ± 0.00089, the 5 % error by
  +0.00038 ± 0.00119.

**For later attention-kernel checks on Ornith:** score the new kernel against the fork's kernel on the same Q8_0
cache at `-ub 512`, stretch by stretch, beside the generic kernel (`GGML_CUDA_FATTN_TILE_PASCAL=0`) on the same cache,
and judge by the KLD and top-token disagreement above that yardstick divided per stretch. That sees uniform errors from
est. about 4.0 % in KLD and 4.4 % in top-token disagreement.

**Limits.** One fault size, so the smallest error seen is an estimate from the square law, not a measurement (Q1811
measures it with fault factors 1.01, 1.02, 1.03 and
1.05). One kind of error (a uniform scale), and the prompt kernel only. The
yardstick also converts the cache to F16, so it is not only a new order of sums. C994's `q8u` is set beside it from
C994's run; the identical reference and the identical `q8f1` make that pairing exact. One load per arm. Per-stretch
values come from `llama-perplexity`'s five-decimal running means, up to 0.00006 a stretch. These cards run the rig
build's compute 6.1 PTX with the fork's kernels forced on.

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the rule: the value, the rule and seen or not; the estimates against both yardsticks; the correlations; the validity checks; each arm's compute buffers |
| [paired.tsv](paired.tsv) | every form: the mean per stretch in KLD, top-token disagreement and ln PPL ratio, its standard error across stretches and t |
| [kld.tsv](kld.tsv) | per arm: scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th, 99th and 90th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [chunks.tsv](chunks.tsv) | each stretch's own ln PPL ratio, KLD and same top token, per arm |
| [runs.tsv](runs.tsv) | every run: arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the pass, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c1032_table.py](run/c1032_table.py) | the tables and the rule from the logs and C994's tables |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for the reference;
the bin folder unpacked from the rig tarball):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  MOUNTS="<the folders of the build, model and text>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # pass s; $W/DONE at the end
# once $W/DONE exists, from this folder:
python3 -I run/c1032_table.py $W ../2026-11-24_c994_attn_fault_selfref .
```

The run took 15 minutes, 2026-11-25 11:58–12:13: three model loads, every one exit 0 ([runs.tsv](runs.tsv)); each
arm read the 24 GB reference from a hard disk.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
