# Q1726: the per-stretch check's floor on Ornith-1.5-122B, measured (2026-11-24)

[C989](../2026-11-24_c989_attn_error_floor/README.md) estimated how small an error in the Pascal prompt attention
kernel the per-stretch quality check on Ornith-1.5-122B Pascal N2P GGUF (`valid1`) sees against the F16 cache: est.
3.43 % in KLD and 1.37 % in top-token disagreement, scaled down from one fault of 5 %. This folder measures it instead:
the same check with the kernel's output multiplied by 1.01, 1.02, 1.03 and 1.05, set by the fault switch's new size
([2026-11-24_c993_fault_scale](../2026-11-24_c993_fault_scale/README.md), `GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE`)
([Q1726](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

**Result: against the F16 cache the check sees a 1 % attention error in KLD and a 2 % error in top-token
disagreement. The KLD floor is 1 % or below, under a third of C989's estimate of 3.43 %; the top-token floor lies
between 1 % and 2 %, around C989's estimate of 1.37 %.**
- **× 1.01 (a 1 % error):** +0.000500 ± 0.000157 KLD per stretch (3.18 standard errors, just over the line; higher on
  11 of the 12 stretches). The top token disagrees with the F16 cache slightly less often than without the error
  (−0.067 ± 0.074 points, t −0.90), so top-token disagreement does not see it.
- **× 1.02:** +0.001080 ± 0.000116 KLD (t 9.30, all 12 stretches) and +0.147 ± 0.041 points of top-token disagreement
  (t 3.56). **× 1.03:** +0.002320 ± 0.000464 (t 5.00) and +0.253 ± 0.054 (t 4.65). **× 1.05** repeats C989's 5 %
  fault to the last digit on every stretch: +0.004280 ± 0.000671 (t 6.38) and +0.403 ± 0.037 (t 10.95).
- **The KLD difference grows more slowly than the square of the error,** which is why C989's estimate was too high.
  Per (error)², it is 5.00 at 1 %, 2.70 at 2 %, 2.58 at 3 % and 1.71 at 5 %. Scaled down as a square from 5 %, a 1 %
  error looked 2.9 times smaller than it is. Top-token disagreement grows about in proportion from 2 % up (0.147, 0.253
  and 0.403 points at 2, 3 and 5 %).
- **Perplexity per stretch sees none of them** (t 1.84, 2.27, 1.37 and 0.30).
- **This floor is against an unfaulted run.** A new kernel brings its own rounding. Set beside C989's rounding arm
  (the F16 cache read in micro-batches of 128), only the 5 % error stands above it (t 4.37 in KLD); 3 % reaches 2.64
  and 1 % sits below it. That matches [C994](../2026-11-24_c994_attn_fault_selfref/README.md)'s estimate of about
  4.4 % for the check against rounding.

**For later attention-kernel checks on Ornith:** the 12 stretches against the F16 cache tell an attention error of 1 %
apart from no error in KLD, and of 2 % in top-token disagreement. Telling such an error apart from a correct kernel
that sums in another order is the harder bar, and today that needs about 5 %. A yardstick that changes only the
attention kernel ([Q1765](../../../quants/experiments.md), LLMRig point 26) may lower it. The same fault ladder in the
form a new-kernel check takes, against that yardstick, would measure that floor too
([Q1811](../../../quants/experiments.md)). Whether the KLD floor lies below 1 % is
[Q1812](../../../quants/experiments.md).

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5-fault-scale` at `ceaf710a1` (`pascal-p106-v5` `af00bb13b` plus the fault's size switch; rig tarball `llama.cpp-20261124-ceaf710a1-pascal-p106-v5-fault-scale.tar.gz`, md5 `76b179182497c2c5766c508f3558a74b`; its `llama-perplexity` md5 `68e10f2a7f64a288da88067a5297ba24`). With the switch unset its kernels are the base's bytes ([2026-11-24_c993_fault_scale](../2026-11-24_c993_fault_scale/README.md)) |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C976's and C989's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096`, `-ub 512`; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only), so each log names the kernel that took the prompt passes and the fault's factor |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before the pass |
| Time | 2026-11-24 19:45–20:24, 6 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |

### The arms (one pass, g)

| Arm | Cache | Fault | What it is |
|---|---|---|---|
| `f16` | F16 | | writes the reference, C989's pass g reference |
| `q8` | Q8_0 | | the unfaulted run, C989's `q8` |
| `q8f101` | Q8_0 | `=1`, `_SCALE=1.01` | every output of the prompt kernel × 1.01: a 1 % error in every attention layer's output on every token |
| `q8f102` | Q8_0 | `=1`, `_SCALE=1.02` | a 2 % error |
| `q8f103` | Q8_0 | `=1`, `_SCALE=1.03` | a 3 % error |
| `q8f105` | Q8_0 | `=1`, `_SCALE=1.05` | a 5 % error, C989's `q8f1` as a repeat |

The faults are C989's `=1` ([its README](../2026-11-24_c989_attn_error_floor/README.md#the-faults)) at a set size.
The prompt kernel multiplies its output, partial outputs of split blocks included, by the factor read from
`GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE`.

## The rule, fixed before the run

- **Pairs:** for each factor f, the fault arm (`q8f101`, `q8f102`, `q8f103`, `q8f105`) minus the unfaulted Q8_0 arm
  (`q8`), stretch by stretch over C976's 12 stretches, both scored against the F16 cache's reference written in the
  same run. That is C989's `q8f1 − q8` form.
- **Seen:** a factor is seen when the mean per-stretch difference is at least 3 standard errors above zero in KLD or in
  top-token disagreement (both one-sided). The ln PPL ratio difference is reported two-sided and is not part of the
  answer.
- **The answer:** the smallest factor seen, by KLD, by top-token disagreement and by either, against C989's est. 3.43 %
  and 1.37 %.
- **Validity:** every log names the compute 6.1 prompt attention kernel at head 256 with its cache type; every fault arm
  logs `GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 (tests of the tests): output x <its factor>` and no other arm logs it.
  Expected, not required: `q8` and `q8f105` repeat C989's `q8` and `q8f1` on every stretch and the reference repeats
  C989's perplexity (this build is `pascal-p106-v5` `af00bb13b` plus the switch, C989's was `3dfad876f`; with the
  switch unset the kernels are the base's bytes). If they differ, the comparison still stands within this run.
- **Expected from C989's and [C994](../2026-11-24_c994_attn_fault_selfref/README.md)'s numbers (est.):** the
  difference against the F16 cache has two parts. The fault's own KLD grows as the square of its size (C994: 0.011030
  at 5 %), and a cross term with the Q8_0 cache's own error grows in proportion to it (C994: −0.006750 at 5 %, the part
  of the fault that the F16 reference hides). If both hold at smaller sizes, the mean KLD difference is est. −0.00091 at
  × 1.01, −0.00094 at × 1.02, −0.00008 at × 1.03 and +0.00428 at × 1.05, so no factor below 1.05 would be seen in KLD,
  and C989's est. 3.43 % (which scaled the whole difference as the square) would be too low.

The rule was pushed (`dd9d9a20`) 2026-11-24 19:48, while the reference was being written.

**How it came out** ([verdict.tsv](verdict.tsv)): every validity check held. Every log names the prompt kernel at head
256 with its cache type; each fault arm logs its factor as the cards use it (1.00999999, 1.01999998, 1.02999997,
1.04999995) and the other arms log no fault. The expected repeats held too: `q8` and `q8f105` give C989's `q8` and
`q8f1` on every stretch to the last printed digit, and the reference's perplexity is C989's (1.579050). The expectation
written with the rule was wrong: the KLD difference is positive at every size (below).

## The result

### Each arm over the 12 stretches

Every arm against the F16 cache's reference, 49,140 scored tokens ([kld.tsv](kld.tsv)). The ± in this table treats
neighbouring tokens as independent; the paired table below has the errors that hold.

| Arm | Mean KLD | Same top token | ln PPL ratio | RMS Δp |
|---|---|---|---|---|
| `q8`: no fault | 0.006699 ± 0.000258 | 98.175 ± 0.060 % | −0.000728 ± 0.000651 | 3.022 % |
| `q8f101`: 1 % | 0.007199 ± 0.000278 | 98.242 ± 0.059 % | +0.000892 ± 0.000697 | 3.175 % |
| `q8f102`: 2 % | 0.007777 ± 0.000224 | 98.028 ± 0.063 % | +0.000089 ± 0.000682 | 3.258 % |
| `q8f103`: 3 % | 0.009024 ± 0.000295 | 97.922 ± 0.064 % | −0.000039 ± 0.000748 | 3.624 % |
| `q8f105`: 5 % | 0.010980 ± 0.000311 | 97.772 ± 0.067 % | −0.000377 ± 0.000798 | 3.945 % |

### Each error against the unfaulted run, per stretch

Mean of (fault arm − `q8`) over the 12 stretches ± the standard error across stretches, and t = mean / SE; "seen" is
t ≥ 3 in KLD and top-token disagreement ([paired.tsv](paired.tsv)):

| Error | KLD | t | stretches up | Top token differs (points) | t | stretches up | ln PPL ratio | t |
|---|---|---:|---:|---|---:|---:|---|---:|
| 1 % | **+0.000500 ± 0.000157** | **3.18** | 11 | −0.067 ± 0.074 | −0.90 | 6 | +0.00162 ± 0.00088 | 1.84 |
| 2 % | **+0.001080 ± 0.000116** | **9.30** | 12 | **+0.147 ± 0.041** | **3.56** | 9 | +0.00082 ± 0.00036 | 2.27 |
| 3 % | **+0.002320 ± 0.000464** | **5.00** | 12 | **+0.253 ± 0.054** | **4.65** | 12 | +0.00069 ± 0.00050 | 1.37 |
| 5 % | **+0.004280 ± 0.000671** | **6.38** | 12 | **+0.403 ± 0.037** | **10.95** | 12 | +0.00035 ± 0.00118 | 0.30 |

Bold: seen by the rule. **The answer:** the smallest error seen is 1 % in KLD and 2 % in top-token disagreement, so
1 % by either measure. Against C989's est. 3.43 % (KLD) and 1.37 % (top token).

How each difference grows with the error ε = f − 1, and what C989's estimate would have said from each size (the
square law for KLD, proportion for top token, with that size's spread):

| Error | KLD / ε² | Top token differs / ε | est. floor from this size, KLD | est. floor from this size, top token |
|---|---:|---:|---:|---:|
| 1 % | 5.00 | −6.7 | 0.97 % | (no estimate: negative) |
| 2 % | 2.70 | 7.35 | 1.14 % | 1.68 % |
| 3 % | 2.58 | 8.43 | 2.32 % | 1.94 % |
| 5 % | 1.71 | 8.06 | 3.43 % | 1.37 % |

### Stretch by stretch

Each stretch's own difference from `q8` ([chunks.tsv](chunks.tsv)): KLD, then top-token disagreement in points, with
`q8`'s own values against the F16 cache:

| Stretch | `q8` KLD | 1 % | 2 % | 3 % | 5 % | `q8` top differs | 1 % | 2 % | 3 % | 5 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00989 | +0.00158 | +0.00097 | +0.00253 | +0.00456 | 2.100 | +0.147 | +0.293 | +0.782 | +0.562 |
| 2 | 0.00293 | +0.00086 | +0.00151 | +0.00217 | +0.00250 | 1.026 | +0.267 | +0.293 | +0.194 | +0.292 |
| 3 | 0.00794 | +0.00071 | +0.00148 | +0.00235 | +0.00386 | 2.589 | +0.024 | +0.122 | +0.221 | +0.538 |
| 4 | 0.01992 | +0.00033 | +0.00188 | +0.00695 | +0.01092 | 3.197 | +0.026 | −0.072 | +0.319 | +0.588 |
| 5 | 0.00517 | +0.00067 | +0.00091 | +0.00150 | +0.00336 | 1.613 | −0.074 | −0.001 | +0.069 | +0.290 |
| 6 | 0.00473 | +0.00047 | +0.00105 | +0.00130 | +0.00300 | 1.733 | −0.024 | +0.145 | +0.221 | +0.490 |
| 7 | 0.00661 | +0.00035 | +0.00109 | +0.00238 | +0.00498 | 2.666 | −0.296 | +0.123 | +0.343 | +0.481 |
| 8 | 0.00201 | +0.00071 | +0.00071 | +0.00098 | +0.00258 | 0.828 | +0.026 | +0.393 | +0.099 | +0.399 |
| 9 | 0.00434 | +0.00035 | +0.00075 | +0.00297 | +0.00483 | 1.537 | −0.168 | −0.054 | +0.245 | +0.410 |
| 10 | 0.00446 | +0.00047 | +0.00135 | +0.00247 | +0.00491 | 1.341 | −0.048 | +0.198 | +0.297 | +0.300 |
| 11 | 0.00174 | +0.00032 | +0.00051 | +0.00091 | +0.00202 | 0.642 | +0.065 | +0.221 | +0.136 | +0.259 |
| 12 | 0.01066 | −0.00082 | +0.00075 | +0.00133 | +0.00384 | 2.628 | −0.749 | +0.103 | +0.110 | +0.227 |

### Beside rounding (context, not in the rule)

Each arm against C989's rounding yardstick, `f16u` (the F16 cache read in micro-batches of 128, so every sum in the
model runs in another order), stretch by stretch. `f16u` ran on C989's build; the pairing holds because this run's
`q8`, `q8f105` and reference repeat C989's to the last digit ([paired.tsv](paired.tsv)):

| Arm − `f16u` | KLD | t | Top token differs (points) | t |
|---|---|---:|---|---:|
| no fault | −0.000710 ± 0.000231 | −3.07 | −0.179 ± 0.058 | −3.07 |
| 1 % | −0.000210 ± 0.000232 | −0.90 | −0.246 ± 0.090 | −2.74 |
| 2 % | +0.000370 ± 0.000255 | 1.45 | −0.032 ± 0.070 | −0.46 |
| 3 % | +0.001610 ± 0.000609 | 2.64 | +0.074 ± 0.070 | 1.05 |
| 5 % | **+0.003570 ± 0.000817** | **4.37** | **+0.224 ± 0.054** | **4.11** |

### What the tables say

- **KLD is the sharper measure at small errors.** At 1 % it is higher than without the error on 11 of 12 stretches; at
  2 % on all 12, with the smallest spread of the four sizes (t 9.30). Top-token disagreement needs 2 %: at 1 % the
  top token flips toward the F16 cache's on some stretches and away on others (−0.749 points on stretch 12, +0.267 on
  stretch 2).
- **Scaling one size down as a square misjudges the floor.** Against the F16 cache the difference also carries a cross
  term with the Q8_0 cache's own change (C994), and that term does not scale with the error as the expectation
  assumed. At 1 % the difference is close to what the fault's own KLD would be by the square law (0.000441 est.); at
  5 % it is 0.39 of it. So the difference per ε² falls as the error grows, and an estimate from 5 % puts the floor
  too high (3.43 % against a measured 1 % or less). From 2 % the same estimate gives 1.14 %.
- **Over the 12 stretches every size adds KLD,** though on one stretch (12) the 1 % error brings the output closer to
  the F16 cache.
- **The floor against zero is not the floor against a correct kernel.** Beside rounding, a 3 % error reaches t 2.64
  and only 5 % is seen, as C994 found in the form a new-kernel check takes (est. 4.4 %).
- **Perplexity per stretch is not a kernel-error test** at any of these sizes.

**Limits.** One kind of error (a uniform scale of the prompt kernel's output, partial outputs included), in one
kernel. 1 % is the smallest size tried and is seen just over the line, so the KLD floor is at 1 % or below, not
bracketed. The 12 stretches are fixed, so a repeat on the same build gives the same values (the 5 % arm repeated
C989's exactly); the spread is the stretches', not the run's. Per-stretch values come from `llama-perplexity`'s
five-decimal running means. The comparison with rounding uses C989's arm from its build. These cards run the rig
build's compute 6.1 PTX with the fork's kernels forced on.

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the rule: per factor the value, the rule and seen or not; the answer; the validity checks; the repeats of C989 |
| [paired.tsv](paired.tsv) | per factor: the mean per stretch of (fault − `q8`) in KLD, top-token disagreement and ln PPL ratio, its standard error across stretches and t, the stretches where it is above zero, the difference per ε² (KLD) and per ε (top token), the floor each size would estimate; and each arm against C989's `f16u` (context) |
| [kld.tsv](kld.tsv) | per arm: scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th, 99th and 90th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [chunks.tsv](chunks.tsv) | each stretch's own ln PPL ratio, KLD and same top token, per arm |
| [runs.tsv](runs.tsv) | every run: arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the pass, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c993_table.py](run/c993_table.py) | the tables and the rule from the logs and C989's tables |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for the reference;
the bin folder unpacked from the rig tarball of `ceaf710a1`):

```bash
export WORK=$W BIN=<ceaf710a1 tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  MOUNTS="<the folders of the build, model and text>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # pass g; $W/DONE at the end
# once $W/DONE exists, from this folder:
python3 -I run/c993_table.py $W ../2026-11-24_c989_attn_error_floor .
```

The run took 39 minutes, 2026-11-24 19:45–20:24: six model loads, every one exit 0 ([runs.tsv](runs.tsv)); the
model was read from a hard disk.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
