# Q1248: does the sparse prompt path change the model's output? KLD at 8,192 to 49,152 tokens through the compute 6.1 PTX (2026-11-11)

Miner Llama's sparse attention below Turing ([Q230](../2026-11-04_qsa_sparse_attn/README.md),
`GGML_CUDA_FATTN_SPARSE_PASCAL`) reads only the cells Qwen3.8-Flash-Next's QSA layers keep. On the rig, prompt
micro-batches of 128 through it read prompts 46.7 % faster ([R52, Q1214](../2026-11-10_deep_lane/prompt_depth/prompt_depth.md)), and
[C705](../2026-11-11_c705_sparse_prompt_rule/README.md) made that a rule of its own (`_PROMPT`). The decode path had
been checked on the whole model ([FN41](../2026-11-09_sparse_attn_whole_model/README.md)); prompts through the kernel
had not, and on the rig R52's `sp128` line wrote another greedy reply at depth 0 than the preset. This folder holds
the check: KL divergence and same top token of each prompt line against the same model on the dense kernels, on
recorded agent text, at four context depths, through the rig build's compute 6.1 PTX on three RTX 3090s. It is a
correctness check only: no time on these cards means anything for a P106.

**Result: the sparse prompt path changes the output no more than rounding does.** Against the dense kernels at
micro-batches of 128, prompt micro-batches of 128 through the kernel give a mean KLD of 0.002309 to 0.026896 over the
four depths (`sp128`), and C705's rule gives 0.002257 to 0.027539 (`pr128`). Reading the same prompt on the dense
kernels in micro-batches of 512 instead of 128 (`d512`, the preset's micro-batch, the same sums in another order)
moves it further at every depth: 0.003238 to 0.035516. Micro-batches of 512 through the kernel (`sp512`) move it as
far as `d512` does, within 1.11 standard errors at every depth. Same top token: 97.638 % to 98.974 % in every one of
these arms. A second load of the dense reference agrees with it to the stored reference's own precision, so every
difference comes from the arm. The kernel's own test faults show that the kernel took the prompt calls, and what the
check can see: dropping every split's last list entry (`_FAULT=3`) shows plainly (KLD 0.052640 and 0.276735, same top
94.725 % and 91.394 %); a 1 % error on the kernel's output (`_FAULT=1`) stays within the rounding yardstick in KLD and
shows only as a one-sided perplexity shift at 32,768. The rig rows that time this path (Q1272, Q1245) may run.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v4` at `08634ce00`, the rig tarball `llama.cpp-20261111-08634ce00-pascal-p106-v4.tar.gz` (md5 `2435f1072847805a3082c4c5bfe65066`); its `llama-perplexity` (md5 `eeb069be99eff00f462304c1fb1449cb`). It holds C705's `_PROMPT` rule. Its GPU code is sm_61 machine code and compute 6.1 PTX only (`CUDA : ARCHS = 610` in every log), so a newer card runs the PTX |
| Model | UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, as three shards (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`) |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight passed first |
| Depths | a depth D is one chunk of the text's first D tokens (`-c D --chunks 1`), scored on its second half, as `llama-perplexity` scores a chunk: 8,192 (positions 4,096–8,191, 4,095 tokens scored), 16,384 (8,192–16,383, 8,191), 32,768 (16,384–32,767, 16,383), 49,152 (24,576–49,151, 24,575) |
| Settings in every run | `-b 2048 -ngl 99 -fa on -fit off --no-host`; the v3 release preset's variables without the keep-alive (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2`); `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, so the kernel picks a P106-090's split counts |
| Time | 2026-11-11 19:05–20:22, 28 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |

The arms (`S` = `GGML_CUDA_FATTN_SPARSE_PASCAL`):

| Arm | `-ub` | Variables | What it is |
|---|---:|---|---|
| `d128` | 128 | `S=0` | the dense kernels: the reference; its log-probabilities are what every other arm is scored against |
| `d128b` | 128 | `S=0` | `d128` loaded again: the floor |
| `sp128` | 128 | `S=2 S_NQ=128` | R52's `sp128` line: every call of up to 128 query tokens through the kernel, at any cache size |
| `pr128` | 128 | `S=1 S_MIN_RATIO=1.0 S_PROMPT=1` | C705's rule: prompt micro-batches through the kernel once the cache holds 2,051 cells (Flash-Next's kept cells per query) |
| `d512` | 512 | `S=0` | the dense kernels at the preset's micro-batch: the same model and sums in another order, the yardstick for rounding |
| `sp512` | 512 | `S=2 S_NQ=512` | micro-batches of 512 through the kernel (R65's `sp512`; the kernel's cases at 512 query tokens pass, [FN63](../2026-11-11_fn63_sparse_256_512/README.md)) |
| `f1` | 128 | `sp128` + `S_FAULT=1` | the kernel's output × 1.01 (8,192 and 32,768 only) |
| `f3` | 128 | `sp128` + `S_FAULT=3` | every split drops the last entry of its list (8,192 and 32,768 only) |

## The result

Mean KLD ± its standard error, and same top token (%) ± its standard error, each arm against `d128` on the same
tokens. From [kld.tsv](kld.tsv):

| Arm | 8,192 | 16,384 | 32,768 | 49,152 |
|---|---|---|---|---|
| `d128b` KLD | −0.000003 ± 0.000000 | −0.000002 ± 0.000000 | −0.000002 ± 0.000000 | −0.000002 ± 0.000000 |
| `d128b` same top | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| **`sp128` KLD** | **0.002309 ± 0.000243** | **0.008160 ± 0.001152** | **0.026896 ± 0.002226** | **0.012169 ± 0.001288** |
| `sp128` same top | 98.852 ± 0.166 | 98.535 ± 0.133 | 97.870 ± 0.113 | 98.755 ± 0.071 |
| **`pr128` KLD** | **0.002257 ± 0.000300** | **0.008958 ± 0.001343** | **0.027539 ± 0.002088** | **0.011819 ± 0.000967** |
| `pr128` same top | 98.974 ± 0.157 | 98.633 ± 0.128 | 97.760 ± 0.116 | 98.645 ± 0.074 |
| `sp512` KLD | 0.002722 ± 0.000312 | 0.012185 ± 0.001920 | 0.039867 ± 0.003031 | 0.017834 ± 0.001656 |
| `sp512` same top | 98.828 ± 0.168 | 98.669 ± 0.127 | 97.638 ± 0.119 | 98.645 ± 0.074 |
| `d512` KLD (yardstick) | 0.003238 ± 0.000482 | 0.010659 ± 0.001461 | 0.035516 ± 0.002494 | 0.016476 ± 0.001374 |
| `d512` same top | 98.657 ± 0.180 | 98.633 ± 0.128 | 97.766 ± 0.115 | 98.669 ± 0.073 |
| `f1` KLD | 0.003390 ± 0.000659 | | 0.037105 ± 0.002833 | |
| `f1` same top | 98.632 ± 0.182 | | 97.772 ± 0.115 | |
| `f3` KLD | 0.052640 ± 0.003834 | | 0.276735 ± 0.009205 | |
| `f3` same top | 94.725 ± 0.349 | | 91.394 ± 0.219 | |

Each arm against the yardstick `d512`, in units of the two standard errors combined (from
[kld_vs_d512.tsv](kld_vs_d512.tsv); the arms are scored on the same tokens, so the errors are not independent and this
is a scale, not a test):

| Arm | KLD − `d512`'s: 8,192 | 16,384 | 32,768 | 49,152 | Same top − `d512`'s: 8,192 | 16,384 | 32,768 | 49,152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sp128` | −0.000929 (−1.72) | −0.002499 (−1.34) | −0.008620 (−2.58) | −0.004307 (−2.29) | +0.195 (0.80) | −0.098 (−0.53) | +0.104 (0.65) | +0.086 (0.84) |
| `pr128` | −0.000981 (−1.73) | −0.001701 (−0.86) | −0.007977 (−2.45) | −0.004657 (−2.77) | +0.317 (1.33) | 0.000 (0.00) | −0.006 (−0.04) | −0.024 (−0.23) |
| `sp512` | −0.000516 (−0.90) | +0.001526 (0.63) | +0.004351 (1.11) | +0.001358 (0.63) | +0.171 (0.69) | +0.036 (0.20) | −0.128 (−0.77) | −0.024 (−0.23) |
| `f1` | +0.000152 (0.19) | | +0.001589 (0.42) | | −0.025 (−0.10) | | +0.006 (0.04) | |
| `f3` | +0.049402 (12.78) | | +0.241219 (25.29) | | −3.932 (−10.01) | | −6.372 (−25.76) | |

What the tables say:

- **The dense reference repeats.** `d128b` against `d128`: mean KLD −0.000003 and −0.000002, largest single-token
  KLD 0.000004 to 0.000024, same top 100 % at every depth. `llama-perplexity` stores the reference's log-probabilities
  at 16 bits each, so equal logits score just below zero. Reading prompts, two loads agree to that precision; FN41's
  loads that did not repeat at depth were decoding.
- **The kernel took the prompt calls.** The faults act only inside the sparse kernel, and both change the `sp128`
  line (`f3` from 0.002309 to 0.052640 at 8,192 and from 0.026896 to 0.276735 at 32,768). `pr128` and `sp512` differ
  from their dense twins too: `pr128` against `d128`, which repeats exactly, and `sp512` against `d512`, whose KLD per
  depth is another.
- **The prompt path moves the output less than rounding does.** At micro-batches of 128 the kernel (`sp128`, `pr128`)
  stays below `d512` at every depth, by 0.86 to 2.77 standard errors in KLD, with the same top token within 1.33. At
  512, `sp512` stays within 1.11 of `d512`.
- **Perplexity moves both ways, no further than rounding moves it.** The mean log-ratio of an arm's perplexity to the
  reference's (1.599942, 1.520363, 1.655457 and 1.498846 at the four depths): `sp128` and `pr128` −0.004694 to
  +0.002522, `sp512` −0.005483 to +0.003799, `d512` −0.007665 to −0.000087 (each ± in [kld.tsv](kld.tsv)).
- **What the check cannot see.** A 1 % error on the kernel's output (`f1`) lands at `d512`'s KLD (0.19 and 0.42
  standard errors from it): on this model at these depths an error of that size cannot be told from rounding by KLD,
  as FN41 found for greedy text. At 32,768 it does show as a one-sided perplexity shift, −0.020497 ± 0.002726 against
  the rounding arms' −0.007665 to +0.003799; at 8,192 it does not (+0.001666 ± 0.001552). A fault that changes which
  cells are read (`f3`) shows at 12.78 and 25.29 standard errors in KLD.
- **The size of rounding on this model.** The largest single-token KLD is 17.610420 in `sp128` at 32,768 and again at
  49,152, and 13.999619 in `d512` at both (the two windows share positions 24,576–32,767): a few positions turn over
  under any change of summation order, while the median stays at 0.000015 to 0.000063. For scale, the whole Q2_0
  quantization costs a KLD of 0.285 against the Q8_0 on this text
  ([quality chain](../2026-11-03_flash_next_quality/README.md)).
- **R52's other greedy reply at depth 0** on the rig is within this: a greedy text parts at the first near-tie, and the
  dense kernels alone move the output this far when the micro-batch changes.

**Limits.** These cards run the fork's sparse kernel and llama.cpp's dense kernels through the compute 6.1 PTX. The
fork's other Pascal kernels ask for compute capability 6.1 exactly at the preset's settings, so they are off on a 3090
(FN41's third finding), and this is the sparse kernel inside the real model against llama.cpp's dense kernels, not
the P106's whole kernel set. One load per arm and depth; the depths overlap in text (32,768 and 49,152 share 8,192
scored positions). `sp256` was not run; it lies between `sp128` and `sp512`, which both pass.

## Files

| File | What |
|---|---|
| [kld.tsv](kld.tsv) | per arm and depth: scored positions and tokens, mean KLD ± SE, same top token ± SE, maximum, 99.9th and 99th percentile and median KLD, RMS of the probability change, the arm's and the reference's perplexity and the mean log-ratio of the two ± SE, and the differences from the floor `d128b` (in standard errors) |
| [kld_vs_d512.tsv](kld_vs_d512.tsv) | the same table with `d512` in place of the floor: the differences from the rounding yardstick |
| [runs.tsv](runs.tsv) | every run: arm, depth, `-ub`, exit status, start and end |
| [run/kld_depth.sh](run/kld_depth.sh) | the driver: one container and one `llama-perplexity` load per arm and depth; the first arm writes the reference log-probabilities, every later arm is scored against them |
| [run/kld_table.py](run/kld_table.py) | the tables from the runs' logs |

## Commands

On a build machine with Docker and three free cards (`W` a work folder; `BIN` the rig tarball's unpacked `bin/`;
`MODEL` the first shard; the cards by index or UUID):

```bash
S=GGML_CUDA_FATTN_SPARSE_PASCAL
export BIN=<build>/bin MODEL=<Swift GSQ-RCO Q2_0, first shard> TEXT=<agent-heldout.txt> OUT=$W/out GPUS=<three cards>
export MOUNTS="<the folders of BIN, MODEL and TEXT>" DOCKER_ARGS="--label <job>=1 --cpu-shares=256"
export PRESET_ENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2"
export ARM_UB_d128=128   ARM_ENV_d128="$S=0"
export ARM_UB_sp128=128  ARM_ENV_sp128="$S=2 ${S}_NQ=128"
export ARM_UB_d128b=128  ARM_ENV_d128b="$S=0"
export ARM_UB_d512=512   ARM_ENV_d512="$S=0"
export ARM_UB_sp512=512  ARM_ENV_sp512="$S=2 ${S}_NQ=512"
export ARM_UB_pr128=128  ARM_ENV_pr128="$S=1 ${S}_MIN_RATIO=1.0 ${S}_PROMPT=1"
export ARM_UB_f1=128     ARM_ENV_f1="$S=2 ${S}_NQ=128 ${S}_FAULT=1"
export ARM_UB_f3=128     ARM_ENV_f3="$S=2 ${S}_NQ=128 ${S}_FAULT=3"
# first pass: the reference and five arms at 8,192, 16,384, 32,768 and 49,152
ARMS="d128 sp128 d128b d512 sp512 pr128" setsid -f nohup bash run/kld_depth.sh > $W/run.console.txt 2>&1 < /dev/null
# second pass, once $OUT/done exists: the faults at two depths (d128 first, so its references are reused)
rm $OUT/done; ARMS="d128 f1 f3" DEPTHS="8192 32768" setsid -f nohup bash run/kld_depth.sh > $W/pass2.console.txt 2>&1 < /dev/null
# when $OUT/done exists again:
python3 -I run/kld_table.py $OUT d128 d128b d128b sp128 pr128 d512 sp512 f1 f3 > kld.tsv
python3 -I run/kld_table.py $OUT d128 d512 sp128 pr128 sp512 f1 f3 > kld_vs_d512.tsv
```

The reference log-probabilities take 1.9, 3.8, 7.6 and 12 GB at the four depths (`$OUT/base.<D>.kld`). The JIT cache
(`$OUT/jit`, `CUDA_CACHE_PATH`) keeps the compiled PTX between loads.

## Provenance and licence

- **Model:** Swift-1.5 GSQ-RCO Q2_0 is
  [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  under the Swift Open License 1.0 and qwen-community-1.0.
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md)
  (`run/agent_text.py` there); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
