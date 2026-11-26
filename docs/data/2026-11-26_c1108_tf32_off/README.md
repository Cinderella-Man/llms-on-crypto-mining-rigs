# C1108: cuBLAS as on a P106 in the GPU checks that run the rig build on newer cards

Registry row [Q1841](../../../quants/experiments.md) (C1108, from Q1772's
[C1039](../2026-11-25_c1039_cohere2_nmse/README.md)). Correctness only, nothing timed.

**Result: with TF32 off and cuBLAS computing in F32, an RTX 3090 running the rig build puts every random model's
card-against-CPU nmse at the P106's level, and no verdict of the card suite or of the pipe check changes.** TF32 off
alone (`NVIDIA_TF32_OVERRIDE=0`, what the row asked for) brings 135 of the 148 random models that have a P106 number
within 10 times of it; the median falls from 9.04e-08 to 3.12e-14. The other 13 stay higher because cuBLAS on this
card also computes F16 sources in F16, which a P106 never does; with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` as well, all
148 are within 4 times of the P106's number (the largest ratio 3.22). So
[scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh) gets `PASCAL_CUBLAS` with both settings, both forced sets
carry it, and every GPU-check driver passes it. One exception remains: a P106 computes BF16 weights' products of 9 to
128 tokens in BF16, and with the compute type at F32 the 3090 computes them in F32. Of the earlier checks on these
cards, one needs a rerun: Flash-Next's verify drift (C1128), whose sizes held TF32 rounding on one side of the
comparison only.

## Why

The GPU checks of Miner Llama's rig build run on a build machine's RTX 3090s: the build holds only compute 6.1 code,
and the card's driver compiles its PTX. Two of llama.cpp's choices about cuBLAS read the card itself, not the build:
- ggml's CUDA backend sets `CUBLAS_TF32_TENSOR_OP_MATH` on every cuBLAS handle (`ggml/src/ggml-cuda/common.cuh`). On
  an Ampere or newer card cuBLAS then rounds the inputs of every product it computes in F32 to TF32, which keeps 10
  bits of mantissa: an F32 weight's product with a prompt of more than 8 tokens, for one. A P106 has no TF32.
  NVIDIA's `NVIDIA_TF32_OVERRIDE=0` overrides the library's own setting.
- `ggml_cuda_mul_mat_cublas` computes F16 and quantized sources in F16 on a card with fast F16
  (`fast_fp16_hardware_available`, false for compute 6.1 only) and in F32 on a P106. BF16 sources it computes in BF16
  on a card with fast BF16, and on a P106 in BF16 up to 128 tokens and in F32 above.
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` makes every source F32 on any card: the P106's choice for every source but BF16
  at 9 to 128 tokens. On its own it does not help, because TF32 then rounds the F32 products (C1039's `f32` set).

In C1039 TF32 put every random model's card-against-CPU nmse near 1e-7 instead of the P106's 1e-14, and hid a real
change of a few units in the last place.

## The change

`scripts/pascal_forced.sh` gets `PASCAL_CUBLAS="NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32"`, and both
forced sets carry it: `PASCAL_FORCED_MMV`, and so `PASCAL_FORCED` and `pascal_force_preset`. Every driver that sources
the file gets it without an edit. It is a set of its own, not one more kernel switch, because it describes the card,
not a kernel choice: a P106 computes cuBLAS's products this way whatever its switches, so a run with no switch
(llama.cpp's own kernels) passes `$PASCAL_CUBLAS` alone. Three drivers set their arms' settings themselves and were
changed too:
- [bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh), the pipe step of every GPU check:
  `CARD_ENV` (default `NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`) goes into every arm, the `stock` arm
  too, and into each arm's `env.txt`; `CARD_ENV=` leaves cuBLAS as the card runs it;
- [C1023's replay driver](../2026-11-24_c1023_suite_record/run/run.sh) for the `MUL_MAT` and `MUL_MAT_ID` records:
  its default `ON` starts with the two settings;
- [C1042's replay driver](../2026-11-26_c1042_suite_records/run/run.sh): its three attention arms (no switch, the tile
  kernel, every Pascal attention switch) pass `$PASCAL_CUBLAS`; its fused-case arms use the forced sets already.
  `FLASH_ATTN_EXT` reaches no cuBLAS product, so its verdicts are C1042's; this driver was not run here.

## What ran

One rig build, `pascal-p106-v5` `2d839da1d` (the landing of 2026-11-25; `test-llama-archs` md5
`29ecef4f0faee82f47d7816874a839f2`, `test-backend-ops` md5 `65048f96fa48b8722c7ae9f03638b6b0`), on one RTX 3090 of
the build machine, 2026-11-26 06:57–07:30, in three steps of [run/run.sh](run/run.sh):

| Step | What | Settings compared |
|---|---|---|
| archs | [run/archs.sh](run/archs.sh): `test-llama-archs -b CUDA0 -s 1`, every setting twice | `stock` nothing set; `notf32` TF32 off alone; `before` the forced set before C1108; `after` that set with TF32 off alone; `f32` the forced set now (`before` and `PASCAL_CUBLAS`) |
| replay | the card suite's `MUL_MAT` share, 5,458 cases, replayed from C1023's kept record by C1023's driver | `before` its settings before C1108 (the fork's matrix-vector kernels at a P106's shapes); `tf32` those and TF32 off; `after` its default now (those and `PASCAL_CUBLAS`) |
| pipe | the pipe check on this one card (layouts `one` and `virt`, three virtual devices), its default arms and checks | `before` `CARD_ENV=`; `tf32` `CARD_ENV=NVIDIA_TF32_OVERRIDE=0`; `after` its default |

`test-llama-archs` writes each random model's CPU logits in its first run and compares every later run's with them:
`stock` and `notf32` against the first `stock` run, the other three against the first `before` run (the forced set's
`LLAMA_NO_MOE_DOWN_RESCALE=tg` changes the MoE models' graphs on the CPU too). The archs step ran twice: four settings
at 06:57, then all five at 07:16; the four settings' 1,256 numbers are the same in both.

## Result: `test-llama-archs`

Every run passes all 159 tests (157 card rows, 28 skipped as in every build), every run's CPU logits are its pair's
first run's bit for bit (152 files), and every repeat gives its first run's card number for all 157 models
([archs-runs.tsv](archs-runs.tsv), [archs-nmse.tsv](archs-nmse.tsv)). The card-against-CPU nmse by setting
([archs-summary.tsv](archs-summary.tsv)); the P106's numbers are C912's on `af00bb13b`
([logits.tsv](../2026-11-24_c912_p106_check/logits.tsv)), whose CPU logits `2d839da1d` repeats bit for bit:

| Setting | Median nmse | Largest (model) | Models at or below 1e-12 | Within 10× of the P106 (of 148) | Card ÷ P106, geometric mean | Largest card ÷ P106 (model) |
|---|---:|---|---:|---:|---:|---|
| `stock` | 9.04e-08 | 7.31e-07 (`gemma4`) | 2 | 15 | 119,000 | 3,980,000 (`granite`) |
| `notf32` | 3.12e-14 | 1.87e-07 (`kimi-k3`) | 125 | 135 | 0.464 | 8,990 (`kimi-k3`) |
| `before` | 9.04e-08 | 7.31e-07 (`gemma4`) | 2 | 15 | 119,000 | 3,980,000 (`granite`) |
| `after` | 3.12e-14 | 1.87e-07 (`kimi-k3`) | 125 | 135 | 0.464 | 8,990 (`kimi-k3`) |
| **`f32`** (now) | **3.08e-14** | 1.43e-07 (`gemma4`) | 136 | **148** | **0.192** | **3.22** (`exaone-moe`) |

`stock` and `before` give the same number for every model, and so do `notf32` and `after`: these random models'
weights are F32, which the fork's Pascal kernels do not take, so the forced set changes nothing here but cuBLAS's
arithmetic. Some models:

| Model | `stock` | `after` (TF32 off) | `f32` (now) | P106 (C912) |
|---|---:|---:|---:|---:|
| `llama` | 1.07e-07 | 2.22e-14 | 2.22e-14 | 8.69e-14 |
| `qwen3next` | 8.72e-08 | 2.34e-14 | 2.34e-14 | 2.78e-14 |
| `qwen35moe` | 8.73e-08 | 2.44e-14 | 2.44e-14 | 2.67e-14 |
| `flint` | 9.22e-08 | 3.15e-14 | 3.15e-14 | 3.51e-14 |
| `cohere2` | 8.92e-08 | 2.44e-14 | 2.44e-14 | 2.80e-14 |
| `gemma4` | 7.31e-07 | 1.43e-07 | 1.43e-07 | 1.42e-07 |
| `deepseek2` | 2.85e-09 | 1.37e-09 | 1.01e-13 | 4.93e-13 |
| `kimi-k3` | 3.29e-07 | 1.87e-07 | 1.21e-12 | 2.08e-11 |
| `kimi-linear` | 2.10e-10 | 9.84e-11 | 2.30e-14 | 5.57e-14 |

The 13 models that TF32 off alone leaves more than 10 times above the P106 (`bailingmoe3`, `deepseek2`,
`deepseek32`, `dots3note`, `glm-dsa`, `hy_v4`, `inkling2`, `kimi-k3`, `kimi-k31`, `kimi-linear`, `mistral-large4`,
`mistral4`, `xing4_0`) are models in which cuBLAS gets F16 sources and computed them in F16 (from the code; which of
each model's products, not traced). The compute type moves 16 models in all, these 13 and `grok`, `inkling2_loop` and
`t5`, and the 14 of them that have a P106 number end at or below 1.28 times it (`xing4_0`; `t5` barely moves,
2.18e-09 to 2.11e-09, and has no P106 number). `gemma4` is the largest in every
setting and on the P106 (1.42e-07): its own level, not the card's. `cohere2` gives 2.44e-14 here, the number C1039's
`notf32` set gave on the candidate with the same fusion.

## Result: the `MUL_MAT` replay

| Replay | Settings | Cases | OK | FAIL | NOREC | Seconds |
|---|---|---:|---:|---:|---:|---:|
| `before` | C1023's settings before C1108: the fork's matrix-vector kernels at a P106's shapes | 5,458 | 5,458 | 0 | 0 | 196 (from the disk) |
| `tf32` | those and `NVIDIA_TF32_OVERRIDE=0` | 5,458 | 5,458 | 0 | 0 | 71 |
| `after` | the driver's default now: those and `PASCAL_CUBLAS` | 5,458 | 5,458 | 0 | 0 | 47 |

No case's verdict differs between the three ([replay.tsv](replay.tsv), [replay-cases.tsv](replay-cases.tsv)). The
cases' errors against the CPU (the largest of each case's compared outputs; 4,314 cases carry them;
[replay-errors.tsv](replay-errors.tsv)):
- **TF32 off moves the F32 weights' products only,** 105 cases: the largest F32 error falls from 7.11e-08 to 3.08e-13
  at 1 to 8 tokens and from 6.93e-08 to 2.39e-13 at 9 or more. Every other case gives its `before` error to the digit.
- **The compute type moves 738 cases more,** the sources cuBLAS computed in F16 or BF16. F16 weights at 9 or more
  tokens: median 4.01e-07 to 1.67e-14 (all 79 cases), F16 with F16 activations 1.60e-07 to 9.88e-15, F8_E4M3
  3.97e-07 to 1.29e-14. The quantized products of 9 or more tokens keep their medians, and in the 297 of their 807
  cases that cuBLAS takes their largest errors fall (Q8_0 2.85e-05 to 1.62e-05, Q4_0 2.53e-05 to 1.46e-05).
- **BF16 weights at 9 to 128 tokens go the other way:** median 6.23e-15 to 2.16e-06 (all 43 cases). The card
  computed them in BF16, rounding the activations to BF16 as the CPU does, and a P106 does the same up to 128 tokens.
  This is the exception named above; [Q1901](../../../quants/experiments.md) takes the P106's exact rule into the
  fork.

## Result: the pipe check

Three passes on one card, each 130 verdict lines (five random models, layouts `one` and `virt`, the default arms
`f32-stock`, `q8-p106` and `mix-fn-c4@one`; [pipe.tsv](pipe.tsv)), 328, 320 and 321 seconds:
- **Every verdict is the same in all three:** 125 OK, 5 INFO (the kernels line). Pipelined decoding gives plain
  decoding's tokens, logits and states in every pass.
- **The settings reached every container:** each arm's `env.txt` lists `NVIDIA_TF32_OVERRIDE=0` in the `tf32` and
  `after` passes and `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` in `after`, and neither in `before`.
- **Where the plain run's numbers moved, they moved where cuBLAS computes.** With TF32 off the plain logits hash
  changed in 15 of 25 rows of `f32-stock` in each layout (the host, sampled and race checks of all five models: their
  F32 products with the prompt) and in 3 of `q8-p106` (`qwen4exp`, whose norms stay F32); with the compute type too,
  in 15 of `q8-p106` (the Q8_0 copies' dense products with the 550-token prompt, which cuBLAS takes). `mix-fn-c4`,
  which reads its prompt 4 tokens a call, changed nowhere. Only in the `after` pass do the plain tokens move, in the
  `flint` Q8_0 copy's sampled and race checks (both layouts alike), as a sampled reply does when its logits move.

## Which earlier checks this touches

The question for each earlier check on the build machine's 3090s (its GPU checks since 2026-11-20: the points of
the GPU-check queue and the code rows' card steps): did it compare numbers finer than TF32's rounding, about 1e-7 in
nmse, where a cuBLAS product sat on the path (an F32 one under TF32, or an F16 or quantized one computed in F16), and
was that rounding on one side of the comparison only? It is deterministic, so where both sides of a comparison
compute the same product the same way, both carry the same rounding and it cancels to first order. It matters where
one side is the processor, or where one side computes a product on cuBLAS and the other on a kernel that keeps F32 or
works in integers. "Est." marks what follows from the code and the setup, not from a run.

**Affected: the verdict or its sizes carry rounding a P106 does not have**

| Check | What it compared | Where TF32 sat | What it means for the verdict |
|---|---|---|---|
| [C1039](../2026-11-25_c1039_cohere2_nmse/README.md) (Q1772), sets `plain` and `f32` | each random model's card-against-CPU nmse (`test-llama-archs`), two builds | on the card side only: every F32 weight's product with the test's prompt | those sets put every model near 1e-7 and could not see `cohere2`'s move; C1039's verdict comes from its `notf32` set, which had TF32 off, and stands |
| [C1128](../2026-11-26_c1128_verify_drift/README.md) (Q1861), Flash-Next's verify drift | a four-token verify pass's next-token numbers against one-token passes, and C964's P106 flips against that drift | on the four-token side only: Flash-Next's router (`ffn_gate_inp`, 2,560 → 512, F32) runs on cuBLAS at four tokens (the F32 multi-token switch takes at most 32 rows since C807) and on `mul_mat_vec_f` at one | the drift's sizes (mean KL divergence 0.000870, median logit-gap move 0.0839, 8 of 768 top tokens changed) hold TF32's rounding of the router's inputs, which a P106 lacks; the comparison with C964's P106 moves ("2.4 times C964's move" at code row 124) is then not like with like. The rule's verdict (rounding everywhere, not a few positions) needs the run again with TF32 off: new row Q1900 (C1167) |
| [C981](../2026-11-25_c981_cuts1_flash_kld/README.md) (Q1714), `cuts=1` on Flash-Next | KL divergence of calls read in other pieces | partly one-sided, est.: where a piece of 4 to 6 tokens and a 64-token micro-batch compute the same product, the piece runs the shared expert's gate (`ffn_gate_inp_shexp`, one row, F32) on `mul_mat_vec_f` in F32 and the dense quantized weights on the matrix-vector kernels, and the micro-batch runs the gate on cuBLAS under TF32 and the dense quantized weights on cuBLAS in F16 (64 tokens is MMQ's limit on these cards; Q1901) | the pass stands, est.: the card's arithmetic adds rounding to the measured 0.004828 that a P106, which takes the matrix-vector kernels and MMQ, does not have, so a P106 would sit at or below it, against the same 0.028 limit. The size is the 3090's, not the P106's |

**Not affected, or to second order only**

| Checks | Why |
|---|---|
| The card suite on the 3090s: [C947](../2026-11-21_c947_v5_merged_gpu/README.md) (a), [C968](../2026-11-23_c968_pascal_mmvq_suite/README.md), [C991](../2026-11-25_c991_pascal_paths_suite/README.md), [C1023](../2026-11-24_c1023_suite_record/README.md), [C1024](../2026-11-24_c1024_mmq_j64_tile/README.md), [C1042](../2026-11-26_c1042_suite_records/README.md), [C1111](../2026-11-26_c1111_bf16_intermittent/README.md), [C1157](../2026-11-26_c1157_graph_src_types/README.md) and the landings' replays | `test-backend-ops` judges each case against the processor at thresholds far above this rounding (5e-4 in nmse for `MUL_MAT`); C1108's replay of the `MUL_MAT` share gives the same verdict in each of its 5,458 cases with cuBLAS as the card runs it, with TF32 off, and with `PASCAL_CUBLAS` (above) |
| Bit identity of the fork's kernels: [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) (a), C1138's four kernels ([IQ2_XXS](../2026-11-25_c1138_iq2xxs_mtok/README.md), [IQ2_S](../2026-11-26_c1138_iq2s_mtok/README.md), [IQ2_XS](../2026-11-26_c1138_iq2xs_mtok/README.md), [IQ1_M](../2026-11-26_c1138_iq1m_mtok/README.md)) | those kernels compute without cuBLAS |
| The same settings on two builds, or pipelined against plain decoding: points 1 (c) to 6, 8, 9 and 11 of the GPU-check queue ([C947](../2026-11-21_c947_v5_merged_gpu/README.md) to [C957](../2026-11-23_c957_batch_api_gpu/README.md)), the pipe step of every check since [C1009](../2026-11-24_c1009_pipe_step_cards/README.md), the landings' replies | both sides compute the same products on the same card, so a same-bits verdict holds. A change smaller than this rounding that sits before such a product can be rounded away on the 3090 where a P106 would show it (est. rare). C1108's pipe check gives the same verdicts in all three passes (above) |
| Ornith-1.5-122B's quality checks: [C976](../2026-11-23_c976_ornith_q8_cache_kld/README.md), [C989](../2026-11-24_c989_attn_error_floor/README.md), [C990](../2026-11-24_c990_q8_cache_long/README.md), [C993](../2026-11-24_c993_floor_measured/README.md), [C994](../2026-11-24_c994_attn_fault_selfref/README.md), [C1032](../2026-11-25_c1032_attn_yardstick/README.md) | each arm is scored against a reference read with the same pass size (prompt arms against a prompt reference, the decode arms against a decode reference), so the card's cuBLAS arithmetic sits on both sides. Their floors and yardsticks (micro-batches of 128 against 512, both above MMQ's 64-token limit; the attention kernel swapped; deliberate attention faults) change the order of the sums or the attention, not which products cuBLAS rounds: est. the same to first order, not measured. Measuring it would be new Ornith work, which the focus rule of 2026-11-25 does not queue |
| Replies that part at near ties between two settings on the 3090s: [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) (b), [C956](../2026-11-23_c956_taper_cuts_gpu/README.md), [C962](../2026-11-23_c962_sampled_reply_gpu/README.md), [C966](../2026-11-23_c966_kv_head_tail_gpu/README.md), [C987](../2026-11-24_c987_slot_cells_gpu/README.md) | the card's cuBLAS arithmetic in both arms; the arms differ in the kernels, the cache layout or the sampler order compared (C956's pieces: see C981 above) |

## New rows

- [Q1900](../../../quants/experiments.md) (C1167): C1128's run again with `PASCAL_CUBLAS` in the forced set, so
  Flash-Next's verify drift is measured without TF32 on its four-token side.
- [Q1901](../../../quants/experiments.md) (C1168): a fork switch that makes two more choices as on compute 6.1 on
  any card. The rig build on a card with FP16 tensor cores leaves a dense quantized product of 64 or more tokens to
  cuBLAS (`ggml_cuda_should_use_mmq` reads the card's compute capability) where a P106 takes MMQ; with `PASCAL_CUBLAS`
  cuBLAS computes those in F32, not F16, but still not the P106's way (from the code, not measured). And the P106's
  compute-type rule exactly, BF16 up to 128 tokens included, in place of `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`.

## Files

| File | What |
|---|---|
| [archs-runs.tsv](archs-runs.tsv) | per `test-llama-archs` run: setting, repeat, exit status, summary, card rows OK / FAIL / SKIP, CPU logits files written, compared and differing |
| [archs-nmse.tsv](archs-nmse.tsv) | per random model: the card-against-CPU nmse and the mixed-batch nmse of all ten runs, the P106's nmse (C912), and whether each setting's repeat gives its first number |
| [archs-summary.tsv](archs-summary.tsv) | per setting: the nmse's median and largest, models at or below 1e-12, and the ratio to the P106 per model (geometric mean, largest, models within 10 times) |
| [replay.tsv](replay.tsv) | per `MUL_MAT` replay: exit status, the passed line, cases, OK, FAIL, NOREC, seconds |
| [replay-cases.tsv](replay-cases.tsv) | the cases whose verdict differs between the replays (none: header only) |
| [replay-errors.tsv](replay-errors.tsv) | the replayed cases by weight type, activation type and token count: per replay the median and largest error, and how many cases' errors moved from the `before` replay's |
| [pipe.tsv](pipe.tsv) | per pipe check verdict line: each pass's verdict, whether the verdicts and the plain run's tokens are the same in all three, each pass's plain logits hash and whether it is the `before` pass's |
| [run/run.sh](run/run.sh) | the three steps on one card (detached, one PTX JIT cache shared) |
| [run/archs.sh](run/archs.sh) | `test-llama-archs` under the five settings, one container per run |
| [run/tables.py](run/tables.py) | every table above from the run's folder (standard library only) |

The raw logs are not published: they name the machine.

## Commands

```bash
B=<folder>/llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3   # the unpacked rig tarball
D=docs/data/2026-11-26_c1108_tf32_off
# on the build machine, with a copy of this repository at $REPO and C1023's kept record at $REC (holding mul_mat/):
setsid nohup env WORK=$OUT BIN=$B/bin GPU=2 REPO=$REPO REC=$REC JIT_SEED=<a PTX JIT cache of the same build> \
  PREFLIGHT=scripts/cuda_preflight.sh bash $REPO/$D/run/run.sh > $OUT/run.log 2>&1 < /dev/null &
# the tables, on any machine, from a copy of $OUT
python3 -I $D/run/tables.py $D $OUT docs/data/2026-11-24_c912_p106_check/logits.tsv
```

The run on 2026-11-26 called `run.sh` three times: all three steps with the `archs` step's four settings and the
replay's and pipe check's `before` and `tf32` passes (the `tf32` passes then ran under the tag `after` and were renamed);
`STEPS=archs` with the five settings; `STEPS='replay pipe' TAGS=after` once `PASCAL_CUBLAS` held both settings. Each
container carries the labels `pascal-quants=1` and `c1108=1`, `--cpu-shares=256` and one GPU.

Provenance: Miner Llama's fork and upstream llama.cpp (MIT); the random models are the ones `test-llama-archs` and the
pipe check generate with seed 1; the kept record is C1023's.
