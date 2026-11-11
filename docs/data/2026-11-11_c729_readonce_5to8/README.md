# Five to eight tokens on the read-once kernels (C729, Q1288, 2026-11-11)

What this is: a code change and its checks on a build machine without a GPU, and the rig run that will time it.
Qwen3.8-Flash-Next's dense projections are IQ4_XS, Q4_K, Q5_K, Q3_K and Q6_K products. For IQ4_XS, Q4_K (C722,
[R79](../2026-11-11_r79_c722_p106/README.md)) and Q5_K (C690, [R75](../2026-11-11_r75_c690_p106/README.md)) a kernel
that reads and decodes each weight once for every token, and gives each token the one-token kernel's sums bit for bit,
takes products of 2–4 tokens; on a P106 it took a verify pass's IQ4_XS and Q4_K products from 16.21 to 8.29 ms and the
output layer from 8.11 to 3.15 ms. Products of 5–8 tokens still go to the k-column kernel, whose sums are not the
one-token kernel's. On the line with the MTP draft every micro-batch of a planned split runs at least five tokens
(C678), so the draft line's prompt passes are five-token passes: there the projections took 39.00 ms of GPU time against
34.13 at four tokens ([R41](../2026-11-10_release_profile/README.md)). A draft of four or more tokens verifies five to
eight.

The code is on the fork's branch `pascal-p106-v4-b12246-c729` (`3f166ab56`, one commit on
`pascal-p106-v4-b12246-day` `18d0fa309`), behind its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_RO8`
([docs/switches.md](../../switches.md)). Registry row [Q1288](../../../quants/experiments.md).

## Result

**Every check that runs without a GPU passes, at every token count from 5 to 8: the 24 new kernels hold 95 to 167
registers with no stack, no local memory and no spill instruction; every other function of the build keeps
`18d0fa309`'s code (9,007 of 9,007 in 195 objects, 8,902 of 8,902 in `llama-server`); at 1–4 tokens the host launches
what `18d0fa309` launches under 16 settings, and at 5–8 tokens only under the switch, always at one row per lane
group; with the switch off, and on, the build computes what `18d0fa309` does on 146 of 146 random models; ctest 70 of
70.** No token count had to be left out: at 8 tokens with the prefetch the largest kernel (Q5_K) holds 167 registers,
under the launch bounds' 168, and every launch's occupancy is set by shared memory (two blocks a SM), not by registers,
which would allow 12 warps a SM (65,536 / (32 × 168)) where two blocks hold at most 12. The native bit identity and the
timing are the rig run's ([R116](#the-rig-run-r116)).

- **The new instantiations** ([sass/sass_kernels.tsv](sass/sass_kernels.tsv), CUDA 12.9's `cuobjdump`, sm_61), one
  super-block step per lane, with the 4-token kernel for comparison:

  | Kernel | Tokens | Registers (without / with the prefetch) | dp4a | Shared loads | FFMA | FADD | I2F |
  |---|---:|---|---:|---:|---:|---:|---:|
  | IQ4_XS | 4 | 92 / 127 | 256 | 114 | 68 | 20 | 1 |
  | IQ4_XS | 5 | 95 / 135 | 320 | 138 | 85 | 25 | 1 |
  | IQ4_XS | 6 | 96 / 136 | 384 | 163 | 102 | 30 | 1 |
  | IQ4_XS | 7 | 103 / 140 | 448 | 187 | 119 | 35 | 1 |
  | IQ4_XS | 8 | 108 / 142 | 512 | 210 | 136 | 40 | 1 |
  | Q4_K | 4 | 96 / 136 | 264 | 105 | 104 | 36 | 2 |
  | Q4_K | 5 | 104 / 141 | 328 | 129 | 130 | 41 | 2 |
  | Q4_K | 6 | 111 / 143 | 392 | 153 | 156 | 46 | 2 |
  | Q4_K | 7 | 114 / 151 | 456 | 177 | 182 | 51 | 2 |
  | Q4_K | 8 | 117 / 155 | 520 | 201 | 208 | 56 | 2 |
  | Q5_K | 4 | 139 / 160 | 264 | 107 | 104 | 36 | 2 |
  | Q5_K | 5 | 136 / 160 | 328 | 131 | 130 | 41 | 2 |
  | Q5_K | 6 | 144 / 161 | 392 | 155 | 156 | 46 | 2 |
  | Q5_K | 7 | 148 / 164 | 456 | 179 | 182 | 51 | 2 |
  | Q5_K | 8 | 151 / 167 | 520 | 203 | 208 | 56 | 2 |

  Each token adds 64 dp4a (the 8 more of Q4_K and Q5_K are the staging loop's block sums), 23 to 25 shared loads of its
  activations and 17 (IQ4_XS) or 26 FFMA to a step whose weight loads and decoding stay those of one token; the k-column
  kernel reads every token's activation record again for every row. The I2F are outside the loop (the staging loop's
  division, and for Q4_K and Q5_K the block sums' conversion). The last FFMA of every Q4_K and Q5_K instantiation
  multiply by `-dmin`, the one-token kernel's last operation on a super-block (`acc = FFMA(summ, -dmin, FFMA(sumd, d,
  acc))`).
- **Every other kernel unchanged** ([sass/sass_objects.tsv](sass/sass_objects.tsv),
  [sass/sass_binary.tsv](sass/sass_binary.tsv), [sass/sass_differ.txt](sass/sass_differ.txt)): CUDA 12.9's
  `cuobjdump` of this rig build and of `18d0fa309`'s, built the same way on the same machine. 9,007 of 9,007 functions
  in 195 objects the same SASS and 193 of 195 objects the same PTX: the two whose PTX differs are the two kernels'
  objects, which hold the new instantiations; the 40 earlier instantiations of the two kernels keep their code and
  registers. `bin/llama-server`: 8,902 of 8,902 kernels the same, the 24 new ones besides.
- **The launch choices** ([pick/](pick/), `run/pick.cu` built against each tree's own two files and run under 16
  settings, [pick/arms.txt](pick/arms.txt)): at 1–4 tokens the tip's 10,240 lines are the base's under every setting,
  the switch's included. At 5–8 tokens the base takes nothing, and the tip takes 215 of the 768 row widths (3 types ×
  4 token counts × 1 to 64 super-blocks) under each setting with the switch, always at R = 1, and none without it; the
  other 553 are left to the k-column kernel (48 KB, or IQ4_XS's odd numbers of super-blocks).
  [pick/ro8_launch.tsv](pick/ro8_launch.tsv) lists every one with its warps, shared memory and kernel,
  [pick/flashnext.tsv](pick/flashnext.tsv) Flash-Next's two row widths at 4–8 tokens.
- **Off = the base:** `test-llama-archs -b CPU -s 1` gives `18d0fa309`'s logits on 146 of 146 random models (151 tests
  pass), with no switch and with `GGML_CUDA_MMVQ_PASCAL_RO8=1` (a CUDA switch, so the CPU results must not move);
  ctest 70 of 70 (the static CPU build; the tests that need a download, the network or torch left out, as in C565)
  ([summary.tsv](summary.tsv)).
- **The replay set `ro8`** ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)): 259 cases (every
  IQ4_XS, Q4_K and Q5_K `MUL_MAT` case of 5–8 tokens: Flash-Next's shapes, the output layer's 248,320 × 2,560, each lane
  count, partial row blocks, views of wider rows, the widths the kernels leave to the k-column kernel, three large-
  activation cases), recorded with this rig build's `test-backend-ops` on the build machine's CPU, 2,546,042,880 B, md5
  `e4839b4ed92fa24c07fb32b31eb1cfc9`; 13 runs: the switch at each setting, with the 2–4-token switches, without the
  preset's k-column switches, off, Q4_K or every type left out, and the fault, which must fail cases.
- **The bit-identity harness** `multitoken_bitid` (new modes `dmt8` and `q5k8`) links against this build, md5
  `28862d6311a217e511e673a9379bd1cf`; both read-once tests it asks are the build's own (`nm`: 2 of 2 defined).
- **The rig tarball:** `llama.cpp-20261111-3f166ab56-pascal-p106-v4-b12246-c729.tar.gz`, 674,193,830 B, md5
  `7de99457a7cac7a5b5b21b8c2a719e39` (`TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38),
  `test-backend-ops` md5 `fbeb3be306a55b69457d7237829f4bdc`.

**What it should win, est.:** in R79's served verify pass the same kernels cut the IQ4_XS and Q4_K products from
16.212 to 8.291 ms at four tokens, where four tokens took 1.23 times one token's time at K = 2,560 (IQ4_XS; Q4_K 1.14).
Each token more adds 64 dp4a and about 24 shared loads per super-block to the read-once kernel, and a whole pass of
activation reads per row to the k-column kernel. So est. the five-token pass's IQ4_XS, Q4_K and Q5_K products at
K = 2,560 drop as the four-token pass's did, to about half; at K = 6,144 five tokens run at 1 to 3 warps a block, which
`perf` must show is not a loss. The rig run decides, product by product.

## The change

- **The kernels at 5–8 tokens.** `mul_mat_vec_q_pascal_dmt<type, NT, R, PF>` (`mmvq-pascal-dmt.cu`, C722) and
  `mul_mat_vec_q5_K_pascal_mt<NT, R, PF>` (`mmvq-pascal-q5k.cu`, C690) get 24 more instantiations: NT = 5 to 8 at R = 1,
  with and without the register prefetch, for IQ4_XS, Q4_K and Q5_K. The code is the same templates with more
  columns: no line of either kernel changed. R = 2 (two rows of a lane group sharing the activation loads) is not
  built for them: on a P106 it was slower than R = 1 at every 2–4-token shape (R75, R79), and it holds twice the
  accumulators.
- **The switch.** `ggml_cuda_mmvq_pascal_dmt_takes` and `ggml_cuda_mmvq_pascal_q5k_mt_takes` also take products of 5–8
  columns when `GGML_CUDA_MMVQ_PASCAL_RO8` is 1 or 2, and then the hooks that send 2–4-token products to the kernels
  send these: `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`) runs them after quantizing the activations, and the k-column kernels
  leave the products they take to them (`mmvq-pascal-kcol.cu`, the tests that were there for 2–4 tokens). The switch
  does not need `GGML_CUDA_MMVQ_PASCAL_DMT` or `_Q5K_MT`, which keep 2–4 tokens.
- **Its settings.** One row per lane group always (`_DMT_R` and `_Q5K_MT_R` act on 2–4 tokens only). `=1`: the register
  prefetch as each kernel's own rule has it (IQ4_XS and Q4_K with it, Q5_K only under `_Q5K_MT_PF=1`); `=2`: without it.
  `_DMT_NW`, `_Q5K_MT_NW` (warps per block) and `_DMT_SKIP` apply as at 2–4 tokens.
- **The fault** `GGML_CUDA_MMVQ_PASCAL_RO8_FAULT=1`: the host gives the kernels a column stride of 0 for the products of
  5–8 tokens, so every token stages the first token's activations. It changes no kernel; the replay and the
  bit-identity run must catch it, which shows that the switch sends those products to the kernels.
- **Off, nothing changes:** with the switch at 0 both tests return false for 5–8 columns as before, and no earlier
  kernel's code changes (below). `mmvq.cu`, `mmvq-pascal-kcol.cu` and the two headers change only in comments, with the
  same line counts, so no device-code line moves.
- **`test-backend-ops`:** the Q5_K cases of C690 and the IQ4_XS and Q4_K cases of C722 run at 1–8 tokens (the k-column
  cases already held 2,560 × 6,144 at 1–8).

### Why each token's result is the one-token kernel's, bit for bit

Nothing in the kernels depends on the token count but the loops over it: each token's activations are staged in their
own slots, and a token's sums are formed exactly as at 2–4 tokens, which C722 and C690 built to be the one-token
kernel's (the same lanes per row, the same lane-to-super-block order and xor shuffles, the same FP32 operations on the
same operands, the integer-to-float conversions exact without I2F). On a P106 that held for 2–4 tokens at every
setting (R79: 0 of 60 comparisons differ; R75: 0 of 21). At one row per lane group a row's sums do not depend on the
other rows of its group either.

### Which products it takes

A block stages every token's activations whole in shared memory, beside each warp's weight staging, within 48 KB.
At five to eight tokens the activations take about NT / 4 times their four-token size, so fewer warps fit, and wide rows
not one. Warps per block and the block's shared memory in bytes at Flash-Next's two row widths (`run/pick.cu` on the
tip; 4 tokens with `_DMT=1` and `_Q5K_MT=1`, 5–8 with `_RO8=1`); every launch holds two blocks per SM (96 KB):

| Type | Row width K | 4 tokens | 5 tokens | 6 tokens | 7 tokens | 8 tokens |
|---|---:|---|---|---|---|---|
| IQ4_XS | 2,560 | 6 (37,840) | 6 (40,720) | 6 (43,600) | 6 (46,480) | 5 (45,008) |
| IQ4_XS | 6,144 | 4 (45,536) | 3 (48,096) | 1 (46,304) | k-column | k-column |
| Q4_K | 2,560 | 6 (40,688) | 6 (43,888) | 6 (47,088) | 5 (45,680) | 5 (48,880) |
| Q4_K | 6,144 | 3 (45,120) | 2 (48,192) | k-column | k-column | k-column |
| Q5_K | 2,560 | 6 (46,832) | 5 (44,400) | 5 (47,600) | 4 (45,168) | 4 (48,368) |
| Q5_K | 6,144 | 3 (48,192) | 1 (44,608) | k-column | k-column | k-column |

The widest rows taken, in super-blocks of 256 weights at 5, 6, 7 and 8 tokens: IQ4_XS 30, 24, 22 and 18 (an even
number: its 136-byte super-blocks need two lanes per row), Q4_K 27, 22, 19 and 17, Q5_K 26, 22, 19 and 16.
In Flash-Next the 2,560-wide rows are `attn_qkv`, `attn_gate`, `attn_q`, `attn_k`, `attn_v`, the shared expert's gate
and up, and the output layer; the 6,144-wide ones are `attn_output` and `ssm_out`.

`run/pick.cu` prints every row width from 256 to 16,384 weights ([pick/ro8_launch.tsv](pick/ro8_launch.tsv)). Rows
that do not fit stay on the k-column kernel; C728's K tiles (Q1287) are what would take them.

## Checks on the build machine (no GPU)

[run/checks.sh](run/checks.sh), every build and test in a container at low CPU priority, one at a time under a lock
shared with the machine's other sessions ([summary.tsv](summary.tsv)):

| Step | What | Verdict |
|---|---|---|
| base | `18d0fa309`'s rig build (`TESTS=1`); its CUDA objects and `llama-server` kept for the SASS step | OK, 195 objects |
| build | this branch's rig tarball | OK, md5 `7de99457…` |
| sass | every function of both builds' 195 objects and `llama-server`; the read-once kernels' instantiations | SAME; the 24 new ones CLEAN (no stack, local memory or spill) |
| pick | `run/pick.cu` against both trees under 16 settings | 1–4 tokens SAME; 5–8 OK |
| bitid | `multitoken_bitid` against this build | OK, md5 `28862d63…` |
| cpu, off | static CPU builds of both; `test-llama-archs -b CPU -s 1`, the base writes, this branch compares, without and with the switch | SAME, 146 of 146 each |
| ctest | this branch's CPU build | 70 of 70 |
| record | the replay set `ro8` | OK, 259 cases, 13 runs |

The first `pick` run stopped at its own sanity check: it expected 4 lines of the code `pick.cu` repeats in each tree,
and each tree has 5 (Q5_K's R is read in two places, the warps' count and the launcher). The count was corrected and
the step run again; its arms are the same files. A first launch of the whole script was stopped by the launching
command itself (a `timeout` around the detached start) 20 s in; it was started again from the top, so every verdict
above comes from one complete run.

## The rig run (R116)

[run/rig_c729.sh](run/rig_c729.sh) on card 0 of an idle rig, detached, with the branch's tarball, the `ro8` set and the
harness; then [run/c729_r42.sh](run/c729_r42.sh) from the machine that drives the rig. Its steps:

- **replay:** the recorded set through `scripts/replay_checks.sh`. Passing: every run's line ends `ok` (`ro8-fault`'s
  `ok` means it failed cases, as it must).
- **bitid:** `multitoken_bitid dmt8` and `q5k8`: every 5–8-token product the kernels take against its columns' one-token
  products, at `RO8=1`, `=2`, with Q5_K's prefetch and with `DMT=1 Q5K_MT=1` too: `0 with differences` each; `dmt` and
  `q5k` (2–4 tokens) with all three switches: 0 (the switch must not reach them); the control (`RO8=0`, `BITID_ALL=1`:
  the k-column kernel's sums) and the fault (`RO8_FAULT=1`) must differ (exit 1). The products a kernel declines are
  printed as declined and not counted; the build itself says which (its own `ggml_cuda_mmvq_pascal_*_takes`).
- **perf:** `test-backend-ops perf` at Flash-Next's shapes of the three types (IQ4_XS and Q4_K 2,560 → 10,240, 12,288
  and 6,144, 6,144 → 2,560; Q5_K the output layer, 6,144 → 2,560 and 2,560 → 512), 1–8 tokens, arms `off` (the k-column
  kernel at 2–8 tokens), `mt` (`DMT=1 Q5K_MT=1`), `on`, `nopf` and `pf` (`mt` with `RO8=1`, `=2`, `=1` and Q5_K's
  prefetch), two passes alternated (`perf/us_run.tsv`). The 1-token cases are the one-token kernel in every arm, the
  2–4-token ones the same kernels in `mt`, `on`, `nopf` and `pf`: they give the spread.
- **prompt:** R42's driver (`run/c729_r42.sh`, three loads per arm) with `DMT=1 Q5K_MT=1` in both arms and `RO8` (the
  perf step's faster of `=1` and `=2` at five tokens) in `on`, then R42's `timeline.py` per arm on the build machine.
  With the MTP draft every micro-batch of a planned split runs at least five tokens (C678), so the fresh 4,097-token
  prompts with the taper on hold five-to-eight-token passes; the graph profile shows how many.

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting (and the fault caught), and the replay passing, say that
  the switch gives every 5–8-token product it takes the one-token kernel's bits on a P106. Any difference is a fault to
  find before any timing counts.
- **Gain, negligible or loss** per shape and token count: `on` against `mt` (and `off`) in each pass, by value; the
  spread is the passes' own. At K = 6,144 five tokens run at 1 to 3 warps a block: a loss there goes to `_DMT_SKIP` or a
  width rule, and to C728.
- **In the prompt lane:** each arm's 4,097-token prompt windows with the taper on and off, each load by value; with the
  taper off the prompt runs in large micro-batches, which the switch does not take, so those windows must not move.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder on the
rig. The model, the draft and R42's and R43's files are R82's.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail, in the order they ran |
| [sass/sass_kernels.tsv](sass/sass_kernels.tsv) | every instantiation of both read-once kernels: code hash and registers in both builds; stack, local memory, spill instructions and instruction counts in this one; the last two FFMA |
| [sass/sass_objects.tsv](sass/sass_objects.tsv), [sass/sass_binary.tsv](sass/sass_binary.tsv), [sass/sass_differ.txt](sass/sass_differ.txt), [sass/compare_sass.txt](sass/compare_sass.txt) | per object and for `llama-server`: functions the same, differing, in one build only; every function that is new, each expected |
| [pick/arms.txt](pick/arms.txt), [pick/ro8_launch.tsv](pick/ro8_launch.tsv), [pick/ro8_nopf_launch.tsv](pick/ro8_nopf_launch.tsv), [pick/flashnext.tsv](pick/flashnext.tsv) | the settings `pick.cu` ran under; what the tip launches at 4–8 tokens with `RO8=1` and 5–8 with `=2`, every row width; Flash-Next's two widths |
| [replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv) | the `ro8` set's cases and runs |
| [run/checks.sh](run/checks.sh), [run/pick.cu](run/pick.cu), [run/compare_sass.py](run/compare_sass.py) | the checks |
| [run/rig_c729.sh](run/rig_c729.sh), [run/c729_r42.sh](run/c729_r42.sh) | the rig run |

Also changed for this row: [scripts/record_checks.sh](../../../scripts/record_checks.sh) (suite `ro8`),
[bench/mmvq_pascal/verify/multitoken_bitid.cpp](../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp) (modes `dmt8`
and `q5k8`).

## Commands

```bash
# on the build machine: $WORK/rig/src a llama.cpp clone with the fork as remote "fork" (https fetch URL), $REPO this
# repository; every step waits for LOCK
WORK=$W REPO=$W/repo BRANCH=pascal-p106-v4-b12246-c729 LOCK=<flock file> JOBS=8 \
    setsid -f nohup bash $W/repo/docs/data/2026-11-11_c729_readonce_5to8/run/checks.sh > $W/logs/checks.out 2>&1 < /dev/null
# afterwards: the build trees, CPU trees, ccaches and object dumps removed (the tarball, the set and the harness stay)
WORK=$W REPO=$W/repo BRANCH=pascal-p106-v4-b12246-c729 STEPS=clean bash $W/repo/docs/data/2026-11-11_c729_readonce_5to8/run/checks.sh
```

```bash
# on the rig, the tarball unpacked in $R, the set's tar unpacked in $R/set, the harness in $R
BIN=$R/llama.cpp-20261111-3f166ab56-pascal-p106-v4-b12246-c729/bin OUT=$R/out SET=$R/set/ro8 \
    REPLAY=$R/scripts/replay_checks.sh BITID=$R/multitoken_bitid \
    setsid -f nohup bash $R/rig_c729.sh > $R/console.txt 2>&1 < /dev/null
# from the machine that drives the rig, after the perf step: off, then on
RIG=$RIG WORK=$R BUILDS=<builds folder> BUILD=llama.cpp-20261111-3f166ab56-pascal-p106-v4-b12246-c729 MODELS=<models> \
    R42=<r42 folder> R43=<r43 folder> ARM=off bash docs/data/2026-11-11_c729_readonce_5to8/run/c729_r42.sh
```

## Provenance

Code and scripts are this repository's and the fork's (MIT, as llama.cpp). No model or data file is involved: the
checks use random weights and activations (`test-backend-ops`, `test-llama-archs`, `multitoken_bitid`).
