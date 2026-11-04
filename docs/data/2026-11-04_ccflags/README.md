# Compiler flags and launch bounds for the Pascal kernels, before any assembly (2026-11-04)

Registry rows Q671 and Q672 in [quants/experiments.md](../../../quants/experiments.md), the first candidate of the
assembly survey ([assembly_optimization.md](../../research/notes/assembly_optimization.md) §8). The question: what do
`-Xptxas -dlcm=ca`, register budgets (`-maxrregcount`, `__launch_bounds__`) and two more compiler options change in the
machine code of the fork's Pascal kernels, which of those changes could pay on a P106-090, and does ptxas turn an FP16
conversion into FP16 arithmetic anywhere on the decode path?

**Status.** Every Pascal kernel file of `pascal-p106-v3-b12076-ladder2` (`a330672c1`) was compiled to an sm_61 cubin with
the rig build's own flags in 27 arms (the build's flags, each flag and register arm, and launch-bound variants), and the
census below is of the machine code. Nothing ran on a GPU: the P106 timing of the five candidates is written
([run/p106_timing.sh](run/p106_timing.sh)), its binaries are built, and it runs next.

## What was compiled

- **Files (15, 552 kernels):** `mmvq-pascal.cu` (277 one-token and 2–4-token matrix-vector kernels, with
  `mmvq-pascal-impl.cuh`), `mmvq-pascal-kcol-<type>.cu` for the seven k-column types and `mmvq-pascal-kcol.cu` (119),
  `mmvq-pascal-mtok.cu` (12, the Q2_0 multi-token expert kernels), `mmvf-pascal.cu` (108, BF16), `fattn-pascal.cu` (1,
  decode attention), `fattn-tile-pascal.cu` (14, prompt attention), `gdn-fused.cu` (8) and `moe-persistent.cu` (13, Flint's
  persistent MoE kernel). [run/files.tsv](run/files.tsv) maps each file to its family and block size.
- **Flags:** the 64 arguments of the build's `compile_commands.json` (CUDA 12.9.2, `nvcc` and `ptxas` 12.9.86): `-O3
  -DNDEBUG -std=c++17 -use_fast_math -extended-lambda`, the `GGML_CUDA_FA_*` macros, `--generate-code` for compute 6.1,
  with `-cubin` and the single sm_61 target instead of the fatbin and `-Xptxas -v` added ([run/extract_flags.py](run/extract_flags.py)).
- **Baseline check:** all 552 kernels of the base arm carry exactly the machine code of the same kernels in the rig
  build's own objects ([baseline_check.tsv](baseline_check.tsv)), so the arms differ from the release code only by
  their flag.
- **Arms** ([run/arms.tsv](run/arms.tsv)): `base`; `dlcm_ca`, `dlcm_cg` (`-Xptxas -dlcm=ca` and `=cg`); `mrr64`, `mrr96`,
  `mrr128` (`-maxrregcount`); `restrict` (`--restrict`); `xvec` (`--extra-device-vectorization`); `iqmb2`, `iqmb4`,
  `mtokmb2`, `mtokmb4` (the launch-bound macros the tree already had, for the grid types and the multi-token kernels); and,
  from the fork branch `pascal-p106-v3-b12076-ccflags` (below), `base2` (its defaults, which must equal `base`),
  `mmvq_mb2/3/4`, `iqmc2`, `kcol_mb1/3/4`, `bf16_mb2/3/4`, `fap_mb2/4` and `ftp_mb2/4`.
- **Per kernel and arm** ([kernels.tsv.gz](kernels.tsv.gz), 7,365 rows): registers, stack frame and local bytes
  (`cuobjdump -res-usage`), spill stores and loads in bytes (`ptxas -v`), `STL` and `LDL` instructions (the stack frame
  and the spills together: ptxas' spill line does not count a stack frame), every load mnemonic with its modifiers and
  count, instruction count, FP16 arithmetic and conversion counts, and the theoretical occupancy on a 5-SM GP106: resident
  blocks of the file's thread count per SM by registers (allocated per warp in units of 256), static shared memory, 2,048
  threads and 32 blocks, and active warps over 64. Dynamic shared memory (the staged and tiled shapes) is not in it and
  can lower the figure at launch.
- **Roll-ups:** [summary_family_arm.tsv](summary_family_arm.tsv) per file family; [groups.tsv](groups.tsv) per kernel
  group (the grid types' 2–4-token kernels, the Q2_0 kernels by token count, the k-column kernels by type, ...);
  [load_forms_family_arm.tsv](load_forms_family_arm.tsv) every load form; [changes_vs_base.tsv.gz](changes_vs_base.tsv.gz)
  every kernel an arm changed in registers, stack, local memory, loads or blocks per SM; [half_math.tsv](half_math.tsv)
  FP16 arithmetic and conversions per file and arm; [compile_times.tsv](compile_times.tsv).

## What the compiler flags do

**Load instructions.** The Pascal files' loads are `LDG.E.CI` (the read-only data path, `ld.global.nc` from `__ldg`: all of
the k-column and BF16 kernels' loads, the weights of the others) and plain `LDG.E` (activations, scales, ids). Counts for
the base arm, from [load_forms_family_arm.tsv](load_forms_family_arm.tsv):

| Family | `LDG` | of them `.CI` | of them 128-bit |
|---|---|---|---|
| Pascal quantized (1–4 tokens) | 23,944 | 14,549 | 9,587 |
| multi-token Q2_0 experts | 2,420 | 870 | 294 |
| k-column | 15,575 | 15,575 | 15,568 |
| BF16 | 3,924 | 3,924 | 3,879 |
| decode attention / prompt attention / GDN | 102 / 394 / 129 | 40 / 294 / 129 | 15 / 66 / 2 |
| persistent MoE | 6,822 | 5,413 | 811 |

- **`-Xptxas -dlcm=ca` changes no load instruction.** Plain loads stay `LDG.E` with and without it, the `.CI` loads are
  the same, and `-Xptxas -dlcm=cg` is the setting that changes them: all 9,395 plain `LDG.E` of `mmvq-pascal.cu` become
  `LDG.E.CG` (L2 only), nothing else moves. On this ptxas the instruction-level default is the `ca` form.
- **`-dlcm=ca` does change other things**: 30 kernels of `mmvq-pascal.cu`, 7 of the multi-token kernels, 42 BF16
  kernels, 6 of the prompt attention kernels and the decode attention kernel come out with another schedule and register
  allocation. The big change is in the grid types' 2–4-token kernels (IQ2_S, IQ3_XXS) and the fused Q2_0 gate/up
  kernels of 4 and 8 tokens: they drop to 123–128 registers and from 3 to 4 blocks per SM, with more local-memory instructions (IQ3_XXS 1,213 to
  3,169, IQ2_S 1,467 to 3,246): the same registers and local-memory counts as launch bounds of 4 blocks per SM (`iqmb4`),
  not a cache effect. The BF16 kernels' change is scheduling only (the same loads).
- **`--restrict` and `--extra-device-vectorization` change nothing** in these kernels (0 of 277, 12, 108, 119, 1, 8
  kernels differ for `xvec`; for `restrict` 6 prompt-attention kernels and 1 persistent-MoE kernel are reordered with the
  same instruction counts): the kernels already use `__restrict__` and explicit 128-bit loads.

## Register budgets

- **`-maxrregcount` is not a cap for a kernel with `__launch_bounds__`.** Every Pascal kernel except the GDN fusions
  carries launch bounds: with `-maxrregcount` 64, 96 and 128 their maximum registers stay at 255 (`mmvq-pascal.cu`),
  244 (k-column) and 254 (BF16), and the kernels that grow past 128 still do. The flag acts as a scheduling target: the
  k-column kernels take 3.2 % fewer instructions at 128 (Q2_0 −5.1 %, IQ4_XS −3.7 %, Q3_K −2.6 %) with 9 more registers
  on average (blocks per SM 4.51 to 4.30), the grid types' 2–4-token kernels 7.3 % (IQ2_S) and 4.4 % (IQ3_XXS) fewer, the
  other one-token kernels 1.0 % and the BF16 kernels 0.4 % fewer. Only the GDN fusions, which have no launch bounds, are
  capped (at most 64 registers at 64; they use more registers under the flag than without, 49 to 57 on average).
- **The register budget of a Pascal kernel is its launch bounds.** The fork branch `-ccflags` makes the minimum-blocks
  argument of each kernel's `__launch_bounds__` a macro with today's value as default, and the grid types' 2–4-token
  kernels get a macro apart from their one-token kernels, so a `-D` flag moves it ([groups.tsv](groups.tsv)):

| Kernel group (arm) | Registers (mean / max) | Local-memory instructions | Blocks per SM | Instructions |
|---|---|---|---|---|
| IQ3_XXS, 2–4 tokens: base | 168 / 168 | 1,213 | 3 | 38,316 |
| IQ3_XXS, 2–4 tokens: `iqmc2` (2 blocks) | 253 / 255 | 0 | 2 | 35,400 (−7.6 %) |
| IQ3_XXS, 2–4 tokens: `iqmb4` (4 blocks) | 128 / 128 | 3,169 | 4 | 41,430 (+8.1 %) |
| IQ2_S, 2–4 tokens: base | 168 / 168 | 1,467 | 3 | 39,918 |
| IQ2_S, 2–4 tokens: `iqmc2` | 255 / 255 | 12 | 2 | 35,208 (−11.8 %) |
| IQ2_S, 2–4 tokens: `iqmb4` | 128 / 128 | 3,246 | 4 | 42,738 (+7.1 %) |
| grid types, one token: `iqmc2` | unchanged code | | | |
| other types, 2–4 tokens: base / `mmvq_mb2` / `mmvq_mb3` | 164 / 163 / 144 (mean) | 64 / 80 / 4,860 | 3.07 / 3.05 / 3.35 | 349,404 / −0.3 % / +2.6 % |
| other types, one token: base / `mmvq_mb3` | 117 / 108 (mean) | 1 / 59 | 4.56 / 4.72 | 71,274 / −0.2 % |
| Q2_0 experts, fused gate/up 8 tokens: base / `mtokmb4` | 152 / 128 (mean) | 0 / 92 | 3 / 4 | 35,748 / +0.4 % |
| k-column, Q3_K: base / `kcol_mb3` / `kcol_mb4` | 165 / 148 / 124 (mean) | 0 / 34 / 214 | 2.88 / 3.25 / 4.06 | 39,618 / +0.6 % / +3.4 % |
| BF16 (all): base / `bf16_mb2` / `bf16_mb3` | 93 / 140 / 125 (mean) | 0 / 0 / 725 | 6.99 / 4.41 / 4.77 | 101,028 / −0.3 % / +1.0 % |
| prompt attention, Q8_0 head-128 split kernel: base / `ftp_mb2` | 168 / 196 | 20 / 0 | 3 / 2 | 17,142 (all 14 kernels) / −0.2 % |

  Without a minimum-blocks argument ptxas picks the BF16 kernels' registers itself (93 on average, 7 blocks per SM); an
  explicit 2 gives it room it uses (140). A budget of 3 or 4 blocks per SM costs local memory almost everywhere except
  where the code already fits (the grid types' one-token kernels, the Q2_0 kernels below 8 tokens).

## Local memory in the Pascal kernels (base)

35 of the 552 kernels keep a stack frame or spill (`STL`/`LDL`; ptxas' spill bytes miss the frames):

| Kernels | Count | Registers | Stack frame | `STL` + `LDL` |
|---|---|---|---|---|
| IQ3_XXS, 2–4 tokens (`mul_mat_vec_q_pascal_mc`) | 9 | 168 | 88–208 B | 1,213 |
| IQ2_S, 2–4 tokens | 9 | 168 | 120–224 B | 1,467 |
| Q8_0 (one token), IQ4_XS, Q5_K, Q4_K and Q4_1 (2–4 tokens) | 7 | 255 | 8–48 B | 65 |
| decode attention (`flash_attn_ext_pascal_f16`) | 1 | 134 | 128 B | 433 (Q675) |
| prompt attention, `<128, Q8_0, Q8_0, split>` | 1 | 168 | 40 B | 20 (Q676) |
| persistent MoE (Flint) | 8 | 128 | 16–48 B | 157 |

The k-column, BF16, multi-token Q2_0 and GDN kernels have none. The grid types' 2–4-token frames are what registry Q714's
second part named; `iqmc2` removes them.

## FP16 arithmetic on the decode path (Q672)

[fp16_sites.tsv](fp16_sites.tsv) lists every FP16-arithmetic site of `-ladder2` (the 7,682-function census of
[2026-11-04_ladder2_merge](../2026-11-04_ladder2_merge/README.md#q672-the-sm_61-fp16-conversion-trap) plus this folder's
per-file counts) with where it is chosen and what a compute 6.1 build runs instead:

- **None executes on a P106.** The F16-accumulating `mul_mat_vec_f<half, half>` (1,480 `HFMA2`/`HMUL2` in 88 kernels)
  and the persistent MoE kernel's F16 router accumulation (384 in 12 kernels) are chosen only where
  `fast_fp16_available(cc)`, which is false for a binary compiled for 6.1 alone; the dispatch sets the precision to F32
  first (`mmvf.cu` lines 660 and 752, `moe-persistent.cu` line 1036), and the kernels that run are the `__half22float2`
  forms: two `F2F.F32.F16` and two `FFMA` per half2. The F2F form is already the one in use.
- **BF16 needs no conversion instruction:** the 108 BF16 kernels hold no `F2F` and no `HADD2` (the value is the upper half
  of an FP32 one).
- **Block scales:** 5,188, 271 and 204 `F2F.F32.F16` in the quantized, k-column and multi-token files, about 1 instruction
  in 100 of `mmvq-pascal.cu`; activation sums are FP32. No `HADD2.F32` conversion exists in any arm.
- **`dequantize_block_q8_0_f16`** (64 `HMUL2` per kernel) is called by `ggml_get_to_fp16_cuda(Q8_0)`, which the generic
  attention kernels use to turn a Q8_0 K/V cache into F16 and the convolution ops use. On a P106 a Q8_0 cache is read
  directly (vector kernel for 1–2 tokens, the Pascal prompt kernel for 3 and more at head 128 or 256), and cuBLAS
  products of a quantized matrix compute in F32 on compute 6.1 (`ggml-cuda.cu` line 1897), through the F32 dequantizer.

## The fork branch

`pascal-p106-v3-b12076-ccflags` (one commit on `-ladder2`): the launch-bound macros (`MMVQP_MIN_BLOCKS`,
`MMVQP_MC_MIN_BLOCKS`, `MMVQP_IQ_MC_MIN_BLOCKS`, `KCOL_MIN_BLOCKS`, `MMVF_PASCAL_MIN_BLOCKS`, `FAP_MIN_BLOCKS`,
`FTP_MIN_BLOCKS`; defaults = today's values, and with them all 552 kernels' machine code equals `-ladder2`'s: arm `base2`
against `base`, 0 kernels differ) and `GGML_CUDA_PASCAL_FLAGS`, a cmake option that adds nvcc options to the Pascal kernel
files only, so a flag A/B recompiles those 15 files and shares everything else through ccache. Checked by
[run/cmake_hook_check.sh](run/cmake_hook_check.sh): exactly the 15 files carry the options (148 other CUDA files do
not, and without the option none does), and a file compiled with them takes effect (the multi-token kernels at 128
registers under `--ptxas-options=-dlcm=ca`); [arm_binary_check.tsv](arm_binary_check.tsv) shows that each arm's `test-backend-ops` carries its flag (the grid types' 2–4-token kernels average 253.9 registers in `iqmc2` against 168, the Q2_0 kernels 114.7 in `mtokmb4` and 130.9 in `mrr128` against 124.0, and `dlcm_cg` has 12,336 `LDG.E.CG` against 1,279). Options are single tokens (CMake does not split a source file's option at
a space): `--ptxas-options=-dlcm=ca` for `-Xptxas -dlcm=ca`.

## The P106 timing (not run yet)

Candidates, in order ([candidates.tsv](candidates.tsv)); each is a `test-backend-ops` build of the branch with
`-DGGML_CUDA_PASCAL_FLAGS=<flags>` ([run/timing_arms.tsv](run/timing_arms.tsv)), and the bench arms where the standalone
kernels have the same code:

| # | Arm | Flags | Why |
|---|---|---|---|
| 1 | `iqmc2` | `-DMMVQP_IQ_MC_MIN_BLOCKS=2` | the grid types' 2–4-token kernels without a stack frame, −8 to −12 % instructions, 2 blocks per SM instead of 3 |
| 2 | `mtokmb4` | `-DMMVQP_MTOK_MIN_BLOCKS=4` | 7 Q2_0 multi-token kernels at 4 blocks per SM (one takes 92 local accesses) |
| 3 | `mrr128` | `-maxrregcount=128` | one flag for every Pascal file: −3.2 % instructions in the k-column kernels, −1 % to −7 % elsewhere, a little less occupancy |
| 4 | `dlcm_ca` | `--ptxas-options=-dlcm=ca` | the flag the survey named: expected to lose on the grid types (4 blocks per SM, more local memory) |
| 5 | `dlcm_cg` | `--ptxas-options=-dlcm=cg` | a control: the plain loads bypass L1; against `base` it shows whether L1 caching of them matters |

The arms that lose statically are not timed (`mmvq_mb3/4`, `bf16_mb3/4`, `kcol_mb4`: spills in every kernel for 1–5 % more
instructions). `ftp_mb2` (no stack in the prompt kernel, one kernel at 2 blocks per SM) belongs to Q676's timing, which
needs a prompt-size case; `fap_mb*` to Q675's.

Commands. On the quant machine (CPU only, built; `<work>` holds the tree and the ccache):

```
# the arms' binaries: one test-backend-ops per arm, ~3 minutes each after the first (ccache)
SRC=<tree of pascal-p106-v3-b12076-ccflags> WORK=<work> CCACHE_SEED=<ccache of the -ladder2 build> LOCK=<lock file> \
    ARMS='base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg' run/build_timing_arms.sh
# the bench arms (mmvq_pascal, mmvq_kcol; sm_61 only)
REPO=<this repository> OUT=<work>/bench ARMS='base mrr128 dlcm_ca dlcm_cg' LOCK=<lock file> run/build_bench_arms.sh
```

Relay to the rig (the rig's `/opt/ccflags/<arm>/bin` and `/opt/ccflags/bench/<arm>`), then run on the rig as its own timed
task, with nothing else running (the script refuses to start when a process uses the card or the load is above 0.5):

```
for a in base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg; do
    ssh $QUANT "tar c -C <work>/timing $a" | ssh rig "mkdir -p /opt/ccflags && tar x -C /opt/ccflags"
done
ssh $QUANT "tar c -C <work>/bench ." | ssh rig "mkdir -p /opt/ccflags/bench && tar x -C /opt/ccflags/bench"
ssh rig "ROOT=/opt/ccflags ARMS='base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg' REPS=1 CASES=iq_down OUT=/tmp/ccflags_timing bash -s" \
    < run/p106_timing.sh          # first: one group, one pass, to see how long a group takes
ssh rig "ROOT=/opt/ccflags ARMS='base iqmc2 mtokmb4 mrr128 dlcm_ca dlcm_cg' REPS=3 OUT=/tmp/ccflags_timing bash -s" \
    < run/p106_timing.sh
scp -r rig:/tmp/ccflags_timing . && ssh rig 'rm -rf /tmp/ccflags_timing'
run/timing_table.py ccflags_timing base     # summary_tbo.tsv and summary_bench.tsv: per case, the median time per call, the change against base, and base's spread between passes
```

The cases of [run/p106_timing.sh](run/p106_timing.sh), from
`make_test_cases_perf` and the bench: `iq_down`, `iq_gateup`, `iq_dense` (IQ2_S and IQ3_XXS at Ornith's shapes, 1, 2 and
4 tokens: token 1 runs the one-token kernels `iqmc2` leaves alone), `q2_0_mtok` (2–8 tokens), `q2_0_one`, `kcol` (the dense
types at 1–8 tokens), `bf16`, and the bench's `bench_mtok` (with its own `mtok_mb4` variant), `bench_q2_0`, `bench_bf16`,
`bench_kcol`. The arms alternate in order on odd and even passes so that drift shows; an arm's change counts when it
exceeds `base_noise_pct`. [run/p106_timing_dryrun.sh](run/p106_timing_dryrun.sh) runs the script and the table tool
against fake binaries without a GPU.

**What would change a default:** `iqmc2` becoming the `_mc` kernels' setting if it beats base at 2 and 4 tokens without
costing the 1-token case (the one-token kernels are untouched, so the 1-token rows are the noise control); `mtokmb4` and
`mrr128` the same way per case. A flag that wins on the standalone case but not on test-backend-ops is not taken: release
settings follow agentic speed (the `bench/agent_check` arms), these cases are diagnostics.

## Files

| File | What |
|---|---|
| [kernels.tsv.gz](kernels.tsv.gz) | one row per arm and kernel: registers, stack, local, shared, spills, `STL`/`LDL`, loads, instructions, FP16 counts, blocks per SM, occupancy, change against base |
| [groups.tsv](groups.tsv), [summary_family_arm.tsv](summary_family_arm.tsv) | the same per kernel group and per file family |
| [load_forms_family_arm.tsv](load_forms_family_arm.tsv) | every load mnemonic per family and arm |
| [changes_vs_base.tsv.gz](changes_vs_base.tsv.gz) | each kernel an arm changed, with both values |
| [baseline_check.tsv](baseline_check.tsv) | the base arm's machine code against the rig build's own objects |
| [half_math.tsv](half_math.tsv), [fp16_sites.tsv](fp16_sites.tsv) | FP16 arithmetic and conversions; the Q672 site list |
| [candidates.tsv](candidates.tsv), [arm_binary_check.tsv](arm_binary_check.tsv) | the five P106 candidates with their static evidence; the registers of each arm's test-backend-ops |
| [compile_times.tsv](compile_times.tsv) | seconds per compile (171 compiles, 6 at a time) |
| `run/compile_arms.sh`, `compile_inner.sh`, `extract_flags.py` | the compiles: flags from `compile_commands.json`, one cubin per file and arm, `cuobjdump -sass` and `-res-usage` |
| `run/analyze.sh`, `analyze.py`, `groups.py` | the tables |
| `run/arms.tsv`, `files.tsv`, `timing_arms.tsv` | arms, files, the timing arms |
| `run/build_timing_arms.sh`, `build_bench_arms.sh`, `bench_inner.sh`, `cmake_hook_check.sh`, `arm_binary_check.sh` | the timing binaries, the hook check, the check of the binaries |
| `run/p106_timing.sh`, `timing_table.py`, `p106_timing_dryrun.sh` | the P106 timing, its tables, its dry run |

## Commands that produced the tables

On the quant machine, in containers with `--cpu-shares 256`, at most 6 compiles at a time, under the shared build lock:

```
SRC=<-ladder2 tree with its build/> WORK=<work> JOBS=6 LOCK=<lock file> run/compile_arms.sh          # 124 compiles: base ... xvec, iqmb*, mtokmb*
SRC=<-ccflags tree + build/compile_commands.json> WORK=<work>/w2 JOBS=6 LOCK=<lock file> \
    ARMS='base2 mmvq_mb2 mmvq_mb3 mmvq_mb4 iqmc2 kcol_mb1 kcol_mb3 kcol_mb4 bf16_mb2 bf16_mb3 bf16_mb4 fap_mb2 fap_mb4 ftp_mb2 ftp_mb4' \
    run/compile_arms.sh                                                                                # 47 compiles
WORKS='<work> <work>/w2' OUT=tables run/analyze.sh
python3 run/groups.py tables/kernels.tsv.gz tables/groups.tsv
SRC=<-ccflags tree> run/cmake_hook_check.sh
```

Provenance: the sources are the fork's (`pascal-p106-v3-b12076-ladder2` `a330672c1`, branch `-ccflags` `9e1a8492e`), built
with the CUDA 12.9.2 toolchain in the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image; the raw `ptxas -v`, `cuobjdump
-res-usage` and `-sass` dumps are not published (the tables summarize them).
