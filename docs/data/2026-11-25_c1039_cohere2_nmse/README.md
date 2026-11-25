# C1039: why `cohere2`'s card result moved in b12674

Registry row [Q1772](../../../quants/experiments.md) (from Q1649's P106 check,
[2026-11-24_c912_p106_check](../2026-11-24_c912_p106_check/README.md)). Correctness only, nothing timed.

**Upstream's new CUDA fusion of `NORM` and `MUL` ([#32454](https://github.com/ggml-org/llama.cpp/pull/32454)) moved
it, and nothing is wrong.** With the CUDA fusions off (`GGML_CUDA_DISABLE_FUSION=1`) the random `cohere2` model gives
the same card-against-CPU nmse on both builds. With them on, only the candidate's number moves, and only
`cohere2`'s: the fused kernel sums the layer norm's mean and variance in another order, so it rounds differently.

To see this on an RTX 3090 the card's TF32 rounding had to be turned off (`NVIDIA_TF32_OVERRIDE=0`). With TF32 on,
every model's nmse sits near 1e-7, ten million times the P106's level, and the fusion's change is invisible.

## What ran

The builds are the ones C912's P106 check compared, both rig tarballs (built for compute capability 6.1):
- the candidate `pascal-p106-v5-cand-b12674` `eee615cdb` (upstream b12674 merged), `test-llama-archs` md5
  `4e5a143af3fe0e4252e323f983973545`;
- the reference `pascal-p106-v5` `af00bb13b`, `test-llama-archs` md5 `bf171d94e7000d14d1933e594eb0d2de`.

Both md5s are C912's. [run/run.sh](run/run.sh) runs `test-llama-archs -b CUDA0 -s 1` six times on one RTX 3090 of a
build machine, in one container per run. The driver compiles the builds' compute 6.1 PTX for the card. The six runs:
- each build with the fusions on (the default) and off;
- each fusion-on run once more, to show the run-to-run spread.

The first run writes every random model's processor logits (`LLAMA_ARCHS_LOGITS`), and every later run compares its
own processor logits with them. So for each model the table says whether the two builds' processor results are the
same bits. Only for those models can the card results be compared across the builds: upstream's #32669 gives 120 of
the random models new weight scales, which changes their logits on b12674 (C912).

There were three sets of six runs, 2026-11-25 12:38:20–12:45:38, on the same card:

| Set | Settings added to every run | Why |
|---|---|---|
| `plain` | none | the run as planned |
| `f32` | the fork's P106 kernels forced (`$PASCAL_FORCED` from [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh)) and `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | `plain` put every model near 1e-7; this set tests whether the card's own kernels or cuBLAS's half-precision compute type did that |
| `notf32` | the same, and `NVIDIA_TF32_OVERRIDE=0` | `f32` gave the same numbers as `plain`; this set turns off TF32 |

## Result

`cohere2`'s card-against-CPU nmse ([nmse.tsv](nmse.tsv)); its processor logits are the same bits on both builds in
every set:

| Set | Reference, fusions on | Candidate, fusions on | Reference, fusions off | Candidate, fusions off | Repeats (reference, candidate) |
|---|---|---|---|---|---|
| `plain` | 8.92e-08 | 8.92e-08 | 8.92e-08 | 8.92e-08 | 8.92e-08, 8.92e-08 |
| `f32` | 8.92e-08 | 8.92e-08 | 8.92e-08 | 8.92e-08 | 8.92e-08, 8.92e-08 |
| `notf32` | **2.43e-14** | **2.44e-14** | **2.43e-14** | **2.43e-14** | 2.43e-14, 2.44e-14 |
| P106, C912 (fusions on) | 2.80e-14 | 2.86e-14 | — | — | — |

The 28 random models whose processor logits are the same bits on both builds ([groups.tsv](groups.tsv)):

| Set | Same card nmse on both builds, fusions on | The same, fusions off | Every repeat gives its first run's number |
|---|---|---|---|
| `plain` | 28 of 28 | 28 of 28 | 28 of 28 |
| `f32` | 28 of 28 | 28 of 28 | 28 of 28 |
| `notf32` | **27 of 28: all but `cohere2`** | **28 of 28** | 28 of 28 |

So with TF32 off the 3090 shows what the P106 showed: one model of 28 moves, `cohere2`. The fusion switch alone
undoes the move. The repeats give their first run's number to the printed digit in every set and every model, so the
moved digit is not noise. All 18 runs pass every card row; each candidate run counts 120 "failed" tests, which are the
120 models whose processor logits differ from the reference's file ([runs.tsv](runs.tsv)), as in C912.

Two more rows of [nmse.tsv](nmse.tsv) are worth a word:
- `cohere2moe` gives 2.54e-14 on both builds in `notf32`, but its processor logits differ between the builds (new
  weight scales), so that equality is a coincidence of the printed digits, not a comparison;
- `gemma4` is the largest of the 28 with TF32 off, 1.43e-07 on both builds; on the P106 it was 1.42e-07 (C912). With
  TF32 on it was 7.31e-07.

## Why the fusion rounds differently

From reading upstream's commit, not measured: before #32454 the `NORM` kernel ran rows shorter than 1,024 values
with one 32-thread warp (`test-llama-archs`' random models are 256 values wide by default), and `MUL` then
multiplied by the norm's weight in a kernel of its own. The fused kernel does
the multiply in the same kernel, with the same float operations in the same order. But it runs rows of up to 2,048
values with a 256-thread block. The mean and the variance are then summed by 256 threads instead of 32, in another
order, so they round differently by a few units in the last place. `cohere2` normalises every block with a layer
norm without a bias, the `NORM` + `MUL` pattern the fusion takes. The release files use RMS norms, which this fusion
does not touch.

## Why TF32 had to be off

ggml's CUDA backend sets `CUBLAS_TF32_TENSOR_OP_MATH` on every cuBLAS handle (`ggml/src/ggml-cuda/common.cuh`). On
an Ampere card cuBLAS then rounds an F32 matrix product's inputs to TF32, which keeps 10 bits of mantissa. A P106 has
no TF32, so its products stay in F32. `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` does not change this: it sets the compute
type, and the handle's math mode still allows TF32. NVIDIA's `NVIDIA_TF32_OVERRIDE=0` overrides a library's own
setting. With it, the 3090's card-against-CPU nmse falls to the P106's level for every model.

The same holds for any check that runs the rig build on these cards and compares a card's numbers at the P106's
precision; registry row Q1841 (planned) adds `NVIDIA_TF32_OVERRIDE=0` to the forced set.

## Files

| File | What |
|---|---|
| [nmse.tsv](nmse.tsv) | per set and random model: whether the two builds' processor logits are the same bits ("no logits file": the test writes none for that model, or only the candidate has the model), the card-against-CPU nmse of all six runs, and three verdicts (the builds agree with the fusions on, with them off, each repeat gives its first run's number) |
| [groups.tsv](groups.tsv) | the verdicts counted per set, apart by whether the processor logits are the same bits |
| [runs.tsv](runs.tsv) | per run: build, fusions, exit status, `test-llama-archs`' summary, card rows OK / FAIL / SKIP, processor logits written or compared and how many differ |
| [run/run.sh](run/run.sh) | one set of six runs (one container per run, the PTX JIT cache seeded from a copy) |
| [run/tables.py](run/tables.py) | the three tables from the sets' output folders (standard library only) |

The `plain` set ran an earlier copy of `run.sh` without the `EXTRA_ENV` lines; with `EXTRA_ENV` unset the published
one runs the same command. The raw logs are not published: they name the machine.

## Commands

```bash
CAND=<folder>/llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674   # the unpacked rig tarballs
REF=<folder>/llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2
D=docs/data/2026-11-25_c1039_cohere2_nmse
. scripts/pascal_forced.sh
COMMON="CAND=$CAND REF=$REF GPU=2 JIT_CAND=<CAND's PTX JIT cache> JIT_REF=<REF's> PREFLIGHT=scripts/cuda_preflight.sh"
env $COMMON WORK=$OUT/plain bash $D/run/run.sh
env $COMMON WORK=$OUT/f32 EXTRA_ENV="GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $PASCAL_FORCED" bash $D/run/run.sh
env $COMMON WORK=$OUT/notf32 EXTRA_ENV="NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $PASCAL_FORCED" \
  bash $D/run/run.sh
python3 -I $D/run/tables.py $D plain=$OUT/plain f32=$OUT/f32 notf32=$OUT/notf32
```

Each container carries the labels `pascal-quants=1` and `c1039=1`, `--cpu-shares=256`, one GPU and no network.

Provenance: Miner Llama's fork and upstream llama.cpp (MIT); the random models are the ones `test-llama-archs`
generates with seed 1.
