# Expert kernels that read and decode each weight once for 2–4 tokens (2026-10-28)

Registry row Q211. Ornith-1.5's built-in MTP layer is right 0.85–0.91 of the time on its first draft and 0.74 / 0.67 on
the next two ([2026-10-28_ornith_mtp_accept](../2026-10-28_ornith_mtp_accept/README.md), Q341, Q366), so llama.cpp's
draft-and-verify (`llama-server --spec-type draft-mtp`) should pay on the rig: two drafts, then one 3-token step of the
model. It did not on v2.0 (8.8 → 7.9 tok/s, [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)),
because the Pascal kernels ran 2–4-token products slower than one token per weight read
([2026-10-28_lowbit_rig](../2026-10-28_lowbit_rig/README.md), [2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md)):

- **Expert products cost 2–3× more per distinct expert at 2–4 tokens than at one, in every format, Q8_0 included,**
  which needs no decoding: an N2P expert down product at 2 tokens took 660 µs, against ~220 µs for its ~15 distinct
  experts at the one-token rate. The earlier multi-token kernel scanned the step's expert ids entry by entry in every
  block, so its overhead grew with the square of the entries.
- **Dense products repeated the decoding per token:** 1.86× one token at 2 tokens for N2P, 1.72× for IQ2_S, against
  1.19× for Q8_0, whose weights need no decoding.

Fork branch `pascal-p106-v3-mtpk` (off `pascal-p106-v3`, with the IQ2_S/IQ3_XXS kernels of `pascal-p106-v3-iqmmvq` and
the N2P/N3P types of `pascal-p106-v3-lowbit` merged) adds `mul_mat_vec_q_pascal_mt` for the types of Ornith's files:
IQ2_S, IQ3_XXS, N2P, N3P, Q8_0 and Q5_0 (the design is at the top of `ggml/src/ggml-cuda/mmvq-pascal.cu`):

- **One block per group of tokens that picked the same expert,** found by each warp from the step's expert ids with
  shuffles and ballots (one load of ids per lane); blocks past the last group exit at once.
- **The block's tokens' activations staged once in shared memory,** padded against bank conflicts, in tiles where a
  row needs more than 16 KB; where a row fits, a block handles several row blocks, so the staging and the IQ grid
  tables serve more rows.
- **Each 32-weight sub-block decoded once** into signed int8 words and multiplied by every token's activations; the
  token count (1–4) is a compile-time constant of the inner loop.
- **Every token's result is bit-identical to the one-token kernel's:** the same integer sums and the same FP32
  operations in the same order. `GGML_CUDA_MMVQ_PASCAL_MT=0` restores the earlier kernel.

## Results so far

- **Correct** natively (V100) and through the compute 6.1 PTX the P106 runs, every 2–4-token product bit-identical to
  the single-token products of its columns: [correctness.md](correctness.md).
- Kernel time on a P106 and the agent check with `--spec-type draft-mtp`: next, in the rig queue.

## Files

| File | What |
|---|---|
| [correctness.md](correctness.md) | test-backend-ops cases passed per type and setting, bit-identity and memcheck results |
| `run/build.sh` | the CUDA build on the quality machine (compute 6.1 + PTX, sm_70), and the rig tarball |
| `run/correctness.sh` | the correctness runs: test-backend-ops (native, PTX; three kernel settings), the Flint types, `multitoken_bitid`, compute-sanitizer |
| `run/rig_kernels.sh`, `run/table.sh` | the kernel timing on one P106 (new, earlier and llama.cpp's kernel; 1–4 tokens; row-block and tile settings) and its table |

## Made with

```bash
D=docs/data/2026-10-28_multitoken_kernels/run
QM=<quality machine> WORK=<its work folder> FORK=<fork checkout> COMMIT=f0f2be53b bash $D/build.sh
# multitoken_bitid against that build's static libraries (bench/mmvq_pascal/verify/build_multitoken_bitid.sh's link line)
BIN=<WORK>/out/build/bin BITID=<multitoken_bitid> GPUS=<a V100> OUT=correct_native.log JITS=0 bash $D/correctness.sh
BIN=<WORK>/out/build/bin BITID=<multitoken_bitid> GPUS=<a newer card> OUT=correct_ptx.log JITS=1 MTS="1|1 1 3" \
  MEMCHECK=0 bash $D/correctness.sh
```

- **Build:** fork branch `pascal-p106-v3-mtpk` at `f0f2be53b`: static, CUDA 12.9, compute 6.1 + compute_61 PTX and sm_70.
- **Machines:** the quality machine's V100 32 GB and RTX 3080 (correctness only; no timings there).

## Provenance and licence

- The weights and activations of the checks are random (test-backend-ops; `multitoken_bitid` draws its rows from 256
  quantized random rows per shape).
- This folder: tables and scripts under this repository's licence.
