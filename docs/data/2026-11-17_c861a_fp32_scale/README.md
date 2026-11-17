# C861 (a): the K-quant prompt kernels' scale multiply in FP32 (`GGML_CUDA_MMQ_KQ_FP32SCALE`)

Registry row [Q1549](../../../quants/experiments.md) (part a). The switch is described in
[docs/switches.md](../../switches.md), "Prompt-processing kernels".

**What it changes.** In the prompt kernels (MMQ, the dp4a layout every Pascal card uses), the Q3_K, Q4_K, Q5_K and
Q6_K dot products multiply each sub-block's integer scale by an integer dot product. Compute 6.1 has no 32-bit integer
multiply and runs each one as three XMAD instructions. With the switch on, that multiply is done in FP32. The FP32 form
is exact: every product and sum involved is at most 2^24 in magnitude, so each FP32 value equals the integer one and
the kernel returns the same floats.

**Where the idea comes from.** [ik_llama.cpp #2876](https://github.com/ikawrakow/ik_llama.cpp/pull/2876) (MIT,
merged 2026-11-10 as `766611c4f`; three lines in `vecdotq.cuh`, compiled in for `sm_61` and AMD). There a Tesla P40 read
512-token prompts 5.71 % and 5.69 % faster on two dense K-quant models (852.48 → 901.11 and 819.53 → 866.13 tok/s) and
0.59 % on an MoE model, inside that run's 14.79 % noise. Only the idea was taken; the code here is written for Miner
Llama's kernels.

**What Miner Llama does differently.** It is a run-time switch, off by default, in one binary:
`GGML_CUDA_MMQ_KQ_FP32SCALE` = `0` off, `1` compute 6.1 only, `2` any NVIDIA GPU (testing).
`GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES` picks the types: default `q4_K,q5_K,q6_K` (the three #2876 measured); `all` adds
Q3_K, whose dot product has one such multiply per two sub-blocks. A template flag carries the choice from the launch to
the dot product, so with the switch off every kernel is compiled from the unchanged code. Fork branch
`pascal-p106-v5-kq-fp32scale`, commits `e2eaa41d8` and `abd6a2deb`, on `pascal-p106-v5` (`eb82916d3`, the v4.0 code).

## Results (CPU only: build and machine code, no card)

1. **Which release files it can touch** (`kquant_census.txt`, every tensor's type read from the GGUF headers):

   | Release file | K-quant tensors | Prompt weights it changes |
   |---|---|---|
   | Flint Pascal Q8 GGUF (`p065`) | none (Q8_0, Q5_0, F16, F32) | none |
   | Flint Pascal Q5 GGUF (K2h-r16) | none (Q5_0, Q8_0, F16, F32) | none |
   | Ornith-1.5-122B Pascal N2P GGUF (`valid1`) | none (N2P, N3P, Q5_0, Q8_0, F32) | none; N2P and N3P prompt products use Q8_0's dot product, with float scales |
   | Swift Qwen3.8-Flash-Next's v4 file (UkisAI's Swift-1.5 GSQ-RCO Q2_0) | Q3_K 91, Q4_K 38, Q5_K 16, Q6_K 12 | dense attention, SSM output and shared-expert matrices: Q4_K 412.9 M weights, Q5_K 121.6 M, Q6_K 102.9 M (the default types, 637.4 M), Q3_K 1,162.0 M |

   The output layer (Q5_K) and the token table (Q3_K) are left out of the last column: a prompt reads the output layer
   for its last token only (the vector kernels) and the token table by row lookup. The routed experts are Q2_0 and
   IQ4_NL. So only Flash-Next can gain, and only on its dense layers; Q3_K holds 1.8 times the weights of the three
   default types.

2. **With the switch off the code is v4.0's** (`sass_compare.py`, `cuobjdump -sass` of `test-backend-ops`, every
   kernel's SASS hashed with whitespace normalized, against the v4.0 release build's `test-backend-ops`): 8,097 of
   the 8,797 kernel names are identical. All 700 that differ are stubs of at most 30 instructions: `mul_mat_q` tile
   widths and fused variants that compute 6.1 never launches, whose body only reports an error. 568 differ only in
   the source line number they print (1083 → 1102: `mmq.cuh` moved down 19 lines); the other 132, in the four K-quant
   files that gained kernels, also in the address of their message strings (`sass_identity.txt`). None of the 5,080
   kernels longer than 200 instructions differs, and every `mul_mat_q` kernel that runs on a P106 (308) is identical.

3. **What the FP32 form changes** (`kq_instructions.tsv`: each K-quant `mul_mat_q` kernel built for compute 6.1, its
   instruction counts off and on). At 64 columns, the widest compute 6.1 builds:

   | Type | Instructions off → on | XMAD off → on | I2F off → on | dp4a (IDP) |
   |---|---|---|---|---|
   | Q3_K | 1,338 → 1,248 | 359 → 166 | 32 → 72 | 256, unchanged |
   | Q4_K | 1,860 → 1,740 | 347 → 153 | 72 → 80 | 512, unchanged |
   | Q5_K | 1,974 → 1,860 | 394 → 204 | 72 → 80 | 512, unchanged |
   | Q6_K | 2,034 → 1,860 | 606 → 222 | 64 → 144 | 512, unchanged |

   These are counts in the code, not timings: whether the prompt gets faster on a P106 is the rig row below.

## The rig row (not run yet)

`run/rig_kq_ab.sh`, on one rig with nine P106-090, Flash-Next's v4 preset: per arm (off; on with the default types;
on with `all`) the MMQ test cases of Q3_K to Q6_K in `test-backend-ops` (every case must pass), `llama-bench` prompts
of 64, 128 and 512 tokens at micro-batches of 64 and 512 (five repetitions, two rounds), and one greedy reply of 128
tokens to `run/prompt.txt` (about 600 tokens, read through the prompt kernels), which must be the same bytes in every
arm.

## Files

| File | What |
|---|---|
| `kquant_census.txt` | per release file: tensors and bytes per type, and every K-quant tensor by role (name with its layer replaced by N), type and shape |
| `sass_identity.txt` | the comparison's summary: kernels identical and different, and every differing line's kind |
| `kq_instructions.tsv` | per K-quant `mul_mat_q` kernel (type, tile width J, fallback, fused): instruction, XMAD, I2F, IDP, FFMA, FMUL and FADD counts with the switch's flag off and on |
| `run/kquant_census.py` | the census (gguf-py, header only) |
| `run/sass_compare.py` | the machine-code comparison: `hash`, `compare`, `extract`, `linediff` |
| `run/rig_kq_ab.sh`, `run/prompt.txt` | the rig row |

## Commands

The build (the quant machine's CPU, Docker, `--cpu-shares=256`, `-j 8`):

```bash
TESTS=1 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-kq-fp32scale
```

Rig tarball `llama.cpp-20261117-abd6a2deb-pascal-p106-v5-kq-fp32scale.tar.gz` (838,885,983 bytes, md5
`b416040d21d355b66ed536ebd7b2784a`), image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, `sm_61`, 0 AVX instructions in
`llama-bench`, highest glibc symbol GLIBC_2.38; `bin/llama-server` md5 `8799f2345b5ffca0311105cd49a5f04b`,
`bin/test-backend-ops` `76d1173dd5e3352bb9029c7504d96cc7`.

The census (per release file, all shards of one model):

```bash
python3 -I run/kquant_census.py <llama.cpp>/gguf-py "<label>" <shard 1> [<shard 2> ...]
```

The machine-code comparison (the CUDA image for `cuobjdump`, any Python 3 for the script):

```bash
cuobjdump -sass $BASE/bin/test-backend-ops | c++filt | python3 -I run/sass_compare.py hash > base.tsv
cuobjdump -sass $NEW/bin/test-backend-ops | c++filt | python3 -I run/sass_compare.py hash > new.tsv
python3 -I run/sass_compare.py compare base.tsv new.tsv cmp     # cmp.identity.tsv, cmp.kq.tsv (= kq_instructions.tsv)
awk -F'\t' 'NR>1 && $4!="yes" {print $1}' cmp.identity.tsv > differ.list
cuobjdump -sass $BASE/bin/test-backend-ops | c++filt | python3 -I run/sass_compare.py extract differ.list > ext-base.txt
cuobjdump -sass $NEW/bin/test-backend-ops | c++filt | python3 -I run/sass_compare.py extract differ.list > ext-new.txt
python3 -I run/sass_compare.py linediff ext-base.txt ext-new.txt
```

`BASE` is the v4.0 release build (`miner-llama-v4.0`, built from `eb312b7bc`, the tag commit's second parent with the
same tree), `NEW` this branch's build.

## Provenance and licence

The idea: ik_llama.cpp #2876 by mb8565, merged by ikawrakow (MIT). The code and these files: Miner Llama, MIT, as
llama.cpp. The tables are read from my own release files' headers and my own builds.
