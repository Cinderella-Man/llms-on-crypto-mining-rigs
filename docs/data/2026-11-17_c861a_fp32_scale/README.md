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

## The rig row

`run/rig_kq_ab.sh`, on one rig with nine P106-090, Flash-Next's v4 preset (`release/flash-next-v4/settings.env`, its
environment and its layer split): per arm, `off`, `on` (the default types Q4_K, Q5_K, Q6_K) and `all` (Q3_K too),
(1) the MMQ test cases of Q3_K to Q6_K in `test-backend-ops` (every case must pass), (2) `llama-bench` prompts of 64,
128 and 512 tokens at micro-batches of 64 and 512 (five repetitions, two rounds in the order off, on, all), and (3)
one greedy reply of 128 tokens to `run/prompt.txt` (621 tokens, read through the prompt kernels at micro-batches of
512), which must be the same bytes in every arm.

## Result (rig 1, 2026-11-18)

Run on `pascal-p106-v5` at `602691153` (upstream b12483 merged; the switch has been in the v5 line since 2026-11-17),
Flash-Next's v4 file, 20:23:59 to 21:16:35, nothing else on the rig. Step (1) was left out (`STEPS="bench greedy"`):
`test-backend-ops` checks each case against the CPU backend, which on the rig's two Celeron cores takes hours. The
greedy reply below and the exactness argument above cover correctness.

**The replies are the same bytes in all three arms** (`rig1_greedy.tsv`: md5 `21e6e02ed314660e84cd3e2d781e7c2c`,
417 bytes each). The FP32 form is exact on a P106, as the argument says.

**Speed, a diagnostic** (`rig1_bench.md`; single requests, so Flash-Next's agent check decides whether its preset
sets the switch):

1. **The first load was slow in every test, and it was an `off` load.** `off` round 1 read 6.10, 4.02, 3.54, 4.16,
   5.07 and 7.61 % slower than `off` round 2 (micro-batch 64: prompts of 64, 128, 512; micro-batch 512: the same),
   every repetition lower. It was the evening's first load, after 12 minutes with the cards idle. The third load
   (`all` round 1) was slow in its first two tests (4.41 and 3.57 % below `off` round 2) and the first repetition of
   its third, then normal. So round 1's `on` against `off` (+6.35, +4.23, +3.67, +4.34, +5.43, +8.79 %) measures that
   first load, not the switch. The comparison below is against `off` round 2, the one warm `off` load; with one
   such load, `off`'s own spread is not measured, and the `on` and `all` loads' spreads stand in for it.
2. **`on` (Q4_K, Q5_K, Q6_K): negligible.** Against `off` round 2, `on` round 1 and round 2:

   | -ub | prompt | `on` r1 | `on` r2 | `on` r2 vs r1 |
   |---|---|---|---|---|
   | 64 | 64 | −0.13 % | +0.08 % | +0.22 % |
   | 64 | 128 | +0.04 % | +0.09 % | +0.05 % |
   | 64 | 512 | +0.00 % | +0.08 % | +0.08 % |
   | 512 | 64 | −0.00 % | +0.05 % | +0.05 % |
   | 512 | 128 | +0.09 % | +0.15 % | +0.06 % |
   | 512 | 512 | +0.51 % | +0.35 % | −0.16 % |

   Every change but one is 0.15 % or less, the size of the arm's own round-to-round spread in the last column;
   512-token prompts at micro-batches of 512 read +0.51 and +0.35 % faster.
3. **`all` (Q3_K too): a small gain at micro-batches of 512, negligible at 64.** Against `off` round 2:

   | -ub | prompt | `all` r1 | `all` r2 | `all` r2 vs r1 |
   |---|---|---|---|---|
   | 64 | 64 | −4.41 % (slow start) | −0.21 % | +4.40 % |
   | 64 | 128 | −3.57 % (slow start) | +0.37 % | +4.09 % |
   | 64 | 512 | −0.07 % | +0.25 % | +0.32 % |
   | 512 | 64 | +0.31 % | +0.28 % | −0.03 % |
   | 512 | 128 | +0.50 % | +0.42 % | −0.07 % |
   | 512 | 512 | +1.39 % | +0.84 % | −0.53 % |

   At micro-batches of 512 both loads are above `off` by more than their own spread, and every repetition is above
   the same repetition of `off` round 2 but one (512 tokens, round 2's first: 103.00 against 103.01 tok/s). Q3_K
   holds 1.8 times the dense weights of the three default types, so `all` gains more than `on`.
4. **Nothing near the P40's 5.7 %.** The switch reaches only Flash-Next's dense layers (637.4 M weights at the
   default types, 1,162.0 M more in Q3_K, of 176.94 B); the routed experts, Q2_0 and IQ4_NL, take other kernels.
   #2876's 5.7 % was on dense K-quant models.
5. **The greedy replies' prompt pass** (`rig1_greedy.tsv`, the first pass of a fresh process, no warm-up, one each):
   `off` 8,981.51 ms, `on` 8,449.99 ms, `all` 8,417.71 ms for the 621 tokens (69.14, 73.49 and 73.77 tok/s). One
   pass per arm, with `off` first in its step as in the benchmark step: not a speed comparison. The warmed,
   repeated `llama-bench` tests do not show a difference of that size.

**Verdict:** exact; `GGML_CUDA_MMQ_KQ_FP32SCALE=1` with the default types is negligible, and with
`GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all` a small gain on prompts read in micro-batches of 512 (+0.28, +0.42 and
+0.84 % in round 2). Both stay off by default. Flash-Next's agent check with `1` and `all` against its preset decides
the preset (registry Q1602), with the arms' order alternated, since the first load here was slow.

## Files

| File | What |
|---|---|
| `kquant_census.txt` | per release file: tensors and bytes per type, and every K-quant tensor by role (name with its layer replaced by N), type and shape |
| `sass_identity.txt` | the comparison's summary: kernels identical and different, and every differing line's kind |
| `kq_instructions.tsv` | per K-quant `mul_mat_q` kernel (type, tile width J, fallback, fused): instruction, XMAD, I2F, IDP, FFMA, FMUL and FADD counts with the switch's flag off and on |
| `run/kquant_census.py` | the census (gguf-py, header only) |
| `run/sass_compare.py` | the machine-code comparison: `hash`, `compare`, `extract`, `linediff` |
| `rig1_bench.md` | the rig row's `llama-bench` loads: every load's tok/s with each repetition, the changes against `off` per round and each arm's round-to-round spread, the means, and every load against `off` round 2 (per cent, and per repetition in tok/s) |
| `rig1_greedy.tsv` | the greedy replies: per arm the md5 and bytes of the reply and its prompt pass |
| `run/rig_kq_ab.sh`, `run/prompt.txt` | the rig row (run with `STEPS="bench greedy"`) |
| `run/kq_ab_table.py` | the tables of `rig1_bench.md` from the runner's `bench-<arm>-<round>.jsonl` |

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

The rig row (2026-11-18, from a machine that reaches the rig as `$RIG`; `BIN` is the rig build of `pascal-p106-v5`
`602691153`, `bin/llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`, whose `llama-bench` and `llama-completion`
name `GGML_CUDA_MMQ_KQ_FP32SCALE`; `MODELS` the folder with Flash-Next's v4 file):

```bash
P=llms-on-crypto-mining-rigs; D=/tmp/c861a
ssh $RIG "mkdir -p $D/run"
for f in $P/docs/data/2026-11-17_c861a_fp32_scale/run/rig_kq_ab.sh $P/docs/data/2026-11-17_c861a_fp32_scale/run/prompt.txt \
         $P/release/flash-next-v4/settings.env $P/scripts/ram_watchdog.sh; do ssh $RIG "cat > $D/run/$(basename $f)" < $f; done
ssh $RIG "BIN=$BIN/bin MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf OUT=$D/out STEPS='bench greedy' \
    setsid nohup bash $D/run/rig_kq_ab.sh > $D/rig_kq_ab.out 2>&1 < /dev/null &"
# when $D/out/DONE exists: copy $D back, then rm -rf $D on the rig
python3 -I run/kq_ab_table.py <copy of $D/out> "off on all" 2 off-2 > rig1_bench.md
cp <copy of $D/out>/greedy.tsv rig1_greedy.tsv
```

## Provenance and licence

The idea: ik_llama.cpp #2876 by mb8565, merged by ikawrakow (MIT). The code and these files: Miner Llama, MIT, as
llama.cpp. The tables are read from my own release files' headers and my own builds.
