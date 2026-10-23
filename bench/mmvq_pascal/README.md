# mmvq_pascal: single-token matrix-vector microbenchmark for the P106-090

When an LLM generates one token, almost all GPU time goes into matrix-vector products: each weight matrix is read once
and multiplied by one activation vector. llama.cpp does this with its MMVQ kernel (`mul_mat_vec_q`). This benchmark
times that kernel, and faster variants written for Pascal, on the exact matrix shapes of one token of
[Flint](https://huggingface.co/RecursiveSI/flint) (39.8 B-parameter MoE, ~2.5 B active), on one P106-090 card
(GP106, compute capability 6.1, 5 SMs, ~173 GB/s measured memory bandwidth).

The question it answers: how fast can single-token decode get on a 5-SM card if the kernel changes but the model file
does not? The static analysis in
[quant_speed_b_formats_kernels.md](../../docs/research/notes/quant_speed_b_formats_kernels.md) predicts that
llama.cpp's MMVQ is limited by instruction issue on this card rather than by memory bandwidth; this benchmark tests
that prediction (see [Expected results](#expected-results)).

## What it measures

**Shapes** (input width K → output rows N), with the number of calls in one Flint token:

| Shape | K → N | Calls per token | Notes |
|---|---|---|---|
| `attn_q` | 2048 → 4096 | 49 | attention Q projection |
| `attn_kv` | 2048 → 512 | 98 | attention K and V projections (2 per layer, same shape) |
| `attn_output` | 4096 → 2048 | 49 | attention output projection |
| `moe_gate_up` | 2048 → 512, 6 of 256 experts | 47 | routed experts, gate and up fused with SwiGLU in one kernel (`MUL_MAT_ID`) |
| `moe_down` | 512 → 2048, 6 of 256 experts | 47 | routed experts, down projection, one input vector per expert (`MUL_MAT_ID`) |
| `shexp_gate_up` | 2048 → 1024 | 47 | shared expert, gate and up fused with SwiGLU |
| `shexp_down` | 1024 → 2048 | 47 | shared expert, down projection |
| `output` | 2048 → 150144 | 1 | output head |

The gate/up projections are timed as one fused kernel because that is what llama.cpp runs on compute capability 6.1
(its MMVQ fusion is disabled only for 6.0 and older). For the MoE shapes the weight tensor holds all 256 experts and each
call reads 6 of them, selected by an ids array.

**Weight types**, all with random weights:

| Type | Bytes per 32 weights | Layout |
|---|---|---|
| `q4_1` | 20 | `half2 (d, m)`, then 16 bytes of nibbles; weight j is the low nibble of `qs[j]`, weight j+16 the high nibble. Value `d*q + m`. The makers' Flint GGUF uses this type. |
| `q4_0` | 18 | `half d`, then 16 bytes of nibbles (same order). Value `d*(q - 8)`. |
| `q8_0` | 34 | `half d`, then 32 int8. Value `d*q`. |
| `q4_1_64` | 18 | 36 bytes per 64 weights: one `half2 (d, m)`, then two 16-byte halves packed like Q4_1 ([patches/q4_1_64-type.diff](../../patches/q4_1_64-type.diff)). Holds the makers' group-64 Flint weights exactly. |
| `q5_0` | 22 | `half d`, 4 bytes `qh` (bit j = fifth bit of weight j), then 16 bytes of nibbles as in Q4_0. Value `d*(q - 16)`. Flint's S4-L0 file stores its routed experts' gate/up in it. |
| `q5_K` | 22 | 176 bytes per 256 weights: `half2 (d, dmin)`, 12 bytes of 6-bit sub-block scales and mins, 32 bytes `qh`, 128 bytes of nibbles. Value `d*sc*q - dmin*m` per 32-weight sub-block. Used by Unsloth's UD-Q4_K_XL. |
| `iq4_xs` | 17 | 136 bytes per 256 weights: `half d`, 6-bit sub-block scales (`scales_h`, `scales_l`), 128 bytes of 4-bit indices into a fixed non-linear table (`kvalues_iq4nl`). Used by UD-Q4_K_XL. |

Activations are quantized exactly as llama.cpp does it (`quantize_q8_1`): blocks of 32 int8 with `half2 (d, s)`, where s
is the FP16 sum of the 32 unquantized values.

**Kernel variants:**

| Variant | What it is |
|---|---|
| `llama` | (a) llama.cpp's MMVQ copied from commit `d77256999`, with the launch shape a compute 6.1 card gets: 4 warps per block, 1 row per block, and for small K ("small-K" rule) 4 rows per block. Verified bit-identical to llama.cpp (see [Checking correctness](#checking-correctness)). |
| `pr25479` | (b) the same kernel with the launch shape of llama.cpp PR #25479: 2 warps per block at one token. |
| `p_r1c2`, `p_r2c2`, `p_r4c2`, `p_r2c4` | (c) Pascal-tuned kernel on the unchanged block formats. A warp is split into lane groups sized to K; each group owns R rows that share one load of the activations (2R weight rows when gate and up are fused); each lane takes C consecutive blocks per iteration and loads them with the widest loads their alignment allows (`q4_1` at C = 4: five 128-bit loads per 128 weights). Row pointers are computed once and advanced by a constant; nibbles are unpacked with two masks per word (an unsigned-by-signed dp4a on the high nibbles, with the factor 16 folded into the activation scale); partial sums are reduced with warp shuffles, with no shared memory or `__syncthreads`. |
| `p_r1c1`, `p_r2c1` | (c) for the 256-weight K-/I-quants: each lane takes one super-block per iteration. Q5_K's activation sums (for its mins) are computed once per chunk and shared by all rows; IQ4_XS looks its indices up with `__byte_perm` in a table held in registers, with the activations pre-permuted into the order the lookup produces. |
| `llama_sk` | (a) for `iq4_xs` with llama.cpp's small-K shape, which llama.cpp withholds from IQ4_XS on Pascal (experiment Q47). |
| `rl_r1c2`, `rl_r2c2`, `rl_r4c2`, `rl_r2c4` | (e) row-local planes (`q8_0`, `q4_1`, `q5_0`): each row keeps its byte range but holds all its codes first and its scales after them, so every code load is a 128-bit load. Needs a load-time repack. |
| `st_r1c2`, `st_r2c2`, `st_r4c2`, `st_r2c4` | (f) staged loads (`q8_0`, `q4_0`, `q4_1`, `q5_0`): the unchanged GGUF blocks; each lane group first copies the contiguous row segment its lanes are about to read into shared memory with coalesced 128-bit loads, then every lane reads its own blocks from there. No repack. |
| `soa_r2`, `soa_r4` | (d) structure-of-arrays repack of `q4_1` (group 32) and `q4_1_64` (group 64): one plane of 16-byte nibble chunks arranged so that each weight load is a fully coalesced 128-bit load, and one plane of `half2 (d, m)` scales. Activations use the same idea: an int8 plane and a `float2 (d, s)` plane, so the block sum is FP32 (which also removes the FP16 overflow that forces llama.cpp's Flint rescale). R rows per lane group as in (c). |

**Method:**
- Every kernel is first checked against a CPU reference that computes the same dot products in double precision from
  the exact quantized weights and activations the GPU uses. The check passes when the error is below 1e-5 of the sum of
  the absolute per-block terms (2e-3 for llama.cpp's Q4_1 and Q4_1_64 kernels on GPUs other than compute 6.1, where they
  multiply the scales in FP16). For the output head, every 37th row is checked.
- Timing: 64 calls are captured in one CUDA graph (8 for the output head) and the graph is launched 30 times after
  3 warm-up launches; the reported time per call is the median over the 30 launches of graph time divided by
  the calls in the graph. Consecutive
  calls read different copies of the weights (≥ 64 MB in total for the dense shapes) or a different random set of 6
  experts, so no call finds its weights in the 1.5 MB L2 cache. The activation vector stays the same, as in a real
  forward pass, where it was just written.
- Times include the per-kernel overhead of a CUDA graph node, as llama.cpp pays it with CUDA graphs; the empty-kernel
  line printed at the start measures it.
- GB/s = weight bytes per call / time per call. `%peak` is that against 173 GB/s (`--ref-gbs` changes it).
- Per-token total = Σ over the shapes of time per call × calls per token. It covers the matrix-vector products of the
  49 attention layers, the 47 MoE layers and the output head. It leaves out layers 0–1 (dense FFN), the router (F32 in
  the makers' file), the per-call `quantize_q8_1` launches and all non-matvec work. `best(c)` and `best(c-f)` take,
  per shape, the fastest correct variant among (c), or among (c) to (f): a llama.cpp port could pick parameters per
  shape the same way. `widek` is llama.cpp's kernel with the launch shape of the wide-rows build (2 warps per block,
  except where llama.cpp's small-K rule applies), assembled per shape from `llama` and `pr25479`.
- Before anything is timed, the program streams memory for ~2 s so the card reaches its sustained clock, then records
  two reference points: the cost of an empty kernel node in a CUDA graph and the streaming-read bandwidth of a plain
  copy-style kernel (128-bit loads, read-only and plain load paths). The latter is the practical ceiling for any
  variant.

## Building

Needs Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image (CUDA 13 can no longer build for Pascal):

```
bench/mmvq_pascal/build.sh
```

This writes `bench/mmvq_pascal/build/mmvq_pascal` (about 3 MB) with sm_61 machine code for the P106, compute_61 PTX,
and sm_120 machine code for an RTX 50-series GPU. Host code is built without AVX (the rig's Celeron has none; the build
reports the count of AVX instructions in `build/BUILDINFO.txt`, which must be 0) and the CUDA runtime is linked
statically, so the binary needs nothing on the rig but the NVIDIA driver. `build/ptxas.log` and
`build/res-usage.txt` list registers per kernel; `build/sm61.sass` is the sm_61 disassembly.

## Running on the rig

From the repository root. `rig` is an SSH alias for the rig (a `Host rig` entry in `~/.ssh/config`) that logs in as a user
allowed to use the GPUs:

```
bench/mmvq_pascal/build.sh
scp bench/mmvq_pascal/build/mmvq_pascal rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --csv /tmp/mmvq_pascal.csv' | tee mmvq_pascal_rig.txt
scp rig:/tmp/mmvq_pascal.csv .
ssh rig 'rm -f /tmp/mmvq_pascal /tmp/mmvq_pascal.csv'
```

The full run takes one to two minutes and uses under 1 GB of GPU memory. It runs on one card; the other cards and the
desktop session can stay as they are. To watch the clock and power while it runs (the 75 W cap holds the core at
~1710–1760 MHz under load):

```
ssh rig 'nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 1'
```

Useful options (`--help` lists all):

| Option | Effect |
|---|---|
| `--check-only` | correctness checks only, no timing (a few seconds) |
| `--types q4_1,q8_0` | subset of weight types (default `q4_1,q4_0,q8_0,q4_1_64,q5_0,q5_K,iq4_xs`) |
| `--variants llama,p_r2c2` | subset of variants (default all) |
| `--shapes attn_q,moe_down` | subset of shapes (default all) |
| `--reps 30`, `--batch 64` | timed graph launches, calls per graph |
| `--verbose` | print the largest deviations the correctness check finds |
| `--list` | list shapes and variants |

The program exits with status 2 if any correctness check fails.

### Reading the output

One line per shape, type and variant: `MB/call` (weight bytes read per call), `us/call` (median) and `us(min)`,
`GB/s`, `%peak` (of 173 GB/s), `regs` (registers per thread), `occ` (resident blocks per SM at that register count),
`grid` (blocks per call), `err/tol` (worst deviation from the CPU reference as a fraction of the tolerance; must be
≤ 1) and `PASS`/`FAIL`. Then the per-token totals per type and variant, with the speed-up against `llama` of the
same type and against `q4_1`/`llama`, the configuration Flint runs today. The CSV holds the same rows plus
`TOKEN_TOTAL`, `GRAPH_NODE_OVERHEAD`, `STREAM_READ_LDG` and `STREAM_READ_PLAIN` rows.

## Checking correctness

On any recent NVIDIA GPU (timings there say nothing about Pascal):

```
bench/mmvq_pascal/build/mmvq_pascal --check-only                          # native code of that GPU
CUDA_FORCE_PTX_JIT=1 bench/mmvq_pascal/build/mmvq_pascal --check-only     # the Pascal code path, JIT-compiled from compute_61 PTX
```

The second form matters because llama.cpp's Q4_1 code takes a different branch on compute 6.1 (FP32 scale products)
than on every other architecture.

**Bit-exactness against llama.cpp.** `verify/` links the copied baseline (`src/llama_mmvq.cuh`) against a static
build of llama.cpp's ggml and runs the same random weights and activations through both: `MUL_MAT`, `MUL_MAT_ID` and
the fused gate/up/SwiGLU path, at every benchmark shape (the MoE cases use 32 experts), for all four types. As a
control it also runs PR #25479's launch shape, which must differ from llama.cpp wherever it changes the summation
order. The llama.cpp tree must be commit `d77256999` with `patches/q4_1_64-type.diff` applied.

```
bench/mmvq_pascal/verify/build_ggml.sh <llama.cpp dir> <out dir> 61-virtual    # or 120 for an RTX 50-series GPU
bench/mmvq_pascal/verify/build_verify.sh <llama.cpp dir> <out dir>/build ptx61  # or sm_120
bench/mmvq_pascal/build/verify_ptx61                                            # or verify_sm_120
```

`build_ggml.sh` mounts the llama.cpp tree read-only and builds only the ggml libraries (~2 minutes). The verify
binaries link cuBLAS statically and are ~1.3 GB each. On an RTX 5070 Laptop GPU all 32 cases are bit-identical in
both forms (native sm_120 and the compute 6.1 path through the JIT), and the variants of `mmvq_pascal` pass the CPU
reference check in both.

## Expected results

The prediction comes from counting the SASS instructions of each kernel for sm_61 and assuming the card is limited by
instruction issue: 5 SMs × 128 lanes × ~1.73 GHz = 1.107 T lane-instructions per second, of which 76–90 % is usable.
At 173 GB/s that budget allows at most 6.4 instructions per weight byte; a kernel that needs more is issue-bound, one
that needs fewer is bandwidth-bound.

- llama.cpp's MMVQ needs 10–17 instructions per byte at Flint's shapes, because with K = 2048 each thread runs a single
  loop iteration (prologue and reduction cost as much as the dot product), and the small-K path for the expert
  down-projection leaves 3 of 4 warps idle. These per-shape figures are the dynamic estimates of the survey note above,
  made from the rig's llama.cpp binary. The copied kernel compiles to the same code: its sm_61 loops have the same
  102 / 112 / 165 instructions per 32 weights for Q4_1 / Q4_0 / Q8_0, and the same register counts (40 / 48 / 64;
  56 for the Q4_1 small-K kernel). `build/sm61.sass` holds the disassembly.
- The tuned kernels (c) need 2.3–5.8 instructions per byte (static count of this binary's sm_61 code, prologue and
  epilogue included): `p_r1c2` 5.8, `p_r2c2` 4.0, `p_r4c2` 3.1, `p_r2c4` 2.6 at K = 2048 for Q4_1. That is below 6.4, so
  they should become bandwidth-bound.

Predictions for `q4_1`:

| Shape | MB/call | `llama` instr/byte | `llama` predicted µs (GB/s) | Tuned instr/byte | µs at 173 GB/s |
|---|---|---|---|---|---|
| `attn_q` | 5.24 | 15.3 | 80–95 (55–66) | 2.6–5.8 | 30 |
| `attn_kv` | 0.66 | 15.3 | 10–12 (55–66) | 2.6–5.8 | 3.8 |
| `attn_output` | 5.24 | 10.3 | 54–64 (82–97) | 2.3–3.9 | 30 |
| `moe_gate_up` | 7.86 | 11.8 | 93–110 (71–84) | 2.6–4.7 | 45 |
| `moe_down` | 3.93 | 16.8 | 66–79 (50–60) | 2.6–5.8 | 23 |
| `shexp_gate_up` | 2.62 | 11.8 | 31–37 (71–84) | 2.6–4.7 | 15 |
| `shexp_down` | 1.31 | 11.9 | 16–18 (71–84) | 2.6–5.8 | 7.6 |
| `output` | 192.2 | 15.3 | 2950–3500 (55–65) | 2.6–5.8 | 1111 |
| **Per token** | 1510 MB | | **20–24 ms** | | **8.7 ms** |

If the issue-bound model holds:
1. `q4_1`/`llama` totals 20–24 ms per token, and its GB/s per shape tracks 1 / (instructions per byte): `attn_output`
   ~1.5× the GB/s of `attn_q`, `moe_down` the lowest.
2. `pr25479` is 1.1–1.25× faster than `llama` on the total, mostly on the K = 2048 shapes.
3. `best(c)` reaches 9.7–11.6 ms per token (75–90 % of 173 GB/s, close to the streaming-read line), i.e. 1.7–2.5× faster
   than `llama`, except on the smallest shapes (`attn_kv`, `shexp_down`), where a few-µs kernel cannot reach full
   bandwidth.
4. With the `llama` kernel, `q4_1_64` is no faster than `q4_1` (its loop costs the same per weight) and `q4_0` is no
   faster, possibly up to 10 % slower, despite 10 % fewer bytes (2-byte-aligned blocks need more instructions). With the tuned
   kernels, time follows bytes: `q4_1_64` and `q4_0` ~10 % faster than `q4_1`, `q8_0` ~1.7× slower.

It is refuted if:
- `llama` already reaches ≥ 100 GB/s on the K = 2048 shapes, or `attn_output` is not clearly faster per byte than
  `attn_q`: the baseline is then not issue-bound.
- `best(c)` is less than 1.3× faster than `llama` on the total: then latency, the L1/texture path or DRAM limits the
  card, not instruction issue. Comparing the tuned kernels' GB/s with the streaming-read line separates these: near it
  means bandwidth; well below it means latency or load-path throughput.
- `q4_1_64`/`llama` is ≥ 8 % faster than `q4_1`/`llama`: bytes already matter with the stock kernel.

Within (c) and (d), the ranking tells which remaining cost matters: `p_r1c2` ≈ `p_r4c2` means re-reading activations
through L1 is not a limit; `p_r2c4` and `soa_*` clearly ahead of `p_r2c2` means the number of load instructions and
coalescing matter.

## Results on the P106-090

One card of the rig (compute 6.1, 5 SMs, 1531 MHz maximum clock): every variant passes the correctness check, an empty
CUDA-graph node costs 1.85 µs, and a plain streaming read reaches 179 GB/s, the practical ceiling.

Matrix-vector time per Flint token (all shapes × calls per token):

| Type | `llama` | `pr25479` | best tuned (c) | best of (c) and (d) | Bytes per token |
|---|---|---|---|---|---|
| `q4_1` | 20.0 ms (75 GB/s) | 17.3 ms (1.16×) | 11.5 ms (1.75×) | 10.6 ms (1.89×, 142 GB/s) | 1510 MB |
| `q4_1_64` | 21.2 ms | 18.2 ms (1.16×) | 10.6 ms (1.99×) | 10.1 ms (2.10×) | 1359 MB |
| `q4_0` | 24.3 ms | 19.7 ms (1.23×) | 12.6 ms (1.93×) | — | 1359 MB |
| `q8_0` | 35.3 ms | 31.7 ms (1.11×) | 23.3 ms (1.52×) | — | 2566 MB |

GB/s per shape for `q4_1`:

| Shape | `llama` | `pr25479` | `p_r2c4` | `soa_r2` |
|---|---|---|---|---|
| `attn_q` (K = 2048) | 68 | 97 | 144 | 142 |
| `attn_kv` (K = 2048) | 60 | 76 | 101 | 88 |
| `attn_output` (K = 4096) | 100 | 107 | 136 | 145 |
| `moe_gate_up` (K = 2048) | 84 | 103 | 126 | 149 |
| `moe_down` (K = 512) | 57 | 52 | 139 | 129 |
| `shexp_gate_up` (K = 2048) | 89 | 101 | 107 | 133 |
| `shexp_down` (K = 1024) | 83 | 61 | 115 | 111 |
| `output` (K = 2048) | 69 | 96 | 159 | 156 |

The issue-bound model holds on every point it predicted:

1. llama.cpp's kernel takes 20.0 ms per token for Q4_1 (predicted 20–24), `attn_output` (K = 4096, more work per
   thread) runs 1.46× the GB/s of `attn_q`, and `moe_down` is the slowest shape.
2. The 2-warp launch shape of PR #25479 is 1.16× faster in total, all of it on the K = 2048 and 4096 shapes; on the
   narrow `moe_down` and `shexp_down` it is slower (52 against 57 and 61 against 83 GB/s). A launch rule that keeps 4
   warps for K ≤ 1024 would gain a little more.
3. The tuned kernels reach 1.75× (`q4_1`) to 2.10× (`q4_1_64`) and up to 159 GB/s, 89 % of the streaming-read ceiling:
   load-instruction count and coalescing were the limit (`p_r2c4` and the plane layout are well ahead of `p_r2c2`).
4. With llama.cpp's kernel, fewer bytes don't help: `q4_1_64` is 6 % slower than `q4_1`, `q4_0` 21 % slower. With the
   tuned kernels time follows bytes for `q4_1_64` (7 % faster than `q4_1`), not yet for `q4_0` (10 % slower) or `q8_0`
   (2.0× the time of `q4_1` for 1.7× the bytes).

None of the refutation conditions occurred. In a real token the matrix-vector products are about 55 % of the time
(20 of the makers' file's 35.6 ms per token), so a llama.cpp kernel at these speeds would take the makers' Q4_1 from
28 to about 36–38 tok/s, and further with the rescale removed and the group-64 type.

### Q5_0, K-/I-quants, row-local planes and staged loads

A second run added the formats of the files at the quality bar (Q5_0 in Flint's S4-L0; Q5_K and IQ4_XS in Unsloth's
UD-Q4_K_XL) and two ways around the 2-byte alignment that holds Q8_0, Q4_0 and Q5_0 back in (c). Every variant passes
the correctness check. Matrix-vector time per Flint token on one P106-090 (`widek` = llama.cpp's kernel with the
wide-rows launch shape, the best llama.cpp build so far):

| Type | `llama` | `widek` | best (c) | best (e), row-local planes | best (f), staged | best of all |
|---|---|---|---|---|---|---|
| `q8_0` | 35.4 ms | 28.1 ms | 23.2 ms (1.53×) | 18.4 ms (1.92×) | **17.1 ms (2.07×, 150 GB/s)** | 16.7 ms (2.13×) |
| `q5_0` | 28.4 ms | 25.2 ms | 14.3 ms (1.98×) | 11.9 ms (2.38×) | 12.6 ms (2.28×) | 11.4 ms (2.49×) |
| `q4_0` | 24.1 ms | 19.8 ms | 12.4 ms (1.94×) | — | 10.3 ms (2.33×) | 10.1 ms (2.38×) |
| `q4_1` | 20.3 ms | 16.9 ms | 11.4 ms (1.77×) | 10.5 ms (1.93×) | 10.6 ms (1.91×) | 10.0 ms (2.04×) |
| `q5_K` | 31.6 ms | 27.7 ms | 13.9 ms (2.27×, 120 GB/s) | — | — | |
| `iq4_xs` | 34.9 ms | 26.6 ms | 13.7 ms (2.55×, 94 GB/s) | — | — | |

GB/s per shape for `q8_0` (and `q5_0`), best variant of each kind:

| Shape | `llama` | `pr25479` | best (c) | best (e) | best (f) |
|---|---|---|---|---|---|
| `attn_q` | 63 (58) | 99 (73) | 116 (120) | 149 (148) | 155 (138) |
| `attn_kv` | 56 (58) | 85 (70) | 89 (87) | 108 (104) | 117 (98) |
| `attn_output` | 99 (80) | 102 (79) | 104 (119) | 136 (154) | 156 (144) |
| `moe_gate_up` | 78 (53) | 90 (65) | 116 (117) | 147 (150) | 160 (138) |
| `moe_down` | 81 (44) | 43 (40) | 105 (118) | 153 (144) | 146 (132) |
| `shexp_gate_up` | 81 (62) | 89 (62) | 110 (105) | 136 (138) | 151 (130) |
| `shexp_down` | 42 (68) | 61 (56) | 94 (103) | 127 (128) | 138 (117) |
| `output` | 63 (60) | 103 (75) | 127 (134) | 158 (158) | 165 (143) |

What this shows:

1. **Q8_0 was held back by its loads, not by its arithmetic.** With the same blocks staged through shared memory, Q8_0
   goes from 1.53× to 2.07× and is the fastest format per byte. In (c), each lane's 32-bit loads at a 68-byte stride
   touch a new cache line almost every time; staged, the lane group reads its segment contiguously.
2. **Staging needs no repack.** (f) matches or beats the row-local planes (e) for Q8_0 and Q4_0 and comes within 6 %
   for Q5_0, without touching the weights in memory, so prompt processing (MMQ), row lookup and dequantization keep
   reading the GGUF layout. Row-local planes in turn match the tensor-wide planes of (d) (`q4_1`: 10.5 against
   10.8 ms), so a repack, if it ever comes, can keep each row in place.
3. **Q5_0's fifth bit is cheap once it stays in place.** The tuned Q5_0 kernel splits each block's sum into
   sum(nibble × a) + 16 × sum(bit × a), with `(qh >> j) & 0x01010101` against activations regrouped by two 4 × 4 byte
   transposes per block, and reaches 1.98× (2.28× staged) against llama.cpp's per-bit shift/and/or. The gate (≥ 1.5×
   on the gate/up shape) is passed: 117 against 53 GB/s.
4. **K- and I-quants pass their gates** (Q5_K ≥ 110, IQ4_XS ≥ 90 GB/s): Q5_K 120 GB/s overall (123 on the expert
   gate/up, 125 on the expert down-projection), IQ4_XS 94 (93, 102). IQ4_XS's fused gate/up needs one row per lane
   group: at two rows it takes 255 registers and drops to 40 GB/s.
5. **llama.cpp's small-K shape for IQ4_XS (Q47) cuts both ways.** IQ4_XS's 256-weight blocks put every row up to
   K = 2048 under the small-K rule. Allowing it doubles the down-projections (`moe_down` 14 → 29 GB/s, `shexp_down`
   25 → 50) and the unfused K = 2048 products (`attn_q` 43 → 81) but slows the fused gate/up (52 → 34): a one-line
   change would need to exclude fused calls.

## Files

| File | Contents |
|---|---|
| `src/common.cuh` | block layouts (identical to ggml's), type traits |
| `src/llama_mmvq.cuh` | (a)/(b): llama.cpp's MMVQ and `quantize_q8_1`, copied from `d77256999` (MIT, the ggml authors), Pascal paths only |
| `src/pascal_mmvq.cuh` | (c) tuned kernel (all seven types), (d) SoA kernel, SoA activation quantizer and repack, (e) row-local-plane kernel and repack, (f) staged kernel |
| `src/data.cuh` | random weight/activation generation on the GPU, CPU reference dot products |
| `src/main.cu` | shapes, variants, correctness check, timing, totals, CSV |
| `build.sh` | Docker build of `build/mmvq_pascal` |
| `verify/` | bit-exactness check against a static llama.cpp build |
