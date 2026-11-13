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
| `q6_K` | 26.25 | 210 bytes per 256 weights: 128 bytes `ql` (low 4 bits), 64 bytes `qh` (high 2 bits), 16 int8 scales (one per 16 weights), `half d`. Value `d*scale*(q - 32)`. The output head of most Q4 files. |
| `q4_K` | 18 | 144 bytes per 256 weights: `half2 (d, dmin)`, 12 bytes of 6-bit sub-block scales and mins (packed as in Q5_K), 128 bytes of nibbles; sub-blocks 2j and 2j+1 are the low and high nibbles of the same 32 bytes. Value `d*sc*q - dmin*m` per 32-weight sub-block. The main type of Q4_K_M files. |

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
| `st_r1c2`, `st_r2c2`, `st_r4c2`, `st_r2c4`, `st_r1c1`, `st_r2c1` | (f) staged loads (`q8_0`, `q4_0`, `q4_1`, `q5_0`; `c1` for `q5_K`, `iq4_xs`): the unchanged GGUF blocks; each lane group first copies the contiguous row segment its lanes are about to read into shared memory with coalesced 128-bit loads, then every lane reads its own blocks from there. No repack. |
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
| `--types q4_1,q8_0` | subset of weight types (default `q4_1,q4_0,q8_0,q4_1_64,q5_0,q5_K,iq4_xs,q6_K,q4_K`) |
| `--variants llama,p_r2c2` | subset of variants (default all) |
| `--shapes attn_q,moe_down` | subset of shapes (default all) |
| `--reps 30`, `--batch 64` | timed graph launches, calls per graph |
| `--verbose` | print the largest deviations the correctness check finds |
| `--list` | list shapes and variants |
| `--spread N`, `--spread-gb X` | N weight sets per shape, or as many as fill X GB, cycled call by call, with fresh expert ids for every call (Q99, below) |
| `--tlb-probe` | pointer-chase probe of the GPU's address translation instead of the kernels (Q99, below) |
| `--f16` | F16 one-token products (the MoE router) instead of the quantized kernels (Q97, below) |
| `--bf16`, `--tokens 1,2,3,4` | BF16 products of 1 to 4 tokens at Qwen3.8-Flash-Next's shapes instead of the quantized kernels (Q546, below) |

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
CUDA-graph node costs 1.85 µs, and a plain streaming read reaches 179 GB/s, the practical ceiling. The CSVs of these runs are in [docs/data/2026-10-23_rig_step1](../../docs/data/2026-10-23_rig_step1/README.md) (first run), [docs/data/2026-10-23_q38](../../docs/data/2026-10-23_q38/README.md) (Q5_0, row-local planes, staged loads) and [docs/data/2026-10-23_q46](../../docs/data/2026-10-23_q46/README.md) (Q5_K, IQ4_XS).

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
| `q5_K` | 31.5 ms | 27.5 ms | 13.9 ms (2.27×, 120 GB/s) | — | 13.0 ms (2.42×, 128 GB/s) | 12.9 ms (2.44×) |
| `iq4_xs` | 34.7 ms | 26.6 ms | 13.6 ms (2.56×, 94 GB/s) | — | 12.3 ms (2.82×, 104 GB/s) | 12.0 ms (2.90×) |

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
4. **K- and I-quants pass their gates** (Q5_K ≥ 110, IQ4_XS ≥ 90 GB/s) with one super-block per lane: Q5_K 120 GB/s
   overall with direct loads (123 on the expert gate/up, 125 on the expert down-projection), IQ4_XS 94 (93, 102).
   Staged with one row per lane group they reach 128 and 104 GB/s; IQ4_XS's unfused products gain most (attn_q
   100 → 123, expert down 103 → 117), while its fused gate/up stays faster with direct loads (94 against 87). At two
   rows the fused IQ4_XS kernel takes 255 registers and drops to 40 GB/s.
5. **llama.cpp's small-K shape for IQ4_XS (Q47) cuts both ways.** IQ4_XS's 256-weight blocks put every row up to
   K = 2048 under the small-K rule. Allowing it doubles the down-projections (`moe_down` 14 → 29 GB/s, `shexp_down`
   25 → 50) and the unfused K = 2048 products (`attn_q` 43 → 81) but slows the fused gate/up (52 → 34): a one-line
   change would need to exclude fused calls.

## Q6_K (Q101)

`q6_K` is the 6-bit rung considered for the tensors that Flint's best files keep in Q8_0 (Q90: +0.014 KLD for all of
G3's Q8_0 tensors except the head, ~0.26 GB less per token). Its 210-byte super-blocks (128 bytes `ql`, 64 bytes `qh`,
16 int8 scales, FP16 `d`) are only 2-byte aligned, so it has only the staged variant (`st_r1c1`) besides llama.cpp's:

- One super-block per lane. The lane group stages its segment with 16-byte loads where the rows and the segment allow
  it, else with 8- or 4-byte loads (K = 1024 and 512 give 840- and 420-byte rows).
- A lane whose super-block starts at an odd half-word in shared memory reads the 53 words around it and shifts them by
  2 bytes (one `PRMT` per word).
- The fused gate/up stages the up and the gate segment one after the other in the same buffer: 27 KB of shared memory
  per block instead of 54 KB, which would not fit Pascal's 48 KB.
- The 6-bit codes are the low or high nibbles of `ql` OR-ed with a 2-bit field of `qh` moved to bits 4–5, one dp4a
  per 4 weights. The offset −32 comes from the activations' half-block sums (32 × the sum of 16 int8 values), computed
  once per chunk and shared by all rows, where llama.cpp spends a second dp4a per word with `0xE0E0E0E0`.

Static count in the sm_61 SASS, main loop per weight byte: 3.2 lane-instructions for a plain product and 2.9 for the
fused gate/up (Q5_K: 3.1 and 2.8), 3.5 and 3.9 with 8- and 4-byte staging. Q5_K's 3.1 runs at 120–128 GB/s, so Q6_K is
expected near 125–135 GB/s. Every Q6_K case passes the CPU reference at all eight shapes, through the compute_61 PTX
on an RTX 3090. The gate for the fork (Q101) is ≥ 130 GB/s on the dense K = 2048 shapes:

```bash
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --types q6_K,q5_K,q8_0 --csv /tmp/q101.csv' | tee q101_rig.txt
```

## Q4_K (Q149)

Q4_K_M files, what most people download, are mostly Q4_K and Q6_K: a token of Ornith-1.5-35B's Q4_K_M reads 57 % Q4_K
and 39 % Q6_K. ggml-org's Qwen3.6-35B-A3B Q4_K_M keeps its routed experts in Q4_K (90 % of the file, 22 % of a token's
2.6 GB), its attention, DeltaNet projections and shared expert in Q8_0 (58 %) and its output head in Q6_K (16 %).
`q4_K` is Q5_K without the fifth bit, so it gets Q5_K's variants (`p_r1c1`, `p_r2c1`, `st_r1c1`, `st_r2c1`):

- One 144-byte super-block per lane. 144 bytes are 9 × 16, so every row width stages and loads with 128-bit loads.
  Staged shared memory per block is 18 KB with one row per lane group, 36 KB with a fused gate or two rows (two rows
  with a gate would need 72 KB and are left out).
- Sub-blocks 2j and 2j+1 take the low and high nibbles of the same 8 code words. The low nibbles go through a signed
  dp4a on `q & 0x0F0F0F0F`; the high nibbles stay in place, an unsigned-by-signed dp4a on `q & 0xF0F0F0F0`, and the
  sub-block's sum, exactly 16 times too large, is shifted right by 4 once instead of shifting all 8 words. That is
  2 instructions per code word and sub-block, against llama.cpp's shift, mask and dp4a plus a second dp4a for the
  activation sum.
- The scales and mins decode 4 per word as Q5_K's do; the mins use the activations' block sums from the int8 values,
  computed once per chunk and shared by all rows.

Static count in the sm_61 SASS, main loop per weight byte: 2.7 lane-instructions staged and 2.5 with direct loads for a
plain product, 2.3 and 2.1 for the fused gate/up (Q5_K: 3.1 and 2.9, 2.8 and 2.6), with 96 and 48 registers for the
plain kernels and no spills. Q5_K runs at 120–128 GB/s with more work per byte, so Q4_K is expected at 125–140 GB/s.
Every Q4_K case passes the CPU reference at all eight shapes through the compute_61 PTX on an RTX 3090. The gate for
the fork (Q149) is ≥ 120 GB/s on the dense K = 2048 shapes and the expert shapes, together with Q6_K's ≥ 130 GB/s.
[`kquant_rig.sh`](kquant_rig.sh) runs both gates, the fork's `test-backend-ops` for Q4_K and Q6_K on a P106, and the
end-to-end A/B on the rig (Qwen3.6 Q4_K_M with each kernel switched off in turn, Flint UD-Q4_K_XL):

```bash
scp bench/mmvq_pascal/build/mmvq_pascal bench/mmvq_pascal/kquant_rig.sh rig:/tmp/q149/
ssh rig 'BIN=/opt/llama-builds/<pascal-p106-v2-kquant build>/bin MMVQ=/tmp/q149/mmvq_pascal QWEN=<Qwen3.6 Q4_K_M> FLINT=<UD-Q4_K_XL> bash /tmp/q149/kquant_rig.sh'
```

## The F16 router (Q97)

Flint's files at the quality bar keep the MoE router in F16: 256 rows of 2,048 weights, 1 MiB per layer, 47 layers per
token. llama.cpp multiplies it with `mul_mat_vec_f`, which gives each row a 256-thread block whose threads read 4 bytes
per iteration (~50 GB/s on a P106 in the S4-L0 profile). On compute 6.1 the fork's persistent MoE kernel (Q88) computes
the router inside its own launch, in `mul_mat_vec_f`'s summation order so that it matches the per-op path bit for bit:
one warp per row, each lane standing in for 8 of the 256 threads. `--f16` ([src/f16_mv.cuh](src/f16_mv.cuh)) times
three ways to multiply it:

- `mmvf`: `mul_mat_vec_f` as a compute 6.1 card runs it for one token (copied from llama.cpp `d77256999`; FP32
  accumulation);
- `moep`: the persistent kernel's router rows as a kernel of their own (work items of 2 rows per warp, the normed input
  computed on the fly from the layer input and the norm weight): its router phase if it had the whole card;
- `f16_rR`, a Pascal kernel with the quantized kernels' work split: a lane group sized to the row owns R rows; per
  iteration each lane reads 8 weights (16 bytes) of each row and the 8 matching FP32 activations (32 bytes), which the R
  rows share, all as coalesced 128-bit loads; FP32 accumulation, one FMA per weight in column order within the lane,
  then shuffles across the lane group.

The shapes are the router's (2048 → 256) and 2048 → 4096 for the steady-state bandwidth; copies of the weights cycle
call by call, and the check compares every row (every 37th of the large shape) with a double-precision product of the
same F16 weights and FP32 activations. The rig run decides whether the persistent kernel's router rows should take the
faster order (worth it if `moep` costs more than ~0.3 ms per token and `f16_rR` is clearly faster; the gate for `f16_rR`
is ≥ 110 GB/s at the router's shape):

```bash
ssh rig 'MMVQ=/tmp/q149/mmvq_pascal OUT=/tmp/q97 bash /tmp/q149/kquant_rig.sh router'
```

## Flash-Next's BF16 products (Q546)

Qwen3.8-Flash-Next keeps its hyper-connection projections, its indexer's query projection and its recurrent layers'
alpha and beta in BF16. On nine P106-090 cards they took 43 of the 136 ms of GPU time per decoded token
([profile](../../docs/data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)), most of it in the 320-wide rows of
`hc_attn_up` and `hc_ffn_up` (~0.33 ms per 6.5 MB call, ~20 GB/s). llama.cpp runs them with `mul_mat_vec_f`, which
gives each of the 10,240 rows a 160-thread block whose threads read 4 bytes once, so the card waits on memory latency
rather than bandwidth. `--bf16` ([src/bf16_mv.cuh](src/bf16_mv.cuh)) times, per shape and token count (`--tokens`):

- `mmvf`: `mul_mat_vec_f<nv_bfloat16>` as a compute 6.1 card runs it (FP32 accumulation, no fusion);
- `gG_rR`: Miner Llama's Pascal BF16 kernel (`ggml/src/ggml-cuda/mmvf-pascal.cu`, branch `pascal-p106-v3-bf16mv` at `a4acc80a3`,
  copied verbatim) at a fixed launch shape: G lanes per row group (8, 32, or 128 over the block's 4 warps), R rows per group
  sharing every activation load, 8 weights per lane per iteration with one 128-bit load, each weight converted exactly
  (BF16 is the upper half of an FP32) and added with one FMA;
- `pascal`: the same kernel at the launch shape the fork picks for the card (`G,R` column).

| Shape | K → N | MB per call | Calls per token | Fork's shape on a P106 (1–2 tokens) | 3–4 tokens (Q611) |
|---|---|---|---|---|---|
| `hc_up` | 320 → 10,240 | 6.55 | 97 (2 per layer, 1 for the output) | G = 8, R = 4 (160 blocks) | the same |
| `hc_down` | 10,240 → 320 | 6.55 | 97 | G = 128, R = 4 (80 blocks) | G = 32, R = 4 (20 blocks) |
| `hc_inject` | 10,240 → 4 | 0.08 | 96 | G = 128, R = 1 (4 blocks) | the same |
| `indexer_q` | 2,560 → 512 | 2.62 | 12 (the attention layers) | G = 32, R = 4 (32 blocks) | the same |
| `ssm_ab` | 2,560 → 48 | 0.25 | 72 (alpha and beta of the 36 recurrent layers) | G = 128, R = 1 (48 blocks) | G = 128, R = 2 (24 blocks) |

At 3 and 4 tokens the lane-group size from the row width stops at 32 and a short grid gets more lanes before fewer
rows (Q611, from the first sweep on a P106: at 2 rows per group the 10,240-wide rows ran 15–16 % faster at 32 lanes
than at 128). From 5 tokens R = 8 serves the rows where the grid allows it.

Copies of the weights cycle call by call (≥ 64 MB per shape), and every row (every 37th of `hc_up`) is checked against a
double-precision product of the same BF16 weights and FP32 activations, tolerance 1e-5 of the sum of the absolute terms.
The last table gives the milliseconds per decoding step summed over the shapes (`best` takes the fastest correct
`gG_rR` per shape). At the ~100–130 GB/s the tuned quantized kernels reach on this card, the Pascal kernel should take
~11–14 ms per token for all five shapes, against 43 ms in the profile. On the rig:

```bash
scp bench/mmvq_pascal/build/mmvq_pascal rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --bf16 --tokens 1,2,3,4,5,6,8 --csv /tmp/bf16.csv' | tee bf16_rig.txt
scp rig:/tmp/bf16.csv . && ssh rig 'rm -f /tmp/mmvq_pascal /tmp/bf16.csv'
```

On another GPU, `--bf16 --check-only --tokens 1,2,3,4,5,6,8` checks the compute 6.1 code through the PTX JIT. Since
Q591 the kernel takes 1 to 8 tokens, with 4 rows per lane group up to 4 tokens and 8 rows from 5 (variants `gG_r8`).

## Lossless BF16 (Q621)

`build/lossless_bench` ([src/lossless_main.cu](src/lossless_main.cu), built by `build_lossless.sh` with
`build/lossless_entropy`) times the BF16 products of the previous section over the model's own weights, in plain BF16
and in two lossless codes ([src/lossless_bf16.h](src/lossless_bf16.h)). Both keep each weight's low byte (exponent
bit 0 and the mantissa) and code its high byte (sign and exponent >> 1), which holds almost all of the redundancy:

- **x12**, 12 bits a weight: a 4-bit index into a 16-entry table per row (a window of 8 exponent pairs times the two
  signs). Two byte permutes and an add give 4 high bytes; 4 more interleave them with the low bytes into the 16 bytes
  the plain kernel would have loaded. Weights outside the window (~1 in 5,000) are kept as (column, BF16) per row and
  put in place before the products. Each lane decodes the plain kernel's own 8-weight chunks and runs the same FMA
  sequence, so **its output equals the plain kernel's bit for bit at the same launch shape**;
- **x11**, 11 bits a weight plus escapes: a 3-bit index (4 pairs times 2 signs), 32 weights packed into three words;
  one permute gives 4 high bytes, but 2–3 % of the weights fall outside its 8 exponents.

[src/lossless_mv.cuh](src/lossless_mv.cuh) is the Pascal BF16 kernel of `src/bf16_mv.cuh` with the weight load
replaced by the decoder (G = 8, 32, 128 lanes per row group for x12, also G = 2 for x11's 10 chunks of a 320-wide row;
R = 1, 2, 4, 8 rows). The weights are a sample the entropy tool writes (`lossless_entropy --sample`, see
[docs/data/2026-11-04_lossless_bf16](../../docs/data/2026-11-04_lossless_bf16/README.md)): 27 tensors of 8 shape
classes, coded once on the CPU and checked weight for weight, then copied on the card to ≥ 32 MB of plain BF16 per
class so that consecutive calls never hit the L2. Variants per class and token count (1–8): `plain/pascal` (the fork's
launch shape), `plain/gG_rR`, `x12/gG_rR`, `x11/gG_rR`, and `read/<code>`, a stream over the code's bytes per call with
nothing computed: the card's read speed for that size, the floor under the codes. Every variant is checked against a
double-precision product of the BF16 weights (every row up to 4,096 rows, else every 37th; tolerance 1e-5 of the sum
of the absolute terms), and each x12 output against the plain kernel's at the same (G, R), bit for bit (`same`). The
summary adds calls per token × the fastest correct variant of each code per class, against `plain/pascal`.

```bash
# CPU only, any machine: every weight of the sample through the kernels' permute steps
build/lossless_bench --sample sample.lbb --cpu-only
# on a newer GPU: the compute 6.1 code through the PTX JIT, no timing
CUDA_FORCE_PTX_JIT=1 build/lossless_bench --sample sample.lbb --check-only
# on the rig: docs/data/2026-11-04_lossless_bf16/run/rig_bench.sh copies, checks, runs and cleans up
```

## Flash-Next's Q2_0 experts (Q223)

The Swift Q2_0 file of Qwen3.8-Flash-Next stores its routed and shared experts in Q2_0: 18 bytes per 64 weights, one
FP16 scale and 2-bit codes for the values −1, 0, 1, 2. On nine P106-090 cards the routed experts took ~28 of the 136 ms
of GPU time per decoded token ([profile](../../docs/data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)),
~11 ps per weight. llama.cpp's `mul_mat_vec_q` runs these rows with its small-K shape (4 warps and 4 rows per block, one
32-weight half block per thread), which leaves 48 of a block's 128 threads idle on the 2560-wide gate/up rows and 108 on
the 640-wide down rows. `--q2_0` ([src/q2_0_mv.cuh](src/q2_0_mv.cuh)) times one token per shape:

- `llama`: llama.cpp's `mul_mat_vec_q` for Q2_0 as a compute 6.1 card runs it, with the fused gate and SwiGLU and the
  expert ids;
- `pascal`: Miner Llama's Pascal Q2_0 kernel (the Q2_0 path of `ggml/src/ggml-cuda/mmvq-pascal.cu`, branch
  `pascal-p106-v3-q2_0mv`, copied for one token) at the launch shape the fork picks: lane groups sized to the row, two
  blocks per lane, the codes looked up with `__byte_perm` in a 4-byte table (4 lookups and 3 shifts per 16 weights),
  the activations split into even and odd bytes once per load and shared by all rows of the lane group;
- `rR_seg`, `rR_tile`, `rR_dir`: the same kernel at R = 1 or 2 rows per lane group, with each lane group's row
  segments staged through shared memory (128-bit loads), with each warp's whole rows staged at once (a tile; for rows
  whose segments aren't 16-byte aligned), or with direct 32-bit loads.

| Shape | K → N | Experts | MB per call | Calls per token | Fork's shape |
|---|---|---|---|---|---|
| `moe_gate_up` | 2,560 → 640, gate and up fused with SwiGLU | 10 of 512 | 9.22 | 48 | 4 lanes per row, R = 2, segment staging |
| `moe_down` | 640 → 2,560 | 10 of 512 | 4.61 | 48 | 1 lane per row, R = 1, tile staging (5,760 bytes per warp) |
| `shexp_gate_up` | 2,560 → 640, fused | shared | 0.92 | 48 | 4 lanes per row, R = 1 (R = 2 would leave fewer than 4 blocks per SM: Q612), segment staging |
| `shexp_down` | 640 → 2,560 | shared | 0.46 | 48 | as `moe_down` |

The routed shapes hold all 512 experts and replay 64 random sets of 10 distinct experts call by call; the shared ones
cycle copies (≥ 64 MB), so no call finds its weights in L2. Every output of the first set is checked against a
double-precision product of the same codes and Q8_1 activations (tolerance 1e-5 of the sum of the absolute terms,
carried through SwiGLU). The last line sums 48 calls of each shape run (`best` takes the fastest correct Pascal
variant per shape). Gate/up and down together were ~23.7 ms per token in the profile (plus ~2 ms where the shared
expert's gate/up joins the launch); at 3–6 ps per weight, the estimate for the Pascal kernel, they would take ~7–14 ms.
On the rig:

```bash
scp bench/mmvq_pascal/build/mmvq_pascal rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --q2_0 --csv /tmp/q2_0.csv' | tee q2_0_rig.txt
scp rig:/tmp/q2_0.csv . && ssh rig 'rm -f /tmp/mmvq_pascal /tmp/q2_0.csv'
```

If an `rR_*` variant beats `pascal` clearly at a shape, change `mmvqp_choose` (gate R) or the host's tile choice in the
fork and `q2p_pick` here; `GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T` selects the same alternatives in the fork without a
rebuild. Several tokens per step (a speculative-decoding verify) run through the fork's multi-token kernel, which
`test-backend-ops perf -o MUL_MAT_ID -p type_a=q2_0` times with `GGML_CUDA_MMVQ_PASCAL=0` and `1`. On another GPU,
`--q2_0 --check-only` checks the compute 6.1 code through the PTX JIT.

## Flash-Next's experts in a k-token pass (R7, Q591)

A speculative verify or a pipelined guess runs k = 2 to 8 tokens through one pass of the model, and each token picks
10 of the 512 routed experts per layer. On nine cards the routed experts of such a pass cost 27.2 ms of GPU time at
k = 1 and 70.7 ms at k = 5 on llama.cpp's kernels ([Q583](../../docs/data/2026-11-03_verify_cost/README.md)): every
(token, slot) entry reads its expert again, whichever other token picked it. `--q2_0-mtok`
([src/q2_0_mtok.cuh](src/q2_0_mtok.cuh)) times one pass's gate/up (fused with SwiGLU) and down products at k = 1 to 8:

- `pertok`: Miner Llama's one-token Pascal Q2_0 kernel (`--q2_0`'s `pascal`) launched once per token, the cost
  without sharing at the best one-token speed;
- `mtok`: the fork's multi-token kernel (`mul_mat_vec_q2_0_pascal_mtok` of `ggml/src/ggml-cuda/mmvq-pascal-mtok.cu`,
  branch `pascal-p106-v3-b12076-mtok`): one launch per pass; the entries that picked one expert form a group whose
  block reads the expert's rows once and applies each decoded weight to every member; each block stages its members'
  activations in shared memory once and loops over 4 row blocks; lane groups, staging and sums as in the one-token
  kernel, so every output is bit-identical to `pertok`'s;
- `mtok_c1`: groups of one entry (no sharing: the kernel's own cost per entry), `mtok_c4`: at most 4 entries per group
  (less shared memory per block at k = 5 to 8), `mtok_rb1`, `mtok_rb2`, `mtok_rb8`: row blocks per block,
  `mtok_gdir`: the gate/up weights loaded directly instead of staged, `mtok_mb4`: compiled for 4 resident blocks per
  SM (at most 128 registers; the fork's `MMVQP_MTOK_MIN_BLOCKS`) instead of 3.

How much a pass shares depends on the routing, and no routing trace of Flash-Next exists yet. Route profiles:
`random` (each token's 10 experts uniform and independent: 19.8, 38.8 and 74.7 distinct experts at k = 2, 4, 8),
`local` (popularity skewed so that a quarter of the experts take 44 % of the picks, and each expert of the token
before kept with probability 0.16: Flint's measured routing in Q51, assumed for Flash-Next), `copy` (the same with
0.6: highly repetitive text, an upper bound on sharing) and `trace` (`--route-trace FILE`: one token per line, its 10
expert ids, as `bench/expert_overlap` dumps them for one layer). The table prints the mean distinct experts per pass
(`dist`) next to each time; GB/s counts only the distinct experts' bytes. The last table sums 48 layers of both shapes
per pass, for `pertok`, `mtok` and `mtok_c1`. Step 0 of every case is checked against a double-precision product and
every multi-token output against `pertok`'s bit for bit. On the rig:

```bash
scp bench/mmvq_pascal/build/mmvq_pascal rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --q2_0-mtok --csv /tmp/q2_0_mtok.csv' | tee q2_0_mtok_rig.txt
scp rig:/tmp/q2_0_mtok.csv . && ssh rig 'rm -f /tmp/mmvq_pascal /tmp/q2_0_mtok.csv'
```

The fork's `GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G` selects the same group size, row blocks and gate loads without a
rebuild (`=0`: the earlier kernels, for an A/B on one binary).

## k-column products (Q591)

A speculative-decoding verify multiplies every dense matrix by k = 2–8 tokens at once. Miner Llama's k-column kernels
(`ggml/src/ggml-cuda/mmvq-pascal-kcol*.cu`, fork branch `pascal-p106-v3-b12076-kcol`) read and decode each weight once
for all k columns: Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS, and Q4_0, Q5_0 and IQ4_NL (Q630, the shared
expert's other down types). The activations are quantized into records laid out for the kernels (int8 values in each
type's decoding order, FP32 scales, the K-quants' block sums, Q3_K's and Q6_K's half-block offsets, Q4_0's and Q5_0's
block offsets) that each block stages in shared memory once; per 32 weights, row and column a kernel spends 8 dp4a and
2–5 FP32 operations. The design is at the top of the kernel header, which `src/kcol_mm.cuh` copies verbatim.

`build_kcol.sh` builds `build/mmvq_kcol` ([src/kcol_main.cu](src/kcol_main.cu)), which times them per shape and column
count (`--tokens`, default 1–6 and 8) at the dense shapes of Qwen3.8-Flash-Next's Swift Q2_0 file (its tensor census:
`attn_qkv` 2,560 → 10,240 in Q3_K ×18, IQ4_XS ×13, Q4_K ×4, Q2_0 ×1; `attn_gate` 2,560 → 6,144; `attn_q`
2,560 → 12,288; `ssm_out` and `attn_output` 6,144 → 2,560; the shared expert's gate and up, and its down projection
640 → 2,560 in Q2_0 ×16, Q4_0 ×16, Q5_0 ×7 and IQ4_NL ×7; the output layer 2,560 → 248,320 in Q5_K and the draft's
Q6_K head; the hyper-connection projections 320 → 10,240 and 10,240 → 320 as the Q8_0 retype). Variants:

- `pascal`: the launch shape the fork picks (lanes per row group from the row width, 2 rows per group unless that
  leaves fewer than 3 blocks per SM, the largest activation tile that fits 48 KB of shared memory);
- `r1`, `r2`: 1 or 2 rows per lane group; `itt1`: one iteration per activation tile (the K-tile loop); `s32k`: a
  32 KB shared-memory budget (3 blocks per SM where registers allow);
- `prod`: the product alone, the records quantized beforehand, which shows what the quantizer costs.

Each call is the quantizer and the product, as in the model; copies of the weights cycle call by call (≥ 64 MB). Every
row (every 37th above 4,096 rows) of every column is checked against a double-precision product of ggml's reference
dequantization ([src/kcol_ref.h](src/kcol_ref.h)) and the activations read back from the records, tolerance 1e-5 of
the sum of the absolute terms. Per shape the table gives µs per call, GB/s and the cost against the same variant's one
column (`vs k1`, the flatness the kernels aim at: a 5-column pass at most ~1.3× one); the last table sums calls per
token × µs per call over the shapes run, per column count. On the rig:

```bash
scp bench/mmvq_pascal/build/mmvq_kcol rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_kcol --csv /tmp/kcol.csv' | tee kcol_rig.txt
scp rig:/tmp/kcol.csv . && ssh rig 'rm -f /tmp/mmvq_kcol /tmp/kcol.csv'
```

On another GPU, `CUDA_FORCE_PTX_JIT=1 build/mmvq_kcol --check-only --nsm 5` checks the compute 6.1 code with the
P106's launch shapes. `verify/kcol_unit_check.sh` checks each type's decoding against ggml's reference on a CPU, without
a GPU. The whole product path through ggml (llama.cpp's and the earlier Pascal kernels against these) is timed by
`test-backend-ops perf` with `GGML_CUDA_MMVQ_PASCAL_KCOL=0` and the default:
[docs/data/2026-11-04_kcol_dense](../../docs/data/2026-11-04_kcol_dense/README.md) has the run for one P106.

## Flash-Next's grid types (Q222)

Miner Llama's Pascal kernels (`mmvq-pascal.cu`, fork branch `pascal-p106-v3-b12076-iq2xs`) take five grid types: IQ2_S
and IQ3_XXS (Q166) and IQ2_XXS, IQ2_XS and IQ3_S (Q222), whose routed gate/up projections and attention fill Swift-1.5
Qwen3.8-Flash-Next's GSQ-RCO IQ2_XS, IQ3_XXS and IQ3_S files. `iq_perf.sh` times them inside ggml with the fork's
test-backend-ops at Flash-Next's shapes, 1–4 tokens, against llama.cpp's kernels in the same binary
(`GGML_CUDA_MMVQ_PASCAL_SKIP`): one routed gate or up projection (2,560 → 640, 10 of 128 experts per token), the fused
routed gate/up + SwiGLU (under `GGML_CUDA_PROFILE`, since perf mode repeats only a graph's last node), and `attn_qkv`,
`attn_q`, `attn_gate` (2,560 → 10,240, 12,288, 6,144) and `attn_output` / `ssm_out` (6,144 → 2,560); every call
reads 4 MB or more. `iq_perf_table.py` turns the logs into a CSV and `iq_perf.md` (µs per call off and on, speed-up,
GB/s at one token), and with `--layout` (a file's
[iq_layout.py](../../docs/data/2026-11-04_iq2xs_kernel/run/iq_layout.py) listing) into the milliseconds a token of that
file spends in these products. On the rig:

```bash
ssh rig "BIN=/opt/llama-builds/<build>/bin OUT=/tmp/iq_perf bash -s" < bench/mmvq_pascal/iq_perf.sh
scp -r rig:/tmp/iq_perf . && ssh rig 'rm -rf /tmp/iq_perf'
python3 -I bench/mmvq_pascal/iq_perf_table.py iq_perf --layout docs/data/2026-11-04_iq2xs_kernel/layout_swift-iq2_xs.txt > iq_perf.csv
```

`verify/iq_unit_check.sh <llama.cpp tree>` checks the three new types' decoding (the fork's `mmvq-pascal-iq.cuh`,
built for the host) against ggml's own dequantization on a CPU, without a GPU; `MUTATE=1` checks that it catches two
planted faults. Results: [docs/data/2026-11-04_iq2xs_kernel](../../docs/data/2026-11-04_iq2xs_kernel/README.md).

## Flash-Next's grid types in a k-token pass (Q713)

Fork branch `pascal-p106-v3-b12076-iqkcol` takes the grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and IQ1_M at
2–8 tokens into the k-column dense kernels above (Swift-1.5's IQ3_S attention, shared experts and `ssm_out`) and into a
multi-token expert kernel in the style of R7's Q2_0 one (`mmvq-pascal-mtok-iq.cuh`: each distinct expert of the step
read once for all the tokens that picked it; the routed gate/up projections of the IQ files). Both decode each
32-weight block once into 8 words of signed int8 weights (`mmvqp_iq::decode32` in `mmvq-pascal-iq.cuh`, which
`src/mmvq-pascal-iq.cuh` copies verbatim) and spend one dp4a per 4 weights and token; the expert kernel's results are
bit-identical to the one-token Pascal kernel's. Where the expert kernel takes a type, MUL_MAT_ID runs its 5–8-token
products there instead of llama.cpp's matrix-matrix kernels.

- `build_kcol.sh` with `IQ_TREE=<llama.cpp tree>` adds these types to `build/mmvq_kcol` (their grids come from that
  tree's `ggml-common.h` into `build/iq_grids.h`): the IQ3_S dense shapes of the Swift IQ2_XS file and each other type at
  `attn_gate`'s 2,560 → 6,144 (`--types iq3_s,...`; the pass total then covers the types run).
- `iq_kcol_perf.sh` times them inside ggml at 1–8 tokens against the kernels before Q713 in the same binary
  (`GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` and `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP` listing the types): the routed gate or up
  projection, the fused routed gate/up + SwiGLU and the dense attention shapes, as `iq_perf.sh`. `iq_kcol_perf_table.py`
  writes the CSV and `iq_kcol_perf.md` (µs off and on, speed-up, each run's k-token cost against its own one-token call,
  and with `--layout` the milliseconds of a k-token pass of a file's timed products).
- `verify/iq_kcol_check.sh <llama.cpp tree>` checks both kernels' decoding against ggml's own dequantization on a CPU,
  and the expert kernel's arithmetic bit for bit against the one-token functions; `MUTATE=1` plants three faults that
  must fail exactly IQ2_S, IQ3_S and IQ1_M.

```bash
ssh rig "BIN=/opt/llama-builds/<build>/bin OUT=/tmp/iq_kcol_perf bash -s" < bench/mmvq_pascal/iq_kcol_perf.sh
scp -r rig:/tmp/iq_kcol_perf . && ssh rig 'rm -rf /tmp/iq_kcol_perf'
python3 -I bench/mmvq_pascal/iq_kcol_perf_table.py iq_kcol_perf --layout docs/data/2026-11-04_iq2xs_kernel/layout_swift-iq2_xs.txt > iq_kcol_perf.csv
```

Results: [docs/data/2026-11-04_iq_kcol](../../docs/data/2026-11-04_iq_kcol/README.md).

## Memory system: spread weights and a TLB probe (Q99)

In the model, the ported kernels take 13–33 µs longer per call than in this benchmark, and the excess fits a constant
~1.3–2.0 µs per 2 MiB page a call touches ([Pascal forks and quants](../../docs/research/notes/pascal_forks_and_quants.md),
section 3): a card sweeps ~5.4 GB of weights per token, while Pascal's second-level TLB covers ~2 GiB (measured on
GP100 and GP104), and the benchmark's working sets stay under ~1.2 GB and are replayed. Two modes test that:

- `--spread N` or `--spread-gb X` (test E2): N weight sets of every matrix of a shape, or as many as fill X GB (at most
  4,096), up to `--spread-max-gb` (default 5) and the free memory. A MoE matrix's set holds all 256 experts, so N sets
  are N layers' worth; a dense shape needs many sets to sweep gigabytes (~950 of `attn_q` for 5 GB). The sets of a
  matrix lie back to back in one allocation, so they sweep one contiguous range of 2 MiB pages. Call j of a graph uses
  set j mod N (the graph then holds at least N calls), and before each graph launch one small kernel copies the
  next slot of a pool of expert ids into the buffer the calls read, so every call of every launch picks fresh experts.
  With `--check-only` the results are checked against the CPU reference and one id refresh against the pool. The CSV
  gains `weight_sets` and `swept_bytes`.
- `--tlb-probe` (test E1): one thread follows chains of dependent loads through L2: slots 2 MiB (the page size of
  `cudaMalloc`) or 32 MiB apart over footprints from 16 MiB to 5.5 GiB, and one slot in each of N random 2 MiB pages of
  the buffer (N = 8 … 2,560). After a warm-up pass the slots sit in L2, so a step costs an L2 hit while the
  translations hit and adds a page walk where they miss. Every chain runs alone and again while the other four SMs
  stream reads. `--tlb-max-gb` caps the buffer (default 5); with `--check-only` each chain is walked once around and
  checked, untimed.

Rig commands (one card, nothing else on it; the binary needs only the driver):

```bash
ssh rig 'mkdir -p /tmp/q99'
scp bench/mmvq_pascal/build/mmvq_pascal rig:/tmp/q99/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/q99/mmvq_pascal --tlb-probe --csv /tmp/q99/tlb.csv' | tee q99_tlb.txt
```
```bash
for gb in 0.5 1 2 3 4 5; do ssh rig "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/q99/mmvq_pascal --types q5_0,q8_0,q4_1 --variants st_r1c2,st_r2c2,st_r2c4 --shapes moe_gate_up,moe_down,attn_q,attn_output,output --spread-gb $gb --csv /tmp/q99/spread$gb.csv" | tee q99_spread$gb.txt; done
```
```bash
scp 'rig:/tmp/q99/*.csv' . && ssh rig 'rm -rf /tmp/q99'
```

The variants are the shapes `mmvqp_choose` picks in the fork for S4-L0's types: Q5_0 fused gate/up `st_r1c2`, Q8_0
`st_r2c2` (fused `st_r1c2`), Q4_1 `st_r2c4`. What decides:

- **TLB probe.** The footprint at which a 2 MiB-stride step gets slower is the TLB reach in pages; the 32 MiB-stride
  curve shows whether the L2 TLB counts 32 MiB regions (the second knee of GP100/GP104) or pages. The step time beyond
  the knee, minus the step time before it, is the page walk, alone and under load.
- **Spread.** Under the TLB model, a call at 0.5 GB (one set of the MoE matrices: 352 MiB of Q5_0 gate/up, 272 MiB of
  Q8_0 down) is no slower than in the default mode, and calls get ~1.3–2.0 µs slower per page they touch (16 for the
  gate/up call, 9 for the down call, ~3.5 for `attn_q`) once the sweep passes the L2 TLB's reach: +20–30 µs on the MoE
  shapes and ~+10 % on the head at ~5 GB. The research note's kill criterion: MoE calls at 5 GB less than 10 µs slower
  than at 0.5 GB.

## Several tokens per step: the fork's kernels

The fork's branch `pascal-p106-v2-multicol` extends the ported kernels to 2–4 tokens per step (a speculative-decoding
verify step), reading every weight row and every distinct expert once per step; the design is in
[pascal_mmvq_integration.md](../../docs/research/notes/pascal_mmvq_integration.md) section 14. These kernels are timed
with the fork's own `test-backend-ops` instead of this benchmark, so the measurement runs the code that llama.cpp
runs:

- `multitoken_perf.sh` runs on the rig: Flint's dense and expert products at 1–4 tokens on one card, with llama.cpp's
  kernels, the Pascal kernels, and the Pascal kernels with staged activations (`GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1`).
  `multitoken_perf_table.py` turns its logs into tables: time per call, speed-up, and the cost of an n-token call
  relative to one token.
- `verify/multitoken_bitid.cpp` (built by `verify/build_multitoken_bitid.sh` against a test build from
  `scripts/build_llamacpp_tests_docker.sh`) checks that every 2–4-token product equals the single-token products of
  its columns bit for bit. Its modes (the first argument) pick the types and shapes; the comment at the top of the file
  lists them. `BITID_CPU=1` runs a mode on ggml's CPU backend, a check of the harness itself that needs no GPU.

```bash
ssh rig "BIN=/opt/llama-builds/<build>/bin bash -s" < bench/mmvq_pascal/multitoken_perf.sh
```
```bash
scp -r rig:/tmp/mmvq_multitoken . && bench/mmvq_pascal/multitoken_perf_table.py mmvq_multitoken
```

## Files

| File | Contents |
|---|---|
| `src/common.cuh` | block layouts (identical to ggml's), type traits |
| `src/llama_mmvq.cuh` | (a)/(b): llama.cpp's MMVQ and `quantize_q8_1`, copied from `d77256999` (MIT, the ggml authors), Pascal paths only |
| `src/pascal_mmvq.cuh` | (c) tuned kernel (all eight types but Q6_K), (d) SoA kernel, SoA activation quantizer and repack, (e) row-local-plane kernel and repack, (f) staged kernel (all nine types) |
| `src/data.cuh` | random weight/activation generation on the GPU, CPU reference dot products |
| `src/memsys.cuh` | `--tlb-probe`: pointer chases over strides and random pages, alone and under a streaming load |
| `src/main.cu` | shapes, variants, correctness check, timing, totals, CSV |
| `build.sh` | Docker build of `build/mmvq_pascal` |
| `verify/` | bit-exactness check against a static llama.cpp build; `multitoken_bitid.cpp`: the fork's 2–4-token products against its single-token ones; the kernel-check kit of Q149: `tbo.sh` and `run_suite.sh` (operator tests through the compute_61 PTX: forced, off, a mutation build), `q4k-mut.diff` (the mutation), `tbo_tally.py` (which cases failed), `sass_cmp.sh` (every shared kernel's SASS identical between two builds), `benchloops_k.py` and `forkloops.py` (main-loop instructions per weight byte) |
| `multitoken_perf.sh`, `multitoken_perf_table.py` | timing of the fork's kernels at 1–4 tokens per step on the rig, and its tables |
| `kquant_rig.sh` | the Q4_K and Q6_K gates, the fork's Q4_K/Q6_K tests on a P106, and the tg128 A/B of the two kernels on the rig (Q149) |
| `src/f16_mv.cuh` | `--f16`: llama.cpp's F16 one-token product, the persistent MoE kernel's router rows and the Pascal F16 kernel (Q97); `kquant_rig.sh router` runs it on the rig |
| `src/bf16_mv.cuh` | `--bf16`: llama.cpp's BF16 `mul_mat_vec_f` and Miner Llama's Pascal BF16 kernel with its launch-shape rule, copied from the fork (Q546) |
| `src/lossless_bf16.h`, `src/lossless_entropy.cpp`, `src/lossless_mv.cuh`, `src/lossless_main.cu`, `build_lossless.sh` | the lossless BF16 codes x12 and x11 (Q621): encoders and the CPU decoder that runs the kernels' byte permutes; `build/lossless_entropy`, the byte entropy and code sizes of a model's BF16 matrices and the bench's weight sample (CPU); the decoding Pascal BF16 kernels; `build/lossless_bench`, their timing against the plain kernel at 1–8 tokens |
| `src/q2_0_mv.cuh` | `--q2_0`: llama.cpp's Q2_0 `mul_mat_vec_q` and Miner Llama's Pascal Q2_0 kernel at one token (Q223) |
| `src/q2_0_mtok.cuh` | `--q2_0-mtok`: Miner Llama's multi-token Q2_0 expert kernel, copied from the fork, its launch rule, and the route profiles (R7, Q591) |
| `tools/` | SASS and register helpers for the fork's Pascal kernels: `regs.py`, `loops.py`, `loops2.py` (registers, spills and main-loop instruction counts from a ptxas log and `cuobjdump -sass`), `classify.py` (sorts a test-backend-ops log's failures into the cases a mutation should hit) (Q78); `benchloops.py` (the bench's staged-kernel loops per weight byte), `forksass.py` (SASS identity of two single-file builds of the fork's `mmvq-pascal.cu`, Q6_K registers), `sasscmp.py` (SASS identity of two bench builds) (Q101) |
| `src/kcol_mm.cuh`, `src/kcol_main.cu`, `src/kcol_ref.h`, `build_kcol.sh` | `build/mmvq_kcol`: Miner Llama's k-column products (Q591, copied verbatim from the fork), timed at 1–8 columns at Qwen3.8-Flash-Next's dense shapes against ggml's reference dequantization |
| `verify/kcol_unit_check.cu`, `verify/kcol_unit_check.sh` | each type's decoding in the k-column products against ggml's reference dequantization, on a CPU (no GPU) |
| `iq_perf.sh`, `iq_perf_table.py` | the five grid types' products at Flash-Next's shapes inside ggml, Pascal kernels against llama.cpp's, and the tables (Q222) |
| `verify/iq_unit_check.cu`, `verify/iq_unit_check.sh` | IQ2_XXS, IQ2_XS and IQ3_S decoding of the fork's Pascal kernels against ggml's own dequantization, on a CPU (no GPU); `MUTATE=1` the check of the check |
| `src/mmvq-pascal-iq.cuh` | the fork's ggml-free decoding of the grid types and IQ1_M (copied verbatim), which `src/kcol_mm.cuh` includes (Q713) |
| `iq_kcol_perf.sh`, `iq_kcol_perf_table.py` | the grid types' and IQ1_M's products at 1–8 tokens inside ggml, the Q713 kernels against the kernels before them, and the tables (Q713) |
| `verify/iq_kcol_check.cu`, `verify/iq_kcol_check.sh` | the grid types' and IQ1_M's decoding in the fork's k-column and multi-token kernels against ggml's own dequantization, on a CPU (no GPU); `MUTATE=1` the check of the check (Q713) |
