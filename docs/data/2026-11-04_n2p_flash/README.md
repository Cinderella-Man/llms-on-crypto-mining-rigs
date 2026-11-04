# Q928 / Q220: N2P and N3P for Qwen3.8-Flash-Next, with 128-weight variants for its 640-wide down projections (2026-11-04)

N2P and N3P are this project's own 2- and 3-bit formats for Pascal ([Q164](../../research/notes/pascal_lowbit_format.md),
[Q185–Q189](../2026-10-27_lowbit/README.md)): a handful of non-uniform int8 levels and a shifted copy held in registers,
decoded with one byte permute per 4 weights, so a P106 reads a weight in 3.8–4.7 ps at one token, Q4_0's speed. On
Ornith-1.5-122B a file with N2P/N3P experts was better than Unsloth's UD-Q2_K_XL on quality and 1.28× faster on the agent
check ([Q189](../2026-10-28_lowbit_rig/README.md)). This folder brings them to Swift-1.5 Qwen3.8-Flash-Next (hidden 2560,
512 routed experts of width 640, 10 used per token):

- **the types on the current engine:** the fork branch `pascal-p106-v3-lowbit` (ids 500, 501) ported onto the newest
  kernels, `pascal-p106-v3-b12076-kshape`, as `pascal-p106-v3-b12076-n2p`;
- **N2P-128 and N3P-128 (Q220):** the same formats in 128-weight blocks. Flash-Next's routed down projections have
  640-wide rows, which hold no whole 256-weight block, so before this only Q2_0 (2.25 bits) and IQ4_NL (4.5) fit them;
- **the multi-token expert kernel** for all four types: 2–8 tokens (a speculative verify) read each distinct expert once
  for every token that picked it;
- **two Flash-Next files:** the routed experts requantized from the Swift-1.5 Q8_0 with FN4's importance matrix (gate and
  up N2P; down N2P-128 in one file, N3P-128 in the other), every other tensor pmix-q2-fast's byte for byte.

## The formats

| Type | Id | Weights a block | Bytes | Bits a weight | Levels | Scales | Block alignment |
|---|---:|---:|---:|---:|---|---|---|
| N2P | 500 | 256 | 86 | 2.6875 | IQ2_K's 4 and a +5 copy | 4-bit signed + shift bit per 8 weights, FP16 per block | 2 B |
| N3P | 501 | 256 | 104 | 3.25 | IQ3_K's 8 and a +4 copy | 5-bit signed + shift bit per 32, FP16 per block | 8 B |
| **N2P-128** | 502 | 128 | 44 | 2.75 | as N2P | as N2P | 4 B |
| **N3P-128** | 503 | 128 | 54 | 3.375 | as N3P | as N3P | 2 B |

The 128-weight types are N2P and N3P with half the codes, scales and shift bits and their own FP16 scale; the quantizer
is the same code with the block size as a parameter (N2P and N3P come out byte for byte as before). Their 32-bit
fields come first, so every field is word-aligned relative to its block: N2P-128 is codes (8 words), scales (2 words),
then the shift bits and d sharing the last word; N3P-128 is the 2-bit plane (8 words), the 1-bit plane (4 words), then
N3P's first word (d, shift bits, bit 4 of the scales) and its 4-bit scale half-word. A 640-wide row is five blocks: 220
bytes in N2P-128, 270 in N3P-128 (Q2_0: 180). `llama-quantize` now falls back from N2P and N3P to their 128-weight types
where a row is a multiple of 128 wide (else to IQ4_NL as before), so `--tensor-type ffn_down_exps=n2p` also gives N2P-128.

## The code (fork branch `pascal-p106-v3-b12076-n2p`)

| Commit | What |
|---|---|
| `c1266a056` | the types (ids 500, 501), CPU reference, importance-matrix quantizer, CPU dot products (from `-lowbit` `d2c86e307`; Q4_1_64, retired on this base, dropped; upstream's `best_index_int8` is gone, so the quantizer has its own) |
| `4127220b1` | CUDA dequantize, get_rows, llama.cpp's matrix-vector kernel, the Pascal one-token and 2–4-token kernels (from `f40132341`, merged beside the grid types' and Q2_0's tile staging of this base) |
| `e51b90f02` | MMQ tile loaders (from `67181fcd9`; upstream's loaders take a gate argument now) |
| `db9b059ed` | test-backend-ops perf cases at Ornith's shapes (from `2942ef390`, `37aae03b2`, `5fd8ffeec`) |
| `0b8427657` | N2P-128 and N3P-128: CPU reference and quantizer, CPU dot products against Q8_0, gguf-py ids, the quantize fallback |
| `c722aa389` | N2P-128 and N3P-128 on CUDA: dequantize, get_rows, llama.cpp's kernel, the Pascal kernels (warp-tile staging of the 640-wide rows, as Q2_0's), MMQ |
| `efce6a24e` | the multi-token expert kernel for N2P, N3P, N2P-128 and N3P-128 (`mmvq-pascal-mtok-np.cuh`) |
| `a85c08643` | test-backend-ops cases at Flash-Next's shapes, 1–8 tokens and prompt batches, eval and perf |
| `1a0832375` | the CPU dequantizers read through const views (no warning; output unchanged) |

**One token** (`mmvq-pascal.cu`): N2P and N3P as before (N2P staged as a half-word type, N3P like Q4_K). N2P-128 loads
44-byte blocks as words; N3P-128 is a half-word type (54-byte blocks at 2-byte boundaries). At K = 640 a row is five
blocks, one lane per row, so the row segments are never whole 16-byte (N2P-128) or 4-byte (N3P-128) units: like Q2_0's
down projection (Q825), each warp stages its 32 rows at once as a tile with coalesced 128-bit loads and every lane reads
its block from shared memory. With a fused gate N2P-128 loads directly.

**2–8 tokens of experts** (`mmvq-pascal-mtok-np.cuh`): the grid types' multi-token kernel (Q713) generalized to the N
types and to 128-weight blocks: each distinct expert of the step is read once, each 32 weights decoded once into int8
words and multiplied by every token that picked it (8 dp4a a token), with the same integer sums, the same FP32
operations in the same order and the same lane reduction as the one-token kernel, so every token's result is
bit-identical to it. Row segments are staged with the widest loads the rows allow; N3P-128's 270-byte rows (no gate)
as warp tiles. `GGML_CUDA_MMVQ_PASCAL_MTOK=0` runs the earlier kernels.

**Prompt batches** (MMQ): the N2P/N3P tile loaders take the block type; a 256-weight MMQ iteration spans two 128-weight
blocks. Both load into the Q8_0 tile layout as N2P and N3P do.

## Status

**Built and correct on the CPU; the GPU checks, the files' KLD and the rig timing follow** (below).

- **Build:** the rig tarball (sm_61 code and compute_61 PTX, `TESTS=1`, no AVX, GLIBC 2.38) of `1a0832375` compiles
  cleanly. The files were made with the CPU tools of `0b8427657` (`llama-quantize`), whose quantizer is the final
  branch's.
- **CPU check** (`test-quantize-fns`, every type, a build without AVX like the rig's): all four pass
  ([cpu_check.tsv](cpu_check.tsv)). N2P-128 and N3P-128 quantize random data a little more closely than N2P and N3P
  (0.004376 against 0.004383, 0.002464 against 0.002475: one FP16 scale per 128 weights instead of 256), and their
  reference and fast quantizers agree exactly. test-backend-ops compares a GPU with the CPU backend, so its runs are
  the GPU checks below.

## Files

| File | What |
|---|---|
| [run/build.sh](run/build.sh) | the builds on the quant machine: the rig tarball (sm_61 + compute_61 PTX), the CPU tools and test-quantize-fns, a compute 8.6 llama-perplexity and test-backend-ops for the KLD and the native MMQ check |
| [run/gpu_checks.sh](run/gpu_checks.sh) | test-backend-ops runs per switch setting (MODE=ptx on another card, MODE=native on a P106) |
| [run/quantize.sh](run/quantize.sh) | the two files: llama-quantize of the experts, then gguf_compose onto pmix-q2-fast, `verify`ed |
| [run/rig_tk.sh](run/rig_tk.sh) | on a rig: the native check (MMQ included), per-kernel perf at Flash-Next's expert shapes, T(k) of the files |
