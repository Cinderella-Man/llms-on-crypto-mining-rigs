# Flint's shared expert on the P106 rig: fusing its gate/up (Q50) and a seventh-expert design (Q37) (2026-10-23)

Scope: the kernel launches that Flint's always-on shared expert costs per generated token in our llama.cpp fork (branch
`pascal-p106`), why its gate and up products did not fuse, the fix (fork branch `pascal-p106-shexp`, commit
`076274d61`), and a design for running the shared expert inside the routed experts' launches the way Recursive's own
engine does ([rec](rec_engine.md), MoE section). Flint: 49 layers, layers 0–1 dense (SwiGLU, intermediate 6,144),
layers 2–48 MoE with 256 routed experts of intermediate 512, top-6, plus a shared SwiGLU expert of intermediate 1,024
added with weight 1. "Claim to test" marks anything not measured on the rig; "(est.)" marks an estimate.

## Launches in one MoE layer at one token

Q = quantization of the activations to Q8_1 (one small kernel), P = matrix-vector product kernel.

| Step | Fork `eeaa59660` | With Q50 | Seventh expert (design below) |
|---|---|---|---|
| FFN norm (RMS_NORM+MUL, fused) | 1 | 1 | 1 |
| Router product (F16 or BF16) | 1 | 1 | 1 |
| Sigmoid, top-6, normalization (fused) | 1 | 1 | 1 |
| Routed gate/up + SWIGLU (fused) | Q + P | Q + P | Q + P, with the shared rows |
| Routed down | Q + P | Q + P | Q + P, with the shared rows |
| Weighted sum of the six experts (fused) | 1 | 1 | 1, with the shared output |
| Shared gate, shared up | Q + P, Q + P | Q + P (fused with SWIGLU) | – |
| SWIGLU | 1 | – | – |
| Shared down + add to the routed sum (fused) | Q + P | Q + P | – |
| **Kernels** | **16** | **13** | **9** |

Per token that is 47 MoE layers × 3 kernels for Q50 (plus one dense layer, see below) and another 47 × 4 for the
seventh expert.

## Q50: why the shared gate/up did not fuse, and the fix

- **Not the node order.** In Flint's graph the shared expert's gate product, up product and SWIGLU are adjacent nodes,
  and the CUDA backend's `{MUL_MAT, MUL_MAT, GLU}` matcher accepts gate and up in either order.
- **The fusion memory check refused it.** Before fusing, `ggml_cuda_check_fusion_memory_ranges` rejects a group whose
  output overlaps the memory of any input that is not one of the fused nodes. llama.cpp builds the shared expert after
  the routed experts, so its up product is the last reader of the FFN input (2,048 floats). ggml-alloc frees that
  buffer right after the up product and places the next allocation, the 1,024-float SWIGLU output, into it. The
  output now overlaps an input, and the group runs unfused: two products, two quantizations of the same activations
  and a SWIGLU kernel. The dense FFNs hit the same check whenever the layout happens to reuse the input's buffer: in a
  profile of S4-L0 on six cards, layer 1's dense FFN was split and layer 0's fused.
- **The overlap is harmless for quantized weights.** `ggml_cuda_mul_mat_vec_q` first quantizes the activations into a
  separate pool buffer and the product kernel reads only that copy. On the same stream the product kernel starts after
  the quantization finishes; on Hopper and newer, where llama.cpp uses programmatic dependent launch, the product kernel
  waits in `cudaGridDependencySynchronize()` before it reads, which also waits for the quantization grid to finish.
  The float path (`mul_mat_vec_f`) reads the activations inside the product kernel, so it keeps the check.
- **The fix** (one function in `ggml-cuda.cu`): for gate/up/GLU groups with quantized weights, the memory check lets
  the activations alias the GLU output. Allocation is unchanged. The alternative, an allocation dependency added in
  `graph_optimize` (`add_alloc_dep`, which upstream uses for the routed + shared launch below), is exact too, but it
  keeps the FFN input alive one node longer in every graph, including prompt batches that never fuse (up to 4 MB more
  at `-ub 512`).
- **Measured on a laptop GPU** (Pascal kernels forced with `GGML_CUDA_MMVQ_PASCAL=2`, a 5-layer cut of Flint with
  S4-L0's FFN types: routed gate/up Q5_0, everything else Q8_0): 109 → 103 launch groups per token, every shared
  expert one `MUL_MAT+MUL_MAT+SWIGLU` group. Logits: KLD against the old build is the same as the old build against
  itself (the storage floor of llama-perplexity's base file, max 5.1·10⁻⁵), same top token 100 %, and perplexity
  differs only in the 14th digit, as much as two runs of the old build differ.
- **On the rig's 6-card S4-L0 graph** 48 gate/up pairs per token ran split (47 shared experts, layer 1's dense FFN):
  fusing them removes 96 of 1,020 launch groups per token, i.e. 48 products, 48 quantizations and 48 SWIGLU kernels.
  The event profile showed 28.7 + 29.7 + 9.2 = 67.6 µs per shared expert; the fused launch reads the same 4.5 MB, est.
  ~35–40 µs. Expected gain est. 0.8–1.3 ms per token, +3–5 % at 37 tok/s (claim to test).

## Q37: the shared expert as a seventh expert

### What already exists, and why Flint misses it

Upstream llama.cpp already runs a shared expert inside the routed gate/up launch: `graph_optimize` moves a shared
`{MUL_MAT, MUL_MAT, GLU}` next to the routed `{MUL_MAT_ID, MUL_MAT_ID, GLU}` (`ggml_cuda_match_shared_expert`), and
`mul_mat_vec_q` (and the fork's Pascal kernel, which follows it) gives the shared expert one extra grid channel that
reads the routed launch's quantized activations (`fusion.shared_up`, `shared_gate`, `shared_dst`). The match requires
the shared matrices to have the same type, row count and row stride as one routed expert. There is no such path for
the down projection.

Flint misses the match in every file on the row count (shared 1,024 rows, routed 512), and in most files on the type:

| File | Routed gate/up | Shared gate/up | Routed down | Shared down |
|---|---|---|---|---|
| S4-L0 | Q5_0 | Q8_0 | Q8_0 | Q8_0 |
| Unsloth Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| Makers' Q4_1 | Q4_1 | Q4_1 | Q4_1 | Q4_1 |
| UD-Q4_K_XL | IQ4_XS, Q5_K, Q4_K, IQ3_S per layer | Q8_0, Q5_K, IQ4_XS | IQ4_XS, Q5_K, Q6_K, Q8_0 | Q8_0, Q5_K, IQ4_XS |

In UD-Q4_K_XL the routed and shared types agree in only 2 of 47 layers for gate/up and 8 of 47 for down. S4-L0's
shared expert must stay Q8_0: at Q4_1 in half the layers it cost 0.019 KLD (registry Q45, X1).

### Design A: one extra channel with its own type and shape (keeps every file as it is)

1. **Gate/up phase.** Extend the extra channel of the Pascal kernel:
   - its own row count and row stride: the grid covers the larger of the two row counts, and blocks past a channel's
     last row return at once;
   - its own weight type: the kernel body becomes a device function templated on the weight type and launch shape,
     and the global kernel calls the routed or the shared instantiation per block (`blockIdx.y == gridDim.y - 1`).
     Shared memory is the larger of the two, registers the larger of the two. Instantiate only the pairs that occur
     (Q5_0 + Q8_0, Q8_0 + Q8_0, Q4_1 + Q4_1, and UD-Q4_K_XL's pairs among the six Pascal types);
   - both channels read the same Q8_1 activations; each type's own activation regrouping (Q5_0's fifth-bit transposes,
     IQ4_XS's permutes) already runs per block;
   - `ggml_cuda_match_shared_expert` accepts a different type and row count when the Pascal kernel will run.

   Saves one quantization and one product launch per layer.
2. **Down phase.** One product launch for the six routed experts (K = 512) and the shared down projection (K = 1,024),
   then the weighted sum adds the shared output with weight 1:
   - the down launch quantizes its input once, so the routed and shared SWIGLU outputs must be one buffer: the
     gate/up launch writes the shared output into two extra 512-wide slots after the six routed ones (a graph change
     that makes the routed SWIGLU output 8 slots wide), or the quantization reads two sources;
   - the down kernel's extra channel takes its own K and row stride; S4-L0's down projections are Q8_0 on both sides,
     so this phase needs no second type for it;
   - the weighted-sum kernel takes a seventh input with weight 1 and adds it last, so the order
     ((e0·w0 + … + e5·w5) + shared) stays as it is today: the output stays bit-identical.

   Saves one quantization and one product launch per layer; the shared down's fused add disappears into the sum.

### Design B: the shared expert as two extra routed experts (Q37's GGUF rewrite)

- Append the shared expert to the routed tensors as experts 256 and 257: rows 0–511 and 512–1,023 of gate and up,
  columns 0–511 and 512–1,023 of down. Both splits fall on block boundaries, so a tensor whose type already matches is
  copied byte for byte.
- The fused top-k kernel writes ids [top-6, 256, 257] and weights [w0 … w5, 1, 1], with the two fixed weights left out
  of the normalization; the products run with 8 experts; the existing kernels need no change.
- The catch is one type per tensor. S4-L0 would need its shared gate/up in Q5_0 (requantized from BF16, a quality loss)
  or all routed gate/up in Q8_0 (registry Q45 GQ8: better quality, but ~42 GB and ~1 ms per token slower).
  UD-Q4_K_XL cannot be rewritten without requantizing most layers. S4-L0's down is Q8_0 on both sides and carries over.
- Splitting the shared down projection's K into two halves changes the summation order: noise-level, not exact.

### Comparison and decision

| | A: kernel | B: GGUF rewrite |
|---|---|---|
| Files | any type mix, no rewrite | one type per tensor; S4-L0 and UD-Q4_K_XL change |
| Output | bit-identical (shared summed last, K not split) | changes (type, K split) |
| Code | Pascal kernel body and dispatch, the matcher, the down input buffer, the weighted sum | top-k kernel, Flint graph, GGUF rewrite script |
| Kernels saved | 4 per MoE layer, 188 per token | the same |

Design A is the one to build: B gives the same launch saving but changes the files we would measure. Expected gain
est. 0.4–0.8 ms per token (+1.5–3 % at ~40 tok/s) on top of Q50, mostly the two quantization kernels and two launch
latencies per layer (claim to test).

Design A is not built yet because it lands on the same code as the FP32 activation-sum change (registry Q38, item 6)
that is in progress: the gate/up phase moves the Pascal kernel body into a per-type device function and reads the
activations in whatever block format that change settles on, and the down phase changes the quantization step. Build
it on top of the FP32-sum branch once that is final, needing from it: the activation block layout and per-type
activation loader, the quantization entry point (an output offset or two sources), and the kernel body as merged.
