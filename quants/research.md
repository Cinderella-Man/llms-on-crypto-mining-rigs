# Making a model decode faster on P106-090 cards

What can be done to a model file, its weight formats, or by retraining parts of it, so that it generates text faster on
the rig: nine P106-090 cards with 5 streaming multiprocessors (SMs) each ([hardware](../docs/mining_rig.md)). Flint
([RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), 39.8 B parameters, ~2.5 B per token) is the worked
example; every lever is also checked against Qwen3.6-35B-A3B. Speed here means single-stream generation, the speed one
user sees. Prompt processing and serving several users at once are left out.

The levers fall into three classes:

- **A. File only.** Anything a GGUF file can hold that llama.cpp runs with at most a small, non-kernel patch: which
  format each tensor gets, how the weights are rounded or rotated, merged tensors, smaller heads, pruning.
- **B. New formats and kernels.** Block layouts and decode kernels written for 5-SM Pascal cards.
- **C. Retraining.** Draft models for speculative decoding, quantization-aware recovery, retraining after pruning.

In short:

- **The cards are limited by instructions, not memory bandwidth.** llama.cpp's matrix-vector kernel spends most of its
  instructions on address arithmetic, so a Pascal-specific kernel and launch shape are the biggest levers (expected
  +20–45 %), ahead of any format. Measuring that on the rig comes first.
- **Measured, file only, exact:** Flint's rescale step can go (Q8_0 down-projections in three layers, +8 %), the makers'
  Q4_1 fits a group-64 format (−9.4 % bytes), fused Q/K/V and an F16 router change nothing but speed. Combined:
  "makers-fast", 23.9 GB, same quality, expected ≥ 30 tok/s against 28.
- **Measured, mixes:** spending 8 bits only where each millisecond buys the most quality reaches Unsloth's UD-Q4_K_XL
  quality at a predicted 23 tok/s against its 18.8. Rotations, better rounding and a 4-bit head don't help; Q4_0 works
  once seven dominant-channel tensors stay 8-bit.
- **Quality has a noise floor on Flint:** rounding differences flip its expert choices, so the same file differs by KLD
  0.065 between two GPU models. On 300 tool-calling tasks every finalist is within noise of BF16; the makers' file, worst
  by KLD, is as good as any.
- **Qwen3.6:** its "Q4_0" file keeps a Q6_K head; as Q8_0 it is expected +13 % faster at identical quality, and a mix
  from its own sensitivity sweep beats the published file on both speed and quality.
- **Retraining:** an EAGLE-3 draft head could give 1.3–2×; a one-hour rig test decides whether 5–6 days of training are
  worth it.

The detailed research behind this page is in four notes: [file-only levers](../docs/research/notes/quant_speed_a_file_only.md),
[formats and kernels](../docs/research/notes/quant_speed_b_formats_kernels.md),
[retraining](../docs/research/notes/quant_speed_c_retraining.md) and
[community reports](../docs/research/notes/quant_speed_d_community.md). Every experiment, done or planned, is in
[experiments.md](experiments.md), and the words used here are defined in the [glossary](../GLOSSARY.md).

## The levers, ranked

Starting point: Flint's makers' Q4_1 file runs at 27.9 tok/s in `llama-bench` (tg128) and 24.3 tok/s on real chat
requests, using llama.cpp `d77256999` with CUDA graphs enabled on Pascal. "Expected" figures are estimates from the cost
model below and still have to be measured on the rig; "measured" figures were. Quality was measured for every file
lever against Flint's BF16 weights; the files combining the exact levers are in the [results table](README.md#results-so-far-flint).

| Lever | Class | Expected or measured effect | Quality | Effort | Qwen3.6 |
|---|---|---|---|---|---|
| A Pascal-specific matrix-vector kernel for the existing formats: several rows per warp, cheaper addressing, wide loads | B | expected +20–45 % (34–40 tok/s), for every file | exact | days (llama.cpp fork) | yes |
| An EAGLE-3 draft head for Flint (speculative decoding) | C | expected 1.3–2× in real chat, if verifying 2–3 extra tokens stays cheap on the rig | exact (only speed changes) | 2–4 days of training on a 7-GPU machine | has its own MTP head already |
| Pascal launch shape for the same kernel: 2 warps instead of 4 at one token, no idle warps for 512-wide inputs (PR [#25479](https://github.com/ggml-org/llama.cpp/pull/25479)) | B | expected +5–20 % | exact | hours | yes |
| Remove the MoE down-projection rescale: Q8_0 expert down-projections in the three layers where Flint's activations overflow (46–48), then `LLAMA_NO_MOE_DOWN_RESCALE=1` | A + small patch | measured +8 % (Q4_1) and +11 % (Q5_0) for the switch alone | exact: same KLD, no NaN in 41,000 tokens | done (file tool + patch) | no rescale there |
| A speed-aware mix: spend 8 bits only where each millisecond buys the most quality | A | measured quality, predicted speed: UD-Q4_K_XL's quality at 23.0 tok/s (Unsloth's file: 18.8), or 26.7 tok/s at KLD 0.219 | from KLD 0.120 to 0.219 | done (recipes) | yes |
| 4-bit attention and shared expert at UD-Q4_K_XL quality, via partial retraining | C | expected UD-Q4_K_XL quality at 1.5–1.7 GB per token instead of 2.3 (≈ +20 % over UD-mirror v1) | the open question; rotations don't do it | days | yes |
| Group-64 Q4_1 (`Q4_1_64`): the makers' file is group-64 data stored as group-32 | B | −9.4 % bytes per token; expected +2–4 % with today's kernel, ~+10 % with a Pascal kernel | exact for the makers' file (verified on GPUs) | prototype built | no |
| Fused Q/K/V tensor (`attn_qkv`) | A | expected +3–4 % (~200 fewer kernel launches per token) | exact (verified) | done (file tool) | partly |
| Router weights F32 → F16 in the makers' file | A | expected +1.8 % | noise level (verified) | done (file tool) | n/a (already 16-bit) |
| Vulkan backend instead of CUDA | engine | unknown; +11–47 % on a GTX 1060 (same GP106 chip) in [#19817](https://github.com/ggml-org/llama.cpp/issues/19817) | exact | hours (build) | yes |
| `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_DISABLE_FUSION=1` | engine switch | expected ±5 % each, never measured on Pascal | exact | minutes | yes |

Levers checked and not worth it on this rig:

- **Fewer experts per token** (measured): 6 → 4 gave +9 % speed for −7 points of same-top-token agreement.
- **K-quants, I-quants, NVFP4, MXFP4:** they decode slower per byte on these cards. NVFP4 needs ~2.6× the instructions
  per weight of Q4_1; the makers' NVFP4 file is expected at 17–20 tok/s against 28 for their Q4_1.
- **Q4_0 instead of Q4_1:** at most +6 %. Q4_0 saves 10 % of the bytes but costs more instructions per byte. It works on
  Flint once seven dominant-channel tensors stay at Q8_0 (measured, KLD 0.315 against Q4_1's 0.280).
- **Hadamard rotations** (measured): exact, but the rotated 4-bit files are slightly worse (Q4_1 KLD 0.308 against
  0.283 unrotated), because llama.cpp's 8-bit activation blocks absorb the spread-out outlier channels.
- **Better rounding (AutoRound, GPTQ-style error feedback):** gains at noise level for 4-bit block formats; it cannot
  bring Q4_1 to Q5_0's quality.
- **A 4-bit output head** for files made from the original weights (measured): KLD 0.280 → 0.338 and perplexity
  +9.5 % → +16.8 % for 1.75 ms per token. Only the makers' file, whose head was trained at 4 bits, can afford it.
- **Trimming the vocabulary:** the output head is 12 % of the bytes, but a 65,000-token head misses 1.9 % of English and
  5.5 % of code tokens. Expected +4–6.5 %.
- **Pruning layers, attention heads or experts:** below the quality bar without retraining. Expert pruning reads the
  same bytes per token.
- **ik_llama.cpp's types, Marlin-style and lookup-table kernels, other engines:** CPU-only, tensor-core-only, or no
  faster on sm_61. ik_llama.cpp also needs AVX2 on the host.

## Where a token's time goes

A P106-090 card does not run out of memory bandwidth while it generates: it runs out of instructions. Its 5 SMs can issue
about 6.4 instructions per byte of a 173 GB/s memory stream, and llama.cpp's matrix-vector kernels need 5–15 per byte.
Disassembling the kernels in the rig's binary shows where they go:

| Share of the kernel's inner loop | |
|---|---|
| Integer and address arithmetic (Pascal has no full-rate 32-bit multiply; 18-, 20- and 34-byte blocks make offsets expensive) | 41–59 % |
| Loads (only 8-, 16- or 32-bit, never 64- or 128-bit vector loads) | 16–19 % |
| The dot products themselves (`dp4a`) | 5–8 % |

Instructions per byte of weights in that loop: Q8_0 4.85, Q4_1 5.1, Q4_0 6.2, Q6_K 7.2, IQ4_XS 7.8, Q5_0 8.7, IQ4_NL 9.0,
Q4_K 10.6, MXFP4 11.3, NVFP4 14.9. That order matches the measured speeds, and the 10-SM P106-100 runs the same Q4_0 file
1.57× faster on the same memory. On Flint's 2048-wide rows each thread does a single loop iteration, so set-up costs
almost as much as the work, and in the experts' 512-wide down-projection three of every four warps sit idle.

That is why the format matters more here than on any other Pascal card: the others turn 49–66 % of their bandwidth into
generation speed, the P106-090 only 39–43 %, because it has 35–38 GB/s of bandwidth per SM against at most 19 elsewhere.

A simple model fits all four Flint files measured on the rig to within 2 %
([token_cost.py](../scripts/token_cost.py)):

> time per token = Σ over formats (bytes read in that format ÷ (decode efficiency × 173 GB/s)) + ~11 ms

| Format | Bits per weight | Decode efficiency on Flint | Time per parameter read (Q4_1 = 1) |
|---|---|---|---|
| Q4_1 | 5.0 | 0.38 | 1.0 |
| Q5_0 | 5.5 | 0.30 | 1.4 |
| Q8_0 | 8.5 | 0.41 | 1.6 |
| K-quants and I-quants (Q4_K, Q5_K, Q6_K, IQ4_XS) | 4.25–6.6 | 0.19 | 1.7–2.6 |
| F32 (the makers' router) | 32 | ~0.6 | ~4 |

The ~11 ms of fixed time per token is per-kernel overhead: about 1,900 kernel launches per token, of which ~470 belong to
the MoE down-projection rescale (2.4–2.8 ms measured). The bytes a Flint token reads split almost evenly between
attention (35 % in the makers' file) and the six routed experts (33 %), then the output head (12 %), the shared expert
(11 %) and the router (6 %, F32). Per MoE layer, attention and the six experts read the same 18.9 M parameters.

## Class A: changes to the file

### Which format each tensor gets

The cost model turns the choice of a mix into a budget: every tensor kind has a quality effect and a time cost per token.
Unsloth's UD-Q4_K_XL keeps attention and the shared expert at Q8_0 in most layers. Re-expressed in fast formats, that
allocation (UD-mirror v1) keeps Unsloth's quality but still reads 2.26 GB per token, 1.49 GB of it Q8_0, and is expected at
22.1 tok/s against 28 for the makers' Q4_1. The bytes that cost speed are attention's, because every token reads them.

Experiment [Q18](experiments.md) measured how much quality each tensor class carries at 4 bits: an importance-matrix
Q4_1 of the whole model (output head Q8_0), then one class at a time raised to Q8_0, all scored against BF16.

| Class raised to Q8_0 | Mean KLD (Q4_1 everywhere: 0.280) | Decode time added | KLD gained per ms |
|---|---|---|---|
| Attention k and v | 0.261 | +0.56 ms | **0.034** |
| Shared expert | 0.251 | +1.6 ms | 0.018 |
| Attention output | 0.246 | +2.25 ms | 0.015 |
| Dense layers 0–1 | 0.274 | +0.41 ms | 0.015 |
| Expert down-projections | 0.261 | +1.6 ms | 0.012 |
| Expert gate and up | 0.249 | +3.2 ms | 0.010 |
| Attention q | 0.275 | +2.25 ms | 0.002 |
| Output head (lowered to Q4_1) | 0.338 | −1.75 ms | keep at Q8_0 |

The gains don't add up linearly, so mixes built from this ranking were scored as whole files
([Q27](experiments.md)). All are rescale-free (layers 46–48 down-projections at Q8_0, rescale switched off); speed is the
cost model's prediction for llama-bench tg128 on the rig.

| File | Predicted tok/s | Mean KLD | Same top token | Perplexity vs BF16 |
|---|---|---|---|---|
| Makers' Q4_1 + F16 router (rescale-free) | 30.5 | 0.322 | 78.5 % | +4.3 % |
| S1: Q4_1, Q8_0 attention k/v, shared expert, dense layers | 26.7 | 0.219 | 83.6 % | +4.6 % |
| S2: S1 + Q8_0 attention output | 25.2 | 0.178 | 85.7 % | +5.4 % |
| UD-mirror v2: v1 with attention q at Q4_1 | 24.4 | 0.142 | 87.1 % | +3.6 % |
| S3: S2 + Q8_0 expert down-projections | 24.3 | 0.149 | 87.1 % | +4.7 % |
| UD-mirror v1 | 23.2 | 0.127 | 87.5 % | +4.0 % |
| **S4: S3 + Q5_0 expert gate and up** (32.6 GB, 6 cards) | 23.0 | **0.120** | **88.7 %** | **+3.4 %** |
| Imatrix Q5_0 (head Q8_0), for comparison | 22.1 | 0.185 | 84.5 % | +5.7 % |
| Unsloth UD-Q4_K_XL, for comparison | 18.8 | 0.120 | 87.7 % | +3.9 % |

S4 and the mirror recipes keep Unsloth's quality about 23 % faster than Unsloth's own file; S4 is even closer to the
original weights than UD-Q4_K_XL (88.7 % against 87.7 % same top token). Every mix here beats a single fast format at
the same speed. Between 23 and 27 tok/s the choice is a straight trade of speed for fidelity. An importance matrix
computed from the BF16 weights with bartowski's calibration text scores the same as Unsloth's in these recipes
([Q12](experiments.md)).

### Removing the MoE down-projection rescale

llama.cpp divides the input of Flint's expert down-projections by a power of two before the matrix product and multiplies
the result back, because the 16-bit sums in its 8-bit activation blocks overflow with Q4_0/Q4_1/Q5_0/Q5_1 weights. The
step costs ~470 launches per token, and switching it off is worth +8 % (Q4_1) to +11 % (Q5_0), but Q4_1 then produces NaN.

The activations that overflow come only from a few dominant channels in the down-projection inputs of layers 46–48:
their channel RMS reaches 25,600, 1,080 and 48,400, against ≤ 302 in every other layer (read from Unsloth's importance
matrix). Two file-level fixes remove the overflow there and leave the model's output unchanged:

- **Q8_0 down-projections in layers 46–48 only.** Q8_0's dot product doesn't use the 16-bit sums. It costs +8 MB per
  token (+0.11 ms) and keeps every other tensor of the makers' file byte for byte
  ([retype_tensors.py](../scripts/retype_tensors.py)).
- **A power-of-two rebalance in those layers.** Multiply each expert's down-projection block scales by 2^k and divide
  its up-projection scales by 2^k. SwiGLU is linear in the up branch, so the output is the same, no weight code changes
  and no byte is added.

Either one, together with `LLAMA_NO_MOE_DOWN_RESCALE=1`
([patch](../patches/moe-down-rescale-env.diff)), gives the measured +8 % without NaN. The first was checked on 41,000
tokens ([Q22](experiments.md)): with the rescale switched off, the makers' file with Q8_0 down-projections in layers
46–48 scores exactly like the unmodified file with the rescale (mean KLD 0.322, 78.4 % same top token against BF16) and
produces no NaN, while the unmodified file with the rescale switched off produces NaN everywhere. The same change keeps
UD-mirror v1's quality (KLD 0.127, 87.5 %).

### The makers' Q4_1 is group-64 data

Recursive trained Flint's 4-bit weights as group-64 affine INT4 (MLX export), and its GGUF stores each group as two
Q4_1 blocks of 32 with identical scale and offset: in every Q4_1 tensor, 100 % of neighbouring block pairs match. A
group-64 format holds those weights exactly in 4.5 instead of 5.0 bits per weight, 23.5 GB instead of 26.0, and 9.4 % fewer
bytes per token. llama.cpp has no such format; [patches/q4_1_64-type.diff](../patches/q4_1_64-type.diff) adds one (CPU
reference, CUDA matrix-vector product, dequantization; no batched kernel yet, so prompt processing is slower), and
[q4_1_to_q4_1_64.py](../scripts/q4_1_to_q4_1_64.py) converts the makers' file losslessly. With today's kernel style the
expected gain is only +2–4 %, because the kernel, not the bytes, sets the speed; with a Pascal kernel it grows to ~+10 %.

### Rotations

Rotating the residual stream by a randomized Hadamard matrix (as in QuaRot and SpinQuant) spreads outlier channels over
whole rows, which suits 4-bit block formats. For Flint it folds into the weights exactly, with no llama.cpp change: every
RMSNorm weight moves into the matrices that read its output, the embedding and every reading matrix get the rotation on
their input side, and every matrix writing to the residual gets its transpose on the output side. Each value head can be
rotated too, with the inverse folded into the attention output projection. [rotate_gguf.py](../scripts/rotate_gguf.py)
does both.

The expert down-projections read the SwiGLU output, which no offline rotation reaches. llama.cpp can rotate those inputs
at run time: GGUF metadata `prism.hadamard.*` (added for Prism Bonsai 2 in
[#29600](https://github.com/ggml-org/llama.cpp/pull/29600)) marks weights stored with a Hadamard folded in, and a fast
Walsh-Hadamard kernel transforms their inputs. Flint needs a one-line loader change to accept it
([patch](../patches/flint-prism-hadamard.diff)); Qwen3.6 is accepted already. The run-time transform does not make the
16-bit sums safe in general: a Sylvester Hadamard leaves each 32-value block sum depending only on the channels whose
index is a multiple of 32, so a dominant channel either drops out of every block sum or is amplified 1.41×.

Rotations are exact here (the rotated model at Q8_0 scores KLD 0.059 against BF16, the unrotated one 0.052), but they
make Flint's 4-bit files slightly worse, not better ([Q13](experiments.md)). With importance matrices computed from the
same calibration text on each model, rotated against unrotated: Q4_1 KLD 0.308 against 0.283, the UD-mirror recipe 0.132
against 0.124. The likely reason is that llama.cpp also quantizes activations, to 8 bits per block of 32: rotating the
residual stream spreads Flint's few huge activation channels into every block, which coarsens all of them, while the
weights themselves are already well-behaved Gaussian blocks with nothing to spread. Weight-only rotation results, which
assume 16-bit activations, don't transfer to this setting. The one exception is Q4_0, whose failure comes from those
same dominant channels: rotated, it scores KLD 0.367 and perplexity +9.8 % instead of 0.563 and +41.9 %, still worse than
plain Q4_1.

### Rounding

llama.cpp's quantizer rounds each block with an importance-weighted search. Error-feedback methods (GPTQ, AutoRound) see
correlations between inputs and do better on tensors whose inputs have dominant channels, but for 4-bit block formats the
measured gains are at noise level: a community AutoRound Q4_0 of Qwen3.8-27B scored KLD 0.0266 against 0.0265 for an
imatrix Q4_0. Draft PR [#12557](https://github.com/ggml-org/llama.cpp/pull/12557) cuts Q4_0's KLD by 28 % and Q5_0's by
13 % and does nothing for Q4_1. Q5_0's weighted error is about 3× lower than Q4_1's, which no rounding closes.

Q4_0's failure on Flint (perplexity +45 %) is not a missing offset. Within each block Flint's weights are Gaussian with no
offset; Q4_0's error is the normal 1.4× of Q4_1's everywhere except on tensors whose inputs have dominant channels: the
down-projections of layers 46–48, the shared expert's down-projection in layer 12, and layer 48's attention output.
Keeping just those seven tensors at Q8_0 fixes it ([Q24](experiments.md)): KLD 0.313, 80.0 % same top token and
perplexity +3.4 %, where plain imatrix Q4_0 gives 0.565, 71.7 % and +45 %.

### Smaller changes

- **Fused Q/K/V** ([fuse_qkv.py](../scripts/fuse_qkv.py)): Flint's loader accepts an `attn_qkv` tensor. Concatenating the
  rows is byte-exact when q, k and v share a format, and saves two matrix-vector products and two activation
  quantizations per layer.
- **Router format:** the makers' file keeps the router in F32 (99 MB per token). F16 halves that. Q8_0 saves bytes but
  adds an activation-quantization launch per layer, which cancels the gain.
- **Merged expert gate+up tensors:** no gain on sm_61, where llama.cpp already fuses gate, up and SwiGLU into one kernel.
- **The shared expert as two extra routed experts:** saves 4–6 launches per layer but needs a graph change.
- **Embedding table:** F32 → Q8_0 saves 0.9 GB of the rig's 4 GB host RAM, no speed.

## Class B: new formats and kernels

Because the cards are instruction-bound, the kernel is the bigger lever and a new format pays mainly together with a new
kernel. In order of cost:

1. **Launch shape** ([#25479](https://github.com/ggml-org/llama.cpp/pull/25479), [#20831](https://github.com/ggml-org/llama.cpp/pull/20831)):
   2 warps per block at one token on Pascal, and no idle warps for 512-wide inputs. Expected +5–20 %.
2. **A Pascal matrix-vector kernel** for Q4_0/Q4_1/Q5_0/Q8_0: several rows per warp sharing the activation registers,
   several blocks per lane, strength-reduced addressing. Expected +20–45 %. A P100 profile found the same kernel
   instruction-bound and gained 33 % from four rows per block. A standalone benchmark at Flint's shapes is in
   [bench/mmvq_pascal](../bench/mmvq_pascal/).
3. **FP32 sums in the activation blocks**, and one activation quantization shared by Q/K/V and by routed plus shared
   gate/up: makes the rescale unnecessary for every format (+7–15 %) and removes ~145 more launches (+2–3 %).
4. **Load-time repacking into separate planes** of 4-bit codes and scales, read with 128-bit loads; llama.cpp's SYCL
   backend does this on Intel GPUs (Q4_0 +31 % on an Arc 770). Expected +5–15 % on top of item 2, and the natural place
   for group-64 compaction of the makers' file.
5. **A cheaper bit order for Q5_0**, whose fifth bit costs the most unpacking among the fast formats: +3–8 % on Q5_0-heavy
   mixes.
6. **More fusion along the MoE chain**: +2–5 %; a single persistent kernel per layer, as vLLM uses on Blackwell, could
   reach +25–40 % but is weeks of work.

Together, items 1–3 are expected to reach 37–53 tok/s on the makers' file, and 43–62 with item 4. None of them changes
the model's output beyond summation order.

## Class C: retraining

The quant machine for this work has 3 × RTX 3090, 2 × V100 32 GB and 2 × RTX 3080 (158 GB of GPU memory, no
peer-to-peer). The V100s have no BF16, and Flint's activations are large enough to overflow FP16, so they are usable only
in FP32. transformers has supported Flint natively since
[PR #49824](https://github.com/huggingface/transformers/pull/49824) (eager or flex attention: its SDPA path can't handle
attention sinks).

- **An EAGLE-3 draft head.** llama.cpp's `draft-eagle3` works for any model whose graph exposes its layer inputs, and
  Flint's does. Expected 1.3–2× in real chat with 2–3 drafted tokens (2.3–2.9 accepted per step, from published heads for
  similar MoE models), and 1.2–1.4× for requests with tools, which lose backend sampling. Training costs 60–150 GPU-hours,
  most of it regenerating ~100,000 responses with Flint. The risk: on gpt-oss-120b EAGLE-3 made llama.cpp *slower*
  (0.46–0.71×, [#18039](https://github.com/ggml-org/llama.cpp/pull/18039#issuecomment-3755925892)) because verifying
  several tokens read more experts. On this rig two parallel conversations cost only 1.0–1.13× one, so verification
  should be cheap, but that is measured first ([Q19](experiments.md), with the scripts in
  [bench/eagle3_flint](../bench/eagle3_flint/)). A 32k draft vocabulary instead of 150k adds 8–10 %.
- **Block drafters** (DFlash and similar): a higher ceiling (1.6–2.6×) at 3–5× the training cost; drafts must stay ≤ 5
  tokens so verification stays on the matrix-vector path.
- **Partial quantization-aware recovery:** retrain only attention, the shared expert, the dense layers and the head to
  Q4_0/Q4_1, layer by layer, against the BF16 model. Each layer fits one 3090; hours per layer class. Bytes per token
  drop from 2.26 GB (UD-mirror v1) to ~1.54 GB, ≈ +10–20 %, if quality holds. Google's Q4_0 QAT of Gemma 3 cut the
  perplexity gap by 54 %; whether Flint reaches UD-Q4_K_XL's KLD this way is unproven.
- **Full layer-by-layer recovery to Q4_0** with IQ4_NL or Q8_0 down-projections: 1.45 GB per token, expected 30–37 tok/s,
  12–40 GPU-hours.
- **Pruning plus retraining** (layers, query heads, experts, vocabulary): +4–19 % for tens of billions of tokens of
  retraining; Qwen-family models collapse after ~20 % of their layers are removed even with retraining.

## Measuring quality on Flint: the noise floor

Flint's outputs move when only the arithmetic changes. The makers' Q4_1 file computed with 8 of its 49 layers on a GPU
instead of the CPU differs from the all-CPU run by mean KLD 0.088, with the same top token only 90.7 % of the time
(4 × 512 tokens). The difference starts at float rounding (10⁻⁷ in the first layer's projections) and grows layer by
layer. Dumping every intermediate tensor of two such runs shows why: the experts each token picks are the same except at
the boundary, where the 6th-ranked expert flips (in layer 2, token 3 uses expert 229 in one run and 76 in the other; in
layer 10, expert 81 or 178). Near-ties between the 6th and 7th router scores are common enough that rounding noise
swaps experts in several layers per token, and each swap changes everything after it.

The same holds between GPUs: the makers' file computed with RTX 3090 kernels against the same file computed with V100
kernels differs by KLD 0.065 and 91.3 % same top token. Q8_0, the closest a quant gets to the original weights,
therefore already sits at KLD 0.052 and 93.3 % same top token against BF16 (perplexity +3.3 %). That is the floor for any Flint file, and two files whose KLDs differ by less than a few
hundredths are not distinguishable by KLD. All scores on this page come from the same GPU build against the same BF16
reference.

Task checks see the same picture. On 300 BFCL v3 tool-calling tasks ([Q30](experiments.md),
[bench/bfcl_flint](../bench/bfcl_flint/)), BF16 calls 80.7 % correctly and every finalist lands within the ±4–5-point
uncertainty: the makers' Q4_1 83.0 %, makers-fast 83.3 %, UD-mirror v1 rescale-free 83.0 %, S4 82.0 %, S1 78.3 %. The
makers' file, worst by KLD, is as good as any on tasks, which is what its distillation tuning was for. Two runs of the
same file on two GPU types agree on only 80 % of the calls, so finer comparisons need thousands of tasks.

## Carry-over to Qwen3.6-35B-A3B

| Lever | Carries over? |
|---|---|
| Pascal kernel, launch shape, FP32 activation sums, repacking | Yes: same 2048/512 expert shapes |
| Rescale removal | Not needed: llama.cpp applies no rescale to Qwen3.6 |
| Group-64 Q4_1 | No: Qwen3.6 has no group-64 4-bit training |
| Fused Q/K/V | Partly: only its 10 full-attention layers have q/k/v |
| Rotations, online Hadamard | Yes, and llama.cpp already accepts `prism.hadamard` for it |
| Draft heads | Already has an MTP head; a 32k draft vocabulary for it is worth +9–13 % ([#29143](https://github.com/ggml-org/llama.cpp/pull/29143)); a DFlash drafter exists ([z-lab](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)) |
| Partial recovery to 4 bits | Improves quality rather than speed: Qwen3.6 already runs Q4_0 |
| K-quant leftovers in "Q4_0" files | Yes, and bigger: bartowski's Qwen3.6 Q4_0 keeps its 248k-token head and four attention q tensors in Q6_K, 22.7 % of each token's bytes. Re-expressed as Q8_0 ([Q29](experiments.md)) the cost model predicts 23.6 instead of 20.8 tok/s at unchanged quality (KLD 0.037 against 0.036, both against Q8_0); the MTP draft file has the same Q6_K head |
| Speed-aware mix | Yes, with a different ranking ([Q32](experiments.md)): Qwen3.6's cheapest quality is `ssm_out`, the shared expert and its 10 full-attention layers. Q8_0 there plus the head gives KLD 0.029 against bartowski's 0.036 at a higher predicted speed (22.5 against 20.8 tok/s, [Q33](experiments.md)); adding the linear-attention qkv and gate halves the KLD (0.020) at 20.2 tok/s ([Q31](experiments.md)). Qwen3.6 is far less sensitive than Flint: its Q4_0 already agrees with Q8_0 on 92 % of top tokens |

## What to run next

The plan, with its order and decision gates, is kept in one place: the top of [experiments.md](experiments.md#plan).
In short: measure everything predicted here on the rig first (kernel microbenchmark, finalist speeds, engine variants,
drafter go/no-go), then the exact file tweaks and a per-layer allocation search, the shared-expert merge, and, depending
on what the rig shows, the Pascal kernel and the EAGLE-3 head.

## Sources

- llama.cpp: [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) (Pascal MMVQ warps),
  [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) (narrow-matrix warps, P40 table),
  [#30334](https://github.com/ggml-org/llama.cpp/issues/30334) (MMVQ fusion on Pascal),
  [#29600](https://github.com/ggml-org/llama.cpp/pull/29600) (`prism.hadamard`),
  [#12557](https://github.com/ggml-org/llama.cpp/pull/12557) (improved legacy-quant rounding),
  [#16991](https://github.com/ggml-org/llama.cpp/pull/16991) (`GGML_CUDA_GRAPH_OPT`),
  [#22571](https://github.com/ggml-org/llama.cpp/pull/22571), [#21652](https://github.com/ggml-org/llama.cpp/pull/21652)
  (activation-sum overflow), [#22710](https://github.com/ggml-org/llama.cpp/pull/22710),
  [#26322](https://github.com/ggml-org/llama.cpp/pull/26322) (MoE fusion),
  [#18039](https://github.com/ggml-org/llama.cpp/pull/18039) (EAGLE-3), [#29143](https://github.com/ggml-org/llama.cpp/pull/29143)
  (MTP draft vocabulary), [#19817](https://github.com/ggml-org/llama.cpp/issues/19817) (Vulkan vs CUDA on a GTX 1060),
  [CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013).
- Flint: [model card](https://huggingface.co/RecursiveSI/flint), [Recursive's announcement](https://www.recursive.com/updates/introducing-recursive-flint),
  [INT4 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf), [MLX INT4](https://huggingface.co/RecursiveSI/flint-int4),
  [transformers support](https://github.com/huggingface/transformers/pull/49824), [terelabs' comparison](https://huggingface.co/terelabs/flint-GGUF).
- Pascal kernels: [P100 profile](https://github.com/qr0n/p100-llama-cpp), [GTX 1080 fusion patch](https://github.com/ArtyomITA/bonsai2-ternary-pascal).
- Rounding and retraining: [AutoRound](https://github.com/intel/auto-round), [EfficientQAT](https://arxiv.org/abs/2407.11062),
  [EAGLE-3](https://arxiv.org/abs/2503.01840), [SpecForge](https://github.com/sgl-project/SpecForge),
  [Gemma 3 QAT](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/),
  [GLU activation spikes](https://arxiv.org/abs/2405.14428).
