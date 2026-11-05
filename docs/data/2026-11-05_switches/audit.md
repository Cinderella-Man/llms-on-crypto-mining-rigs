# The switch audit: every fork behaviour over upstream b12076 (C536, Q1025)

The fork's 197 commits and 33 merges between upstream b12076 (`20a51f24a`) and `-ladder4` (`e3b30303d`), read commit by
commit (`git show`) against the code at `e3b30303d`. For each: what it adds, its switch and default, whether the default
path differs from upstream's, and whether that difference had no switch. Line numbers are at `e3b30303d`. The items
that needed a switch, and what `pascal-p106-v3-b12076-flags` (`b4d6d427b`) did with each:

| Audit item | What it is | Resolution on `-flags` |
|---|---|---|
| `f053a0f64` + `981569357` | 2 warps per block in llama.cpp's `mul_mat_vec_q` at one token on compute 6.1 code | `GGML_CUDA_MMVQ_2WARPS=0` |
| `149ec6eff` | gate/up/GLU fusion when the GLU output reuses the activations (the shared expert, Q50) | `GGML_CUDA_GLU_ALIAS_FUSE=0` |
| `94ed35a40` + `a4020ce3a` | upstream PR #31047's tile list in MMF's MUL_MAT_ID | `GGML_CUDA_MMQ_MOE_LEGACY=1` now covers MMF |
| `f3ff08786` | the legacy MMQ tile width on RDNA3/4 | `GGML_CUDA_MMQ_MOE_LEGACY=1` keeps upstream's RDNA3/4 rule |
| `2c27f86b2` | the reasoning budget's forced token with a -inf logit set to 0 | `LLAMA_REASONING_BUDGET_UNMASK=0` |
| `5f9d3b7e8` | `seq_rm` of a tail puts head on the first removed cell | `LLAMA_KV_HEAD_TAIL=0` |
| `5b5a92a46` | `find_slot` keeps head on a free cell | `LLAMA_KV_HEAD_FREE=0` |
| `237590944` (f) | a sliding-window layer expands K last | `LLAMA_ATTN_K_LAST=0` |
| `237590944` (a) | the CUDA backend reports `COMPUTE_6_1` | no switch: output only; its one consumer, the rescale default, has `LLAMA_NO_MOE_DOWN_RESCALE` |
| `d70c4b29b` | the card mask's cell copies allocated with the mask off | `LLAMA_GPU_KQ_CELLS=0` |
| `be5498051` (a) | `seq_rm` that empties a stream restarts at cell 0 | `LLAMA_KV_HEAD_EMPTY=0` |
| `be5498051` (b) | a state restored into an empty stream at canonical cells | `LLAMA_KV_RESTORE_CANONICAL=0` |
| `c6663558a` | a ring checkpoint past the divergence restored | `LLAMA_SERVER_CKPT_RING_PAST=0` |
| `d537d03dc` | a sliding-window restore resumes after `pos_max` | `LLAMA_SERVER_CKPT_RESUME_NEXT=0` |
| `d87e419ca` | sliding-window checkpoints hold the whole ring | `LLAMA_SERVER_CKPT_RING=0` |
| `c6663558a`, `d799fb934`, `62774f688` | server timing fields and log levels | no switch: output only |
| `1290b48f0` | ggml-alloc plans again for a leaf no node reads | no switch: a fix; without it a plan made during packed uploads crashes |
| `45fbad874` | MUL_MAT_ID's matrix-vector batch limit from the code's architecture | no switch: a fix; without it a PTX run launches more warps than the kernel allows |
| `76788bf4a` | IQ2_S/IQ3_XXS sign masks in registers, 3 blocks per SM | build-time only (`GGML_CUDA_PASCAL_FLAGS`); a run-time switch is Q1040 |
| `29450398c` (BF16) | the BF16 kernel to 8 tokens, R = 4 at 3–4 tokens | `GGML_CUDA_MMVF_PASCAL_MAX_COLS=4`, `GGML_CUDA_MMVF_PASCAL_RULE=0` |
| `00d662c3a` | the BF16 lane-group rule at 3–4 tokens (Q611) | `GGML_CUDA_MMVF_PASCAL_RULE=1` |
| `bdb902d95` | the QSA k-pool pad cells found in linear time | `LLAMA_QSA_PAD_LINEAR=0` |
| `66ba1f654` | the scheduler's context sized per graph | `GGML_SCHED_CTX_PER_GRAPH=0` |
| `3e49981fb` (a) | state buffers not zero-filled | `LLAMA_STATE_NOINIT=0` (zero-filled when allocated) |
| `3e49981fb` (b) | the prompt cache drops a state copied short | no switch: a fix; without it the cache keeps a corrupt entry |
| `5174e7ce0` | the routed Q2_0 down projection on the multi-token kernel (Q825) | `GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,0,2` |
| `39c3ab5f1` | FP32 activation sums in llama.cpp's `mul_mat_vec_q` (Q916) | `GGML_CUDA_MMVQ_FP32_SUMS=0` |
| `f3072cbe8` | -1 expert ids: the host-side copy for ids of any producer; the device-side checks | `GGML_CUDA_SKIP_IDS_ANY=0` for the host side; the device comparisons stay (their cost: Q1028) |
| `3dbf1105a` | CUDA claims CPY IQ4_NL → F32 and F32 ↔ Q6_K | `GGML_CUDA_CPY_HOP_TYPES=0` |
| `-hybrid` × `-gdnfuse` (found while reading) | a pipe slot's conv histories land in state group 0 with the fused conv | a bug, not a switch: Q1042 |

The full reading follows, one section per quarter of the commits; "needs switch" there means "needed one at
`e3b30303d`".

## Inventory, commits 1-50

Line numbers are at HEAD unless marked "upstream". "static" = read once (first use) and cached for the process.

| commit | behaviour | switch (default; values) | read where | default vs upstream | needs switch |
|---|---|---|---|---|---|
| 9c22cc188 | CUDA graphs allowed on compute 6.1 (upstream: Volta+ only); narrowed to exactly cc 6.1 by 981569357 | `GGML_CUDA_DISABLE_GRAPHS` (upstream's own env; unset = graphs on; set = graphs off on every GPU) | env: `common.cuh:1351` (static in `is_enabled()`); arch gate `ggml-cuda.cu:6595-6603` (per graph key, first use) | default on for cc 6.1; `GGML_CUDA_DISABLE_GRAPHS=1` gives upstream's path on 6.1 (no graphs) | no |
| b56137a3f | env that drops the MoE down-projection rescale (Flint, Mistral4, Mellum, Laguna) | `LLAMA_NO_MOE_DOWN_RESCALE` (unset: dropped at 1 token when every CUDA device is cc 6.1, i.e. `cparams.moe_down_rescale_tg` from the `COMPUTE_6_1` backend feature, else kept; `0`: kept everywhere = upstream; `tg`: dropped at 1 token; other: dropped everywhere) | `llama-graph.cpp:2081-2090` (static getenv, evaluated per graph build), used `:3005`; `moe_down_rescale_tg` set at context creation `llama-context.cpp:410-413` | default on (all-6.1 rigs, 1-token graphs lose the amax/scale chain: different numerics); `=0` restores | no (caveat: the unset default keys on `COMPUTE_6_1`, not on the Pascal kernels being active; with `GGML_CUDA_MMVQ_PASCAL=0`, `GGML_CUDA_MMVQ_PASCAL_SKIP` or a Q5_1 down-projection the stock kernel reads FP16 block sums with the rescale still dropped) |
| f60cbd49c | Flint added to the archs that accept `prism.hadamard.*` metadata | none | `llama-model.cpp:2280` (load) | only for files that hold `prism.hadamard.*` (upstream throws for Flint) | no |
| ed0d58427 | Q4_1_64 type (CPU, CUDA MMVQ, dequant) | none; retired by 230506b65, no code at HEAD (ggml-openvino's own `Q4_1_64` is upstream) | - | n/a | no |
| f053a0f64 | `mul_mat_vec_q` at 1 token, non-small-K: 2 warps per block instead of 4 on compute 6.1 code (since 981569357 a table id `MMVQ_PARAMETERS_PASCAL_DP4A`) | none (compile-time per `__CUDA_ARCH__`, host side a PTX-arch probe) | `mmvq.cu:115, 131-132, 145-156, 170-174, 505-513, 657-660, 1227` (constexpr; probe once per device) | default on for every device running compute_61 code; `GGML_CUDA_MMVQ_PASCAL=0` does not undo it | **yes** |
| 56154bdaa | `GGML_CUDA_PROFILE=<file>` times every launch group | `GGML_CUDA_PROFILE` (unset = off; path = on), `GGML_CUDA_PROFILE_MODE` | `ggml-cuda.cu:5916-5950` (static); hooks `:6319-6325, 6515, 6629` gated on `prof` | default off = upstream | no |
| 051d75840 | profile: plain events outside stream capture | as 56154bdaa | as above | default off | no |
| 4a6a2083f | Pascal 1-token MMVQ kernels for Q4_0, Q4_1, Q8_0 | `GGML_CUDA_MMVQ_PASCAL` (default 1 = cc 6.1 only; 0 off; 2 any NVIDIA); `GGML_CUDA_MMVQ_PASCAL_SKIP=<types>` per type | `mmvq-pascal.cu:886-892` (static), `:982-985`, `:993-1005`; hook `mmvq.cu:1709` | default on for cc 6.1 (own reduction order, FP32 block sums); `=0` falls back to `mul_mat_vec_q` (but with the 2-warp table of f053a0f64) | no |
| 63bce2c80 | tests only | - | - | - | no |
| 1bf34299c | Q5_0 in the Pascal 1-token kernels (+tests) | as 4a6a2083f | as 4a6a2083f | default on for cc 6.1; switch restores | no |
| 8ffa7577b | `GGML_CUDA_PROFILE_HOST=<file>` logs host/GPU time per graph evaluation | `GGML_CUDA_PROFILE_HOST` (unset = off) | `ggml-cuda.cu:6542-6556` (static); hooks `:6619-6622, 6711` | default off = upstream | no |
| 634f86a96 | Q5_K and IQ4_XS in the Pascal 1-token kernels (+tests) | as 4a6a2083f | as 4a6a2083f | default on for cc 6.1; switch restores | no |
| 7984c7e0b | Pascal kernels compute activation block sums in FP32; `tg` value of the rescale env; `MMVQ_PASCAL` backend feature | `GGML_CUDA_MMVQ_PASCAL`; `LLAMA_NO_MOE_DOWN_RESCALE` | as rows above; feature pushed only when the kernels are active | default on for cc 6.1; switches restore | no |
| 149ec6eff | gate/up/GLU fusion allowed when the GLU output reuses the activations' memory (quantized weights) | none | `ggml-cuda.cu:4070-4077, 4110` (`src_may_alias`), `:4501-4506` (lambda), `:4518-4521, 4531-4534` (per graph evaluation, fusion check) | default on, every CUDA GPU and type that fuses (not Pascal-only) | **yes** |
| 94ed35a40 | upstream PR #31047 cherry-pick (J. Gäßler): MoE tile list (`tile_spec`) replaces `expert_bounds` in MMQ and MMF | MMQ: `GGML_CUDA_MMQ_MOE_LEGACY=1` (default 0) emulates the old layout through `tile_spec`; MMF: none | `mmq.cu:106-115` (static), `:279-283, 352-356`; `mmid.cu:35-150, 178-`; MMF `mmf.cu:85-110`, `mmf.cuh:310-370, 604-612` | default on; MMQ restorable; MMF MUL_MAT_ID (>16 tokens; F16/BF16/F32 on Volta+/AMD, never on Pascal) has no switch | **yes** (MMF part) |
| f3ff08786 | upstream cherry-pick: MMQ MoE tile width chosen against tokens per expert on every GPU (upstream: RDNA3/RDNA4 only) | `GGML_CUDA_MMQ_MOE_LEGACY=1` -> `ncols_opt = ne12` on every GPU | `mmq.cu:275-284` (per call) | default on; legacy = upstream on NVIDIA, but not on RDNA3/RDNA4 (upstream used tokens per expert there) | **yes** (minor, RDNA3/4) |
| a4020ce3a | upstream cherry-pick follow-up: `mul_mat_f_ids` returns at once for an empty tile | none (part of the MMF tile list) | `mmf.cuh:364-366` | default on; meaningful only with the MMF tile list | covered by the 94ed35a40 MMF item |
| 0c0a36b22 | with pipeline parallelism a graph is reused only when nothing is in flight (`gf_in_flight`) | `LLAMA_GRAPH_REUSE_WAIT=1` (default 0) restores reuse with the full wait | `llama-context.cpp:613-614` (context creation), `:2822`; flag `:1223, 4970` | default on (pipeline parallelism only; speed only, same kernels on same inputs) | no |
| b7d7f0fcf | tests only | - | - | - | no |
| ea01efe7e | `GGML_OP_KQ_MASK`: attention mask built on each device from cell/token positions for ubatches >= n tokens | `LLAMA_GPU_KQ_MASK=<n>` (default 0 = off since a later commit; the commit had 32), `LLAMA_GPU_KQ_MASK_CHECK=1` | `llama-graph.cpp:57-63` (static), `:81-86` per ubatch; `kq_mask` placement in `graph_get_cb` `llama-context.cpp:5081-5098` (names only the fork's nodes) | default off = upstream | no (note: the per-device cell copies of d70c4b29b, another slice, are allocated in `llama-kv-cache.cpp:131-138` even at 0) |
| 2acdafc2f | one-row (residual ADD +) RMS_NORM + MUL writes its Q8_1 copy; `mul_mat_vec_q` skips its own quantization | `GGML_CUDA_NORM_Q8` (default on, every CUDA GPU; 0 off); `GGML_CUDA_NORM_Q8_DEBUG` | `norm.cu:911-917` (static); `ggml-cuda.cu:5227-5233, 5799-5804`; `mmvq.cu:1663-1664` | default on (numerics: Q8_1 `ds.y = d*sum(q)` instead of the float sum, fused add+norm); `=0` restores | no |
| 2c27f86b2 | screened output head; side file; sampler-safety check; server per-batch choice; reasoning-budget forced token with logit -inf set to 0 | `LLAMA_SCREEN_HEAD` (default 0), `LLAMA_SCREEN_HEAD_M`, `LLAMA_SCREEN_HEAD_FILE`; reasoning-budget change: none | `llama-context.cpp:415-432` (creation); `llama-graph.cpp:4928-4937`; `server-context.cpp:2264`; `common/reasoning-budget.cpp:181-189` (per sample) | head: default off = upstream (`output_screen.weight` loaded only for files that hold it); reasoning-budget change always on | **yes** (reasoning-budget part) |
| 7b36b8ce2 | screened head: log on change; op tests in the default list | as 2c27f86b2 | `llama-context.cpp:1843-1849` | off | no |
| 1b7c83f40 | server logs the screened-head decision when it changes | as 2c27f86b2 | server, only when `screen_head_m > 0` | off | no |
| dc4498ad2 | Pascal kernels for 2-4 tokens, each distinct expert read once | `GGML_CUDA_MMVQ_PASCAL_MC` (default 1; 0 = 2+ tokens to llama.cpp's kernels), `GGML_CUDA_MMVQ_PASCAL` | `mmvq-pascal.cu:896-902` (static), `:993` | default on for cc 6.1; switches restore | no |
| 66dd3d050 | multi-token kernels stage activations through shared memory | `GGML_CUDA_MMVQ_PASCAL_STAGE_ACT` (default 0) | `mmvq-pascal.cu:905-911` (static) | default off | no |
| fb9256e61 | tests only | - | - | - | no |
| bbf183f1b | `ggml_backend_sched_set_caller_sync`: skip host waits before split input copies | API (default false); llama sets it only with `LLAMA_INPUT_UPLOAD` (default 0) | `ggml-backend.cpp:1238-1239, 2639-2642`; caller `llama-context.cpp:2831` | default off = upstream | no |
| 5f9d3b7e8 | `seq_rm` of a sequence's tail puts head on the cell of the first removed position | none | `llama-kv-cache.cpp:488-530` (`seq_rm`, per call) | default on, every model and backend | **yes** |
| d9e4cbfef | `llama_pipe_*` pipelined single-token decoding | API only; `pipe_drain()` in `decode()`/dtor is a no-op without slots | `llama-context.cpp:3596-3603` | only when the API is called | no |
| 7ab93dd20 | tool only (examples/pipespec) | - | - | - | no |
| 42238b10b | `llama_pipe_get_sampled` (+ pipespec) | API only | `llama-context.cpp` pipe code | only when called | no |
| 705f74029 | tool only | - | - | - | no |
| 504f95ba7 | tool only | - | - | - | no |
| 11689230f | tool only | - | - | - | no |
| bc250a0fa | tool only | - | - | - | no |
| d1668fa87 | `llama_sampler_set_pos` (position-keyed dist draws); each graph's sampler input tensors re-pointed in `set_input` | API (keyed only after `llama_sampler_set_pos`; the server calls it only in its pipe mode, `pipe_draws` default false); re-pointing always runs | `llama-sampler.cpp` (`llama_sampler_set_pos`, dist `backend_set_input`); `llama-graph.cpp:1694-1697, 5106` | default = upstream (with one graph per sampler the re-pointing sets the tensors it already holds) | no |
| c106ff3e4 | pipe slots for several sequences; graph callback places nodes on the scheduler being built | API; default `graph_get_cb(sched_res())` = the context scheduler | `llama-context.cpp:5064-5101` | default = upstream | no |
| 468243476 | tool only | - | - | - | no |
| f21e0e469 | off switches `GGML_CUDA_MMQ_MOE_LEGACY`, `LLAMA_GRAPH_REUSE_WAIT`, `GGML_CUDA_MMVQ_PASCAL_MC` | the switches themselves (all default = fork behaviour) | see 94ed35a40, 0c0a36b22, dc4498ad2 | n/a | no (gaps: see f3ff08786; legacy falls back to the tile list when `ne02*ceil(ne12/J) > 65535`, `mmq.cu:353-355`) |
| a4b5d2c6e | one persistent kernel per MoE layer at 1 token (skeleton) | `GGML_CUDA_PERSISTENT_MOE` (default 1 = cc 6.1; 0 off; 2 any NVIDIA), `_CHECK`, `_GRID`, `_FAULT` | `moe-persistent.cu:869-875` (static), `:962-965`; `ggml-cuda.cu:4464-4478, 4982, 5230` | default on for cc 6.1 (claimed bit-identical to the per-op path); `=0` restores | no |
| f599ebb65 | persistent kernel for Q5_0/Q8_0 layers; Pascal MMVQ code moved to `mmvq-pascal-impl.cuh` | as a4b5d2c6e / 4a6a2083f | as above | default on for cc 6.1 | no |
| 0fa7779bf | persistent kernel bit-identical; check mode per stage | `GGML_CUDA_PERSISTENT_MOE_CHECK` (default 0; CUDA graphs off only in check mode, `ggml-cuda.cu:6650`) | `moe-persistent.cu:967-968` | default off | no |
| 8cf936de1 | persistent kernel for Q5_K/IQ4_XS layers | as a4b5d2c6e | as above | default on for cc 6.1 | no |
| 981569357 | Pascal-only pieces only on cc 6.1: own MMVQ table id (2 warps), graphs exactly 6.1, `GGML_CUDA_MMVQ_PASCAL` 1/2 semantics | as 9c22cc188, f053a0f64, 4a6a2083f | as those rows | as those rows | **yes** (same item as f053a0f64: the table has no runtime switch) |
| 6605aed8e | persistent MoE: explicit architecture rule, log on first use | as a4b5d2c6e | `moe-persistent.cu:946, 962-965` | default on for cc 6.1 | no |
| 4b7b82637 | persistent MoE on by default on cc 6.1; notice when the rescale blocks it | `GGML_CUDA_PERSISTENT_MOE` | `ggml-cuda.cu:4466-4475` | default on for cc 6.1; `=0` restores | no |
| 4225d88f1 | packed async input uploads (`ggml_backend_sched_set_input_upload`, `_set_input_skip`, `_get_input_copy`), `llama_pipe_submit_fed`, input layer on GPU | `LLAMA_INPUT_UPLOAD` (default 0), `LLAMA_INPUT_ON_GPU` (default 0), `LLAMA_PIPE_UPLOAD` (pipe slots only) | `llama-context.cpp:1022-1026` (context creation); `llama-model.cpp:1789-1795` (load); `ggml-backend.cpp` paths gated on `input_upload`/`caller_sync`/skip list | default off = upstream (only an extra `last_cpu_split` scan per evaluation) | no |
| 5b5a92a46 | `find_slot` keeps head on a free cell instead of restarting the search from cell 0 | none | `llama-kv-cache.cpp:1123-1131` (per ubatch) | default on, every model and backend | **yes** |
| cef08989e | tool only | - | - | - | no |

50 commits.

### Needs a switch

#### f053a0f64 (+ 981569357): 2 warps per block in `mul_mat_vec_q` at one token on compute_61 code

- Where (HEAD): `ggml/src/ggml-cuda/mmvq.cu`
  - `enum mmvq_parameter_table_id`, line 115: `MMVQ_PARAMETERS_PASCAL_DP4A`.
  - `get_device_table_id()` (device), lines 131-132: `#elif __CUDA_ARCH__ == GGML_CUDA_CC_DP4A return MMVQ_PARAMETERS_PASCAL_DP4A;`
  - `mmvq_arch_probe` / `mmvq_device_code_arch()`, lines 138-156; `get_device_table_id(int cc)` (host), lines 170-174.
  - `calc_nwarps()`, lines 505-513: `if (table_id == MMVQ_PARAMETERS_PASCAL_DP4A) { if (ncols_dst == 1 && !small_k) return 2; table_id = MMVQ_PARAMETERS_GENERIC; }`
  - `calc_rows_per_block()`, lines 657-660 (maps to GENERIC, unchanged result).
  - `should_use_small_k` lambda in `mul_mat_vec_q_switch_ncols_dst`, line 1227: `calc_nwarps(type, c_ncols_dst, table_id, /*small_k =*/ true)`.
- Upstream (`git show 20a51f24a:ggml/src/ggml-cuda/mmvq.cu`): no DP4A table; compute_61 code takes `MMVQ_PARAMETERS_GENERIC`, whose `calc_nwarps` gives
  ```
  case 1: case 2: case 3: case 4:
      return 4;
  ```
  and `should_use_small_k` called `calc_nwarps(type, c_ncols_dst, table_id)`.
- Effect at default: every one-token product that reaches `mul_mat_vec_q` on a device running compute_61 code (types the Pascal kernels do not take, shapes they decline, and everything with `GGML_CUDA_MMVQ_PASCAL=0` or `_SKIP`) launches 2 warps per block instead of 4. `GGML_CUDA_MMVQ_PASCAL=0` does not bring back upstream's launch.
- Off value must restore: the GENERIC table for compute_61 code on device and host (4 warps at `ncols_dst == 1`). `nwarps` is a compile-time constant (`__launch_bounds__`, `blocks_per_iter`), so a runtime switch needs both shapes instantiated (e.g. a template flag chosen on the host from an env var read once) or the host to fall back to a GENERIC-table instantiation.
- Tokens/logits: can change. `blocks_per_iter = vdr * nwarps*warp_size / qi` changes which blocks each thread sums and the cross-warp reduction, so FP32 summation order differs (last-bit logit differences; rare token flips).

#### 149ec6eff: gate/up/GLU fusion when the GLU output aliases the activations

- Where (HEAD): `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml_cuda_check_fusion_memory_ranges()`, lines 4070-4077 (new parameter `src_may_alias`) and line 4110 (`|| src == src_may_alias` in the skip test).
  - `ggml_cuda_can_fuse()`, lines 4501-4506 (`mmvq_src1_may_alias` lambda: `ggml_is_quantized(ffn_up->src[0]->type) ? ffn_up->src[1] : nullptr`), calls at 4518-4521 (bias GLU) and 4531-4534 (plain GLU, MUL_MAT and MUL_MAT_ID).
- Upstream (`20a51f24a`, ggml-cuda.cu 3584 and 3794-3797):
  ```
  if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
  ...
  if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
      int out_nodes[] = { node_idx + 2 };
      return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
  }
  ```
  (and the same call without the extra arguments for the bias variant at 3782-3784).
- Effect at default: on every CUDA GPU, a quantized gate/up pair whose GLU output ggml-alloc placed in the FFN input's memory (Flint's shared expert, dense FFNs) now runs fused (one product + in-kernel GLU) instead of two products, two quantizations and a GLU kernel.
- Off value must restore: `src_may_alias = nullptr` in both calls (upstream's refusal of the overlap).
- Tokens/logits: probably bit-identical products, but a different kernel path (fused GLU in the product kernel; on cc 6.1 the Pascal kernel's gate shape `mmvqp_choose(type, gate=true)`); the commit only checked "logits unchanged within run-to-run noise". Mainly speed.

#### 94ed35a40 (+ a4020ce3a): MMF MUL_MAT_ID tile list without an off switch

- Where (HEAD): `ggml/src/ggml-cuda/mmf.cu` lines 85-110 (`tile_spec_dev`, `max_tiles`, `ggml_cuda_launch_mm_ids_helper(..., 16, ...)`), `ggml/src/ggml-cuda/mmf.cuh` `mul_mat_f_ids` lines 310-370 (reads `tile_spec[blockIdx.y]`, early return 364-366 from a4020ce3a) and launch 604-612 (`block_nums_ids(block_nums.x, ids_data->max_tiles, 1)`); shared helper `ggml/src/ggml-cuda/mmid.cu` `mm_ids_helper` 35-150 (writes `tile_spec`, no `expert_bounds`).
- Upstream (`20a51f24a:ggml/src/ggml-cuda/mmf.cu` 72-96):
  ```
  expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);
  ...
  ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
      static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, /*write_inverse =*/ false, ctx.stream());
  ...
  ids_info.expert_bounds_dev = expert_bounds_dev.get();
  ```
- Effect at default: MMF MUL_MAT_ID with more than 16 tokens (F16/BF16/F32 experts on Volta+ NVIDIA or AMD with WMMA/MFMA; MMF is never chosen on Pascal) uses the linearized tile list. `GGML_CUDA_MMQ_MOE_LEGACY` covers only MMQ.
- Off value must restore: the per-expert layout for MMF, e.g. extend `GGML_CUDA_MMQ_MOE_LEGACY` (or a new `GGML_CUDA_MMF_MOE_LEGACY`) to pass `tiles_per_expert = ceil(ncols_max/16)` to the helper, as mmq.cu 352-356 does, with the grid sized `n_experts*tiles_per_expert`.
- Tokens/logits: speed only expected (each output column's dot product does not depend on how columns are grouped into tiles).

#### f3ff08786: `GGML_CUDA_MMQ_MOE_LEGACY=1` is not upstream on RDNA3/RDNA4

- Where (HEAD): `ggml/src/ggml-cuda/mmq.cu` `ggml_cuda_mul_mat_q`, lines 275-284:
  ```
  ncols_opt = ne12;
  if (!ggml_cuda_mmq_moe_legacy()) {
      ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
  }
  ```
- Upstream (`20a51f24a:ggml/src/ggml-cuda/mmq.cu` 243-247):
  ```
  // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
  if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
      ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
  }
  ```
- Effect: default (tokens per expert everywhere) differs from upstream on every non-RDNA GPU, and the legacy value restores upstream there, but on RDNA3/RDNA4 legacy picks against `ne12`, which upstream never did. Also legacy silently uses the tile list when `ne02*ceil(ne12/J) > 65535` (`mmq.cu:353-355`).
- Off value must restore: in the legacy branch keep upstream's RDNA3/RDNA4 condition.
- Tokens/logits: tile width (J) choice: speed; where the MMQ config uses stream-k the k-split points move, which can change last bits.

#### 2c27f86b2: reasoning-budget forced token with logit -inf set to 0

- Where (HEAD): `common/reasoning-budget.cpp` `common_reasoning_budget_apply()`, lines 181-189:
  ```
  if (cur_p->data[i].id != forced) {
      cur_p->data[i].logit = -INFINITY;
  } else if (cur_p->data[i].logit == -INFINITY) {
      cur_p->data[i].logit = 0.0f;
  }
  ```
- Upstream (`20a51f24a:common/reasoning-budget.cpp` 181-186):
  ```
  // set all logits to -inf except the forced token
  for (size_t i = 0; i < cur_p->size; i++) {
      if (cur_p->data[i].id != forced) {
          cur_p->data[i].logit = -INFINITY;
      }
  }
  ```
- Effect at default: always active, not tied to `LLAMA_SCREEN_HEAD`. Without the screened head it matters only when the forced token's logit is already -inf when the budget sampler forces it (e.g. a -inf logit bias or an earlier sampler masking it); upstream then leaves an all -inf distribution.
- Off value must restore: leave the forced token's logit untouched (or apply the `else if` only while the screened head is on).
- Tokens/logits: changes the sampled token in that degenerate case only.

#### 5f9d3b7e8: `seq_rm` of a tail puts head on the first removed position's cell

- Where (HEAD): `src/llama-kv-cache.cpp` `llama_kv_cache::seq_rm()`, lines 488-530: `pos_min_head`/`pos_min` tracking (489-514) and
  ```
  } else if (p1 == std::numeric_limits<llama_pos>::max() && pos_min_head != cells.size()) {
      head = pos_min_head;
  } else if (new_head != cells.size() && new_head < head) {
  ```
  (lines 523-530; the `cells.get_used() == 0 -> head = 0` branch at 517-522 is a separate, later commit).
- Upstream (`20a51f24a:src/llama-kv-cache.cpp` 467-477):
  ```
  if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
      if (new_head == cells.size()) {
          new_head = i;
      }
  }
  ...
  // If we freed up a slot, set head to it so searching can start there.
  if (new_head != cells.size() && new_head < head) {
      head = new_head;
  }
  ```
- Effect at default: every tail removal (`p1 < 0`), used by the server's prompt-cache trims, speculative rollbacks and context shifts, can move head (also forward) to the cell of the lowest removed position.
- Off value must restore: upstream's rule only (`new_head < head`). One switch could cover this and 5b5a92a46 together (they work as a pair).
- Tokens/logits: can change. Different cells for later tokens change the KV order that attention sums over (and can change `n_kv`), so logits can differ in the last bits.

#### 5b5a92a46: `find_slot` keeps head on a free cell

- Where (HEAD): `src/llama-kv-cache.cpp` `llama_kv_cache::find_slot()`, lines 1123-1131:
  ```
  if (head_cur > cells.get_used() + 2*n_tokens && !(head_cur < cells.size() && cells.is_empty(head_cur))) {
      head_cur = 0;
  }
  ```
- Upstream (`20a51f24a:src/llama-kv-cache.cpp` 1054-1058):
  ```
  if (head_cur > cells.get_used() + 2*n_tokens) {
      head_cur = 0;
  }
  ```
- Effect at default: every model, every ubatch; when head sits on an empty cell the slot search no longer restarts from cell 0.
- Off value must restore: upstream's condition.
- Tokens/logits: can change, as for 5f9d3b7e8 (cell placement changes attention's key order; fill pattern and `n_kv` can differ).

## Inventory, commits 51-100 (432414d17 .. 15b5de992)

unless marked "upstream". "cc61" = every CUDA device of compute capability 6.1 (P106). 50 commits.

| commit | behaviour | switch (default; values) | read where | default vs upstream | needs switch |
|---|---|---|---|---|---|
| 432414d17 | server: pipelined speculation for a slot generating alone (pipe slots, position-keyed draws, copy/cont guesser); `llama_sampler_chain_on_backend` API | `--pipe-depth N` (0 = off), `--pipe-guess none/copy/cont` (cont), `--pipe-copy-min` (4) | common/arg.cpp:1668-1697; gate server-context.cpp:2535 (`pipe_depth > 0`, at load), 4995 (`pipe_eligible`), 5303 (per turn) | only with `--pipe-depth` > 0 (and `-bs`); default = upstream | no |
| 237590944 | (a) CUDA reports `COMPUTE_6_1` feature; (b) LLAMA_INPUT_ON_GPU / LLAMA_INPUT_UPLOAD defaulted on for cc61 (reverted by 3ad372ae9); (c) `synchronize()` skip + reused-graph sync skip, gated on plain_upload; (d) scheduler `input_upload==1` (was >=1) and pack reset; (e) `gf_in_flight=true` moved into `graph_compute`; (f) sliding-window `build_attn` expands q, v, k (upstream q, k, v) | (a) none; (b) `LLAMA_INPUT_ON_GPU` (unset = off at HEAD), `LLAMA_INPUT_UPLOAD` (unset = 0 at HEAD; nonzero = 2); (c)/(d) follow LLAMA_INPUT_UPLOAD; (e) none (LLAMA_GRAPH_REUSE_WAIT=1 of an earlier commit neutralises it); (f) none | (a) ggml-cuda.cu:8114-8122 (`ggml_backend_cuda_get_features`); (b) llama-model.cpp:1789-1790 (load), llama-context.cpp:1022-1025 (`sched_new`); (c) llama-context.cpp:1216-1219, 2831, 2848; (e) llama-context.cpp:4970; (f) llama-graph.cpp:4228-4240 | (a) default on (extra feature entry on cc61); (b)-(d) default off = upstream; (e) no observable default effect (graph is rebuilt after a K-shift anyway; sync conditions reduce to upstream's with plain_upload 0); (f) **default on, every backend, every SWA model** | **yes: (f) k-last order; (a) feature entry (cosmetic)** |
| ec25ca92a | a fed pipe token waits on its source slot's event and uploads from pinned memory | none (inside `llama_pipe_submit_fed`) | llama-context.cpp `pipe_submit_impl` | only through the pipe API | no |
| a6a329c0a | packed-upload buffers grow 2x (min 64 KiB); `LLAMA_KV_PAD` pads n_kv to a power of 2 >= 256 | `LLAMA_KV_PAD` (unset = 256 = upstream); pack growth only with input_upload 2 | llama-kv-cache.cpp:1394-1405 (static, read once); ggml-backend.cpp `ggml_backend_sched_pack_inputs` | default = upstream (pad 256; packs only with LLAMA_INPUT_UPLOAD or pipe slots) | no |
| 8d7e4c0a8 | llama-pipespec `--pipe-floor` | tool flag | examples/pipespec | tool only | no |
| 6a33c6567 | server `--pipe-floor N` filler guesses | `--pipe-floor` (0 = off) | common/arg.cpp:1699; server `pipe_top_up` | only with `--pipe-depth` | no |
| 7216a09b7 | CUDA: Q/K/V products + norms + RoPE + cache writes of one token in two launches; `mmvqp_rows_1t` shared; rope.cu host refactor (`rms_norm_mul_rope_params`, SASS identical per commit) | `GGML_CUDA_ATTN_BLOCK` (1 = cc61 only, default; 0 off; 2 any NVIDIA) | ggml-cuda.cu:4893-4900 (static, read once), called from `ggml_cuda_try_fuse` 4970-4975 | default on (cc61); 0 restores the per-op path; rope.cu refactor changes no behaviour | no |
| b55d88c1d | MoE down-projection rescale (upstream `build_amax_src1_f16`) dropped at one token on cc61 | `LLAMA_NO_MOE_DOWN_RESCALE` (unset: as "tg" on cc61, kept elsewhere; "0": kept everywhere = upstream; "tg"; other: dropped everywhere) | llama-graph.cpp:2081-2090 (static env, first graph); `cparams.moe_down_rescale_tg` llama-context.cpp:410-413 (context creation) | default on (cc61), numerics change at one token; `=0` restores upstream | no |
| 141b20976 | staircase drafter (llama_pipe_draft_*), Flint FFN-half lambda (+ `t_embd_all` with drafter), tap layers as outputs, pos-uniform API | `--pipe-guess staircase --pipe-draft FILE` (+ `--pipe-draft-*`); llama-pipespec flags | server-context.cpp:2537-2550; llama-context.cpp `pipe_draft_init`; flint.cpp `cparams.pipe_embd_all` (false unless drafter) | only with a drafter; Flint graph identical at default (same ops/order in a lambda) | no |
| 4651a3eb4 | CUDA init param `priority=high` (streams at greatest priority); `ggml_backend_event_query` proc | backend init params string (nullptr by default) | ggml-cuda.cu `ggml_backend_cuda_device_init_backend`; common.cuh `stream()` (`stream_priority != 0`) | only when a caller passes `priority=high` (LLAMA_PIPE_CHUNK_STREAM); default = upstream | no |
| f0f42ef69 | pipe chunk slots on own lower-priority streams; `llama_pipe_ready`; norm-placement callback iterates the scheduler's own backends | `LLAMA_PIPE_CHUNK_STREAM` (unset = off) | llama-context.cpp:683-688 (context creation); callback 5090-5099 | default = upstream (main scheduler's backends = context backends, same order) | no |
| 7fc017ace | llama-pipespec `--pipe-chunks` | tool flag | examples/pipespec | tool only | no |
| a9ce0359e | llama-pipespec waits for the prompt | none | examples/pipespec | tool only | no |
| 66c5a4c11 | continuation store for pipelined guesses | `--pipe-store FNAME` (none) | common/arg.cpp:1752; common/pipe-store.cpp | only with `--pipe-depth` + `--pipe-store` | no |
| b29d4c63c | llama-pipespec `--pipe-store` | tool flag | examples/pipespec | tool only | no |
| fcad3c226 | Pascal one-token decode attention kernel | `GGML_CUDA_FATTN_PASCAL` (0 off = default since 3ad372ae9; 1 cc61; 2 any NVIDIA); `_FAULT` test hook | fattn-pascal.cu:256-262 (static), fattn.cu:803-806 | default off = upstream | no |
| d70c4b29b | card-built attention mask reads per-device cell copies (F32 [2, kv_size*n_stream+1]), update rows per step, stale tracking; KQ_MASK op layout | `LLAMA_GPU_KQ_MASK=<min tokens>` (0 = off = default since b6db46d8c); `LLAMA_GPU_KQ_MASK_CHECK` (off) | llama-graph.cpp:57-62 (static); llama-context.cpp:2893-2900; **allocation unconditional:** llama-kv-cache.cpp:130-138, 267-273 | graph path default = upstream host mask; **cell copies are still allocated (VRAM) and tracked by default** | **yes: cell-copy allocation** |
| d799fb934 | backend sampling kept with a grammar / reasoning budget (device token checked on host, full row fetched on rejection); `llama_get_logits_ith_fetch`; sampler stats; server grammar stats in timings + log | `LLAMA_GRAMMAR_HOST_SAMPLING=1` restores the switch-off (unset = keep backend sampling) | common/sampling.cpp:437 (per `common_sampler_init`) | only with `-bs` (backend sampling) plus a grammar/budget; switch exists. **Timings JSON keys and info log line for every grammar request, no switch** | **yes (cosmetic): grammar stats in timings/log** |
| 17d91b10e | unsigned loop index | none | llama-context.cpp decode | no behaviour change | no |
| 7612006f4 | test hook: every Nth grammar-checked token counts as rejected | `LLAMA_GRAMMAR_TEST_REJECT=N` (unset = 0 = off) | common/sampling.cpp:659-666 (static) | default = off | no |
| 1226b4c18 | pipelined speculation for tool requests until the grammar can reject | via `--pipe-depth` | server `pipe_eligible`, `pipe_step` | only with `--pipe-depth` | no |
| 6bf2495e9 | Q6_K in the Pascal matrix-vector kernels (1-4 tokens) | `GGML_CUDA_MMVQ_PASCAL` (1 = cc61, default; 0 off; 2 any); `GGML_CUDA_MMVQ_PASCAL_SKIP=q6_K` (716c5374c) | mmvq-pascal.cu:886-892, 948-976 (static) | default on (cc61); switches restore `mul_mat_vec_q` | no |
| 081d33e13 | Q4_K in the Pascal matrix-vector kernels | same; `GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K` | same | default on (cc61); switches exist | no |
| 716c5374c | per-type skip list for the Pascal MMVQ kernels | `GGML_CUDA_MMVQ_PASCAL_SKIP` (empty) | mmvq-pascal.cu:948-976 (static), 1003, 1192 | default empty = no change | no |
| af2b1c225 | compute-6.1 prompt attention kernel (replaces stock TILE; reads F16/Q8_0 K/V without F16 copy) | `GGML_CUDA_FATTN_TILE_PASCAL` (1 = cc61, default; 0 off; 2 any); `_MIN` (9 F16 / 3 Q8_0) | fattn-tile-pascal.cu:539-548 (static); fattn.cu:779, 815 | default on (cc61); 0 restores stock TILE and its F16 copy | no |
| 2c2fe51fc | prompt attention: head-128 scores split over head halves | `GGML_CUDA_FATTN_TILE_PASCAL_QK` (2 split, default; 1 old layout) | fattn-tile-pascal.cu:534-537 (static) | inside the fork kernel only | no |
| 6b605eef7 | prompt attention: per-tile staging row pointers | covered by GGML_CUDA_FATTN_TILE_PASCAL | fattn-tile-pascal.cu | inside the fork kernel only | no |
| db4d734be | prompt attention: info log on first use | covered by GGML_CUDA_FATTN_TILE_PASCAL | fattn-tile-pascal.cu:645 | only when the fork kernel runs | no |
| ee6e6391f | test at 1024 tokens | n/a | tests/ | tests only | no |
| cc16884ec | no scheduler reserve when attached backend sampler chains are covered by the last reserve; detach never reserves | `LLAMA_SAMPLER_RESERVE=1` restores a reserve per attach/detach (unset = skip) | llama-context.cpp:366-367 (context creation); 2685, 2695, 2706 | default on (speed only); switch restores upstream | no |
| be5498051 | (a) `seq_rm` leaving a stream empty puts head at 0; (b) state restore into an empty stream places cell p at p (p mod size for SWA) and head after the newest; (c) `LLAMA_STATE_SEQ_FLAGS_SWA_RING` flag | (a), (b) none; (c) API flag | (a) llama-kv-cache.cpp:517-529, 554-558; (b) 2798-2828, 2843, 2920-2949; (c) 2538 | (a), (b) **default on, all models, numerics (cell order)**; (c) only when a caller passes the flag (the server does by default, see d87e419ca) | **yes: (a) and (b)** |
| c6663558a | server agent sessions: `--prefix-cache`, `--checkpoint-gen`, one ring tail checkpoint for SWA models, ring checkpoint past the divergence still restorable, timings keys, log levels | `--checkpoint-tail2` restores the two tail checkpoints and the mid-prompt rule; `--prefix-cache DIR` (off), `--checkpoint-gen` (off); none for the restore rule, timings keys, log level | common/arg.cpp:1812-1870; `ckpt_trim` server-context.cpp:2654 (load: SWA + PART seq_rm + no draft); tail rule 6372-6375, 6401-6403, 6427; restore rule 6057; timings server-common.cpp:167-171; logs 6097, 6106 | tail checkpoint: default on, switch exists; prefix/gen: opt-in; **restore-past-divergence rule, timings keys, SLT_TRC->SLT_INF: default on, no switch** | **yes: ring-past-divergence restore rule; timings keys/log level (cosmetic)** |
| d537d03dc | SWA checkpoint restore resumes at pos_max + 1 (stock: pos_max) | none (keyed on `ckpt_trim`) | server-context.cpp:6089-6093 | **default on for SWA models** | **yes** |
| d87e419ca | prefix files and every SWA checkpoint saved with the whole sliding-window ring; `llama_state_seq_save_file_ext` | none for checkpoints (`ring = ckpt_trim`); prefix files only with `--prefix-cache` | server-context.cpp:3928-3932, 3999-4003 | **default on for SWA models: bigger checkpoints, different restore cells** | **yes** |
| c01d7ca33 | staircase drafter: windows from the layer split, `--pipe-draft-kind` | drafter flags | llama-pipe-drafter.cpp; llama-context.cpp `pipe_draft_init` | only with a drafter | no |
| 9c0b3d7ba | staircase drafter: old in-flight tokens read the final state | drafter only | llama-pipe-drafter.cpp | only with a drafter | no |
| 14a475cb1 | fed tokens refused for M-RoPE/PLE models | none (API error path) | llama-context.cpp `pipe_submit_impl` | only through the pipe API | no |
| 467e22a5b | staircase + store guess order | drafter/store flags | server-context.cpp:2579; common/pipe-guess.cpp | only with `--pipe-depth` | no |
| c74514879 | staircase drafter: flush at most the pending commits (OOB read fix) | drafter only | llama-pipe-drafter.cpp | only with a drafter | no |
| 1290b48f0 | ggml-alloc: `needs_realloc` also tests every leaf against its plan entry | none | ggml-alloc.c:1049-1059 (`ggml_gallocr_needs_realloc`, every non-reused allocation) | **default on, all backends (upstream code)** | **yes (bug fix)** |
| 3ad372ae9 | release defaults: FATTN_PASCAL, INPUT_ON_GPU, INPUT_UPLOAD off unless set | the three env vars | fattn-pascal.cu:256-262; llama-model.cpp:1789-1790; llama-context.cpp:1022-1025 | restores upstream defaults | no |
| b6db46d8c | card-built mask off unless `LLAMA_GPU_KQ_MASK` set | `LLAMA_GPU_KQ_MASK` (unset = 0) | llama-graph.cpp:57-62 (static) | restores upstream host mask | no |
| 39e9423ca | persistent MoE quantizer and fused norm's Q8_1 copy use upstream's new block sum (d * sum of int8 codes, #31247) | `GGML_CUDA_PERSISTENT_MOE` (1), `GGML_CUDA_NORM_Q8` (on) cover the kernels | moe-persistent.cu:871-874; norm.cu:913-916 (static) | inside fork kernels; makes them bit-identical to upstream `quantize_q8_1` | no |
| b8abbf252 | compute-6.1 attention kernels decline kv_rows (src[5]) -> upstream MMA kernel | covered by the two kernels' switches | fattn-tile-pascal.cu (supported check), fattn-pascal.cu | restores upstream routing for kv_rows | no |
| 230506b65 | retire Q4_1_64 (type, ftype, gguf-py) | n/a | ggml.h, llama.h, gguf-py | restores upstream enums | no |
| db6ed9f99 | prompt attention kernel takes upstream's KV_rows argument (unused) | covered by GGML_CUDA_FATTN_TILE_PASCAL | fattn-tile-pascal.cu | signature only | no |
| 165b306f6 | comments of the card-mask port | n/a | llama-graph.*, llama-kv-cache.* | comments only | no |
| 45fbad874 | MUL_MAT_ID MMVQ batch limit clamped to the arch of the device's code | none | mmvq.cu:327-331 (`get_mmvq_mmid_max_batch_llama`) | **default on, NVIDIA (upstream code)**; identical where device runs its own arch (P106) | **yes (pure bug fix)** |
| 879f3b86f | kv_layer members set by name (the cell copy had landed in `k_rot`) | none | llama-kv-cache.cpp:259-268 | fixes a fork-introduced crash; restores upstream `k_rot` | no |
| 15b5de992 | KQ_MASK test fills sentinel tensors | n/a | tests/ | tests only | no |

### Needs a switch

#### Y1. 237590944 (f): sliding-window `build_attn` expands K last

- Where: `src/llama-graph.cpp`, `llm_graph_context::build_attn(llm_graph_input_attn_kv_iswa * inp, ...)` starting at
  line 4198; the expand order at lines 4228-4240:
  `ggml_build_forward_expand(gf, q_cur); if (v_cur) expand(v_cur); if (k_cur) expand(k_cur);`
- Upstream (`git show 20a51f24a:src/llama-graph.cpp`, lines 3566-3576):
  ```
  ggml_build_forward_expand(gf, q_cur);

  if (k_cur) {
      ggml_build_forward_expand(gf, k_cur);
  }

  if (v_cur) {
      ggml_build_forward_expand(gf, v_cur);
  }
  ```
- Off value restores: q, k, v order in the iswa variant (upstream's plain-KV variant already uses q, v, k, so only the
  iswa one changed).
- Effect: node order of every sliding-window model on every backend. It lets upstream CUDA's
  RMS_NORM+MUL+ROPE+VIEW+SET_ROWS fusion fire for K (Flint: 35 kernels fewer per token), and the fork's
  `GGML_CUDA_ATTN_BLOCK` matcher relies on it (an off switch here also stops ATTN_BLOCK matching iswa layers). The commit
  claims logits unchanged; expected speed only, but the order changes which fusions run and can change graph splits, so
  bit-identity is per-backend, not guaranteed.

#### Y2. 237590944 (a): CUDA backend reports a `COMPUTE_6_1` feature (cosmetic)

- Where: `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_get_features` (starts 8058), lines 8114-8122: pushes
  `{ "COMPUTE_6_1", "1" }` when every device has cc 6.1.
- Upstream: no such entry.
- Off value restores: leave the entry out. Its only consumer is `cparams.moe_down_rescale_tg`
  (llama-context.cpp:410-413), which `LLAMA_NO_MOE_DOWN_RESCALE=0` already overrides.
- Effect: one more `COMPUTE_6_1 = 1` in the system-info line and in `ggml_backend_get_features` on P106 rigs. No tokens,
  no speed. Low priority; listed for completeness.

#### Y3. d70c4b29b: card-mask cell copies allocated even with the card mask off

- Where: `src/llama-kv-cache.cpp`, `llama_kv_cache::llama_kv_cache` (starts 67): `kq_cells_ctx` (112), context
  `mem_size` `+ 1` overhead (119), the allocation in `ctx_for_buft` (130-138:
  `if (!other && offload && n_seq_max <= 24 && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) { ... ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, kv_size*n_stream + 1) ... }`),
  `layer.kq_cells` and `kq_cells_il` (267-273). Bookkeeping that runs at default: `kq_cells_mark`/`kq_cells_stale` in
  `clear`, `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`, `update`, `apply_ubatch` (452-938, 1300-1305,
  2108-2120), host-only.
- Upstream (lines 116 and 248):
  ```
  /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()),
  ...
  layers.push_back({ il, k, v, k_stream, v_stream, });
  ```
- Off value restores: no cell-copy tensor (`kq_cells_devs` empty, so `get_kq_cells_resident()` is false and every graph
  keeps the host mask, as it already does with LLAMA_GPU_KQ_MASK unset). Natural gate: allocate only when
  `llama_kq_mask_dev_min_tokens() > 0` (llama-graph.cpp:57-62).
- Effect: memory only. 8 bytes per cell per device per cache in the KV buffers (e.g. about 2 MiB per card for a
  262,144-cell cache), plus a larger reported KV buffer size. No tokens, no speed beyond the allocation.

#### Y4. be5498051 (a): `seq_rm` that empties a stream resets head to 0

- Where: `src/llama-kv-cache.cpp`, `llama_kv_cache::seq_rm` (466). Single sequence: lines 517-529
  (`if (cells.get_used() == 0) { head = 0; } else if (p1 == max && pos_min_head != cells.size()) {...} else if ...`);
  any sequence: 554-558.
- Upstream (lines 473-477 and 497-501):
  ```
  // If we freed up a slot, set head to it so searching can start there.
  if (new_head != cells.size() && new_head < head) {
      head = new_head;
  }
  ```
  (The `pos_min_head` tail-head rule in the same block comes from 5f9d3b7e8, outside this slice.)
- Off value restores: drop the `cells.get_used() == 0` branches (keep the rest).
- Effect: KV cell placement for the next prompt in that stream (e.g. a server slot cleared after a wrapped SWA ring).
  Attention sums keys in cell order, so logits can differ in the last bits from upstream (the commit's aim: match a
  freshly started context). Can alter tokens through rounding. No crash either way.

#### Y5. be5498051 (b): canonical cell placement for a state restored into an empty stream

- Where: `src/llama-kv-cache.cpp`, `canonical_slot` (2798-2828, new) and `state_read_meta` (2830): `canonical` flag 2843;
  2920-2926 (`sinfo = canonical_slot(...)`, falling back to `find_slot`); 2937-2949 (head set after the newest restored
  cell).
- Upstream (line 2584):
  ```
  sinfo = find_slot(ubatch, false);
  ```
  and no head adjustment after `apply_ubatch`.
- Off value restores: always `find_slot`, no head update.
- Effect: where restored cells sit (sequence state loads, server checkpoints, slot files, prefix files): position p in
  cell p (p mod size for a sliding-window ring) instead of from head in saved order. Logits can differ in the last bits
  from upstream (intended to match re-processing). No crash either way.

#### Y6. c6663558a: a ring checkpoint past the divergence is still restored

- Where: `tools/server/server-context.cpp`, checkpoint search lambda, lines 6051-6060:
  `if (cur.pos_max > pos_next && !(ckpt_trim && cur.ring)) { return false; }`. Active by default because since d87e419ca
  every checkpoint of a sliding-window model is a ring one (`ckpt_trim`, line 2654: `n_swa > 0 && seq_rm PART && ctx_dft == nullptr`).
- Upstream (lines 4260-4263):
  ```
  // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
  if (cur.pos_max > pos_next) {
      return false;
  }
  ```
- Off value restores: the plain `cur.pos_max > pos_next` test. `--checkpoint-tail2` does not turn this off at HEAD.
- Effect: which checkpoint a diverging prompt restores and how much of the prompt is re-read (speed); the restore point
  sets where batches split, so logits can differ in the last bits from upstream.

#### Y7. d537d03dc: sliding-window checkpoint restore resumes at pos_max + 1

- Where: `tools/server/server-context.cpp`, lines 6089-6093:
  `pos_next = std::min(pos_next, std::max(cur.pos_min + 1, cur.pos_max + (ckpt_trim ? 1 : 0)));`
- Upstream (line 4287):
  ```
  pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
  ```
- Off value restores: `+ 0` (re-evaluate the token at pos_max, as stock).
- Effect: one token less re-evaluated per restore, and the batch after a restore starts one position later. Logits can
  differ in the last bits from upstream (the commit's aim: match the server that took the checkpoint). Speed: one token.

#### Y8. d87e419ca: every checkpoint of a sliding-window model holds the whole ring

- Where: `tools/server/server-context.cpp`, `create_checkpoint` (3931): `const bool ring = ckpt_trim;` (3932) and
  `flags_tgt = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (ring ? LLAMA_STATE_SEQ_FLAGS_SWA_RING : 0)` (4001-4003); the flag is
  honoured in `llama_kv_cache::state_write` (llama-kv-cache.cpp:2538).
- Upstream (line 2778):
  ```
  cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
  ```
- Off value restores: `ring = false` (no SWA_RING flag). Prefix files take the flag only with `--prefix-cache` (opt-in).
- Effect: memory (commit: +52 MiB per checkpoint for Flint at -ub 512, host or VRAM wherever checkpoints are kept) and a
  restore that occupies more cells, so attention sums over a different extent: last-bit logit differences from
  upstream. Also enables Y6.
- Note: Y6, Y7, Y8 and the one-tail-checkpoint rule all key on `ckpt_trim` (line 2654). One server flag forcing
  `ckpt_trim = false` would restore upstream for all of them at once (and make `tail2` true, i.e. stock offsets
  `{4 + n_ubatch, 4}`); `--checkpoint-gen` already refuses to run without `ckpt_trim`.

#### Y9. c6663558a, d799fb934: server timings keys and log lines (cosmetic)

- Where: `tools/server/server-common.cpp`, `server_slot_stats::to_json` (141): lines 161-165 (`sampled_on_device`,
  `grammar_n`, `grammar_rejected_n` for requests with a grammar) and 167-171 (`cache_src`, `cache_ckpt_n`,
  `ckpt_created_n`, `cache_restore_ms`, always). `tools/server/server-context.cpp`: "restored context checkpoint" and
  "forcing full prompt re-processing" raised from `SLT_TRC` to `SLT_INF` with new text (6097, 6106-6107); new
  "sampling = ..." info line for grammar requests (1199-1205).
- Upstream: `to_json` ends after the `draft_n` block (`return base;`, lines 156-161 of upstream server-common.cpp); both
  checkpoint messages were `SLT_TRC` (upstream server-context.cpp 4289, 4293).
- Off value restores: no extra keys, trace-level messages.
- Effect: response JSON and log output only. No tokens, no speed.

#### Y10. 1290b48f0: ggml-alloc re-plans when a leaf lacks memory its plan entry gave it

- Where: `ggml/src/ggml-alloc.c`, `ggml_gallocr_needs_realloc` (1009), new loop at lines 1049-1059 over
  `graph->leafs` with `ggml_gallocr_node_needs_realloc(galloc, leaf, &galloc->leaf_allocs[i].leaf)`.
- Upstream: after the node/src loop the function ends with
  ```
      }

      return false;
  }
  ```
- Off value restores: no leaf loop.
- Effect: speed only (one loop over the leaves per non-reused allocation, and a re-plan or scheduler reserve where a
  leaf no node reads needs memory). No numerics.
- Bug fix: the off value would reintroduce the segfault at `galloc->buffers[-1]` (ggml_gallocr_alloc_graph+570), but
  only when a plan was made while a leaf held memory from a packed upload: `LLAMA_INPUT_UPLOAD=1`, or pipe slots (whose
  default `pipe_upload` is 2). Not reachable on the default plain decode path.

#### Y11. 45fbad874: MUL_MAT_ID matrix-vector batch limit from the device's code architecture

- Where: `ggml/src/ggml-cuda/mmvq.cu`, `get_mmvq_mmid_max_batch_llama` (323), lines 327-331:
  `cc = std::min({ cc, ggml_cuda_highest_compiled_arch(cc), mmvq_device_code_arch() });` (NVIDIA, not HIP/MUSA).
- Upstream (lines 285-294):
  ```
  int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
      // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
      if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
          if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
              return MMVQ_MAX_BATCH_SIZE;
          }
  ```
- Off value restores: the device's own cc.
- Effect: identical on a P106 running this build (6.1 everywhere). Differs only where a GPU runs code built for an
  older architecture (compute_61 PTX on newer cards, `CUDA_FORCE_PTX_JIT=1`, or a build lacking the device's arch);
  there some MUL_MAT_ID batches go to MMQ instead of MMVQ (different kernel, so numerics differ).
- Pure bug fix: the off value reintroduces the launch failure ("invalid argument": too many warps for the kernel's
  launch bounds) in exactly those cases. Upstream fails there.

### Notes

- 237590944 (e): `gf_in_flight = true` moved from `process_ubatch` into `graph_compute` (llama-context.cpp:4970), so a
  K-shift also marks it. At default it has no observable effect: `memory_update` invalidates the reused graph after a
  K-shift anyway, and with `plain_upload == 0` the reuse sync (2848) and `synchronize()` (1216) reduce to upstream's.
  The `reuse_safe` rule it feeds comes from 0c0a36b22 (outside this slice).
- c6663558a's one-tail-checkpoint default is covered by `--checkpoint-tail2` for the offsets and the mid-prompt rule,
  but not for the ring flag (Y8) or the restore rule (Y6).
- d799fb934's change to sampling with a grammar applies only with `-bs`; `LLAMA_GRAMMAR_HOST_SAMPLING=1` restores
  upstream's switch-off. With `-bs` and a grammar, the default output differs from upstream's (device draws are kept).

## Inventory, commits 101-150

Line numbers are at HEAD. "static" = read once (first use) and cached for the process; "ctx" = read when a
llama_context is created. Kernel-level switches that already existed before this slice: `GGML_CUDA_MMVQ_PASCAL`
(mmvq-pascal.cu:886-892, static, default 1 = compute 6.1 only, 0 off, 2 any NVIDIA) and
`GGML_CUDA_MMVQ_PASCAL_SKIP=<type,...>` (mmvq-pascal.cu:950-974, static, default empty: per-type back to llama.cpp's
`mul_mat_vec_q`).

| commit | behaviour | switch (default; values) | read where | default vs upstream | needs switch |
|---|---|---|---|---|---|
| b05ecc2d1 | A device without a backend registry (the tensor-split meta device) reports no features and no event query, instead of aborting in `ggml_backend_reg_get_proc_address` | none (crash fix in fork-only calls) | src/llama-impl.cpp `llama_backend_dev_has_feature` (fork function, per call); llama-context.cpp `pipe_submit_impl` | same as before for every device that has a registry; only LLAMA_SPLIT_MODE_TENSOR changes (it aborted) | no |
| 854da512c | IQ2_S and IQ3_XXS in the Pascal mmvq kernels (1-4 tokens, MUL_MAT_ID, fused gate/up + GLU, shared expert) | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs` (default empty); `GGML_CUDA_MMVQ_PASCAL=0` (all) | mmvq-pascal.cu:953 (static), :888 (static) | default on for files holding IQ2_S/IQ3_XXS on cc 6.1 (FP32 scale products, sums in another order than mul_mat_vec_q); other types' code paths unchanged (table words 0) | no |
| c3dbca232 | IQ2_S/IQ3_XXS stage a fused gate/up's two matrices one after the other (`mmvqp_staged_mats` = 1, as Q6_K) | none of its own; the type's kernel via `_SKIP` | mmvq-pascal.cu:125 (constexpr) | inside 854da512c's kernel only | no (first-version detail, 12 min after 854da512c, never measured alone; see Notes) |
| 3f4f9c379 | Compile-time tuning macros of the grid types `MMVQP_IQ_SIGN_LUT`, `MMVQP_IQ_MIN_BLOCKS`, `MMVQP_IQ_R`, at the then-current values (1, 1, 1); perf cases | `-D` macros (build time) | mmvq-pascal-impl.cuh:39-59 | defaults left the kernel as it was | no |
| 02cd9905a | `MMVQP_IQ_PERSIST`: one-token grid-type blocks loop over row blocks | `-DMMVQP_IQ_PERSIST=n` (default 0 = off) | mmvq-pascal-impl.cuh:39-41; mmvq-pascal.cu:416, :1153 | off = previous (tables filled under `bx == blockIdx.x`, always true at 0) | no |
| 2c82e8ffe | tests only (`GGML_TEST_RAW_QUANT`) | - | tests/test-backend-ops.cpp | test only | no |
| 76788bf4a | IQ2_S/IQ3_XXS defaults changed: sign masks computed in registers (`SIGN_LUT` 1 -> 0), registers for 3 blocks per SM (`MIN_BLOCKS` 1 -> 3, both the 1-token and the 2-4 token kernels) | no env var; previous only by rebuild: `-DMMVQP_IQ_SIGN_LUT=1 -DMMVQP_IQ_MIN_BLOCKS=1` (via `GGML_CUDA_PASCAL_FLAGS`) | mmvq-pascal-impl.cuh:42-50 (compile time) | tuning of a fork kernel, default on | **yes (tuning; build-time only)** |
| 9b3dd8c1c | The hybrid memory's attention part (Qwen3.5/3.6, Ornith, Qwen3.8) may build its KQ mask on the devices | `LLAMA_GPU_KQ_MASK=<n>` (default 0 = off: host mask); `LLAMA_GPU_KQ_MASK_CHECK=1` compares | llama-graph.cpp:57-63, :67-73 (static) | off = upstream (`kq_dev.upd` null: host mask, upstream `can_reuse_attn_mask`) | no |
| 983a08ae4 | Pipelined decoding for delta-net hybrids: one recurrent state group per pipe slot + stash; recurrent-memory refactor (`alloc_tensors`, `ring_pos`, `src_grp0`, `set_n_groups`) | `llama_pipe_init` API / `--pipe-depth N` (default 0 = off) | common/common.h:670, common/arg.cpp:1668 (CLI); server-context.cpp:2535 | only when `llama_pipe_init` runs; with one group (default) offsets, `s_copy`, `seq_rm`, `state_write` equal upstream's (grp 0) | no |
| b1ec1414c | tool only (`llama-pipe-recurrent-check`) | - | examples/ | tool only | no |
| 5e4e304fb | Device-built mask also for M-RoPE token batches | `LLAMA_GPU_KQ_MASK` (default 0) | llama-graph.cpp:78-82 | off = upstream | no |
| 2382b44ed | `LLAMA_GPU_KQ_MASK_CHECK=1` synchronizes before a pipe slot's inputs | `LLAMA_GPU_KQ_MASK_CHECK` (default 0) with `--pipe-depth` | llama-context.cpp:3356-3366 (static) | opt-in debug | no |
| 2626a3353 | Pipelining for qwen4exp: conv rows into the slot's state group; `llama_pipe_init` accepts QWEN4EXP; stash copy on the slot's own backend | `--pipe-depth` (default 0) | qwen4exp.cpp:1575-1584 (`grp = get_grp()`, 0 without a pipe) | default = previous (grp 0 = slot 0 at n_rs_seq 0) | no (but see Notes: the fused conv path ignores the group) |
| 4308deced | `llama_pipe_init` logs each device's pipe-slot compute buffers | `--pipe-depth` | llama-context.cpp `pipe_init` | log only, opt-in | no |
| 8c25952e4 | tests only (HC_STREAM_MEAN) | - | tests/ | test only | no |
| ab8deade8 | Pascal BF16 matrix-vector kernel (1-4 tokens), hooked ahead of upstream's mmvf/cuBLAS choice in `ggml_cuda_mul_mat` and inside `ggml_cuda_mul_mat_vec_f`; backend feature `MMVF_PASCAL` | `GGML_CUDA_MMVF_PASCAL` (default 1 = cc 6.1; 0 off; 2 any NVIDIA); `_NSM`; `_SHAPE=G,R` (forces one shape for all) | mmvf-pascal.cu:47-51, :53-57, :59-73 (static); hooks ggml-cuda.cu:2479, mmvf.cu:653 | default on for BF16 weights on cc 6.1 (FP32 sums in another order) | no |
| 59c5d4363 | tests only (same HC_STREAM_MEAN case) | - | tests/ | test only | no |
| 98bbd6a2f | Q2_0 in the Pascal mmvq kernels (byte_perm lookup, warp-tile staging of 180-byte rows) | `GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0`; `GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T` (shape; default 0,0,1) | mmvq-pascal.cu:953, :924-936 (static) | default on for Q2_0 tensors on cc 6.1; other types unchanged (tile_mode 0) | no |
| 62774f688 | Server: context checkpoints in device stores, checkpoints spread inside prompts; `llama_state_seq_*_store` API; `ckpt_save_ms`/`ckpt_dev_n` in timings | `--ctx-checkpoints-vram N` (0), `--ctx-checkpoints-vram-sync` (off), `--ctx-checkpoints-spread N` (0) | common/arg.cpp:1771, :1783, :1790; common.h:683-685 (start-up) | off = previous; on the default path only a save timer and two extra JSON timing fields | no |
| c70687382 | Q2_0 MUL_MAT_ID of 2-8 tokens reads each distinct expert once (mtok), ahead of `mul_mat_vec_q_pascal_mc` (2-4) and llama.cpp's kernels (5-8) | `GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G,T` (default on; `0` off); `GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0` | mmvq-pascal-mtok.cu:502-516, :519-536 (static); hook mmvq.cu:1702 | default on for Q2_0 expert products of 2-8 tokens on cc 6.1 (stated bit-identical to the one-token kernel) | no |
| 2bfbb54fc | `--ctx-checkpoints-spread` rule: N/2 / N spacing, gap eviction counts the new checkpoint | `--ctx-checkpoints-spread` (0) | arg.cpp:1790 | off; the spread = 0 eviction order unchanged | no |
| fd2381805 | New op `GGML_OP_GDN_CONV` (CPU) | used only when a graph builds it (`LLAMA_GDN_FUSE`) | ggml.c, ggml-cpu/ops.cpp | not on a default path by itself; the enum value is inserted before `GGML_OP_UNARY` (see Notes) | no |
| cf7763b6d | CUDA recurrent-layer fusions: GDN initial state read in place (get_rows gather skipped), GDN_CONV kernel + its cpy, gated rms norm (RMS_NORM, MUL, SIGMOID, MUL) in one launch; match any graph of that shape (upstream Qwen3.5/3.6/Next too) | `GGML_CUDA_GDN_FUSE` (default 1; 0 off; also off with `GGML_CUDA_DISABLE_FUSION`) | ggml-cuda.cu:186-191 (static); checks :3544, :3632, :3652, :3709 | default on for CUDA GDN graphs; upstream GDN kernels unchanged when `rows == nullptr` | no |
| 3f4f0853e | qwen4exp graph: conv window + q/k L2 norm as one `ggml_gdn_conv`; the gate's producers first so the gated norm's four nodes are consecutive | `LLAMA_GDN_FUSE` (default 1; 0 = graph as before) | qwen4exp.cpp:12-21 (static), used :1283 and in `build_norm_gated` | default on for qwen4exp | no (see Notes: pipe group) |
| 6cee2174b | tests only (fusions at 1-8 tokens) | - | tests/ | test only | no |
| 455cc53b4 | tests only (`LLAMA_ARCHS_LOGITS`, `LLAMA_ARCHS_N_RS_SEQ`) | - | tests/ | test only | no |
| b5ed2f4fe | Log line per state-store allocation | none | llama-context.cpp state-store writer | only when a device store is allocated (ON_DEVICE / store API) | no |
| 1e0caf230 | Draft memory: draft reads the target's embeddings/head, draft head over a token shortlist, settable direct-I/O staging; `llama_model_params.share_src/share_tensors` | `LLAMA_DRAFT_SHARE=embd[,head]` (unset = own copies); `LLAMA_DRAFT_SHORTLIST=k[,m[,r]]` (unset = full head); `LLAMA_DIO_STAGING=<n>x<MiB>` (unset = 4 x 64 MiB, the old constants) | common/speculative.cpp:3398, :2154; llama-mmap.cpp:52-84 (static) | off = previous (constants unchanged, `share_src` null, `head_short_on` false) | no |
| 2989d6b59 | Fused GDN conv L2 norm and gated rms norm sum their squares as `rms_norm_f32`'s vector path (bit for bit the unfused ops); gated norm limited to rows <= 128 floats, 16-byte aligned | the fusion switches (`GGML_CUDA_GDN_FUSE`, `LLAMA_GDN_FUSE`) | gdn-fused.cu | inside the default-on fusion; after it fused = unfused numerically | no (removes a numeric difference; see Notes) |
| 29450398c | k-column (kcol) kernels for plain MUL_MAT of 1-8 tokens (Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS), hooked first in `ggml_cuda_mul_mat_vec_q`; **and** BF16 Pascal kernel taken to 8 tokens, R = 4 (was 2) at 3-4 tokens | kcol: `GGML_CUDA_MMVQ_PASCAL_KCOL` (default 1; 0 off; 2 everywhere), `_SKIP`, `_R`, `_ITT`, `_SMEM`, `_NSM` (static). BF16: only `GGML_CUDA_MMVF_PASCAL=0` (whole kernel) | mmvq-pascal-kcol.cu:21-24, :46-66; hook mmvq.cu:1584; mmvf-pascal.cu:28, :82 | kcol default on (1 column only for Q3_K, IQ4_NL, short-row Q8_0); BF16 5-8 tokens now the Pascal kernel instead of upstream's path; BF16 3-4 tokens R 4 | kcol: no. **BF16 part: yes (tuning/extension)** |
| 766c92a22 | GDN conv kernel: the output pointer no longer shadows the shared row (2989d6b59 wrote the norm row into the output and read it back from there) | fusion switches | gdn-fused.cu | bug fix inside the default-on fusion | no |
| e1fa4c573 | kcol decoding also buildable for the host (`KCOL_HD`, `prmt`, `h2f`) | - | mmvq-pascal-kcol-impl.cuh | device code unchanged | no |
| 7b9e8c6db | kcol for Q4_0, Q5_0, IQ4_NL | `GGML_CUDA_MMVQ_PASCAL_KCOL`, `_SKIP=q4_0,q5_0,iq4_nl` | mmvq-pascal-kcol.cu | default on for those types | no |
| 00d662c3a | BF16 Pascal launch shape at 3-4 tokens: lane-group size from the width capped at 32, a short grid grows G before lowering R | none for the rule; `GGML_CUDA_MMVF_PASCAL_SHAPE=G,R` forces one shape for every product (not the old per-tensor rule) | mmvf-pascal.cu:81-110 (per call) | tuning of a fork kernel, default on; changes G, so the sum order | **yes (tuning)** |
| af46311af | Q2_0 one-token gate/up takes R = 1 when R = 2 leaves fewer than 4 blocks per SM | `GGML_CUDA_MMVQ_PASCAL_Q2_0=2` forces R = 2 (the previous default exactly); `GGML_CUDA_MMVQ_PASCAL_NSM` | mmvq-pascal.cu:924-936 (static), rule :1031-1040 | tuning, default on; switch restores it | no |
| f379fe30b | Sparse decode attention below Turing over the QSA mask's cells | `GGML_CUDA_FATTN_SPARSE_PASCAL` (default 0 off; 1 below Turing; 2 any) + `_NQ`, `_SPLITS`, `_NSM`, `_FAULT`, `_MIN_RATIO` | fattn-sparse-pascal.cu:24-40 (static); hook fattn.cu:808 | off = upstream | no |
| ff8e1ec24 | Pascal decode attention kernel's transpose-reduce kept in registers (same arithmetic) | `GGML_CUDA_FATTN_PASCAL` (default 0 = kernel off) | fattn-pascal.cu:256-262 (static) | kernel is opt-in; default = upstream | no |
| fd549b754 | Bytes per token: `LLAMA_MOE_TAIL` (drop low-share routed experts), `LLAMA_HEAD_BOUND` (main head over a shortlist with an exact bound and full-head fallback) | `LLAMA_MOE_TAIL=p=..,w=..,renorm=..,hist=..` (unset = off); `LLAMA_HEAD_BOUND=k[,m]` (unset = off) | llama-context.cpp:527 (ctx); common/common.cpp:1663 | off = previous (`build_moe_ffn` unchanged, `head_bound_ith` returns INFINITY) | no |
| 081ffb81f | Sparse FA mode 1 threshold per query token | `GGML_CUDA_FATTN_SPARSE_PASCAL=1` | fattn-sparse-pascal.cu | opt-in | no |
| bdb902d95 | Upstream `set_input_kpool`: padded k-pool entries' cells found by marking the reps below n_pad + n_reps instead of sorting all reps + binary search; `LLAMA_QSA_INPUT_STATS` logging | pad search: none. Stats: `LLAMA_QSA_INPUT_STATS=N` (default 0) | llama-memory-hybrid-idx.cpp:880-905 (per ubatch); qwen4exp.cpp:791-800 (static) | pad search on by default inside upstream's code (same cells, host time only) | **yes (upstream code, speed only)** |
| d86fc66c2 | Port of #31156: one graph result per batch size of up to 8 tokens; optional per-size schedulers | `LLAMA_GRAPH_PER_SIZE` (default 1; 0 = the two arenas of before; added by a476ab83f); `LLAMA_SCHED_PER_SIZE` (default 0) | llama-context.cpp:617-630 (ctx) | default on; 0 restores the old keying (`gf_res_key`) and the shared scheduler | no |
| 66ba1f654 | Port (scheduler scratch per graph): the scheduler's context buffer is sized for the graph being split and grown by realloc; allocation-dependency views go to a separate `dep_ctx` | none | ggml-backend.cpp:1517-1535, :1937-1948, :2010, :2466 | default on in upstream's scheduler (host memory only) | **yes (memory only)** |
| 601525521 | Port of #31274: graph metadata buffer calloc'd once, untouched pages stay non-resident | `LLAMA_GRAPH_META_CALLOC` (default 1; 0 writes one byte per page, as the old vector) | llama-graph.cpp:1768-1782 (static) | default on; 0 restores residency | no |
| a476ab83f | The switches above for d86fc66c2 / 601525521, per-size logs, memory breakdown counts per-size schedulers | (see d86fc66c2, 601525521) | llama-context.cpp:617-630; llama-graph.cpp:1768 | - | no |
| 314b7d1e3 | IQ2_XXS, IQ2_XS, IQ3_S in the Pascal mmvq kernels | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_xxs,iq2_xs,iq3_s` | mmvq-pascal.cu:953 | default on for those types on cc 6.1; existing types unchanged | no |
| 4085845a6 | tests only | - | tests/ | test only | no |
| 6c88b6991 | tests only | - | tests/ | test only | no |
| dafb09a4a | `LLAMA_SPEC_DUMP`: one JSON line per speculative verify step | `LLAMA_SPEC_DUMP=<file>` (unset = off), `_K`, `_TGT_K`, `_NGRAM`, `_FULL_BLOCK` | tools/server/server-context.cpp:249 (start-up) | off = previous (nothing recorded) | no |
| 9e1a8492e | `__launch_bounds__` minimum blocks as macros (defaults = previous, machine code identical); CMake `GGML_CUDA_PASCAL_FLAGS` for the Pascal files | build time (`-D`, CMake cache, default empty) | ggml/src/ggml-cuda/CMakeLists.txt:135-146; the impl headers | identical by default | no |
| 04d37fe26 | Port of #31569: fit adds the skipped NextN layers to `n_gpu_layers` and to the output device's tensor split | `LLAMA_FIT_NEXTN_SKIP` (default on; 0 = left out, as before; added later by 6ce691d8f) | common/fit.cpp:60-70, :171 (getenv per fit call) | default on (`--fit` defaults on) for models with NextN layers loaded without MTP | no |

### Needs a switch

#### 76788bf4a: IQ2_S / IQ3_XXS sign masks in registers, 3 blocks per SM

- Where at HEAD: ggml/src/ggml-cuda/mmvq-pascal-impl.cuh:42-50 (`#define MMVQP_IQ_SIGN_LUT 0`, `#define MMVQP_IQ_MIN_BLOCKS 3`,
  `MMVQP_IQ_MC_MIN_BLOCKS` defaults to it), used by `mmvqp_init_tabs` (:102, :110, the sign-mask table fill),
  `mmvqp_iq_masks` (`#if MMVQP_IQ_SIGN_LUT`, ~:511-525) and the `__launch_bounds__` of `mul_mat_vec_q_pascal`
  (mmvq-pascal.cu:414) and `mul_mat_vec_q_pascal_mc` (:448) through `mmvqp_min_blocks` / `mmvqp_min_blocks_mc` (impl :64-69).
- Previous (3f4f9c379 / 02cd9905a):
  ```
  #define MMVQP_IQ_SIGN_LUT 1     // masks from the shared-memory table
  #define MMVQP_IQ_MIN_BLOCKS 1   // __launch_bounds__(128, 1) for the 1-token and 2-4 token kernels
  ```
- Today the previous state is reachable only by rebuilding (`GGML_CUDA_PASCAL_FLAGS="-DMMVQP_IQ_SIGN_LUT=1;-DMMVQP_IQ_MIN_BLOCKS=1"`),
  and that also moves IQ2_XXS / IQ2_XS / IQ3_S (314b7d1e3, born with 3 blocks per SM) off their own default. A runtime
  off value must give IQ2_S and IQ3_XXS the table masks and the `(128, 1)` bound: a second instantiation (a template
  flag for the mask source plus kernels with the other bound), picked per launch from a static env read in
  `ggml_cuda_mmvq_pascal`.
- Tokens/logits: no. The table and the register computation give the same mask bytes, and the register limit does not
  change the arithmetic. Speed only.

#### 29450398c (BF16 part): Pascal BF16 kernel to 8 tokens, R = 4 at 3-4 tokens

- Where at HEAD: ggml/src/ggml-cuda/mmvf-pascal.cu:28 `#define MMVF_PASCAL_MAX_COLS 8`, the limit in
  `ggml_cuda_mmvf_pascal_supports` (:170, `src1->ne[1] <= MMVF_PASCAL_MAX_COLS`), which gates both hooks
  (ggml-cuda.cu:2479 in `ggml_cuda_mul_mat`, mmvf.cu:653 in `ggml_cuda_mul_mat_vec_f`); `mmvfp_pick_shape` :82
  `const int rmax = ncols_dst <= 4 ? 4 : 8;`.
- Previous (ab8deade8):
  ```
  #define MMVF_PASCAL_MAX_COLS 4
  const int rmax = ncols_dst <= 2 ? 4 : 2;
  ```
  so BF16 products of 5-8 tokens went through upstream's dispatch (`ggml_cuda_should_use_mmvf` -> `mul_mat_vec_f`, or
  cuBLAS), and 3-4 tokens used at most 2 rows per group.
- What an off value must restore: (a) a token limit, e.g. `GGML_CUDA_MMVF_PASCAL_MAX_COLS=4`, read once (static) and used
  in `supports()`; (b) the 3-4 token rule `rmax = 2`. (b) can share one "rule version" switch with 00d662c3a below.
- Tokens/logits: (a) yes. 5-8 token batches (speculative verify) change kernel, so the FP32 summation order changes and
  low bits of the logits differ (it can flip near-ties). (b) no. R only changes how many rows share an activation load:
  each row's lanes and its reduction tree depend on G only. Speed only.

#### 00d662c3a: BF16 launch shape at 3-4 tokens (Q611)

- Where at HEAD: ggml/src/ggml-cuda/mmvf-pascal.cu `mmvfp_pick_shape`, :81-110: `const bool mid = ncols_dst == 3 || ncols_dst == 4;`,
  `while (G < (mid ? 32 : 128) && nchunks > 16 * G)`, and the `if (mid) { while (G < 128 && blocks < target && nchunks >= 8*G) G *= 4; }`
  block before the R loop.
- Previous (29450398c):
  ```
  G = 8;
  while (G < 128 && nchunks > 16 * G) { G *= 4; }
  const int target = 4 * nsm;
  R = rmax;
  while (R > 1 && mmvfp_blocks(nrows, G, R) < target) { R /= 2; }
  while (G < 128 && mmvfp_blocks(nrows, G, R) < target && nchunks >= 8 * G) { G *= 4; }
  ```
- `GGML_CUDA_MMVF_PASCAL_SHAPE=G,R` does not give this back: it forces one shape on every BF16 product, while the old
  rule picks per tensor (10,240-wide rows G = 128, 2,560 x 48 G = 128 R = 1, and so on). An off value (e.g.
  `GGML_CUDA_MMVF_PASCAL_RULE=0`, static) must set `mid = false`, which is the code above.
- Tokens/logits: yes, slightly. G sets the number of lanes per row and the shuffle / shared-memory reduction over them,
  so the FP32 summation order of 3-4 token BF16 products (Flash-Next's hyper-connection and alpha/beta projections in
  verify batches) changes. Low bits only, but it can flip a near-tie.

#### bdb902d95: QSA k-pool pad cells found in linear time (upstream code)

- Where at HEAD: src/llama-memory-hybrid-idx.cpp `llama_memory_hybrid_idx_context::set_input_kpool`, :880-905 (the
  `is_rep` vector, `bound = min(kv_size*n_stream_kv, n_pad + n_reps)`, `while (pad_cell < bound && is_rep[pad_cell])`,
  `GGML_ASSERT(pad_cell < bound)`). Runs every ubatch of a qwen4exp QSA model (upstream architecture).
- Upstream b12076:
  ```
  std::vector<int64_t> reps(pcell, pcell + pool_end.size());
  std::sort(reps.begin(), reps.end());
  int64_t pad_cell = 0;
  for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
      while (std::binary_search(reps.begin(), reps.end(), pad_cell)) { ++pad_cell; }
      GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
  ```
- An off value (e.g. `LLAMA_QSA_PAD_SORT=1`, static) restores the sort + binary search.
- Tokens/logits: no. Both pick the same lowest non-rep cells, so the inputs are identical. Host time per ubatch only
  (the commit: about 8k reps sorted per decoded token at 32k cells). The value of the switch is the A/B rule, not risk.

#### 66ba1f654: scheduler context sized for the active graph, dependency views in `dep_ctx` (upstream code)

- Where at HEAD: ggml/src/ggml-backend.cpp `ggml_backend_sched_split_graph` :1517-1535 (frees `dep_ctx`, sizes the context
  buffer from `max(n_nodes, n_leafs)`, `realloc` when it grows), :1937-1948 (`dep_ctx = ggml_init(n_dep_nodes * ggml_tensor_overhead())`),
  :2010 (`ggml_view_tensor(sched->dep_ctx, keep[k])`); `ggml_backend_sched_new` :2466 (only `context_graph_overhead`,
  no buffer); `ggml_backend_sched_free` :2513.
- Upstream b12076:
  ```
  // ggml_backend_sched_new
  sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
  sched->context_buffer = (char *) malloc(sched->context_buffer_size);
  // ggml_backend_sched_split_graph: ggml_init on that fixed buffer, and
  struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
  ```
- a476ab83f put "each behaviour change of #31156 and #31274" behind a switch but left this one out. An off value
  (e.g. `GGML_SCHED_CTX_PER_GRAPH=0`, read once in `ggml_backend_sched_new`) must allocate the fixed buffer in
  `sched_new`, skip the per-graph realloc and create the dependency views in `sched->ctx` (no `dep_ctx`).
- Tokens/logits: no. Only the size and residency of the scheduler's host scratch (one per scheduler, so it multiplies
  with the per-size schedulers and pipe slots). Splits, tensors and results are the same. Memory only, which counts
  on the 4 GB rig host.

### Notes

- **Likely bug at HEAD (only with `--pipe-depth` on qwen4exp):** `llama_model_qwen4exp::graph::build_gdn_conv_at`
  (qwen4exp.cpp:1593-1648, the default path since 3f4f0853e, `LLAMA_GDN_FUSE=1`) writes the new conv histories at
  `kv_head * row_size` with `mem_size` rows per rollback slot. It never adds `mctx_cur->get_grp() * mem_size`, which
  2626a3353 added to the unfused `build_conv_state_at` (:1575-1584) and 983a08ae4 to delta-net-base (:479, :568). The
  CUDA cpy fusion writes the same view. The reads go through `s_copy`, which does include the group
  (`src_grp0 * size + src0`). So a pipe slot would read its conv history from its own group and write it to group 0.
  Not run here. `LLAMA_GDN_FUSE=0`, or `grp * mem_size` in the `dst` view's offset at n_rs_seq 0, would avoid it.
- **Enum shift:** fd2381805 inserts `GGML_OP_GDN_CONV` before `GGML_OP_UNARY` (ggml.h:599), so `GGML_OP_UNARY` through
  `GGML_OP_GLU` get values one higher than upstream's. No runtime switch can cover that; it matters only for anything
  that exchanges op numbers between builds (ggml-rpc). Appending the op after the upstream ops would avoid it.
- **Borderline, marked no:**
  - c3dbca232 changed the gate/up staging of a kernel that was 12 minutes old and never measured. Only
    `GGML_CUDA_MMVQ_PASCAL_SKIP` covers it (the whole type). If the owner wants every staging choice A/B-able, the off
    value is `mmvqp_staged_mats` returning `M` (not 1) for IQ2_S/IQ3_XXS. Speed and shared memory only.
  - 2989d6b59 changed the sum order inside the fused GDN conv and gated-norm kernels. The new order is
    `rms_norm_f32`'s, so the default (fused) now matches `GGML_CUDA_GDN_FUSE=0` bit for bit. The older order existed for
    9 minutes (cf7763b6d 22:39 to 2989d6b59 22:48) and differed from upstream. Restoring it would bring back a
    difference from upstream, so no switch.
  - ff8e1ec24 is a codegen-only change (same arithmetic, no stack frame) in a kernel that is off by default
    (`GGML_CUDA_FATTN_PASCAL=0`).
- **Dispatch-order hooks in this slice, each with its own off switch:** kcol first in `ggml_cuda_mul_mat_vec_q`
  (mmvq.cu:1584, `GGML_CUDA_MMVQ_PASCAL_KCOL=0`), mtok before `ggml_cuda_mmvq_pascal` (mmvq.cu:1702,
  `GGML_CUDA_MMVQ_PASCAL_MTOK=0`), BF16 Pascal before upstream's mmvf/cuBLAS choice (ggml-cuda.cu:2479,
  `GGML_CUDA_MMVF_PASCAL=0`), GDN fusions before topk-moe and the other fusions in `ggml_cuda_try_fuse`
  (`GGML_CUDA_GDN_FUSE=0`), sparse FA after the Pascal FA in `ggml_cuda_flash_attn_ext` (fattn.cu:808, off by default).
- **Default-path additions with no behaviour change:** 62774f688 adds `ckpt_save_ms` and `ckpt_dev_n` to the server's
  timings JSON and times every checkpoint save. fd549b754 calls `llama_head_bound_ith` twice per `common_sampler_sample`,
  and it returns INFINITY when the switch is off.

## Inventory, commits 151-197 (47 commits) and the 33 merges of 20a51f24a..e3b30303d

Line numbers are at HEAD. "static" = read once per process (function-local static); "ctor" = once per llama_context.
Read only: git show / git diff / grep; nothing built or run.

### Commits

| commit | behaviour | switch (default; values) | read where | default vs upstream | needs switch |
|---|---|---|---|---|---|
| db2f29ea9 | upstream #31569 (cherry-pick, 2 of 3): `hp_ngl_skip` passed by reference to every measurement call (refactor of 04d37fe26's pointer arg) | covered by `LLAMA_FIT_NEXTN_SKIP` (on unless "0"; 0 = `hp_nsk` 0, the NextN block at fit.cpp:691 never runs = b12076) | common/fit.cpp:67 `common_fit_switch`, :171; getenv on every measurement call | default on = upstream's later fix; `=0` restores b12076 exactly | no |
| d7630f4b5 | #31569 (3 of 3): public `common_get_device_memory_data` gains the `hp_ngl_skip` out-param (API signature only) | same | common/fit.h:101-108; no caller outside fit.cpp | API only | no |
| 65e3effbc | upstream #31610: a draft (extra) model is measured with a no-alloc main context as `ctx_other` | `LLAMA_FIT_DRAFT_CTX` (on unless "0"; 0 = measured alone, as b12076) | common/fit.cpp:324, per measurement | default on; `=0` restores b12076 | no |
| 6ce691d8f | the two switches above; with `LLAMA_FIT_DRAFT_CTX` a `LLAMA_DRAFT_SHARE` draft is measured without the tensors it reads from the target | `LLAMA_FIT_DRAFT_CTX`, `LLAMA_FIT_NEXTN_SKIP` (both on unless "0"); `LLAMA_DRAFT_SHARE` (unset = 0) | fit.cpp:67/171/324 per call; common/common.cpp:1294 (`mparams_dft.share_tensors`) | share bits only with `LLAMA_DRAFT_SHARE` | no |
| 3e49981fb | #30137 reference-counted checkpoints (shared_ptr list, `server_prompt_checkpoints_copy`), `--ctx-checkpoints-share`; ALSO always on: state buffers not zero-filled (`common_state_data`) and the prompt-cache save drops a state copied short | `--ctx-checkpoints-share` / `LLAMA_ARG_CTX_CHECKPOINTS_SHARE` (default off; on = cache entries and child slots share host checkpoints) | common/arg.cpp:1803; server-context.cpp:2641, 6191, 6697 (params, at load) | sharing off = deep copies, same as before. **Not switched:** (a) no-zero-fill allocator, (b) whole-state check in `prompt_save` | **YES (a), (b)** |
| b52064398 | `LLAMA_SPEC_ADAPT`: draft length per step from drafter probabilities and a verify-cost table; `LLAMA_SPEC_ADAPT_LOG` | `LLAMA_SPEC_ADAPT` (unset/0 off, 1 on), `_COST`, `_WIDTHS`, `_DRAFT_MS`, `_S`, `_LOG`, ... | common/speculative.cpp:238-270, read once by `server_spec_adapt::init` (server-context.cpp:2494, at load) | off: `result_p` null, no probabilities computed, nothing cut = upstream | no |
| 113cdc36d | k-column kernels for IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_M dense products of 2-8 tokens (mode 1; 1 token untouched) | `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=<type names>` (default empty) per type; `GGML_CUDA_MMVQ_PASCAL_KCOL=0` all kcol; `GGML_CUDA_MMVQ_PASCAL=0` all Pascal | mmvq-pascal-kcol.cu:46-68 (static), :21 (static) | default on (cc 6.1), subject to C268's table; existing types' kcol code unchanged (grid_bytes 0, r_max 2) | no |
| 278e4e42b | multi-token expert kernels for the same IQ types at 2-8 tokens; `get_mmvq_mmid_max_batch` returns 8 for them on cc 6.1 (5-8-token MUL_MAT_ID leaves MMQ for MMVQ); a product mtok then declines runs in chunks within llama.cpp's limit | `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=<types>` or `GGML_CUDA_MMVQ_PASCAL_SKIP=<types>` (default empty) make `ggml_cuda_mmvq_pascal_mtok_takes` false = upstream limit, MMQ at 5-8 tokens, chunk path unreachable; `GGML_CUDA_MMVQ_PASCAL_MTOK=0` | mmvq-pascal-mtok.cu:576-590 (static per type), :503-516 (static); limit mmvq.cu:364-369; chunks mmvq.cu:1716-1740 | default on; per-type skip restores b12076 dispatch exactly (chunks only reachable when the limit was raised) | no |
| 0fd6ce5c1 | tests (test-backend-ops cases) | - | - | test only | no |
| 8ff12e276 | card hops: `LLAMA_HOP_CAST` casts the residual at layer-split boundaries; `GGML_CUDA_HOP_STAGED` pinned-host ring between cards without peer access | `LLAMA_HOP_CAST` (unset/""/0/off/f32 = off; f16, bf16, q8_0), `LLAMA_HOP_CAST_AT`; `GGML_CUDA_HOP_STAGED` (0; 1 no-peer pairs, 2 all), `_SLOTS`, `_SLOT_KB`, `_CHUNKS`, `_STREAM` | src/llama-context.cpp:446 (ctor); ggml-cuda.cu:807 (per destination context, first hop) | off: `build_hop_cast` returns cur, `cudaMemcpyPeerAsync` as before; `hop_guard` stores and name compares inert | no |
| 5b69b0574 | test-hop-cast | - | - | test only | no |
| 973bcf09d | light CUDA profiler modes | `GGML_CUDA_PROFILE=<file>` (unset = off), `GGML_CUDA_PROFILE_MODE=group` (default) `\|layer\|sample=<N>\|graph` | ggml-cuda.cu:5918-5922 (static) | off: prof null, `profile_phases.stale()` false without phases, `capture_gen++` inert | no |
| 7b1632aee | micro-batch taper for prompt calls | `LLAMA_UBATCH_TAPER`, `--ubatch-taper` / `LLAMA_ARG_UBATCH_TAPER`, request `"ubatch_taper"` (default off; on, head=N:S, tail=N:S, short=N:S, min=N) | llama-context.cpp:633 (ctor); arg.cpp:1633; server per batch | off: no plan, `plan_cap` returns n_ubatch (llama-batch.cpp:810-818); the server's per-call `llama_set_ubatch_taper_next` hint is inert | no |
| 3e22fd7c1 | `LLAMA_SPEC_DUMP_FULL_BLOCK` decodes a second DFlash2 block for the dump record | `LLAMA_SPEC_DUMP_FULL_BLOCK=<n>` (0 off; 1 trained block; n >= 2), only with `LLAMA_SPEC_DUMP` | server-context.cpp:265 (static) | off: `draft_full_block` returns early (its `batch.clear()` is inert: each draft starts with one) | no |
| 308cd232b | block verification; grammar-forced tokens appended to drafts; drafts cut | `LLAMA_SPEC_BLOCK_VERIFY`, `LLAMA_SPEC_FORCE` (+`_CHARS`, `_MAX`), `LLAMA_SPEC_DRAFT_KEEP` (all off) | server-context.cpp:135-150 via `server_spec_verify::init` (:2480, at load) | off = upstream; `common_speculative_accept`'s `GGML_ASSERT(n_accepted == 0)` became a loop for n_accepted > 0 (unreachable off); new APIs only | no |
| ba068ae6e | ggml-rpc on the fork's op table: `GGML_OP_COUNT == 104` assert, proto major 108, reject unknown op | compile-time `GGML_RPC=ON` only | ggml/include/ggml-rpc.h; ggml-rpc.cpp `deserialize_tensor` | RPC builds only; a fork RPC peer refuses a stock one (required by the fork's op numbering) | no |
| b8f42a3ff | `LLAMA_SPEC_KEEP_DRAWS`: token rejection keeps its draws across a checkpoint replay | `LLAMA_SPEC_KEEP_DRAWS` (0 off, 1 on) | server-context.cpp:146 (at load) | off = previous | no |
| 61abd4cb3 | route once per verify block | `LLAMA_MOE_ROUTE_ONCE` (unset off; k[,max,hist,dry]); `llama_set_moe_route_once` API | llama-context.cpp:573 (ctor) | `moe_route_once == 0`: build_moe_ffn and decode unchanged | no |
| b3825c215 | route once: leader's experts ordered by the token's own scores | same | same | opt-in only | no |
| 5174e7ce0 | C269/Q825: Q2_0 MUL_MAT_ID without a gate whose row segments aren't 16-byte units (routed down, K = 640) now runs in the mtok kernel (warp tiles, else direct loads) instead of being declined; `GGML_CUDA_MMVQ_PASCAL_MTOK` 5th field T | `GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G,T` (T default 1 = tiles where they fit else direct; 0 = direct always) | mmvq-pascal-mtok.cu:503-516 (static) | default differs from the fork's previous dispatch; **T=0 does not restore the decline**; only whole-kernel switches (`_MTOK=0`, `_SKIP=q2_0`) do | **YES** |
| 4ba3f424d | C267/Q823 k-column launch shapes retuned; C268/Q824 (type, K, k) table sends some 2-8-column products to the other kernels | old shapes: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=49152` (or any of `_KCOL_R`, `_ITT`, `_SMEM`: `pick_old`); old dispatch: `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` (default 1); `_KCOL_SMSM` | kcol.cu:34-43 (static), :95-98 (static), :227-241; kcol-impl.cuh:1295 | both together give the pre-C267/C268 fork exactly (`pick_old` = the old loop; fallback flag 0 there) | no |
| 39c3ab5f1 | Q916: llama.cpp's own mul_mat_vec_q takes the activation block sum from the int8 values in FP32 (Q4_0, Q4_1, Q5_0, Q5_1, IQ1_S) instead of block_q8_1's FP16 `ds.y` | **none** (template arg hard-wired `true`) | ggml/src/ggml-cuda/vecdotq.cuh (compile time) | differs from upstream for every stock-MMVQ product of those types; the Pascal switches do not reach it | **YES** |
| 7071bbc60 | test comment | - | - | test only | no |
| d27845b67 | test comment | - | - | test only | no |
| c1266a056 | N2P / N3P types (ids 500/501), CPU reference, quantize fallback; `GGML_TYPE_COUNT` 44 -> 502 | file-driven (only files holding those types); llama-quantize target | - | no effect on other files: gap ids 44-499 still rejected by gguf (blck_size 0, gguf.cpp:740); type loops guard NULL names | no |
| 4127220b1 | CUDA N2P/N3P: dequantize, get_rows, generic mul_mat_vec_q (vec_dot_n2p/n3p_q8_1), Pascal one-token/mc kernels; block_n2p fields reordered | file-driven; Pascal kernels: `GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p` (accepted: any exact lowercase ggml type name) -> generic mul_mat_vec_q | mmvq-pascal.cu:950-973 (static) | only N files; Q6_K/grid half-word code paths unchanged | no |
| e51b90f02 | MMQ (prompt) for N2P/N3P | file-driven | - | only N files | no |
| db9b059ed | perf tests | - | - | test only | no |
| 0b8427657 | N2P-128 / N3P-128 (ids 502/503), CPU; `GGML_TYPE_COUNT` 504; llama-quantize falls back N2P -> N2P-128 on 128-multiple rows | file-driven / quantize target | - | only N files | no |
| f1c692ef7 | Q238 cold tier: a layer's routed experts in hot + cold tensors (`*_exps_cold`, `ffn_exps_tier`) | file-driven (only layers with `blk.N.ffn_down_exps_cold`) | llama-model.cpp `create_tensor_exps_cold`; qwen4exp | no cold tensors: same tensors and same graph (build_moe_ffn body moved into a `build_exps` lambda, same nodes and names); MoE cache skips tiered layers only | no |
| f6c157897 | llama-imatrix `--span-mask`, `--span-dump` | CLI flags of llama-imatrix (LLAMA_EXAMPLE_IMATRIX) | tools/imatrix | tool only | no |
| c722aa389 | CUDA N2P-128/N3P-128 incl. Pascal kernels and MMQ; get_rows `qk` template param (default QK_K), `tile_smem` lambda, N loaders templated | file-driven; `GGML_CUDA_MMVQ_PASCAL_SKIP=n2p_128,n3p_128` | mmvq-pascal.cu:950-973 | existing types unchanged (qk = QK_K; half-word branch: `s.tile` false for Q6_K and grid types; Q2_0 tile_mode unchanged) | no |
| 6356509ef | cold tier: per-tier expert scales | cold-tier files only | - | - | no |
| efce6a24e | mtok kernels for N2P/N3P/N2P-128/N3P-128 (2-8 tokens) | `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=n2p,n3p,n2p_128,n3p_128` (accepted, `mtok_iq_skipped` covers N types) / `_PASCAL_SKIP` / `_MTOK=0` | mtok.cu:576-590 (static per type) | only N files | no |
| a85c08643 | tests | - | - | test only | no |
| 1a0832375 | N2P/N3P CPU dequantizers read through const views, output unchanged | N only | - | - | no |
| e9c545cc4 | cold tier: each slot a MUL_MAT_ID column of its own | cold-tier files only | - | untiered layers unchanged | no |
| f5e600bd2 | test-moe-ids | - | - | test only | no |
| 76683f4e4 | test-backend-ops `--record` / `--replay` | - | - | test only | no |
| a89d6b281 | replay records named by case | - | - | test only | no |
| f3072cbe8 | C120/Q660 `GGML_OP_MOE_TAIL` (op after `GGML_OP_KQ_MASK`, `GGML_OP_COUNT` 105); `LLAMA_MOE_TAIL` defaults to the op (graph=1 keeps the graph ops); -1 ids handled in CUDA kernels (llama.cpp's mul_mat_vec_q, mul_mat_vec_q_moe, get_rows, add_id, MMQ/MMF/MMVF wrappers, host fallback; Pascal and mtok kernels) | `LLAMA_MOE_TAIL` (unset = off; p, w, renorm, hist, graph, skip, repeat); `llama_set_moe_tail_mode` | llama-context.cpp:527 (ctor) | graph unchanged when off. **Not switched:** the -1 checks compiled into upstream kernels and `ggml_cuda_skip_ids` (copy + zero-rows kernels) for any ids not made by ARGSORT/TOP_K | **YES (speed only)** |
| 6e57e208b | a tiered layer takes LLAMA_MOE_TAIL's graph-level cut | needs cold tier and `LLAMA_MOE_TAIL` | - | - | no |
| 3dbf1105a | C408/Q963 `LLAMA_HOP_CAST` below 8 bits; new CUDA CPY F32->Q6_K, Q6_K->F32, IQ4_NL->F32; `supports_op` lists them | `LLAMA_HOP_CAST=q6_k\|q5_1\|q5_0\|q4_1\|q4_0\|iq4_nl` (any case) | llama-context.cpp:446 (ctor) | hop off = upstream. **Not switched:** CUDA now claims the three CPY pairs for every graph (e.g. K-shift of an IQ4_NL K cache) | **YES (minor)** |
| 5aebbddd9 | C143/Q682 card hops through a mailbox in mapped host memory | `GGML_CUDA_HOP_MAILBOX` (0; 1 no-peer pairs, 2 all), `_CHUNKS`, `_SLOTS`, `_SLOT_KB`, `_MAX_KB`, `_STREAM`, `_THREADS`, `_POLL_US`, `_TIMEOUT_MS` | ggml-cuda.cu:827 (per destination context, first hop); hop-mailbox.cu:231-238 | off: falls through to staged / peer copy as before | no |
| c140f9569 | Q961 the MTP draft's token rows from a cache on its layer's card | `LLAMA_DRAFT_EMBD_CACHE=<rows>[,<max tokens>]` (unset = off) | common/speculative.cpp:2123 (drafter init) | off: `cparams.embd_cache` null, table read as before | no |
| 648c720c9 | Q961 the MTP draft's hidden rows kept on the card between passes | `LLAMA_DRAFT_H_CARRY=1` (unset/0 off) | common/speculative.cpp:2103 (drafter init) | off: `inp->h` input as before | no |
| 6fd71e172 | route once's histogram reads the ids before LLAMA_MOE_TAIL's cut | needs both switches and hist=1 | - | - | no |

N2P question: `GGML_CUDA_MMVQ_PASCAL_SKIP` and `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP` accept `n2p`, `n3p`, `n2p_128`, `n3p_128` (exact lowercase ggml type names) and send those types to llama.cpp's generic mul_mat_vec_q / mul_mat_vec_q_moe (the fork's vec_dot_n2p/n3p_q8_1); `_PASCAL_SKIP` also skips them in mtok. `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` parses the names too, but the k-column kernels have no N types (`kcol_type` returns -1), so nothing to skip. With MTOK skipped, 5-8-token N expert products still use MMVQ: the upstream pascal_older table's default case is 8 for unlisted types.

### Merges: changes of their own

Checked with `git show --remerge-diff` (the merge against git's automatic merge of its parents; empty = no own change) and `--cc`.

| merge | own change beyond its parents | effect |
|---|---|---|
| e3b30303d (-hybrid) | none (automatic) | - |
| d29c67616 (-n2p) | none (automatic; --cc hunks are both sides' edits) | - |
| 02a8e196f (-draft1card) | none (automatic) | - |
| 78e22dfcd (-hopcast4) | none | - |
| ffc33bf63 (-mailbox) | none | - |
| b861d7b58 (-replay) | none | - |
| 6a9021389 (-rpc) | ggml-rpc.h: `static_assert(GGML_OP_COUNT == 105)` (was 104), `RPC_PROTO_PATCH_VERSION` 2 -> 3, comment names GGML_OP_MOE_TAIL | RPC builds only; follows -moeskip's new op. No switch needed |
| 3327a3720 (-routeonce) | conflicts resolved as the union: `llama_set_moe_tail_mode` and `llama_set_moe_route_once`/`_counts` (llama-context.cpp, llama-ext.h), `build_moe_tail_op` and `build_inp_route_once`/`build_moe_route_once` (llama-graph.cpp/.h), both cparams groups in `allow_reuse`, both tests in CMake | none of its own |
| 9485352e8 (-spancount) | none | - |
| 450f22c57 (-specdump2) | speculative.cpp DFlash: drops -specdump's `if (i > params.n_max) { walk.push_back(...); continue; }` (as -specdump2 itself did) and keeps -blockver's `\|\| dp.result_p`; speculative.h: both declaration sets | union of the two sides' intents; none of its own |
| 0b25e37bf (-blockver) | none (automatic) | - |
| ef34f7d71 (-coldtier) | none | - |
| 30bb223c2 (-moeskip) | none | - |
| 7bc1552f4 (-taper) | union: `LLAMA_SCHED_PER_SIZE` block + taper env (ctor), `sched_active = nullptr` + taper hints (decode), head-bound block + `llama_set_ubatch_taper_next` (server), test list | none of its own |
| 6da8ee8b0 (-adraft) | union at every drafter: `spec_info_add_marginal` (-specdump) and `spec_push_p` (-adraft); DFlash keeps the `i > n_max` walk branch, then `p_min > 0 \|\| sample \|\| dp.result_p`, then both `walk.push_back` and `result_p->push_back`; server: both `info` and `result_p` draft params | none of its own |
| 11f98affb (-specdump) | none | - |
| 3e7e044e8 (-ckptrc) | none | - |
| 2454d6db5 (-bytesw) | union of the hop-cast block and `LLAMA_MOE_TAIL` parsing (ctor), plus **own**: `moe_tail_collect`'s readbacks and LLAMA_HEAD_BOUND's `t_head_h` readback use `sched_res()` instead of `sched.get()` (now llama-context.cpp:2169, :2174) | only with `LLAMA_MOE_TAIL` hist / `LLAMA_HEAD_BOUND`; `sched_res()` is `sched.get()` without a per-size scheduler. No switch needed |
| 9334adfb7 (-vgraphs) | none | - |
| c34625f1c (-hops) | none | - |
| cb547461b (-lightprof) | none | - |
| b2101bf56 (-qsa) | conflict in qwen4exp.cpp includes: keeps `qwen4exp_gdn_fuse_enabled` | none of its own |
| 298773a4d (-fap) | none | - |
| c96d4a575 (-ccflags) | none (automatic) | - |
| ff6b713e0 (-iqkcol) | none | - |
| 9bcbfee3f (-kcol into -ladder3) | none (automatic) | - |
| 9dab19185 (-kcol into -iqkcol) | none (automatic) | - |
| a330672c1 (-draftmem) | none | - |
| d03e234ac (-ckpt) | none | - |
| 3db80fa67 (-gdnfuse) | none | - |
| 7269203a7 (-mtok) | union: both includes in mmvq.cu (kcol, mtok), both test lists | none of its own |
| b92467b2e (-kcol) | union of perf-test lists | tests only |
| 75b642b90 (-iqmmvq) | union of Q2_0 (98bbd6a2f) and IQ2_S/IQ3_XXS (76788bf4a) in mmvq-pascal: Q2_0's `__launch_bounds__(..., 1)` became `mmvqp_min_blocks<type>()` (1 for non-grid types at that commit), `type == Q6_K` became `mmvqp_halfword`, `sm[]` sized by `seg`, tile staging and IQ tables both kept, type list has all three | inside the Pascal kernels; no own behaviour |

### Needs a switch

#### 1. 3e49981fb (a): state buffers not zero-filled
- Where: common/common.h:1354-1368 (`common_no_init_allocator`, `common_state_data`), used by `common_prompt_checkpoint::data_tgt/data_dft` (common.h:1383-1384), `common_prompt_checkpoint::tgt_to_host` (common/common.cpp:2787), `server_prompt_data::main/drft` (tools/server/server-task.h:731-732), `server_prompt_cache::alloc` (tools/server/server-task.cpp:1804-1810), the spill jobs (tools/server/server-context.cpp:2052, 4098, 4207), `ckpt_read_buf`/`ckpt_write_buf` templated.
- Upstream: `std::vector<uint8_t> data_tgt; std::vector<uint8_t> data_dft;`, and in `alloc` `std::vector<uint8_t> state_data_tgt; ... state_data_tgt.resize(state_size_tgt);`. `resize` zero-fills every byte.
- Off value: zero-fill again. The simplest form is a static flag read in `common_no_init_allocator::construct` (`::new (p) U()` when on). The alternative is the plain `std::vector<uint8_t>`.
- Effect: speed and memory timing only. It saves one memset pass per checkpoint and cache entry, and pages are committed when written rather than at resize. It cannot change tokens, because every buffer is written whole before it is read: `update_tgt`/`update_dft` abort on a short write, `update_tgt_store`/`tgt_to_host`/the spill clear on one, and the cache save is covered by (b).

#### 2. 3e49981fb (b): the prompt-cache save drops a state copied short
- Where: tools/server/server-context.cpp:729-736, `server_slot::prompt_save`.
- Upstream (b12076 server-context.cpp:361-366):
  ```
  llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
  if (ctx_dft) {
      llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
  }
  return true;
  ```
- Off value: ignore the returned byte counts, keep the entry and return true. Tie it to (a)'s off value: with (a) on, a kept short entry would hold uninitialized bytes instead of zeros.
- Effect: this is a pure bug fix. The two versions differ only when `llama_state_seq_get_data_ext` writes fewer bytes than its own size query. Upstream then keeps a corrupt cache entry, and restoring it could change tokens. The fork drops the entry and reprocesses the prompt.

#### 3. 5174e7ce0 (C269/Q825): Q2_0 gate-less expert products of unaligned rows go to mtok
- Where: ggml/src/ggml-cuda/mmvq-pascal-mtok.cu, `ggml_cuda_mmvq_pascal_mtok` (Q2_0 path), lines 886-902 (`tile_ok`, and `st = MTOK_DIRECT` in the last else).
- Previous (5174e7ce0^):
  ```
  const bool tile_ok = !gate && stride_row_x == blocks && wbytes % 16 == 0 && (nrows_x * row_bytes) % 16 == 0 &&
      x_aligned(16) && MMVQP_MTOK_NWARPS * wbytes <= 48 * 1024;
  ...
  } else if (tile_ok) {
      st = MTOK_TILE;
  } else {
      return false;
  }
  ```
  `x_aligned(16)` requires `row_bytes % 16 == 0`, so the routed down projection (K = 640, 180-byte rows) was always declined. It ran on `mul_mat_vec_q_pascal_mc` at 2-4 tokens and on llama.cpp's `mul_mat_vec_q` at 5-8 tokens.
- Off value: the old `tile_ok` (with `x_aligned(16)`) and `return false` instead of `MTOK_DIRECT` for gate-less products. This could be a new value of T (today T=0 means "direct loads always", not "decline") or its own variable.
- Effect: a different kernel computes these products. Per-row FP32 summation order can differ, so logits may change in the last bits; the change is mainly speed.

#### 4. 39c3ab5f1 (Q916): FP32 activation sums inside llama.cpp's own mul_mat_vec_q
- Where: ggml/src/ggml-cuda/vecdotq.cuh.
  - The impl templates gain `bool s32`: Q4_0 at lines 118-148 (s32 branch 139-146), Q4_1 at 154-~180, Q5_0 at 199-232, Q5_1 at 239-~275.
  - The MMVQ callers pass `true` at 860 (`vec_dot_q4_0_q8_1`), 879 (`vec_dot_q4_1_q8_1`), 899 (`vec_dot_q5_0_q8_1`) and 919 (`vec_dot_q5_1_q8_1`).
  - `vec_dot_iq1_s_q8_1` is at 1361-1391 (`sumu`, `d8`).
- Upstream:
  ```
  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
  ...
  return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
  ```
  (likewise Q4_1/Q5_0/Q5_1), and for IQ1_S:
  ```
  const float2 ds    = __half22float2(bq8_1[iqs].ds);
  return d1q * (ds.x*sumi + ds.y*delta);
  ```
- Off value: `s32 = false` at the four MMVQ call sites, and IQ1_S's `ds.y*delta` form.
  - At compile time this is a macro.
  - At run time it needs a bool template parameter threaded through `mul_mat_vec_q` / `mul_mat_vec_q_moe` and chosen on the host. That doubles those instantiations for the 5 types.
- Reach: no Pascal switch covers this, because it sits in the stock kernel. That kernel runs for:
  - Q5_1 and IQ1_S always (they have no Pascal kernel);
  - Q4_0/Q4_1/Q5_0 wherever the Pascal kernels decline (5-8-token experts, shapes);
  - every product on a GPU that is not cc 6.1, and every product with `GGML_CUDA_MMVQ_PASCAL=0`.
- Effect: logits change (FP32 exact block sums instead of the FP16-rounded `ds.y`). It is a bug fix: the off value brings back inf/NaN once a block's activations sum past 65504 (llama.cpp #23606).

#### 5. f3072cbe8 (C120/Q660): -1-id handling always compiled into upstream kernels
- Device checks, compiled in (each branch is never taken for ids >= 0):
  - ggml/src/ggml-cuda/mmvq.cu:732-739 (`mul_mat_vec_q`) and :1020-1026 (`mul_mat_vec_q_moe`): `if (... (int32_t) channel_x < 0) { zeros; return; }`
  - getrows.cu:33-38 (`k_get_rows`), :74-82 (`k_get_rows_kq`), :114-124 (`k_get_rows_float`), :153-160 (`k_get_rows_float_vec`)
  - add-id.cu:22-29 (`add_id_kernel`)
  - topk-moe.cu:170 (`ids_stride`) and :322 (`if (!config.tail)`)
- Host path: mmq.cu:192-199, mmf.cu:15-21, mmvf.cu:637-645 and ggml-cuda.cu:2599/2714 (the `ggml_cuda_mul_mat_id` fallback) wrap the product in `ggml_cuda_skip_ids` (moe-tail.cu:155-179).
  - `ggml_cuda_ids_skip_mode` (moe-tail.cu:88-104) returns SKIP_NEG for any ids whose producer, after views, is not ARGSORT, TOP_K or MOE_TAIL.
  - Such ids get a pool allocation and two extra kernels: `k_skip_ids_fix` copies the ids, and `k_skip_ids_zero_rows` launches one block per slot and token.
  - At default this hits upstream DeepSeek-V4/4.1 hash layers (`ggml_get_rows(ffn_gate_tid2eid, ...)`), GroveMoE (`ggml_cast` ids), the fork's MoE-cache slots and test inputs.
  - The fork's Pascal and mtok kernels carry the same -1 checks.
- Upstream: none of these branches; MMQ/MMF/MMVF read `ids` directly.
- Off value: no -1 checks and no skip-ids copy. One form is a static flag that makes `ggml_cuda_ids_skip_mode` return NONE; the device checks need a macro or template flag. It must stay on whenever `LLAMA_MOE_TAIL` runs in op mode, since that emits -1 ids.
- Effect: speed only (a compare per thread; two small kernels per MUL_MAT_ID with non-argsort ids). Tokens and logits are unchanged for ids >= 0: the copy is byte-identical with dedup off, and zero_rows touches only -1 slots.

#### 6. 3dbf1105a (C408/Q963): CUDA supports_op claims three new CPY pairs
- Where: ggml/src/ggml-cuda/ggml-cuda.cu:7685-7694 (`ggml_backend_cuda_device_supports_op`, GGML_OP_CPY), with the kernels in cpy.cu and cpy-utils.cuh.
- Upstream: IQ4_NL->F32, F32->Q6_K and Q6_K->F32 returned false (the scheduler ran them on the CPU).
- Reach at default: any graph with such a CPY.
  - The main case is the K-shift of an IQ4_NL K cache: llama-kv-cache.cpp:2415 `ggml_cast(ctx, cur, GGML_TYPE_F32)` with `-ctk iq4_nl`, on context shift or cache reuse.
  - A Q6_K cache set through the API also qualifies; the CLI's cache types exclude Q6_K.
- Off value: drop the three `return true`. With `LLAMA_HOP_CAST=q6_k|iq4_nl` the casts would then fall back to the CPU, so tie the off value to those hop types being unset.
- Effect: the dequantization is exact on both, but the scheduler may now run the following ops (Hadamard, RoPE, re-quantize) on the card instead of the CPU. Shifted K can therefore differ in low bits; the change is mainly placement and speed.

