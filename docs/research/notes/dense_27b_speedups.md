# Faster generation for a dense ~27B model (Qwen3.8-27B) on 9 × P106-090 (sm_61, 5 SMs, PCIe 1.0 x1, no P2P), as of 2026-10-22

Scope: every technique that could raise batch-1 generation speed of Qwen3.8-27B on this rig with llama.cpp-family engines. The techniques covered are speculative decoding, tensor parallelism, quantization choice, activation sparsity and layer skipping, runtime and kernel levers, and smaller or pruned variants. All sources were read on 2026-10-22 unless a date is given. Conventions:
- "Claim to test" marks anything not confirmed by a primary source or not measured on this rig.
- "(my calc)" marks this note's own arithmetic.
- "P" and "S" in source lists mean primary (PR, issue, source code, model card, paper) and secondary (blog, community repo, aggregator).

The rig's facts are in the [hardware spec](../../mining_rig.md); measured numbers are in [docs/benchmarks/](../../benchmarks/). Nothing in this note was run on the rig beyond what those benchmark pages report.

**Starting point** (measured, [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md), llama.cpp `85d5703a3`, flash attention on, CUDA graphs inactive):
- Unsloth UD-Q4_K_M (16.46 GB), `-sm layer`:
  - 4 cards: 3.94 tok/s generation (tg128), 50.6 tok/s pp512.
  - 9 cards: 3.43 tok/s generation, 49.1 tok/s pp512.
- Single-card efficiency on Llama-2-7B ([calibration](../../benchmarks/2026-10-22-single-card-calibration.md)):
  - Q4_0: 73 GB/s (42 % of 173 GB/s).
  - Q4_K_M: 60 GB/s.
  - Q8_0: 93 GB/s (on 2 cards).
  - The 5-SM card is limited by instruction issue and unpacking, not by DRAM.
- On the same rig, Qwen3.6-35B-A3B Q4_0 with MTP went from 15.5 to 22.3 tok/s in llama-server, with 77 % of 3 drafted tokens accepted.

Headline findings (each is sourced below):
1. **The main Qwen3.8-27B GGUFs already contain the MTP layer.** All 12 Unsloth and bartowski files whose headers were read carry `blk.64.nextn.*` and `qwen35.nextn_predict_layers = 1`. `--spec-type draft-mtp` works without a separate head file.
   - Unsloth's separate file `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB) is not Q4_0 inside: its NextN block is Q4_K/Q6_K and its output head and embeddings are Q3_K.
   - That makes one draft step read ~0.81 GB with the separate file, against ~1.39 GB when the target's own Q6_K head is used (my calc from GGUF headers).
2. **On a 5-SM card the decisive unknown for speculative decoding is the verify cost.** That is the time to run 2–5 tokens through the target compared with 1. On a P40 (30 SMs), Q4_0 costs 1.02× / 1.11× / 1.45× of a single token for 2 / 3 / 4 tokens, while Q4_K_S costs 1.16× / 1.48× / 1.82× ([PR #20831](https://github.com/ggml-org/llama.cpp/pull/20831) benchmark table).
   - The P106-090 has roughly 2.6× less compute per byte of bandwidth than a P40 (my calc), so its ratios will be worse.
   - Expected MTP gain: about 1.0–1.5× on UD-Q4_K_M and 1.3–1.9× on a Q4_0 file (claim to test).
3. **UD-Q4_K_M is a mix of eight formats.** By bytes read per token it is Q5_K 31 %, IQ4_XS 31 %, Q4_K 23 % and Q6_K 10 %. Unsloth's Q4_0 is 82.5 % Q4_0.
   - Against the single-card efficiencies measured on this rig, the Q4_0 file should generate ~15–20 % faster: 4.5–4.9 tok/s (claim to test).
   - Its quality cost is measurable: mean KLD 0.0265 for bartowski's Q4_0 against 0.0139 for Q4_K_M and 0.0158 for Q4_1.
4. **Tensor parallelism in mainline (`-sm tensor`) supports `qwen35`, at 128 AllReduces per token.** On Pascal the fast internal AllReduce is unavailable (it needs exactly 2 devices and compute capability ≥ 7.0), so reductions fall back to NCCL or to a host-staged F32 "butterfly".
   - The closest measured analog, 12 × P104-100 on PCIe 1.0 x1 with a 2-core host, decoded Llama-3-70B at 0.96 tok/s in tensor mode against 4.07–4.09 tok/s in layer mode.
   - A loss is expected here too (claim to test). Only a low-latency AllReduce that works on Pascal, which exists as unmerged fork patches, could change that.
5. **The 4 → 9 card slowdown for the 27B is not a per-hop cost.** It is 37.7 ms per token, ~13 %: the same percentage as on 7B-class models, where it was ~7 ms. A cost proportional to work fits the calibration's finding that lightly loaded cards clock down (~1595 MHz average on 9 cards). A fixed copy latency does not fit.
6. **Backend sampling (`-bs`, merged January 2026) removes most of llama-server's ~10–15 ms per token** of logits copy and CPU sampling over the 248,320-token vocabulary. With MTP the saving grows, because each verify step returns one logits row per drafted token plus one for the target's own token. It is unavailable in tensor mode, and it silently falls back to the CPU with grammars or tool-call constraints.
7. **No activation-sparsity or layer-skipping code exists in mainline llama.cpp or ik_llama.cpp.** GGUF stores every FFN weight with its quantization blocks along the input dimension, so TEAL-style input sparsity cannot skip whole blocks without a transposed copy of the weights and a new kernel. The byte ceiling is ~1.3–1.5× at 40–50 % FFN sparsity.
8. **There is no official smaller Qwen3.8.** Community layer-pruned cuts of ~20B and ~17B exist without standard benchmarks. Qwen3.6-35B-A3B (MoE) is the measured alternative: 4.7× faster on this rig.

### Lever summary

Expected effects are this note's estimates (claims to test) unless marked measured.

| Lever | Status (2026-10-22) | Expected effect on this rig | Effort |
|---|---|---|---|
| MTP, built into the GGUF or from the separate head file | Mainline since 2026-05-16 | 1.0–1.5× on UD-Q4_K_M; 1.3–1.9× on Q4_0 | Low |
| Q4_0 file instead of UD-Q4_K_M | Unsloth Q4_0, 16.06 GB | 4.5–4.9 tok/s (+15–20 %); quality lower | Low (download) |
| Q4_1 file | Unsloth Q4_1, 17.5 GB | 4.3–4.9 tok/s; quality ≈ Q4_K_M | Low (download) |
| Q8_0 file | Unsloth Q8_0, 29.05 GB, 6 cards | ~3.2–3.4 tok/s alone; flattest verify cost, so possibly the best base for MTP; best quality | Low–medium |
| Backend sampling `-bs` | Merged 2026-01-04 | Removes ~10–15 ms per served token (~4–6 %); more with MTP | Low |
| CUDA graphs on sm_61 (PR #27721) | Open PR | +2–5 % for the dense model | Low (build exists) |
| Pascal MMVQ patches (#25479 nwarps, #30334 fusion gate) | Open PR / open issue | +3–10 % | Medium (rebuild) |
| MTP draft-vocabulary trim (PR #29143) | Open PR | +5–15 % on top of MTP | Medium |
| DFlash2 / DSpark drafters | Mainline; drafters exist for Qwen3.8-27B | 0.8–1.5× (8-token verify blocks are costly here) | Low–medium |
| EAGLE-3 | Mainline; no Qwen3.8-27B head | n/a | — |
| Classic draft model (Qwen3.5-0.8B) | Works | Likely < 1× (host-RAM state checkpoints) | Low |
| n-gram drafting | Mainline | ≤ 1× on chat; possibly several × on edit-style output | Low |
| `-sm tensor` on 4 cards | Experimental, merged 2026-04-09 | 0.5–1.5×, most likely a loss | Medium; lock-up risk |
| ik_llama.cpp `-sm graph --max-gpu 2` | ik fork; host CPU support unclear | Unknown | High |
| Pascal low-latency AllReduce plus TP | Fork patches only | 0.6–1.8× | Very high |
| Activation sparsity (TEAL-like) | No implementation | Ceiling 1.3–1.5×; realistic gain uncertain on an issue-bound card | Very high |
| Card count, placement, clock mode | Configuration | 0 to +13 % | Low |
| KV type at long context | Configuration | Matters only beyond ~16–32k tokens | Low |
| Pruned 20B / 17B, or the MoE model | Community / official | ~1.4× / ~1.65× by bytes; 4.7× measured for the MoE; quality cost | Low–medium |

---

## 0. The model and where the 254 ms per token go

### Takeaway
Qwen3.8-27B is 64 layers:
- 48 Gated DeltaNet (GDN) linear-attention layers and 16 gated full-attention layers.
- Hidden size 5120, FFN width 17408, vocabulary 248,320, untied output head, one MTP layer.

Every token reads ~15.4 GB of weights on the GPUs. At the measured 3.94 tok/s that is ~61 GB/s, the same efficiency as Q4_K_M on one card. The FFN is two thirds of the bytes, the output head ~7 %. Fixed per-node overhead, GDN state traffic and KV reads are each a few percent at short context. So for the dense model the dominant term is how fast 5 SMs can unpack the weights. That is why the quant format, speculative decoding (which reuses each weight read for several tokens) and splitting matmuls across cards are the only large levers.

### Cited Findings
- `config.json`: `hidden_size` 5120, 64 layers, `full_attention_interval` 4, 24 query heads, 4 KV heads, head_dim 256.
  - GDN: 16 key heads and 48 value heads of dimension 128, conv kernel 4.
  - `intermediate_size` 17408, vocab 248,320, `tie_word_embeddings` false, `mtp_num_hidden_layers` 1.
  - Source: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) (repo created 2026-08-05). The Qwen3.5-27B and Qwen3.6-27B configs have the same shape (sources in section 6).
- GGUF headers of Unsloth and bartowski files, read with HTTP range requests (first ~12.6 MB of each file, no tensor data downloaded):
  - Metadata: `general.architecture = qwen35`, `qwen35.block_count = 65` (64 layers plus the MTP block), `nextn_predict_layers = 1`, `ssm.inner_size = 6144`, `ssm.group_count = 16`, `ssm.time_step_rank = 48`.
  - The GGUFs' default sampling metadata is top_k 20, top_p 0.95, temperature 1.0.
  - Sources: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (large upload 2026-08-19), [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF).
- llama.cpp keeps the token-embedding table in host memory ("very little benefit to offloading the input layer"), so its quant type does not affect GPU time — [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp). The calibration page uses the same convention ("file size minus the token-embedding table").
- With `-sm layer`, the output head and the MTP layer are placed on the last device — [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp). [Issue #27428](https://github.com/ggml-org/llama.cpp/issues/27428) (open, 2026-08-20) reports the last GPU using ~2 GB more VRAM with MTP.
- GDN recurrent state per sequence: 48 layers × 48 heads × 128 × 128 × 4 B = 144 MiB, plus ~5.6 MiB of conv state. A server log in [issue #23577](https://github.com/ggml-org/llama.cpp/issues/23577) shows 720 MiB of S-state for 5 slots of Qwen3.6-27B, which matches.
- On the rig, small MoE models cost ~10–11 µs per graph node per token with the GPU 95–98 % busy, so the cost is GPU-side ([small MoE benchmark](../../benchmarks/2026-10-22-small-moe-overhead.md)). Qwen3.6-35B-A3B has 3,727 nodes per token ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).

### Inferences
**Parameters read per token (my calc from the config):**

| Part | Parameters | Share |
|---|---|---|
| FFN, 64 × 3 × 5120 × 17408 | 17.11 B | 67 % |
| GDN projections (qkv, gate, out, alpha/beta), 48 layers | 5.56 B | 22 % |
| Full attention (q with gate, k, v, o), 16 layers | 1.68 B | 7 % |
| Output head, 248,320 × 5120 | 1.27 B | 5 % |
| Token embedding (host side, one row per token) | 1.27 B | — |
| MTP layer (read only when drafting) | ~0.42 B | — |

**Bytes read per token on the GPUs, by file (my calc from tensor types in the GGUF headers):**

| File | GPU bytes per token | Format mix (share of those bytes) | Output head |
|---|---|---|---|
| Unsloth UD-Q4_K_M (16.46 GB) | 15.39 GB | Q5_K 31.4 %, IQ4_XS 30.9 %, Q4_K 22.7 %, Q6_K 9.6 %, IQ4_NL 2.1 %, Q3_K 1.7 %, IQ3_S 1.0 %, Q8_0 0.5 % | Q6_K, 1.04 GB |
| Unsloth Q4_0 (16.06 GB) | 15.07 GB | Q4_0 82.5 %, Q6_K 6.9 % (head), Q5_K 6.9 % (all 48 `ssm_out`), Q4_1 3.0 % (8 of 64 `ffn_down`), F32 0.7 % | Q6_K, 1.04 GB |
| Unsloth Q4_1 (17.53 GB) | 16.45 GB | Q4_1 86.7 %, Q6_K 6.3 %, Q5_K 6.3 % | Q6_K |
| Unsloth Q8_0 (29.05 GB) | 27.23 GB | Q8_0 100 % | Q8_0, 1.35 GB |
| bartowski Q4_0 (16.34 GB) | 15.38 GB | Q4_0 81.7 %, Q6_K 9.5 %, Q8_0 5.2 %, Q4_1 2.9 % | Q6_K |
| bartowski IQ4_NL (17.43 GB) | 16.48 GB | IQ4_NL 65.1 %, Q6_K 25.0 %, Q8_0 8.5 % | Q6_K |

**Time budget of the current token (UD-Q4_K_M, 4 cards, llama-bench, my calc):**
- 15.39 GB / 0.254 s = 60.6 GB/s. The benchmark page's ~65 GB/s divides the whole file size instead. This matches single-card Q4_K_M (60 GB/s), so the multi-card run adds no visible inefficiency beyond the format.
- Output head: 1.04 GB, ~17 ms (6.8 %).
- Fixed per-node cost: an estimated ~2.1k compute nodes per token. The estimate is ~35 nodes per GDN layer and ~26 per attention layer, from the qwen35 graph code in [src/models/qwen35.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp). At the rig's ~10 µs per node that is an upper bound of ~21 ms (8 %), overlapping with weight streaming.
- GDN state: ~3 MiB read and ~3 MiB written per GDN layer per token, ≈ 0.3–0.6 GB per token, ≈ 2–4 % of the token time.
- KV cache: 16 layers × 4 KV heads × 256 × 2 × 2 B = 64 KiB per token of context (f16). At 8k context that is 0.5 GiB read per generated token (~3 %); at 32k, 2 GiB (~10–14 %).
- Layer boundaries: 3 hops on 4 cards. Measured ~1 ms each on 7B-class models, so ~3 ms.
- llama-server adds ~5 ms for the 0.99 MB logits copy at ~190–213 MB/s and ~10 ms of CPU sampling (section 5).

### Gaps
- No measured node count for the qwen35 decode graph. Running with `-ub 1` and verbose logging would print a decode-shaped `graph nodes` line.
- No per-kernel timing of the 27B on the rig. `nsys` is the tool, but it needs host RAM, which is scarce.

---

## 1. Speculative decoding

### Takeaway
- **MTP is the only mature option for Qwen3.8-27B with evidence on Pascal.**
  - On other hardware it gives 1.4–2.2×: Unsloth's figure, and community sweeps on 3090, 5090, DGX Spark and Strix Halo.
  - On 2 × P40 with tensor split it gives +92 % at `--spec-draft-n-max 4 --spec-draft-p-min 0.75`.
  - On the rig itself it gave +44 % on the MoE model.
- **What decides the gain here** is how much more a 2–5-token verify step costs than a 1-token step on a compute-starved card.
  - With the current K-quant mix, MTP may give anything from no gain to ~1.5×.
  - With a Q4_0 or Q8_0 file the verify step is much cheaper on Pascal, and 1.3–1.9× is plausible (claim to test).
  - n-max 1–2 is the likely optimum, and p-min gating around 0.75 helped on Pascal.
- **EAGLE-3:** no head exists for Qwen3.8-27B.
- **DFlash2 and DSpark** drafters exist for Qwen3.8-27B. They draft 7–8-token blocks, and verifying blocks that wide is exactly what a 5-SM card does worst.
- **Classic draft models and n-gram drafting on this hybrid target** make the server checkpoint the ~150 MB recurrent state into host RAM on every round, over the x1 links. Chat-style output will likely get slower. Long, highly repetitive edit-style output may still gain.

### Cited Findings
**MTP in mainline**
- PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) "llama + spec: MTP Support" (opened 2026-05-04, merged 2026-05-16).
  - It folded in the partial recurrent-state rollback from closed PR [#22400](https://github.com/ggml-org/llama.cpp/pull/22400).
  - Results in the PR thread:
    - DGX Spark, Qwen3.6-27B Q8_0, 9 prompts: wall time 201.07 s without speculation, 83.8 s with MTP n=3 (acceptance 0.72), 90.44 s with n=2 (0.83).
    - RTX 3090 + 3060, Qwen3.6-27B Q6_K: 22.97 → 42.45 tok/s at n=3 (76 %); prefill 1315 → 665 tok/s.
    - 3 × RTX 3060, Q4_K_M: 18.51 → 32.24.
    - RTX 5090, q4_0: 63.72 → 105.47 (0.797).
- Follow-ups (all merged, dates are merge dates):
  - [#23269](https://github.com/ggml-org/llama.cpp/pull/23269) (2026-05-19): default `--spec-draft-n-max` 3; `--spec-draft-p-min` re-enabled; spec types chainable.
  - [#23287](https://github.com/ggml-org/llama.cpp/pull/23287) (2026-05-20): draft sampling on the GPU, `--spec-draft-backend-sampling`, default on.
  - [#23433](https://github.com/ggml-org/llama.cpp/pull/23433) (2026-05-21): logits skipped in the MTP catch-up pass.
  - [#26177](https://github.com/ggml-org/llama.cpp/pull/26177) (2026-07-27): `--fit` accounts for the NextN block.
  - [#26296](https://github.com/ggml-org/llama.cpp/pull/26296) (2026-07-31): MTP tensors loaded only when used.
  - [#27005](https://github.com/ggml-org/llama.cpp/pull/27005) (2026-08-13): MTP type auto-detected from a draft GGUF.
  - [#28549](https://github.com/ggml-org/llama.cpp/pull/28549) (2026-09-16): CUDA graphs for the MTP draft, +4–5 % on an RTX 5090. Not in the rig's 2026-09-04 build.
  - [#27694](https://github.com/ggml-org/llama.cpp/pull/27694) (2026-10-02): `--spec-draft-sampling probabilistic` (rejection sampling); +0.93 % on DGX Spark.
  - [#30459](https://github.com/ggml-org/llama.cpp/pull/30459) (2026-10-14): `--spec-draft-n-ubatch` to shrink the draft compute buffer.
- Drafting is autoregressive with the single MTP layer. One draft decode runs per drafted token, fed the previous step's hidden row and the sampled token. Drafting stops at `n_max`, or when the top candidate's probability falls below `p_min`.
  - Hidden rows pass through host memory, ~20 KB per token at hidden size 5120.
  - Source: `common_speculative_impl_draft_mtp` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp). ik_llama.cpp [PR #2369](https://github.com/ikawrakow/ik_llama.cpp/pull/2369) describes the same: "Multi-token drafting reuses the one MTP layer on each step".
- Two ways to load the head ([common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp), master):
  - Without `-md`, an MTP draft context is created against the target model and uses its NextN block and output head.
  - With `-md`, a separate MTP GGUF is loaded as the draft model. `--spec-draft-device`/`-devd` and `-ngld` then place it, for example on an otherwise idle card.
  - A community README for Qwen3.8-27B uses only `--spec-type draft-mtp --spec-draft-n-max 2` on the main UD-Q4_K_XL file, which "contains `qwen35.nextn_predict_layers=1` and the MTP tensors" — [sudoingX/qwen38-mtp, sweeps/multi-gpu.md](https://github.com/sudoingX/qwen38-mtp/blob/master/sweeps/multi-gpu.md) (S, updated 2026-09-16).
- Cost of a draft step:
  - ik_llama.cpp [PR #1809](https://github.com/ikawrakow/ik_llama.cpp/pull/1809) (merged 2026-05-16): the Qwen3.5/3.6 MTP layer reuses the 248k-row output tensor, "so drafting is dominated by its matrix multiply".
    - It added `-mtprot` to use a requantized copy (~645 MiB for 27B), +5–10 % generation with no measurable acceptance loss.
    - One user measured 54.36 → 64.09 tok/s on Qwen3.6-27B (5060 Ti + 3090).
  - Mainline equivalent: open [PR #29143](https://github.com/ggml-org/llama.cpp/pull/29143) (2026-09-19), a d2t draft-vocabulary trim for qwen35 MTP.
    - A 32K-entry head is ~87 % smaller.
    - Q8 target on an RTX PRO 6000: 43 → 97.9 tok/s at N=3 and 104.5 at N=5.
- VRAM:
  - Unsloth says MTP needs ~1–2 GB of headroom — [unsloth.ai/docs/models/mtp](https://unsloth.ai/docs/models/mtp).
  - RTX 5090, Qwen3.8-27B Q4_K_M: +~680 MB at n=1, then ~+150 MB per extra draft level, which is one GDN state snapshot — [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings) (S, 2026-08).
  - On 8 × 3060 Ti over PCIe 2.0 x1, the 2209 MiB MTP compute buffer failed to allocate on the last card, and the server silently "retried without pipeline parallelism"; `-ts 9,9,9,9,9,9,9,3` fixed it — [#27428](https://github.com/ggml-org/llama.cpp/issues/27428).
  - On this rig Qwen3.6-35B-A3B's MTP cost 2.2 GB of extra GPU memory ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).
- Multi-GPU issues:
  - [#27428](https://github.com/ggml-org/llama.cpp/issues/27428) (open): draft-mtp roughly halves prompt processing on a multi-GPU layer split, while a single GPU loses only 3–22 %.
  - [#27122](https://github.com/ggml-org/llama.cpp/issues/27122) (open, 2026-08-15): MTP with Qwen3.8-27B in `-sm tensor` causes CUDA lock-ups, including hard power-offs on 4 × A4000. Workaround: `LLAMA_GRAPH_REUSE_DISABLE=1`.

**MTP results for dense 27B models (tok/s, baseline → MTP)**

| Hardware | Model, split | Flags | Result | Source |
|---|---|---|---|---|
| 2 × Tesla P40 (sm_61), tensor split, no P2P | Qwen3.8-27B UD-Q5_K_XL, 131K, q4_0 KV | n-max 2 / 4 / 6 at p-min 0.75; n-max 6 at p-min 0.60 | 11.7 → 21.1 / **22.6** / 21.1 / 20.1; acceptance 0.73–0.94 / 0.68–0.87 / 0.63–0.81 / 0.46–0.74; prefill 354 → 295 | [sudoingX/qwen38-mtp multi-gpu.md](https://github.com/sudoingX/qwen38-mtp/blob/master/sweeps/multi-gpu.md) (S, Sep 2026) |
| same | UD-Q4_K_XL, 262K | n-max 4, p-min 0.75 | 13.2 → 23.4 | same |
| RTX 2070 + 5060 Ti, PCIe 2.0 x1 riser, layer split | Qwen3.8-27B UD-Q4_K_XL | n-max 1 / 2 / 3 / 4 | 19.7 → 30.6 / 37.1 / **37.8** / 35.4 | same |
| same, tensor split 5:2 | same | n-max 2 / 3 / 4 | 25.5 → **37.7** / 36.1 / 32.5 | same |
| 2 × V100 NVLink, layer split | Qwen3.8-27B Q4_K_M | n-max 4 | 35.7 → 59.1 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) (S) |
| CMP 170HX, PCIe 1.0 x4 | Qwen3.8-27B Q4_K_M | n-max 2 | 33.0 → 46.7, acceptance 0.53–0.94 | same |
| RTX 3090 | Qwen3.8-27B UD-Q4_K_XL, greedy | n-max 2 / 3 / 5 | 41.55 → **66.39** / 63.29 / 54.89; at n2: code +92 %, prose +48.6 % | [hackmd write-up](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090) (S, 2026-08-29) |
| RTX 5090 | Qwen3.8-27B Q4_K_M | n-max 1 / 2 / 3 / 4 / 5 | 73.6 → 104.8 (acc. 0.860) / 125.5 (0.766) / **133.6** (0.674) / 119.5 (0.592) / 108.8 (0.520) | [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings) (S, 2026-08) |
| Strix Halo, Vulkan | Qwen3.8-27B Q6_K | n-max 2 / 5 | 9.59 → 18.01 / 15.16 | [sleepingrobots](https://sleepingrobots.com/dreams/qwen38-mtp-strix-halo/) (S, 2026-08-15) |
| 8 × 3060 Ti, PCIe 2.0 x1, layer split | Qwen3.8-27B UD-Q4_K_XL | fork's adaptive MTP, n-max 12, ungated / p-min 0.75 | 22.93 (acc. 0.597) / 29.82 (0.708); no baseline given | [#27428](https://github.com/ggml-org/llama.cpp/issues/27428) (P) |
| 2 × 3090 (ik_llama.cpp), layer / graph mode | Qwen3.5-27B Q8_0 | draft 1 / 2 | layer 24.8 → 32.2 (83.4 %) / 25.9; graph 37.5 → 43.2 / 28.0 | ik [PR #1698](https://github.com/ikawrakow/ik_llama.cpp/pull/1698) (P, 2026-04-28) |

- The sweep author's remark on the P40 run: "The verification batch on sm_61 is expensive enough that the deeper draft stops paying." On p-min: "On this class of card the gate rescues the deep draft". p-min 0.60 hurt — [multi-gpu.md](https://github.com/sudoingX/qwen38-mtp/blob/master/sweeps/multi-gpu.md).
  - The same file reports p-min as a large net loss on a 2 × 5060 Ti box (71.3 → 46.2 tok/s at n-max 4). The setting does not generalise across hardware.
  - It also finds that the n-max optimum moves with topology: lower for layer split and for slow links.
- Unsloth: "`--spec-draft-n-max 2` is the best starting point", ~1.4–2.2× — [unsloth.ai/docs/models/mtp](https://unsloth.ai/docs/models/mtp).

**Verify-step cost on Pascal**
- Mainline sends quantized matmuls with ≤ 8 tokens to MMVQ (`#define MMVQ_MAX_BATCH_SIZE 8`), with no Pascal special case — [mmvq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cuh).
  - The generic launch table uses 4 warps for 1–4 columns and 2 warps for 5–8 — [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu).
  - A code comment notes that MMVQ redoes K-quant decoding per column.
- Tesla P40, Llama 8B: tokens/s at physical batch b (pp512 with `-ub b`). These are the "before" columns of a maintainer's benchmark posted 2026-04-05 in open [PR #20831](https://github.com/ggml-org/llama.cpp/pull/20831), which targets RDNA, so the P40 numbers are a neutral baseline:

  | Quant | b=1 | b=2 | b=3 | b=4 | b=5 | b=8 | Cost of a b-token step vs 1 token (b=2 / 3 / 4 / 8, my calc) |
  |---|---|---|---|---|---|---|---|
  | Q4_0 | 58.82 | 115.28 | 158.52 | 162.46 | 197.04 | 217.47 | 1.02 / 1.11 / 1.45 / 2.16 |
  | Q4_K_S | 58.89 | 101.22 | 119.67 | 129.77 | 140.95 | 137.53 | 1.16 / 1.48 / 1.82 / 3.43 |
  | Q6_K | 39.26 | 71.42 | 95.51 | 107.00 | 121.07 | 141.99 | 1.10 / 1.23 / 1.47 / 2.21 |
  | Q8_0 | 34.66 | 69.50 | 100.15 | 107.86 | 150.31 | 165.57 | 1.00 / 1.04 / 1.29 / 1.67 |
  | IQ4_XS | 60.13 | 115.81 | 150.61 | 138.84 | 180.50 | 183.97 | 1.04 / 1.20 / 1.73 / 2.61 |
  | IQ4_NL | 57.45 | 113.08 | 156.59 | 161.61 | 192.95 | 210.02 | 1.02 / 1.10 / 1.42 / 2.19 |

- L4 (Ada), Qwen3.8-27B: MMVQ verify adds ~9.3 ms per extra token on a 66.9 ms step. A local patch limiting MMVQ to 2 columns (wider batches going to MMQ) gave +16 % with MTP — [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (S, Aug 2026). The L4's MMQ uses int8 tensor cores; Pascal's MMQ uses dp4a, so this does not transfer directly.

**Recurrent state on rejection (Gated DeltaNet)**
- For `draft-mtp`, `draft-eagle3`, `draft-dflash` and `draft-dspark`, the context keeps `n_rs_seq = n_max` on-device state snapshots and rolls back to the last accepted token without replay.
  - `need_n_rs_seq()` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h); `llm_arch_supports_rs_rollback` includes QWEN35 ([src/llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp)).
- For `draft-simple` and n-gram types used alone, `n_rs_seq` is 0. In that case, and whenever a draft is longer than `n_rs_seq`, the server takes a "speculative checkpoint" of the sequence's recurrent state before verifying, restores it on rejection, and re-runs the accepted tokens.
  - Source: [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp), master.
  - On-device checkpoints were disabled on 2026-06-04 ([#24108](https://github.com/ggml-org/llama.cpp/pull/24108)), so the checkpoint lives in host memory.
  - The checkpointing came from [#19493](https://github.com/ggml-org/llama.cpp/pull/19493) (merged 2026-04-19).

**EAGLE-3**
- [#18039](https://github.com/ggml-org/llama.cpp/pull/18039) (merged 2026-06-12) added `draft-eagle3`: a one-layer drafter fed hidden states from three target layers, converted with `convert_hf_to_gguf.py --target-model-dir`. It crashes with `-sm tensor`.
- [#24593](https://github.com/ggml-org/llama.cpp/pull/24593) (merged 2026-06-19) added Qwen3.5/3.6 hybrid targets.
  - DGX Spark, Qwen3.6-27B Q4_K_M with Ex0bit's PRISM EAGLE3 head: 12.56 → 22.31 tok/s, acceptance 0.50, n-max 3.
  - Users in the thread reported slowdowns on V100 (15–17 against a 27 baseline).
- Heads:
  - [Ex0bit/Qwen3.6-27B-PRISM-EAGLE3](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3) (2026-05-19): about 0.6 B parameters; a "compressed" variant with a 32K draft vocabulary is 1.10 GB in bf16.
  - VirVen heads exist for Qwen3.5-27B and Qwen3.6-27B.
  - **No EAGLE-3 head for Qwen3.8-27B was found.** An EAGLE-4 head exists for it (EAGLE-Corp), but the llama.cpp EAGLE-4 PR (#30930) was closed on 2026-10-21.

**DFlash, DFlash2, DSpark**
- What they are:
  - DFlash: a lightweight block-diffusion drafter conditioned on target hidden states; it drafts a whole block in one forward pass — [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) (2026-02-05, ICML 2026).
  - DFlash2 adds a local convolution and a candidate selector — [#27342](https://github.com/ggml-org/llama.cpp/pull/27342) / [#27816](https://github.com/ggml-org/llama.cpp/pull/27816) (merged 2026-08-27).
  - DSpark: a parallel drafting backbone plus a small semi-autoregressive head with confidence-scheduled verify length — [arXiv 2607.05147](https://arxiv.org/abs/2607.05147) (2026-07-06).
- llama.cpp:
  - `draft-dflash` [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) (merged 2026-06-28). DGX Spark, Qwen3.6-27B Q4_K_M: 12.57 → 33.76 tok/s.
    - On hybrid targets verification writes state for the whole block, so a rejection can cost an extra target pass.
  - `draft-dspark` [#25173](https://github.com/ggml-org/llama.cpp/pull/25173) (merged 2026-07-28). [#26275](https://github.com/ggml-org/llama.cpp/pull/26275) (merged 2026-08-17) added speculators-format drafts, including a Qwen3.8-27B DSpark at 1.93× with acceptance 0.27; the hardware is not stated.
  - DFlash2 with `-sm tensor` asserts: [#28777](https://github.com/ggml-org/llama.cpp/issues/28777) open; fix [#27858](https://github.com/ggml-org/llama.cpp/pull/27858) open.
- Drafters for Qwen3.8-27B:
  - [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (3.85 GB safetensors, 2026-08-15).
  - [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) ships `dflash-Qwen3.8-27B-Q4_0.gguf` (1.09 GB). Its header: 5 layers, hidden 5120, FFN 17408, `block_size` 8, `selector_top_k` 16, no output head of its own (a low-rank selector instead).
  - The same repo has `mtp-Qwen3.8-27B-Q4_0.gguf` (1.68 GB): Q4_0 NextN block, Q4_0 head and Q4_0 embeddings.
  - [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) (3.7 GB, 2026-08-14) and [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) (4.0 GB, 2026-09-16).
- Comparisons:
  - RTX 3090, Qwen3.8-27B: DFlash2 n4 +51.9 % and n7 +22.6 %, against MTP n2 +59.8%; with DFlash2 n7, prose −11 % — [hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090) (S).
  - 2 × R9700, llama.cpp: baseline 30.5, DFlash n3 53, MTP n3 56 tok/s; accepted length ~3 in llama.cpp against ~5.5 in vLLM — [z-lab/dflash #170](https://github.com/z-lab/dflash/issues/170) (S).

**Classic draft models**
- Draft and target must match vocabulary type and BOS/EOS, sizes may differ by ≤ 128, and token text must match from id 5 on; otherwise loading fails — `common_speculative_are_compatible` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp).
- The only small official models with the 248,320-token vocabulary are Qwen3.5-0.8B, 2B, 4B and 9B; there is no Qwen3.6 or Qwen3.8 below 27B (section 6).
  - Unsloth Qwen3.5-0.8B Q4_0 is 0.507 GB — [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF).
- Results:
  - DGX Spark, Qwen3.6-27B + 0.8B at n=16: 81.39 s wall against 201 s without speculation and 83.8 s with MTP ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673)).
  - RTX 3090, Qwen3.8-27B + 0.8B: −29.8 % ([hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090), S).

**n-gram drafting** ([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md); defaults from common.h)
- The methods:
  - `ngram-simple`: drafts the continuation of the most recent earlier match of the last n tokens (defaults n 12, m 48).
  - `ngram-map-k`: drafts only after the key has recurred `min-hits` times.
  - `ngram-map-k4v`: experimental; keeps up to four continuations per key.
  - `ngram-mod`: a rolling-hash table of ~16 MB shared across slots, defaults n-match 24 / n-min 48 / n-max 64 ([#19164](https://github.com/ggml-org/llama.cpp/pull/19164), merged 2026-01-30).
  - `ngram-cache`: n-gram statistics, optionally loaded from file.
- Flags: `--spec-ngram-mod-n-match/-n-min/-n-max`, `--spec-ngram-simple-size-n/-size-m`, etc. ([common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)).
- Results:
  - gpt-oss-120b translating code comments: 181 → 446 tok/s ([#18471](https://github.com/ggml-org/llama.cpp/pull/18471)).
  - Line-ending mismatch (CRLF against LF) changed acceptance from 0.11 to 0.83 ([#19164](https://github.com/ggml-org/llama.cpp/pull/19164)).
  - Qwen3.8 with `ngram-mod` on cold chat prompts: 0.97× ([KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings), S).

**Combinations and options**
- `--spec-type a,b,c` chains methods. The priority is fixed (n-gram types first, then draft-simple, eagle3, mtp, dflash, dspark); the first one that returns a draft wins the round ([#23269](https://github.com/ggml-org/llama.cpp/pull/23269); common/speculative.cpp).
  - MTP plus `ngram-mod` is reported working.
  - Tree or multi-branch verification is not in llama-server: [#30931](https://github.com/ggml-org/llama.cpp/pull/30931) "[Draft] Add tree spec-dec support" opened 2026-10-21.
- `--spec-draft-p-min` (default 0) stops drafting at the first position whose top probability is below the threshold. `--spec-draft-backend-sampling` (default on) samples draft tokens on the GPU with top-k 10, so only 10 logits per draft step come back to the host; it falls back to the CPU in tensor mode ([#23287](https://github.com/ggml-org/llama.cpp/pull/23287)).
- llama.cpp has no asynchronous drafting: drafting and verification alternate on the same thread, so idle cards cannot run the drafter in parallel.

### Inferences
**A model of the gain (my calc).** Notation:
- `T1`: one target step for one token.
- `d`: one MTP draft step.
- `r(n)`: the cost of an n-token target step relative to `T1`.
- `a`: llama.cpp's acceptance rate (accepted ÷ drafted).
- `k`: drafted tokens per round.

A round yields `1 + a·k` tokens and costs `k·d + d_catchup + r(k+1)·T1`. The catch-up pass runs the MTP layer over the accepted tokens without computing logits, so it reads only the NextN block.

- **`d` on this rig** (bytes per step at ~60 GB/s, plus ~60 graph nodes):
  - Built-in path on UD-Q4_K_M: NextN 0.35 GB + Q6_K head 1.04 GB ≈ 1.39 GB ≈ 23 ms ≈ 0.09·T1.
  - Unsloth separate file: NextN 0.27 GB (Q4_K/Q6_K) + Q3_K head 0.55 GB ≈ 0.81 GB ≈ 13–15 ms ≈ 0.07·T1 for a Q4_0 target (T1 ≈ 212 ms).
  - ggml-org Q4_0 MTP file: ≈ 0.96 GB.
  - Catch-up ≈ 0.02·T1.
- **`r(n)` on this rig.** The P106-090 has 5 SMs at ~1.7 GHz under its power cap for 173 GB/s, against a P40's 30 SMs at ~1.5 GHz for 347 GB/s: ~0.38× the compute per byte (my calc). Its verify cost must therefore lie above the P40 row.
  - Scaling the P40's extra cost (r − 1) by 1.5–2.5× gives:

  | Base | r(2) | r(3) | r(4) |
  |---|---|---|---|
  | UD-Q4_K_M (byte-weighted P40 mix: 1.11 / 1.34 / 1.75) | 1.17–1.28 | 1.5–1.85 | 2.1–2.9 |
  | Q4_0 | 1.03–1.05, up to ~1.5 if fully issue-bound | 1.17–1.28, up to ~2.0 | 1.7–2.1, up to ~2.5 |
  | Q8_0 | ~1.0–1.1 | ~1.1–1.3 | ~1.4–1.7 |

- **On-rig anchor.** Qwen3.6-35B-A3B Q4_0 with 3 drafts made 1 + 0.77·3 = 3.31 tokens per round at 22.34 tok/s, i.e. 148 ms per round, against 64.5 ms per plain step: a round cost 2.3·T1.
  - Taking 4 MTP passes at ~0.1·T1 each, the 4-token verify cost ~1.7–1.9·T1.
  - An MoE verify also reads more distinct experts per step than a dense one, so a dense Q4_0 r(4) below ~1.9 is plausible.
- **Resulting speed-ups** (acceptance taken from the Qwen3.8-27B 5090 sweep, 0.86 / 0.77 / 0.67 for k = 1 / 2 / 3; claims to test):

  | Target | k = 1 | k = 2 | k = 3 |
  |---|---|---|---|
  | UD-Q4_K_M, built-in head | 1.34–1.45× | 1.24–1.49× | 0.94–1.26× |
  | Q4_0, separate head, optimistic r | ~1.65× | ~1.8–1.9× | ~1.3–1.6× |
  | Q4_0, separate head, fully issue-bound r | ~1.2× | ~1.2× | ~1.1× |

  - Combined with Q4_0's own +15–20 %, the plausible range is ~5.5–9 tok/s, against 3.94 today.
  - On Q8_0 (T1 ≈ 293 ms, ~3.4 tok/s alone) the flatter verify cost could give ~1.6–2× at k = 3, i.e. ~5.5–6.5 tok/s with the best quality. This depends on Q8_0 keeping its P40-like batch scaling on 5 SMs.
- **llama-server overhead grows with k unless `-bs` is used.** Without backend sampling each verify step brings back k+1 logits rows of ~1 MB (~5 ms each over x1) and samples each on the CPU (~10 ms each). That is ~45–60 ms per round at k = 3, 10–15 % of the round (section 5).
- **Placement.** The separate head can go on a fifth, otherwise idle card (`-md … -devd CUDA4`). That keeps the NextN block, the MTP compute buffer (1.2–2.2 GB on other rigs) and the draft KV off the four full target cards. Hidden rows pass through host memory either way, so this adds no new transfer.
- **Snapshots.** `n_rs_seq = n_max` snapshots of ~150 MB each are spread over the cards holding GDN layers: ~37 MB per card per level on 4 cards.
- **Prefill.** Expect pp512 to fall well below 50.6 tok/s with MTP on a multi-card layer split (#27428). For long prompts this cost must be weighed against the decode gain.
- **DFlash2 and DSpark.** A draft pass reads ~1.1 GB of drafter plus the target head for 8 positions, and the verify step is up to 8–9 tokens wide (MMVQ up to 8, then dp4a MMQ). If an 8-token step costs 2.5–4·T1 on this card (P40 Q4_K_S already 3.43× at b=8) and llama.cpp accepts ~3 tokens per block (z-lab #170), the result is roughly break-even. Worth one run with the 1.09 GB ggml-org DFlash file, not a primary path.
- **Qwen3.5-0.8B as a draft model.** Draft cost is ~15–20 ms per token (0.5 GB plus ~1k nodes at ~10 µs). Acceptance is probably below MTP's. Every round also copies ~150 MB of recurrent state to host RAM over x1 links: ~0.2 s if the four cards copy in parallel, ~0.7 s if serially. That is 1–3·T1 of overhead per round, so a slowdown is likely.
- **n-gram drafting on this hybrid** has the same per-round checkpoint cost, so it only pays when rounds accept many tokens. For edit-style output (rewriting a file with small changes) `ngram-mod` drafts 48–64 tokens.
  - A 64-token verify runs in MMQ at close to prompt speed: ~1.3 s at the measured 50 tok/s pp512.
  - A fully accepted round could reach ~20–40 tok/s on such output, against 3.94.
  - Chaining `draft-mtp,ngram-mod` with both capped at 8 (`--spec-draft-n-max 8 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 8`; the default n-min of 48 must come down too) keeps every draft within the on-GPU snapshot path. The cost is 8 snapshots (~1.2 GB spread over the cards).
- **EAGLE-3:** only a Qwen3.6-27B head exists, and its compressed 32K-vocabulary variant would make draft steps cheap. On the Qwen3.8 target its acceptance is unknown and probably low, because the head was trained on another checkpoint's hidden states. Not a priority.

### Gaps
- No speculative-decoding measurement on any 5-SM Pascal card, P104/P106 included. No Pascal pp at physical batch 1–8 for Q4_1, Q5_K or the UD mixes.
- Whether Unsloth's Q3_K-headed separate MTP file lowers acceptance compared with the built-in Q6_K head.
- Hardware behind #26275's Qwen3.8-27B DSpark 1.93×.
- Whether llama-server's speculative checkpoint copies are pinned or pageable, and serial or parallel across cards. This sets the n-gram and draft-model round cost on this rig.
- No measurement of MTP plus n-gram chains on a dense 27B that excludes warm-cache repetition.

---

## 2. Tensor parallelism for the dense model

### Takeaway
Tensor parallelism is the only way to read weights on several cards at once for a single token, so for a dense model it is the only scheme that adds up the cards' bandwidth.
- **Mainline `-sm tensor` supports `qwen35`.** GDN heads and recurrent state are split per key-head group, and there are 2 AllReduces per layer: 128 per token, each 20 KB per card in F32.
- **On Pascal the reduction is slow.** The fast internal AllReduce requires exactly 2 devices and compute capability ≥ 7.0. Pascal therefore falls back to NCCL, which must be built for sm_61 and runs through host shared memory without P2P, or to a host-staged F32 butterfly.
- **Measured analogs on slow links:**
  - 12 × P104-100 on PCIe 1.0 x1 with a 2-core host: tensor mode decoded 4.3× slower than layer mode.
  - 2 × 3090 on PCIe 2.0 x1 without P2P, the same Qwen3.8-27B UD-Q4_K_M file: break-even (41.1 against 40.6 tok/s), with prefill collapsing ~10×.
- **Fit:** the model needs at least 3–4 cards at 4 bits, and its 4 KV heads and 16 GDN key heads divide evenly only by 2 and 4, so TP-4 is the natural configuration.
- **Expected result:** a loss for decode with the stock butterfly (claim to test). A gain is plausible only with a low-latency AllReduce that works on Pascal plus CUDA graphs, which is fork-level engineering.
- **Hybrid TP × PP:** not possible in mainline. ik_llama.cpp's `--max-gpu N` does it, but ik's support for this host CPU is unverified.

### Cited Findings
**Mainline implementation**
- [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) "ggml: backend-agnostic tensor parallelism (experimental)" (opened 2026-02-05, merged 2026-04-09).
  - A "meta" backend wraps N backends. Weights are split along output rows (QKV, gate, up) or input columns (attention or SSM output, FFN down), and each partial result is AllReduced.
  - The author: TP "works comparatively better for slow GPUs with fast interconnect"; "the key factor is how long the base runtime is vs. the synchronization overhead".
  - Device cap 16 (`GGML_BACKEND_META_MAX_DEVICES`).
- Reduction paths on CUDA ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu), [allreduce.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu), [ggml-backend-meta.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-meta.cpp), master):
  - The order is NCCL, then internal, then a fallback; `GGML_CUDA_ALLREDUCE=nccl|internal|none` forces one.
  - NCCL reduces small tensors in F32 (below 262,144 elements for ≥ 4 GPUs, a heuristic "tuned on RTX 4090s connected via 16x PCIe 4.0") and larger ones in BF16. A decode-step AllReduce of 5120 elements is therefore F32.
  - The internal AllReduce ([PR #22299](https://github.com/ggml-org/llama.cpp/pull/22299), merged 2026-05-10) polls pinned host memory with `__nanosleep` (sm_70+). Its init returns null unless `n_devices == 2` and cc ≥ 700, logging "internal AllReduce init failed (n_devices != 2?)", which also appears on 2 × P100 because of the cc gate ([#29466](https://github.com/ggml-org/llama.cpp/issues/29466)).
  - The fallback butterfly works in F32 by recursive doubling. Each step copies the whole partial tensor to a partner (`cudaMemcpyPeerAsync`, which the driver stages through host memory without peer access) and runs a one-node ADD graph there. Extra GPUs beyond a power of two are folded in first and copied back last.
- `qwen35`, `qwen35moe` and `qwen3next` are not on the tensor-mode deny list (`llm_arch_supports_sm_tensor` in [src/llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp)).
  - Qwen3.5 support arrived inside #19378 (commit 2026-03-25). [PR #28965](https://github.com/ggml-org/llama.cpp/pull/28965) (merged 2026-09-16) fixed fused-QKV qwen35 splitting and was tested on 4 × P100.
  - GDN weights and the conv/SSM states are split per key-head group; `ssm_out` is split on its input dimension, so each card keeps only its own heads' state and no recurrent state crosses cards ([src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)).
  - The output head is split by vocabulary, and logits are gathered from every card.
- Limits:
  - Flash attention is required.
  - `--fit` is not implemented for tensor mode, so `-c` must be set by hand.
  - Backend sampling falls back to the CPU ("backend sampling not supported with SPLIT_MODE_TENSOR") — [src/llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp).
  - Quantized KV is supported since [PR #23792](https://github.com/ggml-org/llama.cpp/pull/23792) (merged 2026-06-01); [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) still says otherwise.
  - Compute buffers are over-allocated ~N× ([#30317](https://github.com/ggml-org/llama.cpp/issues/30317), open 2026-10-11; on 4 × 5060 Ti the proposed fix took 1736 → 874 MiB per device).
  - A no-P2P deadlock report: [#27110](https://github.com/ggml-org/llama.cpp/issues/27110), open.
- CUDA graphs: each subgraph between AllReduces is its own graph, and graphs are disabled below Volta in master ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)). Graphs matter a lot in tensor mode:
  - 8 × V100, qwen35 27B F16, tg128: layer 14.98; tensor 20.00 without graphs, 40.96 with ([PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749), merged 2026-07-16).
  - 4 × P100, Qwen3.6-35B-A3B Q8_0: tensor 39.58 without graphs, 70.62 with, against layer 61.44 ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721), open).
- NCCL:
  - Without P2P, NCCL uses the SHM transport through host memory — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
  - NCCL's makefile builds sm_60/61 only with CUDA 12.x and drops archs below sm_75 for CUDA ≥ 13 — [NCCL makefiles/common.mk](https://github.com/NVIDIA/nccl/blob/master/makefiles/common.mk). So NCCL for this rig must be built against CUDA 12.

**ik_llama.cpp `-sm graph`**
- QWEN35 and QWEN35MOE are supported, recurrent layers included ([ik PR #1388](https://github.com/ikawrakow/ik_llama.cpp/pull/1388), merged 2026-03-10); QWEN3NEXT is disabled.
- `--max-gpu N` ([ik PR #1051](https://github.com/ikawrakow/ik_llama.cpp/pull/1051)): "all GPUs get used, it is just that only max-gpu GPUs are being used in any given model layer". With an even split it groups contiguous layer blocks per GPU group: effectively TP-N × PP.
- `-grt q8_0|bf16|f16|f32` sets the reduce type ([ik PR #1087](https://github.com/ikawrakow/ik_llama.cpp/pull/1087), [#1154](https://github.com/ikawrakow/ik_llama.cpp/pull/1154)), but the cast only applies above 32 tokens. Decode reduces in F32.
- Without NCCL or P2P, small reductions gather every card's vector to one GPU, add, and broadcast; larger ones use a ring (ik `ggml/src/ggml-cuda/reduce.cu`).
- CUDA graphs are disabled below Ampere in ik.
- ik's README says only Turing and newer are fully supported, and that the CPU side needs AVX2 — [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). This host has no AVX at all.
- Results:
  - 2 × 3090, Qwen3.6-27B IQ4_XS: layer 51 tok/s, graph 72 ([ik issue #1730](https://github.com/ikawrakow/ik_llama.cpp/issues/1730), 2026-05).
  - 4 × 3090, LLaMA-3-70B Q4_0: graph pp 1803 against mainline tensor 288, tg ~50 against ~48 ([ik discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247), 2026-02).
  - No Pascal or slow-link ik numbers found.

**Measurements on slow links, Pascal and dense ~27B**

| Rig | Link | Model | Layer | Tensor | Source |
|---|---|---|---|---|---|
| 12 × P104-100 (sm_61), 2-core Pentium, no NCCL, butterfly fallback | PCIe 1.0 x1 | Llama-3-70B Q4_K_M, f16 KV | tg 4.07–4.09, pp 37–38 | tg **0.96**, pp **1.18** | [TurboPrefill-VLM-Validation](https://github.com/sergey-automation/TurboPrefill-VLM-Validation) (P, raw logs, runs 2026-06-27; fork build b9100) |
| 2 × RTX 3090, no P2P | PCIe 2.0 x1 | Qwen3.8-27B UD-Q4_K_M | tg 40.56, pp4096 1616 | internal AllReduce: tg 41.10, pp 140; NCCL: tg 31.25, pp 169 | [note.com/agentworkflowlab](https://note.com/agentworkflowlab/n/nc4683f6203e9) (S, 2026-09-03, build 10775) |
| RTX 2070 + 5060 Ti | one card on PCIe 2.0 x1 riser | Qwen3.8-27B UD-Q4_K_XL | tg 19.7 | tg 25.5 (5:2 split) | [sudoingX multi-gpu.md](https://github.com/sudoingX/qwen38-mtp/blob/master/sweeps/multi-gpu.md) (S) |
| 3 × CMP 50HX (TU102), P2P working, NCCL | PCIe 2.0 x8 | Qwen3.8-27B UD-Q4_K_XL | tg 25.16 (short), 7.92 at 250k | tg 39.27, 15.61 at 250k; prefill ~equal; aggregate with 5 clients 36.2 against 54.4 for layer | [FASTCHIP/cmp50hx-llama-research](https://github.com/FASTCHIP/cmp50hx-llama-research) (P, runs 2026-09-21..25) |
| 2 × P100 (sm_60), different sockets | PCIe 3.0 | Qwen3.8-27B Q4_K_XL | — | tg 19.40 → 22.70 with patches; with the internal AllReduce enabled on Pascal (fork patch 0003) tg128 22.3 → 24.5, pp8192 323.6 → 378.7 | [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp) (P/S, 2026-09/10) |
| 8 × V100 | PCIe 3.0 x8 | qwen35 27B F16 | tg 14.98 | 20.00 (no graphs), 40.96 (graphs) | [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749) (P) |

- key-sh measured the stock fallback on 2 × P100 doing 128 exchanges per card per forward pass ("64 layers x 2 exchanges per layer"). A 2048-token prefill moved 20.5 GB over PCIe, 23.1 % of GPU time.
  - Its patch 0006 extends the internal AllReduce to more than 2 devices, but is "correctness-tested only" (virtual devices) — [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp).
- Small-copy latency: on 2 × 3090 with PCIe 4.0 x16, GPU-to-GPU latency with P2P disabled was 18.4–32.3 µs, against ~1.1 µs with P2P ([ik discussion #1380](https://github.com/ikawrakow/ik_llama.cpp/discussions/1380)). No PCIe 1.0 x1 measurement was found.
- Mainline has no TP × PP hybrid. All selected devices join one meta device, and `-dev` or `CUDA_VISIBLE_DEVICES` can only shrink the set.

### Inferences
**AllReduce traffic per generated token for Qwen3.8-27B (my calc).** 128 AllReduces, each over a 5120-element vector, 20,480 B in F32 (10,240 B in BF16).

| Scheme | Per card per AllReduce | Per card per token, each direction | Sequential stages per AllReduce |
|---|---|---|---|
| Mainline butterfly, TP-4 | 2 × 20 KB out, 2 × 20 KB in | ~5.1 MB | 2 |
| Mainline butterfly, TP-9 | busiest card 4 × 20 KB each way | ~10 MB (busiest card) | 5 (fold, 3 rounds, copy back), 26 copies |
| NCCL ring (SHM), TP-4 | 2(N−1)/N × 20 KB = 30 KB each way | ~3.8 MB | 6 ring steps |
| ik gather-add-broadcast, TP-4 | reducing card 3 × 20 KB in and out | ~7.7 MB (reducing card) | 2 |
| Internal-style one-shot, BF16, TP-4 (fork patch 0006) | 10 KB up, 30 KB down | 1.3 MB up, 3.8 MB down | 1 |

- **Bandwidth is not the limit.** 5 MB per card per token at ~190–213 MB/s is ~25 ms, and cards 0–3 share one PCIe 2.0 x1 switch uplink (~400 MB/s usable when all four are busy, [hardware spec](../../mining_rig.md)). The limit is the 256 or more sequential host-staged hops per token.
  - At the 2 × 3090 PCIe 2.0 x1 point (internal AllReduce, Ampere, fast host), each AllReduce cost ≈ 0.1 ms: 128 × 0.1 ≈ 12 ms of the 24.3 ms token.
  - On the P104 rig with the butterfly, it cost ≈ 6.4 ms: (1042 − ~20) ms / 160 AllReduces, i.e. ~1.3 ms per stage.
- **TP-4 decode on this rig** ≈ compute/4 + 128 × stages × t_stage + exposed launch time.
  - With a Q4_0 file: 53–69 ms (allowing up to 30 % efficiency loss on quarter-size matmuls) + 128 × 2 × (0.25–1.3 ms) + 15–30 ms ≈ 130–430 ms.
  - That is 2.3–7.5 tok/s, against ~4.7 for layer split on the same file. The P104 evidence puts the likely outcome at the slow end: a loss (claim to test).
  - CUDA graphs (#27721 build) remove launch overhead but not hop latency.
  - The stock butterfly's per-stage cost (copy, event wait, separate ADD graph, host involvement) is what a Pascal-capable one-shot AllReduce would cut to one hop per AllReduce. That is the only route to the optimistic end.
- **With MTP, communication is paid per verify step, not per token.** Each AllReduce carries (k+1) × 20 KB, and the latency term stays about the same. TP-4 plus MTP therefore loses less than TP-4 alone, which matches the community finding that tensor and MTP gains "stack" on fast links.
- **Prefill in TP-4** moves 20 KB × 512 = 10 MB per AllReduce per 512-token ubatch: ~1.3 GB (one-shot) to ~2.6 GB (butterfly) per card per ubatch each way (my calc). At ~200 MB/s that is several seconds per ubatch, so prefill would drop well below today's 50 tok/s. The 3090 PCIe 2.0 x1 run fell ~10×.
- **Split granularity.** TP-4 gives each card 1 KV group (6 query heads) and 4 GDN key heads. FFN 17408/4 = 4352 = 17 × 256 is aligned for K-quant superblocks; `ssm_out` input 6144/4 = 1536 is aligned too.
  - With 9 cards, attention runs on only 4 of 9 cards, and with 256-element blocks in `ssm_out`, 8 GDN units cover 9 cards, leaving one idle per GDN layer. TP-9 is a poor fit.
- **TP × PP** (2 groups of TP-2) would halve the hops per AllReduce. Mainline cannot express it; ik's `--max-gpu 2` can, if ik builds and runs on this CPU (see [engines_llamacpp_family.md](engines_llamacpp_family.md), section 5).
- **Risks:** experimental code with open crash and lock-up issues on Pascal and on Qwen3.8 MTP (#29466, #27122). Backend sampling is unavailable. NCCL's host-memory buffers and proxy threads compete for 4 GB of RAM and 2 cores.

### Gaps
- No tensor or graph-mode measurement on P106 cards, with 4 cards on x1 links, or with this model on Pascal.
- No measured per-hop latency for 20 KB host-staged copies on PCIe 1.0 x1, nor whether `cudaMemcpyPeerAsync` without peer access blocks the host thread.
- NCCL on sm_61 over SHM with 4 GB host RAM: buffer sizes and CPU cost are unknown.
- Whether the MTP layer is itself split in tensor mode.

---

## 3. Quantization choices for speed on 5-SM Pascal cards

### Takeaway
- **Measured on this rig and confirmed on other Pascal cards:** the cheaper a format is to unpack, the more bandwidth it turns into tokens. Per byte, Q8_0 beats Q4_0, which beats Q4_K_M and the other K-quants and lookup-table formats.
- **The current file is mostly in the slow formats.** UD-Q4_K_M's bytes are Q5_K, IQ4_XS, Q4_K and Q6_K.
- **Options:**
  - Unsloth Q4_0, 82.5 % Q4_0: should run ~4.5–4.9 tok/s, at a measurable quality cost (bartowski Q4_0 KLD 0.0265 against 0.0139 for Q4_K_M).
  - Q4_1: KLD 0.0158, 4-byte-aligned loads, no Pascal speed data; the candidate that may give Q4_K_M quality at Q4_0 speed.
  - Q8_0: slower alone (~3.4 tok/s), but near-lossless and with the flattest multi-token cost, which matters for MTP.
- **Formats to avoid:** MXFP4 and NVFP4 decode through lookup tables (NVFP4 also converts FP8 scales in software) and suit this card worse than Q4_0.
- **Pending mainline Pascal patches:** a few percent each.
- **ik_llama.cpp's Pascal work** doubled IQ4_K on a P40, but that only brought IQ4_K up to Q4_K's speed.

### Cited Findings
**Kernel facts** (current master [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh), [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu))
- Pascal has no MMVQ parameter table of its own and uses the generic one: 4 warps, 1 row per block at 1 column.
- Weight loads:
  - Q4_0, Q5_0, Q8_0, IQ4_NL and Q6_K use `get_int_b2`: two 16-bit loads per 32-bit word, because their blocks are only 2-byte aligned.
  - Q4_1, Q5_1, IQ4_XS and Q4_K use `get_int_b4`: one 32-bit load.
- Inner loops:
  - Q4_0: 2 dp4a plus mask and shift per 32-bit word of weights.
  - Q8_0: 1 dp4a per word.
  - Q4_K: twice Q4_0's dp4a count (dot product plus min-term sum) plus scale and min unpacking.
  - IQ4_NL, IQ4_XS, MXFP4 and NVFP4 decode nibbles through a 16-entry table with `__byte_perm`: ~8 PRMT plus ~5 ALU instructions per word, against ~3 for Q4_0.
- MXFP4 and NVFP4 have MMVQ and MMQ paths on sm_61, with no architecture gate in dispatch (lookup plus dp4a).
  - NVFP4's E4M3 scales are converted with a `__nv_fp8_e4m3` cast, gated only on CUDA ≥ 11.8, so it runs in software on pre-sm_89 cards.
  - The native FP4 MMQ path is Blackwell-only.
- The `__byte_perm` lookup came from [PR #15451](https://github.com/ggml-org/llama.cpp/pull/15451) (merged 2025-08-25): gpt-oss-20B MXFP4 on a P40, pp512 at physical batch 1, 63.60 → 81.98 (+29 %). Lookup formats were instruction-bound on Pascal.

**Per-format speed on sm_61**
- GTX 1060 6GB: the same GP106 chip with 10 SMs and 192 GB/s. Llama-3.2-3B tg128, `-r 10`, baseline → with open [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479) (2026-07-09, 2 warps instead of 4 at batch 1 for cc 6.1):
  - Q4_0 49.41 → 51.68.
  - Q4_K_M 43.49 → 45.85.
  - Q5_K_M 39.92 → 41.22.
  - Q6_K 34.10 → 33.99.
  - Q8_0 32.62 → 33.43.
  - Using standard file sizes, the effective bandwidth ranks Q8_0 ≈ 114 GB/s, Q4_0 ≈ 99, Q5_K_M ≈ 96, Q4_K_M ≈ 93, Q6_K ≈ 90 (my calc).
- Tesla P40, Llama-3.1-8B tg128 ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334), 2026-10-11):
  - Q4_0: 52.28 stock, 54.75 with MMVQ fusion off.
  - Q4_K_M: 48.94 stock, 48.34 with fusion off.
- This rig, Llama-2-7B, one card: Q4_0 19.45 tok/s (73 GB/s), Q4_K_M 14.99 (60 GB/s), Q8_0 on 2 cards 13.23 (93 GB/s) ([calibration](../../benchmarks/2026-10-22-single-card-calibration.md)).
  - The 10-SM P106-100 does ~30.4 tok/s on the same Q4_0 file ([scoreboard #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)): 1.56× for twice the SMs at the same memory, consistent with issue-bound decode.
- Multi-token cost by format on a P40: section 1's table from [PR #20831](https://github.com/ggml-org/llama.cpp/pull/20831). Q8_0, Q4_0 and IQ4_NL scale far better than Q4_K_S and IQ4_XS.
- No Pascal decode numbers were found for Q4_1, Q5_0, Q5_1, IQ4_NL or IQ4_XS on a 2026 build, MXFP4 on a dense model, or NVFP4.

**Mainline changes and pending Pascal patches**
- [PR #30517](https://github.com/ggml-org/llama.cpp/pull/30517) "CUDA: faster MMVQ for low-bit types" (merged 2026-10-17, so in the rig's 2026-10-22 build but not the 2026-09-04 one):
  - VDR 1 → 2 for Q2_K, Q3_K and Q6_K; an extra dp4a replaces `__vsub4` in Q6_K; 2-row blocks for Q2_K and Q3_K in the generic table.
  - RTX 3090: q4_K 1.04× at 1 column and 1.12× at 2; q6_K 1.06× at 1 and 1.19× at 4. No Pascal data.
  - Relevant to the Q6_K output head every 4-bit file here uses.
- [PR #26705](https://github.com/ggml-org/llama.cpp/pull/26705) (merged 2026-09-07; only in the newer build): branchless Q4_K/Q5_K unpack. Predicated instructions at 8 columns drop from 226 to 34; gains appear at batch ≥ 4–5 on 4090/5090.
- [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479) (open): Pascal-specific 2-warp MMVQ, +2.5 % to +5.4 % on a GTX 1060 (above).
- [Issue #30334](https://github.com/ggml-org/llama.cpp/issues/30334) (open):
  - `ggml_cuda_should_fuse_mul_mat_vec_q` excludes `cc <= 600`, so cc 6.1 cards keep MMVQ fusion (introduced by [#16715](https://github.com/ggml-org/llama.cpp/pull/16715)).
  - On a P40, fusion off gave Q4_0 +4.7 %, Q3_K_M +5.3 %, Q4_K_M −1.2 %, Q2_K −5.5 %.
  - `GGML_CUDA_DISABLE_FUSION=1` turns off all fusions, not just this one.
- [Issue #30235](https://github.com/ggml-org/llama.cpp/issues/30235) (open): Q8_0 MMQ tile I = 128 gives +12 % pp512 on a Tesla P4 (prefill only; tg −0.3 %).
- Fork patches for P100 (sm_60) do not transfer to sm_61:
  - The fp16 matvec ([key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp) 0007) relies on P100's full-rate FP16.
  - Q4_K q8_1 sum reuse (0001, +4 %) is not bit-identical; a similar idea was rejected elsewhere for KLD 0.019 ([Kmic-68/llama.cpp PR #1](https://github.com/Kmic-68/llama.cpp/pull/1), 2026-09-25).
  - The same PR's main change, Q4_K MMVQ at VDR 4 (scale and min unpacking paid once per 32 values instead of 16), took a P100 Q4_K matvec from 121 to 81 µs with exact results. It was kept sm_60-only because "it was only measured on sm_60, where dp4a is emulated". Whether it helps the issue-bound sm_61 is untested, and relevant to every K-quant file here.
  - [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp) changes only MMQ tiles, i.e. prefill.
- Rows per block on sm_61 are unsettled: ik [PR #2701](https://github.com/ikawrakow/ik_llama.cpp/pull/2701) found 2 rows at 1 column neutral on a P40 (+0.3 % Llama, −4.5 % DeepSeek) but +12 % on a P100.

**ik_llama.cpp**
- [PR #2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509) (merged 2026-09-23) replaced IQ4_K's per-byte global table gather with `__byte_perm`.
  - Qwen3.5-4B IQ4_K tg128: P40 31.75 → 67.88, P100 34.48 → 60.26; perplexity identical.
  - On a P100 with Llama-3.1-8B, IQ4_K went 22.2 → 42.7 against ik's Q4_K_M at 42.36. The doubling fixed an outlier rather than beating Q4_K.
  - IQ4_NL, IQ4_XS, IQ4_KS, IQ4_KSS and MXFP4 had already moved to `__byte_perm` in ik [PR #713](https://github.com/ikawrakow/ik_llama.cpp/pull/713) (2025-08-21).
- Still on scalar per-byte tables in ik: IQ5_K, IQ5_KS and IQ6_K. The trellis (KT) types use a 32-bit multiply hash per weight plus dp4a, likely expensive on Pascal (inference from source; no data).
- ik's Pascal status doc says 1–8 tokens use MMVQ on sm_61 and that head-256 models (Qwen3.5/3.6) run flash attention on the GPU on a P40 — [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md) ([PR #2670](https://github.com/ikawrakow/ik_llama.cpp/pull/2670), 2026-10-14).
- ik supports dense Qwen3.5-family models with fused delta-net kernels and Qwen3.8 MTP (README). ubergarm publishes an ik-only Qwen3.8-27B IQ4_KS (15.75 GiB, PPL 6.9938 against BF16 6.9540) — [ubergarm/Qwen3.8-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.8-27B-GGUF).

**Quality: bartowski Qwen3.8-27B** (build b10896, wiki.test.raw, 100 chunks, against BF16 PPL 6.744) — [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF):

| Quant | Size | PPL | Mean KLD | Same top token |
|---|---|---|---|---|
| Q8_0 | 29.12 GB | 6.7514 | 0.00092 | 98.75 % |
| Q6_K | 23.86 GB | 6.7400 | 0.00298 | 97.39 % |
| Q5_K_M | 20.92 GB | 6.7539 | 0.00529 | 96.87 % |
| Q4_K_L | 18.82 GB | 6.7620 | 0.01058 | — |
| Q4_1 | 17.83 GB | 6.7607 | 0.01579 | 94.51 % |
| Q4_K_M | 17.44 GB | 6.7645 | 0.01388 | 95.00 % |
| IQ4_NL | 17.44 GB | 6.7645 | 0.01496 | 94.78 % |
| Q4_K_S | 16.36 GB | 6.7705 | 0.01560 | — |
| Q4_0 | 16.35 GB | 6.8643 | 0.02652 | 92.78 % |
| IQ4_XS | 15.48 GB | 6.7803 | 0.01884 | 94.14 % |

- Other quality data:
  - ubergarm Qwen3.6-27B: BF16 6.9066; IQ4_KS 6.9740 (14.69 GiB); smol-IQ4_NL 7.0040 (15.41 GiB, mainline-compatible, nearly pure IQ4_NL with Q8_0 SSM tensors and head) — [ubergarm/Qwen3.6-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.6-27B-GGUF).
  - Unsloth's own Qwen3.8-27B KLD charts are images only.
  - A secondary test gives UD-Q4_K_XL KLD 0.00955 — [kingy.ai](https://kingy.ai/blog/qwen3-8-27b-best-quantization-gguf/) (S).
- Files on offer:
  - Unsloth: Q4_0 16.1 GB, Q4_1 17.5 GB, Q8_0 29.0 GB, UD-IQ4_XS 14.3 GB, UD-Q4_K_S/M/XL 15.4/16.5/17.6 GB, UD-Q5_K_M 19.8 GB, UD-Q6_K 22.0 GB; no IQ4_NL or Q5_0.
  - mradermacher's i1 repo: Q4_0 15.9 GB, Q4_1 17.4 GB, IQ4_XS 15.4 GB.
  - A Qwen3.8-27B NVFP4 GGUF exists, measured only on Blackwell ([cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF](https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF)).
  - No dense MXFP4 GGUF was found.
  - No published file is pure Q4_0 for every tensor: the head stays Q6_K, and Unsloth keeps `ssm_out` at Q5_K.
- Output head measurement: on a P100 a 606 MiB Q4_1 head of Qwen3.5-9B took 1.19 ms per token, ~8 % of decode wall time — [sbendhiba/llama-cpp-p100-patches](https://github.com/sbendhiba/llama-cpp-p100-patches) (S).

### Inferences
**Projected single-token speed by file on 4 cards (my calc).** Each format gets the rig's measured single-card efficiency: Q4_0/Q4_1 ~73 GB/s, K-quants ~60 GB/s, Q8_0 ~93 GB/s. Claims to test.

| File | GPU bytes per token | Estimated time | Estimated tok/s | Mean KLD (bartowski equivalent) |
|---|---|---|---|---|
| UD-Q4_K_M (now) | 15.39 GB | 254 ms (measured) | 3.94 (measured) | ~0.012–0.014 (Q4_K_M class) |
| Unsloth Q4_0 | 15.07 GB | 205–220 ms | 4.5–4.9 | ~0.027 |
| Unsloth Q4_1 | 16.45 GB | 205–235 ms (4-byte loads may beat Q4_0 per byte) | 4.3–4.9 | ~0.016 |
| bartowski IQ4_NL | 16.48 GB | ~250–270 ms (25 % Q6_K) | ~3.7–4.0 | 0.015 |
| Unsloth Q8_0 (6 cards) | 27.23 GB | ~290–310 ms | ~3.2–3.4 | 0.0009 |

- **The output head is 7 % of a 4-bit file's bytes, and Q6_K, one of the slowest per-byte formats here.** A custom requantization could set `--output-tensor-type q4_0` (0.72 GB instead of 1.04 GB) or q8_0 (1.35 GB but the cheapest to unpack), and put `ssm_out` in Q4_0. It needs `llama-quantize` with an imatrix from BF16 (54 GB) or with `--allow-requantize` from Q8_0, done on another machine because of this host's 4 GB of RAM. Saving: ~2–5 % of the token time; quality impact unmeasured.
- **Which build is used shifts the ranking.** The 2026-10-22 build has #26705 (K-quant unpack) and #30517 (Q6_K), which narrow the K-quant gap, so the current file may already be a little faster on the new build. Compare files on the same build.
- **For speculative decoding, the multi-token cost matters as much as the single-token speed** (section 1). Q8_0 and Q4_0 scale best on a P40.
- **Expected effect of the Pascal patches on Q4_0:** #25479 (+3–6 %) and the #30334 fusion gate (+0–5 %), so together roughly +3–10 %.
- **ik_llama.cpp quants** (IQ4_KS, IQ4_K) would need ik to build and run on this CPU first. Their Pascal decode speed is unmeasured except IQ4_K, which reaches Q4_K speed.

### Gaps
- No measurement of Q4_1, IQ4_NL or Q5_0 decode on any 5-SM card, and none of multi-token costs on such a card.
- No KLD for Unsloth's Q4_0 or Q4_1 of Qwen3.8-27B. Their imatrix mixes differ from bartowski's.
- No data isolating the effect of the output-head quant on quality for Qwen3.x.

---

## 4. Activation sparsity and other training-free compute skipping

### Takeaway
Training-free activation sparsity is real for SwiGLU models. TEAL, R-Sparse, LaRoSA, WINA and Prox reach 40–50 % FFN or input sparsity with ~1–3 points of quality loss at 7–9B and 1.2–1.8× decode on fast GPUs with custom FP16 kernels. One 2026 paper ran a TEAL-style method on Qwen3.5 hybrids and another on Qwen3.6-27B. But:
- Nothing is implemented in mainline llama.cpp or ik_llama.cpp. Maintainers declined PowerInfer-style work, and LayerSkip requests went stale.
- In GGUF every FFN matrix is quantized along its input dimension, so input-channel skipping cannot skip whole blocks without a second, transposed copy of the weights and a new sparse quantized kernel.
- On an issue-bound 5-SM card a sparse kernel's extra index work eats into the byte savings.

The byte ceiling is 1.37–1.5× at 40–50 % FFN sparsity. Layer skipping and early exit have no implementation either, and skipping a GDN layer leaves its recurrent state stale. The trained MTP head is the practical form of self-speculation for this model.

### Cited Findings
**Methods (all batch-1 decode)**
- **TEAL** — [arXiv 2408.14690](https://arxiv.org/abs/2408.14690) (v1 2024-08-26; ICLR 2025):
  - Training-free magnitude thresholds on the inputs of all 7 projections; the output head stays dense.
  - Llama-3-8B on an A6000, gpt-fast with a Triton kernel and column-major weights in FP16: 45.3 → 55.4 / 64.2 / 73.9 tok/s at 25 / 40 / 50 % (1.22 / 1.42 / 1.63×); on an A100, 1.11 / 1.25 / 1.33×.
  - Quality: WikiText PPL 5.87 → 5.94 / 6.21 / 6.67.
  - Accuracy was tested with 4-bit AWQ and 2/3-bit weights, but the repo has no quantized sparse kernel — [FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL).
- **R-Sparse** — [arXiv 2504.19449](https://arxiv.org/abs/2504.19449) (2025-04, ICLR 2025): input sparsity plus a low-rank path; ~40–42 % faster at 50 % in an FP32 Hugging Face implementation; no fused quantized kernel.
- **CATS** — [arXiv 2404.08763](https://arxiv.org/abs/2404.08763) (2024): thresholds the SiLU gate output, MLP only, ~15 % latency gain at 50 %. TEAL reports CATS collapsing on Llama-3-8B at 40 %.
- **LaRoSA** — [arXiv 2507.01299](https://arxiv.org/abs/2507.01299) (ICML 2025):
  - PCA rotation folded into the weights.
  - Qwen2.5-7B PPL 6.85 → 7.42 at 50 % (TEAL 7.81).
  - 1.42× at 50 % on an A100 (107.5 → 152.7 tok/s).
- **WINA** — [arXiv 2505.19427](https://arxiv.org/abs/2505.19427) (v2 2026-02-18): ranks by |x|·‖W column‖; at 50 % Llama-3-8B average 72.99 → 71.05 (TEAL 70.03); GEMV microbenchmarks only.
- **Prox** — [arXiv 2607.27591](https://arxiv.org/abs/2607.27591) (2026-07-30):
  - FFN only; ranks intermediate channels with an INT4 proxy of up/gate.
  - Abstract: up to 1.99× end-to-end at 70 % FFN sparsity.
  - Per the paper body, it is tested on Qwen3.5-4B/9B: Qwen3.5-9B downstream 70.9 → 70.4 / 69.5 / 68.4 at 40 / 50 / 60 %; TEAL 69.6 at 50 %.
  - The proxy adds nothing when the weights are already 4-bit.
- **SpAx** — [arXiv 2610.02598](https://arxiv.org/abs/2610.02598) (2026-10-01):
  - Three tiers per column: full, approximate, skip.
  - The abstract targets offloaded weights (CPU memory or flash): up to 2.74× with 4-bit weights, ≤ 10 % perplexity increase.
  - The body reports Qwen3.6-27B Q4_K_M: dense PPL 7.47, TEAL-FFN at a 30 % read budget 8.27 (+10.7 %). No GPU-resident results.
- **SpQt** — [arXiv 2511.04477](https://arxiv.org/abs/2511.04477) (2025-11):
  - A llama.cpp fork with a "zigzag" Q4_K layout grouping superblocks along the input-channel dimension.
  - Metal only; 1.33–1.55× end-to-end on Apple M2-class chips; Llama-2-7B Q4_K PPL 5.28 → 5.65 at 50 %.
  - No code link found.
- **GRIFFIN** — [arXiv 2404.01365](https://arxiv.org/abs/2404.01365) (2024): picks FF neurons once per prompt and keeps them for the whole generation (structured, compactable); 50 % of FF parameters, ~1.25–1.29× (S, abstract summary).
- **Training-required, not applicable to stock Qwen weights:** DejaVu ([arXiv 2310.17157](https://arxiv.org/abs/2310.17157), trained predictors), ReLU-fication and ProSparse ([arXiv 2310.04564](https://arxiv.org/abs/2310.04564), [arXiv 2402.13516](https://arxiv.org/abs/2402.13516)), TurboSparse, Q-Sparse, and PowerInfer and PowerInfer-2 ([arXiv 2312.12456](https://arxiv.org/abs/2312.12456), llama.cpp forks needing ReLU models and trained predictors).
- **GDN hybrids:** a 2026 repo applies TEAL to Qwen3.5 0.8B/2B/4B in Triton FP16 — [srikara-V/sparse-gdn-decode](https://github.com/srikara-V/sparse-gdn-decode) (S):
  - Qwen3.5-4B at 40 %: −0.78 points.
  - L40S speed 1.12× (2B) and 1.27× (4B); neutral on H100.
- **Layer skipping and early exit:**
  - LayerSkip ([arXiv 2404.16710](https://arxiv.org/abs/2404.16710)) needs training.
  - SWIFT ([arXiv 2410.06916](https://arxiv.org/abs/2410.06916), ICLR 2025) is training-free self-speculation by skipping 40–50 % of layers: 1.3–1.6× on Llama-2 in Hugging Face transformers.
  - CLaSp ([arXiv 2505.24196](https://arxiv.org/abs/2505.24196)): 1.3–1.7×.
  - ShortGPT ([arXiv 2403.03853](https://arxiv.org/abs/2403.03853)) removes layers outright; generative tasks collapse at 25 % removal.
  - No study of layer skipping on GDN hybrids was found.

**llama.cpp status**
- No activation-sparsity, TEAL, sparse-GEMV or layer-skip code in mainline or ik_llama.cpp (GitHub searches, 2026-10-22).
- Sparsity prediction [#4759](https://github.com/ggml-org/llama.cpp/issues/4759) was closed. In [discussion #4534](https://github.com/ggml-org/llama.cpp/discussions/4534) (Dec 2023) maintainers judged PowerInfer integration not worth it (ReLU-only, partial-offload benefit).
- LayerSkip requests ([#10820](https://github.com/ggml-org/llama.cpp/issues/10820)) closed as stale.
- Weight layout: `ffn_gate`/`ffn_up` are `{n_embd, n_ff}` and `ffn_down` is `{n_ff, n_embd}` ([src/models/qwen35.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)). ggml's contiguous dimension ne0 is the input dimension, and quant blocks run along it.

### Inferences
- **Input sparsity in the stock layout saves nothing.** A zero input channel is one element inside every row's 32- or 256-element block, so a block is skippable only if all its inputs are zero.
  - Output-row skipping of `ffn_gate`/`ffn_up` is block-aligned, but needs the mask before the matmul.
  - Input skipping on `ffn_down` needs a transposed copy (`{n_embd, n_ff}`, as PowerInfer's `ffn_down_t`) and an "axpy"-style kernel. A transposed Q4_0 copy of all down projections is ~3.2 GB, which fits in the rig's ~38 GB of unused VRAM. This is one of the few places where the rig's spare memory is an asset.
- **Byte ceilings (my calc, from section 0's parameter split):**
  - TEAL on the FFN only: 1.37× at 40 % and 1.50× at 50 %.
  - On all projections: 1.61× and 1.9×.
  - CATS-style, with the gate kept dense: ~1.29× at 50 %.
  - Published kernels reach ~75 % of the ideal on fast GPUs.
- **On this card the realistic gain is uncertain and could be negative.** A sparse axpy kernel dequantizes with FP32 FMAs instead of dp4a, and needs index or threshold work per channel. On an issue-bound 5-SM card that can cost more than the saved bytes. Quality at 27B scale is known only at extreme sparsity (Qwen3.6-27B +10.7 % PPL at ~70 %).
- **The no-new-kernel variant is per-prompt structured compaction** (GRIFFIN-like): pick ~50 % of FFN neurons after the prompt, gather them into compact matrices on the GPU, decode with ordinary MMVQ. The gather reads the FFN once per prompt (~10 GB, ~0.15 s on-GPU). It needs graph and weight-management changes in llama.cpp, and its quality on long generations is the risk.
- **A post-hoc MoE carve** is static, block-aligned sparsity. One community carve exists (section 6), with weak quality.

### Gaps
- No activation-sparsity result for Qwen3.8-27B, none on Pascal, and no quantized sparse CUDA kernel for K-quants or Q4_0 in any maintained codebase.
- No layer-skip or early-exit study on GDN hybrids.

---

## 5. Other llama.cpp-level levers

### Takeaway
For the dense 27B these levers are worth a few percent each:
- **CUDA graphs** (#27721; +2–5 % expected; the build exists on the rig).
- **Backend sampling** (`-bs`; removes ~10–15 ms per served token, and ~45–60 ms per MTP round at k = 3).
- **The Pascal MMVQ patches** (+3–10 %).
- **Card count and clock behaviour.** The 4 → 9 card penalty is clock-down, not copies; use the minimum card count, or test a "prefer maximum performance" clock mode.
- **KV f16 versus q8_0**, beyond ~16–32k tokens of context.

The GDN kernel is a small share of the token (~2–4 %) and has no Pascal-specific issues. The power limit cannot be raised above 75 W. Clock offsets via Coolbits are possible because Xorg runs on every card, but they are a persistent system change with stability risk.

### Cited Findings
**CUDA graphs**
- Master still disables graphs for `cc < GGML_CUDA_CC_VOLTA` in `ggml_cuda_graph_set_enabled` — [ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu). The rig's log confirms "disabling CUDA graphs due to GPU architecture" ([small MoE benchmark](../../benchmarks/2026-10-22-small-moe-overhead.md)).
- [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) "CUDA graphs on Pascal (sm_61): +40% on MoE, +7% on dense, no regression" (opened 2026-08-25, open, no review):
  - A one-line change of the gate to `GGML_CUDA_CC_DP4A`.
  - GTX 1080, dense Qwen-based 9B Q5_K_M: tg64 28.5 → 30.1 (+5.6 %).
  - P102-100 (Linux, driver 580, CUDA 12.9), Ling-3.0-tiny Q8_0: tg128 70.80 → 88.42; pp flat.
- Graphs are captured after 2 identical calls; any change in node shapes resets the warm-up ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)).
  - `GGML_CUDA_DISABLE_GRAPHS` disables graphs when set to anything.
  - `GGML_CUDA_GRAPH_DEBUG=1` logs disable and recapture events ([#30327](https://github.com/ggml-org/llama.cpp/pull/30327), merged 2026-10-13).
  - [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) (merged 2026-10-20) fixed graph recapture after a memory-pool flush.
- Graph memory is ~128 B per node per graph ([PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749)).

**Backend sampling**
- [PR #17004](https://github.com/ggml-org/llama.cpp/pull/17004) (merged 2026-01-04): flag `-bs` / `--backend-sampling` (env `LLAMA_ARG_BACKEND_SAMPLING`), "experimental", default off ([common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)).
- GPU implementations exist for greedy, dist, top_k, top_p, min_p, temp, temp_ext, penalties and logit_bias; the longest supported prefix of the sampler chain is offloaded, and no-op samplers count as supported ([src/llama-sampler.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)). The model's default chain with its default top_k 20 / top_p 0.95 / temperature 1.0 is therefore fully offloadable.
- It falls back to the CPU with a grammar (including tool-call and JSON-schema grammars), with a reasoning-budget sampler, and in tensor mode.
- The logits copy is skipped when every output row has a backend sampler. With `-sm layer` the sampler runs on the device holding the output head.
- Measured gains:
  - +1.3 % at 1 slot and +48 % at 32 slots on an RTX 5090 ([#27050](https://github.com/ggml-org/llama.cpp/issues/27050)).
  - +6.6 % on Qwen3.8-27B with DFlash2 when greedy chains use target and draft backend sampling ([#29797](https://github.com/ggml-org/llama.cpp/pull/29797), merged 2026-10-07).
- On this rig, llama-server decoding was 16–18 % slower than llama-bench for Qwen3.6-35B-A3B (15.5 against 18.5–18.9 tok/s, ~11 ms per token), attributed to CPU sampling over the large vocabulary ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).

**Fusion and MMVQ**
- Fusions active for qwen35 on sm_61 ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)):
  - RMS_NORM + MUL (+ADD).
  - SSM_CONV (+ADD) + SILU ([#22478](https://github.com/ggml-org/llama.cpp/pull/22478)).
  - Unary + MUL; chained ADD/MUL.
  - GDN with its state write.
  - MMVQ gate + up + SwiGLU.
- Fusions that do not apply to qwen35: the RoPE fusions need NORMAL or NEOX rope, and qwen35 uses multi-section rope; the MoE fusions.
- `GGML_CUDA_GRAPH_OPT=1` (concurrent Q/K/V streams) returns early without graphs and is single-GPU only.

**Gated DeltaNet kernel**
- History:
  - Op added [#19504](https://github.com/ggml-org/llama.cpp/pull/19504) (2026-03-07).
  - Redundant copies removed [#23940](https://github.com/ggml-org/llama.cpp/pull/23940) (2026-07-03).
  - Four state columns per warp [#30087](https://github.com/ggml-org/llama.cpp/pull/30087) (2026-10-08): Qwen3.8-27B pp512 +8.2 % on a 4090, decode unchanged. A P100 fork cited in the PR found two columns better for decode shapes (13.2 → 8.4 µs per call).
  - Chunked MMA prefill kernel [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) (2026-10-14), Ampere and newer only.
- No sm_61 issue or measurement was found. The rig's earlier Qwen3.8-27B run shows the op runs on sm_61.

**KV cache and flash attention on sm_61**
- sm_61 uses FP32 FA math. f16 K/V with this model's GQA ratio takes the TILE kernel; quantized K/V takes the VEC kernel ([fattn.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu)).
- Default compiled K/V pairs: q4_0, q8_0, f16 and bf16, each paired with itself ([docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)).
- P100, Qwen3.8-27B, 74,919-token request: tg 20.92 with f16 against 16.85 with q8_0 ([key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp)). No sm_61 data.

**Placement, card count, clocks, power**
- The output head and MTP layer sit on the last device. `-ts` splits by layer count, not bytes ([src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)).
- Calibration on this rig ([calibration](../../benchmarks/2026-10-22-single-card-calibration.md)):
  - One busy card hits the 75 W cap in 50–90 % of samples at 1710–1760 MHz.
  - 2 cards: ~1830 MHz, +4–5 %.
  - 9 cards: each busy ~10 %, average core clock ~1595 MHz, −13 % against 2 cards, ~7 ms per token on Llama-2-7B.
- The power limit is adjustable only between 60 and 75 W. The only memory clocks are 4004 and 405 MHz, and application clocks are unsupported ([hardware spec](../../mining_rig.md)).
- `nvidia-smi -lgc` is documented as Volta+ ([nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html)). Coolbits offsets through `nvidia-settings` work under Xorg on Pascal ([multigpu_pascal_performance.md](multigpu_pascal_performance.md), section Q8).

**Host-RAM costs of hybrid models in llama-server** (master [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h), [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp))
- Context checkpoints default to up to 32 per slot (`-ctxcp`), spaced at least 8192 tokens apart (`-cms`). For this model each one is a ~150 MB copy of the recurrent state in host memory.

### Inferences
- **The 4 → 9 card penalty for the 27B is clock-down (my calc).** 1/3.43 − 1/3.94 s = 37.7 ms per token for 5 extra boundaries, ~7.5 ms each, against ~1 ms each on Llama-2-7B. Both runs lose ~13 %, so the cost scales with work per token: lightly loaded cards clocking down between bursts, as the calibration logged. A copy latency would be a fixed cost per hop.
  - Use the fewest cards that fit: 4 for 4-bit files, 6 for Q8_0.
  - A clock mode that holds P0 boost while idle (`nvidia-settings -a '[gpu:N]/GPUPowerMizerMode=1'`, a runtime setting under the running Xorg) might recover part of it, even on 4 cards where each is busy ~25 % of the time (claim to test).
- **CUDA graphs for the dense model.** ~2.1k nodes × a few µs of launch overhead saved ≈ 5–10 ms of 212–254 ms: +2–5 %. With MTP, the draft graphs (#28549, master) and a fixed verify width help keep graphs captured. Variable draft lengths from p-min change node shapes and force recaptures (claim to test with `GGML_CUDA_GRAPH_DEBUG=1`).
- **Backend sampling.** Without it a served token carries ~5 ms of logits copy (248,320 × 4 B over x1) plus ~10 ms of CPU sampling: ~6 % at 3.94 tok/s, ~7 % at 4.7, and k+1 times that per MTP round. On the GPU, a top-k over 248k entries on 5 SMs costs an unmeasured amount.
  - With CUDA 12.9's bundled CCCL, top-k uses a radix path; `-DGGML_CUDA_CCCL_VERSION` can fetch a newer CCCL with CUB DeviceTopK (≥ 3.4.3, [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)).
- **GDN** costs ~2–4 % of the token (state traffic), so a better GDN kernel is not a lever for decode. With MTP, verifying k+1 tokens also writes k+1 intermediate states for rollback (~150 MB each), ~1–2 % per round.
- **KV.** Up to ~8k tokens of context the KV share is ~3 %. At 32k it is ~10–14 %, where q8_0 halves the bytes but switches to the VEC kernel. Measure before choosing; the P100 data favours f16.
- **Clock offsets.** Under a power cap, a positive core offset shifts the V/F curve and can raise clocks at the same power. Since the card is issue-bound, that would raise decode by up to the clock gain (~+5–8 % at most, claim to test). Memory offsets help less because DRAM is not the limit. Coolbits is a persistent Xorg configuration change, and unstable clocks corrupt output silently, so validate numerically.
- **Context checkpoints:** each new checkpoint is a ~150 MB transfer over the x1 links (~0.2–0.75 s) and ~150 MB of the 4 GB host RAM. Capping `-ctxcp` protects the host on long conversations.

### Gaps
- No qwen35 dense measurement on any sm_61 card with or without graphs. No P104/P106 data.
- Cost of GPU top-k over 248k entries on a 5-SM card.
- Whether `GPUPowerMizerMode` sticks on these cards and changes clocks between bursts.
- P106-090 response to core clock offsets.

---

## 6. Smaller, pruned and distilled variants

### Takeaway
Qwen3.6 and Qwen3.8 have no dense model below 27B; Qwen3.5 has 0.8B, 2B, 4B and 9B. Community depth-pruned Qwen3.8-27B cuts exist:
- A ~19.7B "Minitron"-style model with 44 of 64 layers, healed with distillation.
- A security-tuned 19.3B cut, whose general scores collapse.
- A 16.8B layer-and-width cut.

None has standard benchmarks except the security cut. By bytes they would decode ~1.4× and ~1.65× faster. The measured alternative is Qwen3.6-35B-A3B (MoE): 18.5 tok/s in llama-bench on this rig, 22.3 with MTP in llama-server, close to Qwen3.6-27B on Qwen's own benchmarks but behind Qwen3.8-27B.

### Cited Findings
- **Official lineup** (Hugging Face creation dates):
  - Qwen3.5 dense: 0.8B and 2B (2026-02-28), 4B and 9B (02-27), 27B (02-24).
  - Qwen3.6: 27B (04-21) and 35B-A3B (04-15) only.
  - Qwen3.8: 27B (08-05), Flash-Next (MoE, 08-24) and 2.4T-A95B.
  - Sources: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
  - Qwen3.5-9B: 32 layers, hidden 4096, FFN 12288.
- **Official benchmarks:**
  - Qwen3.5-27B / 9B: MMLU-Pro 86.1 / 82.5, GPQA-D 85.5 / 81.7, LiveCodeBench v6 80.7 / 65.6, IFBench 76.5 / 64.5.
  - Qwen3.6-27B against Qwen3.6-35B-A3B: GPQA 87.8 / 86.0, LiveCodeBench 83.9 / 80.4, SWE-bench Verified 77.2 / 73.4, Terminal-Bench 2.0 59.3 / 51.5.
  - Qwen3.8-27B against Qwen3.6-27B: GPQA 89.2 / 87.8, LiveCodeBench v6 90.3 / 83.9, SWE-bench Pro 61.7 / 53.5, Terminal-Bench 2.1 73.0 / 63.4.
  - Model cards above.
- **Community derivatives of Qwen3.8-27B:**
  - [exnivo/Qwen3.8-20B-Minitron](https://huggingface.co/exnivo/Qwen3.8-20B-Minitron) (2026-08-16):
    - 64 → 44 layers by removing 5 hybrid groups (15 GDN + 5 attention layers); 27.36 B → 19.75 B.
    - Healed with top-k distillation from Qwen3.8-27B logits plus a merged LoRA.
    - Only an internal evaluation is reported.
    - GGUFs from mradermacher (static and i1).
  - [hcnote/Qing-Sec-20B-Qwen3.8-27B-Slim](https://huggingface.co/hcnote/Qing-Sec-20B-Qwen3.8-27B-Slim) (2026-09-12): 44 layers, 19.29 B, LoRA on security and code data; MMLU 81.6 → 59.0, CEval 82.2 → 58.1.
  - [logic65/Qwen3.8-Whittle-16B](https://huggingface.co/logic65/Qwen3.8-Whittle-16B) (2026-08):
    - 20 layers removed plus ~25 % of MLP width; 16.8 B; QLoRA heal.
    - Its own 39-task battery only; repetition loops reported.
    - Q4_K_M GGUF 10.14 GB.
  - logic65's Qwen3.8-Whittle-MoE-27B-A17.8B: a post-hoc MoE carve (64 experts, 16 routed, ~17.8 B active) running as `qwen35moe`; weak on its own evaluations.
  - [erokhins/Qwen3.6-27B-FFN-pruned-25](https://huggingface.co/erokhins/Qwen3.6-27B-FFN-pruned-25): FFN 17408 → 13056, no healing, no evaluations.
  - Vocabulary-trimmed Qwen3.6-27B GGUFs exist (248k → 146k tokens, ~2 GB smaller file), without measurements.
  - NVIDIA publishes only NVFP4 quantizations of Qwen3.6/3.8-27B, unusable on Pascal; no Nemotron distillation of a Qwen3.x 27B was found.
- **On this rig:** Qwen3.6-35B-A3B Q4_0 runs at 18.46 tok/s in llama-bench on 4 cards, and 15.5 → 22.3 tok/s with MTP in llama-server ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).

### Inferences
**Speed by bytes, same quant family (my calc):**

| Model | GPU weights per token, relative to 27B | Speed relative to 27B |
|---|---|---|
| Qwen3.8-20B-Minitron | ~71 % | ~1.4× |
| Whittle-16B | ~60 % | ~1.65× |
| Qwen3.5-9B | ~31 % | ~3× |

- Each of those smaller models trades away quality that is unmeasured (pruned cuts) or large (9B: LiveCodeBench −15 points against Qwen3.5-27B, and further behind Qwen3.8-27B).
- A pruned cut keeps the 248k-row output head, so its share of bytes grows.
- The pruned cuts also keep the GDN/attention pattern, and they may or may not keep a working MTP layer: removing layers changes the final hidden states the MTP head was trained on (claim to test).

### Gaps
- No standard benchmarks (MMLU-Pro, LiveCodeBench, SWE-bench) for the Qwen3.8 20B or 16B cuts.
- No KLD of the cuts against Qwen3.8-27B.

---

## Experiment ideas

Ranked by expected gain per hour of effort. Each expected outcome is a claim to test, not a prediction to rely on.

**Conventions for every experiment:**
- Record the build commit, file checksum, cards used, context size and flags.
- Use llama-bench for raw speed and the repo's [scripts/server_bench.sh](../../../scripts/server_bench.sh) for served speed: four fixed prompts at temperature 0, 256 tokens.
- Log `nvidia-smi` clocks, power and throttle reasons alongside.
- On the 4 GB host keep `-ub` ≤ 512 and `-c` moderate (see [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md) for the pinned-buffer limit).

1. **MTP on the current file.**
   - Hypothesis: the built-in NextN layer gives 1.25–1.5× at n-max 1–2 even on UD-Q4_K_M.
   - Setup:
     - `llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf -ngl all -fa on -c 32768 -np 1 --spec-type draft-mtp --spec-draft-n-max {1,2,3}`, without and with `--spec-draft-p-min 0.75`, on cards 0–3 (`CUDA_VISIBLE_DEVICES=0,1,2,3`).
     - Repeat with Unsloth's separate file: `-md MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB), first on the same cards, then with `-devd` on a fifth card (`CUDA_VISIBLE_DEVICES=0,1,2,3,4 -dev CUDA0,CUDA1,CUDA2,CUDA3 -devd CUDA4`).
   - Measure: served tok/s per prompt, acceptance (log), per-card VRAM, prefill tok/s with MTP on (#27428), and whether the log reports "retried without pipeline parallelism".
   - Expected: built-in head at n = 1–2 gives 1.25–1.5× the plain served speed; n = 3 is no better. The separate file is equal or slightly faster, with acceptance within a few points. Prefill drops noticeably.
   - Effort: 1–2 h (no download for the built-in path).
   - Risk: low. MTP compute-buffer OOM on the last card; fix with `-ts` or `-devd`.

2. **Multi-token cost per quant: the number every speculative decision rests on.**
   - Hypothesis: on 5 SMs a 3-token step costs ~1.5–1.9× a 1-token step for UD-Q4_K_M and ~1.15–1.3× for Q4_0.
   - Setup: `llama-bench -m <file> -ngl 99 -fa on -p 64 -n 0 -ub 1,2,3,4,5,6,8,9,12,16 -r 2` on 4 cards (6 for Q8_0). Do UD-Q4_K_M now, and each new file from experiments 3 and 7 when it arrives. Run on both builds; #26705 and #30517 change K-quant and Q6_K MMVQ.
   - Measure: r(b) = b × tok/s(ub=1) ÷ tok/s(ub=b); the jump between b = 8 (MMVQ) and b = 9 (MMQ).
   - Expected: r rises faster for UD-Q4_K_M than for Q4_0 or Q8_0, and is worse than the P40's ratios in section 1.
   - Effort: 0.5 h per file.
   - Risk: none.
   - Use: picks n-max and the base quant for MTP, and says whether DFlash's 8-wide blocks can ever pay.

3. **Q4_0 file, then Q4_0 plus MTP.**
   - Hypothesis: Unsloth Q4_0 alone gives 4.5–4.9 tok/s; with MTP n=2 and the separate head it gives 5.5–9 tok/s.
   - Setup: download `Qwen3.8-27B-Q4_0.gguf` (16.06 GB). Run `llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3` on 4 cards, then experiment 1's server runs with `-md MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, then experiment 2's sweep.
   - Measure: tg128, served tok/s, acceptance, and perplexity on a fixed small text compared with UD-Q4_K_M. Use `llama-perplexity`; KLD base files get large with a 248k vocabulary, so use published KLD (section 3) or compute KLD on another machine.
   - Expected: +15–20 % alone; the combination is the largest gain available without code changes.
   - Effort: 2–3 h including download and load (~2 min to load over the x1 links).
   - Risk: low. Q4_0 quality is measurably lower (bartowski KLD 0.027).

4. **Backend sampling in llama-server.**
   - Hypothesis: `-bs` removes ~10–15 ms per plain served token and more with MTP.
   - Setup: experiment 1's or 3's best configuration with and without `-bs`; one run with a tool-call or JSON-schema request to confirm the CPU fallback.
   - Measure: served tok/s against llama-bench tg; server log for fallback warnings.
   - Expected: +4–7 % without MTP, +8–15 % with MTP at k = 2–3.
   - Effort: 0.5 h.
   - Risk: low; "experimental" flag.

5. **CUDA graphs (PR #27721 build) on the dense model.**
   - Hypothesis: +2–5 % for plain decode; graphs stay captured with fixed-width MTP verify.
   - Setup: the rig's #27721 variant against stock, same file and flags; llama-bench tg128 and experiment 1's best MTP run. On a build that also has #30327, add `GGML_CUDA_GRAPH_DEBUG=1` to count recaptures, with and without `--spec-draft-p-min`.
   - Measure: tg, served tok/s, recapture rate, VRAM.
   - Expected: small plain gain; p-min gating causes recaptures.
   - Effort: 0.5–1 h.
   - Risk: low. Unmerged; P100 (sm_60) regressed with graphs, but sm_61 data was positive.

6. **Card count, last-card placement and clock mode.**
   - Hypothesis: the 4 → 9 card loss is clock-down, and holding clocks or using fewer, busier cards recovers it.
   - Setup:
     - Q4_0 on 4, 5, 6 and 9 cards (`CUDA_VISIBLE_DEVICES`).
     - On 4 cards, `-ts` variants that give the last card (output head, NextN block, MTP buffer) fewer layers.
     - Then `nvidia-settings -a '[gpu:N]/GPUPowerMizerMode=1'` for the cards in use: a runtime setting under the running Xorg, which reverts when Xorg restarts.
   - Measure: tg128, average core clock per card, throttle reasons.
   - Expected: 4 cards fastest. PowerMizer mode recovers most of the 9-card loss and 0–5 % on 4 cards.
   - Effort: 1–2 h.
   - Risk: low. Higher idle power.

7. **Q4_1 and Q8_0 as bases.**
   - Hypothesis: Q4_1 matches Q4_0's speed at Q4_K_M quality. Q8_0 is slower alone but the best MTP base.
   - Setup: Unsloth Q4_1 (17.5 GB) on 4 cards; Q8_0 (29.05 GB) on 6 cards. Repeat experiments 2 and 3 for each.
   - Measure: tg128, r(b), served tok/s with MTP, perplexity.
   - Expected: Q4_1 4.3–4.9 tok/s. Q8_0 3.2–3.4 alone and ~5.5–6.5 with MTP n=3 if r(4) stays ≤ ~1.6.
   - Effort: 3–4 h (downloads dominate).
   - Risk: low. Q8_0 needs ~6 cards plus the MTP buffer.

8. **Pascal MMVQ patches.**
   - Hypothesis: 2-warp MMVQ for cc 6.1 (#25479) and turning off MMVQ fusion for cc 6.1 (#30334) each add a few percent on Q4_0.
   - Setup:
     - Apply #25479's diff to the current master.
     - Separately, change `cc <= GGML_CUDA_CC_PASCAL` to `cc < GGML_CUDA_CC_VOLTA` in `ggml_cuda_should_fuse_mul_mat_vec_q`; as a quick proxy first, run with `GGML_CUDA_DISABLE_FUSION=1`.
     - Build for sm_61 on another machine with `GGML_NATIVE=OFF` (see [engines_llamacpp_family.md](engines_llamacpp_family.md), section 8).
   - Measure: tg128 for UD-Q4_K_M and Q4_0, and r(b) for b ≤ 4, interleaving A/B runs.
   - Expected: +3–10 % combined on Q4_0, smaller or negative on K-quants.
   - Effort: 2–4 h.
   - Risk: low. Patch drift.

9. **n-gram drafting for edit-style workloads.**
   - Hypothesis: on outputs that copy the prompt (refactors, file rewrites) `ngram-mod` gains a lot despite host-RAM checkpoints; on chat it loses.
   - Setup:
     - Two prompt sets: a 2–4k-token code file to return with a small rename, and the four standard chat prompts.
     - Run `--spec-type ngram-mod` (defaults), `ngram-simple`, and the chain `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 8 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 8`, which keeps drafts within the on-GPU snapshot path.
   - Measure: served tok/s, accepted tokens per round, host RAM, time spent in checkpoints (debug log "created speculative checkpoint").
   - Expected: several × on the rewrite task, ≤ 1× on chat; the capped chain best overall.
   - Effort: 1–2 h.
   - Risk: low. Watch host RAM.

10. **KV type at long context.**
    - Hypothesis: beyond ~16–32k tokens the KV read becomes visible, and f16 (TILE kernel) beats q8_0 (VEC) on sm_61, as on P100.
    - Setup: `llama-bench -p 0 -n 64 -d 0,8192,32768,65536 -ctk f16,q8_0 -ctv f16,q8_0 -fa on` on 4 cards with Q4_0, watching per-card VRAM.
    - Measure: tg at each depth, VRAM.
    - Expected: equal up to 8k; f16 faster at 32k+ but uses twice the KV memory.
    - Effort: 1 h.
    - Risk: low.

11. **MTP draft-vocabulary trim (PR #29143).**
    - Hypothesis: a 32K-entry draft head cuts the draft step from ~0.8–1.4 GB to ~0.4 GB read, adding 5–15 % on top of MTP.
    - Setup: build master plus #29143; use the trimmed sidecar named in the PR (or produce one as the PR describes); repeat experiment 3's MTP runs.
    - Measure: served tok/s, acceptance.
    - Expected: equal acceptance, +5–15 %.
    - Effort: 3–4 h.
    - Risk: medium; unmerged code.

12. **Other drafters: DFlash2, DSpark, Qwen3.5-0.8B.**
    - Hypothesis: they are break-even or worse here, because of 8-wide verifies and host checkpoints.
    - Setup:
      - `--spec-type draft-dflash -md dflash-Qwen3.8-27B-Q4_0.gguf` (ggml-org, 1.09 GB) with `--spec-draft-n-max 3,5,7`.
      - A DSpark GGUF if one converts cleanly.
      - `--spec-type draft-simple -md Qwen3.5-0.8B-Q4_0.gguf --spec-draft-n-max 4` with `-devd` on a spare card.
    - Measure: served tok/s, acceptance, accepted length.
    - Expected: DFlash2 0.8–1.5×, 0.8B draft < 1×.
    - Effort: 2 h.
    - Risk: low.

13. **Mainline tensor parallelism on 4 cards.**
    - Hypothesis: TP-4 with the stock butterfly loses to layer split because of 256 host-staged hops per token; CUDA graphs narrow the gap but do not close it.
    - Setup:
      - `llama-bench -sm tensor -fa on -p 64 -n 32` with `CUDA_VISIBLE_DEVICES` set to (a) 0,1,2,3 (one switch) and (b) two cards per switch (0,1,5,6).
      - Builds: stock and #27721; optionally NCCL built against CUDA 12 for sm_61 (`GGML_CUDA_ALLREDUCE=nccl` against `none`).
      - Then llama-server with `--fit off -c 8192`, without MTP first.
    - Measure: tg, pp, per-card VRAM (#30317 over-allocation), host RAM and CPU (NCCL), output sanity.
    - Expected: 0.5–1.0× of layer split for decode, much worse prefill.
    - Effort: 2–4 h (NCCL build adds more).
    - Risk: medium. Lock-up reports exist (#27122, #29466), and a hang may need a power cycle.

14. **Core clock offset (only with the owner's go-ahead; persistent Xorg configuration change).**
    - Hypothesis: a positive core offset raises clocks under the 75 W cap, and issue-bound decode follows.
    - Setup: enable Coolbits. On ONE card raise `GPUGraphicsClockOffset` in steps, running single-card Llama-2-7B Q4_0 tg and a perplexity check at each step; stop at the first regression, perplexity drift or Xid error.
    - Measure: clock, power, tg, perplexity.
    - Expected: up to +5–8 % on tg.
    - Effort: 2–3 h.
    - Risk: high. Silent corruption or driver hangs; see [multigpu_pascal_performance.md](multigpu_pascal_performance.md), section Q8.

15. **Custom requantization with a cheaper head and `ssm_out`.**
    - Hypothesis: Q4_0 everywhere (or a Q8_0 head) saves 2–5 % over Unsloth Q4_0.
    - Setup: on another machine, `llama-quantize --imatrix <imatrix> --output-tensor-type q4_0 --tensor-type ssm_out=q4_0` from BF16, or with `--allow-requantize` from Q8_0. Keep `blk.64` (MTP) intact. Run experiments 2 and 3 again.
    - Measure: tg, served tok/s with MTP, perplexity against Unsloth Q4_0.
    - Expected: +2–5 %; quality change unknown.
    - Effort: 4–6 h.
    - Risk: low (quality).

16. **Pruned 20B as a speed/quality point.**
    - Hypothesis: Qwen3.8-20B-Minitron decodes ~1.4× faster than the 27B at the same quant, and its quality loss is measurable.
    - Setup: mradermacher Q4_0 or Q4_K_M GGUF. Run llama-bench, a perplexity check, and a small task check (a few coding prompts) against the 27B. Test whether `draft-mtp` still works and what acceptance it gets.
    - Measure: tg, perplexity, task results, MTP acceptance.
    - Expected: ~5.5 tok/s at Q4_0; noticeable quality drop; MTP acceptance lower.
    - Effort: 2–3 h.
    - Risk: low.

17. **Pascal-capable one-shot AllReduce plus TP-4 (fork work).**
    - Hypothesis: replacing the butterfly with one host-staged hop per AllReduce, plus CUDA graphs, makes TP-4 1.2–1.8× faster than layer split.
    - Setup: port key-sh/p100-llama-cpp patches 0003 (internal AllReduce below Volta) and 0006 (more than 2 devices) to current master, plus #27721. Test 4 cards with Q4_0, then with MTP.
    - Measure: tg, pp, per-AllReduce time (nsys on a short run if host RAM allows).
    - Expected: highly uncertain, 0.6–1.8×.
    - Effort: 1–3 days.
    - Risk: high; unmaintained code path.

18. **ik_llama.cpp route.**
    - Hypothesis: if ik builds and runs on this non-AVX host, `-sm graph --max-gpu 2` (TP-2 × PP-2) and IQ4_KS can beat mainline layer split.
    - Setup: the feasibility gate first ([engines_llamacpp_family.md](engines_llamacpp_family.md), experiment 12). Then ubergarm's IQ4_KS (15.75 GiB) with `-sm graph --max-gpu 2` and `-mtp`.
    - Measure: tg, served tok/s, GPU placement of every node (`GGML_SCHED_DEBUG`).
    - Expected: unknown.
    - Effort: medium–high.
    - Risk: may not run at all on this CPU.

19. **Activation-sparsity prototype (research).**
    - Hypothesis: TEAL-style 40–50 % FFN input sparsity on transposed Q4_0 `ffn_down` (plus row-skipping on gate/up) gives >1.2× on 5-SM cards.
    - Setup: offline thresholds from a calibration set; a transposed `ffn_down` copy in VRAM; a custom CUDA sparse axpy kernel and a graph change in a llama.cpp fork. Start from TEAL's thresholds and PowerInfer's Q4_0 axpy kernel as references.
    - Measure: kernel time against MMVQ on the rig, end-to-end tg, perplexity.
    - Expected: ceiling 1.37–1.5×; on an issue-bound card possibly no gain.
    - Effort: 1–2 weeks.
    - Risk: high.
