# tqLLM ("tqllm") and the TurboQuant formats behind it: identification, comparison with Strata, and what transfers to 9 × P106-090 (sm_61, 5 SMs, PCIe 1.0 x1, no P2P), as of 2026-10-22

Scope: what the name "tqllm" refers to, how that project works, how it differs from Strata and llama.cpp, and whether the engine or any idea in it can be used on this rig: 9 × P106-090 (compute capability 6.1, 6 GB, 5 SMs, ~173 GB/s, FP16 at 1/64 rate, dp4a, no tensor cores), PCIe 1.0 x1 per card without peer-to-peer, a 2-core Celeron without AVX, 4 GB of host RAM, CUDA 12.9 and driver 580. The earlier Strata and ninfer findings are in [engines_other.md](engines_other.md) and [engine_infer.md](engine_infer.md). All sources were read on 2026-10-22 unless a date is given. Conventions:
- "Claim to test" marks anything not confirmed by a primary source or not measured on this rig.
- "(my calc)" marks this note's own arithmetic, including tensor types and sizes read from GGUF headers with HTTP range requests (no tensor data downloaded).
- "(my SASS count)" marks instruction counts from compiling the decode dot-product functions, copied from llama.cpp master and from the TQ3_4S fork with only the type definitions adapted, into a minimal one-row matrix-vector loop with CUDA 13.4 for sm_75. CUDA 13 cannot target sm_61, so Turing serves as the closest proxy: it has the same integer instructions used here (LOP3, SHF, PRMT, IDP4A), but the numbers are relative costs, not Pascal cycle counts.
- "P" and "S" in source lists mean primary (repository, code, model card, paper) and secondary (blog, aggregator, social media).
- tqLLM source links point to `main` at commit `5763f51` (release 1.2.2); fork links point to `main` at `ecc7678`; llama.cpp links point to master at `970b51c`.

The rig's facts are in the [hardware spec](../../mining_rig.md); measured numbers are in [docs/benchmarks/](../../benchmarks/). Nothing in this note was run on the rig.

**Starting point** (measured on the rig, see the [single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md), [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md), [speed-ups](../../benchmarks/2026-10-22-speedups.md) and [Flint](../../benchmarks/2026-10-22-flint.md) pages):
- Decode on one card is limited by unpacking, not memory: Llama-2-7B Q4_0 reaches 73 GB/s (42 % of the measured peak), Q4_K_M 60 GB/s, Q8_0 93 GB/s.
- Qwen3.6-35B-A3B, bartowski Q4_0 (20.8 GB): 19.6 tok/s with CUDA graphs (51 ms per token); 27.7 tok/s served with MTP and backend sampling. Its KV cache is ~20 KiB per token, so 262,144 tokens take ~5 GiB in f16.
- Flint Q4_1: ~28 tok/s.
- Qwen3.8-27B Q4_0 (dense): 4.33 tok/s plain (231 ms per token), 7.5 tok/s with MTP and two drafted tokens. A 2-token step costs 1.33× a 1-token step and an 8-token step 2.69×.
- Two copies of Qwen3.6 on cards 0–4 (f16 KV, 262,144 tokens) and cards 5–8 (q8_0 KV, 131,072 tokens) gave ~31–35 tok/s combined.

**Headline findings** (each is sourced below):
1. **"tqllm" is almost certainly tqLLM** (github.com/turbo-tan/tqllm, site tqllm.com), confidence ~85 %. It is a small, new, from-scratch C++/CUDA inference engine (Apache-2.0, 30 stars, repository created 2026-10-02, first release 1.0.0 on 2026-10-14, release 1.2.2 on 2026-10-22). It serves Qwen3.5/3.8 dense models, `qwen35moe` MoE models and Qwen3.8-Flash-Next (125B MoE) from **one** NVIDIA GPU of the RTX 30 series or newer, or a GB10. "TQ" stands for TurboQuant: the engine is built around rotation-plus-codebook weight and KV formats (TQ3_4S, TQ3, TQ4, tq3 KV) that follow Google Research's TurboQuant (ICLR 2026). A companion llama.cpp fork (llama.cpp-tq3, 235 stars) and a Hugging Face account with TQ3_4S GGUFs come from the same maintainer. The remaining ~15 % covers the wider TurboQuant KV-cache ecosystem (the turboquant_plus project and its llama.cpp fork, which LocalAI packages as a `llama-cpp-tq` backend) and the TurboQuant method itself; both are covered in Q3.
2. **tqLLM is not a quantization method, but it ships several.** TQ3_4S (4.0 bits per weight, GGUF type 46) uses 32-weight blocks with a 32-point Walsh-Hadamard rotation, an 8-level codebook and four 8-bit scales. TQ3 (3.125 bpw) and TQ4 (4.125 bpw) rotate 128-weight blocks and store one FP16 scale per block, using type ids 300 and 301 that only tqLLM reads. The KV formats include tq3 (3.5 bits per value), turbo2/3/4, trellis-coded tcq2/tcq3, q8 and q4. Around them sit an automatic memory fit, TqCache (the oldest KV positions in host RAM), exact DFlash2 tree verification, native MTP and multi-user batched verification.
3. **tqLLM competes with Strata on Strata's own showcase and borrows from it.** Both run Qwen3.8-Flash-Next on one consumer GPU by keeping the routed experts in host RAM with a GPU expert cache and an AVX2 CPU path for misses. tqLLM's code credits several Strata designs: its Gated DeltaNet kernel, fused IQ expert kernel structure, per-32 activation quantization, dense FP16 cache, prompt-guided expert-bank refill and route-tail skip. It does not include Strata code. On the same RTX 3090 and GGUF, the maintainer reports 1.06× / 1.34× / 1.63× Strata 0.2.2's decode speed after 32K / 64K / 128K prompts and about 3× faster first token on short chats (claims to test). Unlike Strata, tqLLM has no multi-GPU mode and no Pascal build.
4. **The tqLLM engine cannot run on this rig, and the compiler stops it first.** Its shared device headers and attention and expert kernels use `mma.sync` (m16n8k16 and m16n8k32), `ldmatrix` and `cp.async` in inline PTX without architecture guards, and these need sm_75/sm_80. After that it would still fail on every other axis. It has no multi-GPU mode, so each model would have to fit one 6 GB card: its own release files start at 11.85 GB, and the models this rig runs today are 16–26 GB. Its MoE and Flash-Next modes need tens of GB of pinned host RAM and an AVX2+FMA+F16C CPU, and TqCache streams KV over PCIe, which here is ~190 MB/s.
5. **TQ3_4S weights through the llama.cpp fork are the only TurboQuant-style weight path with Pascal-compatible code, and they are likely slower than Q4_0 here.** The fork's TQ3_4S decode uses dp4a and `__byte_perm`, and it folds the rotation into the activation quantization kernel, so it adds no graph nodes. But its unpacking costs ~149 instructions per 32 weights, against 59 for Q4_0, 78 for Q4_K and 113 for IQ4_NL (my SASS count). On these 5-SM cards decode time follows instruction count (Q4_K: 1.33× the instructions, measured 1.30× slower), so TQ3_4S is predicted to be ~2× slower than Q4_0 per weight despite 11 % fewer bytes. The fork's own PR #90 says TQ3_4S decode is "unpack-bound" even on an RTX 3090. Predictions: the published Qwen3.6-35B-A3B TQ3_4S file ~13–17 tok/s against 19.6, the dense 27B ~2 tok/s against 4.3 (claims to test).
6. **The TurboQuant idea that pays off here is already in mainline llama.cpp.** Since PR #21038 (merged 2026-04-01), llama.cpp rotates Q, K and V with a Hadamard matrix whenever the KV cache is quantized. Upstream rejected separate TurboQuant KV types (PR #21089, closed 2026-06-02), and the turboquant_plus project itself says rotation plus q4_0 approximates its turbo4 format. On this rig a rotated q4_0 KV cache stores Qwen3.6's 262,144-token context in ~1.4 GiB instead of ~5 GiB. That could give a four-card copy ~245K tokens of context instead of 131K. The costs: four extra matmul kernels per attention layer (estimated ~1 % of Qwen3.6's token time and ~5 % of Flint's), plus the sm_61 flash-attention paths for quantized KV (the VEC kernel for 1–2-token steps; the TILE kernel with a full f16 copy of K and V for MTP verification steps of 3 or more tokens).
7. **The fork-only 3-bit KV types (tq3_0, turbo3) add little on this rig.** They store ~12 % (3-bit V only) to ~29 % (3-bit K and V) more context than rotated q4_0, and they run on the same VEC kernel with heavier unpacking.
8. **Three smaller tqLLM ideas could transfer, each worth a few percent:**
   - A 4-bit LM head for ranking with int8 rescoring of the top 16. The cheap proxy, a Q4_0 head instead of the Q6_K one, saves ~2 ms of Qwen3.6's 51 ms token.
   - Running the shared expert inside the routed-expert launches, which tqLLM measured at +6 % on an RTX 3090. The rig would need a CUDA patch, with an upper bound around +4 %.
   - The "2-bit experts, 4-bit dense" recipe of the Qwen3.6 TQ3_4S file. Mainline llama.cpp can reproduce it with Q2_K/Q3_K experts and Q4_0 dense tensors at ~13.4 GB, which fits on three cards.
9. **A side finding for other notes:** on the same proxy IQ4_NL compiles to ~1.9× Q4_0's instructions per weight. On these unpack-bound cards it may decode well below Q4_0's speed, which matters wherever IQ4_NL is suggested as a rescale-safe or quality format (claim to test in the same single-card run as TQ3_4S).

### Option summary

Expected effects are this note's estimates (claims to test) unless marked as measured.

| Option | Status (2026-10-22) | Expected effect on this rig | Effort |
|---|---|---|---|
| Run tqLLM itself | Images for sm_86/89/120/121 only; source needs sm_75/80 instructions; single GPU | Does not build for sm_61; does not fit even if ported | — |
| Port tqLLM to sm_61 | Would need SIMT replacements for every tensor-core kernel plus a multi-GPU split | Weeks of work; Ampere-tuned design unlikely to beat llama.cpp here (the ninfer gfx906 port reached 0.52–0.66× llama.cpp) | Very high |
| TQ3_4S weights via the llama.cpp-tq3 fork | Fork rebased on upstream 2026-09-24; TQ3_4S MMVQ uses dp4a/PRMT; no Pascal testing | ~0.4–0.6× Q4_0 decode per weight; Qwen3.6 TQ3_4S file ~13–17 tok/s, dense 27B ~2 tok/s | Medium (fork build) |
| Rotated q4_0 KV in mainline (`-ctk q4_0 -ctv q4_0`) | Mainline; rotation on by default for quantized KV | Qwen3.6 262K context in ~1.4 GiB; a four-card copy at ~245K instead of 131K; decode speed vs f16 depends on the sm_61 FA path | Low |
| `LLAMA_ATTN_ROT_DISABLE=1` with q8_0 KV | Mainline env var | Removes 4 kernels per attention layer: ~+1 % Qwen3.6, ~+3–6 % Flint, with q8_0 quality unchanged per PR #21038's KLD data | Very low |
| tq3_0 / turbo3 KV cache (forks) | Fork-only types; VEC kernel only | ~12 % (3-bit V) to ~29 % (3-bit K and V) more context than rotated q4_0; slower unpack | Medium |
| Q4_0 LM head (proxy for tqLLM's 4-bit ranking head) | `llama-quantize --output-tensor-type` | +2–4 % on Qwen3.6; top-1 agreement to measure | Low |
| Shared expert inside the routed-expert kernel | tqLLM-only; would be a ggml-cuda fusion | ≤ +4 % on Qwen3.6 | High |
| Mainline copy of the "2-bit experts, Q4_0 dense" recipe | `llama-quantize --tensor-type` | ~13.4 GB, fits 3 cards; ~17–19 tok/s; quality cost unknown | Medium |
| DFlash2 tree verification | In mainline llama.cpp as `draft-dflash`; covered in [dense_27b_speedups.md](dense_27b_speedups.md) | 7–8-token verify blocks are what these cards do worst | — |
| TqCache / host-RAM experts | tqLLM-only | Not applicable: 4 GB RAM and ~190 MB/s per link | — |

---

## Q1. Which project is "tqllm"?

### Takeaway
The exact name belongs to **tqLLM** (stylised "tqLLM", "TQLLM" in its NOTICE): github.com/turbo-tan/tqllm, with a website at tqllm.com and Docker images `ghcr.io/turbo-tan/tqllm` and `ytan2000/tqllm`. Confidence ~85 %: it is the only project with that name, and its name, formats and documentation all tie it to TurboQuant. It is a niche project with 30 stars and visibility mainly through X posts and a fork's blog post, so a reader who heard "tqllm" may also have meant the broader TurboQuant family in llama.cpp forks (~10 %) or the TurboQuant method itself (~5 %). No project named "tq-llm", "tqllm.cpp", "tq_llm" or a paper named "TQ-LLM" was found.

### Cited Findings
**tqLLM (confidence ~85 %)**
- Self-description: "C++/CUDA inference for supported Qwen models and TurboQuant GGUF weights, with an OpenAI-compatible HTTP server" — [tqllm README](https://github.com/turbo-tan/tqllm) (P).
- Website: "a C++/CUDA inference engine for open-weight models and TurboQuant GGUF weights"; Apache-2.0; "One Docker image, an OpenAI-compatible API, and speculative decoding that never changes the answer"; RTX 30/40/50 and DGX Spark; macOS "coming soon" — [tqllm.com](https://tqllm.com/) (P).
- Repository facts (GitHub API): Apache-2.0, 30 stars, 5 forks, created 2026-10-02, last push 2026-10-22, 104 commits in the visible history (oldest 2026-10-11), no GitHub releases. Releases are tracked as Docker tags and in the changelog: 1.0.0 on 2026-10-14 ("First public release"), then 1.2.0, 1.2.1 and 1.2.2 on 2026-10-22 — [GitHub API: turbo-tan/tqllm](https://github.com/turbo-tan/tqllm); [CHANGELOG.md](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).
- The same maintainer publishes the llama.cpp fork **llama.cpp-tq3**: "llama.cpp fork with TQ3_1S/4S CUDA kernels — 3.5-bit WHT quantization achieving Q4s quality at 10% smaller size". MIT, 235 stars, 19 forks, created 2026-03-30, last push 2026-10-13 — [GitHub turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) (P).
- The matching Hugging Face account (YTan2000) holds 27 model repos. TQ3_1S/TQ3_4S GGUFs run from 2026-03-30 (Qwen3.5-27B) to Qwen3.6-35B-A3B (2026-04-16), Qwen3.8-27B (2026-08-14; 2,718 downloads) and GLM-5.3-Flash (2026-09-17). The tqLLM-only `Qwen3.8-27B-TQ` repo (2026-10-08; 3,325 downloads) and `Qwen3.8-Flash-Next-tqLLM` (2026-10-22) follow — [Hugging Face API, author YTan2000](https://huggingface.co/YTan2000) (P).
- A community fork, **azllm**, "set up to run Qwen3.8-27B as a daily driver for coding agents on a single consumer GPU". It changes only the serving layer and reports TQ_L decode of 118–211 tok/s on an RTX 3090 Ti and 233–333 on an RTX 5090 (2026-10-14). 10 stars — [GitHub xhinker/azllm](https://github.com/xhinker/azllm) (S). A blog post by the fork's author is titled "An Inference Engine Nobody Has Heard Of, Runs Qwen3.8–27B at 207 tok/s on One RTX 3090 Ti, Full 262k Context" (page returned HTTP 403; title from search results) — [Medium](https://xhinker.medium.com/an-inference-engine-nobody-has-heard-of-and-it-runs-qwen3-8-27b-a76dc6df3a36) (S).
- A Chinese-language X post says someone wrote their own engine, named tqllm, running Qwen3.8-27B TQ3_4S-v2 at 273 tok/s on an RTX 3090 (page returned HTTP 402; content from search results) — [X post](https://x.com/lxfater/status/2106710930226585958) (S).

**The wider TurboQuant family (~10 %)**
- **turboquant_plus**: a research repo implementing TurboQuant for KV caches, with turbo2/3/4 formats. 7,020 stars, created 2026-03-25 — [GitHub TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) (P).
- Its llama.cpp fork **llama-cpp-turboquant** (2,435 stars, last push 2026-10-22) — [GitHub](https://github.com/TheTom/llama-cpp-turboquant) (P). LocalAI added it as a backend called `llama-cpp-tq` ("feat(turboquant.cpp): add new backend", 2026-04-03) — [LocalAI commit mirror](https://gitea.varghacsongor.hu/GithubMirror/LocalAI/commit/6e11f882f79b4dbb47754f332e4eba1812b4fb91) (S).
- Other "TQ" tooling found: PyPI packages `tqkit` (TurboQuant+ benchmarking) and `tqai` (TurboQuant KV compression for PyTorch/MLX) — [tqkit](https://pypi.org/project/tqkit/), [tqai](https://pypi.org/project/tqai/) (S).

**The method (~5 %)**
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate, arXiv 2504.19874 (submitted 2025-04-28), ICLR 2026 poster — [arXiv](https://arxiv.org/abs/2504.19874) (P); Google Research blog, 2026-03-24 — [research.google](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (P).

**Not found:** searches for "tq-llm", "tqllm.cpp", "tq_llm", "TQ LLM" and "TQ-LLM" on GitHub, arXiv and the web returned no other project (2026-10-22).

### Inferences
- The two plausible meanings lead to the same practical answer for this rig. The engine is unusable here (Q5), and what transfers is the TurboQuant lineage of rotation-based quantization (Q3).
- tqLLM is a one-maintainer project less than a month old with daily releases, so its interfaces, file formats and numbers are still moving.

### Gaps
- No Reddit or Hacker News discussion of tqLLM was found, so there is no independent reproduction of its numbers.

---

## Q2. What tqLLM does technically

### Takeaway
tqLLM is a single-GPU engine tuned for Ampere and newer. It has its own TurboQuant-derived weight formats (TQ3_4S, TQ3, TQ4) and also reads most GGUF block formats natively. It offers a menu of compressed KV caches, exact speculative decoding (DFlash2 trees, native MTP, prompt lookup) with greedy output identical to plain decoding, a memory fit that picks the KV format and context at start-up, and TqCache to spill old KV positions to host RAM. For Qwen3.8-Flash-Next it keeps 51 GB of routed experts in pinned host RAM, caches experts on the GPU, and computes misses on the CPU with AVX2. Hardware support is RTX 30/40/50 (sm_86/89/120) and GB10 (sm_121). There is no multi-GPU support of any kind.

### Cited Findings
**Models and formats**
- Supported: "Qwen3.5 / Qwen3.8 dense models (hybrid attention + Gated DeltaNet) and qwen35moe mixture-of-experts models such as MiniCPM-V 4.7 35B-A3B", plus Qwen3.8-Flash-Next via a manifest. Tensor types: "TQ3, TQ4, TQ3_4S, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q4_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q2_K, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, F16 and BF16"; "IQ1_M does not load yet" — [quickstart.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/quickstart.md) (P).
- Weight formats: "All three rotate the weights before quantising (the TurboQuant idea: a rotation spreads outliers so a small codebook fits every block)" — [formats.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/formats.md) (P):

  | Format | Block | Scales | Bits per weight | Runs in |
  |---|---|---|---:|---|
  | TQ3_4S (GGUF type 46) | 32 weights, 32-point rotation, 8-level code | four 8-bit scales | 4.0 | tqLLM and llama.cpp-tq3 |
  | TQ3 (type 301) | 128 weights, 128-point rotation, 8-level code | one FP16 per 128 | 3.125 | tqLLM only |
  | TQ4 (type 300) | 128 weights, 128-point rotation, 16-level code | one FP16 per 128 | 4.125 | tqLLM only |

- Rotated-basis caveat: in a TQ3/TQ4 model, "rotate the input of every matrix that shares an input with a TQ matrix, whatever its storage type". The BF16 `ssm_alpha`/`ssm_beta` matrices are stored pre-rotated — [formats.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/formats.md) (P).
- Quality of the tqLLM-only 27B files (KLD against BF16): TQ_XS 11.85 GB 0.079, TQ_S 12.55 GB 0.055, TQ_M 13.14 GB 0.045, TQ_L 14.72 GB 0.022 — [HF YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) (P, maintainer-measured).

**KV cache and long context**
- `--kv FORMAT`, default bf16; "tq3 = 3-bit, trellis tcq3 / tcq2, q8, q4, splits" — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P).
- Code: tq3 is "TQ3_0 K and V: 8 x 32-value blocks per 256-dim row, fp16 RMS each, 3-bit rotated codebook (112 B per head per token)", i.e. 3.5 bits per value, and requires head_dim 256 — [kv_tq3.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/formats/kv/kv_tq3.cpp) (P). The split codecs are tq3, tq2s (2-bit, 80 B), q8 (rotated int8, 272 B), tcq2 (trellis 2-bit, 72 B), tcq3 (trellis 3-bit, 104 B) and q4 (rotated 4-bit, 144 B) — [kv_split.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/formats/kv/kv_split.cpp) (P). One turbo variant is labelled "fails the free-running quality gate" — [kv_turbo.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/formats/kv/kv_turbo.cpp) (P).
- TqCache: "Ask for more context than the GPU holds and the memory fit keeps the oldest KV positions in system RAM (exact…)". On an RTX 5060 Ti a 200K prompt decodes at 113 tok/s with all KV on the GPU and at 70 tok/s when 81,920 positions are read from RAM — [long-context.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/long-context.md) (P).
- `--kv-compact R` compresses old KV R-fold ("not exact") — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P).

**Speculative decoding**
- DFlash2 drafter (`-md`, up to 15 drafts; "hybrid Qwen3.5 / 3.8 models verify at most 8 columns, so 7 is the effective maximum"), token-tree verification by default, `--mtp` with the model's own head, and `--lookup` prompt-lookup drafts — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P).
- "Plain decoding and every verify path (tree, chain, MTP) compute attention the same way, so greedy output with a drafter equals greedy output without one" — [ARCHITECTURE.md](https://github.com/turbo-tan/tqllm/blob/main/docs/ARCHITECTURE.md) (P).
- Target-head shortlist: "The default greedy verifier ranks the full vocabulary with a 4-bit head, then re-scores its top 16 candidates with an int8 head… This is an empirical speed optimisation, not a certified full-vocabulary argmax" — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P).

**Memory placement and MoE**
- Memory fit order: "token embeddings and the LM head to host memory, smaller prefill buffers, a smaller KV format…, prefill weight blocks, the KV tail to system RAM, then a shorter context" — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P).
- qwen35moe: `--moe-host-mb N` "keeps the least-used experts in system RAM when the file is larger than the GPU"; `--moe-cache-mb C` keeps recently used host experts in a VRAM cache managed inside the decode graph — [quickstart.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/quickstart.md); [CHANGELOG 1.2.0](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).
- "MoE decode runs the shared expert inside the routed expert launches. MiniCPM-V 4.7 35B-A3B on one RTX 3090: IQ3_S 207.8 -> 220.1 tok/s, Q5_K_M … 176.5 -> 187.1 tok/s" — [CHANGELOG 1.2.0](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).
- Flash-Next: "The routed experts live in system RAM; the GPU keeps a cache of expert slots, misses go over PCIe by the copy engine and the next layer's experts are prefetched while the current one runs" — [BENCHMARKS.md](https://github.com/turbo-tan/tqllm/blob/main/docs/BENCHMARKS.md) (P). It needs "a 24 GB NVIDIA GPU (RTX 30 or newer) or a GB10 / DGX Spark, 64 GB of system memory or more (51 GB of experts live there…)". The model is the ISTA-DASLab GSQ-RCO IQ3_S GGUF (two shards, 54.8 + 28.8 GB) plus a 1.46 GB MTP layer and an optional 50.9 GB packed-expert file — [HF YTan2000/Qwen3.8-Flash-Next-tqLLM](https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM); [HF ISTA-DASLab IQ3_S tree](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (P).
- Flash-Next CPU miss pool: "x86 AVX2: missed experts computed on the CPU from the pinned packed copy". Functions are compiled with `__attribute__((target("avx2,fma")))` and `"avx2,fma,f16c"`, with no runtime CPU check found — [fn_cpu_experts.hpp](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/fn_cpu_experts.hpp) (P). The generic CPU expert matvec is "x86-64 = AVX2+FMA (original). aarch64 = NEON" — [cpu_expert_matvec.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/experts/cpu_expert_matvec.cpp) (P).

**GPU requirements**
- "One image tag covers x86-64 PCs with RTX 30, 40 or 50 series GPUs (CUDA architectures 86, 89, 120) and arm64 GB10 / DGX Spark systems (121)". Native build: CMake ≥ 3.25, "CUDA 12.9 and 13.0 were tested on RTX 3090" — [quickstart.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/quickstart.md) (P).
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native`; CMakeLists has no architecture whitelist — [CMakeLists.txt](https://github.com/turbo-tan/tqllm/blob/main/CMakeLists.txt) (P).
- Tensor-core and Ampere instructions in inline PTX without `__CUDA_ARCH__` guards: `mma.sync.aligned.m16n8k16` (f16 and f32 accumulate) and `mma.sync.aligned.m16n8k32…s8` and `ldmatrix.sync.aligned.m8n8.x4` in the shared helper header — [common.cuh](https://github.com/turbo-tan/tqllm/blob/main/src/core/common.cuh); `cp.async.cg.shared.global` in the attention kernels — [attention.cu](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen35/attention.cu); `cp.async` in the INT8 expert GEMM — [quant.cu](https://github.com/turbo-tan/tqllm/blob/main/src/formats/weight/quant.cu) (P). Only the Flash-Next Gated DeltaNet kernel wraps `cp.async` in `__CUDA_ARCH__ >= 800` — [qwen4exp.cu](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/qwen4exp.cu) (P).
- Runtime gates: the safetensors Qwen2 path throws "Qwen BF16 execution requires compute capability 8.0 or newer" — [model.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/engine/model.cpp); INT8 GEMM is supported only at CC ≥ 8.0 — [int8_gemm.cu](https://github.com/turbo-tan/tqllm/blob/main/src/core/int8_gemm.cu) (P).
- Multi-GPU: the source has no `cudaSetDevice` call and the docs never mention more than one GPU. Several users share one GPU through `--np N` ("requests served at once (batched verify)") — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md) (P; code search of the cloned repo).

**Activity, maturity, license**
- Apache-2.0. Third-party code: llama.cpp's Jinja engine, ggml's IQ grids and IQ4_NL codebook, stb_image — [NOTICE](https://github.com/turbo-tan/tqllm/blob/main/NOTICE) (P). Also llama.cpp's MMQ files at commit `fb4b273` (MIT), copied with modified tile loaders; "Idea of launching MMQ directly per expert group from the prefill path: Strata" — [third_party/ggml_mmq/NOTICE](https://github.com/turbo-tan/tqllm/blob/main/third_party/ggml_mmq/NOTICE) (P).
- The MoE lane's own status file says: "Pending gates: same-file standard llama.cpp KLD on Wikitext-2 (100 x 512), sane text, and matched no-spec speed… Do not promote this lane or quote performance before those gates pass" — [MOE_STATUS.md](https://github.com/turbo-tan/tqllm/blob/main/docs/MOE_STATUS.md) (P).
- Release 1.2.1 and 1.2.2 fixed, among others, prefill in chunks of 8 tokens ("a 32K prompt took 263 s instead of 15 s") and failing GB10 requests on the day of release — [CHANGELOG](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).

**Published performance** (all maintainer-measured; claims to test)
- Qwen3.8-27B TQ3_4S-v2 + DFlash2-TQ3 drafter, tq3 KV, RTX 3090: 55.5 / 54.5 / 50.6 tok/s without the drafter and 155.2 / 149.9 / 131.0 with it, at 2K / 8K / 32K prompts; prefill 2,155–2,426 tok/s — [BENCHMARKS.md](https://github.com/turbo-tan/tqllm/blob/main/docs/BENCHMARKS.md) (P).
- LocalMaxxing prompts, TQ_XS reasoning / code: RTX 3090 271.3 / 149.8, RTX 5060 Ti 175.9 / 98.8, GB10 99.7 / 67.6 tok/s — [HF YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) (P).
- MoE without a drafter: MiniCPM-V 4.7 IQ3_S on an RTX 3090, tqLLM 202.8 against llama.cpp 136.4 tok/s; Qwen3.6-35B-A3B UD-Q4_K_XL with MTP 246.2–329.6 tok/s (177.5–182.8 without) — [tqllm.com](https://tqllm.com/); [CHANGELOG 1.2.0](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).
- Flash-Next against Strata: see Q4.

### Inferences
- The feature list targets three things this rig does not have: tensor cores for cheap wide verification, a single large GPU, and a host with tens of GB of RAM and AVX2.
- The design choices that do matter here are the formats (Q3) and kernel-count reductions such as the shared-expert fusion, because per-kernel overhead is about a third of the token time on these cards (see [flint_speedups.md](flint_speedups.md), section 0).

### Gaps
- No tqLLM numbers on any card below sm_86, and no independent benchmark of any tqLLM claim.
- How tqLLM uses the second Flash-Next shard (28.8 GB, the per-layer/n-gram table): disk, page cache or RAM. Not read in code.

---

## Q3. The TurboQuant method and its llama.cpp descendants

### Takeaway
TurboQuant is a **data-oblivious vector quantizer**. It applies a random rotation so each coordinate follows a known, concentrated distribution, then a precomputed scalar codebook per coordinate. A 1-bit QJL stage on the residual makes inner products unbiased. The paper's target is the KV cache and vector search, not weights. In llama.cpp the idea split three ways:
1. **Rotation alone, merged into mainline** (PR #21038, 2026-04-01): Q, K and V are rotated with a Hadamard matrix whenever the KV cache is quantized, which makes q4_0 KV much more accurate. On by default.
2. **New KV types** (tbq3_0/tbq4_0, turbo2/3/4, tq3_0, trellis tcq), **rejected upstream** (PR #21089 closed 2026-06-02) and living in forks.
3. **Rotated weight formats** (TQ3_1S, TQ3_4S) in the llama.cpp-tq3 fork. They claim Q4-class quality at ~10 % smaller size (maintainer PPL only).

Production implementations drop the QJL stage.

### Cited Findings
**The method**
- "randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates… applying optimal scalar quantizers per each coordinate"; a two-stage design with "a 1-bit Quantized JL (QJL) transform on the residual" for unbiased inner products. Distortion within a factor of ~2.7 of the lower bound. KV cache: "absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel" — [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (P, 2025-04-28).
- The Google blog adds the companion PolarQuant (random rotation, polar coordinates) and QJL ("zero-overhead, 1-bit trick"). It claims KV memory reduced "by a factor of at least 6x" on needle-in-a-haystack, and up to 8× faster 4-bit attention logits on H100 against a JAX 32-bit baseline. It does not mention weight quantization — [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (P, 2026-03-24).
- turboquant_plus: "Extract the norm, apply a WHT with random sign flips, then quantize with Lloyd-Max centroids". turbo4 4.25 bits, turbo3 3.5 bits (3.125 at block 128), turbo2 2.5 bits. QJL is kept "as a reference implementation only" because it "adds variance that softmax turns into attention noise". Wikitext-2 PPL against q8_0 (6.111, Apple M5 Max): turbo4 +0.23 %, q4_0 +0.52 %, turbo3 +1.06 %, turbo2 +6.48 %. K precision dominates quality, so the README suggests a higher-precision K with a compressed V for some models. It says rotation plus q4_0 approximates turbo4 and the full codec is not upstream — [turboquant_plus README](https://github.com/TheTom/turboquant_plus) (P, last push 2026-07-20).

**Mainline llama.cpp**
- PR #21038 "llama : rotate activations for better quantization", merged 2026-04-01. It rotates Q, K and V before they enter the cache and rotates the attention output back, described as "backend agnostic". Qwen3.5-9B q4_0/q4_0 KV: mean KLD ~0.0080 → ~0.0048. gpt-oss-20b AIME25 with q4_0 KV: 2.0 % → 21.7 % (f16 37.9 %). q8_0/q8_0 roughly unchanged. On CUDA (Qwen3 30B-A3B Q4_0) token generation dropped ~7–12 % — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) (P).
- Current code: rotation for K is on when `ggml_is_quantized(type_k)` and the head size is a multiple of 64 (rotation size grows to the head size); V uses 64 × 64 blocks. `LLAMA_ATTN_ROT_DISABLE` turns it off — [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp) (P). Each rotation is one `ggml_mul_mat` with a Hadamard matrix between two reshapes, applied to Q, K and V before the cache write and to the attention output after it, also for sliding-window caches — [llama-impl.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-impl.h); [llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp) (P).
- PR #21089 (CPU-only `tbq3_0` ~3.06 bits and `tbq4_0` ~4.06 bits) was closed on 2026-06-02 without merging. On Qwen3.5-4B, tbq4_0 KLD was 0.00960 against q4_0's 0.00912, tbq3_0 0.02647, and CPU generation dropped from ~15.7 to ~8.3 tok/s. A maintainer wrote that "so far no one has presented evidence that would justify the opportunity cost of adding TurboQuant support" — [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (P).
- A status article (August 2026): vLLM merged a TurboQuant attention backend (PR #38479, v0.20.0) but concluded "FP8 remains the best default"; 3-bit keys cost about 20 points on reasoning benchmarks; Ollama's TurboQuant PR closed 2026-05-30 — [SOTAAZ status check](https://sotaaz.com/post/turboquant-status-2026-en) (S).
- Mainline also loads "prism.hadamard" folded-weight GGUFs (normalized Sylvester-Walsh-Hadamard blocks folded into weights, with the activation rotated at run time by one extra matmul per folded weight) for LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE and QWEN3NEXT — [llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) (P).

**The llama.cpp-tq3 fork (TQ3_4S weights, TQ3_0 and turbo KV)**
- Types: `GGML_TYPE_TQ3_1S = 44` ("TurboQuant 3-bit with two half-block scales"), `GGML_TYPE_TQ3_4S = 46` ("four u8 per-8 scales (4.0 bpw)"), KV types `TQ3_0 = 200`, `TURBO3_0 = 201`, `TURBO4_0 = 202`, `TURBO2_0 = 203`. The fork numbers `Q2_0 = 41` and `Q1_0 = 42`, while mainline numbers them `Q1_0 = 41` and `Q2_0 = 42` — [fork ggml.h](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/ggml/include/ggml.h); [mainline ggml.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h) (P).
- TQ3_4S decode dot product: one 16-byte load per block, `__funnelshift_r` to extract two 24-bit index groups, a selector built from shifts and masks, then `__byte_perm` to map four 3-bit indices to int8 levels {−127, −79, −45, −14, 14, 45, 79, 127}, then `dp4a`. Scales are an E3M5 byte decoded with bit operations — [fork vecdotq.cuh](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/ggml/src/ggml-cuda/vecdotq.cuh) (P).
- Activation rotation: "TQ3_4S activations get the Walsh-Hadamard rotation fused into quantization": a 32-point butterfly with `__shfl_xor_sync` inside `quantize_q8_1` — [fork quantize.cu](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/ggml/src/ggml-cuda/quantize.cu) (P).
- TQ3_4S MMQ has a dp4a tile path for GPUs without MMA. Blackwell can map TQ3_4S to FP4 tiles for prefill — [fork mmq.cuh](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/ggml/src/ggml-cuda/mmq.cuh); [fork README](https://github.com/turbo-tan/llama.cpp-tq3) (P).
- The fork's PR #90 says: "On Ampere TQ3_4S decode is unpack-bound, so a 3-token MTP verify batch cost 1.55x a single token". Decoding once per block for multi-column steps cut a 3-token step on an RTX 3090 from 36.2 to 25.7 ms (1 token: 23.0 ms) — [fork PR #90](https://github.com/turbo-tan/llama.cpp-tq3/pull/90) (P, 2026-09-25).
- Quantized KV in the fork's flash attention: TQ3_0 and turbo types have VEC-kernel cases (q8_0/q4_0 K with tq3_0 V, turbo3/turbo4 K with f16/q8_0/turbo V). A fused turbo4 MMA path requires Turing MMA. PR #76 "force VEC FA kernel for all tq3_0 KV combos" — [fork fattn.cu](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/ggml/src/ggml-cuda/fattn.cu); [fork PR #76](https://github.com/turbo-tan/llama.cpp-tq3/pull/76) (P).
- Fork base: "fork: rebuild clean snapshot on verified upstream" (2026-09-24, upstream #29396); last commit 2026-10-12 — [fork history](https://github.com/turbo-tan/llama.cpp-tq3/commits/main) (P).
- Quality claims (maintainer, PPL only): Qwen3.5-27B wiki.test c=2048, TQ3_4S 6.8224 (12.9 GiB) against Q3_K_S 6.8630 (11.4 GiB) — [HF Qwen3.5-27B-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.5-27B-TQ3_4S). TQ3_1S 7.2570 (12.9 GB) against Q4_0 7.2431 (14.4 GB) at c=512 — [HF Qwen3.5-27B-TQ3_1S](https://huggingface.co/YTan2000/Qwen3.5-27B-TQ3_1S) (P).

**Published TQ3_4S files relevant to this rig** (header contents read with HTTP range requests, my calc)
- `Qwen3.6-35B-A3B-TQ3_4S.gguf`, 13.3 GB: routed gate/up experts **Q2_K** (7.05 GB), routed down **Q3_K** (4.61 GB), attention, Gated DeltaNet projections and shared expert **TQ3_4S** (0.71 GB), output head and embeddings Q6_K, router F32. The card says "2-bit experts, 4-bit attention" and needs llama.cpp-tq3; on an RTX 5060 Ti it reports TG128 107 tok/s — [HF Qwen3.6-35B-A3B-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-TQ3_4S) (P).
- The MTP variant (13.98 GB) scored 74/86 against UD-Q3_K_M's 73/86 on the maintainer's coding checks, both run with `-ctk q4_0 -ctv tq3_0` — [HF Qwen3.6-35B-A3B-MTP-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S) (P).
- `Qwen3.8-27B-TQ3_4S-v2.gguf`, 13.8 GB (4.24 BPW): TQ3_4S throughout with a Q6_K head and embeddings; 64.78 tok/s with MTP on an RTX 3090 — [HF Qwen3.8-27B-TQ3_4S](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ3_4S) (P).
- For comparison, the rig's bartowski Qwen3.6 Q4_0 (20.8 GB): routed experts Q4_0 (Q4_1 for five down projections), attention and Gated DeltaNet mostly Q4_0 with Q8_0 K/V and some Q6_K `attn_q`, shared expert Q8_0/Q5_0, output head Q6_K (0.417 GB), embeddings Q4_0 — [HF bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) (P; header read).

### Inferences
- **Bytes per token** (my calc from the headers, routed experts at 8 of 256): the TQ3_4S Qwen3.6 file reads ~1.57 GB per token (routed 0.36, dense 0.79 including the 84 MB F32 router, head 0.42) against ~2.00 GB for the Q4_0 file (routed 0.57, dense 1.02, head 0.42). The saving comes almost entirely from the Q2_K/Q3_K experts. The TQ3_4S tensors are 0.71 GB; at Q4_0 they would be 0.79 GB.
- **KV bytes per token for Qwen3.6** (10 attention layers × 2 KV heads × 256, my calc): f16 20.0 KiB, q8_0 10.6, q4_0 5.6, q4_0 K + tq3_0 V 5.0, tq3 K and V 4.4. At 262,144 tokens: 5.0 / 2.66 / 1.41 / 1.25 / 1.09 GiB. For Qwen3.8-27B (16 layers × 4 KV heads × 256): 64 KiB per token in f16, so 16 GiB at 262,144 tokens; 4.5 GiB with q4_0.
- The rejection of new types and the KLD numbers in PR #21089 and turboquant_plus agree: at ~4 bits, rotation plus q4_0 equals turbo4/tbq4_0, and the TurboQuant-specific gain is only at ~3 bits and below, where quality starts to drop.

### Gaps
- No KLD for TQ3_4S against Q4_0 or Q4_K_M on any model from a common reference. Only PPL and small task suites, all maintainer-run.
- Whether the fork builds and runs correctly for sm_61 (it has no architecture gate, but nobody has reported a Pascal run).

---

## Q4. tqLLM against Strata (and llama.cpp)

### Takeaway
Strata and tqLLM solve the same headline problem: a 125B MoE (Qwen3.8-Flash-Next) on one consumer PC by tiering experts across GPU, RAM and the CPU. tqLLM's repository appeared eight days after Strata's (2026-10-02 against 2026-09-24) and its first release three weeks after, although its formats and llama.cpp fork date from March 2026. It reimplements several Strata kernel and placement designs (credited in code), adds its own formats, dense-model speculative decoding and multi-user serving, and benchmarks itself against Strata on the same GGUF. Strata is the more general and far more popular engine. It has a ggml base, a multi-GPU layer split without P2P, and an experimental Pascal build. tqLLM is narrower and newer, single-GPU, and Ampere-and-newer only. llama.cpp remains the only one of the three that runs on this rig today.

### Comparison

| | tqLLM | Strata | llama.cpp (as used on this rig) |
|---|---|---|---|
| What it is | Single-GPU C++/CUDA engine for Qwen3.5/3.8 dense, qwen35moe and Flash-Next, with TurboQuant formats | Local engine for Qwen3.8-Flash-Next and Qwen3.6-35B-A3B that splits a model across GPU, RAM and SSD | General GGUF engine |
| Code base | From scratch; copies llama.cpp's MMQ, Jinja and IQ tables; Strata designs reimplemented | "Built with parts of llama.cpp / ggml (MIT)" | ggml |
| License, popularity | Apache-2.0; 30 stars; first release 2026-10-14 | MIT; ~30k stars; v0.2.2 (engine 0.1.47) | MIT |
| Weight formats | TQ3_4S, TQ3, TQ4 plus most GGUF types | GGUF (Flash-Next quants Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Qwen3.6 UD-Q4_K_XL) | All GGUF types |
| KV formats and placement | bf16, q8, q4, tq3, turbo2/3/4, tcq2/tcq3, K/V splits; TqCache keeps the oldest positions in RAM | KV in VRAM; from 64K only its most-read part, the rest streams from RAM | f16, q8_0, q4_0 and others, with Hadamard rotation when quantized |
| MoE placement | Flash-Next: 51 GB of experts in pinned RAM, GPU expert cache, copy-engine misses, AVX2 CPU miss pool; qwen35moe: least-used experts in RAM | All experts pinned in RAM (experts + ~10 GB); uncached experts computed on the CPU with AVX2/AVX-512; 29 GB n-gram shard on SSD | Whatever `-ngl`/`-ot`/`-ncmoe` place; on this rig everything sits in VRAM across 9 cards |
| Host requirements | 64 GB RAM for Flash-Next; AVX2+FMA+F16C for CPU experts | RAM ≥ experts + ~10 GB; AVX2/AVX-512 | Runs on this rig's 2 cores and 4 GB with full GPU offload |
| Multi-GPU | None | Layer split, no P2P needed, x1 slots work; auto placement ≤ 4 cards; cards ≤ 8 GB start Off | Layer split over all 9 cards without P2P |
| Minimum GPU | sm_86 images; source needs sm_75/sm_80 instructions | CC 7.5 in Limits; experimental sm_60/61 CUDA 12.9 build; P40 community reports 37–38 tok/s at IQ3_S | sm_61 supported (CUDA graphs need a patch) |
| Speculative decoding | DFlash2 trees, MTP with adaptive length, prompt lookup; exact greedy | The model's MTP layer drafts up to 3 tokens, plus prompt lookup | MTP, DFlash/DFlash2, DSpark, EAGLE-3, n-gram |
| Serving | OpenAI, Anthropic Messages and Responses APIs; `--np` up to 8; built-in monitor | Server and CLI | llama-server, slots |

### Cited Findings
- tqLLM against Strata 0.2.2 (engine 0.1.47, `--spec 4`), same RTX 3090 24 GB and same ISTA-DASLab GSQ-RCO IQ3_S GGUF, greedy: short chats 91.6 against 89.6 tok/s; time to first token on a short chat 0.32–0.41 s against 1.07–1.29 s; at 32K / 64K / 128K, 14.98 / 28.01 / 56.53 s against 15.14 / 29.18 / 58.25 s; decode after 32K / 64K / 128K 76.4 / 85.1 / 77.1 against 72.2 / 63.6 / 47.2 tok/s — [BENCHMARKS.md](https://github.com/turbo-tan/tqllm/blob/main/docs/BENCHMARKS.md) (P, maintainer-measured).
- An earlier comparison against Strata 0.1.45 (same file, RTX 3090): first token 14.09 vs 15.12 s at 32K, decode after the prompt 82.0 / 80.1 / 66.0 / 54.1 vs 69.5 / 68.5 / 53.8 / 48.1 tok/s at 32K / 64K / 128K / 245K. On an IQ4_XS file, tqLLM matched Strata at 61.1 vs 61.2 tok/s — [CHANGELOG 1.2.0](https://github.com/turbo-tan/tqllm/blob/main/CHANGELOG.md) (P).
- Strata designs credited in tqLLM code:
  - The Gated DeltaNet prefill kernel: "Design after Strata's gdn_rec_kh_kernel (MIT…); own code on our layouts" — [qwen4exp.cu](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/qwen4exp.cu).
  - "Activations quantised per 32-value block (Strata's scheme)" and "Structure after Strata's fused IQ kernel (moe_fused_iq.cu, credited; own implementation)" — [quant.cu](https://github.com/turbo-tan/tqllm/blob/main/src/formats/weight/quant.cu).
  - "dense FP16 cache (Strata's dense path, --dense-cache-mb)" — [fn_checkpoint.hpp](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/fn_checkpoint.hpp).
  - Expert banks "filled with the prompt's most-routed experts (Strata does the same)" — [fn_engine.hpp](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/fn_engine.hpp).
  - "TQFN_TAIL_SKIP=R (APPROXIMATE…; idea: Strata v0.1.42 route-tail-skip)" — [fn_layer_forward.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/fn_layer_forward.cpp).

  All P.
- Credits list Strata only as a "comparison baseline", alongside Unsloth, HyperQwen, ExLlamaV3, vLLM and LocalMaxxing — [CREDITS.md](https://github.com/turbo-tan/tqllm/blob/main/docs/CREDITS.md) (P).
- Strata facts (Pascal build, multi-GPU limits, memory model, sizes) are from [engines_other.md](engines_other.md), Q4, which cites Strata's OLDER_GPUS.md, MULTI_GPU.md, HOW_IT_WORKS.md and MODELS.md.
- Strata's VRAM holds "the MTP draft layer, the KV cache (from 64K: only its most-read part, the rest streams from RAM), and an expert cache"; "the model's own MTP layer drafts up to 3 tokens", with prompt lookup for repeated text; "the 28.8 GB n-gram table, read a few rows per token through the OS cache" stays on the SSD — [Strata HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md) (P). Repository created 2026-09-24, 30.5k stars — [GitHub API: Niko1221/Strata](https://github.com/Niko1221/Strata) (P).

### Inferences
- tqLLM builds on Strata in ideas, not in code. The two share the GGUF ecosystem (both read ISTA-DASLab's GSQ-RCO files), and both depend on llama.cpp/ggml heritage: Strata through bundled ggml code, tqLLM through its copied MMQ and IQ tables.
- For this rig Strata remains the less distant of the two. Its experimental sm_61 build and no-P2P layer split exist; its blockers are RAM, the CPU, card count and card size (see [engines_other.md](engines_other.md)). tqLLM's blockers start at the GPU architecture and include the lack of any multi-GPU path.
- The Strata comparison is the maintainer's, on one GPU type, with the same GGUF on both engines. The long-context gap (1.63× at 128K) comes from tqLLM's own long-prompt attention and expert-bank logic, not from its TQ formats, which the Flash-Next path does not use.

### Gaps
- No third-party benchmark of tqLLM against Strata.
- Strata's KV-cache precision (f16 or quantized) is not stated in its README or HOW_IT_WORKS.md.

---

## Q5. Applicability to the rig

### Takeaway
- **The engine:** unusable. It does not compile for sm_61. Even ported, it could not spread a model over nine 6 GB cards, and its offload modes assume RAM, CPU features and PCIe bandwidth the rig lacks.
- **What transfers:**
  - Rotation-before-quantization for the KV cache, already in mainline and usable today, mostly as a context-capacity lever with a small kernel-count cost.
  - A few kernel-count and head-size ideas worth a few percent each.
- **What does not transfer:** TQ3_4S weights. They are the only TurboQuant weight format with Pascal-compatible code, but on unpack-bound 5-SM cards their extra unpack work outweighs the 11 % byte saving.

### Cited Findings
**What blocks the engine, in order**
1. **Build:** unguarded `mma.sync` m16n8k16/m16n8k32, `ldmatrix` and `cp.async` PTX in [common.cuh](https://github.com/turbo-tan/tqllm/blob/main/src/core/common.cuh), [attention.cu](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen35/attention.cu) and [quant.cu](https://github.com/turbo-tan/tqllm/blob/main/src/formats/weight/quant.cu) (P). Per the PTX ISA (general CUDA knowledge, not from tqLLM docs), `ldmatrix` and `mma.m16n8k8` need sm_75, while `mma.m16n8k16` (f16), `m16n8k32` (s8) and `cp.async` need sm_80, so ptxas rejects these kernels for sm_61. The safetensors path also refuses CC < 8.0 at run time — [model.cpp](https://github.com/turbo-tan/tqllm/blob/main/src/engine/model.cpp) (P).
2. **One GPU only:** no multi-GPU code (no `cudaSetDevice`). tqLLM's own release files start at 11.85 GB (27B TQ_XS) — [HF Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) (P). A 6 GB card could hold only small Qwen3.5 dense models (under ~9B at 4 bits), using one card of nine. The models this rig targets (Qwen3.6-35B-A3B, Qwen3.8-27B, Flint, Flash-Next) do not fit.
3. **Offload paths:** qwen35moe host experts and Flash-Next need pinned host RAM (51 GB of experts for Flash-Next; 64 GB minimum), and the CPU miss pool is AVX2+FMA(+F16C) only — [fn_cpu_experts.hpp](https://github.com/turbo-tan/tqllm/blob/main/src/models/qwen/qwen4exp/fn_cpu_experts.hpp) (P). The rig has 4 GB of RAM (pinned buffers above ~1 GB trigger the OOM killer), no AVX, and ~190 MB/s per link.
4. **TqCache:** "Reading that part over PCIe costs decode speed" — [long-context.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/long-context.md) (P). At ~190 MB/s, even 100 MB of host-resident KV per token would add ~0.5 s per token (my calc).
5. **Kernel design:** prefill uses int8/FP8 tensor-core GEMMs on RTX 30+, attention and verification use BF16 MMA, and FP16 half2 helpers would run at 1/64 rate on these cards — [options.md](https://github.com/turbo-tan/tqllm/blob/main/docs/reference/options.md); [common.cuh](https://github.com/turbo-tan/tqllm/blob/main/src/core/common.cuh) (P). The only no-matrix-core port of a comparable engine (ninfer on gfx906) reached "0.52-0.66x of llama.cpp's speed" and was abandoned — [engine_infer.md](engine_infer.md) (P via that note).

**Decode unpack cost per format** (my SASS count, sm_75 proxy, one-row loop; instructions per 32 weights)

| Format | Bits per weight | Instructions per 32 weights | Relative to Q4_0 | Main instructions |
|---|---:|---:|---:|---|
| Q4_0 | 4.5 | 59 | 1.00 | 18 LDG, 8 LOP3, 8 IDP4A |
| Q8_0 | 8.5 | 60 | 1.02 | 26 LDG, 8 IDP4A |
| Q4_K | 4.5 | 78 | 1.33 | 16 IDP4A, 14 LDG, 10 LOP3, 8 I2F |
| Q2_K | 2.625 | 78 | 1.33 | 16 IDP4A, 12 LOP3, 10 SHF |
| Q6_K | 6.5625 | 90 | 1.52 | 23 LDG, 16 LOP3, 16 IDP4A |
| Q3_K | 3.4375 | 90 | 1.53 | 20 LOP3, 16 IDP4A, 12 SHF |
| IQ4_NL | 4.5 | 113 | 1.92 | 36 PRMT, 18 LDG, 16 LOP3 |
| TQ3_4S | 4.0 | 149 | 2.53 | 44 LOP3, 33 IMAD, 23 SHF, 9 PRMT, 10 LDG |

- Calibration against the rig: Llama-2-7B Q4_K_M generated 1.30× slower than Q4_0 on one card for 6.5 % more bytes, close to the 1.33–1.40× instruction ratio of its Q4_K/Q6_K mix. Q8_0 (1.02× the instructions, 1.89× the bytes) was 1.47× slower, so it meets a memory-side limit at ~93 GB/s instead — [single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md) (measured); ratios my calc.
- Context from other Pascal cards: a 30-SM P40 decodes IQ4_NL (57.45 tok/s) as fast as Q4_0 (58.82) at batch 1 — [PR #20831 table in dense_27b_speedups.md](dense_27b_speedups.md) (P via that note). The P106-090 has ~3× the bytes per SM-cycle of a P40 (173 GB/s over 5 SMs against 347 GB/s over 30, my calc), which is why instruction count shows up here and not there.

**Flash-attention path for quantized KV on sm_61** (llama.cpp master)
- Without tensor cores, f16 KV uses the GQA-aware TILE kernel. Quantized K or V uses the VEC kernel when the step has ≤ 2 query tokens, and otherwise the TILE kernel — [fattn.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu) (P). The TILE launch requests f16 K and V (`launch_fattn(…, true, true, …)`), and `launch_fattn` then converts the quantized K/V views to a temporary f16 buffer on every call — [fattn-tile.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-tile.cuh); [fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh) (P).
- The VEC kernel does not reuse K/V across GQA heads; open PR #30792 routes q8_0 into the TILE kernel natively — see [flint_speedups.md](flint_speedups.md), item 9 and section 4.

### Inferences
**TQ3_4S on this rig** (my calc; effective bandwidth taken as proportional to bytes per instruction from Q4_0's measured 73 GB/s, capped near Q8_0's 93 GB/s)
- Predicted effective bandwidth: TQ3_4S ~26 GB/s, Q2_K ~32, Q3_K ~36, IQ4_NL ~38, Q4_K ~55, Q6_K ~70, Q4_0 73.
- **Qwen3.6-35B-A3B:** the Q4_0 file's weight reads come to ~27 ms of the measured 51 ms token, which leaves ~24 ms of per-kernel overhead, attention, recurrent-state work and card hops. The TQ3_4S file's weight reads come to ~45 ms (Q2_K/Q3_K experts ~11 ms, TQ3_4S dense ~27 ms, Q6_K head ~6 ms, F32 router ~1 ms). Predicted ~69 ms per token, **~14.5 tok/s**; a range of 13–17 allows for the ALU model being 30 % pessimistic. This is a claim to test. It still fits on three cards (13.3 GB).
- **Qwen3.8-27B dense:** ~11.7 GB of TQ3_4S weights at ~26 GB/s is ~450 ms per token against ~205 ms of weight reads for Q4_0. Predicted **~2 tok/s plain** against 4.33. MTP verification would also cost more: TQ3_4S multi-column steps were unpack-bound even on an RTX 3090 (fork PR #90).
- **The recipe is the useful part.** The 13.3 GB size comes from Q2_K/Q3_K experts. A mainline file with the same expert types and Q4_0 dense tensors would be ~13.4 GB. Its predicted decode is ~17–19 tok/s (experts +3 ms, dense −1 ms against the Q4_0 file). It needs no fork, fits on three cards, and loads ~35 % faster. The quality cost of 2-bit experts is unknown.

**Rotated KV cache in mainline**
- **Capacity:** at the memory q8_0 uses for 131,072 tokens (~1.33 GiB), q4_0 holds ~245,000 tokens of Qwen3.6 context (my calc). A four-card copy could therefore run near the full 262,144, though the card holding the most attention layers sets the limit. Combined with an f16 copy on cards 0–4, both copies would run at (nearly) full native context.
- **Kernel cost:** four Hadamard matmuls per attention layer per step (Q, K, V and the output inverse). At the measured ~10 µs per graph node that is ~0.4 ms per token for Qwen3.6 (10 attention layers, ~1 % of 51 ms), ~0.6 ms for Qwen3.8-27B (16 layers, ~0.3 %) and ~2 ms for Flint (49 layers including sliding-window ones, ~5 % of 36 ms). All claims to test.
- **`LLAMA_ATTN_ROT_DISABLE=1`** removes those kernels. PR #21038's KLD data shows rotation barely matters for q8_0/q8_0, so with a q8_0 cache disabling it should be free. With q4_0 it should stay on, since rotation is what makes q4_0 KV usable.
- **MTP interaction:** with MTP and two drafts, every verification step has 3 query tokens. With a quantized cache it takes the TILE kernel, which converts the whole K/V view to f16 first. At 131,072 tokens that is ~2.6 GB of f16 writes per step spread over the attention cards, and could cost more than the cache saves (my calc; claim to test). Plain decoding (1 token) and MTP with a single draft (2-token steps) stay on the VEC kernel.
- **Fork-only 3-bit KV** (q4_0 K + tq3_0 V): 5.0 KiB against 5.6 KiB per token, ~12 % more context than rotated q4_0 (~29 % with 3-bit K as well, at a larger quality risk, since K precision dominates quality per turboquant_plus), on the same kernels with heavier unpacking. Not worth a fork build for this alone.

**Smaller ideas from tqLLM**
- **Cheaper LM head:** the head is ~12 % of Qwen3.6's token time in this model (Q6_K, 0.417 GB, ~6 ms). A Q4_0 head (0.286 GB, ~3.9 ms) saves ~2 ms (+4 %). The MTP draft step shares the head, so served speed with MTP gains a little more. tqLLM keeps exact-ish greedy output by rescoring the top 16 with an int8 head. llama.cpp has no such op, so the cheap proxy is a lossy Q4_0 head whose top-1 agreement must be measured. A Q8_0 head (0.54 GB at ~93 GB/s, ~5.8 ms) is about speed-neutral and more accurate than Q6_K.
- **Shared-expert fusion:** Qwen3.6 has a shared expert in all 40 layers. If fusing it into the routed launch removed ~5 kernels per layer, that is ~200 kernels or ~2 ms per token, an upper bound of +4 %. It needs a ggml-cuda fusion pass and may overlap with existing MMVQ gate/up fusion.
- **Exact tree verification and DFlash2:** not useful here. A 7-draft tree needs 8-wide verification, and an 8-token step already costs 2.69× (dense) and 3.82× (Qwen3.6) a 1-token step on these cards. See [dense_27b_speedups.md](dense_27b_speedups.md).

### Gaps
- No Pascal measurement of TQ3_4S, IQ4_NL, Q2_K or Q3_K decode. The predictions above rest on a Turing instruction-count proxy calibrated with one Q4_K_M data point.
- No measurement of the rotation kernels' cost or of quantized-KV decode with MTP on sm_61.
- Whether the llama.cpp-tq3 fork compiles for sm_61. Its TQ3_4S and FA code paths look architecture-neutral, but it carries much new CUDA code that has never been built for Pascal.

---

## Experiment ideas

Ranked by expected gain per hour of effort. Each expected outcome is a claim to test, not a prediction to rely on.

**Conventions for every experiment:**
- Record the build commit and patches, file checksums, cards used, context size, KV types, environment variables and flags.
- Use llama-bench for raw speed and [scripts/server_bench.sh](../../../scripts/server_bench.sh) for served speed; [scripts/parallel_bench.sh](../../../scripts/parallel_bench.sh) for two-copy or multi-slot runs.
- Measure quality off the rig with [scripts/quality_kld.sh](../../../scripts/quality_kld.sh) against a Q8_0 (or BF16) reference with an f16 KV cache, and report mean KLD and same-top-token rate.
- Build any fork off the rig with CUDA 12.9 for sm_61 (see [ops_and_benchmarking.md](ops_and_benchmarking.md)) and remove test binaries from the rig afterwards.
- Watch `MemAvailable` on the 4 GB host and keep `-ub 128` at long context.

1. **Cost of the mainline KV rotation with a q8_0 cache.**
   - Hypothesis: with a quantized KV cache llama.cpp adds four Hadamard matmuls per attention layer. Turning them off with a q8_0 cache gains ~1 % on Qwen3.6 and ~3–6 % on Flint at no quality cost.
   - Setup:
     - Current CUDA-graphs build. `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 3 -d 0,32768` with `-ctk q8_0 -ctv q8_0`, once with and once without `LLAMA_ATTN_ROT_DISABLE=1`, plus an f16 baseline.
     - Qwen3.6 Q4_0 on 5 cards and Flint Q4_1 on 5 cards.
     - Read the `n_rot_k`/`n_rot_v` lines in the log to confirm the rotation state.
     - Off the rig: KLD of q8_0 KV with and without rotation against f16 KV.
   - Measure: tg128 at both depths, KLD, graph node count from a verbose run.
   - Expected: +0.5–2 % (Qwen3.6) and +3–6 % (Flint) with rotation off; KLD change < 0.001.
   - Effort: 0.5–1 h.
   - Risk: none. Keep rotation on for q4_0 caches.

2. **Rotated q4_0 KV for a full-context four-card copy, and its interaction with MTP.**
   - Hypothesis: q4_0/q4_0 KV lets the cards 5–8 copy of Qwen3.6 run at ~245K–262K context instead of 131K. Decode speed matches q8_0 for 1- and 2-token steps, but a quantized cache loses to f16 at long context for 3-token steps (MTP with two drafts), because those take the TILE kernel with a full f16 conversion of K/V.
   - Setup:
     - `llama-server` on cards 5–8 with `-ctk q4_0 -ctv q4_0 -c 262144` (fall back to 229,376 or 196,608 if it does not fit), `-bs`.
     - Step cost by width and depth: `llama-bench -ngl 99 -fa on -p 1,2,3 -n 0 -d 0,32768,131072 -r 2` for f16, q8_0 and q4_0 KV. The 2- and 3-token rows stand in for MTP verify steps with one and two drafts.
     - Served runs with `--spec-type draft-mtp --spec-draft-n-max 1` and `2` on a long-context prompt (fill the context with a document, then ask for 256 tokens).
     - Off the rig: KLD of q4_0 KV (rotation on) against f16 KV.
   - Measure: largest context that loads, per-card VRAM, step time by width, depth and KV type, served tok/s, KLD.
   - Expected: ~245K fits; 1- and 2-token steps with q4_0 within ±10 % of q8_0; 3-token steps at 131K with a quantized cache 10–30 % slower than with f16; KLD ~0.005.
   - Effort: 1–2 h.
   - Risk: low. OOM on the fullest card; the 4 GB host if both copies run at once.

3. **Cheaper LM head (proxy for tqLLM's 4-bit ranking head).**
   - Hypothesis: replacing Qwen3.6's Q6_K `output.weight` with Q4_0 gains +2–4 % decode and slightly more with MTP; Q8_0 is speed-neutral and more accurate.
   - Setup:
     - Off the rig, requantize from a Q8_0/BF16 source with the same per-tensor types as the bartowski Q4_0 file except `--output-tensor-type q4_0` (and a second file with `q8_0`), using the repo's [scripts/quantize_docker.sh](../../../scripts/quantize_docker.sh).
     - On the rig: llama-bench tg128 on 5 cards, and the server benchmark with MTP and `-bs`.
   - Measure: tok/s, KLD and same-top-token rate against the Q8_0 reference.
   - Expected: Q4_0 head +2–4 % with top-1 agreement dropping by ≤ 1 point; Q8_0 head ±1 %.
   - Effort: 1–2 h (quantize and copy dominate).
   - Risk: low.

4. **Single-card format calibration including TQ3_4S and IQ4_NL.**
   - Hypothesis: on one P106-090, decode time per weight follows the instruction counts in Q5: TQ3_4S ~0.4–0.6× Q4_0's tok/s, IQ4_NL ~0.5–0.7×, Q2_K and Q3_K slower than Q4_0 despite fewer bytes.
   - Setup:
     - Build turbo-tan/llama.cpp-tq3 off the rig with `-DCMAKE_CUDA_ARCHITECTURES=61` and CUDA 12.9. A build failure is itself the answer for the fork.
     - From a Llama-2-7B F16 source (as in the [calibration](../../benchmarks/2026-10-22-single-card-calibration.md)), quantize Q4_0, IQ4_NL, Q2_K, Q3_K_S and TQ3_4S with the fork's `llama-quantize`.
     - Run `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` on one card. Run Q4_0 with both the fork and the mainline build to check that the fork's base performs the same.
     - Optionally `-p 1,2,3,4` for the multi-column cost.
   - Measure: tg128, pp512, effective GB/s per format; core clock and power-limit samples.
   - Expected: TQ3_4S 8–12 tok/s against Q4_0's ~19.5; IQ4_NL 10–14.
   - Effort: 2–3 h.
   - Risk: low. The fork may not build or may produce wrong output on sm_61; check perplexity on a few chunks.
   - Use: rules TQ3_4S in or out for good, and settles the IQ4_NL question raised in [flint_speedups.md](flint_speedups.md) and [dense_27b_speedups.md](dense_27b_speedups.md).

5. **A mainline copy of the "2-bit experts, Q4_0 dense" recipe for Qwen3.6.**
   - Hypothesis: Q2_K gate/up experts, Q3_K down experts and Q4_0 for everything else gives a ~13.4 GB file that decodes at ~17–19 tok/s on three cards, with a quality loss to quantify.
   - Setup:
     - Off the rig, with an imatrix, from Unsloth's Q8_0 or BF16: `--tensor-type 'ffn_(gate|up)_exps=q2_k' --tensor-type 'ffn_down_exps=q3_k'`, Q4_0 as the base type, and `--output-tensor-type q6_k` (or q4_0 per experiment 3).
     - On the rig: llama-bench on 3 and 5 cards; server benchmark.
     - If it fits, try three copies on 3 cards each, watching host RAM.
   - Measure: tok/s, file size, load time, KLD and top-1 against Q8_0, and the same against the published TQ3_4S file if experiment 4 shows the fork works.
   - Expected: 17–19 tok/s per copy; KLD several times the Q4_0 file's.
   - Effort: 3–4 h.
   - Risk: medium. 2-bit experts may be too lossy for coding and tool calls; three llama-server processes may not fit in 4 GB of RAM.

6. **Fork-only 3-bit V cache.**
   - Hypothesis: q4_0 K + tq3_0 V in the fork gives ~12 % more context than rotated q4_0/q4_0 at similar speed.
   - Setup: only if experiment 4 shows a working sm_61 fork build. Repeat experiment 2's capacity and depth runs with `-ctk q4_0 -ctv tq3_0`.
   - Measure: largest context, tok/s by depth, KLD.
   - Expected: ~+12 % context, speed within ±10 % of q4_0/q4_0.
   - Effort: 1–2 h after experiment 4.
   - Risk: low, but the gain is small and ties the rig to a fork that renumbers mainline types.

7. **Shared expert inside the routed-expert kernel.**
   - Hypothesis: fusing Qwen3.6's shared-expert matvecs and gate into the routed MMVQ launch removes ~150–200 kernels per token, +2–4 %.
   - Setup:
     - First count the shared-expert nodes per layer in a verbose graph dump of the current build, to bound the gain before writing code.
     - Then prototype a ggml-cuda fusion (MUL_MAT_ID plus the shared MUL_MATs and the sigmoid gate) and compare it with the stock build with `test-backend-ops` and llama-bench.
   - Measure: node count, tg128, output equality.
   - Expected: +2–4 % if more than ~3 kernels per layer can be removed; nothing if existing fusions already cover them.
   - Effort: 1–3 days.
   - Risk: medium (correctness across MoE paths).

**Not worth running:** building tqLLM itself for sm_61. Its source fails at the PTX level, and a port would mean rewriting every tensor-core kernel and adding multi-GPU support, for a design whose speed comes from tensor-core verification (see Q5).
