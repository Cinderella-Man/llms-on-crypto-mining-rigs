# Flint's speed and quality tricks on RTX 3090 cards: what transfers from nine P106-090s, and how much (2026-10-24)

Scope: which of this project's tricks for Flint (40B-A2.5B MoE), developed for nine P106-090 mining cards, would make
Flint faster or better in llama.cpp on NVIDIA GeForce RTX 3090 cards (Ampere, compute capability 8.6), and by how much.
Three setups: one 3090; two to four 3090s as a layer split over PCIe without NVLink or peer-to-peer access (the same
topology as the rig, with faster links and a faster host); one 3090 with routed experts in system RAM. For each trick
the note says what limits a Flint token on a 3090, whether the trick transfers, the expected gain, what code or setting it
needs, the prior art, and how a 3090 owner can measure it. A sibling note covers the V100
([transfer to V100](transfer_v100.md)).

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it. "(est., sim.)" is a result of
  the project's pipeline simulator or n-gram replay on Flint's own transcripts with 3090 stage times assumed here:
  measured offline, an estimate for any real card.
- "P" and "S" in the source list mean primary (code, PRs, NVIDIA documents, measurements in this repository or in a
  repository that states its settings) and secondary.
- Code citations are `branch:path:line` in the fork's worktrees on 2026-10-24: `v2` = `pascal-p106-v2` (`032b130f4`),
  `comb` = `pascal-p106-v2-combined` (`b3052a4f8`), `pers` = `pascal-p106-v2-persistent` (`8de13900f`, not pushed),
  `pipe2` = `pascal-p106-v2-pipespec2`, `tool` = `pascal-p106-v2-toolsample` (not pushed). `ggml-cuda/` stands for
  `ggml/src/ggml-cuda/`. The upstream base is `d77256999`.
- "UD" is Unsloth's UD-Q4_K_XL, the quality bar. "T" is the time of one plain token. "The rescale" is llama.cpp's MoE
  down-projection rescale for Flint.
- Nothing ran on the rig, the quant machine or any 3090 for this note. On a laptop: GGUF headers read with the standard
  library, and the two CPU tools [pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) and
  [replay.py](../../../bench/ngram_replay/replay.py) run on Q51's transcripts with 3090 timings.

Starting point, measured:
- **The card** (NVIDIA GA102 whitepaper, Table 9): 82 SMs, 24 GB GDDR6X at 936 GB/s, 6 MB L2, 35.6 TFLOPS FP32, 71
  TFLOPS FP16 tensor with FP32 accumulate, 284 TOPS INT8 tensor, PCIe 4.0 x16, NVLink for two cards only, 350 W.
  Without NVLink the stock driver reports no peer-to-peer between two 3090s; a community-patched driver turns it on
  (one 2 × 3090 report, no bandwidth measured).
- **llama.cpp on one 3090** (CUDA scoreboard, discussion #15013): Llama-2-7B Q4_0 tg128 158–178 tok/s, pp512
  5,154–5,974. That is ~0.62–0.68 TB/s of weights read, two thirds of the peak, for a dense model with ~400 kernels per
  token (est.).
- **Flint on NVIDIA cards:** one RTX 5090 (1.79 TB/s), upstream build b11825, UD, 131,072-token f16 cache: 145.7 tok/s
  generation and 3,433 tok/s prompts, from llama-server's timings over agent runs, 27.4 GB of VRAM (atomicagent.io). No
  llama-bench number for Flint on a 3090 is published; one 3090 owner reports 50–60 tok/s on short prompts with
  UD-IQ2_XXS and 20 expert layers in RAM (Unsloth's model discussion, anecdotal).
- **Similar MoE models on one 3090:** Qwen3.6-35B-A3B (256 experts × 512, top 8) 115.7–139.6 tok/s with all weights on
  the card, 89.1 with 10 expert layers in RAM and 84.9 with 12, prompts 3,360 → 1,153 tok/s (llama-server sweeps,
  UD-IQ4_NL_XL); gpt-oss-20b tg128 161.8; gpt-oss-120b with its experts in DDR5-6400 ~30 tok/s.
- **Small batches on a 3090 for an MoE** (PR #31047's table, gemma4 26B-A4B Q4_0, pp8192 on master): 173.8 / 306.9 /
  414.5 / 497.2 tok/s at micro-batches of 1 / 2 / 4 / 8 tokens, so a 2-, 4- and 8-token step costs 1.13, 1.68 and
  2.80 T. On the P106s the same steps cost 2.2–3.6 T (Q19).
- **Flint's files** (tensor sizes read from the GGUF headers; the token embedding stays in host memory in llama.cpp):

  | File | Tensors | On the GPU | Routed experts per MoE layer | GB read per token |
  |---|---|---|---|---|
  | UD-Q4_K_XL (25.05 GB) | 23.33 GiB | 23.03 GiB | 0.457 GiB | 2.21 |
  | Unsloth Q8_0 (42.30 GB) | 39.39 GiB | 39.09 GiB | 0.797 GiB | 2.70 |
  | UD-Q3_K_XL (18.49 GB) | 17.2 GiB | ~16.9 GiB (est.) | ~0.33 GiB (est.) | ~1.8 (est.) |
  | Makers' Q4_1 (25.99 GB) | 24.20 GiB | 23.06 GiB (F32 embedding on the host) | 0.469 GiB | 1.66 |
  | Our K2h-r16 (~30 GB) | ~28 GiB | ~28 GiB | — | 2.09 |

- **KV cache:** 28 KiB per token of context in f16 (14 full-attention layers × 4 KV heads × 128 × K and V), 14.9 KiB in
  q8_0: 0.88 / 1.75 / 3.5 / 7.0 GiB at 32k / 64k / 131k / 262k in f16. The 35 sliding-window layers add 52–88 MiB.
- **Where the rig's v2 decode gain comes from** ([fork scope](fork_scope.md)): of 17.6 → 37.5 tok/s on six cards, 87 %
  of the logarithmic gain is Pascal-only (CUDA graphs on compute 6.1, the 2-warp launch shape, the Pascal kernels),
  12 % the FP32 activation sums that let the rescale go, 1 % a generic fusion fix.

## Verdict

- **On a 3090 a Flint token is mostly fixed per-kernel cost, not weight reading.** UD reads 2.21 GB per token, ~3 ms at
  0.7–0.8 TB/s, while a token runs ~1,900 kernels at ~2.5–3.3 µs each inside a CUDA graph: est. 8.3–10.3 ms per token,
  **est. 95–120 tok/s** if the file fit, with 55–65 % of the token in per-kernel cost (section 1.1). The 5090's 145.7
  tok/s on 1.9× the bandwidth points the same way. This inverts the P106 picture:
  - **the Pascal kernels, 87 % of the rig's gain, do not transfer:** they fix instruction issue on 5-SM cards, and
    llama.cpp's own matrix-vector kernels are not issue-bound on 82 SMs;
  - **the launch cuts transfer, and weigh more than on the rig:** the rescale off (~470 kernels per token), the
    shared-expert fusion fix, the norm writing the Q8_1 activations, and the persistent MoE-layer kernel;
  - **byte savings transfer weakly** (screened head, GPTQ head, F16 router: 0–2 % each, est.), and so do host-side fixes
    (tool-request sampling, the on-card mask, the CPU-free loop: 0–3 %), because PCIe 4.0 x16 and a desktop CPU make
    the copies and the host gap small.
- **Top items for 3090 owners (est.):**
  1. **The rescale off at one-token steps: +11–18 % decode, exact.** `LLAMA_NO_MOE_DOWN_RESCALE=tg` in the fork drops it
     where decode lives. On a 3090 that needs no kernel change for UD: llama.cpp's one-token kernels compute the Q5_K
     block sums in FP32 and IQ4_XS, Q6_K and Q8_0 read none. Unsloth's Q8_0 takes `=1`, prompts included. Our
     rescale-free mixes (Q8_0 down-projections in layers 46–48) were scored with `=1` in batches on non-Pascal GPUs;
     at one token their Q5_0 down-projections of layers 2–45 still read the FP16 sums, so they need a one-token check
     first. The PR that added Flint names the cost: "Auto-scaling leads to slowdown of 10-30%".
  2. **The right file and settings.** One 3090 cannot hold UD with a useful context (23.0 GiB of weights on a 24 GB
     card): keep 2–4 expert layers in RAM (`--n-cpu-moe`, 0.46 GiB each) for 32–64k of context, est. 85–110 tok/s.
     UD-Q3_K_XL fits whole with 128k+ of context (its quality is not scored here). Two 3090s hold Unsloth's Q8_0
     (KLD 0.052 against UD's 0.120) at est. 0.92–0.95× UD's speed with the rescale off on both. The makers' sampling
     (temperature 1.0, top-p 0.95, top-k 20, min-p 0, reasoning kept in the history) applies on any card.
  3. **The fork's generic launch fusions: +4–12 % together.** The shared-expert gate/up/GLU fusion fix (Q50, est.
     +2–5 %) and the norm writing the Q8_1 activations and absorbing the residual add (Q100, est. +2–8 %), both exact,
     both checked on RTX 3090s for correctness, never timed there. `GGML_CUDA_GRAPH_OPT=1` (upstream, one GPU) adds est.
     +3–8 %.
  4. **Several 3090s: pipelined speculation transfers directly.** Simulated on Flint's transcripts with 3090 stage
     times: 1.25× / 1.35× / 1.40× overall on 2 / 3 / 4 cards with the untrained copy-and-table guesser (chat
     1.11–1.17×, edits 1.43–1.77×, file rewrites 2–3.5×), 1.6× / 1.9× / 2.2× with a guesser right 70 % of the time (est.,
     sim.). The executor is bit-identical to plain decoding on two RTX 3090s already; it needs llama-server integration.
  5. **Prompts and copies:** PR #31047 (MoE tiles sized by tokens per expert) est. 1.1–1.35× on prompts at
     `-ub 512`; batched n-gram speculation, which Q19 ruled out on the P106s, pays modestly on a 3090: est. 1.1–1.2× over
     all text, ~1.0× on chat, 1.3–1.5× on edits (est., replay).
  - **The largest code lever is the persistent MoE-layer kernel (Q88):** one launch per MoE layer instead of the FFN's
    ~13–14 kernels with the rescale off, est. +8–20 % once its grid and products are tuned for Ampere. It runs
    bit-exact on an RTX 3090 today (`GGML_CUDA_PERSISTENT_MOE=2`), covers 42 of UD's 47 MoE layers, and was never
    timed off the P106.
  - Together, items 1 and 3 est. ×1.15–1.35 on one-token decode for UD on one card; with the persistent kernel est.
    ×1.25–1.55.
- **Quality transfers as files and settings, not as kernels.** Our K2h-r16 and G3 (KLD 0.105 and 0.104 against UD's
  0.120; level with UD on written tokens at 8k, ~4 % behind on tool-schema tokens; non-inferior on 3,477 BFCL tasks)
  need two 3090s (~28 GiB) and give est. +1–3 % speed over UD there. On two cards Unsloth's Q8_0 is the bigger quality
  step for the same hardware.
- **What to upstream:**
  1. A backend-aware gate for the rescale (skip it where the one-token kernel computes the block sums itself or reads
     none), then FP32 activation sums in the stock Q8_1 readers (MMVQ for Q4_0/Q4_1/Q5_0/Q5_1, MMQ's DS4 layout), so
     the rescale can go everywhere. It closes the graph's own TODO (`v2:src/llama-graph.cpp:2523-2525`) and is worth
     est. +11–18 % to every Flint user on a fast GPU.
  2. The gate/up/GLU fusion fix (Q50): small, generic, any GPU.
  3. The norm writing Q8_1 (Q100), once a 3090 A/B shows ≥ 3 %.
  4. 3090 data with Flint on PR #31047, and pipelined decoding as an RFC for layer splits.
  - **Publish a recipe page** "Flint on RTX 3090": the fit table of section 2, the commands of section 5 and the
    rescale switch, with owners' measurements.
  - **Fork hygiene first:** the fork's 2-warp launch shape edits llama.cpp's generic table, which Ampere uses
    (`v2:ggml-cuda/mmvq.cu:456-461`), so a fork build for a 3090 changes its one-token launch shape, unmeasured. Gate it
    to compute 6.1 before pointing 3090 owners at the fork.

## Ranked table

Gains are decode speed for UD unless stated, against upstream llama.cpp with Flint's support as merged (rescale on),
CUDA graphs on. "1 card" assumes the file fits; "3090 + RAM" is one card with 2–8 expert layers in RAM.

| # | Trick (registry) | Transfers to a 3090? | 1 card (est.) | 2–4 cards, layer split (est.) | 3090 + RAM (est.) | What it needs | Effort | How to verify |
|---|---|---|---|---|---|---|---|---|
| 1 | Rescale off at one-token steps (Q76's graph switch; Q22) | Yes for UD and Unsloth's Q8_0 by construction; our rescale-free mixes after a one-token check; never with Q4_0/Q4_1/Q5_0 down-projections in layers 46–48 (makers' Q4_1) | +11–18 % | +11–18 % per card | +8–14 % | Fork env `LLAMA_NO_MOE_DOWN_RESCALE=tg` (`=1` for Q8_0 and rescale-free mixes); upstream: a backend-aware gate | Hours | Same binary with and without the variable; one-token perplexity bit-identical (§5) |
| 2 | Fit, file and sampling (Q1, Q8, Q58, Q75, rec note) | Yes | UD + `--n-cpu-moe 2–4`; UD-Q3_K_XL whole | Q8_0 at 131–262k on 2 cards; UD at 262k | UD + `--n-cpu-moe` up to ~16 (262k f16) | Settings, upstream | Minutes | `llama-bench -ncmoe`, `nvidia-smi` |
| 3 | Shared-expert gate/up/GLU fusion fix (Q50) | Yes (generic, `v2:ggml-cuda/ggml-cuda.cu:3517`) | +2–5 % | same | +2–4 % | Fork commit `032b130f4`; first upstream PR | Small | Fork against upstream, same file |
| 4 | Norm writes Q8_1 and absorbs the residual add (Q100) | Yes (generic, default on; checked on RTX 3090s) | +2–8 % (rig: within noise) | same | +2–6 % | `comb` build; `GGML_CUDA_NORM_Q8=0` turns it off | Small | Env A/B |
| 5 | `GGML_CUDA_GRAPH_OPT=1` (upstream, not ours; Q34 left it untested) | Yes, one GPU only | +3–8 % | — | +2–6 % | Env | Minutes | Env A/B |
| 6 | Persistent MoE-layer kernel (Q88) | Partly: correct on an RTX 3090, grid scales with SMs, products tuned for compute 6.1, Flint-only, 42 of 47 UD layers | +8–20 % after Ampere tuning; unknown as built | same | +6–15 % | `pers` (not pushed), `GGML_CUDA_PERSISTENT_MOE=2` | Weeks | Env A/B with `GGML_CUDA_PERSISTENT_MOE_CHECK=1` |
| 7 | Pipelined speculation (Q87, Q96, Q104) | Yes (generic; bit-identical on two RTX 3090s) | — | 1.25× / 1.35× / 1.40× overall (sim.), several conversations near S × one | — | `pipe2` (`llama-pipespec`); llama-server integration (1–2 weeks) | Tool today; server weeks | `bench/pipespec/check_identity.sh`, `rig_run.sh` with `GPUS=0,1` |
| 8 | Trained one-token guesser (Q20) | Yes | As a full-depth head with batched verify: 1.2–1.45× | 1.6× / 1.9× / 2.2× at p 0.7 (sim.) | 1.1–1.3× | Training (40–70 GPU-hours, pilot running) | Days | Offline acceptance, then the executor |
| 9 | PR #31047, MoE tiles by tokens per expert (Q108) | Yes (all architectures; its author's 3090: gemma4 26B-A4B 1.39× at micro-batch 128, granitemoe F16 0.80× at 32) | Prompts 1.1–1.35× at `-ub 512`, ~1.05× at 2,048; decode 0 | same | Prompts dominated by expert uploads | Cherry-pick (in `comb`) | Hours | `GGML_CUDA_MMQ_MOE_LEGACY=1` A/B |
| 10 | Batched n-gram speculation (Q53, Q64 re-priced) | Yes: verify is cheap on a 3090 | 1.1–1.2× overall; chat ~1.0, code 1.1, edits 1.3–1.5 (replay) | same per stream | smaller | `--spec-type ngram-map-k` (upstream); the copy-gated drafter is not built | Minutes | Server timings, draft acceptance |
| 11 | First-finish race of 2 samples (Q93) | Yes | Client-side race on `-np 2`: 1.1–1.15× per answer | 1.2× per answer through the executor | ~1.1× | Two requests, cancel the slower at `</think>` | Hours | Time to first `</think>` |
| 12 | Screened output head (Q91, Q92) | Yes (generic engine, side file, exact) | +1–2 % | same | +1 % | `comb` + side file | Hours | `LLAMA_SCREEN_HEAD=1` A/B |
| 13 | GPU sampling for tool requests (Q55, Q67) | Yes, but the 600 KB copy costs ~25 µs on PCIe 4.0 | +1–2 % on tool requests | same | same | `tool` (not pushed) | Days | Same tools request, both paths |
| 14 | Packed uploads, device-fed token (Q120, Q121) | Yes (executor only) | — | +1–3 % single stream; needed for streams | — | `pipe2` | — | `bench/pipespec/rig_hostloop.sh` |
| 15 | On-card mask (Q110, Q116), graph-reuse fix (Q109) | Yes, small on PCIe 4.0 | Decode ~0; prefill +2–8 % at ≥ 64k | Frees up to ~0.75 GiB per card of mask copies at 262k, `-ub 512` | — | `comb`, `gpumask2` | — | `llama-batched-bench` at 131k, VRAM per card |
| 16 | Our files: K2h-r16, G3, GPTQ head, F16 router (Q56, Q73, Q77, Q112, Q113) | As files yes; as speed little | Don't fit | Fit 2 cards: +1–3 % over UD, better 512-token KLD | — | ~30 GB download (not published) | — | KLD runs exist (Q41, Q106) |
| — | Not transferring: the Pascal kernels (Q38, Q46, Q78, Q101), the 2-warp shape (a risk), CUDA graphs on compute 6.1, Q4_1_64, PowerMizer (Q141), the q8_0-KV slowdown (Q54, likely reversed on Ampere), the memory offset (Q133, est. ≤ 2–4 %) | | | | | | | |

## 1. Trick by trick (question 1)

### 1.1 What limits a Flint token on a 3090

UD on one 3090, upstream llama.cpp, short context, CUDA graphs on (est. unless marked):

| Part of a token | Est. ms | Basis |
|---|---|---|
| Weight streaming: 2.21 GB at 0.70–0.80 TB/s | 2.8–3.2 | 75–85 % of 936 GB/s on large shapes, less on the six-expert calls (3–4 MB each); the scoreboard's Llama-2-7B reads ~0.65 TB/s including its kernel overhead |
| ~1,900 kernels at ~2.5–3.3 µs each (dependency gap plus ramp-up and tail of small grids) | 4.8–6.3 | ~40 kernels per MoE layer after llama.cpp's fusions, 10–11 of them the rescale ([Flint speed-ups](flint_speedups.md)); the same fusions run on Ampere (`v2:ggml-cuda/ggml-cuda.cu:3862`, `:4538-4544`, `:4160`) |
| of which the rescale's ~470 | 1.2–1.5 | 11 graph ops per MoE layer (`v2:src/llama-graph.cpp:1685-1722`) |
| Attention at ≤ 4k of context | 0.2–0.4 | 49 flash-attention kernels |
| Host: graph launch, sampling, the CPU embedding row, input copies | 0.15–0.4 | desktop CPU; 600 KB of logits cross PCIe 4.0 in ~25 µs |
| **Total** | **8.3–10.3 (97–120 tok/s)** | |
| Context depth | +1.3–1.9 ms per 32k in f16 (est.) | 0.88 GiB of KV per 32k at 0.5–0.7 TB/s |

- **Check against the 5090.** Its served 145.7 tok/s is 6.9 ms per token; weights at 1.79 TB/s take ~1.5–1.8 ms, which
  leaves ~5.1–5.4 ms of fixed cost on a card with ~1.4× the 3090's clock. Scaled, the 3090 pays ~6–7 ms of fixed cost and
  ~3 ms of weights: 9–10 ms, inside the table's range.
- **Check against Qwen3.6-35B-A3B on a 3090**, a similar MoE: 115.7–139.6 tok/s for ~1.6–1.9 GB per token (est.), i.e.
  ~2 ms of weights at peak inside a 7–8.6 ms token.
- **Consequence.** Removing one kernel saves ~0.03 % of a 3090 token against ~0.007 % of a rig token (26–31 ms, ~1.85 µs
  per graph node, Q100's estimate), so the same launch cut is worth ~4× more here. The rig's main lever, faster
  streaming per byte, acts on a third of a 3090 token, and llama.cpp's kernels already reach most of the bandwidth there.

### 1.2 Each trick

| Trick | On the P106s | Mechanism | On a 3090 | Expected gain (est.): 1 / 2–4 / + RAM | Prior art |
|---|---|---|---|---|---|
| Pascal one-token kernels (Q38, Q46) | +28–39 % decode | Lane groups sized to the row, shared activation loads, staged 128-bit loads: fixes instruction issue on 5 SMs | Gated to compute 6.1 (`v2:ggml-cuda/mmvq-pascal.cu:596`; `GGML_CUDA_MMVQ_PASCAL=2` forces them, used for correctness on sm_86). llama.cpp's generic table runs instead (`v2:ggml-cuda/mmvq.cu:109-147`) | ~0 or negative (est.; unmeasured) | Upstream tunes Ampere itself; ik_llama.cpp has its own kernels |
| 2-warp launch shape (v1) | +13–27 % | 2 warps per block at one token for rows ≥ 2,048 | **Not gated:** edits `MMVQ_PARAMETERS_GENERIC`, which Ampere uses (`v2:ggml-cuda/mmvq.cu:456-461`) | unknown sign; a risk | PR #25479 (Pascal only, open) |
| CUDA graphs on compute 6.1 | +6–14 % | Gate lowered from Volta to dp4a (`v2:ggml-cuda/ggml-cuda.cu:5017`) | Already on upstream for compute ≥ 7.0 | 0 | PR #27721 (open) |
| FP32 activation sums + rescale off (Q76, Q22) | +9.6 % | The rescale (abs, max-pool, 7 scalar ops, a divide and a multiply per MoE layer) keeps Q8_1's FP16 block sums below 65,504 in layers 46–48 (channel RMS 25,600–48,400); the Pascal kernels compute the sums from the int8 values | The FP32 sums are Pascal-only (`v2:ggml-cuda/mmvq-pascal.cu:157-168`). The switch is generic (`v2:src/llama-graph.cpp:1677-1683`). llama.cpp's one-token kernels compute the Q4_K/Q5_K sums by dp4a in FP32 (`v2:ggml-cuda/vecdotq.cuh:567-596`) and Q6_K, IQ4_XS, Q8_0 use none; Q4_0/Q4_1/Q5_0/Q5_1 read the FP16 sum (`:136`, `:160`, `:202`, `:231`), and MMQ's tensor-core path reads it for Q4_1/Q5_1/Q4_K/Q5_K (`mmq-vec-dot.cuh:349,361`). So `tg` is safe for UD (its layer-46–48 downs are Q5_K and IQ4_XS) and `=1` for Q8_0, by construction. Mixes with Q8_0 downs in 46–48 were scored with the rescale off in batches on the quant machine's non-Pascal GPUs (Q27, Q73, Q113); at one token their Q5_0 downs in layers 2–45 read the FP16 sum, which the overflow analysis (Q22) puts out of reach but no one-token run on those GPUs has checked | +11–18 % / same / +8–14 % | Upstream TODO (ref PR #30656); PR #30928: "Auto-scaling leads to slowdown of 10-30%"; open BF16 proposal #22571 |
| Shared-expert fusion fix (Q50) | +0.7 % | The fusion memory check refused gate/up/GLU when the allocator reused the FFN input for the GLU output | Generic (`v2:ggml-cuda/ggml-cuda.cu:3517-3545`); the one-token MMVQ fusion is active on sm_86 | +2–5 % (3 of ~40 kernels per MoE layer) | Upstream's fusion (#29184) has the same check |
| Norm writes Q8_1 (Q100) | within noise (+0–2 %) | One RMS-norm kernel writes the row as Q8_1 and does the residual add; Q/K/V and gate/up skip 244 quantizations, 43–48 adds go | Generic, default on (`comb:ggml-cuda/norm.cu:909-926`, widths ≤ 4,096); tests and logits identical on RTX 3090s | +2–8 % (~290 kernels) | Fast P100/GTX 1080 forks fuse norm and quantization ([Pascal forks](pascal_forks_and_quants.md)) |
| Persistent MoE layer (Q88) | 1.16–1.17× per MoE layer, est. +5 % per token | One launch runs router, top-6, routed and shared experts with counter dependencies | Mode 2 runs on any NVIDIA card (`pers:ggml-cuda/moe-persistent.cu:947-950`); grid = blocks per SM × SMs (`:886-897`); 12 type combinations (`:841-856`), Flint's shapes hard-coded; products are the Pascal dp4a code with FP32 sums, so it needs no rescale; bit-identical on an RTX 3090 natively and through the compute_61 PTX | +8–20 % after tuning (est.: the FFN's ~13–14 kernels per layer take ~55–65 µs for ~22 µs of streaming) | Cohere's 30B-A3.3B megakernel 1.58× vLLM at batch 1 (H100); rec 1.4–2.6× llama.cpp (Apple); ik_llama.cpp `-fmoe` +7 % decode (DeepSeek-Lite) |
| Pipelined speculation (Q87) | 1.3–1.55× with untrained guesses | Guesses enter card 1 while later cards finish token t; one-token steps only; sample-and-match | Generic (`pipe2:src/llama-context.cpp:1626`, `:5072`); bit-identical on two RTX 3090s and on four virtual devices over them | — / 1.25–1.40× (sim.) / — | PipeInfer (llama.cpp, MPI), SPD, SpecPipe (8 × RTX 3090), FlowSpec; nothing in llama.cpp |
| Staircase drafter (Q20) | pilot running | Partial-depth features of the tokens in flight, Gumbel-coupled | Generic; on one card a full-depth head is an EAGLE-style drafter | 1.2–1.45× / 1.6–2.2× at p 0.7 / ~1.2× | SPD; DFlash +26 % and MTP +23 % for Qwen3.6 on one 3090 |
| Screened head (Q91, Q92) | +1.5–2.3 % (est. 6 %) | Q4_0 screen ranks 150,144 rows, exact Q8_0 logits for 256 | Generic (`comb:src/llama-graph.cpp:4390`); no new kernels | +1–2 % (0.15 GB at ~0.8 TB/s ≈ 0.19 ms, minus top-k and 4 launches) | HiRE |
| GPTQ head, GPTQ body, F16 router (Q56, Q112, Q113) | −0.116 GB, recovers 0.45–0.79 of Q5_0's cost, −0.049 GB | Better rounding and precision per byte | File-level, GPU-independent; UD's router is already BF16 | +1–3 % where the file fits | NVIDIA ModelOpt GPTQ for MoE (#2840); no GPTQ in llama-quantize |
| Byte-priced mixes, K2h-r16 (Q73, Q77) | 39.3 tok/s, +4.9 % over UD | Priced by the Pascal kernels' bytes-to-time rates | Priced wrong for Ampere, where K- and I-quants decode at full speed; still 0.12 GB lighter than UD and better on 512-token KLD | +1–3 % on 2 cards | Unsloth's dynamic quants balance the same trade |
| PR #31047 for prompts (Q108, Q117) | +22–29 % prompts | MMQ expert tiles by tokens per expert | All architectures (`comb:ggml-cuda/mmq.cu:93-98`, `:209-210`); Flint sees ~12 tokens per expert at `-ub 512` | prompts 1.1–1.35× | Upstream, open, by a CUDA maintainer |
| On-card mask (Q110, Q116), reuse fix (Q109) | +1.3 % at 8k; targets 32k+ and `-ub 128` | Mask expanded per card from cell positions; overlapping micro-batches | Generic. PCIe 4.0 moves a 131k mask row in ~10 µs; the host fill already copies rows (PR #18842) | decode ~0, prefill +2–8 % at ≥ 64k; up to ~0.75 GiB VRAM per card freed at 262k on a split | PR #29963 (inputs once per device, open) |
| CPU-free loop (Q120, Q121) | built, untimed | One async upload per card, embedding on card 1, sampled id fed card to card | Generic, executor only | — / +1–3 % / — | SGLang overlap scheduler, vLLM Model Runner V2 |
| GPU sampling for tools (Q55, Q67) | tool requests 12–16 % slower | Host sampling copies 600 KB per token over PCIe 1.0 x1 (2.8 ms) | Same switch-off upstream (`common/sampling.cpp:422-432`); the copy costs ~25 µs on PCIe 4.0 | +1–2 % on tool requests | Backend sampling #17004; #25919 (reasoning budget) |
| Staggered conversations (Q104) | est. 4–8× total | Several conversations one token each through idle cards | Generic | — / near S × one stream at S conversations / — | Orca, SpecPipe-DB |
| First-finish race (Q93) | offline: reasoning 0.72× at k = 2 | First sample to close `</think>` wins | Client-side on one card (two requests at `-np 2` cost ~1.13 T per step) or the executor | 1.1–1.15× / 1.2× per answer | First Finish Search, short-m@k |
| Task-level quality (Q41, Q106) | E1 non-inferior for G3, K2h | Paired task checks | Results hold on any GPU within compute-path noise (Q21: 3090 against V100 kernels KLD 0.065) | — | — |

**What did not work on the rig, and on a 3090:**
- **Q141, PowerMizer "Prefer Maximum Performance":** nothing on the P106s; the cold-clock effect it targeted is a
  Pascal utilization policy. Not relevant to a 3090.
- **Q19, batched verify:** a no-go on the P106s (a 3-token step 2.2 T). On a 3090 an MoE's 2- and 4-token steps cost
  1.13 and 1.68 T (PR #31047's gemma4 table), so batched speculation pays modestly (row 10).
- **Q43, MTP:** Qwen3.6's head gave 1.04× on the fork because its verify steps ran on stock multi-token kernels. Flint has
  no MTP head. Qwen3.6 MTP gives +23 % on one 3090 (thc1006).
- **Q13 rotations, Q6 fewer experts, Q94 reflection suppression, Q107 global attention in Q8_0:** file or text effects,
  the same on any card.
- **Q54, q8_0 KV ~30 % slower on the P106s:** a Pascal kernel effect (the VEC kernel at FP32). On a 3090 a q8_0 cache
  halves the KV bytes at depth (est. faster past ~32k); Q62 found no measurable quality cost at 512 tokens.

## 2. Flint on one 3090 (question 2)

### 2.1 What fits

A headless 3090 leaves ~23.7–24.0 GiB to CUDA; a desktop on the same card takes 0.3–1 GiB more. The 5090 run reported
27.4 GB for UD at 131k, which read as GiB is 23.0 GiB of weights, 3.5 GiB of KV and ~0.8 GiB of compute buffers and
context (est.); the table below uses those ~0.8 GiB and is good to ±1 layer.
Each `--n-cpu-moe` layer frees 0.457 GiB (UD) or 0.797 GiB (Q8_0); llama.cpp keeps the experts of the first N MoE layers
in host RAM and everything else on the card.

| File | Free for KV with no offload (est.) | `--n-cpu-moe` for 32k / 64k / 131k / 262k, f16 KV | same, q8_0 KV | Quality against BF16 |
|---|---|---|---|---|
| UD-Q4_K_XL | 0–0.2 GiB: no useful context | 2 / 4 / 8 / 16 | 1 / 2 / 4 / 8 | KLD 0.120, 87.7 % same top token (the bar) |
| UD-Q3_K_XL | ~6 GiB | 0 / 0 / 0 / 3 | 0 everywhere | not scored here |
| Unsloth Q8_0 | none: ~16 GiB over | 22 / 23 / 25 / 29 | 21 / 22 / 23 / 25 | KLD 0.052, 93.3 % |
| Makers' Q4_1 | 0–0.2 GiB | as UD | as UD | KLD 0.316, 78.6 % (below the bar) |
| Our K2h-r16, G3 (~28–31 GiB) | none | 11+ | 10+ | KLD 0.105, 0.104 |

- The compute buffer grows with `-ub`: llama.cpp reserves logits for every row of a micro-batch (150,144 × 4 B per row:
  0.29 GiB at `-ub 512`, 1.2 GiB at 2,048), so a large `-ub` costs 1–3 expert layers of VRAM on one card.
- `--fit` (on by default) sizes the context and layers to memory; an explicit `--n-cpu-moe` keeps attention, the shared
  expert and the head on the card, and `llama-fit-params` prints what fit would choose.

### 2.2 What expert offload costs

- **Per offloaded layer:** a UD layer reads ~11.5 MB of experts per token (6 × 1.83 MiB), Q8_0 ~20 MB. At 45–70 GB/s of
  dual-channel RAM plus a GPU→CPU→GPU round trip per layer, est. 0.3–0.4 ms per UD layer. Qwen3.6 on a 3090 paid
  ~0.41 ms per layer (139.6 → 89.1 tok/s for 10 layers, with 8 experts of ~1.8 MB per token each).
- **UD on one 3090 (est., short context, rescale on):** N = 2: 90–112 tok/s; N = 4: 84–105; N = 8: 74–93; N = 16:
  60–76. At depth add the attention of the context (est. +1.3–1.9 ms per 32k in f16, half in q8_0): UD at 262k with q8_0
  KV and N = 8 est. 45–60 tok/s.
- **Q8_0 on one 3090 (est.):** ~22 layers in RAM, ~0.44 GB per token from RAM: 40–55 tok/s on DDR5, 30–45 on DDR4.
- **Prompts with offload:** llama.cpp copies the CPU-resident experts to the card for micro-batches of 32+ tokens, so
  each micro-batch uploads N × 0.46 GiB over PCIe 4.0 (~70 ms at N = 4). Larger `-ub` amortizes it, against the
  compute-buffer cost above. Qwen3.6's prompts fell 3,360 → 1,153 tok/s with 10 layers out at `-ub 512`.
- **What the project adds here:** the rescale switch (row 1) keeps its value (est. +8–14 %, a smaller share of a longer
  token); every other GPU-side gain shrinks by the RAM share.

### 2.3 Community numbers

| Card | Model, file | Setting | Result | Source |
|---|---|---|---|---|
| RTX 5090 | Flint UD-Q4_K_XL | b11825, `-fa on -ngl 999 -c 131072`, f16 KV | 145.7 tok/s generation, 3,433 tok/s prompts (server timings, agent runs) | atomicagent.io |
| RTX 3090 | Flint UD-IQ2_XXS | q4_0 KV, `ncmoe 20`, unified KV | 50–60 tok/s short prompts, 10–15 after long prompts (anecdotal) | Unsloth model discussion #1 |
| RX 7800 XT 16 GB (Vulkan) | Flint UD-Q4_K_XL | `-ngl 99 -ncmoe 24 -fa 1 -ub 2048`, DDR5-6200 | pp2048 1,886 / tg128 49.0; 1,288 / 48.3 at 16k | same discussion |
| RTX 3090 | Qwen3.6-35B-A3B UD-IQ4_NL_XL | `-sm none -fa on`, `-ub 512`, ncmoe 0 / 10 / 12 | tg 139.6 / 89.1 / 84.9; pp 3,360 / 1,153 / 1,008 | gilesthomas.com |
| RTX 3090 | Qwen3.6-35B-A3B UD-Q4_K_XL | 8k context, greedy | 115.7 tok/s; MTP n2 141.9, DFlash n2 146.2; ngram-map −0.3 %, ngram-mod −11 % | thc1006 |
| RTX 3090 | gpt-oss-20b MXFP4 | build 6210, `-fa 1 -b 4096 -ub 2048` | pp2048 5,171, tg128 161.8 | discussion #15396 |
| RTX 3090 + 128 GB DDR5-6400 | gpt-oss-120b Q8_0 | experts in RAM | ~30 tok/s | discussion #21112 |
| RTX 3090 | Llama-2-7B Q4_0 | scoreboard | pp512 5,154–5,974, tg128 158–178 | discussion #15013 |

### 2.4 Which file for one 3090

- **UD-Q4_K_XL with `--n-cpu-moe 2–4` and the rescale off at one-token steps** for chat and agents to 64k: the quality
  bar, est. 95–125 tok/s with row 1. For long documents, `--n-cpu-moe 8` with q8_0 KV to 262k.
- **UD-Q3_K_XL** if everything must stay on the card (no RAM traffic, 128k+ of context): faster, quality unscored. A
  KLD run with the project's [quality script](../../../scripts/quality_kld.sh) against reference logits from Q8_0 or
  BF16 (made once, on two cards or the CPU) settles it.
- **Unsloth Q8_0** only with fast RAM and patience (est. 30–55 tok/s): near-lossless.
- **Not:** the makers' Q4_1 (fits, below the bar, NaN with the rescale off); our ~28–31 GiB mixes (don't fit).
- **Not built:** a mix priced for Ampere. On a 3090 K- and I-quants cost no more time per byte than Q5_0/Q8_0, so the
  byte-for-quality search of Q73/Q77 would land elsewhere; a ~21 GiB file within the bar would fit one card whole.

## 3. Several 3090s without P2P (question 3)

### 3.1 The same topology as the rig, with different weights

- **Same:** a layer split runs one card at a time; each token hops through host memory between cards (no P2P); the other
  cards idle. `-sm tensor` is closed to Flint: PR #31081 (merged 2026-10-24) denies it for architectures with the MoE
  amax rescale, because an axis-0 split breaks the rescale's row maximum.
- **Different:** a hop of 8 KB staged over PCIe 4.0 costs est. 20–40 µs (0.3 ms on the rig); the host's gap from a sample
  to the next launch is est. 0.1–0.3 ms (1.5 ms on the Celeron); a stage is 2–4.3 ms instead of 3–6 ms. Plain decode on
  2–4 cards therefore runs at one card's speed minus 1–2 % (est.).
- **NVLink** (two cards) gives P2P hops but nothing for decode on a layer split, and nothing else while `-sm tensor` is
  denied (est.).

### 3.2 Pipelined speculation on 2–4 cards

`pipeline_sim.py` on Q51's 65,558 tokens of Flint transcripts, with 3090 stages: a plain token of 8.6–8.7 ms, the output
head 0.45 ms on the last card, hops 0.05 ms, host gap 0.3 ms, layers 25/24, 17/16/16 and 13/12/12/12; depth = cards − 1
(est., sim.):

| Cards | Every card busy (tokens/s) | Copy ≥ 4 only | Copy ≥ 4, else table + request counts (Q96) | Copy, else a guesser right 50 % | Copy, else right 70 % | 95 % right |
|---|---|---|---|---|---|---|
| 2 | 232 (2.0×) | 1.14× | **1.25×** (chat 1.11, code 1.20, edits 1.43, agent 1.22, rewrites 1.96) | 1.43× | 1.59× | 1.90× |
| 3 | 328 (2.8×) | 1.20× | **1.35×** (1.15, 1.28, 1.64, 1.30, 2.77) | 1.64× | 1.93× | 2.60× |
| 4 | 415 (3.6×) | 1.22× | **1.40×** (1.17, 1.32, 1.77, 1.35, 3.48) | 1.77× | 2.16× | 3.19× |

- **Sensitivity:** with hops of 0.15 ms and a 1.0 ms host gap the table moves up by 0.01–0.03 (a slower host leaves more
  to hide): 1.27× / 1.36× / 1.42× with the table guesser.
- **Against the rig:** six P106 cards simulated 1.44× with the same guesser and measured 1.3–1.55× (Q87), so the
  simulator has not overstated the rig.
- **Against batched speculation on the same split:** pipelining needs no verify batch, so it stacks with a fast card's
  cheap verify steps in principle; the executor today runs one-token steps only.
- **What a 3090 owner needs:** `llama-pipespec` from `pipe2` (pushed) builds for sm_86 as is, with the 2-warp caveat;
  wrong guesses cost no time while depth ≤ S_eff − 1 (1, 2, 3 on 2, 3, 4 cards), inside the depth ≤ 3 that is exact
  today (Q87's depth-≥ 4 divergence is under investigation). For a server it needs the llama-server integration
  (1–2 weeks, not built).

### 3.3 The CPU-free loop, several conversations, the race

- **Packed uploads and the device-fed token (Q120, Q121):** on the rig they target a 1.5 ms host gap and ~50
  synchronous copies per token. On a desktop the gap is ~0.1–0.3 ms and each small copy ~10–20 µs, so est. +1–3 %
  single stream; they matter for the streams mode, where one host thread submits for N conversations.
- **Staggered conversations (Q104):** on a layer split llama-server's `-np N` batches the slots into one step that passes
  every card; on a 3090 that batch is cheap (4 tokens 1.68 T), so `-np` already does well there. Staggering adds the idle
  cards: est. near S × one stream at S conversations on S cards, +20–60 % over `-np` at N = S (est.). Built in
  `llama-pipespec` only.
- **First-finish race (Q93):** two samples, keep the first to close `</think>`: reasoning 0.72× of the mean at k = 2,
  pass rate +0.2–0.4 points (not significant). Through the executor on 2+ cards both samples run at full speed: est.
  1.2× per answer. On one card a client can race two requests on `-np 2` today (the batch costs ~1.13 T): est.
  1.1–1.15× per answer.

### 3.4 Files and context on 2–4 cards

- **Two cards:** UD with the full 262,144-token f16 context (est. 23.0 + 7.1 + ~2 GiB); Unsloth Q8_0 at 131k in f16 or
  262k in q8_0. With pipeline parallelism (four input copies) the mask copies take n_ctx × `-ub` × 2 B × 4 per card,
  1 GiB at 262k and `-ub 512`: use `-ub 256`, or the on-card mask (Q110) which holds it once.
- **Q8_0 against UD:** 2.70 against 2.21 GB per token is ~0.65 ms more streaming on a ~9.5 ms token: est. 0.92–0.95× UD's
  speed with the rescale off on both (`=1` for Q8_0, `tg` for UD), for KLD 0.052 against 0.120.
- **Our mixes on two cards:** K2h-r16 (2.09 GB per token, Q8_0 downs in 46–48, so the rescale can go after a one-token
  check) est. +1–3 % over UD; better 512-token KLD (0.105), level written-token KLD at 8k, ~4 % worse on tool-schema
  tokens. Not published.
- **Three or four cards:** Q8_0 at 262k in f16; spare memory for several conversations (0.27 / 0.93 / 3.55 GiB of f16
  KV per conversation at 8k / 32k / 131k, Q105).

## 4. Ranked list for 3090 owners, and what to publish (question 4)

| Order | Do this | Expected (est.) | Effort | Code |
|---|---|---|---|---|
| 1 | Pick the file and offload by section 2; makers' sampling with `--min-p 0` (llama-server's default is 0.05) and reasoning kept in the history; `-fa on`; `--ctx-checkpoints` for prefix reuse past the 513-token window; explicit `-np` | Fits; the makers' quality setting | Minutes | Upstream |
| 2 | Rescale off at one-token steps: `LLAMA_NO_MOE_DOWN_RESCALE=tg` with UD, `=1` with Q8_0 | +11–18 % decode, exact | Hours (fork build) | Fork; upstream gate to propose |
| 3 | `GGML_CUDA_GRAPH_OPT=1` on one card | +3–8 % | Minutes | Upstream |
| 4 | The fork's `comb` build for sm_86 with the 2-warp edit gated: shared-expert fusion fix and norm writes Q8_1 | +4–12 % | A build | Fork |
| 5 | Two to four cards: `llama-pipespec` for copy-heavy work now; the server integration when it lands | 1.25–1.40× overall, edits 1.4–1.8×, rewrites 2–3.5× | Tool today | Fork |
| 6 | PR #31047 for prompts (in `comb`, or cherry-picked) | 1.1–1.35× prompts | Hours | Upstream PR |
| 7 | Batched n-gram speculation for edits and rewrites (`ngram-map-k` key 4, draft 8) | 1.3–1.5× on edits, ~1.0× on chat | Minutes | Upstream |
| 8 | q8_0 KV past ~32k | capacity ×1.9; est. faster at depth | Minutes | Upstream |
| 9 | Persistent MoE-layer kernel, once tuned and pushed | +8–20 % | Weeks (project) | Fork |
| 10 | A trained Flint drafter (EAGLE-style head for one card; the staircase head for splits) | 1.2–1.45× / 1.6–2.2× | Training | Project |

**What to upstream to llama.cpp, in order:**
1. **The rescale gate.** Today every backend pays the rescale (`v2:src/llama-graph.cpp:2528`). A backend capability
   ("this device's Q8_1 readers never use FP16 block sums for type X at one token") lets CUDA skip it for K-quant,
   IQ4_XS, Q6_K and Q8_0 down-projections at decode; FP32 activation sums in MMVQ for Q4_0/Q4_1/Q5_0/Q5_1 (8 dp4a per
   activation block, the way Q4_K/Q5_K already do it) and in MMQ's DS4 layout remove it everywhere. It needs a 3090
   A/B and a bit-identical one-token perplexity, both in section 5. The V100 note proposes the same change from the
   Volta side ([transfer to V100](transfer_v100.md)).
2. **The gate/up/GLU fusion fix** (Q50), already ranked first in [fork scope](fork_scope.md) §4.2.
3. **Norm writes Q8_1** (Q100) with a 3090 measurement; it is generic for hidden sizes ≤ 4,096.
4. **Flint data on PR #31047** from a 3090 at `-ub 128/512/2048`, and **pipelined decoding** as a discussion/RFC with
   multi-3090 numbers, since every layer-split user without P2P gets it.

**What to publish as a recipe:** a "Flint on RTX 3090" page in this repository: the fit table (2.1), the offload
estimates replaced by owners' measurements, the commands of section 5, the rescale switch with its safety rule (never
with Q4_0/Q4_1/Q5_0 down-projections in layers 46–48), and the sampling settings. Before it points at the fork: gate the
2-warp edit to compute 6.1 ([fork scope](fork_scope.md) §5.3 step 4) and push the persistent branch with an Ampere grid.

## 5. What a 3090 owner can measure (question 5)

The project's quant machine runs no speed measurements, so every number above waits for a 3090 owner. `M` is the model
file, `B` a build's `bin/`. Run each A/B in one process order, warm the card first, `-r 3` or more.

**Builds:**
```bash
git clone https://github.com/ggml-org/llama.cpp upstream
git clone --branch pascal-p106-v2-combined https://github.com/Cinderella-Man/llama.cpp fork
# in fork/ggml/src/ggml-cuda/mmvq.cu, case 1 of MMVQ_PARAMETERS_GENERIC: remove "return small_k ? 4 : 2;" for a
# stock Ampere launch shape (or build both ways and compare: that measures the 2-warp edit on Ampere)
for d in upstream fork; do cmake -S $d -B $d/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && \
  cmake --build $d/build -j --target llama-bench llama-perplexity llama-server; done
```

**1. Baseline and fit** (upstream; one card, then `--n-cpu-moe` steps):
```bash
$B/llama-bench -m $M -ngl 99 -fa on -ncmoe 0,2,4,8 -p 512,2048 -n 128 -r 3 -o md
$B/llama-bench -m $M -ngl 99 -fa on -ncmoe 8 -ctk q8_0 -ctv q8_0 -d 0,32768,131072 -n 64 -r 2 -o md
nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw --format=csv -lms 500   # beside a run
```

**2. The rescale switch** (fork build, same binary; never set the variable to an empty string, which drops the rescale
everywhere):
```bash
$B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5 -o md
LLAMA_NO_MOE_DOWN_RESCALE=tg $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5 -o md
# exactness: one token per step, KLD must be 0 and the same top token 100 %
$B/llama-perplexity -m $M -ngl 99 -fa on -f wiki.test.raw -c 512 --chunks 8 -ub 1 --kl-divergence-base base.kld
LLAMA_NO_MOE_DOWN_RESCALE=tg $B/llama-perplexity -m $M -ngl 99 -fa on -f wiki.test.raw -c 512 --chunks 8 -ub 1 \
  --kl-divergence-base base.kld --kl-divergence
# Q8_0 (two cards): =1, then a default-batch perplexity must stay finite
```

**3. Launch fusions and graph optimization** (fork build):
```bash
GGML_CUDA_NORM_Q8=0 $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5   # Q100 off, then default
GGML_CUDA_GRAPH_OPT=1 $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5  # one card
# Q50 has no switch: compare upstream and fork tg128 on the same card and file
```

**4. Prompts, PR #31047** (fork build):
```bash
for e in 1 0; do GGML_CUDA_MMQ_MOE_LEGACY=$e $B/llama-bench -m $M -ngl 99 -fa on -p 512,2048,8192 -n 0 \
  -ub 128,512,2048 -r 2 -o md; done
```

**5. Pipelined speculation on 2–4 cards** (build `pipe2` with the `llama-pipespec` target; scripts from this
repository's [bench/pipespec](../../../bench/pipespec/)):
```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg CUDA_VISIBLE_DEVICES=0,1 BIN=$B MODEL=$M N=128 DEPTHS='1 2 3' MISSES='0 0.3' \
  bash bench/pipespec/check_identity.sh
BIN=$B MODEL=$M TAG=2x3090 GPUS=0,1 DEPTHS='1 2' ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg' bash bench/pipespec/rig_run.sh
```
Read `bench/pipespec/out/2x3090/summary.tsv` (`same_as_plain` must say yes; `cont-dD-*` against the simulator's 1.25×
on two cards), then compare with `pipeline_sim.py` on the run's own `plain.jsonl` as [pipelined speculation](pipelined_speculation.md)
describes.

**6. Batched n-gram speculation** (upstream llama-server): the same edit-heavy request with and without
`--spec-type ngram-map-k` (key and draft sizes per `docs/speculative.md`), reading `draft_n` and `draft_n_accepted` from
the response timings; expect ~1.0× on chat and 1.3–1.5× on whole-file edits.

**7. Persistent kernel** (when `pers` is pushed): `GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1` once for
correctness, then tg128 with and without `GGML_CUDA_PERSISTENT_MOE=2`.

Report per run: llama.cpp commit, file, `-ncmoe`, context, KV type, card count, driver, PCIe link (`nvidia-smi -q -d
PCIE`), RAM type and channels, and the warm-up.

## What this changes in the plan

- **The fork's message to non-Pascal owners is "launch cuts", not "kernels":** the rescale switch, the two fusions and
  the persistent kernel are the pieces a fast GPU feels; the Pascal kernels stay Pascal.
- **The rescale gate moves up the upstream list** for every fast GPU, next to the fusion fix ([fork scope](fork_scope.md)
  ranked it as fork-only).
- **The 2-warp edit must be gated before any non-Pascal build** of the fork is recommended.
- **Q19's no-go is a P106 result:** on a 3090, batched speculation and a trained Flint drafter for one card are worth
  revisiting; the staircase head's full-depth variant (head U of the Q20 pilot) is that drafter's starting point.
- **A 3090 measurement round** (section 5, items 1–4, ~2 hours on one card) would replace most estimates here.

## Already covered

| Topic | Where |
|---|---|
| Which fork changes are Pascal, MoE, generic or Flint, with file:line | [fork scope](fork_scope.md) |
| The same question for V100 cards | [transfer to V100](transfer_v100.md) |
| Executor design and correctness on two RTX 3090s | [pipelined speculation](pipelined_speculation.md) |
| Drafter choice and its training pilot | [pipeline drafter](pipeline_drafter.md) |
| The rescale, FP32 sums, norm writing Q8_1, persistent kernel design | [kernel integration](pascal_mmvq_integration.md) §8, §17; [breakthrough candidates](breakthrough_candidates.md) §2 |
| Prompt processing, PR #31047, the on-card mask | [prompt processing](prefill_pascal.md) |
| Host loop, packed uploads, device-fed token | [host decode loop](host_decode_loop.md) |
| Tool-request sampling | [tool-request sampling](tool_request_sampling.md) |
| File quality at 8k on agent text, release rule | [long-context mix](long_context_mix.md), [long-context quality](long_context_quality.md) |
| GPTQ for the head and body | [GPTQ for the body](gptq_body.md) |
| Flint's makers' engine and settings | [rec](rec_engine.md) |
| llama.cpp's split modes, P2P, MoE flags | [llama.cpp-family engines](engines_llamacpp_family.md) |
| A dense model on one 3090 with speculation | [HyperQwen lessons](hyperqwen_lessons.md) |

## Not verified

- **Every 3090 speed in this note is an estimate.** No Flint run on a 3090 exists in this project; the token budget of
  section 1.1 rests on the kernel count from code reading, ~2.5–3.3 µs per kernel (not measured on a 3090), the
  scoreboard's dense-model bandwidth and one 5090 run of served agent traffic with unstated batch settings.
- **The rescale's share:** ~470 kernels is the count from the fork's commit; their cost on a 3090 and the claim of PR
  #30928 (10–30 %, hardware not named) are not measured here.
- **The fusions on a 3090:** Q50 and Q100 are correct on RTX 3090s, never timed there; Q100 gave less than estimated on
  the rig.
- **The persistent kernel on Ampere:** correct, untimed; its products are the Pascal dp4a code and its gain depends on
  tuning.
- **The 2-warp edit on Ampere:** sign and size unknown.
- **Pipelined speculation on 3090s:** simulated with assumed stage times, hop and host gap; the executor's speed was
  measured only on the rig.
- **Batched speculation on a 3090:** replayed with gemma4 26B-A4B's step costs from PR #31047's table (an MoE with 128
  experts, top 8) and with a flatter Flint-shaped curve; llama-bench's small prompt steps may also run without CUDA
  graphs, which overstates verify cost.
- **Offload speeds:** scaled from Qwen3.6 on a 3090 and from bytes per layer; RAM speed, CPU and PCIe generation move
  them.
- **UD-Q3_K_XL's quality** and the makers' NVFP4 file on Ampere (llama.cpp's NVFP4 path on compute 8.6 was not checked).
- **The memory table:** usable VRAM, compute buffers and `--fit`'s choices are calibrated on one 5090 run, not on a 3090.
- **q8_0 KV speed at depth on Ampere,** and which flash-attention kernel it takes at one token.
- **Community numbers** are third-party runs with the settings they state; the Flint 3090 report is anecdotal.

## Sources

NVIDIA and llama.cpp (P):
- NVIDIA, "NVIDIA Ampere GA102 GPU Architecture" whitepaper v2, Table 9,
  <https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf>
- llama.cpp CUDA scoreboard, [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) (RTX 3090
  rows of 2025-08-01 and 2026-09-07)
- [PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928) (Flint support, the 1/1024 rescale, "Auto-scaling leads
  to slowdown of 10-30%", merged 2026-10-21), [PR #30861](https://github.com/ggml-org/llama.cpp/pull/30861) (per-expert
  amax rescale), [PR #31081](https://github.com/ggml-org/llama.cpp/pull/31081) (`-sm tensor` denied for Flint, merged
  2026-10-24)
- [PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047) (MoE tile list; RTX 3090 tables for granitemoe,
  gemma4 26B-A4B and llama 8B; open)
- [PR #16991](https://github.com/ggml-org/llama.cpp/pull/16991) and [#18496](https://github.com/ggml-org/llama.cpp/pull/18496)
  (`GGML_CUDA_GRAPH_OPT`), [#16130](https://github.com/ggml-org/llama.cpp/pull/16130) (fused top-k MoE),
  [#15077](https://github.com/ggml-org/llama.cpp/pull/15077) (`--n-cpu-moe`), [#17004](https://github.com/ggml-org/llama.cpp/pull/17004)
  (backend sampling), [#18842](https://github.com/ggml-org/llama.cpp/pull/18842) (mask rows copied), [#29963](https://github.com/ggml-org/llama.cpp/pull/29963),
  [#25919](https://github.com/ggml-org/llama.cpp/pull/25919), [#25479](https://github.com/ggml-org/llama.cpp/pull/25479),
  [#27721](https://github.com/ggml-org/llama.cpp/pull/27721), [#6766](https://github.com/ggml-org/llama.cpp/pull/6766)
  (CUDA graphs, RTX 3090 +10 % on Llama-2-7B)
- ik_llama.cpp [PR #229](https://github.com/ikawrakow/ik_llama.cpp/pull/229) (`-fmoe`), [#947](https://github.com/ikawrakow/ik_llama.cpp/pull/947)
  (graph reuse), [#1022](https://github.com/ikawrakow/ik_llama.cpp/pull/1022) and [#1080](https://github.com/ikawrakow/ik_llama.cpp/pull/1080)
  (split mode graph on two 3090s)

Flint and community measurements:
- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) and [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF)
  (file list and sizes via the Hub API), [Unsloth's Flint guide](https://unsloth.ai/docs/models/flint) (P)
- [unsloth/flint-GGUF discussion #1](https://huggingface.co/unsloth/flint-GGUF/discussions/1) (RTX 3090 and RX 7800 XT
  reports) (S)
- [atomicagent.io, "Recursive Flint"](https://atomicagent.io/blog/recursive-flint/) (RTX 5090, 2026-10-22) (S)
- [gilesthomas.com, Qwen3.6-35B on an RTX 3090](https://www.gilesthomas.com/2026/07/benchmarking-qwen-3-6-35b-moe-rtx-3090) (S)
- smcleod.net, 2026-02-25: two RTX 3090s with a community-patched driver, `nvidia-smi topo -p2p` from NS to OK (S)
- [thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) (P)
- llama.cpp discussions [#15396](https://github.com/ggml-org/llama.cpp/discussions/15396) (gpt-oss-20b) and
  [#21112](https://github.com/ggml-org/llama.cpp/discussions/21112) (gpt-oss-120b with experts in RAM) (P)

The fork, read-only (P): `v2:ggml-cuda/mmvq.cu:109-147, 455-470`; `v2:ggml-cuda/mmvq-pascal.cu:157-168, 596`;
`v2:ggml-cuda/vecdotq.cuh:136, 160, 202, 231, 567-596`; `v2:ggml-cuda/quantize.cu:100, 548`; `v2:ggml-cuda/mmq.cu:408`;
`v2:ggml-cuda/ggml-cuda.cu:3517-3545, 5017`; `v2:src/llama-graph.cpp:1670-1722, 2518-2530`; `comb:ggml-cuda/norm.cu:909-926`;
`comb:ggml-cuda/mmq.cu:93-98`; `comb:src/llama-graph.cpp:4390`; `pers:ggml-cuda/moe-persistent.cu:841-856, 886-897,
947-950`; `pipe2:src/llama-context.cpp:1626, 1652, 5072`; `pipe2:src/llama-model.cpp:1699-1705`;
`tool:common/sampling.cpp:432-442`.

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): Q1, Q4, Q8, Q13, Q19–Q22, Q27, Q38, Q41, Q43, Q46, Q50,
  Q54–Q56, Q62, Q64, Q73, Q76, Q77, Q87, Q88, Q91–Q93, Q96, Q100, Q104–Q113, Q116–Q121, Q125, Q133, Q141 and the
  decision log.
- Notes: [v2 synthesis](v2_synthesis.md), [breakthrough candidates](breakthrough_candidates.md),
  [pipelined speculation](pipelined_speculation.md), [pipeline drafter](pipeline_drafter.md),
  [prompt processing](prefill_pascal.md), [host decode loop](host_decode_loop.md),
  [tool-request sampling](tool_request_sampling.md), [GPTQ for the body](gptq_body.md),
  [long-context mix](long_context_mix.md), [rec](rec_engine.md), [llama.cpp-family engines](engines_llamacpp_family.md),
  [kernel integration](pascal_mmvq_integration.md), [fork scope](fork_scope.md), [Flint speed-ups](flint_speedups.md),
  [HyperQwen lessons](hyperqwen_lessons.md), [unconventional ideas](unconventional_ideas.md).
- Tools run for this note on Q51's transcripts (65,558 generated tokens): [pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py)
  with the stage times of section 3.2, and [replay.py](../../../bench/ngram_replay/replay.py) with `--vn
  1=5.752,2=6.516,4=9.650,8=16.089` (gemma4 on a 3090, ms per step) and `--vn 1=1,2=1.15,3=1.25,4=1.35,8=1.75` (a
  Flint-shaped curve, est.).
