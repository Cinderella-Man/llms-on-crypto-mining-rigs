# Magnitude (github.com/magnitudedev/magnitude): what it is, why its decode looks fast, and what Miner Llama can use (2026-10-28)

Read-only study through GitHub web pages, `raw.githubusercontent.com`, the GitHub API, the project's site and benchmark
page, its drafters' Hugging Face cards, llama.cpp PR #22105, the Launch HN thread and secondary write-ups. Nothing was
cloned, downloaded, installed or run; the install script was read, not executed. Paths are relative to
`https://github.com/magnitudedev/magnitude/blob/main/`. "(est.)" marks my own arithmetic. Every Magnitude speed figure
is the vendor's own.

## 1. What it is

| | |
|---|---|
| What | A local inference engine with a desktop app (Electron), a CLI and a background service. It serves GGUF models over an OpenAI-compatible API and writes their settings into coding-agent harnesses (Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, Cline). Models run on the user's machine: it is not a hosted model, and it needs no account or API key |
| Company | Magnitude AI Inc., Y Combinator S25 (the site's footer, the Launch HN post). Two founders, Anders and Tom (Tom Greenwald posts the launches on X), who earlier published an open-source browser agent (4k+ stars, per the HN post). They plan to sell hybrid local and cloud inference billed per token |
| Licence | Apache-2.0 |
| Age and activity | Repository created 2026-06-12. Launch HN on 2026-09-30 with v0.2.0, which replaced the earlier llama.cpp-based engine with the project's own (maintainer comment on issue #82). 1,631 commits on `main`, several a day (last push 2026-10-28); 7.4k stars, 524 forks, 30 open issues and 12 open pull requests. The two founders wrote 1,537 of the commits, a release bot 75, eleven other people 19 |
| Code | Rust 24.6 MB (engine and service), TypeScript 10.8 MB (desktop, CLI, web; Bun, Effect, Electron), C++ 1.7 MB, Python 1.4 MB (evaluation), CUDA 1.05 MB, Metal 0.71 MB, GLSL 0.32 MB (Vulkan) |
| Kernels | Written once in the project's own language, Seismic (`inference/engine/kernels/kernels/*.seismic`; compiler and runtime in `inference/seismic/`), with native code per backend (`kernels/cuda/`: 88 `.cu` files, plus Metal, Vulkan and CPU). On NVIDIA the model kernels are CUDA C++ that a bundled NVRTC 12.9 compiles on the user's machine into a CUBIN for the exact card; compiler-generated kernels ship as PTX 7.1 for the driver to compile (`inference/docs/compatibility.md`) |
| Hardware | CUDA needs compute capability 8.0 or newer. Pascal is not served by CUDA; Pascal, Volta and Turing are routed to Vulkan, whose Pascal paths are validated offline only, not on a device. Driver R525 or newer. No multi-GPU: distributed tensor execution is out of scope in `inference/docs/overview.md`, and the founders put multi-GPU on the roadmap |
| Models | A pinned catalog (`inference/catalog/models.json`, Hugging Face commits locked in `models.lock.json`): Qwen3.5/3.6/3.8 (Qwen3.8-Flash-Next included), Gemma 4, Nemotron 3.5, LFM, MiniCPM and others, mostly Unsloth GGUFs, each with one drafter (section 2.3). Kernels are written per architecture family, so the catalog is narrower than llama.cpp's |
| Fit for this rig | None. Nine 6 GB cards on PCIe 1.0 x1 need a layer split across GPUs, which it does not do, and its CUDA backend does not serve compute 6.1 |

[engines_other.md](engines_other.md) names it once, in the 2026 wave of new engines, without a study.

## 2. The speed claim and how it is achieved

### 2.1 What is claimed

- **Headline** (README, site, the launch post on X of 2026-10-15): over 2× faster than Ollama, on average across popular
  local models, on Mac and NVIDIA.
- **The numbers** ([benchmark page](https://magnitude.dev/benchmarks)), average of 11 models at about 4 bits:

| Machine | Baseline | Prefill | Decode | Decode range per model |
|---|---|---|---|---|
| Mac mini M4 Pro, 48 GB | Ollama 0.35.1 (MLX) | 1.32× | 2.19× | 1.29–3.51× |
| DGX Spark (GB10, 128 GB) | Ollama 0.35.1 (llama.cpp) | 1.34× | 2.17× | 0.76× (Nemotron 3.5) – 4.84× (Qwen3.5-4B, 61 → 296 tok/s) |

- **The launch claim before it** (Launch HN, 2026-09-30), against llama.cpp: Qwen3.6-35B-A3B at 4 bits, 64k context,
  no speculative decoding: decode 30 → 57 tok/s on Metal and 49 → 58 (+19 %) on CUDA; prefill 466 → 507 and
  2,033 → 2,507.

### 2.2 The method behind the numbers

From the benchmark page and the harness (`inference/evaluation/bench`, `inference/evaluation/fixtures`):

| Condition | Value |
|---|---|
| Workload | One chat request: ~64k tokens of Moby-Dick passages, the model asked to repeat the latest one, 256 output tokens. This is the harness's `prose-repeat` fixture: a 512-word passage, and the answer is its latest paragraph-aligned suffix of at least 256 words. Every output token is already in the prompt |
| Sampling | Greedy, thinking off |
| Cache | A fresh engine process and an empty prompt cache per request (the harness disables prefix caching) |
| Speculation | On for both engines: Ollama with each tag's own drafter (MTP for most), Magnitude with its catalog drafter (DFlash for the Qwen, Nemotron and Muse Glimmer models, MTP for Gemma 4). The page notes that Ollama's MLX runner drafted very few tokens for three Gemma 4 models at 64k |
| KV cache | Ollama's defaults (16-bit); Magnitude's 8-bit keys and 4-bit values (the HN thread: llama.cpp also ran 16-bit, because the 8/4-bit setting slowed its decode) |
| Sample size | Median of two requests; some Ollama rows are single requests from an earlier run of the harness |
| Average | Arithmetic mean of the per-model ratios |
| Quality | Not scored; the engines run different weights, templates and numerics, and every answer reached 256 tokens |

### 2.3 Where the decode speed comes from

1. **A block drafter on a recitation task.** DFlash drafts a whole block of positions (16 for the Qwen drafts in
   llama.cpp's implementation) in one pass from the target model's hidden states (`kernels/dflash.seismic`: residual
   rows tapped from several target layers, `cuda/tap_rows.cu`, fused by a projection; `cuda/draft_path_step.cu`,
   `draft_top_k.cu`). When the answer is a copy of the prompt, nearly every drafted position is accepted. This rig
   shows the same effect with simple guessers: inside a copy a 6-token match continues correctly 95–99 % of the time
   (Q64), and file rewrites run 4.08× under the pipelined executor (Q96). On new text DFlash gains far less, least on
   MoE models: llama.cpp's DFlash PR (#22105, merged June 2026) reports Qwen3.6-27B at 2.69× on SpeedBench and
   GPT-OSS-20B at 0.61–1.27×, because a wide verify reads many more experts (as Q19 and Q366 found here). The drafters
   are third-party checkpoints that Magnitude converted to GGUF: Z-Lab's DFlash drafters (Qwen3.6-35B-A3B's is 0.4 B
   parameters, 421 MB at Q8_0), NVIDIA's LiLiCorr drafters and DSpark drafters.
2. **Split for one model** (est., across two vendor runs a fortnight apart, Qwen3.6-35B-A3B on the DGX Spark): the
   engine alone 49 → 58 tok/s against llama.cpp (1.18×); with drafters on the recitation task Ollama 85.8 → Magnitude
   216 (2.52×). About 2.1× of the headline ratio is the drafter on copied text.
3. **The cache format at 64k.** An 8-bit key and 4-bit value cache reads (8 + 4) / (16 + 16) = 0.375 of a 16-bit
   cache's bytes per step (`cuda/attention_decode_k8v4.cu`, `attention_prefill_k8v4.cu`, `attention_append_k8v4.cu`).
   At 64k the full-attention layers' cache is a large share of a decode step's bytes, so part of the decode lead is
   the cache format against Ollama's default.
4. **Tuned kernels.** The prefill ratios (0.91–2.20×) involve no drafting and are the kernels' doing. The first-load
   tuner (`inference/docs/tuning.md`) spends up to 60 s per model load searching the configurations each kernel's
   contract allows (structurally different implementations first, then refinement of the best; the site names thread
   grouping, chunk size and prefetch depth). Its objective is device time over prefill at 4k and 64k, one decode step
   at 4k and 64k of history, and drafting, each point weighted by its share of the default configuration's time. A
   candidate must match the default on identical inputs (an error ratio over every output and every written state
   region, plus edge-case shapes with no timing weight) before it is timed further; results are cached per model,
   device, toolchain and computation, and a later load reuses them.
5. **On new text the lead shrinks or reverses.** On the HN thread, users measured llama.cpp about 2× faster on an M5
   Max (v0.2.1) and 20–30 % faster at decode on an RTX 5070 Ti; on an M5 Pro Magnitude led oMLX on decode (82.8
   against 76.5 tok/s) and trailed on prefill (709 against 1,843). A wiki page on the engine (aiwiki.ai) reports a
   re-test posted on X on 2026-10-16 that reproduced the gain on the recitation workload and found Magnitude slower
   than Ollama when the model wrote new text, each engine with its default drafter; the post itself was not found, so
   its numbers are not given here. Issue #82 (Magnitude 9.05 against llama.cpp's 55.7 tok/s on an M4, same GGUF) was
   closed as obsolete with the old engine, without a new measurement.

The 2× is a real measurement of a narrow case: reciting a passage at 64k context with a strong block drafter, a compact
cache and a cold process. It says little about the speed of a coding agent writing new text, which is what
[bench/agent_check](../../../bench/agent_check/README.md) measures here.

## 3. Red flags and signs of legitimacy

**Flags:**
- The headline generalizes a recitation probe (section 2.2) to "popular local models".
- Two requests per point, different drafters on each side, no quality scoring, and different weights and templates per
  engine.
- 7.4k stars against two people writing the code: the shape of a Launch HN and X spike. Nothing found either confirms or
  rules out bought stars.
- The architecture documents describe intended design and say they are not a record of validated capability; the
  founders say coding agents help optimize the kernels, and the inference README says a passing local build does not
  qualify every release host.
- Installation is a remote script piped into a shell. On Linux it asks for sudo and keeps sudo's timestamp alive with a
  background `sudo -n true` every 50 s while it runs; each run is counted by the vendor's server (operating system,
  architecture, package type).

**Signs of legitimacy:**
- Source under Apache-2.0 with readable kernels, a public harness, and a method page that lists the losing rows and the
  Ollama MLX drafting caveat.
- A Y Combinator company with an earlier open-source product, and steady daily commits.
- The installer downloads only from the project's GitHub releases and checks size and SHA-256 against an offer signed
  with an embedded publisher key (`openssl pkeyutl -verify`) on Linux, and the Apple code signature with a fixed team
  identifier on macOS.
- The desktop and CLI manifests carry no third-party analytics, crash-reporting or tracking package (the internal
  `@magnitudedev/*` packages were not read; the CLI has an update checker). No wallet, crypto or credential-collection
  code was seen, and no account or API key is needed.

**Scam verdict:** not a scam. A real engine with real engineering, sold with a headline that overstates its general
speed.

## 4. Technique → our rig

| Technique (Magnitude source) | Transfers to a P106 (sm_61, 5 SMs, x1, Celeron)? | Already in Miner Llama or the registry? | Expected effect on agentic speed | Effort |
|---|---|---|---|---|
| Progressive output head: Q8_0 codes in bit planes, every row scored on a 4-bit view with a per-row error bound, survivors refined at 5 bits, then exact (`kernels/readout.seismic`; `cuda/readout_top_rows.cu`, `readout_refine_rows.cu`, `readout_exact_rows.cu`, `readout_planes_rows.cu`) | Yes: taking a high nibble is cheaper than decoding Q4_0 | Q92's screen is a separate GPTQ Q4_0 copy (173 MB side file), shown exact for top-k ≤ 20 on 40,527 held-out rows (Q91); no bound-based version | Flint: Q92's bytes, no side file, exact for any sampler; Ornith: a head at Q8_0's values for fewer bytes than Q5_0 (5.1) | 3–5 days after an offline survivor count |
| First-load tuner over each kernel's legal launch configurations, numerical gate, cached results (`inference/docs/tuning.md`) | Yes | Hand-tuned for the P106-090: Q25, Q34, Q38, Q46, Q88's grid, Q166's 11 variants, Q285, Q351. [uzu_lessons.md](uzu_lessons.md) found per-device tables nothing new for this rig | est. 0–2 % on the rig; est. 0–10 % on other Pascal cards, unmeasured (5.2) | 1–2 weeks |
| 8-bit key, 4-bit value cache read directly by decode and prompt attention (`cuda/attention_*_k8v4.cu`) | Partly: q8_0 KV runs llama.cpp's VEC kernel 30 % slower here (Q54), so it needs Q210's decode kernel | q8_0 KV costs nothing at 512 tokens (Q62); Q61; Q210 | Ornith at ~40k est. 1.2–1.35× with the kernel; est. ≤ 2–4 % at the agent check's 5–8k (5.3) | Medium |
| DFlash / DFlash2 block drafter from target-layer taps (`kernels/dflash.seismic`, `cuda/draft_*.cu`) | As a block guesser for the pipelined executor, maybe; not as a 16-token batched verify on an MoE | llama.cpp `--spec-type draft-dflash` (PR #22105); Q271 for Flint; [uzu_lessons.md](uzu_lessons.md) | Flint est. +3–8 % (Q271); Qwen3.6 parked (5.4) | High for Flint (training); low for Qwen3.6 (a drafter exists) |
| DSpark's per-slot confidence cutoff (`cuda/draft_confidence.cu`) | Yes | Q409, Q342, Q57's cumulative-confidence cutoff | none new | — |
| Clustered draft readout: the draft scores vocabulary clusters and computes logits only inside the chosen ones (`kernels/clustered_readout.seismic`) | Yes | Q400 (shortlist), Q401 (low-rank ranker) | a Q401 variant (5.6) | Low |
| Counter-based Philox and Gumbel sampling keyed by vocabulary id, seed and position (`inference/docs/engine/generation.md`) | Yes | v2's position-keyed Gumbel coupling (Q96) and seeded GPU sampling | none | — |
| Grammar mask built on the host during the forward pass, logits kept on the device | Yes | Q67 (sample on the GPU, check on the host, full row only on a rejection) | none | — |
| Replay of accepted input after a state is evicted; one unresolved advance per sequence (`inference/docs/engine/state.md`) | — | the accepted-path replay in [uzu_lessons.md](uzu_lessons.md) | none new | — |
| Prefix tree shared by concurrent sessions | One conversation: no; streams mode: maybe | Q111's prefix files and ring checkpoints for one slot | 0 on the agent check; memory for parallel agents (5.5) | Medium |
| Kernels compiled on the user's machine (NVRTC) | Not needed: prebuilt sm_61 tarballs (Q139); compiling on the Celeron would take minutes per model | — | none | — |
| NVFP4 expert GEMMs and other Ampere-and-newer paths (`cuda/routed_experts.cu`) | No: compute 8.0+ | — | — | — |
| Qwen3.8-Flash-Next's hashed n-gram embedding on the card (`kernels/ngram.seismic`; it is a model input, not prompt-lookup drafting) | No: the 28.8 GB table cannot sit on 6 GB cards | Q214, Q228, Q229 | none | — |
| Recitation benchmark, cold process per request (`prose-repeat`) | Methodology | the agent check is this project's metric | such probes measure guessers, not agentic speed | — |

## 5. Transferable ideas, ranked

### 5.1 An exact progressive output head without a side file

- **Mechanism.** Load the Q8_0 head's codes as three planes: the high nibble of each code (with its block scale, ~4.5
  bits per weight), bit 3, and the low three bits. Every vocabulary row is scored on the 4-bit view, whose value is the
  scale times the nibble's level with the missing bits taken at their midpoint; each row carries an error radius that
  bounds |exact − view| for a normalized input. Rows whose upper bound reaches the best lower bound survive; the
  survivors are rescored with bit 3, then exactly with all eight bits. The sampler's own Gumbel noise is added to both
  bounds, so the sampled token is the one the full head would give, for any temperature, and for top-k with the
  k-th lower bound as the threshold.
- **Against Q92.** Q92 ranks with a separate GPTQ Q4_0 copy of the head (+173 MB on disk, on the last card and during
  loading) and takes exact logits for a fixed 256 candidates, exact for top-k ≤ 20 as checked on held-out rows. The
  progressive head reads the same ~0.17 GB per Flint token (est.: 18 of every 34 bytes of a Q8_0 block), needs no side
  file, and is exact by construction for any sampler. Full-logit requests and grammar rejections still fall back to
  the full head, as with Q92 and Q360.
- **Expected effect.** Flint: the gain Q262 and Q360 target, est. +1.0–1.6 % per generated token on the agent check,
  as an alternative to the side file rather than an addition, and 173 MB freed on the card that carries the head
  (Q277's lighter last card), along with the host RAM the side file needs while loading (Q262). Ornith-1.5-122B: its Q8_0 head (811 MB) screened at ~430 MB per read (est.) against 524 MB for the Q5_0 head
  of Q171/Q212, est. +1 % per token and −0.6 ms per MTP draft (more if the draft head is Q8_0 today); a shortlist
  (Q400) costs acceptance, this costs none.
- **Risk.** A worst-case bound can keep many rows when the top logits are close, and a 2,048-wide product's bound is
  much looser than its typical error (est.). First an offline count of survivors per level on Q56's dumped final
  hidden states (~1 h on the quant machine); kill if more than ~1–2 % of the vocabulary survives the 5-bit level.
- **Effort.** 3–5 days: load-time repack, the bound per row, two gather kernels, exactness checks against the full head
  under greedy decoding and the makers' sampling. New to this project.

### 5.2 First-load tuning of the Pascal kernels, for owners of other Pascal cards

- **Mechanism.** The fork's Pascal kernels have discrete launch choices: lane-group size and rows per group, staged or
  direct loads, two or four warps, the persistent MoE kernel's grid (ten blocks on five SMs, Q88), the attention split,
  MMQ tiles (Q285, Q351), `GGML_CUDA_MMVQ_MAX_COLS`. All are fixed for the P106-090. A tuning pass at first load times
  each choice on the owner's card and the model's own shapes, with cold weights (a rotating working set larger than
  L2, as in [uzu_lessons.md](uzu_lessons.md)), weights the time by the agent check's shares (generation 76–79 %,
  prompts 19–21 %, contexts 5–24k; [flint_v3_plan.md](flint_v3_plan.md)), keeps only choices whose output is
  bit-identical to the default (or within the summation-order rule of Q38 for choices that change the order), and
  caches the table per card model, SM count, driver, build and model.
- **Expected effect.** est. 0–2 % on this rig, whose shapes were tuned by hand. On other Pascal cards (P106-100 with 10
  SMs, P104-100, GTX 1060–1080, P40 with 30 SMs) est. 0–10 %, unmeasured: launch shape alone moved P106 decode by
  13–27 % in Q34, and the best shape depends on the SM count. It serves the "any card count" goal of the public fork
  more than this rig's agent check.
- **Effort.** 1–2 weeks: a tuning table consulted at launch, a tuning tool built on `bench/mmvq_pascal`, the identity
  gate. New as a mechanism.

### 5.3 An 8-bit key, 4-bit value cache for Ornith at long context

- **Mechanism.** Q210 measured attention at 32 % of an Ornith-1.5-122B token after 39k tokens of context, each layer
  reading its 16-bit cache at ~21 GB/s. An 8/4-bit cache read directly by Q210's planned decode kernel cuts the bytes to
  0.375× (q8_0's 0.5×).
- **Expected effect.** est. 1.2–1.35× at ~40k together with the kernel; est. ≤ 2–4 % at the agent check's 4.6–7.5k
  context, where attention is ~6 %. Needs agent-text KLD at 8k or more first: Q62 covered q8_0 keys and values at 512
  tokens only, and 4-bit values are new here.
- **Effort.** Medium, after Q210's kernel. Partly new (the 4-bit values).

### 5.4 Parked: Z-Lab's DFlash drafter for Qwen3.6-35B-A3B

A pretrained drafter exists (0.4 B parameters, 421 MB at Q8_0) and llama.cpp runs it (`--spec-type draft-dflash`). On
the rig a batched 16-position verify reads many more experts per layer (the PR's MoE result 0.61–1.27×; Qwen3.6's MTP
gave 1.04× in Q43). Reopens with Q211-class multi-token expert kernels, or as a block guesser in the pipelined executor
(Q271) once Q162's hybrid rollback runs on the rig.

### 5.5 Parked: one shared prefix for every conversation in streams mode

Magnitude's sessions share prefix caches. In Q104's streams mode each conversation holds its own copy of the system
and tool prompt's full-attention cells: est. ~140 MB per extra Flint conversation at a 5k-token prefix (28 KiB per
token), plus reading it. Zero on the single-agent check; reopens when parallel sub-agents run on the rig.

### 5.6 A variant for Q401: a clustered draft readout

Score vocabulary clusters by their centroids, then compute exact logits only inside the top clusters. Approximate like
Q400 and Q401; compare at equal cost with Q396's acceptance tool.

### 5.7 A copy-then-fresh trace for Q342

Magnitude's `prose-shift` fixture asks the model to repeat a passage, write ~150 new words, then repeat the passage
again (Moby-Dick is in the public domain). As a trace it tests how fast an adaptive pipe-depth rule (Q342) switches
between copied and fresh text; offline in Q87's simulator. A diagnostic, no speed gain of its own.

## 6. Verdict

- **What it is:** a young, real, Apache-2.0 local engine from a Y Combinator company, with its own kernel language and
  a first-load tuner, built for single modern GPUs and Macs.
- **The speed claim:** true for what was measured, not in general. The "over 2× faster than Ollama" average comes
  from reciting a passage at 64k context with a block drafter that copies well, an 8-bit-key, 4-bit-value cache
  against a 16-bit default, two requests per point and a cold process. The engine-only gain at launch was 1.18× on CUDA, and on
  new text users and a reported re-test find it level with or behind llama.cpp and Ollama.
- **Scam:** no. The marketing overreaches; the code, harness and method page are public.
- **For Miner Llama:** it cannot run on the rig (no compute 6.1 on CUDA, no multi-GPU). Two ideas are new here: the
  exact progressive output head (5.1) and first-load kernel tuning for other Pascal cards (5.2). The 8/4-bit cache
  (5.3) extends Q210. Its drafters, confidence cutoffs, Gumbel sampling and prefix caching are already covered by
  Q271, Q409, Q342, Q96 and Q111.

## 7. Security observations

- No text in the pages and files read addresses an AI agent. The repository carries `AGENTS.md` and `CLAUDE.md` at its
  root and in `inference/` (instructions for the project's own coding agents); they were not opened. The README's
  install one-liners address human readers and were not run.
- Things that run on install or in normal use (none was run): the install script (sudo on Linux, a sudo keep-alive
  loop, a counted request to the vendor's installer endpoint, signature and checksum checks); a systemd or launchd
  service in server mode; an update checker in the CLI; model downloads from Hugging Face commits pinned in the
  catalog.

## 8. Not verified

- None of Magnitude's speed figures; the per-model split in 2.3 combines two vendor runs.
- The re-test on X reported by aiwiki.ai (the post was not found).
- How tight the progressive head's error bound is in practice, and how many rows survive each level; its exact CUDA
  arithmetic lives in a header (`lib/readout/progressive.cuh`) that was not read.
- What the internal `@magnitudedev/*` packages send over the network.
- My Pascal estimates (screen bytes, survivor gate, tuning gains on other cards, cache-format gains) are arithmetic,
  not rig measurements.

## Sources

- Repository and API: https://github.com/magnitudedev/magnitude, https://api.github.com/repos/magnitudedev/magnitude
  (and `/contributors`, `/languages`)
- Files cited above (raw views at `https://raw.githubusercontent.com/magnitudedev/magnitude/main/...`): `README.md`,
  `inference/README.md`, `inference/docs/overview.md`, `inference/docs/tuning.md`, `inference/docs/compatibility.md`,
  `inference/docs/seismic/overview.md`, `inference/docs/engine/generation.md`, `inference/docs/engine/state.md`,
  `inference/catalog/models.json`, `inference/evaluation/fixtures/README.md`,
  `inference/evaluation/bench/src/magnitude_bench/{requests,plan,cli}.py`,
  `inference/engine/kernels/kernels/{readout,clustered_readout,draft,dflash,ngram}.seismic`,
  `inference/engine/kernels/kernels/cuda/{readout_top_rows,routed_experts}.cu` and the `cuda/` listing,
  `package.json`, `desktop/package.json`, `cli/package.json`
- Issues: https://github.com/magnitudedev/magnitude/issues/82, https://github.com/magnitudedev/magnitude/issues/192
- Site and method: https://magnitude.dev, https://magnitude.dev/benchmarks, https://magnitude.dev/install.sh (read
  only), https://docs.magnitude.dev/reference.md
- Launch post on X: https://x.com/tomgreenwald/status/2110718297255670071 (found by search, not opened)
- Launch HN: https://news.ycombinator.com/item?id=49911995
- Drafter cards: https://huggingface.co/magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF,
  https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr-GGUF
- llama.cpp DFlash: https://github.com/ggml-org/llama.cpp/pull/22105; DFlash paper https://arxiv.org/abs/2602.06036
- Secondary: https://www.developersdigest.tech/blog/magnitude-self-tuning-local-inference-engine-2026,
  https://explainx.ai/blog/magnitude-open-source-inference-engine-2x-faster-than-ollama-claim-checked-2026,
  https://aiweekly.co/alerts/magnitude-yc-s25-launches-self-compiling-inference-engine-claims-92-faster,
  https://aiwiki.ai/wiki/magnitude_inference_engine (via search results; the page refused direct reads)
- This repository: [the registry](../../../quants/experiments.md), [flint_v3_plan.md](flint_v3_plan.md),
  [uzu_lessons.md](uzu_lessons.md), [cheap_mtp_draft.md](cheap_mtp_draft.md),
  [agent_prompt_cache.md](agent_prompt_cache.md), [tool_request_sampling.md](tool_request_sampling.md),
  [multi_conversation_serving.md](multi_conversation_serving.md),
  [pascal_attention_long_context.md](pascal_attention_long_context.md), [engines_other.md](engines_other.md),
  [bench/agent_check](../../../bench/agent_check/README.md)
