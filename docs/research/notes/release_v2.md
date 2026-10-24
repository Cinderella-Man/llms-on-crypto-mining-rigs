# Shipping v2: files, engine builds, licences, settings and honest results for owners of the same rig (2026-10-24)

Scope: v2 is a Flint setup for the Manli 9 × P106-090 rig that should be faster than Unsloth's UD-Q4_K_XL on stock
llama.cpp at the same quality. Thousands of these rigs exist, so the release has to work for a stranger with the same
box: download, check, run, in minutes, with the right to do so and numbers they can reproduce. Which file and which
engine pieces win is decided on the rig today (Q117 engine builds, Q125 release candidates) and is the owner's call;
this note covers everything around that decision:
- what stock llama.cpp does with our files, above all with the screened head's extra tensor, and how to ship it;
- the licences of Flint, Unsloth's importance matrix, llama.cpp and the CUDA libraries, and what the cards must say;
- how to distribute the engine, on which systems it runs, and what a quick start must check;
- the server command lines, environment switches and client configurations to ship;
- a one-page results format that reports speed and quality honestly;
- a ranked release checklist with efforts and the owner's decisions.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (licences, model cards, code, official docs, measurements in this
  repository) and secondary.
- Nothing ran on the rig or the quant machine. Model-card text, licences and GGUF headers were read over HTTP (headers
  by range requests, the first 25 MB of each file); llama.cpp sources were read at upstream `d77256999` (the fork's
  base), upstream master `0ae0e1976` (2026-10-23), and the fork's `pascal-p106-v2` (`032b130f4`) and
  `pascal-p106-v2-combined` (`b3052a4f8`) branches. File:line references are to those trees.
- "UD" is Unsloth's UD-Q4_K_XL; "the screen" is the screened output head of Q91/Q92 (`output_screen.weight`, a GPTQ
  Q4_0 copy of the head, 173 MB); "side file" is that tensor alone in its own GGUF, loaded with
  `LLAMA_SCREEN_HEAD_FILE`.

Starting point, measured ([registry](../../../quants/experiments.md)):
- **v1** (released 2026-10-23): fork branch `pascal-p106` at `eeaa59660`, README instructions to build from source;
  Miner S4-L0 named but not published. No file, binary or tag has been published yet.
- **Engines (Q117, UD, llama-bench tg128):** v2 base 34.7 tok/s on six cards and 36.5 on nine against stock 15.2 and
  16.5 and v1 31.2 and 32.6. The combined build processes prompts 1.22–1.29× faster than v2 base but decodes 3.6–9.8 %
  slower; its ablations decide which pieces ship. Served chat on the v2 fork shows no server gap and streaming costs
  nothing at 1,000 tokens (Q118).
- **Candidates:** K2sr (K2 + screen + F16 router, 2.053 GB per token) for the fork, K2h-r16 (GPTQ Q5_0 head, 2.091 GB)
  for stock llama.cpp, and UD + screen (2.060 GB) as the file to beat (decision log, 08:50 and 09:45). With a tie the
  release recommends UD + screen on the fork ([long-context mix](long_context_mix.md) §5.2).
- **The screen** is exact for the makers' sampler (Q91) and bit-identical to the full head in the engine on G3 and UD,
  in the file or from a side file (Q92). It is off by rule for any request with a grammar, which includes every tool
  request (Q55's rule, [agent workload check](agent_workload_check.md)).

## Verdict

- **Ship the screen as a side file, never inside the model GGUF.** Stock llama.cpp refuses a file with an extra
  tensor: the loader counts every tensor in the file, each architecture creates only the tensors it knows, and
  `done_getting_tensors` throws "wrong number of tensors; expected N, got N−1" (`src/llama-model-loader.cpp:1400-1406`,
  called without its `partial` flag at `src/llama-model.cpp:1873` in `d77256999` and `:1880` on master). llama.cpp
  ignores only tensors it knows but does not use. The fork's v2 base (`pascal-p106-v2`) refuses it too; only builds
  that contain Q92 load it. A side file keeps the model a standard GGUF that every llama.cpp build and app loads, costs
  173 MB to download instead of a second 25–33 GB file, and works with any Flint GGUF whose head is 150,144 × 2,048:
  the same file is exact on UD and on our mixes (Q92). If the release recommends UD, v2 redistributes no model weights
  at all, only the engine, the side file and settings. The side file ships only if Q92 is in the release engine.
- **Our mixes and the F16 router are standard GGUF.** Q4_1, Q5_0, Q8_0, F16 and F32 are core ggml types; UD itself
  ships a BF16 router. Any llama.cpp from release b11803 (Flint support, `a37cc4610`) loads them, as do apps built on
  it; Flint's original chat template parses from b11798 on (`\x` escapes, #30929). The speed gain needs the fork; on
  other GPUs and on Macs a larger simple-format file is only slower ([releasing.md](../../../quants/releasing.md)).
- **Licences allow everything v2 needs.** Flint's `LICENSE` is the unmodified Apache-2.0 text, its model card declares
  `apache-2.0` and adds no use policy, and the weights carry no NOTICE file. Quantized derivatives may be
  redistributed, commercially too, under four duties: include the licence, mark the changes prominently, keep the
  attribution, and use "Flint" and "Recursive" only to describe the origin (§4, §6). Unsloth's repository, importance
  matrix included, is Apache-2.0 as well; v2 credits it and redistributes nothing of Unsloth's. llama.cpp is MIT, and
  NVIDIA's CUDA EULA lists `libcudart.so`, `libcublas.so` and `libcublasLt.so` as redistributable (Attachment A).
- **Two blockers found on the way:**
  - **This repository has no licence file**, so the scripts and recipes the release depends on cannot legally be
    reused. The owner picks one (MIT, as llama.cpp, is the natural choice).
  - **GGUF headers inherited from Unsloth's files are wrong for a release.** UD and Unsloth's Q8_0 say
    `general.license = other`, `general.license.name = flint-preview`, `general.name = Flint Preview` and
    `general.quantized_by = Unsloth`; files composed from them inherit all of it, since `compose_gguf.py` takes metadata
    from its base file. Every released GGUF gets its header checked and rewritten (licence, name, author, base model)
    before upload, plus `general.sampling.min_p = 0`, so that stock llama-server applies the makers' sampling without
    flags (Flint's `generation_config.json` has temperature, top-p and top-k but no min-p, and llama-server's default
    min-p is 0.05).
- **Distribute the engine as prebuilt tarballs on the fork's GitHub Releases, with NVIDIA's runtime libraries in a
  second asset, as upstream llama.cpp does** (`cudart-llama-…-bin-ubuntu-cuda-….tar.gz`,
  `.github/workflows/release.yml:435-452`). The binaries need glibc ≥ 2.38 and GCC 13's libstdc++ (measured on the v2
  build: `GLIBC_2.38`, `GLIBCXX_3.4.32`, no AVX), so they run on Ubuntu 24.04, 25.10 and 26.04 LTS and Debian 13, not
  on Ubuntu 22.04 or Debian 12. Every one of those Ubuntu releases ships the proprietary `nvidia-driver-580`; the
  `-open` modules support only Turing and later. With the CUDA libraries bundled, the rig needs no CUDA toolkit
  (~7–9 GB on a 64 GB SSD), and a 25–33 GB model fits on the stock SSD (est.). No Docker image: it needs the NVIDIA
  container toolkit and a 2.4 GB base image on a 4 GB-RAM host. Build-from-source stays the documented fallback. v1
  gets the same treatment: a tag on `eeaa59660` with its own tarball.
- **Settings:** one command line per use case on nine cards (section 4): the makers' sampling (T 1.0, top-p 0.95,
  top-k 20, min-p 0), `-bs`, `--ctx-checkpoints 4`, `--cache-ram 0`, `LLAMA_NO_MOE_DOWN_RESCALE=tg` for every file on
  v2, the screen switches for chat only, and `-b 2048 -ub 512` only if the on-card mask (Q110) is in the release
  engine (otherwise v1's `-b 512 -ub 128`). Nine cards beat six even at short context on v2 (36.5 against 34.7 tok/s).
  The pi and OpenCode configurations are the makers' own, pointed at the rig through an SSH tunnel.
- **Reporting:** one page that compares files with the same head treatment, reports chat and tool requests
  separately, says the requests were streamed, gives 512-token KLD and 8k agent-text KLD split into written and read
  tokens against BF16, task checks with intervals and margin verdicts, time per correct answer, and the exact engine
  commits and file hashes (section 5). Reputable quant releases publish KLD and recipes but no intervals, no task
  checks across serving paths and no speed method; v2 must not omit those.
- **Top of the checklist:** (1) the owner's file and engine decision, (2) a licence for this repository,
  (3) release tarballs with bundled libraries and licence notices, (4) GGUF header fixes, the model card and the
  Hugging Face upload, (5) validated command lines and a quick-start check. Est. 3–4 working days after the decision,
  plus upload time.

## Release checklist, ranked

Ranked by what blocks the release, then by order of work. Efforts are (est.) and count work, not waiting.

| # | Item | Blocks the release? | Effort (est.) | Owner decision or action | Source |
|---|---|---|---|---|---|
| 1 | **What ships:** release file (K2sr, K2h-r16, or none with UD + screen) from Q125, Q41 and the 8k agent-text rule; engine pieces from Q117's ablations; whether Q92 (screen) and Q110 (on-card mask) are in | yes | rig blocks already running; 0.5 h to record | **yes**: the release decision | decision log 08:50, 09:45, 11:00; [long-context mix](long_context_mix.md) §5.2 |
| 2 | **A licence for this repository** (none today; scripts, recipes and tools are not reusable without one) | yes | 15 min | **yes**: which licence (MIT for code; optionally CC BY 4.0 for docs) | repository tree; GitHub API reports no licence |
| 3 | **Engine tarballs:** v2 at the tagged release commit and v1 rebuilt from `eeaa59660`, each with `BUILDINFO.txt`, `LICENSE` (MIT), third-party notices, `SHA256SUMS`; a separate CUDA-runtime asset (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` from the build image) with an `$ORIGIN` RPATH; `--list-devices` and one chat request on the rig | yes | 0.5–1 day (script change, two builds on the quant machine, 30 min rig test) | **yes**: bundle NVIDIA's libraries under the CUDA EULA, or require NVIDIA's apt packages | [build script](../../../scripts/build_llamacpp_docker.sh); upstream `release.yml:404-452`; CUDA EULA §1.1.2, Attachment A |
| 4 | **GGUF header fix** of every released model file (licence `apache-2.0`, name, `quantized_by`, base model, `general.sampling.min_p = 0`; keep `quantize.imatrix.*`), then `sha256sum`; the side file gets the same header fields | yes | 1–2 h per file (a full rewrite with gguf-py's metadata script, then a load test) | **yes**: the name and author written into the file | headers of UD, Unsloth Q8_0, our UD-mirror (§1.5); `compose_gguf.py` |
| 5 | **Model card(s)** with `LICENSE`: base model and revision, Apache-2.0 §4 change list, credits, not-affiliated line, who it is for, required build, settings, results summary and links | yes | 0.5 day | approve the text | Apache-2.0 §4, §6; [releasing.md](../../../quants/releasing.md); terelabs' change notice |
| 6 | **Hugging Face:** account or organisation, repository name, login on the uploading machine, upload of the GGUF (if any), the side file, `SHA256SUMS`, client configs | yes | 1 h plus upload time (30 GB at 50–100 Mbit/s is 40–80 min) | **yes**: account, names, the login itself | HF CLI docs; storage-limit docs |
| 7 | **GitHub releases on the fork:** tags (e.g. `p106-v1.0` on `eeaa59660`, `p106-v2.0` on the release commit), assets, notes; immutable releases on; default branch a P106 branch instead of upstream's `master`; issues on | yes | 1 h | **yes**: tag names, repository settings | GitHub docs (releases, immutable releases); fork has no releases, issues off |
| 8 | **Validated command lines** on the rig: chat and agent at 262,144 tokens on nine cards with the release engine and file; `-ub 512` fit and host MemAvailable; the screen's +173 MB on the last card with the default split; `LLAMA_NO_MOE_DOWN_RESCALE=tg` on each released file | yes | 1–2 h of rig time | rig time | §4; Q117; [gptq_head](../../../quants/gptq_head/README.md) |
| 9 | **Quick start** page and a pass/fail check script (driver branch, nine P106-090, libraries resolve, RAM, free disk, model checksum, source read speed) | yes | 1 day | — | §3.4; [setup.md](../../setup.md); `rig_inventory.sh` |
| 10 | **One-page results** in the format of §5.2, from Q117, Q118, Q125, Q55, Q41, Q106/Q126 | yes | 1 day once the data exist | approve claims | §5 |
| 11 | **README and tags of this repository:** v2 section, v1 kept linked, tag v1 at `1a35d59`, merge `v2` into `main` | yes | 2–3 h | **yes**: merge and tags | repository history |
| 12 | Client configs: the makers' `pi-models.json` and `opencode.json` with the tunnel's URL, context = `-c`, a 32,768–65,536 output cap | no (agent users need it) | 1–2 h | — | makers' GGUF repository |
| 13 | Multi-turn decision run (799 tasks × K ≥ 2, E3 at 65,536 tokens) for the shipped file | no, if the card states what was tested; yes for any multi-turn claim | 1–2 days of the quant machine | **yes**: booking | Q41; [task-level checks](task_level_quality_checks.md) |
| 14 | Second file for stock llama.cpp (K2h-r16) next to K2sr | no | +1 h card, +30 GB upload | yes | decision log 09:45 |
| 15 | Upstream PRs for the screen and FP32 sums (would let the side file work on stock builds) | no | days | — | [upstream status](upstream_pascal_patches.md) |
| 16 | Docker image | no (not recommended) | — | — | §3.1 |

## 1. Files and compatibility

### 1.1 What stock llama.cpp does with an extra tensor

- **The rule.** The loader indexes every tensor of the file (`n_tensors = weights_map.size()`,
  `src/llama-model-loader.cpp:719` in `d77256999`). Each architecture's `load_tensors` then calls `create_tensor` for
  the names it expects; each call counts one tensor. At the end `done_getting_tensors` compares the counts and throws
  if fewer were created than the file holds, unless `partial` is set (`:1400-1410`). On `d77256999` and on master
  `0ae0e1976` no caller sets it: `ml.done_getting_tensors()` has no argument (`src/llama-model.cpp:1873`, master
  `:1880`). The `partial` path came with MTP support (#22673) and is unused.
- **What is tolerated.** A tensor the architecture asks for but does not use (operation NONE, or the `TENSOR_SKIP`
  flag) is skipped with "model has unused tensor … -- ignoring" (`:1188`). A tensor the architecture never asks for,
  such as `output_screen.weight` for Flint, is not.
- **What a user sees** with a screen-inside file on stock llama.cpp, LM Studio or any other llama.cpp-based app (est.,
  same loader): the load fails with `done_getting_tensors: wrong number of tensors; expected 780, got 779` (UD has 779
  tensors). Unknown metadata keys such as `flint.output_screen.top_m` are ignored.
- **In the fork.** Builds with Q92 create the screen as an optional tensor of the same shape as `output`
  (`src/llama-model.cpp:1862-1863` on `pascal-p106-v2-combined`), so they load files with and without it. Without
  `LLAMA_SCREEN_HEAD=1` the tensor is still loaded onto the card that holds the head: 173 MB of VRAM for nothing. The
  v2 base without Q92 (`pascal-p106-v2`, `032b130f4`) has no such tensor and refuses the file as stock does.
- **The side file** joins the loader like an extra split (`src/llama-model-loader.cpp:678-703` on the combined
  branch): only `output_screen.weight` is taken, a model that already has the tensor is refused, and the side file's
  metadata is not read (`get_key` reads the main file only, `:422-436`). So `flint.output_screen.top_m` in the side
  file has no effect; M is 256 (`src/llama-hparams.h:237`) unless `LLAMA_SCREEN_HEAD_M` overrides it. Stock builds and
  v1 ignore `LLAMA_SCREEN_HEAD_FILE`, so the side file can never break a load.

### 1.2 Screen inside, side file, or both

| Option | Stock llama.cpp and apps | Fork without Q92 (v1, v2 base) | Fork with Q92 | Download | Verdict |
|---|---|---|---|---|---|
| Screen inside the GGUF | refuses to load | refuses to load | loads; screen on with `LLAMA_SCREEN_HEAD=1` | a second 25–33 GB file per candidate | no |
| Side file only | loads the model; env ignored | loads; env ignored | loads; screen on with both variables | 173 MB, once, for every Flint file with the usual head | **yes** |
| Both | — | — | — | doubles the uploads for convenience | no |

- **One side file for all files.** Q92 checked the same GPTQ Q4_0 screen, calibrated on another quant's hidden states,
  on G3 inside the file, G3 from the side file and UD from the side file: candidate logits bit-identical to the full
  head on 262,144 of 262,144, greedy and seeded sampling identical. The exact rows come from each file's own head, so
  the screen only ranks.
- **When it pays.** Only for files with a Q8_0 head (UD, K2sr, G3): it reads 174 MB instead of 327 MB per token. On a
  GPTQ Q5_0 head (K2h, K2h-r16, 211 MB) it saves 37 MB and its exactness there is untested. The model card says which
  files it is for.
- **Name.** Keep the name used in the docs, `flint-output-screen-gptq-q4_0.gguf`, and add the model revision it was
  made from to the card.

### 1.3 The F16 router and the Q4_1/Q5_0/Q8_0 mixes

- **Types.** Everything in the candidates is a core ggml type with CPU, CUDA, Vulkan and Metal kernels: Q4_1, Q5_0,
  Q8_0 for weights, F16 for the router, F32 for norms. UD already ships a BF16 router (type 30 in its header), so a
  non-F32 router is nothing new for llama.cpp.
- **Minimum build.** b11803 for the Flint architecture; b11798 for the original chat template (earlier builds fail on
  its `\x00` filter, which is why terelabs ship a modified template). Our files keep the original template.
- **The rescale.** Stock llama.cpp always applies Flint's MoE down-projection rescale, which is safe for every file.
  In v1, `LLAMA_NO_MOE_DOWN_RESCALE=1` gives NaN with Q4_0/Q4_1/Q5_0/Q5_1 down-projections, and K2's are Q5_0 in
  layers 2–45. In v2, `tg` drops the rescale only for one-token steps, where compute 6.1 runs the Pascal kernels with
  FP32 sums (`src/llama-graph.cpp:1829-1841`); it is unsafe on any other GPU and with `GGML_CUDA_MMVQ_PASCAL=0` or
  `GGML_CUDA_MMVQ_PASCAL_MC=0` (Q117).
- **Where FP16 overflows without the rescale.** The down-projection inputs of Flint's last MoE layers are large. On a
  long agent conversation (`multi_turn_long_context_10`, K2h in FP32 on the CPU, first 2,048 positions), routed layer
  46 reaches 1.6 × 10⁴ with 32-value block sums of 3.8 × 10⁴, layer 47 3.5 × 10³, the shared expert of layer 45
  4.5 × 10³; routed layers 2–45 stay below 450. Three FP16 quantities on that path can overflow:
  1. block_q8_1's FP16 sum of 32 activations (limit 65,504), read by upstream `mul_mat_vec_q` for Q4_0, Q4_1, Q5_0 and
     Q5_1 and by MMQ's DS4/D2S6 layouts (Q4_1, Q5_1, Q4_K, Q5_K, Q2_K, and Q4_0 without tensor cores);
  2. block_q8_1's FP16 scale, max/127 (limit max ≤ 8.3 × 10⁶), read by every type;
  3. cuBLAS's FP16 GEMM, which converts the activations to FP16 and accumulates in FP16 (`CUBLAS_COMPUTE_16F`).
     ggml takes it instead of MMQ for batches of 64 or more tokens on GPUs with FP16 tensor cores but no int8 ones
     (V100). That is how K2h, and G3 with UD's attention, gave NaN on the V100s with the rescale off (Q126, Q127).
- **On the P106 (compute 6.1) `tg` reaches none of them.** Prompt batches keep the rescale; without FP16 tensor cores
  they always run MMQ, which for Q5_0 and Q8_0 keeps float scales (D4 layout) and int32 dp4a sums. A one-token step's
  routed down projection is a six-row `mul_mat_id`, run by `mul_mat_vec_q_pascal_mc` (the persistent MoE kernel uses
  the same functions): each block's sum is d8 times the sum of its int8 values, in FP32, and block_q8_1's FP16 sum is
  never read. The one FP16 value left, d8, has 500× headroom over the largest measured input. The magnitudes belong
  to a token and a layer, not to its position, so a long context does not change this. The shared expert, which the
  rescale never covers, reads no FP16 sum on these paths either. **Release setting: `LLAMA_NO_MOE_DOWN_RESCALE=tg` on
  the P106 for every file, never `=1`; on any other GPU leave it unset.**
- **Speed off the fork.** Simple formats decode a little faster than K-quants even on stock llama.cpp here (S4-L0
  16.06 against UD 14.20 tok/s in the v1 chat benchmark), but the 2× comes from the fork. On modern GPUs and Macs the
  larger file is slower; the card says so.

### 1.4 How others ship fork-only tensors and side files, and how they name files

| Publisher | File names | Extra tensors and side files | Fork-only types | What the card records |
|---|---|---|---|---|
| Unsloth ([flint-GGUF](https://huggingface.co/unsloth/flint-GGUF)) | `<model>-UD-<TYPE>.gguf` ("UD" = Unsloth Dynamic), `<model>-Q8_0.gguf`, splits in a folder as `-0000N-of-0000M.gguf` | vision projectors as side files (`mmproj-F16.gguf`); MTP heads inside the GGUF but in a separate repository (`Qwen3.6-35B-A3B-MTP-GGUF`), once mainline could load them; the importance matrix as `imatrix_unsloth.gguf_file`, an extension no tool mistakes for a model | none | base model, licence, KLD and top-1 against size as plots, a link to the run guide; the selection rule is not published |
| bartowski ([RecursiveSI_flint-GGUF](https://huggingface.co/bartowski/RecursiveSI_flint-GGUF)) | `<Org>_<model>-<TYPE>.gguf`; `_S`/`_M`/`_L` say how much of the body stays at the base type (90 / 70 / 50 %, the rest placed by a computed layout) | `-imatrix.gguf`, the calibration corpus as `-calibration-v6.txt`, per-file `layouts/<file>.tensor-types.txt` (the exact `--tensor-type-file`) | none | llama.cpp release used (b11804) and "that release or newer"; `quantized_by`, `base_model`, `base_model_relation: quantized`; KLD with ± for three files |
| ubergarm (ik_llama.cpp quants, e.g. [Qwen3.8-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.8-27B-GGUF)) | `<model>-<TYPE>.gguf`, with MTP in the name when the head is inside | MTP head inside the file | IQ*_K/KS/KT types; tag `ik_llama.cpp`; a note at the top that most files need the fork and which ones run on mainline | recipes as tensor regexes, quantize, imatrix and perplexity logs, the imatrix `.dat` |
| ik_llama.cpp itself | — | — | — | publishes no binaries; Docker images on ghcr.io; third parties publish Windows builds |
| terelabs ([flint-GGUF](https://huggingface.co/terelabs/flint-GGUF)) | `flint-<TYPE>.gguf` | its imatrix, a `comparison.csv` | none | KLD mean/median/p99, BFCL with McNemar tests and measurement error, an Apache-2.0 §4(b) change list, a not-affiliated line |

- **The pattern:** nobody puts an engine-specific tensor into a file aimed at mainline users. Extra tensor sets go into
  separate repositories or side files, and files that need a fork say so in the first lines and in a tag.
- **For v2 (owner decides):** a repository named after the model and the hardware (e.g. `flint-P106-GGUF`), a file name
  with the recipe (e.g. `flint-P106-K2.gguf`), the side file next to it, a `p106` tag. A recipe name tells a reader
  more than a pseudo-type such as "Q4_K_XL": the mixes hold Q5_0 and Q8_0 with some Q4_1.

### 1.5 Metadata the release files must carry

Headers read by range requests:

| File | `general.name` | `general.license` | `general.quantized_by` | `quantize.imatrix.dataset` | `general.sampling.*` |
|---|---|---|---|---|---|
| Unsloth UD-Q4_K_XL (revision `20468ed`) | Flint Preview | other, name `flint-preview` | Unsloth | `unsloth_calibration_flint-preview.txt` (687 chunks of 512) | temp 1.0, top-p 0.95, top-k 20 |
| Unsloth Q8_0 | Flint Preview | other, `flint-preview` | Unsloth | same | same |
| Our UD-mirror v1 (made from Unsloth's Q8_0) | Flint Preview | other, `flint-preview` | Unsloth | same | same |
| The makers' Q4_1 | Flint INT4 Preview | — | — | — | same |

- **Consequences.** A file composed from Unsloth-derived donors claims to be Unsloth's and to fall under a preview
  licence. The public Flint is Apache-2.0 (Hugging Face repository `RecursiveSI/flint`, revision `2a247be`), and so is
  Unsloth's repository card.
- **Before upload:** `general.name`, `general.license = apache-2.0`, `general.quantized_by`, `general.repo_url`,
  `general.base_model.*` (name, organisation, repository URL of `RecursiveSI/flint`), and
  `general.sampling.min_p = 0` (read by `common/common.cpp:1124-1146`; user flags override it). Keep
  `quantize.imatrix.*`: it names the importance matrix, which is part of the attribution.
- **Check after the rewrite:** byte-identical tensor data (tensor hashes against the source), a load on stock
  llama.cpp, a load on the release engine, then `sha256sum` for the card.

## 2. Licences

### 2.1 Flint

- **The licence.** `RecursiveSI/flint`'s `LICENSE` is the Apache License 2.0 text, identical to apache.org's apart
  from a trailing newline; the model card's metadata says `apache-2.0`; the card's "License and reporting" section
  says the weights are under Apache 2.0 and hosted API terms are separate. The makers' GGUF (`flint-int4-gguf`) and
  Unsloth's repository carry the same licence. The only NOTICE in the repository belongs to the SGLang kit, not to
  the weights.
- **What it allows:** reproduction, modification and redistribution of derivative works, commercially or not, under
  Apache-2.0 or other terms for the modifications (§2, §4). There is no use policy in the licence; the card's
  "Limitations and responsible use" section is advice.
- **What it requires of a quantized derivative (§4):**
  - a copy of the licence with the files;
  - prominent notices in modified files stating the changes: converted to GGUF, requantized per the recipe with the
    named importance matrix, router in F16, screen tensor (side file) as a GPTQ Q4_0 copy of the head;
  - all copyright and attribution notices of the source kept (Flint has no NOTICE file to carry).
- **Names (§6).** No trademark rights are granted beyond describing the origin. "Flint" in a repository name is the
  customary descriptive use (Unsloth, bartowski, terelabs all do it); the card adds that the release is not affiliated
  with or endorsed by Recursive, as terelabs' does.

### 2.2 Unsloth's importance matrix and UD-Q4_K_XL

- **Our files' rounding** used `imatrix_unsloth.gguf_file` from `unsloth/flint-GGUF`, a repository whose card and
  `LICENSE` say Apache-2.0. The matrix itself carries no licence field (its header holds only the dataset name and
  chunk counts) and was computed on Unsloth's unpublished calibration text.
- **What v2 redistributes:** none of Unsloth's files. The card credits the matrix by name and repository, and the
  GGUF keeps `quantize.imatrix.file`/`.dataset`. If the release recommends UD, users download it from Unsloth at a
  pinned revision (`20468edfc38ab43d6914cc88b08cbecb05d0fc62`; UD is 25,054,688,704 bytes, sha256 `63c2ecc4…072f` per
  the Hub's LFS record, the same size as the file measured here).
- **An alternative the owner may prefer:** our own importance matrix, which scores the same (Q12), from a calibration
  text with a stated licence. It removes the dependency at the cost of rebuilding and rescoring the file.

### 2.3 llama.cpp, the fork and the bundled CUDA libraries

- **llama.cpp** is MIT ("Copyright (c) 2023-2026 The ggml authors"); the fork inherits it (GitHub reports MIT for
  `Cinderella-Man/llama.cpp`). Its vendored code is permissive: cpp-httplib and nlohmann/json under MIT
  (`vendor/cpp-httplib/LICENSE`, `licenses/LICENSE-jsonhpp`), stb_image MIT or public domain, miniaudio public domain
  or MIT-0, sheredom's subprocess.h the Unlicense. The licences are embedded only into the multi-command `llama`
  binary (`cmake/license.cmake`; target `llama-app` in `app/CMakeLists.txt`, `llama licenses`), which the build
  script does not build, so the tarball carries them as files.
- **The fork's own changes** are released under the same MIT licence; the release notes say so in one line.
- **NVIDIA's libraries.** The CUDA EULA (current text, v13.4) lets developers distribute the portions listed in
  Attachment A inside an application, under conditions: material additional functionality beyond the SDK, the
  libraries accessed only by the application, and distribution terms consistent with the EULA (§1.1.1 item 3,
  §1.1.2). Attachment A lists `libcudart.so` and `libcublas.so`/`libcublasLt.so` for Linux. llama.cpp's own release
  workflow ships exactly these three as a separate asset. The CUDA-runtime asset carries a notice naming the files,
  their version (CUDA 12.9.x) and the EULA.
- **System libraries** (`libgomp`, `libstdc++`, `libc`) come from the distribution and are not bundled.

### 2.4 This repository

- No licence file exists, so the default is all rights reserved: anyone may read the scripts but not legally copy or
  adapt them. The project's purpose is reuse, so this blocks the release.
- MIT matches llama.cpp and keeps patches portable between the fork and this repository; the owner decides.

### 2.5 What the model card and the README must say

**Model card (Hugging Face):**
- metadata: `license: apache-2.0`, `base_model: RecursiveSI/flint`, `base_model_relation: quantized`, `quantized_by`,
  `pipeline_tag: text-generation`, tags (`gguf`, `llama.cpp`, `pascal`);
- the first lines: what the file is for (P106-090 and other 5-SM Pascal cards with this project's engine), what it is
  not for (stock llama.cpp gains little, modern GPUs and Macs lose), and the required build;
- the change list of §2.1, the source revision, the recipe, the importance matrix and its licence;
- credits: Recursive (Flint, Apache-2.0), Unsloth (importance matrix, Apache-2.0), llama.cpp (MIT);
- "not affiliated with or endorsed by Recursive";
- the makers' limitations, by reference to their card;
- results in the format of §5.2, files with sha256, settings of §4.

**README of this repository:**
- the repository's licence;
- the v2 quick start, with v1 kept and linked;
- third-party licences of the engine tarballs (MIT, the vendored notices, NVIDIA's EULA for the runtime asset).

## 3. Engine distribution

### 3.1 Tarball, Docker image or source

| Way | What the user needs | Size | Fit for the rig | Verdict |
|---|---|---|---|---|
| **Prebuilt tarball + CUDA-runtime asset** (GitHub Release) | driver 580, glibc ≥ 2.38 | 176 MB today (seven tools and `test-backend-ops`, each binary with its own CUDA kernels); runtime libraries est. 0.5–0.7 GB unpacked (cuBLASLt is most of it; NVIDIA's cuBLAS 12.9.1 archive is 934 MB with static libraries) | no compiler, no toolkit, nothing in the system changed | **primary** |
| Docker image (e.g. on `nvidia/cuda:12.9.2-runtime-ubuntu24.04`, ~2.4 GB compressed) | Docker, NVIDIA container toolkit | 2.5+ GB | package changes on the rig, a daemon on a 4 GB host, the 64 GB SSD ([ops note](ops_and_benchmarking.md)) | no |
| Build from source on the rig | CUDA 12.9 toolkit (~7–9 GB), cmake | — | slow on two cores | fallback, documented in setup.md |
| Build on another PC with `build_llamacpp_docker.sh` | Docker on any x86-64 Linux PC | — | as today | for contributors |

- **Layout (as upstream):** `llama-p106-v2.0-bin-ubuntu-cuda12.9-sm61.tar.gz` (binaries, `LICENSE`, notices,
  `BUILDINFO.txt`) and `cudart-llama-p106-v2.0-bin-ubuntu-cuda12.9.tar.gz`, unpacked into one folder; binaries built
  with `-DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON` (`release.yml:410-421`). Building ggml as
  shared libraries would store the CUDA kernels once instead of in every binary (est. 2–3× smaller binaries).
- **Contents:** `llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, plus `llama-screen-check` if the screen
  ships; `test-backend-ops` in a separate test asset.
- **A smaller alternative:** upstream's multi-command `llama` binary (`app/llama.cpp`: `serve`, `cli`, `bench`,
  `perplexity`, `licenses`, …) links those tools into one executable, so the CUDA kernels are stored once and the
  licences are printed by the binary itself. The documented commands would change from `llama-server` to
  `llama serve`; the fork's own tools stay separate.
- **Upstream's own Linux CUDA 12.8 release** is built for a broad architecture set with CPU variants chosen at run time
  (`release.yml:404-421`), so it may serve as a no-build stock baseline on the rig (PTX for compute 6.1; not tested).

### 3.2 Which systems the tarball runs on

| System | glibc | libstdc++ ≥ 3.4.32 | NVIDIA 580 in the distribution | Runs |
|---|---|---|---|---|
| Ubuntu 22.04 | 2.35 | no | — | no |
| Ubuntu 24.04 LTS | 2.39 | yes (GCC 14) | `580.178.04` | yes (the tested system) |
| Ubuntu 25.10 | 2.42 | yes | `580.159.03` | yes (est.) |
| Ubuntu 26.04 LTS | 2.43 | yes | `580.178.04` | yes (est.) |
| Debian 12 | 2.36 | no | — | no |
| Debian 13 | 2.41 | yes | not checked | yes (est.) |

- **The binary's floor** is measured from the v2 build: highest symbols `GLIBC_2.38`, `GLIBCXX_3.4.32`,
  `CXXABI_1.3.13`, zero AVX instructions; it needs `libgomp.so.1`, `libcuda.so.1` (driver) and the three CUDA
  libraries.
- **Driver.** The 580 branch is the last for Pascal and a long-term-support branch with end of life in June 2028
  (NVIDIA's datacenter driver table). Use the proprietary `nvidia-driver-580`; NVIDIA's open kernel modules support
  Turing and later only. The driver reports CUDA 13.0 and runs CUDA 12.9 binaries.
- **Other cards.** A rig with P106-100 cards (10 SMs, also compute 6.1) runs the same binaries; the kernels and
  splits were tuned on 5-SM cards and it decodes faster (the CUDA scoreboard puts the P106-100 at ~1.56× the
  P106-090; [community reports](quant_speed_d_community.md)). Untested. The quick start names the card it found.

### 3.3 GitHub releases, checksums, keeping v1

- **Limits:** each asset under 2 GiB, up to 1,000 assets per release, no total-size or bandwidth limit. GitHub
  computes and shows a SHA-256 digest for every asset; immutable releases also lock assets and tag and produce a
  verifiable release record.
- **Releases:** `p106-v1.0` on `eeaa59660` (the v1 engine of the chat benchmark), `p106-v2.0` on the release commit
  of `pascal-p106-v2`, each with both assets, `SHA256SUMS`, and notes that link the results page and this
  repository's setup and quick start. v1 stays downloadable as it was measured; its branch stays frozen.
- **Discoverability:** the fork's default branch is upstream's `master` and its issues are off. Visitors should land
  on the P106 branch; issues give owners of other rigs a place to report.
- **Models:** the Hugging Face Hub records each file's sha256 as its LFS object ID and shows it per file;
  `hf cache verify <repo> --local-dir <dir>` checks local files against it. The card also prints the hashes.
  Recommended per-file size on the Hub is under 200 GB, so 25–33 GB files need no splitting.

### 3.4 What the quick start checks

A read-only `p106_check.sh` that prints PASS/FAIL per line (the existing [rig_inventory.sh](../../../scripts/rig_inventory.sh)
stays the full inventory):

| Check | Command | Pass |
|---|---|---|
| Driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | 580.x on every card; module not `-open` (`modinfo nvidia`, license NVIDIA) |
| Cards | `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv` | nine "P106-090" with ~6 GB and < 50 MiB used; P106-100 reported as a different variant |
| Engine | `ldd bin/llama-server`, `bin/llama-server --list-devices` | no "not found"; CUDA0–CUDA8 with ~6068 MiB |
| CPU | `grep -c avx /proc/cpuinfo` | informational only: the build needs no AVX |
| Host memory | `/proc/meminfo` | MemTotal ≥ 3.7 GiB; MemAvailable ≥ 2 GiB before start (est.); the 4 GB note on `--cache-ram 0` and `-ub` |
| Disk | `df -h` on the model folder | free space ≥ file size + 1 GB; ext4 for USB sticks |
| Model source speed | `dd if=<model> of=/dev/null bs=16M count=128 iflag=direct` | ≥ 400 MB/s keeps a 25 GB load near one minute (the nine links take ~675 MB/s together, 54 GB in ~80 s); a USB 2 port turns it into ~12 minutes (est.) |
| Model file | `sha256sum -c SHA256SUMS` (est. 1.5–3 min for 25 GB on the Celeron) or `hf cache verify` | matches the card |

The page itself (est. 10 minutes plus the download):
1. Ubuntu 24.04 or later with `nvidia-driver-580` ([setup.md](../../setup.md)); `nvidia-smi` lists nine cards.
2. Download both engine assets, `sha256sum -c`, unpack into one folder, `--list-devices`.
3. Download the model at a pinned revision (and the side file), verify; 25 GB take ~35 min at 100 Mbit/s and
   ~4 min at 1 Gbit/s (est.).
4. Run the check script, then the chat command of §4; open the built-in web UI through an SSH tunnel.

## 4. Settings to ship

### 4.1 Server command lines

All on the release engine, nine cards, loopback only. The makers' own llama.cpp command uses the same sampling and
`-c 262144 -np 1 --cache-ram 0 -fa on` (their GGUF card).

**Chat (one user, long reasoning, full context):**

```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg \
LLAMA_SCREEN_HEAD=1 LLAMA_SCREEN_HEAD_FILE=flint-output-screen-gptq-q4_0.gguf \
bin/llama-server -m <file>.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 \
  -b 2048 -ub 512 -bs --ctx-checkpoints 4 --cache-ram 0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

**Agent and tool use (pi, OpenCode):** the same without the two screen variables. Every tool request carries a
grammar, which turns off both backend sampling and the screen for that request, so the screen would only hold 173 MB
of VRAM. Tool requests then decode est. 3.2–3.5 ms per token slower than chat (Q55, on today's rig runbook); Q67
would close that gap.

**Long context and the 4 GB host:**
- `-b 2048 -ub 512` only if the on-card mask (Q110) is in the release engine and the rig check of item 8 passes;
  otherwise `-b 512 -ub 128`, v1's setting, which keeps the pinned attention-mask copies at 262,144 tokens within the
  host's memory. Between them, `-ub 256`.
- `--cache-ram 0` always; `--ctx-checkpoints 4` keeps a cached conversation reusable (each checkpoint holds the
  sliding-window cells, ~52–88 MiB of host memory); the checkpoint policy for long agent sessions is Q132's pilot
  question.
- f16 KV cache (the default); a q8_0 cache is slower on these cards and saves little
  ([attention at long context](pascal_attention_long_context.md)).

**Card counts:**
- Nine cards are the default: on v2 they decode faster than six even at short context (36.5 against 34.7 tok/s,
  Q117), and only nine hold a 25–33 GB file plus 7 GiB of f16 KV at 262,144 tokens.
- Six cards (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`) leave three cards free for another model: UD up to ~128k tokens, the
  K2-class files up to ~32k (est., from the files' sizes and 28 KiB of KV per token).
- A Q8_0 file needs `-ts 6,6,5,5,6,5,5,6,6` at full context (v1). The screen adds 173 MB to the last card; if that
  card runs out, move one step of the split off it.

**Stock llama.cpp** takes the same flags; it ignores the environment variables, always rescales, and has no screen.

### 4.2 Environment switches

| Switch | Engine | Use | Never |
|---|---|---|---|
| `LLAMA_NO_MOE_DOWN_RESCALE=tg` | v2 | every file, on the P106 | on other GPUs; with `GGML_CUDA_MMVQ_PASCAL=0` or `GGML_CUDA_MMVQ_PASCAL_MC=0` |
| `LLAMA_NO_MOE_DOWN_RESCALE=1` | v1, v2 | only files whose expert down-projections are Q8_0, Q6_K, IQ4_XS, IQ4_NL or MXFP4 (Unsloth Q8_0, S4-L0) | UD, K2, G3 (NaN) |
| `LLAMA_SCREEN_HEAD=1` + `LLAMA_SCREEN_HEAD_FILE=<side file>` | v2 with Q92 | chat with Q8_0-head files (UD, K2sr) | agent-only servers (no effect, 173 MB of VRAM) |
| `LLAMA_SCREEN_HEAD_M` | v2 with Q92 | leave unset (256) | — |
| The Q117 off switches (`GGML_CUDA_MMQ_MOE_LEGACY`, `LLAMA_GRAPH_REUSE_WAIT`, `LLAMA_GPU_KQ_MASK=0`, `GGML_CUDA_NORM_Q8=0`, `GGML_CUDA_MMVQ_PASCAL_MC=0`) | combined build | diagnosis only | in the quick start |

### 4.3 Clients: pi and OpenCode

- **The makers' files** (`pi-models.json`, `opencode.json` in `RecursiveSI/flint-int4-gguf`) point at
  `http://127.0.0.1:8080/v1`, declare a 262,144-token context and a 65,536-token output budget, and keep
  `reasoning_content` in the history (OpenCode `interleaved.field`, pi's provider replays it). v2 ships them with the
  context matched to `-c` and points them at an SSH tunnel to the rig (`ssh -N -L 8080:127.0.0.1:8080 <rig>`).
- **Sampling lives on the server.** pi sends a temperature only when set and never top-k, top-p or min-p; OpenCode's
  agent entries send temperature 1.0 and top-p 0.95. Top-k 20 and min-p 0 must therefore come from the server's
  flags or the file's `general.sampling.*` keys (§1.5).
- **Output cap.** The makers allow 65,536 tokens; the agent check uses 32,768, the makers' minimum for hard problems,
  so that a loop cannot run for half an hour.
- **Compaction.** pi compacts at its context window minus 16,384 tokens, and on the rig that costs one full re-read
  of the summarized span (est. 12–50 minutes for 100k tokens, [agent workload check](agent_workload_check.md)): set
  the client's window equal to `-c`.
- **Exposure.** llama-server stays on loopback; serving a LAN needs `--api-key` and a firewall rule.

## 5. Honest reporting

### 5.1 What reputable releases include and omit

| Release | Includes | Omits |
|---|---|---|
| Unsloth Flint | KLD and top-1 against size (plots), the importance matrix | numbers in text, intervals, the reference and text length, task checks, any speed method |
| bartowski Flint | llama.cpp release, calibration corpus, per-file tensor layouts, KLD ± for three files | task checks, speed, hardware |
| ubergarm | recipes, quantize/imatrix/perplexity logs, the fork requirement up front | task checks, speed method |
| terelabs Flint | KLD mean/median/p99, BFCL with McNemar tests and measurement error, `comparison.csv`, change notice | sampled decoding (greedy only), speed, long-context checks |
| Red Hat quantized models | task recovery against the base on many benchmarks, eval commands ([task-level checks](task_level_quality_checks.md)) | KLD, serving path |
| Recursive's Flint card | settings, BF16 against NVFP4 on six benchmarks, "evaluate the specific checkpoint and runtime" | quantized GGUF scores, tool-calling benchmarks |

What v2 must not omit, because each one changed a conclusion in this project: intervals and the noise floor (Q30,
Q41); paired differences against the bar file (Q41); the reference (BF16, not Unsloth's Q8_0, whose tensors UD
shares) and the KLD context length (512 against 8k, Q106); the head treatment (Q125); chat against tool requests
(Q55); streaming (Q118); card count and context; engine commit and flags for every speed; failure rates (malformed,
runaway); reasoning length and time per correct answer ([answer time](answer_time.md)); who the file is for.

### 5.2 The one-page results format (`docs/benchmarks/<date>-flint-v2.md`)

1. **One paragraph of setup:** engines with commit and tarball sha256 (stock `d77256999` built with the same script,
   v1 `eeaa59660`, v2 release commit), files with sha256 and source revision, nine cards, driver and CUDA versions,
   context, server flags and environment per arm, sampling, streamed requests, dates.
2. **Speed on the rig** (each cell from the same build, cards and flags; tok/s with the run-to-run spread):

   | File (head treatment) | Engine | Chat decode, streamed | Tool-request decode, streamed | Prompt tok/s at 2k / 8k / 32k new tokens | First token of an agent turn | llama-bench tg128 (microbenchmark) |
   |---|---|---|---|---|---|---|

   Rows: UD (Q8_0 head) on stock, v1 and v2; UD + screen on v2; the release file on stock and v2 (+ screen if it has a
   Q8_0 head). Chat compares screened with screened; tool requests run without the screen for every file by rule.
3. **Quality against BF16:**

   | File | KLD at 512 (80 × 512 wikitext) | Same top token | PPL change | 8k agent text, written tokens, ratio to UD (95 % CI) | 8k, read tokens | Decision tokens |
   |---|---|---|---|---|---|---|

   8k ratios with a cluster bootstrap by schema set (≥ 24 chunks, ≥ 15 clusters), as in the release rule of the
   [long-context mix](long_context_mix.md) §5.2.
4. **Task checks** (Q41's protocol): per lane E1, E2, E3 at 65,536 tokens, pass@1 with 95 % interval, tasks × attempts,
   the paired difference from UD with its 90 % interval and the margin verdict ("within the 1.5-point margin"), malformed
   and runaway rates, the length ratio. Lanes not run for the shipped file are listed as not run.
5. **Time per correct answer** against UD per lane with its interval, from measured per-token times and lengths
   ([answer time](answer_time.md) §5.1).
6. **Noise and limits:** run-to-run disagreement, what the sample sizes cannot resolve, one conversation at a time,
   quality scored on the quant machine (Pascal kernels change summation order only).
7. **Reproduce:** the commands, the per-task CSVs (task IDs and verdicts, no problem text), the recipe.

### 5.3 Rules

- A "faster" claim compares the same head treatment, the same build, cards and flags, and names the request type.
- A "better" claim needs the interval to exclude zero; otherwise "within m of UD-Q4_K_XL".
- llama-bench numbers are labelled microbenchmarks and never presented as chat speed.
- Every speed against stock names the stock commit; every quality number names BF16 as the reference.

## 6. Plan

### 6.1 Order of work

1. **Today, while the rig decides (no dependency on the outcome):** a repository licence chosen; the tarball packaging
   change (`$ORIGIN` RPATH, runtime asset, notices) and a v1 rebuild from `eeaa59660` on the quant machine; the check
   script; the card and results-page skeletons; the header-fix script, tested on one file.
2. **After the owner's decision (item 1):** the v2 release build, its rig test and the command-line validation
   (items 3, 8, ~2 h of rig time); header fix and hashes; the card; the results page.
3. **Publishing:** Hugging Face repository and upload; fork releases and tags; this repository's README, tags and the
   `v2` → `main` merge.

Total after the decision: est. 3–4 working days plus upload time; the multi-turn decision run, if wanted, adds 1–2
days of the quant machine and can follow as a card update.

### 6.2 What the owner must decide or do

- **The release content:** which file (or UD + screen), which engine pieces, whether the screen and the on-card mask
  are in.
- **Licences:** one for this repository; whether to bundle NVIDIA's runtime libraries under the CUDA EULA; whether to
  keep Unsloth's importance matrix (with credit) or rebuild with our own.
- **Names:** Hugging Face account or organisation, repository and file names, the name and author written into the
  GGUF, the fork's tag names; the "Miner" name from v1 or another.
- **Accounts and settings:** create the Hugging Face repository and log in on the uploading machine (tokens are
  entered by the owner); fork settings (default branch, issues, immutable releases); tags and the merge in this
  repository.
- **Machine time:** ~2 h of rig time for the release build and command lines; the quant machine for builds and
  upload, and for the decision run if wanted.

## Already covered

| Topic | Where |
|---|---|
| Release checklist and card template (v1 era) | [quants/releasing.md](../../../quants/releasing.md) |
| Building sm_61 binaries off the rig, linking options, Docker on the rig | [ops and benchmarking](ops_and_benchmarking.md) §2, [setup.md](../../setup.md), the build script |
| The makers' engine, sampling, history and client settings | [rec](rec_engine.md) |
| Task protocol, margins, the quality section of the cards (Q42) | [task-level checks](task_level_quality_checks.md) §7, Q41 |
| Time per correct answer and the length gate | [answer time](answer_time.md) §5 |
| The release rule against UD at equal speed, 8k written/read KLD | [long-context mix](long_context_mix.md) §5.2, Q106, Q126 |
| The agent check's one-page format, pi and OpenCode details | [agent workload check](agent_workload_check.md) §2, §5.2 |
| Why tool requests sample on the host | [tool-request sampling](tool_request_sampling.md), Q55, Q67 |
| The screen, its tools and its correctness | [gptq_head](../../../quants/gptq_head/README.md), Q91, Q92 |
| ik_llama.cpp's AVX2 requirement and lack of releases; Pascal coverage of wrappers | [llama.cpp family](engines_llamacpp_family.md) §5–6 |
| The P106-100 variant of the same rig | [community reports](quant_speed_d_community.md) X1–X3 |
| Upstreaming the Pascal changes | [upstream status](upstream_pascal_patches.md) |

## Not verified

- **What stock llama.cpp prints** for a screen-inside file: derived from the loader code, not run.
- **Which pieces and which file ship:** open until Q117's ablations, Q125 and the owner's decision.
- **The release files' headers:** the K2-class files are on the quant machine; whether they inherit Unsloth's header
  depends on their composition chain. Their sizes are not recorded here (est. 31–34 GB from S4-L0's 32.5 GB).
- **The screen on a GPTQ Q5_0 head:** exactness checked only on Q8_0 heads.
- **The CUDA libraries' unpacked size** and the 12.9 toolkit's own EULA copy: the current EULA (v13.4) was read; its
  Attachment A has listed these libraries across releases (est.).
- **The tarball on Ubuntu 25.10/26.04 and Debian 13:** from symbol versions, not run. Debian 13's NVIDIA 580 package
  was not checked.
- **`-ub 512` at 262,144 tokens on the rig** with the release engine, and the screen's VRAM under the default split.
- **Six-card context limits:** arithmetic from file sizes and KV per token.
- **P106-100 rigs:** same compute capability; behaviour and speed untested.
- **Upstream's Linux CUDA 12.8 release on the rig:** PTX JIT for compute 6.1 and CPU-variant selection on the Celeron.
- **Whether `hf download` verifies checksums itself:** the CLI docs describe only `hf cache verify`.
- **Download times, `sha256sum` speed on the Celeron and load times:** arithmetic.
- **Unsloth's source weights:** its headers name "Flint Preview"; whether those weights equal the public BF16 is not
  known ([long-context mix](long_context_mix.md), not verified).
- **Licence reading:** this note reads the licence texts; it is not legal advice.

## Sources

Licences and model cards (P), read 2026-10-24:
- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) at `2a247be`: `README.md` (metadata, "Recommended
  generation settings", "Limitations and responsible use", "License and reporting"), `LICENSE` (compared with
  [apache.org's text](https://www.apache.org/licenses/LICENSE-2.0.txt)), `generation_config.json`, `sglang/NOTICE`,
  file list via the Hub API.
- [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) at `9c47151`: `README.md`
  (llama-server command, pi and OpenCode instructions), `pi-models.json`, `opencode.json`; GGUF header of
  `flint-int4-q4_1.gguf`.
- [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) at `20468ed`: `README.md`, `LICENSE`, the Hub's tree
  API (sizes, LFS sha256), headers of `flint-UD-Q4_K_XL.gguf`, `flint-Q8_0.gguf` and `imatrix_unsloth.gguf_file` (range
  requests).
- [bartowski/RecursiveSI_flint-GGUF](https://huggingface.co/bartowski/RecursiveSI_flint-GGUF) at `a04b26f`,
  [terelabs/flint-GGUF](https://huggingface.co/terelabs/flint-GGUF) at `fb43f2c`,
  [ubergarm/Qwen3.8-27B-GGUF](https://huggingface.co/ubergarm/Qwen3.8-27B-GGUF),
  [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) and
  `unsloth/Qwen3.6-35B-A3B-GGUF` file lists: cards and file naming.
- NVIDIA, [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) (v13.4): §1.1.1, §1.1.2, Attachment A;
  [CUDA 12.9.1 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)
  (cudart 12.9.79, cuBLAS 12.9.1.4 archive sizes).

Code (P):
- llama.cpp upstream `d77256999` and master `0ae0e1976`: `src/llama-model-loader.cpp` (`:719`, `:1188`,
  `:1400-1410`), `src/llama-model.cpp` (`:1873`; master `:1880`), commit history for #22673 (MTP, `partial`), #30928
  (Flint, `a37cc4610`, in b11803) and #30929 (hex escapes, `9febe63d5`, in b11798); `.github/workflows/release.yml`
  (`:328-452`), `docs/docker.md`, `.devops/cuda.Dockerfile`, `cmake/license.cmake`, `app/CMakeLists.txt`, `app/llama.cpp`, `vendor/`,
  `licenses/`, `LICENSE`.
- The fork, `pascal-p106-v2-combined` at `b3052a4f8`: `src/llama-model-loader.cpp:422-436, 678-703`,
  `src/llama-model.cpp:1862-1863`, `src/llama-graph.cpp:1829-1841, 4390-4426`, `src/llama-context.cpp:380-392`,
  `src/llama-hparams.h:237`, `common/common.cpp:1124-1146, 1422-1480`, `common/common.h:241-243, 647-650, 661, 672`,
  `tools/server/server-context.cpp:1295-1299, 4567-4577`, `common/arg.cpp` (flag names); `pascal-p106-v2` at
  `032b130f4` (no screen tensor; rescale switch at `src/llama-graph.cpp:1670-1682`).
- [Cinderella-Man/llama.cpp](https://github.com/Cinderella-Man/llama.cpp) via the GitHub API: branch heads, licence,
  no releases, issues off, default branch `master`.
- The v2 build's `BUILDINFO.txt` and `objdump -T`/`-p` of its `llama-server`: symbol versions and needed libraries.

Distributions, drivers and hosting (P):
- Ubuntu archive via the Launchpad API: `nvidia-graphics-drivers-580` and `glibc`/`gcc-14` per release (jammy, noble,
  questing, resolute); Debian `glibc` versions via sources.debian.org.
- NVIDIA, [supported drivers and CUDA toolkit versions](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html)
  (R580 long-term support branch, end of life June 2028);
  [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) README (Turing or later).
- Hugging Face: [storage limits](https://huggingface.co/docs/hub/storage-limits),
  [model cards](https://huggingface.co/docs/hub/model-cards) (`base_model`, `base_model_relation`, licence fields),
  [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf auth login`, `hf repos create`,
  `hf upload`, `hf download --revision`, `hf cache verify`).
- GitHub: [about releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
  (asset limits), [immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases),
  [asset digests changelog](https://github.blog/changelog/2025-06-03-releases-now-expose-digests-for-release-assets/)
  (2025-06-03).

Secondary (S):
- Phoronix, [NVIDIA confirms 580 is the last Linux driver for Maxwell, Pascal and Volta](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW).

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): Q12, Q41, Q42, Q55, Q56, Q67, Q73, Q77, Q91, Q92, Q106, Q113,
  Q117, Q118, Q125, Q126, Q131, Q132; the decision log.
- [README](../../../README.md), [setup.md](../../setup.md), [mining_rig.md](../../mining_rig.md),
  [quants/README.md](../../../quants/README.md), [releasing.md](../../../quants/releasing.md),
  [recipes](../../../quants/recipes/README.md), [gptq_head](../../../quants/gptq_head/README.md),
  [build script](../../../scripts/build_llamacpp_docker.sh), [compose_gguf.py](../../../scripts/compose_gguf.py),
  [patches](../../../patches/README.md), [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md).
- Notes: [rec](rec_engine.md), [task-level checks](task_level_quality_checks.md), [answer time](answer_time.md),
  [long-context mix](long_context_mix.md), [agent workload check](agent_workload_check.md),
  [tool-request sampling](tool_request_sampling.md), [llama.cpp family](engines_llamacpp_family.md),
  [community reports](quant_speed_d_community.md), [ops and benchmarking](ops_and_benchmarking.md).
