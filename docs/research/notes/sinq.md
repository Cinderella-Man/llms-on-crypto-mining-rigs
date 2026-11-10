# SINQ, the BeeLlama.cpp claim, and Flash-Next's 2–3-bit files (2026-11-10)

SINQ (Sinkhorn-Normalized Quantization) is a weight-quantization method from Huawei that gives each weight matrix a
second scale, one per input column, chosen from the weights alone. On 2026-11-10 a post on X announced it in
BeeLlama.cpp v0.5.0 Preview, a llama.cpp fork, as the first SINQ in the llama.cpp ecosystem. The post says that adding
SINQ to any GGUF recipe lowers its error against BF16. As evidence it reproduces Bartowski's Qwen3.5-9B collection
with SINQ added: every recipe gets a lower KLD, the gains are largest at 6 and 8 bits, and "Bartowski + SINQ" beats
Unsloth's files on most of the size–KLD frontier. It also lists its limits:
- the SINQ settings were picked by hand for each recipe with A/B tests;
- the files run only on BeeLlama.cpp v0.5.0;
- some recipes may gain nothing;
- larger models (Qwen3.8-27B) come later.

This note covers four questions:
1. What SINQ is.
2. Whether the claim can be checked.
3. Whether the fork is safe.
4. What SINQ could do for Miner Llama's 2–3-bit Flash-Next files on nine P106 cards.

The paper (arXiv v4) and these pages were read as web pages on 2026-11-10:
- the official code, its Pre-SINQ GGUF script and model cards;
- the KVarN paper;
- the fork's repository at branch `v0.5.0` (`d8b5685b0`) and release commit `d51f2f5e0`, its releases, its issues
  and its Hugging Face account.

Nothing was cloned, built, run or downloaded from the fork or its model pages. The X post itself could not be opened
without signing in, so its figures are known only from the text quoted above. Numbers marked **est.** are this note's
arithmetic.

## Summary

1. **SINQ is real and published.**
   - Authors: Lorenz K. Müller, Philippe Bich, Jiawei Zhuang, Ahmet Çelik, Luca Benfenati and Lukas Cavigelli, all of
     Huawei. The code's copyright holder is Huawei Technologies Switzerland AG.
   - Paper: [arXiv 2509.22944](https://arxiv.org/abs/2509.22944), v1 on 2025-09-26, v4 on 2026-01-29, accepted at
     ICML 2026. Code: Apache-2.0, [huawei-csl/SINQ](https://github.com/huawei-csl/SINQ).
   - The method in one line: before rounding, a few damped Sinkhorn–Knopp steps balance the standard deviations of
     the matrix's rows and columns. The column factors become a stored per-input-column scale, and the row factors
     merge into the usual per-group scales. No calibration data is needed.
   - KVarN ([arXiv 2606.03458](https://arxiv.org/abs/2606.03458), 2026-06-02), the KV-cache quantizer BeeLlama
     already ships, comes from the same Huawei group: Müller, Bich, Zhuang and Cavigelli are on both papers.
   - BeeLlama's author (GitHub and Hugging Face "Anbeeld") is on neither paper. The fork ports their
     methods; "from the authors of KVarN" describes SINQ, not the fork.
2. **BeeLlama.cpp shows no sign of malware in what was read.**
   - It is a fork of `ggml-org/llama.cpp` under MIT, one developer's project since 2026-05-05. Its `v0.5.0` branch
     sits 1,122 commits ahead of and 273 behind upstream `master`, on a merge base from 2026-10-30.
   - Its SINQ code is a C++ port that carries Huawei's Apache-2.0 notice. Neither it nor the Python optimizer opens a
     network connection or runs a process.
   - Release binaries are built from the tagged tree by GitHub Actions. They are not signed, and preview assets are
     replaced in place.
   - GitHub's compare lists only the first 300 changed files. The rest, among them most of `src/` and `tools/`, were
     not scanned, so the verdict is "looks legitimate, not fully audited" (section 3).
3. **The claim cannot be checked today, and part of it is not accurate.**
   - As read, the Hugging Face repository the post points to, `Anbeeld/Qwen3.5-9B-MTP-GGUF-Bartowski-SINQ`, held only
     a `.gitattributes`: no GGUF, no model card, no recipes, no KLD method. It was created 27 seconds after the
     release.
   - "First time in the llama.cpp ecosystem" holds only for SINQ's runtime form, where the file carries a scale vector
     and the engine applies it. The SINQ authors published GGUF results in the paper (Table 9) and four Pre-SINQ GGUFs
     that run on stock llama.cpp (Qwen3 0.6B to 8B, February 2026).
   - The gains are plausible at 8 bits, because llama.cpp's Q8_0 quantizer ignores the importance matrix, so a column
     scale is the only activation-aware part there.
   - "Every recipe improves" is what hand selection is expected to give. The candidates include a strength of 0, which
     is the plain file, and selection was by hand A/B.
   - The one independent test with an importance matrix found SINQ neutral at 3 bits (section 2.3).
4. **For Miner Llama the evidence at 2–3 bits is thin, but a free form exists and a cheap test fits the tools.**
   - The paper reports only 3 and 4 bits, and its 3-bit gains are measured against round-to-nearest without an
     importance matrix.
   - With an importance matrix, a Qwen3.8-27B 3-bit recipe gained nothing, on code and agent traces too
     ([beellama.cpp #199](https://github.com/Anbeeld/beellama.cpp/issues/199)). This project's rotations and AWQ-style
     scaling also gave nothing once the importance matrix weighted the quantizer (Q13, Q304,
     [GPTQ for the body](gptq_body.md) §1).
   - Folded form: the column scale moves into the preceding norm weight or the preceding matrix's rows. It is exact,
     costs no bytes or launches, and the file runs on Miner Llama's and stock llama.cpp's kernels unchanged.
   - Runtime form (BeeLlama's): est. +0.3 to +0.9 ms a pass in launches and 157–283 MB of scales on the cards for
     Flash-Next's 512 experts per layer. It also needs a scaled copy of the Q8_1 activations for each distinct scale.
5. **Rows:**
   - Q1258 (planned): the folded form tested first in Q164's importance-weighted error proxy on Flash-Next's expert
     and dense matrices (CPU), then a KLD file on Q928's recipe.
   - Q1259 (parked): the runtime form in the kernels.

## 1. What SINQ is

### 1.1 The method

A low-bit format stores each matrix as integers plus a few higher-precision numbers per tile. llama.cpp's block
formats keep one scale (sometimes a minimum) per block of 32 or 256 weights along a row. SINQ adds a second axis
(paper eq. 3):

> W ≈ s ⊙ (Q + z) ⊙ t

- s is the usual per-row scale of each tile (in a GGUF block format, the block scale), z an optional shift, Q the
  integers.
- t is one value per input column, shared by all rows.

An outlier in row i and column j can then be absorbed partly by s_i and partly by t_j, instead of stretching the
whole block's step.

How t is found (Algorithm 1):
1. Start from log-scales of zero for rows and columns.
2. Repeatedly divide each row and each column by its standard deviation relative to the smallest one, with every step
   clamped (damped).
3. Keep the iterate with the lowest imbalance, the ratio of the largest to the smallest row or column standard
   deviation.
4. Round the rescaled matrix with an ordinary quantizer.

The procedure reads only the weights.

The paper's reason this helps: a column's weight spread predicts the typical size of that column's input. Under Adam
training the per-column standard deviation of the weights falls as the inverse square root of the input scale (eq. 4,
Fig. 2). Scaling by t is therefore a weight-only stand-in for activation-aware scaling such as AWQ ("pseudo
activation-aware"). Two consequences:
- It does not always lower the matrix's own reconstruction error, but it lowers the error of the layer's output.
- Hadamard rotation does the opposite (Fig. 3).

Cost of the second scale:
- **Storage:** M extra values per matrix (one per input column), stored in FP16 or int8.
- **Compute:** at run time `x·Wᵀ ≈ (x ⊙ t)·[s ⊙ (Q + z)]ᵀ` (eq. 7), one element-wise multiply of the input per
  matrix, as in AWQ. In a naive W4A16 kernel test (gemlite, a separate PyTorch multiply) this cost 1.8 % at batch 1
  and width 1,024, and 0.8–0.9 % at batch 64 (Table 5). In SGLang the end-to-end decode speedups over FP16 were
  within 0.1× of AWQ's (Table 6).

Three variants:
- **SINQ:** calibration-free, as above. Quantizing takes 1.09× round-to-nearest's time (Table 10).
- **A-SINQ:** SINQ, then AWQ's calibrated per-column search on top (a 1-norm form of AWQ's objective). Takes 8.54×
  round-to-nearest's time.
- **No-overhead SINQ / Pre-SINQ:** t is folded into what produces the input: a norm weight, or the rows of the matrix
  before (§2.3.1). Matrices that read the same input must then share one t:
  - in Qwen3, q, k and v share one, and gate and up share one;
  - the down projection's t goes into the up projection's rows.

  The model stays mathematically identical, so any quantizer can run on it afterwards (GGUF, AWQ, GPTQ, HQQ). Sharing
  costs some of the gain: Qwen3-1.7B at 4 bits reaches WikiText-2 perplexity 17.63 against 17.14 for SINQ, and 18.96
  for HQQ (Table 8).

### 1.2 Reported results

Weight-only, group 64, perplexity on WikiText-2 / C4, lower is better (Tables 1 and 4; BF16 in the first column):

| Model, bits | BF16 | Round-to-nearest | HQQ | GPTQ | SINQ | A-SINQ |
|---|---|---|---|---|---|---|
| Qwen3-1.7B, 3 | 16.67 / 19.21 | 32.43 / 31.10 | 32.10 / 30.54 | 32.21 / 31.05 | 22.39 / 24.88 | 22.30 / 24.00 |
| Qwen3-14B, 3 | 8.64 / 12.01 | 10.50 / 14.88 | 10.73 / 14.39 | 9.54 / 13.03 | 9.33 / 12.90 | 9.31 / 12.71 |
| Qwen3-32B, 3 | 7.60 / 10.77 | 30.78 / 35.83 | 9.09 / 12.58 | 9.03 / 12.38 | 8.79 / 11.83 | 8.45 / 11.54 |
| Qwen3-32B, 4 | 7.60 / 10.77 | 8.92 / 12.80 | 8.62 / 12.20 | 7.80 / 10.99 | 7.74 / 10.96 | 7.78 / 10.93 |

More results:
- **MoE models, 3 bits, WikiText-2** (Tables 13 and 19):
  - Qwen3-30B-A3B: round-to-nearest 12.28, HQQ 10.52, SINQ 10.19 (BF16 8.70).
  - Qwen3-235B-A22B: round-to-nearest 10.11, HQQ 13.07, SINQ 6.27 (BF16 5.37).
- **llama.cpp, no-overhead SINQ before GGUF quantization** (Table 9, GPU; prompt and decode speed unchanged):

  | Model | FP16 | Q4_0 | Pre-SINQ + Q4_0 | Q3_K_S | Pre-SINQ + Q3_K_S |
  |---|---|---|---|---|---|
  | Qwen3-0.6B | 21.88 | 25.25 | 24.01 | 35.59 | 30.20 |
  | Qwen3-1.7B | 17.13 | 21.30 | 17.65 | 24.02 | 19.49 |
  | Qwen3-4B | 14.31 | 14.97 | 14.77 | 19.03 | 18.06 |

- **Qwen3-8B Pre-SINQ GGUF on Hugging Face:** Q3_K_S 11.3619 against 10.6786 with Pre-SINQ (FP16 10.1019), from
  the model card of [huawei-csl/Qwen3-8B-PreSINQ-GGUF](https://huggingface.co/huawei-csl/Qwen3-8B-PreSINQ-GGUF).

What the published results leave out:
- **No result below 3 bits.** The code accepts 2 bits; neither the paper nor the README reports any.
- **No GGUF baseline with an importance matrix.** The Pre-SINQ script (`presinq_models_generation/presinq_gguf.py`)
  calls `llama-quantize` without `--imatrix`. It defaults to A-SINQ, searches 30 settings (group 32/64/128,
  2–32 Sinkhorn iterations, 1–2 passes), and keeps the one with the lowest perplexity of the quantized file on
  `pile_val_5k.txt`. No other text is scored, so the reported gain is also the selection's score.
- **The script's MoE fold is incomplete.** For DeepSeek-V2-Lite it multiplies the FFN norm by the shared gate/up scale
  and divides the routed experts' gate/up columns, but it leaves the router and the shared experts as they were.
  Those also read the norm's output, so the transformed model's routing differs from the original's. An exact fold
  must rescale every reader of the norm (as [GPTQ for the body](gptq_body.md) §3.3 lists for Flint).

## 2. The BeeLlama.cpp claim

### 2.1 What v0.5.0 Preview adds

From the release notes, `CHANGELOG.md` and `docs/sinq.md` at `d51f2f5e0`:
- **Quantizing:** `llama-quantize --sinq-config FILE` turns on "input-column preconditioning" for Qwen3.5 GGUFs from
  BF16, F16 or F32. Only the `qwen35` architecture is accepted, and only the trunk's `attn_*`, `ssm_*` and
  `ffn_gate/up/down` matrices; MoE expert tensors are not among them.
- **SINQ's procedure on BeeLlama's existing codecs:** the Sinkhorn balancing runs over strips of input columns of a
  configurable width (8–512), with a "strength" exponent in [0, 1]. The stored diagonal is `u = F16(t^strength)`, and
  a strength of 0 gives the identity. Only u survives into the file; the row factors guide the search and are dropped,
  because the codec's own block scales replace them. The codecs are llama.cpp's own (Q3_K to Q8_0, IQ1 to IQ4).
- **Importance matrix:** the quantizer encodes W/u with the importance matrix scaled by u² (by default). This is the
  exact transform of the output error onto the stored weights.
- **File format:** five `beellama.qk_sinq.*` metadata keys and one F16 tensor `<name>.sinq_scale2` of length K (the
  input width) per conditioned matrix, 16/K extra bits per weight.
- **Inference:** BeeLlama computes `native_matmul(W/u, u·x)` on the CPU and on "qualified" CUDA backends. Equal
  vectors are shared and scaled inputs reused. Row and tensor split modes are refused. "No absorbed or
  companion-free export is provided", so there is no folded form.
- **Compatibility:** "These GGUFs require BeeLlama". An engine that ignored the extra tensors would multiply W/u by
  unscaled inputs. Stock llama.cpp refuses a GGUF with tensors it does not expect ([release v2](release_v2.md)).
- **Tooling:** an optional offline optimizer (`tools/sinq/optimizer`, Python) plans, scores and selects mixed
  candidates per tensor (stock, SINQ, AWQ). Its corpora are split into calibration, selection and final-test parts,
  and its notes say "Selection scores do not establish held-out quality gains". `llama-perplexity` gains
  `--ppl-full` and per-token KLD output.

### 2.2 What is public, and what a check needs

| Needed to reproduce the post | State on 2026-11-10 |
|---|---|
| The GGUFs | not uploaded: `Anbeeld/Qwen3.5-9B-MTP-GGUF-Bartowski-SINQ` (created 2026-11-10 18:00:47 UTC) holds only `.gitattributes` |
| The per-recipe SINQ settings (`--sinq-config` files: width, iterations, strength per tensor) | not published |
| The KLD method: text, context, tokens, BF16 reference, backend | not published; the docs leave context, batch and token count to the user |
| Bartowski's importance matrix and recipes | public ([bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF), made with llama.cpp b8192) |
| The engine | public source; the SINQ path needs BeeLlama |

A check needs:
- the files and their config files;
- the text and settings of the KLD runs;
- the same numbers on a text that played no part in the A/B choices.

### 2.3 Is it plausible?

- **At 8 bits, yes.** llama.cpp's Q8_0 quantizer ignores the importance matrix (`quantize_q8_0` begins
  `(void)quant_weights; // not used`), so Bartowski's Q8_0 is plain absmax rounding, and a column scale brings it the
  activation awareness it lacks. At 6 bits Q6_K does use the importance matrix.
  - At these widths the KLD against BF16 is small, so a large percentage can be a small absolute change.
  - Such a change is meaningful only with the reference and the quantized file run on the same backend and build. On
    Flint one Q8_0 file differed by KLD 0.065 between two GPU types (Q21).
- **"Every recipe improves" follows from the selection.** With a strength of 0 (the plain file) among the candidates
  and the best one picked by A/B per recipe, a recipe can come out worse on the A/B text only through noise. Lower
  KLD on that same text is expected, and the size of the gain on other text is unknown.
- **With an importance matrix at 3 bits, the one independent test found no gain.** In
  [#199](https://github.com/Anbeeld/beellama.cpp/issues/199) (2026-11-08, a user of the fork), the setup was
  Qwen3.8-27B with Unsloth's UD-Q3_K_XL recipe and importance matrix, built with v0.5.0's SINQ at `f0beb9f60`. Mean
  KLD against BF16 with a forced BOS:

  | Text | imatrix only | + SINQ | + A-SINQ, calibrated on BF16-written text | Unsloth's file |
  |---|---|---|---|---|
  | WikiText-2 | 0.0399 | 0.0402 | 0.0377 | 0.0305 |
  | Chat | 0.0615 | 0.0590 | 0.0532 | 0.0513 |
  | Django source | 0.0600 | 0.0601 | 0.0605 | 0.0630 |
  | Code | 0.01026 | 0.01054 | 0.01022 | 0.00976 |
  | Agent traces | 0.0208 | 0.0209 | 0.0209 | 0.0209 |
  | Same top token, agent traces | 93.69 % | 92.48 % | 92.44 % | 93.59 % |

  SINQ moved KLD by 3 % or less on five texts and lost 1.2 points of top-token agreement on agent traces. A-SINQ
  calibrated on the user's own coding sessions was level or slightly worse. That run used an older calibration path
  than the post's per-recipe settings.

## 3. Safety assessment of BeeLlama.cpp

### 3.1 Who and what

- **Owner:** GitHub user Anbeeld, account since 2021-12-10, 14 public repositories, 54 followers. The profile describes
  a full-stack engineer, links anbeeld.com, and links the X account @Anbeeld.
- **Hugging Face:** Anbeeld since 2026-04-27, 49 model repositories, almost all DFlash, DFlash2 and DSpark drafter
  GGUFs for other makers' models (up to 6,195 downloads each).
- **Repository:** [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp), created 2026-05-05 as a GitHub
  fork of `ggml-org/llama.cpp`, MIT, with 1,189 stars, 82 forks and 18 watchers.
  - It started as DFlash speculative decoding plus TurboQuant KV-cache types, by way of `llama-cpp-turboquant` and
    `buun-llama-cpp`. From v0.4.0 it moved to KVarN and KV-cache precision tails.
  - Seven releases, v0.4.3 to v0.4.9, came between 2026-08-12 and 2026-10-28, and most of them update the llama.cpp
    base.
- **Funding:** a "Support my work!" link to anbeeld.com/support, which lists Buy Me a Coffee and Bitcoin, Ethereum
  and USDT addresses for donations. There is no token, sale or investment offer on the repository or the site.

### 3.2 What was read and what it shows

| Part | What it is | Finding |
|---|---|---|
| `src/llama-sinq.cpp` | the Sinkhorn balancing per strip, the strength exponent, FP16 rounding | standard library and `std::thread` only; header "Copyright 2025 Huawei Technologies Switzerland AG", Apache-2.0; no network, no process, no file writes |
| `src/llama-sinq-activation.cpp` | loader for activation samples (A-SINQ) | reads a manifest and sample files and checks SHA-256 of each (Windows CNG for the hash); refuses absolute paths and `..`; no network, no environment reads |
| `tools/sinq/export_activations.py`, `optimizer/corpora.py`, `optimizer/reference_loader.py` | activation export and the optimizer's corpus and reference loaders | Hugging Face loads with `local_files_only=True` and `trust_remote_code=False`; safetensors; no `requests`, `urllib`, `subprocess`, `pickle` or `eval` |
| `licenses/LICENSE-SINQ`, `NOTICE-SINQ`, `docs/sinq.md` | attribution | credits huawei-csl/SINQ at a fixed commit; says BeeLlama "is not the original Python SINQ serving implementation" |
| GitHub compare `master...v0.5.0`, the first 300 files' patches (`.devops`, `.github`, top-level docs, `cmake`, `common`, `conversion`, `docs`, `examples`, `ggml` headers, `ggml-cuda` up to the KVarN template instances) | the fork's changes | no added socket, HTTP-client or process calls, no encoded blobs; added URLs are documentation links (anbeeld.com articles, the fork's GitHub and GHCR pages, Wikipedia); `--hf-token` passes to router children only through `HF_TOKEN` and is marked sensitive; one binary file, `beellama.jpg` |
| `.gitattributes` | line-ending rules | marks five experiment paths `-text` so their digests survive checkout; nothing hidden from diffs |
| `AGENTS.md`, `CLAUDE.md` | notes for coding assistants | build and test conventions only |

### 3.3 Releases and CI

- `.github/workflows` holds only `release.yml`, `release-dispatch.yml` and `release-preview-dispatch.yml`; upstream
  has 63 workflows, so the fork runs no tests in CI.
- `release.yml` runs only on manual dispatch from `main`. It builds every package from the checked-out source on
  GitHub's runners and uploads with the built-in `GITHUB_TOKEN` (no custom secrets) to GitHub Releases and
  `ghcr.io/anbeeld/beellama.cpp`.
  - Vendor SDKs and installers come from Intel, AMD, LunarG and NVIDIA hosts, some run as root without a hash check.
  - Docker base images are not pinned by digest.
- `SHA256SUMS.txt` is written in the same job as the assets, so it guards the transfer only. There are no signatures,
  no build provenance (`provenance: false`) and no SBOM.
- "Preview" releases are rolling: the tag is force-moved and assets are replaced by newer builds. A checksum recorded
  on one day does not identify the binary a later download gets.
- Downloads are modest: v0.4.9's 17 assets have 2,304 downloads in all, and v0.5.0 Preview's 17 assets 27 on the day
  of release.

### 3.4 Attention and outside reports

- **Stars:** gradual over the window Trendshift tracks (+126 in the 61 days from 2026-09-11 to 2026-11-10, at most +9
  a day), with one day in its top 25 (2026-11-04). Most stars predate that window, from the DFlash posts of May–August.
- **Reception:** the May post on r/LocalLLaMA drew mixed reports:
  - speedups of 33 % to 2× on 3090 and 5090 cards;
  - crashes on one KV-cache type and tool-call breakage;
  - criticism of AI-generated code and of a three-deep fork chain.

  Third-party write-ups (DataCamp, blogs) report the DFlash speedups. No report of malware, telemetry or a
  compromised binary was found.
- **Issues:** a search for malware, virus, trojan, telemetry or suspicious finds only two speed regressions.

### 3.5 Verdict

**Looks legitimate; not fully audited.** The evidence for:
- an identifiable developer with a four-year-old account and a public track record of releases;
- a fork whose merge base with upstream is 11 days old;
- the SINQ code is a faithful, attributed port of an Apache-2.0 method, with no network or process calls in the parts
  read;
- binaries built by CI from the public tree.

The evidence against is hygiene, not malice:
- unsigned, replaced-in-place preview assets;
- no CI tests;
- one maintainer and much AI-written code.

What would settle the rest:
- a line-by-line scan of the diff the compare API does not list (`src/`, `tools/` with the server, `ggml-rpc` and the
  other backends, `scripts/`) for network, process and file-system calls, read against the merge base `3c809f443`;
- a reproducible build of one release asset compared with the published one;
- for the claim itself, the model card with files, settings, method and a held-out text.

Building from source at a fixed commit avoids the binary question entirely.

## 4. What it would mean for Miner Llama

### 4.1 Where the column scale can go, and what each costs on a P106

| Form | What changes | Run-time cost on a P106 | Bytes | Runs on |
|---|---|---|---|---|
| Folded (no-overhead, Pre-SINQ) | t multiplies the preceding norm weight (every reader of that norm shares one t, and the readers kept at high precision, such as the router, take the inverse exactly), or the preceding matrix's rows (down's t into up's rows, the attention output's t into v's rows) | none | none | Miner Llama and stock llama.cpp, unchanged |
| Runtime (BeeLlama's `sinq_scale2`) | an F16 vector per matrix, x·u before the product | x·u must be applied before the activations are quantized to Q8_1 for `dp4a`: one scaled Q8_1 copy per distinct scale, where the norm would otherwise write one shared copy (Q100). Est. +3 to +8 graph nodes a layer, so est. +0.3 to +0.9 ms a pass at ~2.3 µs a node ([floor note](flash_next_floor_and_engine.md)), est. 0.2–0.7 % of the 131.57 ms served step at shallow depth (Q961) | gate/up +0.006 and down +0.025 bits a weight; est. ~4.3 MB a token of scales (0.1 % of 4,155 MB) | a new path in the expert and dense kernels |

Applying the scale inside the `dp4a` loop instead would add float work per weight. Below 4 bits the P106 is limited by
instructions per weight ([2–3-bit format](pascal_lowbit_format.md) §2), so that form is the most expensive.

### 4.2 Flash-Next's tensors

Flash-Next (`qwen4exp`) has:
- 48 layers: 36 Gated DeltaNet, 12 attention;
- width 2,560, with four hyper-connection streams mixed before each half's norm;
- 512 routed experts of width 640, ten used per token, plus a shared expert.

Swift Q2_0 reads 4,155 MB a token:
- the dense tensors are 84 % of it;
- the routed experts (Q2_0) are 16 %;
- the attention and recurrent projections (Q3_K, IQ4_XS, Q4_K and others) are 33.5 %.

Where a folded scale fits:

| Input | Fold into | Readers that share t | Exact if |
|---|---|---|---|
| Norm before a recurrent half | its norm weight | `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` | all readers compensated |
| Norm before an attention half | its norm weight | `attn_q` (with its output gate), `attn_k`, `attn_v`, and any indexer projection that reads the norm | as above |
| Norm before the FFN half | its norm weight | all 512 experts' gate and up, the shared expert's gate and up and its sigmoid gate, **the router** (F32/BF16, compensated exactly) | the router and shared expert are rescaled, which the official script's MoE branch omits (§1.2) |
| SwiGLU output, per expert | that expert's `up` rows | its own down (640 columns) | always: silu(gate)·up is linear in up, and a row scale leaves up's block quantization unchanged |
| Attention result | `attn_v` rows | the attention output | t tied across the 12 query heads of each KV head |
| Gated DeltaNet output | the gated output norm's weight | `ssm_out` | t in the shape of that weight |

The per-expert down fold is the strongest case. It gives every one of the 512 experts its own scale at zero cost, and
it lands on the 2-bit expert tensors that hold most of the quantization error. Gate/up get only one scale per layer
across 512 experts. Per-expert gate/up scales exist only in the runtime form.

The hyper-connection projections (31 % of the bytes) are BF16 or Q8_0. At Q8_0 they already cost no measurable KL
([Flash-Next news](flash_next_news_2026-11.md) §10), so there is nothing to gain there.

### 4.3 Stacking with the importance matrix and learned rounding

- **The importance matrix and SINQ are different degrees of freedom.** The importance matrix weights each column's
  error in the quantizer's search, but all columns in a block share the block's step. SINQ's scale changes each
  column's effective step (u_j × the block scale). They stack in one of two ways:
  - quantize W/u against the importance matrix times u², as BeeLlama does;
  - recompute the importance matrix on the folded model, which gives the same thing.
- **The evidence says the importance matrix already takes most of what column scaling offers** (§2.3,
  [GPTQ for the body](gptq_body.md) §1 and §3.3): AWQ lost to imatrix Q4_1 in llama.cpp, rotations hurt here (Q13,
  Q304), and #199 found SINQ neutral at 3 bits.
- **Learned rounding** (GSQ-RCO's Gumbel-Softmax rounding, GPTQ) takes any reparameterized model as input. SINQ would
  go first, as a fold, and the rounding would then run on the folded weights.
  - The published GSQ-RCO files cannot take it without rerunning GSQ (Q236, Q237).
  - What can take it today are the files this project quantizes itself from the BF16 with FN4's importance matrix: the
    N2P, N2P-128 and N3P-128 experts (Q928, Q220) and the dense tensors of a Pascal mix (Q880).
  - llama.cpp's own Q2_0 quantizer is not one of them: it is ternary absmax and unusable after training
    ([2–3-bit format](pascal_lowbit_format.md) §4).

### 4.4 How to try it without BeeLlama (Q1258), and what waits (Q1259)

**Q1258** runs on the quality machine, CPU first, from the official code and this repository's tools:
1. **Proxy.** Port SINQ's Sinkhorn normalization (`sinq/sinkhorn.py` in huawei-csl/SINQ, Apache-2.0, credited) into
   `scripts/lowbit_proxy.py` as a pre-transform, and score with Q164's importance-weighted error (run with the
   original importance, on the effective weights u ⊙ Q).
   - Sample: Swift-1.5's BF16 safetensors with FN4's importance matrix, Q164's scheme (seven layers, four experts
     each, gate, up and down), plus the dense projections of the same layers.
   - Arms:
     - (a) a per-expert scale on down;
     - (b) one scale per layer over all experts' gate/up;
     - (c) a per-expert scale on gate/up, the runtime form, for comparison;
     - each at strength 1 and 0.5;
     - each in N2P, N2P-128, N3P-128, IQ2_XS, IQ3_XXS and IQ3_S, quantized against importance × u².
2. **KLD**, only for an arm whose error falls beyond the spread across layers at equal bits: Q928's file 2 recipe with
   that fold applied to the BF16 before quantizing.
   - The fold rewrites norm weights, up rows and router columns in place and adds no tensor.
   - The importance matrix is scaled by u².
   - KLD against Swift's Q8_0 on the three texts, beside file 2's agent KLD of 0.172660.
   - The file is the same size and runs on the same kernels, so its speed is unchanged by construction. A rig run
     follows only if the KLD gains.

**Q1259** stays parked:
- it needs kernel work for est. +0.3 to +0.9 ms a pass and 157–283 MB on the cards;
- no published result shows SINQ helping below 3 bits or on top of an importance matrix.

It reopens if Q1258's proxy puts arm (c) clearly ahead of arms (a) and (b).

## 5. Not verified

- The X post itself (x.com and its mirrors refused the page): its figures, its frontier plot and the account's
  history.
- The SINQ GGUFs, their settings and the KLD method: not published when read.
- The fork's changes beyond the first 300 files GitHub's compare lists: most of `src/` (apart from the SINQ files), `tools/`
  (including the server), `ggml-rpc` and the other backends, `scripts/`. Also the release binaries and Docker images,
  which were not downloaded.
- Whether #199's author is independent of the maintainer beyond being a separate account.
- Flash-Next's exact norm shapes for the folds in §4.2: the gated output norm's weight shape and whether the sparse
  indexer reads the attention norm's output. Both are read from the GGUF before Q1258's KLD step.
- SINQ at 2 bits: no result exists to check.

## Sources

- L. K. Müller, P. Bich, J. Zhuang, A. Çelik, L. Benfenati, L. Cavigelli, "SINQ: Sinkhorn-Normalized Quantization
  for Calibration-Free Low-Precision LLM Weights", [arXiv 2509.22944](https://arxiv.org/abs/2509.22944) (v4,
  2026-01-29), Tables 1–10, 13, 19 and Algorithm 1; [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63404).
- [huawei-csl/SINQ](https://github.com/huawei-csl/SINQ) (Apache-2.0): README, `presinq_models_generation/` (README and
  `presinq_gguf.py`), [discussion #21](https://github.com/huawei-csl/SINQ/discussions/21);
  [PreSINQ GGUF collection](https://huggingface.co/collections/huawei-csl/presinq-gguf).
- L. K. Müller, P. Bich, C. Boretti, H.-M. Chang, J. Zhuang, L. Cavigelli, "KVarN: Variance-Normalized KV-Cache
  Quantization Mitigates Error Accumulation in Reasoning Tasks", [arXiv 2606.03458](https://arxiv.org/abs/2606.03458);
  [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN).
- [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp): branch `v0.5.0` (`d8b5685b0`), release
  [v0.5.0 Preview](https://github.com/Anbeeld/beellama.cpp/releases) (`d51f2f5e0`), `docs/sinq.md`, `CHANGELOG.md`,
  `licenses/NOTICE-SINQ`, `.github/workflows/release.yml`, the files in §3.2;
  [issue #199](https://github.com/Anbeeld/beellama.cpp/issues/199); the GitHub REST API's repository, user, release
  and compare records.
- [Anbeeld on Hugging Face](https://huggingface.co/Anbeeld);
  [anbeeld.com/projects/beellama-cpp](https://anbeeld.com/projects/beellama-cpp);
  [Trendshift](https://trendshift.io/repositories/29106); the r/LocalLLaMA launch thread (May 2026).
- llama.cpp `ggml/src/ggml-quants.c` (`quantize_q8_0`, `quantize_q6_K`).
- This repository: [GPTQ for the body](gptq_body.md), [2–3-bit format for Pascal](pascal_lowbit_format.md),
  [Flash-Next floor](flash_next_floor_and_engine.md), [Flash-Next variants](flash_next_variants.md),
  [Q164's proxy data](../../data/2026-10-27_lowbit/README.md), the registry rows named above.
