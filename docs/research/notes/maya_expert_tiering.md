# Project Maya's VRAM–RAM–NVMe expert tiering and its quants: what they would mean for the P106 rig (2026-10-23)

Scope: [Project Maya](https://github.com/mw00/project-maya) (read at commit `8412d5e`, v1.1.25, 2026-10-23) runs
GLM-5.3-Flash (321B MoE, ~18B active) and DeepSeek-V4.1-Flash-1017 (748B MoE) on one to sixteen GPUs. It keeps
the experts a conversation uses most in VRAM, the next ones in RAM and the rest on an NVMe SSD. This note covers
what that tiering and Maya's quants are, and whether either could make the nine-P106 rig faster. Maya's
Pascal status and decode kernels are covered in [Q48](../../../quants/experiments.md) and in the engine
notes ([engines_other.md](engines_other.md), Q1/Q4). Conventions:
- "(my calc)" marks this note's own arithmetic.
- "Claim to test" marks anything not measured on this rig.
- "P" and "S" in source lists mean primary (code, docs, model cards) and secondary.

Starting point on the rig, measured: Flint's S4-L0 mix (the quality-bar file) decodes at 37.4 tok/s on six cards
(26.8 ms per token) and reads 2.24 GB of weights per token. All of its experts sit in VRAM: the file is 32.5 GB,
and the cards hold 54 GB.

## Headline findings

1. **The "NVMe quants" are ordinary GGUF files. The NVMe part is Maya's engine, which keeps a three-tier expert
   cache.** Maya-S/S24/M/L use standard ggml types: IQ2/IQ3 routed experts, Q4_K–Q6_K attention and shared
   experts. Each GGUF tensor has one type. What makes 95–156 GB models run on a 24–32 GB card is the engine:
   - a VRAM cache ranked by an expert-usage profile and refreshed by decayed usage counts;
   - a pinned RAM tier;
   - the rest mapped from the SSD, with each layer's routed experts prefetched as soon as the router picks them.
2. **Tiering solves a capacity problem, and the rig has a speed problem.** Maya's machines have too little memory
   for the model but fast links: PCIe 4.0/5.0 x16, NVMe at several GB/s, and DDR5 for a CPU lane that computes
   the misses. On the rig the models we run already fit: Flint at 25–43 GB and Qwen3.6-35B-A3B at ~20 GB, against
   54 GB of VRAM. Decode time follows the bytes each token reads. No model in the [plan](../../../quants/experiments.md)
   needs a lower tier, so tiering has nothing to speed up here.
3. **The rig cannot feed a lower tier fast enough even if it needed one.**
   - Every way into a GPU is a PCIe 1.0 x1 link: ~190 MB/s per card and ~715 MB/s for all nine together.
   - The disk is an M.2 **SATA** SSD at ~485 MB/s, and the chipset has only an AHCI controller, with no NVMe.
   - The host has 3.8 GB of RAM and a 2-core Celeron without AVX.

   One missed Flint expert at Q8_0 (3.3 MB) takes ~18 ms to reach its card. That is two thirds of a whole token.
   On Maya's PCIe 4.0 x16 machines the same miss takes ~0.13 ms (my calc).
4. **Maya itself cannot run on the rig.**
   - It asks for 32 GB of RAM and pins all but ~6 GB of it for its expert tier.
   - Its prebuilt CPU backend assumes AVX2.
   - On Pascal it runs only GLM-5.3-Flash, labelled "experimental, not validated", with no reports for P106/P104
     cards.
   - Its smallest GLM file (94.7 GB) is nearly twice the rig's VRAM. Modelled with Maya's own tiering, GLM-5.3-Flash
     would decode at roughly 0.4–1 tok/s here (see below).
5. **Little of it carries over: two small control ideas, both for work already planned.**
   - Maya rounds all of a call's activations to int8 in one launch (`MAYA_DS4_Q8_BATCH`). It saves a few hundred
     ~2 µs launches per token per GPU. That is the same kind of saving as Q38's remaining item, one activation
     quantization per shared input.
   - Its drafter verifies a draft window only while the product of the drafts' confidences stays ≥ 0.5 (+6.5 % on a
     Radeon 8065S). On Flint each extra verify position costs ~20–25 % of a token, so this cutoff belongs in the
     drafter controllers of Q53 and Q43.

   Per-expert bit allocation by routing frequency, skipping lightly weighted experts, error-feedback rounding and
   Maya's expert-blob relayout do not help the rig (reasons below).

## How Maya's tiering works

**Tiers.** These come from the README, the GLM study (`docs/GLM5-FLASH.md` §4) and `docs/SETTINGS.md`.
- **VRAM**: free VRAM, less a 700 MiB reserve, becomes a pool of expert slots (`--expert-cache auto`). The pool starts
  from a ranked profile (`expert-profile.bin`: the most-routed experts first). A live LFU with decayed counts then
  swaps in what the conversation routes. The counts are saved after every answer, so the next start begins warm.
- **RAM**: most of the free RAM is pinned for the next tier (all but ~6 GB). Experts evicted from VRAM drop into
  RAM, not back to the SSD. An expert still being copied down to RAM is waited for rather than re-read from the SSD
  (#24: SSD reads per token 3.0 → 2.2 on two V100s).
- **SSD**: the rest of the experts stay mapped in the GGUF files.
  - The mapping carries no access hint. `MADV_RANDOM` / `FILE_FLAG_RANDOM_ACCESS` were measured at disk speed,
    because the page cache then drops pages that are reused on every token.
  - As soon as the router has chosen a layer's experts, `posix_fadvise(WILLNEED)` / `PrefetchVirtualMemory` is
    issued for those few blobs, so the reads overlap the activation quantization and the previous layer's GPU work.
  - Prompts use several reader threads per expert (`STRATA_GLM_DISK_READERS`, `STRATA_GLM_DISK_QD`, up to 32 reads
    in flight) and a read-ahead buffer of up to 3 % of free RAM.
  - A cold start reads experts in file order, so a hard disk reads them in one pass.
- **Misses during decode** are computed by a CPU lane (AVX2/AVX-512 kernels, one thread per core). Alternatively they
  are copied over PCIe into staging slots and computed on the GPU, which DeepSeek's engine calls the "PCIe lane".
  - The choice is made per token from measured times of both lanes. It is kept only where it makes whole tokens at
    least 3 % faster.
  - "A fast CPU with a slow link streams little or nothing."
- **Skipping light experts (opt-in, changes answers)**: a routed expert that is in neither VRAM nor RAM and carries
  under ~10–14 % of its token's route weight is left out instead of read from the SSD (`STRATA_GLM_DISK_SKIP_W`,
  `MAYA_DS4_DISK_SKIP_W`). An auto mode applies it only while the cache is cold.

**How skewed routing is.**
- On Strata's 4-GPU IQ3_S machine (Qwen3.8-Flash-Next, 512 experts × 48 layers), holding 24 % of the
  (layer, expert) pairs in VRAM served **69.2 %** of reads. A frequency curve read off the profile ranking had
  predicted 94.9 % (`tools/make_profile.py`).
- The split cost model takes the routed mass of rank r as (r+1)^−1.2. It matched measured hit rates of 99.0–99.5 % on
  a two-GPU split that held nearly all experts (`src/program/generate.cpp`).
- For GLM, the study estimates ~25 % of experts in VRAM → ~70 % of reads; it is an estimate, not a measurement.
- The GLM profile shipped with Maya ranks experts and is not a frequency table: its counts give exactly 2x − x² of the
  reads for the top x of experts (my calc), which is a linear ramp.

**Speed when the tier works.**
- Two CMP 170HX cards (PCIe 2.0 x4 each) with every Maya-M expert in VRAM: 61.3 tok/s.
- One RTX 3090 with most experts in 141 GB of RAM on an 8-channel EPYC: 27.0 tok/s.
- The study's figure for a miss-heavy token is ~1.2 GB of NVMe reads ≈ 100–150 ms, i.e. ~5–8 tok/s cold.

## The quants

| | Maya-S | Maya-S24 | Maya-M | Maya-L |
|---|---|---|---|---|
| Size | 96.5 GB | 94.7 GB | 116 GB | 156.3 GB |
| Expert gate/up | IQ2_XXS | IQ2_XXS | IQ2_S | IQ3_S |
| Expert down (first/last four MoE layers; rest) | IQ3_XXS; IQ2_S | as Maya-S | IQ3_S; IQ3_XXS | Q5_K; IQ4_XS |
| Attention, shared experts | Q6_K | Q4_K | Q6_K | Q6_K |
| Same next token as FP8 | 83 % | 83 % | 86 % | 90 % |
| Zero-shot accuracy vs FP8 | 97.9 % | 97.7 % | 97.9 % | 99.2 % |

- Dense layers, embeddings, output and MLA k_b/v_b are Q6_K. Small KDA projections and the DSA indexer are Q8_0.
  Router, norms and stream mixing are F32. The MTP (NextN) layer is kept, at Q2_K–Q5_K.
- Calibration: 128 × 2,048 tokens in GLM's chat template (chat, multilingual, reasoning, web and other code, tool
  calls). M and L lean further toward tool calls and front-end code.
- Statistics come from the FP8 model layer by layer, with **a separate importance matrix per expert**.
- Expert gate/up tensors are rounded with **GPTQ-style error feedback** in 256-weight blocks, using each expert's
  own input statistics, with llama.cpp's quantizer choosing scales inside each block. The card says this cuts expert
  output error by about a quarter at the same size.
- Maya's own loader can relayout a GGUF into a "pack": one contiguous blob per expert, with gate and up rows
  interleaved so one pass over the activation serves both. Codes and scales are copied unchanged, so dequantization
  is bit-exact. The point is one contiguous read per expert from the SSD and one activation pass.
- The plan (`docs/QUANT-PLAN.md`) also proposes **per-expert types** in the pack: hot experts get more bits, the long
  tail fewer.

## Why it does not help the rig

**Opposite regime.** Maya spends VRAM where it saves the most slow-tier traffic. On the rig nothing goes to a slow
tier: the quality-bar files use 25–43 GB of 54 GB. A token's time is the bytes it reads from VRAM (~150 GB/s per card
with the fork's kernels) plus fixed per-layer costs. A cache in front of VRAM cannot reduce either.

**The rig's I/O, against one expert miss** (my calc; link and disk speeds measured, see the
[hardware spec](../../mining_rig.md)):

| Path | Bandwidth | Flint expert, Q8_0 (3.3 MB) | GLM expert, Maya-S24 (~7 MB) |
|---|---|---|---|
| Host RAM → the expert's card (PCIe 1.0 x1) | ~190 MB/s | ~18 ms (66 % of a 26.8 ms token) | ~37 ms |
| SATA SSD → host RAM | ~485 MB/s | ~7 ms | ~14 ms |
| Maya's reference: PCIe 4.0 x16 / Gen4 NVMe | ~25 / ~7 GB/s | ~0.13 / ~0.5 ms | ~0.3 / ~1 ms |

In a layer split each layer's misses travel over the single link of the card that holds that layer, so they queue
on ~190 MB/s. All nine links together reach only ~715 MB/s, because two PCIe 2.0 x1 switch uplinks each carry four
cards. With Flint-sized experts, a 1 % miss rate (~3 misses per token) would more than double the token time.

**GLM-5.3-Flash on the rig, modelled with Maya-style tiering** (my calc):
- Maya-S24's routed experts: 12,096 × ~7.0 MB ≈ 85 GB. The rest of the file is ~10 GB.
- VRAM for experts: 9 × ~5.9 GB usable, less ~10 GB of dense weights and ~0.5 GB per card for KV and buffers, leaves
  ~38 GB ≈ 5,400 experts ≈ 45 %. On Strata's measured curve that serves ~80–85 % of reads (claim to test).
- Misses: 15–20 % of 336 routed experts × 7 MB ≈ 0.35–0.47 GB per token. With 3.8 GB of RAM nearly all of it comes
  from the SSD:
  - 0.7–1.0 s at 485 MB/s;
  - 1.8–2.5 s to cross the owning cards' links one layer at a time;
  - at best ~0.5–0.65 s at the 715 MB/s aggregate with perfect prefetch.

  A CPU lane is no alternative: two 1.8 GHz cores without AVX.
- Result: **~0.4–1 tok/s**.
- Even with every expert in VRAM, GLM reads ~8–9 GB per token: ~2.4 GB of experts at ~2.3 bits plus ~9.5 B other
  active parameters at 4–6.5 bits. That caps it at ~15 tok/s before per-layer overheads, against Flint's 2.24 GB
  and 37 tok/s.

**Would an NVMe drive or more RAM change this?**
- **NVMe drive**: the chipset exposes only AHCI, so the M.2 slot takes SATA drives. An NVMe adapter in a GPU slot
  would run at PCIe 2.0 x1 behind a switch that four GPUs already share (≤ 500 MB/s). A USB 3.0 enclosure gives
  ~400 MB/s. Neither beats the SATA SSD, and the GPU links stay the limit either way.
- **32 GB of RAM** (the free SO-DIMM slot): it removes the SSD wait but not the x1 links. GLM would still run at
  ~0.4–2 tok/s (my calc). The upgrade matters for other reasons ([engines_other.md](engines_other.md)), not for
  this.

## What carries over

| Maya idea | Where it fits here | Expected effect |
|---|---|---|
| One int8 activation rounding per call instead of one launch per job (`MAYA_DS4_Q8_BATCH`, bit-exact) | Q38's remaining item: one activation quantization per shared input (Q/K/V, routed and shared gate/up); Q50 for the unfused shared expert | Fewer launches per token; the fixed per-launch cost matters on the 2-core host (claim to test) |
| Verify a draft window only while the product of draft confidences ≥ 0.5 (`MAYA_DS4_DSPARK_CUM`) | Drafter controllers: Qwen3.6's MTP (Q43), an EAGLE-3 head for Flint (Q20), the copy gate of n-gram drafts (Q53) | Fewer rejected verify positions, each worth ~20–25 % of a Flint token |
| Fusing SiLU·up with q8_1 output in the gate/up epilogue; the weighted combine and shared expert inside the down product | Not yet planned. The fork's kernels already fuse gate, up and SWIGLU; writing the down product's q8_1 input from that epilogue would drop its separate quantization launch | Fewer launches and one less pass over the activations |
| Per-expert input statistics for the importance matrix | Our imatrix files are per tensor (one for all 256 experts of a layer) | Unknown for 4- and 8-bit types; Maya uses it at 2–3 bits |

## What does not carry over

- **The tiering itself.** There is no slow tier on the rig, and one miss costs two thirds of a token (above).
- **Per-expert bits ranked by routing frequency.** Let expert e be routed with frequency f_e, stored in b_e bytes,
  and do s_e harm per use per unit of rounding error D(b_e). Then a token's time is proportional to Σ f_e·b_e, and
  the damage to Σ f_e·s_e·D(b_e).
  - At a fixed time per token, the optimum sets s_e·D′(b_e) equal for all experts. The frequency f_e drops out
    (my calc).
  - Frequency decides bits only when the budget is memory (Σ b_e fixed), which is Maya's case and not the rig's.
  - What could matter is the per-use harm s_e, but GGUF stores all experts of a layer in one tensor of one type, and
    llama.cpp has no per-expert types.
- **Skipping lightly weighted experts.**
  - Dropping one of Flint's six experts on every token gave only +4 % speed, and KLD against Q8_0 went from 0.316 to
    0.379 on the Q4_1 file (E2.8,
    [Flint benchmark](../../benchmarks/2026-10-22-flint.md)). A weight threshold drops fewer experts and saves less.
  - The files at the quality bar have no KLD margin to spend.
  - If the router dumps for Q51 record route weights, the share of route weight on each token's sixth expert is a
    free by-product. That is the only check worth making.
- **GPTQ-style error feedback.** The measured gains for 4-bit block formats are at noise level, and no rounding
  brings Q4_1 to Q5_0's quality ([quants/research.md](../../../quants/research.md), "Rounding"). Maya's quarter
  lower error is at 2–3 bits, where rounding error dominates.
- **The pack relayout.** It saves seeks and one activation pass. The fork's matrix-vector kernels already read the
  activation once per launch for several rows, and llama.cpp already fuses gate/up for the routed experts.
- **L1-cached global loads** (`-Xptxas -dlcm=ca`). llama.cpp's sm_61 kernels already use the read-only cached path
  ([Q48](../../../quants/experiments.md)).

## Sources

- Maya README, CHANGELOG (v1.0.x–v1.1.25), `docs/GLM5-FLASH.md` §3.5, §4, §5, `docs/SETTINGS.md` (the expert
  pool; DeepSeek PCIe lane, disk skip, explicit RAM residency, expert profile), `docs/QUANT-PLAN.md`,
  `tools/pack_layer.py`, `tools/make_profile.py`, `tools/glm_cache_sim.py`, `src/program/generate.cpp` (split cost
  model), `data/expert-profile-glm.bin`, all at `8412d5e` — [mw00/project-maya](https://github.com/mw00/project-maya)
  (P).
- Model card and file list — [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
  (P, fetched 2026-10-23).
- Rig I/O, measured: [hardware spec](../../mining_rig.md) (PCIe topology and host↔GPU bandwidth); disk interface
  and CPU flags read on the rig on 2026-10-23 (AHCI SATA controller only; `sse4_2`, no AVX).
