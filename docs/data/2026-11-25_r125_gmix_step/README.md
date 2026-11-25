# Where `gmix-q2`'s extra step time goes (R125, Q1409, 2026-11-25)

What this measured: why Qwen3.8-Flash-Next's better-quality file `gmix-q2` (agent KLD 0.236062 against 0.279228,
[its pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)) decodes slower than `pmix-q2-fast`, the file
it was built from (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF), product by product. R115 measured it 18.06 ms a step
slower on v4's line ([R115](../2026-11-12_r115_gmix_step/README.md)). Here, on nine P106-090 cards with the v5 engine
and Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)), each
file was loaded once under each of R121's two profilers ([R121](../2026-11-17_r121_flash_step_map/README.md)): the
group profile times every launch group of the four-token verify pass, and the host profile splits the step into the
verify pass, the draft and the server. Then R115's step pair ran again on both files, with and without
`GGML_CUDA_MMVQ_PASCAL_MT=1`. Registry row [Q1409](../../../quants/experiments.md).

Each value is one load. Every step and pass time is the repeated request (pass 2) unless a table says otherwise.

## Result

**87 % of the extra time is one kind of product: the routed experts' gate/up projections in `gmix-q2`'s IQ2_XXS,
IQ2_S and IQ2_XS layers.** At four tokens they take 0.94 to 1.05 ms a layer, where the same projection in Q2_0 takes
0.27 ms at about the same number of bytes, so the cost is in decoding the grid types, not in reading them. The dense
switches add about 2.3 ms. The draft, the server's own work and the hops between cards do not change.

### The step on v5 (no profiler)

ms a step on the repeated request, R115's step pair (`pu1`, `gu1`):

| Context | `pmix-q2-fast` | `gmix-q2` | `gmix` − `pmix` | % |
|---:|---:|---:|---:|---:|
| 0 | 95.73 | 113.78 | +18.05 | +18.86 |
| 8,192 | 110.71 | 128.37 | +17.66 | +15.95 |
| both (mean of the two) | 103.22 | 121.08 | +17.86 | +17.30 |

On v4's line R115 measured 107.80 and 107.77 against 125.83 and 125.86 (+18.06 ms, +16.75 %): the v5 engine made both
files faster and left the gap about where it was.

### Where it goes in the step (host profile, `ph` and `gh`)

Medians over each request's typical steps (four drafting passes a step), ms ([split/](split/)):

| Context | Part | `pmix-q2-fast` | `gmix-q2` | difference |
|---:|---|---:|---:|---:|
| 0 | the step | 96.68 | 114.83 | +18.15 |
| 0 | verify pass: GPU time of the nine cards | 80.63 | 98.78 | +18.15 |
| 0 | verify pass: the rest (hops, rows back, acceptance, server) | 3.56 | 3.57 | +0.01 |
| 0 | the draft's span | 12.45 | 12.48 | +0.03 |
| 8,192 | the step | 111.71 | 130.01 | +18.30 |
| 8,192 | verify pass: GPU time of the nine cards | 94.27 | 112.42 | +18.15 |
| 8,192 | verify pass: the rest | 3.64 | 3.65 | +0.01 |
| 8,192 | the draft's span | 13.78 | 13.83 | +0.05 |

The whole difference is GPU time of the verify pass. By card at depth 0 (verify GPU time, median ms): cards 0 to 8 take
10.56, 9.42, 9.33, 9.31, 7.98, 9.26, 7.62, 9.40, 7.71 with `pmix-q2-fast` and 15.01, 14.13, 12.09, 10.14, 11.52, 11.30,
7.65, 9.40, 7.70 with `gmix-q2`: cards 0 to 5 carry it (the switched groups are layers 0 to 31); cards 6 to 8 are
unchanged.

### The verify pass by part (group profile, `pg` and `gg`)

GPU time of a four-token verify pass by R41's parts, all nine cards, ms ([products.md](products.md), table 1). This
profiler synchronizes the stream after every evaluation, so a pass here is the sum of its cards' times:

| Part | `pmix-q2-fast`, depth 0 | `gmix-q2`, depth 0 | difference | `pmix-q2-fast`, 8,192 | `gmix-q2`, 8,192 | difference |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 20.388 | 36.210 | +15.822 | 20.965 | 36.462 | +15.497 |
| attention and recurrent projections | 18.345 | 20.300 | +1.955 | 18.471 | 20.367 | +1.896 |
| shared expert | 3.905 | 4.269 | +0.364 | 3.875 | 4.305 | +0.430 |
| every other part | 37.354 | 37.361 | +0.007 | 50.314 | 50.383 | +0.069 |
| **the pass** | **79.992** | **98.140** | **+18.148** | **93.625** | **111.517** | **+17.892** |

### The routed gate/up products by type

The fused gate/up product of the routed experts (`MUL_MAT_ID+MUL_MAT_ID+SWIGLU`, 2560 × 640 per expert), ms a verify
pass and ms a layer ([products.md](products.md), table 3; layers = launch groups a pass):

| Type | Layers in `pmix-q2-fast` | Layers in `gmix-q2` | `gmix-q2` ms, depth 0 | a layer | ms, 8,192 | a layer | a layer over Q2_0's, depth 0 | × layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2_0 | 42 (+6 fused with the shared expert) | 24 | 6.506 | 0.271 | 6.651 | 0.277 | — | — |
| IQ2_XXS | 0 | 11 | 10.310 | 0.937 | 10.300 | 0.936 | +0.666 | +7.33 |
| IQ2_S | 0 | 7 | 6.848 | 0.978 | 6.773 | 0.968 | +0.707 | +4.95 |
| IQ2_XS | 0 | 4 | 4.167 | 1.042 | 4.202 | 1.051 | +0.771 | +3.08 |
| IQ1_M | 0 | 2 | 1.094 | 0.547 | 1.102 | 0.551 | +0.276 | +0.55 |

- `pmix-q2-fast`'s Q2_0 layers take 11.381 ms over 42 layers (0.271 a layer) at depth 0 and 11.697 (0.279) at 8,192;
  its 6 layers whose shared gate/up is Q2_0 too run routed and shared gate/up as one launch group, 1.799 ms (0.300 a
  layer with the shared expert in it). In `gmix-q2` those layers' routed gate/up changed type, so their shared expert
  runs on its own.
- **The four grid types cost est. 15.91 ms a pass over Q2_0** at depth 0 (15.73 at 8,192): IQ2_XXS 7.33, IQ2_S 4.95,
  IQ2_XS 3.08, IQ1_M 0.55, against the routed experts' measured +15.82.
- `gmix-q2`'s 24 changed layers hold 2.224 bits a weight against Q2_0's 2.25
  ([C827](../2026-11-13_c827_gmix_iq_readonce/README.md)), so they read the same bytes and the time is the decode. The
  routed IQ products of 2–8 tokens run the multi-token grid kernel (`mmvq-pascal-mtok-iq`, Q713), which already reads
  and decodes each expert once for all its tokens (C827's reading of the dispatch).

### The dense switches

`denseiq31` and `denseiq32` (layer groups 1 and 2, the dense tensors from `pmix-iq3-fast`) cost +2.32 ms a pass at
depth 0 (+1.955 in the attention and recurrent projections, +0.364 in the shared expert). By product
([products.md](products.md), table 2): `attn_qkv` +0.952 (three layers in Q6_K at 0.504 ms a layer, where IQ4_XS takes
0.140 and Q4_K 0.110), `ssm_out` +0.674 (more layers in Q5_K and Q6_K), `attn_output` +0.196, `attn_gate` +0.140, the
shared expert's gate/up in IQ3_S and Q6_K +0.168 and +0.131, `attn_q` −0.071. Most of it is the larger types' bytes;
Q6_K's products with 2,560 inputs run at about 19 to 20 ps a weight (`attn_qkv` 0.504 ms for 26.2 M weights,
`attn_gate` 0.317 for 15.7 M) against 14 ps with 6,144 inputs (`ssm_out` 0.225, `attn_output` 0.221), est. 0.49 ms a
pass in `gmix-q2`.

### The read-once kernels (`GGML_CUDA_MMVQ_PASCAL_MT=1`)

ms a step on the repeated request, no profiler:

| Context | `pmix-q2-fast` unset (`pu1`) | `=1` (`pm1`) | difference | `gmix-q2` unset (`gu1`) | `=1` (`gm1`) | difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95.73 | 95.53 | −0.20 | 113.78 | 113.57 | −0.21 |
| 8,192 | 110.71 | 110.61 | −0.10 | 128.37 | 128.85 | +0.48 |

One load an arm, so no class. The switch moves no IQ product (C827: in these files it can take only the shared
expert's Q5_0 and Q8_0 down products), and it leaves `gmix-q2`'s extra time where it is. Every greedy reply with it
is the reply without it, on both files and both depths.

### Prompts

The 8,128-token prompt (the prime at 8,192 on an empty cache, no profiler) read at 218.9 tok/s with `pmix-q2-fast` and
148.0 with `gmix-q2` (−32.4 %); with `=1` 217.9 and 142.4 ([table.md](table.md)). R115 measured −17.43 % on v4's line
(152.2 and 150.6 against 122.5 and 127.5): v5's prompt gains reached Q2_0's products and not the grid types'. The
prompt was not profiled by product here.

### What would pay the quality back

- **A faster four-token routed gate/up kernel for IQ2_XXS, IQ2_S and IQ2_XS** (Q713's `mmvq-pascal-mtok-iq`), brought
  to Q2_0's 0.27 ms a layer, removes est. 15.36 ms of a verify pass (IQ2_XXS 7.33, IQ2_S 4.95, IQ2_XS 3.08), 85 % of
  the 18.15. IQ1_M (2 layers, 0.55 ms) is the same kernel's fourth type. Registry Q1871.
- **The one-token IQ2_XS and IQ2_XXS kernels (Q222) and an IQ1_M one-token kernel (Q711)** do not touch this cost: the
  verify pass has four tokens, so it runs the multi-token kernel, not the one-token ones.
- **The prompt:** `gmix-q2` reads long prompts 32.4 % slower on v5; the prompt products by type are the next
  measurement (registry Q1872).
- **The dense switches** cost about 2.3 ms, mostly their bytes; Q6_K's 2,560-input shape is est. 0.49 ms of it
  (in Q1871).
- What the file buys: agent KLD 0.236062 against 0.279228. With the routed kernel at Q2_0's rate the step gap would be
  est. 2.8 ms (+2.9 % at depth 0) instead of 18.15.

## How it ran

- **Engine:** `pascal-p106-v5`'s candidate `pascal-p106-v5-cand-b12674` at `eee615cdb` (`llama-server` 1.0.0-dev,
  build 13218). Nine P106-090 cards at stock clocks, the 4 GB host, nothing else running (the deep lane's quiet check
  before every load: no llama process, 0 MiB on every card).
- **Files:** `pmix-q2-fast` (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-0000{1,2,3}-of-00003.gguf`) and `gmix-q2`
  (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-0000{1,2,3}-of-00003.gguf`), shard 3 of both the same
  n-gram table; the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf`.
- **The line:** Flash-Next's v5 preset unchanged ([settings.env](../../../release/flash-next-v5/settings.env)), plus
  per arm only the profiler's variables or `GGML_CUDA_MMVQ_PASCAL_MT=1` ([run/r125.sh](run/r125.sh)). The deep lane's
  `MODEL` is `pmix-q2-fast`; every `gmix` arm's arguments end in `-m <gmix-q2's first shard>`, which llama-server
  takes as the last `-m` given (as in R115). A dry run checked that each `gmix` arm's server line is its `pmix` arm's
  plus that `-m`; each server log's `loading model` line names `gmix-q2` in `gg`, `gh`, `gu1` and `gm1` and
  `pmix-q2-fast` in the others. [loads.tsv](loads.tsv) holds every load's line as it ran.
- **Text:** the release check's recorded agent requests (`TAG='^t1swift/'`), prompts of exactly 64 and 8,192 tokens,
  256 tokens out at temperature 0, two passes per depth, then the sampler pass (`SAMPLER_PASS=1`), four idle seconds
  before every request (`PAUSE=4`).
- **Loads**, in order: `warm` (`pmix-q2-fast`, untimed), `pg`, `gg` (group profile), `ph`, `gh` (host profile),
  `pu1`, `gu1` (no profiler), `pm1`, `gm1` (`GGML_CUDA_MMVQ_PASCAL_MT=1`); 20:15–21:28. Loading took 306, 307, 307, 305
  and 307 s with `pmix-q2-fast` and 345, 346, 346 and 343 s with `gmix-q2`.
- **The checks:** both files wrote the same reply on every load at each depth and pass (the two files' replies
  differ, as in R115); no graph reallocation in any request.
- **Tables:** on a CPU, standard-library Python ([run/tables.sh](run/tables.sh)): R121's `split_trace.py` cuts each
  profile into one file per request, S37's `step_split.py` reads the host profiles, R49's `verify_parts.py` the group
  profiles (R41's parts and the launch groups by label), [run/products.py](run/products.py) compares the two files
  part by part and product by product, and the deep lane's `deep_table.py` makes [table.md](table.md).

## Files

| File | What it holds |
|---|---|
| [products.md](products.md), [products.tsv](products.tsv) | the verify pass by part, by product (label without types) and by label, both files, depth 0 and 8,192, both passes |
| [split/](split/) | the host profile's step split per load and request (`ph`, `gh`; `d0p1` … `d8192s`) |
| [verify/](verify/) | `verify_parts.py`'s tables per load and request (`pg`, `gg`): `parts.tsv`, `labels.tsv`, `per_card.tsv`, `accounting.tsv` |
| [table.md](table.md), [lane_classes.tsv](lane_classes.tsv) | the deep lane's own tables: tok/s, the step on each pass, prompts, the draft, the sampler pass, the checks |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran; per request its counts, times and reply hashes |
| [run/r125.sh](run/r125.sh) | the run through `bench/deep_lane/remote.sh` (dry run with its check, start, wait, collect) |
| [run/tables.sh](run/tables.sh), [run/products.py](run/products.py) | the tables from the lane's results folder |

The profile traces themselves (the group profiles are 82 and 95 MB a load) are not published; the tables above are
what the tools read from them.

## Commands

```bash
# from a machine that reaches the rig over ssh, at the repository root (paths are the rig's)
export RIG WORK=/tmp/r125 BIN=<llama-server> MODEL=<pmix-q2-fast shard 1> GMIX=<gmix-q2 shard 1> \
  MODEL_DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> TEXT_LOCAL=<proxy.jsonl.bodies.gz here>
D=docs/data/2026-11-25_r125_gmix_step/run
STEP=dry bash $D/r125.sh && STEP=start bash $D/r125.sh
STEP=wait bash $D/r125.sh && STEP=collect OUT=<folder here> bash $D/r125.sh
# the tables, on any machine with Python 3
S=$PWD OUT=<folder here> RES=<tables> bash $D/tables.sh
```

Provenance and licence: measurements on my own rig; the recorded agent text is the release check's
([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)); the model files are derived from
UkisAI's Swift-1.5 GSQ-RCO files under their licence.
