# The Qwen-V100 llama.cpp fork and Flash-Next on nine P106s (2026-11-05)

[pentaCoxian/Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp) (branch `qwen38-v100`, MIT, pushed
2026-11-03) is a llama.cpp fork that runs Qwen3.8-Flash-Next on two V100-SXM2-32GB cards without NVLink, with
the GGUF [pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF).
A post about it claims up to 187–188 tok/s on code editing and JSON extraction (speculative acceptance up to 95 %), 174
on code generation, 130 at 142K context and 146 on maths with thinking. It credits a tuned layer split, "over 50
low-level PCIe optimizations and operator fusions", a 200 W power cap, an old Xeon E5 host, MTP speculative decoding
with "a custom 64K multilingual draft vocabulary", a Q8_0 vision projector (904 → 613 MB), and 262K context in
32.26 of 32.77 GB on the first card with 32 GB of host RAM.

This note covers three questions about the fork: whether it is safe, whether its numbers hold up against the V100's
bandwidth, and which of its ideas carry over to Miner Llama on nine P106-090 cards. The repository at `6209237`,
Warabii1248's patch series it imports and the model card at revision `75e6411` were read as text on 2026-11-05.
Nothing was built, run or downloaded beyond text files, and no weights were fetched. Numbers marked **est.** are
arithmetic from the linked figures.

The rig's reference line is S35 ([data](../../data/2026-11-04_s35_stack/README.md)): pmix-q2-fast, `-kshape`, the MTP
draft with `LLAMA_DRAFT_SHORTLIST=32768,8192,2`, and Q8_0 hops. It generates 18.8 tok/s on the agent check at 0.708
acceptance with 3 drafts, which is est. 3.1 tokens and ~165 ms a step. Plain decode is 17.06 tok/s. A saving of x ms a
step is worth x / 3.1 ms a token.

## Summary

1. **The fork shows no sign of malware.** It is upstream llama.cpp 0.5.0 (`7fe450e`, the same commit upstream) plus
   ten commits: 138 files, ~25,000 lines of code and a 65,537-line list of token ids. It contains no binaries, pickles,
   release assets, new CI workflows or build hooks. The added C++ and CUDA make no network calls and run no processes.
   The two imported patch sets match their published sources exactly: the model card's `llama-v0.5.0-v100-v3.patch`
   and Warabii1248's `pcie-20261016`, each with only a licence file added (same `git patch-id`). The deploy scripts
   talk only to a llama-server on 127.0.0.1. Section 2 has the details.
2. **The speed numbers are plausible, and the headline ones are copy tasks.** A step is one 4-token verify plus three
   draft passes. It reads est. ~8 GB: ~4.6 GB of dense weights, ~3 GB for the union of experts, and ~0.6 GB of drafts.
   At ~830 GB/s that is a floor of est. ~10 ms. The measured 20–24 ms a step is 2.0–2.5× that floor, similar to the
   rig's efficiency. The 187–188 tok/s come from rewriting a 130-line file with one symbol renamed and from JSON
   extraction from a given text. There the draft is accepted 95 % of the time, which gives 3.85 tokens a step. The
   fork's own mixed benchmark gives the representative figure: **115 tok/s at 54 % acceptance**. Nothing was measured
   on multi-turn agent sessions, and quality was checked only by smoke tests, needle checks and a perplexity of 2.85.
3. **The IQ3E-Q8D file does not carry over.** Its routed experts are Unsloth's IQ2_S gate/up (2.56 bits) and IQ4_NL
   down (4.5 bits). Its dense tensors are Q8_0, with a Q6_K output head, a separate Q4_0 head for the draft and a Q8_0
   MTP layer. That is ~58 GB of weights on the cards, against the rig's ~54 GB of VRAM, and est. 5.5 GB a token,
   against Swift Q2_0's 4.16 GB. The parts that do carry over are already rows here: Q8_0 hyper-connection mixers (Q218)
   and the per-type Pascal mix (pmix).
4. **The 64K draft vocabulary is mostly what the rig already has.** The list holds every id below 32,768 plus 32,768
   ids chosen by frequency in the model's own replies and in local text, with Chinese, Japanese and Korean (CJK) tokens
   included. The rig's shortlist is the ids below 32,768 plus the special tokens and up to 8,192 of the context's
   tokens. Doubling the rows costs a draft pass est. +0.4 ms for est. +0.5–1.5 points of acceptance, which is a wash.
   Ranking by frequency is the part worth testing (Q1103, with Q636).
5. **What carries over, ranked by est. ms a token on S35's line:**
   - **Window lookahead** (Q1109): the next 4-token window enters the first cards while the last cards verify the
     current one. Est. −2 to −8 ms. This is the least certain estimate: its first test is an offline count of how often
     a window would hit, and Q550's one-token form lost 1.6× on the agent check.
   - **Fused half-layer ops for the verify window** (Q1101), from the fork's "A7" kernels. They use only `dp4a` and no
     tensor cores, so they can be ported. Est. −1.3 to −2.6 ms, in 3–6 weeks of work.
   - **Prefetching the n-gram table's rows** (Q1100). The rig's engine already has the helper but never calls it for
     qwen4exp. Est. −0.1 to −0.7 ms, and prompts +1–5 %, for a few lines of code.
   - **`dp4a` sparse-attention scores over a Q8_0 KV cache** (Q1108): est. −0.3 to −1 ms at 16–32k of context, and
     half the KV memory.
   - **Smaller items:** a count of graph rebuilds per step (Q1104, 0 to −1 ms), a frequency-ranked shortlist (Q1103,
     0 to −0.8 ms), a compact Q4_0 draft head (Q1102, −0.15 ms and ~450 MiB on the draft's card).
   - **Prompt-side items:** a draft catch-up that writes only its KV cache (Q1105), a catch-up that no longer drains
     the pipeline (Q1106) and a count of scheduler re-reservations (Q1107).
6. **The fork also gives evidence for rows already queued here:**
   - a 16-bit router copy is safe: 2 of 276,000 top-10 sets differed, both exact ties (Q865);
   - greedy drafts cost 5 points of acceptance and 7 % of speed against probabilistic drafts (Q731);
   - fused hyper-connection kernels for 1–8 tokens exist as a reference implementation (Q439, Q557);
   - tensor split loses to layer split without NVLink (34 against 22.7 ms a step), as the rig's notes expected.

## 1. What the fork is

| Commit | Source | What it adds |
|---|---|---|
| `84cd765` | pentacoxian-dev's `llama-v0.5.0-v100-v3.patch` from the model card (identical; plus `LICENSE-freetoken`) | the `qwen4exp` architecture with MTP, the FreeToken MoE port (Apache-2.0), sparse QSA attention, fused hyper-connection kernels, GPU argmax, a tensor-split AllReduce, and the "A7" fused ops |
| `e1a030f` | Warabii1248's [qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches) `pcie-20261016` (identical; plus `LICENSE-pcie-patches`) | 56 prefill and decode features, all behind switches; about eight of them are traces or measurement modes |
| `e58e25e`, `618a48a`, `8b24c5c`, `e0a9ff2` | the fork's own | graph reuse when only output rows moved, the IQ2_S gate/up stage at 512 threads, the F16 router copy, and the fused FFN op on the full-attention layers; each commit message gives its timing (−0.4, −0.22, −0.10 and −0.79 ms a step) |
| `6626df6`, `80a5c99`, `6209237`, `c8e099e` | the fork's own | `deploy/qwen38-v100/`: the launcher, the environment (52 switches), the draft list, benchmark scripts and a landing page |

The model card (revision `75e6411`, created 2026-09-24) holds one 85.9 GB GGUF and the patch history (v1 to v3; v4 and
v5 were withdrawn). The same author published the IQ3_XXS_M64 and `_HC8` files that
[the Flash-Next news note](flash_next_news_2026-11.md) lists. Licences: the code is MIT, the FreeToken files are
Apache-2.0, and the weights are under the Qwen Community License 1.0. That licence requires a separate licence from
Qwen for a commercial model-as-a-service or coding-assistant business. Internal use is exempt.

## 2. The safety audit

The repository was handled as untrusted. It was cloned at depth 12 with hooks, symlinks, LFS filters and submodules
off, then made read-only. It was read with `git`, `grep`, `sed` and `cat` only. Nothing was built, run or installed.

| Check | Result |
|---|---|
| Base | the merge base `7fe450e` is upstream's own commit, so every file outside the 138 changed ones is upstream's |
| Imports | `git patch-id` of each import commit, with its added licence file excluded, equals that of the published patch: the v3 patch from the model card, and `pcie-20261016` from Warabii's repository, whose own SHA256SUMS line matches |
| Binaries, pickles, archives, release assets | none (no binary entry in the diff; no releases or tags) |
| CI and build | `.github/` gains only a README; the CMake changes only add source files and one test |
| Exec, network, sockets, `dlopen`, `system` in the added C++/CUDA | none |
| File writes in the added C++ | the prompt-cache spill directory (`O_EXCL`, mode 0600, its own directory), an opt-in acceptance log (`LLAMA_SPEC_ACCLOG`), an opt-in MoE capture file (mode `wx`), and a read of an opt-in lookahead-level file |
| Inline assembly | two upstream lines (an AMD dot product and an L2 prefetch) |
| Hidden Unicode, control characters, base64 runs, long hex blobs | none in the changed files |
| Python and shell in `deploy/` | HTTP to `127.0.0.1:18080` only; `nvidia-smi` and `pgrep` for VRAM and RSS; the vocabulary tools tokenize local text (Python's standard library, `/usr/share/doc`, the repository's docs) through the local server |
| Draft list | 65,537 lines: one comment and 65,536 integers from 0 to 248,319 |

Verdict: **clean**. If it is ever built, the risks are the usual ones for a large CUDA patch from a stranger: an
unreviewed 25,000 lines in the server process, and the model card's suggested `pip install nvidia-nccl-cu12` for its
build. It cannot run on a P106 anyway: it targets sm_70 with CUDA 12.8 or later, and its tensor-core kernels need Volta.

## 3. The file: IQ3E-Q8D against the rig's files

The types come from the model card and from the fork's kernel headers (`ggml-a7.h`, `a7-gdn.cu`). They are known
without reading the GGUF header.

| Part | IQ3E-Q8D (V100) | Swift Q2_0 (the rig's base, [census](flash_next_floor_and_engine.md)) |
|---|---|---|
| Routed experts (10 of 512, 2,560 × 640) | gate/up IQ2_S (820 B a row), down IQ4_NL (360 B a row): 1.97 MB an expert, est. 0.95 GB a token, 48.4 GB in all | Q2_0: 0.66 GB a token |
| Attention and recurrent projections | Q8_0 (`wqkv` 2,560 → 10,240, `ssm_out` 6,144 → 2,560): est. 2.87 GB a token | mixed Q3_K / IQ4_XS / K-quants: 1.39 GB |
| Hyper-connection mixers | Q8_0: est. 0.68 GB | BF16 1.28 GB (Q8_0 in the retyped files, Q218) |
| Output head | Q6_K, 0.52 GB; the draft passes use a separate Q4_0 head | Q5_K, 0.44 GB |
| Router | F32 in the file (a 16-bit copy at load with `GGML_A7_ROUTER_F16`) | F32 (Q865 plans a 16-bit copy) |
| MTP layer | Q8_0, appended from the original checkpoint | the separate MTP draft file |
| n-gram (PLE) table | ~27.5 GB, memory-mapped in host RAM | the same table, memory-mapped from the SSD |
| **On the cards** | **~58 GB with the MTP layer; est. 5.5 GB a token** | **~30 GB; 4.16 GB a token** |

The file does not fit the rig's ~54 GB of VRAM and reads 1.33× the bytes of a token. Its two ideas are dense tensors in
Q8_0, which suit the V100's matrix-vector kernels, and experts at IQ2_S/IQ4_NL. On Pascal, pmix already chooses each
dense type by its measured kernel speed. Swift IQ3_XXS (43.8 GiB of weights, on hand) is a better donor of
higher-bit experts than a download of Unsloth's IQ2_S would be (Q1110, parked).

## 4. The speed claim against the V100's bandwidth

Tokens a step = 1 + 3 × acceptance, at 3 drafts a step. ms a step = tokens a step / tok/s. The fork's own `bench.py`
computes the same.

| Task (the fork's table) | tok/s | Acceptance | Tokens a step | ms a step |
|---|---:|---:|---:|---:|
| Code edit: rewrite a given 130-line file with one rename | 188 | 0.95 | 3.85 | 20.5 |
| JSON extraction from a given text | 187 | 0.94 | 3.82 | 20.4 |
| Code generation | 174 | 0.84 | 3.52 | 20.2 |
| Maths, thinking on | 146 | 0.71 | 3.13 | 21.4 |
| Long-context Q&A, 142K prompt | 130 | 0.73 | 3.19 | 24.5 |
| **Mixed benchmark (five prompts × reps)** | **115.1** | **0.542** | **2.63** | **22.8** (measured 22.75) |
| Creative prose | 94 | 0.42 | 2.26 | 24.0 |

- **Floor of a step (est.):** a 4-token verify reads the dense ~4.6 GB once plus the union of the four tokens' experts.
  On real text that is ~30 experts a layer (68–77 % of independent routing, [MoE speculation](moe_speculation_union.md)),
  ~2.9 GB. The three draft passes each read the MTP layer plus a 64K-row Q4_0 head, ~0.2 GB. In all ~8 GB, or ~10 ms
  at ~830 GB/s. In a layer split one card works at a time, so one card's bandwidth is the limit.
- **Measured:** 20–24 ms, 2.0–2.5× the floor. The fork says the remaining ~20 ms of a verify window is bound by
  Volta's integer throughput on these formats, not by memory. That matches the rig's finding that the 2-bit expert
  kernels are instruction-bound ([floor note](flash_next_floor_and_engine.md) §2.2).
- **Plausible:** every row of the table, the 262K context in 32.26 GB (~58 GB of weights, a ~3.3 GB Q8_0 KV cache at
  262K, the projector and buffers across 64 GB), and the 32 GB host with a 27.5 GB memory-mapped table (peak anonymous
  memory 5.6 GB).
- **Not shown:**
  - any multi-turn agent session;
  - quality beyond smoke tests, needle checks and a 4K-chunk perplexity of 2.85;
  - default sampling settings in the benchmarks (no temperature is set);
  - "over 50 optimizations": it is 56 switches, of which about eight are traces, debug modes or ghost runs.

  "125B MoE" is the trunk alone: the file holds 180.2 B parameters with the ~51 B-parameter n-gram table and the MTP
  layer.

## 5. Idea by idea

"Est. ms a token" is on S35's line (about 165 ms and 3.1 tokens a step). "Ours" names what Miner Llama has today.

| Their feature | What it does | Ours | On 9 × P106 | Est. ms a token | Row |
|---|---|---|---|---:|---|
| Stage-0 lookahead + draft-ahead (`LLAMA_LOOKAHEAD`, `LLAMA_DRAFT_AHEAD=4`) | the MTP chain drafts past its window. The first card starts the next window on the guess that all drafts and the bonus token are accepted, and drops it otherwise. Turning it off cost 2 ms of a ~24 ms step | one-token pipelined speculation (Q87, Q550: exact, but 1.6× slower on the agent check, Q593) | 8 of 9 stages overlap instead of 1 of 2, so a hit saves ~8/9 of a verify. Needs the share of windows that hit and a cheap state rollback | **−2 to −8** | Q1109 |
| A7 fused ops (`GGML_OP_A7_GDN_FFN`, `_ATTN`) | one op for each half of a layer at T = 4: hc mix, router with a routing table built on the card, the experts used by the four tokens, gate/up, down with the hc combine; and the attention/GDN half. Uses `dp4a`, no tensor cores | per-op kernels, plus the recurrent fusions (Q436–Q438) | portable in principle (`dp4a`, ~26 KB of shared memory a block); needs retuning for 5 SMs and Q2_0 / pmix types | **−1.3 to −2.6** | Q1101 |
| n-gram row prefetch (Warabii's backport of #29599) | `madvise(WILLNEED)` on the rows of a micro-batch's n-gram indices when they are built | the helper is in the base and called only by gemma4 and deepseek41 | the rig's table is on the SSD: 3.6–5.7 page reads a token, 0.15–0.87 ms of serial latency | **−0.1 to −0.7**; prompts +1–5 % | Q1100 |
| `dp4a` QSA scores over a Q8_0 KV cache (`GGML_CUDA_QSA_DP4A`, `_HPW6`) | q quantized once per 32 values, int8 dot products against the Q8_0 cache; six query heads per warp at GQA 12 | the Pascal sparse kernel (`GGML_CUDA_FATTN_SPARSE_PASCAL`, off; Q230) over an F16 cache | P106 has `dp4a`; the cache's bytes halve | −0.3 to −1 at 16–32k | Q1108 |
| `LLAMA_MTP_PAD=4` + `LLAMA_REUSE_OUT_ORDER` | draft passes padded to one shape, and graph reuse when only output rows moved: 58 → 27–31 rebuilds per 200 micro-batches, −0.4 ms a step | per-size graphs (`LLAMA_GRAPH_PER_SIZE=1`) | a graph build costs more on a 2-core Celeron; whether the draft rebuilds is unmeasured | 0 to −1 | Q1104 |
| 64K draft list (`LLAMA_MTP_DRAFT_VOCAB`) | the draft head over 65,536 rows: every id below 32,768 plus 32,768 by frequency | ids below 32,768 + specials + ≤ 8,192 context tokens (Q592) | +0.4 ms a draft pass for est. +0.5–1.5 points; frequency ranking at equal rows is the better test | 0 to −0.8 | Q1103 |
| Draft-only Q4_0 head | the draft passes read a separate Q4_0 head; the target keeps its Q6_K head | the draft file's Q6_K head over the shortlist | 68.8 → 47.2 MB a pass at 32K rows; ~450 MiB less on the draft's card if only the shortlist rows are kept | −0.15 (memory) | Q1102 |
| KV-only MTP catch-up (`LLAMA_MTP_KV_ONLY`; llama.cpp #29523, closed unmerged) | prompt rows of the MTP layer write its KV and skip the rest | the full catch-up | MTP costs prompts −5 % today ([prefill note](flash_next_prefill.md)) | 0; prompts +1–3 % | Q1105 |
| `mtplag` + `mtpasync` | the catch-up of a mid-prompt batch is deferred and runs on a worker thread, so the pipeline does not drain | the catch-up drains at each `-b` boundary | nine stages drain longer than two; the Celeron has two cores | 0; prompts +2–4 % | Q1106 |
| `reservetrace` / `reserveheal` / `planahead` / `qsareserve` | finds and removes scheduler re-reservations during a prompt, each of which drains the pipeline | none; the prefill note leaves 60–75 % idle cards unexplained | a count is cheap; the fix only if any are found | 0; prompts 0 to +30 % | Q1107 |
| `hcfused` / `hcpar` (`LLAMA_HC_FUSED=2`) | hyper-connection mix and combine as fused kernels for 1–8 tokens | Q439 (Strata's two-kernel read), Q557 (inject fusion) | yes | (Q439's est.) | Q439 |
| `GGML_A7_ROUTER_F16` | F16 router copy at load; 2 of 276,000 top-10 sets differed (exact ties); −0.10 ms a step | Q865 (planned) | yes | (Q865's est. −2.1) | Q865 |
| Probabilistic drafts (`--spec-draft-sampling probabilistic`, the backported #27694) | rejection-sampled drafts; greedy drafts cost them 5 points and 7 % | in the base, unused on the agent line (Q731) | yes, at the agent check's temperature 1.0 | (Q731's est.) | Q731 |
| `GGML_CUDA_GRAPH_OPT=1`, `CUDA_SCALE_LAUNCH_QUEUES=4x` | concurrent Q/K/V branches; a larger launch queue | Q445; the [4 GB host note](host_ram_4gb.md) (4x costs ~120 MB of host RAM per card) | doubtful on 5 SMs; 4x is too much RAM for 4 GB | — | Q445 |
| `ckpttail`, `--batch-size 3072`, `--checkpoint-min-step` | no lone last micro-batch from the tail checkpoint; checkpoints only between batches | Q263 (`LLAMA_SERVER_TAIL_CKPT_MIN`), the prefill note's plan | yes | (Q263) | Q263 |
| MoE dedup over the verify tokens, L2 launch order | each selected expert read once for 2–8 tokens; the row tile as the fastest grid dimension | the multi-token expert kernels (C70, Q591, Q825) | the dedup is in place; L2 order has little to catch in 1.5 MB | ~0 | Q1114 |
| Tensor split with a P2P AllReduce | both cards on every layer | none | no peer access, x1 links; they measured 34 against 22.7 ms | — | ([engines note](engines_llamacpp_family.md)) |
| Chunked GDN prefill on tensor cores | the WY form of the delta rule | the per-token scan | no tensor cores; the scan is ~1.4 % of a prompt token here (prefill note §4) | — | — |
| Q8_0 vision projector + 2,048-token image cap, on-demand eviction | images at 262K | no image input | ~1.1 GB on one card | — | Q1111 |
| Prompt-cache spill to disk (`pcdisk`) | `--cache-ram` evictions written to their own directory; 3.8 s to restore 142K tokens | `--prefix-cache DIR` (Q111, Q132) | matters only with several conversations | — | Q1112 |
| CPU sampler top-k only (`sampletopk`) | candidates only for the top-k | sampling on the card, for the target and the draft | only where the host samples | — | Q1113 |
| FreeToken expert cache (`--moe-expert-cache`, `--ft-native-moe`) | experts kept in VRAM slots when they do not fit | none | the experts fit; a miss over x1 costs ≥ 5 ms | — | Q1115 |
| 200 W cap, NUMA binding, `--parallel 1`, `--spec-draft-n-max 3` | — | the keep-alive (C533), a single-socket host, the same draft settings | — | — | — |

## 6. What to take

### 6.1 Window lookahead (Q1109)

On a layer split, cards 0–7 sit idle for ~150 of a step's ~165 ms. The V100 fork's draft-ahead lets the MTP chain
draft N more tokens on the draft's card while the current window runs. Its stage-0 lookahead starts the next 4-token
window on the first card as soon as that card has finished the current one, on the guess that all three drafts are
accepted and the bonus token equals the draft's fourth guess. A miss drops the window and restores the recurrent
state. The fork's rollback depth is 3, and its recurrent-state checkpoints cost ~35 ms when a draft goes deeper. On
two stages it saves 2 ms of ~24 even at 54 % acceptance, which means est. ~17 % of windows hit.

On nine stages a hit saves ~8/9 of a verify, est. ~120 ms. At the same hit rate that is est. −10 to −25 ms a step,
or −3 to −8 ms a token, before the miss costs. On a layer split those costs are a state restore on each card
(~12 MiB a card in the card's own memory) and the draft-ahead passes. The passes run on the draft's card while it
would otherwise wait for the window.

Q550's one-token pipelined speculation lost 1.6× on the agent check, so the first step is offline. From existing
`LLAMA_SPEC_DUMP` logs, count the share of steps whose three drafts are all accepted and whose bonus token equals a
fourth draft. Below ~10 %, park it with that number.

### 6.2 The A7 fused half-layers (Q1101)

`a7-gdn.cu` (2,333 lines) implements Flash-Next's FFN half of a layer at T = 4 as one op of about seven stage kernels.
`a7_ffn_rquant` makes the int8 activations. `hc1f` and `hc2f` do the hyper-connection mix. Then come the router, which
also builds the routing table on the card (softmax, top-10, the union of the four tokens' experts), the gate/up stage,
the `qh` stage and the down stage with the hc combine and the next layer's int8 input. The attention/GDN half is a
second op of the same kind.

The fork's last extension of it, to the 12 full-attention layers, replaced ~10 kernels each and took the step from
23.60 to 22.81 ms (+3.4 % tok/s). The kernels use `dp4a` and `half2` only, so they can be ported to compute 6.1. The
port needs other shared-memory tiles for Pascal's 48 KB a block and 5 SMs, and stages for Q2_0 experts and pmix's
dense types.

This is the floor note's option (b), a Flash-Next-only executor, for the verify window, with a working reference for
this exact model. Est. −4 to −8 ms a pass (~1,000 of ~2,900 kernels removed, plus the ramp and tail cost of the
small ones), or −1.3 to −2.6 ms a token, in 3–6 weeks of work. The gate is Q867's: a fused real layer ≥ 15 % faster
than the fork's graph. Start with the FFN half.

### 6.3 Prefetching the n-gram rows (Q1100)

Upstream #29599's `llama_prefetch_rows` is in the base and called by gemma4 and deepseek41 only. The V100 fork calls
it for qwen4exp when it builds a micro-batch's n-gram indices. On the rig the 27.5 GB table stays on the SSD behind a
~0.75 GB page cache. The get_rows op then faults 3.6–5.7 pages a generated token one after another, est. 0.15–0.87 ms.

Two changes: issue the whole micro-batch's rows at once when the indices are built (parallel reads), and issue each
drafted token's rows as soon as it is drafted, before the verify needs them. Est. −0.1 to −0.7 ms a token, and
prompts +1–5 % (the rows are 0.1–0.7 ms of a prompt token, prefill note §3). It is a few lines, exact by construction,
and needs one check that `WILLNEED` does not push the draft or the server out of a 4 GB host.

### 6.4 `dp4a` sparse-attention scores over a Q8_0 cache (Q1108)

The fork's QSA kernel reads a Q8_0 KV cache directly, quantizes q once per 32 values and scores with `__dp4a`. Its
`hpw6` variant puts six query heads in a warp at GQA 12, so each K row is read once for six heads. Compute 6.1 has
`dp4a`. On top of Q230's Pascal sparse kernel, a Q8_0 cache halves the bytes of the indexer's and the selected cells'
reads and the KV memory (12 layers × 2 KiB a cell in F16 → ~1.06 KiB: −0.7 GB at 64k).

Est. −1 to −3 ms a pass at 16–32k, or −0.3 to −1 ms a token. A KLD check of the Q8_0 cache comes first, on the quant
machine.

### 6.5 Smaller items

- **Graph rebuilds per step (Q1104).** In a profiled load, count the graph builds and their host time per step. With
  backend samplers on, the fork found that the MTP draft context alternated between two output layouts and rebuilt
  twice a step. If the rig's draft does the same, port `LLAMA_MTP_PAD` and `LLAMA_REUSE_OUT_ORDER`. Est. 0 to −3 ms a
  step.
- **A frequency-ranked shortlist (Q1103)** at 32K rows (Q636's agent-output list), and the fork's 64K list, each as a
  frequency-permuted head so that the extra rows are a plain view and not a `mul_mat_id` gather. Replay coverage on
  the agent check's replies first.
- **A compact Q4_0 draft head (Q1102)** with only the shortlist rows. The fallback rounds read the target's head
  (`LLAMA_DRAFT_SHARE=head`). It saves ~450 MiB on the draft's card, which Q840 and C409r's draft-on-one-card arms
  need more than they need the 0.15 ms.
- **The draft catch-up on prompts (Q1105, Q1106):** write only the MTP layer's KV for prompt rows, and defer the
  catch-up so that the nine-card pipeline does not drain at each `-b` boundary. Together they recover most of the 5 %
  that MTP costs prompts.
- **Scheduler re-reservations during a prompt (Q1107)**, counted with one log line each, as a suspect for the
  prefill note's 60–75 % idle cards, beside Q157 and Q264's per-card profile.

## 7. Parked

- **Unsloth's IQ2_S experts as a pmix donor (Q1110).** IQ4_NL down alone is 22.6 GB, and IQ2_S gate/up + Q2_0 down
  is est. +3.3 GB on the cards. Swift IQ3_XXS, already on hand, is a higher-bit donor of the same tensors. Reopens if
  pmix's KLD shows ~2.5-bit gate/up as the sweet spot and no GSQ-RCO file has it.
- **Image input on the rig (Q1111):** a 613 MB Q8_0 projector and a ~0.5 GB vision buffer on one card. The agent check
  is text only. Reopens when the rig is to take images.
- **Prompt-cache spill to disk (Q1112):** one agent session does not evict. Reopens with multi-conversation serving.
- **A CPU sampler that builds only the top-k candidates (Q1113):** served tokens and drafts sample on the card.
  Reopens if a profile shows host sampling time.
- **L2 launch order for the multi-token expert products (Q1114):** the multi-token kernels read each expert of the
  union once. Reopens if Q866's chained bench shows repeated expert reads.
- **A VRAM expert cache for experts that do not fit (Q1115):** Flash-Next's experts fit, and a miss over a x1 link
  costs ≥ 5 ms. Reopens for a model whose experts exceed the cards.

## 8. How it was read

- **GitHub API:** the repository, its branches, commits, events and forks; the compare against `7fe450e`; the account;
  Warabii1248's repository, its tree, its `README.md`, `fn-192k.env`, `SHA256SUMS-pcie-20261016` and the 745 KB
  cumulative patch.
- **Hugging Face API:** the model's metadata with its GGUF summary, the file list and the sha256 values; then
  `README.md`, `LICENSE` and `patches/llama-v0.5.0-v100-v3.patch` at revision `75e6411`. The viewer's per-tensor table
  is built in the browser from a range read of the 85.9 GB file, so it was not used. The tensor types come from the
  fork's kernel headers.
- **The clone:** shallow, read-only, deleted after the review. The rig-side facts come from the linked notes and from
  `git grep` on Miner Llama's `-ladder4`.

## Sources

- [pentaCoxian/Qwen-V100-llama.cpp](https://github.com/pentaCoxian/Qwen-V100-llama.cpp), branch `qwen38-v100` at
  `6209237df53569f07429f3f79fc5354de5fe704f`; [`deploy/qwen38-v100/README.md`](https://github.com/pentaCoxian/Qwen-V100-llama.cpp/blob/qwen38-v100/deploy/qwen38-v100/README.md).
- [pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF),
  revision `75e6411f0157bb4b6d492a9868bc93c57d2b061c` (file sha256 `42506dd9…e3e50e8`, 85,923,705,344 bytes).
- [Warabii1248/qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches), `pcie-20261016` (patch sha256
  `51faaf10…8291b3`).
- The draft list: `deploy/qwen38-v100/draftvocab-64k-ml.txt` at `6209237` (git blob `dc366c94`, sha256 `bba286c9…199989`).
- llama.cpp #29599 (row prefetch, merged 2026-09-30), #27694 (probabilistic MTP drafts, merged 2026-10-02), #29523
  (KV-only MTP catch-up, closed unmerged), #29353 (single-kernel GDN).
