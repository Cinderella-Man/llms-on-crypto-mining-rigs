# Sampling for tool requests: what Flint loses without backend sampling, and how to keep it (2026-10-23)

Scope: llama-server's backend sampling (`-bs`) picks the next token on the GPU, so the 150,144 logits of a Flint token
never cross the PCIe 1.0 x1 link. llama-server switches it off for every request that carries a grammar, and agent
clients send `tools` with almost every request. This note covers four things:
- exactly when backend sampling is off, in the fork (`pascal-p106`, head `eeaa59660`, upstream base `d77256999`) and on
  upstream master (`36e36e6`, 2026-10-23);
- what a tool request costs per token on the rig;
- which designs keep tool requests on the GPU path, and whether they preserve llama-server's sampling semantics;
- ranked experiments with commands.

Conventions are those of the other notes:
- "Claim to test" marks anything not measured on this rig.
- "(my calc)" marks this note's own arithmetic; "(est.)" marks a desktop measurement scaled to the rig's Celeron.
- "P" and "S" in source lists mean primary (code, docs, measurements in the repo) and secondary.

**Host-side numbers not taken on the rig** come from a host-only benchmark run on one desktop Zen 4 core (up to 5.4 GHz).
- **Build:** the fork's `libllama` and `common`, compiled for x86-64-v2 (SSE4.2, no AVX: the Celeron's instruction set).
- **What it replays:** llama-server's sampler for a Flint tool request, built from Flint's vocabulary and chat template
  (read from the GGUF) and nine OpenCode-style tools (bash, read, write, edit, glob, grep, list, todowrite, webfetch).
- **Sampling:** Flint's defaults (temperature 1.0, top-p 0.95, top-k 20) plus the server's min-p 0.05.
- **Token stream:** 285 tokens of reasoning and text, then a `write` call carrying a 2 KB Python file (411 tokens), then a
  `todowrite` call with a JSON array (64 tokens). The logits are synthetic, with the intended token winning.
- **Scaling to the rig:** the Celeron 3865U (1.8 GHz, 2 MB L3, single-channel DDR4) is taken as 4–6× slower on these
  short loops (est.). The full-row grammar pass, which touches every token's string, is taken as 4–10× slower.

**Starting point on the rig, measured:**
- **Bench:** S4-L0 decodes 37.4 tok/s in llama-bench tg128 on six cards (26.8 ms per token). llama-bench requests logits
  for every generated token and binds no sampler (`tools/llama-bench/llama-bench.cpp:2210-2219`). Its 26.8 ms therefore
  include the 600 KB logits copy, ~2.8 ms (my calc), which is about a quarter of
  [token_cost.py](../../../scripts/token_cost.py)'s 11 ms fixed term.
- **Served chat:** with `-bs` on nine cards (five-turn chat, temperature 1.0), S4-L0 generates 31.6–32.2 tok/s
  (31.1–31.6 ms per token). 81 % of its 12,185 generated tokens were reasoning (my calc from the per-turn answer counts).

## Headline findings

1. **Every Flint tool request runs on the host sampling path, on the fork and on master alike, and nothing upstream changes
   that.**
   - **Why it is off:** `common_sampler_init` turns backend sampling off for the whole request as soon as a grammar sampler
     or a reasoning-budget sampler exists (`common/sampling.cpp:422-432`). A tool request with `tool_choice` "auto" brings
     both: the lazy tool-call grammar, and a reasoning-budget sampler that, with no budget set, never forces anything. That
     sampler exists only so that a `<tool_call>` inside the reasoning does not trigger the grammar.
   - **The lazy grammar's state does not matter:** before its trigger it does nothing at all
     (`src/llama-grammar.cpp:1362`), but the switch is decided once, at sampler creation.
   - **Upstream:** no GPU-side grammar, token-mask upload or partial fallback exists. The only related open PR is #25919,
     which lets the reasoning budget coexist with `-bs`.
2. **The cost is the copy, not the sampler.** Per sampled position (est.):
   - 2.82 ms for 600,576 bytes of logits at ~213 MB/s (my calc);
   - 0.4–0.7 ms of host work: 0.09–0.12 ms measured on the desktop core (filling the 150,144-entry candidate array, then a
     top-20 partial sort), × 4–6;
   - together 3.2–3.5 ms, close to the measured pre-fork `-bs` gain of 3.3 ms per token. On a served S4-L0 token that is
     9–10 % of the speed (est.).
3. **Under speculation the loss multiplies.** A verify step copies one 600 KB row per position: 11.3 ms at 3 drafted tokens
   and 45 ms at 15 (my calc). The planned copy-gated lookup drafting (Q53) targets file rewrites and edits, which in agent
   use are tool-call arguments, so it needs a fix for this first.
4. **The grammar is cheap while it accepts the sampled token, and a rejection is expensive.**
   - Accepted token: the single-token check plus the accept take ~6 µs on the desktop core.
   - Rejection: the server resamples after masking all 150,144 tokens with llama.cpp's GBNF engine, which takes 12.6–15.0
     ms on average (max 32–129 ms) on the desktop core and est. 50–150 ms on the Celeron, i.e. two to five tokens' worth.
   - How often Flint's sampled token is rejected is the main unknown.
5. **Recommended design: sample on the GPU, check the sampled token against the grammar on the host, and fetch the full row
   only after a rejection.**
   - llama-server already samples first and checks second, so this keeps its semantics exactly.
   - Per token it adds ~0.02–0.05 ms of host work (est.); a rejection costs what it costs today plus a 2.8 ms row fetch.
   - It subsumes the "lazy-grammar-aware" variant and PR #25919. Effort: 2–3 days in the fork. Expected: tool requests at
     the plain `-bs` speed, +9–11 % (est.).
   - Built in the fork (Q67, §5), with the screened head kept until the grammar can reject a token.
6. **Token-mask decoding with llguidance is fast enough on the Celeron, but it is not the first step.**
   - Speed: for the nine-tool grammar and Flint's vocabulary a mask takes 25 µs on average (p99 0.95 ms) on the desktop
     core, est. 0.1–0.15 ms on the Celeron, plus ~0.1 ms to upload 18.8 KB.
   - What llama.cpp's llguidance path lacks: lazy triggers, tool-call grammars (only GBNF reaches it; the mechanical
     GBNF-to-Lark conversion fails on Flint's special tokens), a GPU mask op, and a Rust toolchain in the rig's Docker
     builder.
   - Its "mask first" sampling also differs slightly from llama-server's "sample, then check".
   - Use llguidance as the rejection fallback of design (b) if rejections turn out frequent: that keeps the semantics
     exact.
7. **vLLM, SGLang and TensorRT-LLM all mask first and hide the CPU mask work behind the forward pass.**
   - vLLM splits the forward from the sampling call; SGLang builds the next mask while the GPU runs; TensorRT-LLM runs the
     mask work as a CUDA host function inside its CUDA graph.
   - At ~27–31 ms per Flint token against ~0.1 ms per mask, overlap would save under 1 % on the rig.
   - Their transferable lesson is the one in design (a): no mask while nothing is constrained (reasoning, text before the
     tool call).

## 1. When backend sampling is off

### 1.1 The switch

llama-server creates one `common_sampler` per request. Its creation decides the path for the whole request (fork base and
master, P):

| Condition | Effect | Where |
|---|---|---|
| Any grammar: tool calls (lazy or not), `grammar`, `json_schema`, `response_format` | Off for the whole request, with "backend sampling is not compatible with grammar, disabling" in the log | `common/sampling.cpp:422-426`; the assert at `:629` |
| A reasoning-budget sampler exists | Off, with "... not compatible with reasoning budget ..." in the log | `common/sampling.cpp:428-432` |
| ... which is created when the template has thinking tags and the grammar is lazy, a budget ≥ 0 is set, or `reasoning_control` is on | Every Flint tool request (thinking tags `<think>`/`</think>`, lazy grammar) | `common/sampling.cpp:314-327`; `tools/server/server-common.cpp:1512-1527` |
| `logprobs` / `n_probs` > 0 without `post_sampling_probs` | Off | `tools/server/server-context.cpp:2021-2026` |
| TTS tasks | Off | `tools/server/server-context.cpp:2029` |
| `-sm tensor` | Off for the context | `src/llama-context.cpp:1417` |
| DRY, XTC, typical-p, top-n-sigma, mirostat, adaptive-p, infill enabled | Not off, but the GPU runs only the chain up to the first of them; the rest runs on the host with whatever the prefix left. DRY sits before top-k in the default order, so it brings the full row back | `src/llama-sampler.cpp:733-770`; default order `common/common.h:264-274` |
| Per-request `"backend_sampling": false` | Off (useful for A/B tests) | `tools/server/server-schema.cpp:187` |

How the `-bs` default propagates: `common_sampler_init` takes the request's sampling parameters by reference and clears
`backend_sampling` on them. The server reads that field right after (`server-context.cpp:2013-2036`) and unbinds the slot's
GPU sampler with `llama_set_sampler(ctx, slot, nullptr)`. Logit bias and penalties have GPU implementations; Flint's
defaults (temperature, top-p, top-k) and the server's min-p run entirely on the GPU.

### 1.2 What a tool request installs, and what the lazy grammar does before and after its trigger

**Tool requests and their grammars** (P):
- **Which grammar a request gets:** with tools and `tool_choice` "auto" the template's autoparser builds a lazy grammar that
  triggers on the tool-section marker (`common/chat-auto-parser-generator.cpp:66-92`); `apply_sampling` installs it
  (`common/chat.cpp:1744-1747`). `tool_choice` "required" and `response_format` build a non-lazy grammar that applies from
  the first token. `tool_choice` "none" renders the tools into the prompt but builds no grammar.
- **Flint's trigger is a single token:** a trigger word that tokenizes to one token becomes a token trigger
  (`common/sampling.cpp:1065-1091`), and `<tool_call>` is id 150010.
  - Flint's tool syntax: `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`.
  - Ids 150010–150015 are `<tool_call>`, `</tool_call>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`
    (Flint's `tokenizer_config.json`).
- **The grammar for the nine-tool set** (as generated by the fork, 16 KB of GBNF):
  - required arguments in a fixed order, then optional ones in any order;
  - string values as "anything until `</arg_value>`" automata of 12 rules each;
  - typed values (numbers, booleans, arrays, the todo objects with their enums) as JSON grammars.
  - With `parallel_tool_calls` off it allows exactly one call, then only end-of-generation.

**Before the trigger** (P):
- `apply` returns at once (`src/llama-grammar.cpp:1362-1364`).
- `accept` appends the token's text to a buffer and compares the token with the trigger tokens
  (`src/llama-grammar.cpp:1407-1450`). The `std::regex` search over the whole buffer (`:382-413`) runs only for word and
  pattern triggers; Flint has neither.
- Inside the reasoning the grammar is not fed at all (`grammar_should_apply`, `common/sampling.cpp:461-473`, and the accept
  path at `:476-505`; PR #20970).

**After the trigger:** llama-server calls `common_sampler_sample` with `grammar_first = false`
(`server-context.cpp:4743`, and per draft position via `common_sampler_sample_and_accept_n` at `:4821`). The sequence
(`common/sampling.cpp:606-688`) is:
1. reasoning budget, then the sampler chain, on the full row;
2. a grammar check of the one sampled token (`:658-668`);
3. only if that token is rejected: refill the full row, apply the reasoning budget, apply the grammar to all 150,144
   candidates, and run the chain again (`:671-687`).

So the output distribution is "sample; if invalid, resample from the masked distribution". The lazy grammar's state never
reaches the backend-sampling decision, which was made at creation.

### 1.3 Upstream state

**Master is unchanged:** on `36e36e6` (2026-10-23) these files are byte-identical to the fork base:
- `common/sampling.cpp`, `common/reasoning-budget.cpp`, `common/llguidance.cpp`, `common/chat-auto-parser-generator.cpp`;
- `src/llama-sampler.cpp`, `src/llama-grammar.cpp`, `include/llama.h`.

**History of backend sampling** (P):
- **[#17004](https://github.com/ggml-org/llama.cpp/pull/17004)** (merged 2026-01-04) added backend sampling and listed "Not
  compatible with grammar sampling" among its limitations.
  - It kept a hybrid mode (a GPU prefix of the chain, then host samplers on the filtered rows) so that grammars could run on
    the host after GPU filtering.
  - No grammar path was ever built on it.
- **Follow-ups:**
  - [#18547](https://github.com/ggml-org/llama.cpp/pull/18547): new scheduler reserve when the graph topology changes;
  - [#18550](https://github.com/ggml-org/llama.cpp/pull/18550): `ggml_build_forward_select`, so inactive sampler branches
    are skipped;
  - [#25532](https://github.com/ggml-org/llama.cpp/pull/25532) (merged 2026-08-10): several outputs per sequence, so `-bs`
    works in speculative verification; it also made the random draw deterministic between the CPU and GPU paths;
  - [#24655](https://github.com/ggml-org/llama.cpp/pull/24655) and [#26958](https://github.com/ggml-org/llama.cpp/pull/26958):
    `-bs` for EAGLE-3, DFlash and DSpark drafters;
  - [#29797](https://github.com/ggml-org/llama.cpp/pull/29797): greedy selection for temperature-0 chains;
  - [#30223](https://github.com/ggml-org/llama.cpp/pull/30223): a static sampling graph across ubatches.

**The reasoning-budget coupling** (P):
- [#20970](https://github.com/ggml-org/llama.cpp/pull/20970) (2026-03-27) always creates the budget sampler next to a lazy
  grammar, so that a `<tool_call>` inside the reasoning does not trigger the grammar.
- [#21209](https://github.com/ggml-org/llama.cpp/pull/21209) (2026-03-31) turned `-bs` off whenever that sampler exists.
- [#21870](https://github.com/ggml-org/llama.cpp/pull/21870) (2026-04-14) skips the sampler when no budget is set, except
  next to a lazy grammar. It followed a reported 98 → 70 tok/s regression on Vulkan
  ([#21784](https://github.com/ggml-org/llama.cpp/issues/21784)).
- [#25919](https://github.com/ggml-org/llama.cpp/pull/25919) (open since 2026-07-20, 45 lines) keeps `-bs` with a reasoning
  budget: in the FORCING state the forced token replaces the GPU's choice. It reports ~5 %, with no change in pass@1. This
  is exact: on the host path the forcing step also leaves only the forced token.

**Other related PRs and issues:**
- **Host overhead per token** (open): [#30196](https://github.com/ggml-org/llama.cpp/pull/30196) reads all outputs with
  one synchronize; [#29796](https://github.com/ggml-org/llama.cpp/pull/29796) skips redundant stream synchronizes;
  [#29576](https://github.com/ggml-org/llama.cpp/pull/29576) shrinks the host output buffers;
  [#30918](https://github.com/ggml-org/llama.cpp/pull/30918) fuses the GPU penalties.
  - Issue [#27050](https://github.com/ggml-org/llama.cpp/issues/27050) measured `-bs` at +48 % throughput with 32 slots on
    an RTX 5090, single slot unchanged.
- **GBNF engine speed:** [#4218](https://github.com/ggml-org/llama.cpp/issues/4218) (open since 2023);
  [#26885](https://github.com/ggml-org/llama.cpp/pull/26885) (merged, 1.2–1.3×).
  - Open: [#28371](https://github.com/ggml-org/llama.cpp/pull/28371) (memoized candidate rejection),
    [#29557](https://github.com/ggml-org/llama.cpp/pull/29557), [#30117](https://github.com/ggml-org/llama.cpp/pull/30117).
  - Open: [#30172](https://github.com/ggml-org/llama.cpp/pull/30172) (grammar clone 8.4 → 0.40 ms for 107 tools; the
    server clones the sampler, grammar included, on every verify step, `server-context.cpp:4801`).

**llguidance in llama.cpp** (P):
- **Build:** `LLAMA_LLGUIDANCE=ON`, which needs Rust.
- **Scope:** only grammars that start with `%llguidance`, plus JSON schemas (`common/json-schema-to-grammar.cpp:994-1000`).
  Tool-call grammars come from the PEG parser as GBNF (`common/peg-parser.cpp:1606`) and never reach it.
- **Limits:** the llguidance sampler (`common/llguidance.cpp`) has no lazy triggers and no GPU implementation.

## 2. What a tool request costs per token

### 2.1 Where the logits are and how they move

- **Card:** in a layer split the output head is placed like a layer after the last one
  (`src/llama-model.cpp:1699`), so it sits on the last card: GPU5 on six cards, GPU8 on nine. Both are behind the second
  ASM1184e switch. When the logits are copied, no other transfer is running, so the copy gets the single-GPU rate of
  ~213 MB/s ([mining_rig.md](../../mining_rig.md)).
- **Copy:** the host buffer is pinned memory of the output device (`src/llama-context.cpp:2321-2329`).
  - The copy is queued after the graph (`:2065-2077`) whenever a sequence has no GPU sampler (`needs_raw_logits`,
    `:1820-1835`).
  - `common_sampler_sample` waits for it in `llama_synchronize` (`common/sampling.cpp:607`).
- **Sizes:**
  - Flint: 150,144 × 4 = 600,576 bytes, 2.82 ms (my calc).
  - Qwen3.6: 248,320 × 4 bytes = 993 KB, 4.66 ms, which is why `-bs` gained +18 % there with MTP
    ([speed-ups](../../benchmarks/2026-10-22-speedups.md)).
- **With `-bs`:** the GPU chain ends with the sampled id plus the ≤ 20 surviving ids, logits and probabilities. That is under
  0.5 KB in four small asynchronous copies (`src/llama-context.cpp:2169-2177`). The chain is built per output row
  (`src/llama-graph.cpp:4204-4298`) from a padded copy of the logits (`:4232`), so `t_logits` itself stays intact in GPU
  memory until the next decode.

### 2.2 Host work, measured on a desktop core

| Per sampled position | Desktop core (5 runs) | Rig (est.) |
|---|---|---|
| Fill the 150,144-entry candidate array from the logits | 0.055–0.071 ms | 0.25–0.4 ms |
| Chain on the full row (top-k 20 partial sort, top-p, min-p, temperature, draw) | 0.037–0.043 ms | 0.15–0.25 ms |
| Grammar check of the sampled token (after the trigger) | 0.0033–0.0039 ms | 0.015–0.025 ms |
| Accept (grammar, reasoning budget, chain) | 0.0002 ms before the trigger, 0.002–0.004 ms after | ≤ 0.025 ms |
| **Sum on the host path** | **0.092–0.117 ms** | **0.4–0.7 ms** |
| Rejection: grammar over the full row, then the chain | 12.6–15.0 ms on average over 197 tool-call positions, max 32–129 ms; 14.8–17.4 ms where most tokens are allowed, 7.4–8.9 ms where few are | 50–150 ms |
| Clone of the grammar sampler (once per verify step) | 0.025 ms (nine tools) | 0.1–0.15 ms |
| Chat re-parse of the streamed output (HTTP thread) | ≤ 0.024 ms per token at 760 tokens | ≤ 0.15 ms |
| llguidance mask, same nine tools (Python wheel 1.9.1, same core) | mean 0.025 ms, p50 0.011, p90 0.018, p99 0.94, max 1.0; consume 0.8 µs | mean 0.1–0.15 ms, p99 4–6 ms |

**Constrained positions:** in 56 of the 197 measured tool-call positions (28 %) fewer than 20 tokens of the whole
vocabulary were allowed; at the fewest, 2. These are the structural positions: tool name, argument keys, JSON
punctuation, enum values.

### 2.3 Per-token budget on the rig (my calc, S4-L0)

| Item | Tool request today (host path) | `-bs` (chat today; tool requests with design b) |
|---|---|---|
| Forward pass, six cards | ~24.0 ms (26.8 ms llama-bench minus its 2.8 ms copy) | same |
| Logits to the host, 600,576 B at ~213 MB/s | 2.82 ms | — |
| GPU sampler chain (top-k 20 over 150,144, softmax, cumulative sum, draw) | — | 0.1–0.5 ms (claim to test; the pre-fork gain implies 0–0.2 ms) |
| Copy of the sampled id and survivors | — | ~0.05 ms (est.) |
| Host sampling (§2.2) | 0.4–0.7 ms | 0.02–0.05 ms (grammar check and accept only) |
| Synchronize and server loop | same in both | same |
| **Sampling cost per token** | **3.2–3.5 ms** + p_rej × (50–150 ms) | **0.15–0.6 ms** + p_rej × (53–153 ms) |

**Check against measurement:** before the fork, Flint (makers' Q4_1, nine cards) served 44.5 ms per token without `-bs`
and 41.2 ms with it, a difference of 3.3 ms ([speed-ups](../../benchmarks/2026-10-22-speedups.md)).

**On a served S4-L0 token:** about 31.4 ms with `-bs` on nine cards, so a tool request runs at ~34–35 ms (est.), i.e.
9–10 % fewer tokens per second.

**Under speculation:** a verify step copies every position's row, then samples positions until the first mismatch:

| Drafted tokens k | 1 | 3 | 7 | 15 |
|---|---|---|---|---|
| Rows copied | 2 | 4 | 8 | 16 |
| Logits copied | 1.2 MB | 2.4 MB | 4.8 MB | 9.6 MB |
| Copy time on the host path (my calc) | 5.6 ms | 11.3 ms | 22.5 ms | 45.1 ms |

Each accepted position adds a further 0.4–0.7 ms of host sampling. With `-bs` the same step costs k + 1 small GPU chains.
For the 7–15-token blocks of Q53, which pay only on copied text (file contents, edits: tool-call arguments), the host path
adds 22–45 ms to a step that reads several token-times of experts ([HyperQwen lessons](hyperqwen_lessons.md)).

### 2.4 What to measure

- The A/B on the rig of experiment 1: the same tool request on the GPU path, on the host path without a grammar, and on the
  host path with the lazy grammar (expected difference 3.0–3.3 ms per token).
- The GPU chain's time on a P106 (`GGML_CUDA_PROFILE`), and the gap between tokens on the host timeline
  (`GGML_CUDA_PROFILE_HOST`).
- The host costs of §2.2 on the Celeron itself (experiment 4), which replaces every "(est.)" above.
- The rejection rate in real agent traffic, and the share of a tool request's tokens generated before `<tool_call>`
  (experiment 2).

## 3. Designs that keep tool requests on the GPU path

### 3.1 (a) Lazy-grammar awareness: keep `-bs` until the trigger fires

**Mechanism:**
- Leave the GPU sampler bound while the grammar awaits its trigger, while it is not applied (inside the reasoning), and
  while the reasoning budget is not forcing. In the forcing state, take the forced token, as in #25919.
- When the trigger token is accepted, unbind the slot's GPU sampler (`llama_set_sampler(ctx, slot, nullptr)`), so the next
  decode copies logits again.
- In a verify step, stop accepting at the trigger position and re-decode the rest.

**Exactness:** identical to today.
- Before the trigger the grammar is a no-op (`apply` returns, the single-token check passes), so the token comes from the
  chain alone on either path.
- The trigger token itself is sampled unconstrained on both paths.
- The GPU and host chains compute the same distribution (llama.cpp treats `-bs` as the same semantics; #25532 made the
  random draw deterministic between the two paths).

**Code:**
- `include/llama.h` and `src/llama-sampler.cpp:2679-2700`: a query for "grammar awaiting trigger".
- `common/sampling.cpp:422-432` (drop both disables) and `:621-641` (backend branch: forcing override, otherwise return the
  GPU token while the grammar is inactive).
- `tools/server/server-context.cpp`: after `common_sampler_accept` (`:4747`) and in the verify path (`:4810-4825`), unbind
  when the grammar becomes active.
- Cost: unbinding changes the graph topology once per request, so one graph rebuild and CUDA-graph capture at the trigger.

**Effort:** ~1 day plus tests (`tests/test-sampling.cpp`, `tools/server/tests/unit/test_tool_call.py` with `-bs`).

**Expected:** 3.0–3.3 ms saved per token generated before the trigger (est.), nothing after it.
- If agent turns look like the chat run (81 % reasoning), that recovers most of the loss (claim to test).
- It does nothing for long tool arguments: `write` contents, `edit` strings.

### 3.2 (b) GPU sampling, host grammar check, full row only on a rejection (recommended)

**Mechanism:** keep `-bs` for every request with a grammar.
1. The GPU chain samples the token as it does for chat.
2. The host runs the same single-token grammar check the host path runs today.
3. If the token passes (always before the trigger and inside the reasoning, almost always inside a string argument),
   accept it.
4. If it fails, fetch that position's full logits row from GPU memory (2.82 ms) and run the existing resample code: budget,
   grammar over the full row, chain.
5. In the forcing state, take the forced token (#25919).

**Exactness:** identical to llama-server's semantics, which already are "sample from the chain; if the grammar rejects the
token, resample from the masked row".
- Step 1 draws from the same chain distribution as the host path.
- Steps 2–4 are the same code on the same numbers.
- Speculative verification keeps working position by position, because each verify row is sampled on the GPU and checked
  on the host in order.
- Not covered: the rejection-sampling verifier for probabilistic drafters (`common/sampling.cpp:758`, masked candidate
  arrays). It keeps the host path for grammar requests; Flint has no such drafter.

**Code:**
- **Context:** `src/llama-context.cpp` gets an explicit row fetch, e.g. `llama_get_logits_ith_fetch(ctx, i)`. It runs
  `ggml_backend_tensor_get` on the last result's `t_logits` row for an output that had a GPU sampler. Details:
  - `t_logits` is a graph output and is not touched by the sampler subgraph (it works on a padded copy), so the row stays
    valid until the next decode;
  - the batch index maps to the device row through `output_ids`, before `output_reorder`;
  - a decode that spanned several ubatches has only the last ubatch's rows left on the GPU, so it keeps the raw copy.
  - Today `get_logits_ith` (`:1064-1082`) would return the stale host row for a backend-sampled output.
- **Sampler:** `common/sampling.cpp:422-432` drops both disables. The backend branch (`:621-641`) gets the forcing override,
  then the check of `:658-668`, then, on rejection, `set_logits` from the fetched row and the resample block of `:671-687`.
  No new grammar API is needed: before the trigger the check passes by itself.
- **Server:** no change. It derives `use_backend_sampling` from the field that `common_sampler_init` no longer clears
  (`server-context.cpp:2013-2036`).
- **Unchanged:** `logprobs` requests stay on the host path.

**Effort:** 2–3 days including (a)'s tests and #25919's `common_sampler_get_candidates` guard.

**Expected:** every token of a tool request at `-bs` cost: 3.0–3.3 ms per token saved (est.), +9–11 % tok/s on S4-L0; under
speculation, (k + 1) × 2.82 ms per verify step. A rejection costs 2.8 ms more than today.
- Graph topology also stays constant between chat and tool requests, which removes today's rebuild when a client
  alternates between them.

**Making the rejection path cheaper (only if experiment 2 finds rejections frequent):**
- **llguidance computes the fallback mask** (§3.3): ~0.1–0.15 ms instead of 50–150 ms (est.). Applying a mask, then the
  chain, is exactly what the resample step does, so semantics stay exact.
- **Top-k-aware scan:** when the chain starts with top-k (llama-server's default order with penalties and DRY off), the
  resample needs only the 20 highest-logit allowed tokens. Check tokens in descending logit order until 20 pass; this is
  exact.
  - On the synthetic rows it found them within 256 checks at 134 of 197 positions.
  - At constrained positions it degenerates to a full scan, which is slower than the batched pass (6.0–6.7 ms for 1,338 checks
    on average where most tokens are allowed; 26–32 ms averaged over all positions on the desktop core).
  - Worth it only together with real logits, which concentrate on plausible, mostly allowed tokens (claim to test).
- **Upstream GBNF speed-ups** (#28371, #30172) shorten the pass and the per-step clone (cherry-pick candidates).

### 3.3 (c) Token-bitmask decoding (XGrammar / llguidance style)

**Mechanism:**
1. The host computes the allowed-token mask for the next position: 150,144 bits = 18,768 bytes, a 4,693-word buffer in
   llguidance.
2. It uploads the mask (~0.1 ms at ~190 MB/s).
3. A GPU sampler sets disallowed logits to −∞ before the chain, and only the token id comes back.

**Exactness:** this is "mask first", the semantics of llama.cpp's `grammar_first = true` and of vLLM, SGLang and TRT-LLM.
- It matches today's distribution at every step where the tokens surviving top-k, top-p and min-p are all allowed.
- It differs where the chain's survivors include a disallowed token. Today such a step resamples with probability
  P(disallowed); mask first renormalizes over the masked top-20.
- Both are valid grammar-constrained sampling, but tokens and acceptance numbers would not match today's runs.

**Costs on the rig (est.):**
- llguidance mask: 0.1–0.15 ms mean, p99 4–6 ms. Computed after each accept and before the next decode, serially; overlap
  is not needed at this size.
- Upload 0.1 ms, mask kernel ~0.02 ms.
- About 0.25–0.3 ms per constrained token, with no rejection tail.
- llama.cpp's own GBNF engine is unusable for full masks here: 12.6–15.0 ms per mask on the desktop core, est. 50–150 ms on
  the Celeron.

**What llama.cpp would need** (P):
- **A Lark emitter for the PEG grammar:** next to the GBNF one in `common/peg-parser.cpp:1606`. Special-token literals must
  become `<[id]>`, because llguidance matches special tokens only that way, and the "until" parsers must become lazy
  lexemes.
  - Measured: `gbnf_to_lark` on the fork's GBNF fails at the first `<tool_call>` token.
  - A Lark grammar generated in that shape for the same nine tools works and gives the timings above.
- **Lazy triggers:** create the matcher at the trigger and feed it the trigger token. `common/llguidance.cpp` has no
  triggers.
- **A GPU mask sampler** in the backend-sampler interface (`backend_init`/`backend_apply`/`backend_set_input`,
  `src/llama-sampler.cpp`). Two options:
  - upload the smaller of the allowed and disallowed id lists and scatter −∞ with existing `set_rows`/`get_rows` ops (the
    adaptive storage XGrammar uses);
  - a new bit-test op in ggml-cuda for the dense mask.
- **Rust in the rig's Docker builder** (`LLAMA_LLGUIDANCE=ON`).

**Effort:** 1–2 weeks.

**Expected:** 2.8–3.1 ms saved per token after the trigger (est.), the same as (b) minus the mask cost, and no
50–150 ms rejections. Before the trigger it needs (a) anyway.

### 3.4 (d) What vLLM, SGLang and TensorRT-LLM do

All three mask first, compute masks on the CPU, apply them on the GPU, and hide the CPU work behind the forward pass (P
unless marked S):
- **XGrammar** ([arXiv 2411.15100](https://arxiv.org/abs/2411.15100); default backend of vLLM and SGLang):
  - Masks: under 40 µs per token for JSON schemas and JSON CFGs on a Ryzen 9 7950X with Llama-3.1's 128k vocabulary, from
    an adaptive cache of context-independent tokens (>99 % precomputed).
  - The engine starts mask generation as soon as the previous token is accepted and synchronizes only before sampling.
    MLC-LLM's time per output token went 6.2 → 6.3 ms with masks at batch 1.
  - `fill_next_token_bitmask` returns `need_apply = False` when every token is allowed, so the apply is skipped
    ([integration docs](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html)).
- **vLLM V1:**
  - The scheduler's `StructuredOutputManager` fills the batch bitmask (xgrammar, llguidance or outlines); the GPU runner
    applies it to the logits before sampling.
  - With async scheduling, forward and sampling are separate calls, and the mask is computed between them
    ([#26866](https://github.com/vllm-project/vllm/pull/26866), merged 2025-11-01).
  - [#29821](https://github.com/vllm-project/vllm/pull/29821) (merged 2026-01-06) extends this to speculative decoding:
    draft tokens are copied to the CPU asynchronously, masks are built for every verify position while the next forward
    runs, and draft tokens the grammar rejects are replaced by −1 and their positions masked.
- **SGLang:**
  - In overlap mode the next batch's vocab mask is computed after the previous batch's results are processed, while the
    GPU runs (`model_runner.py`, `_preprocess_logits`).
  - `ReasonerGrammarBackend` defers all constraints until the think-end token sequence: no mask during reasoning, the same
    idea as llama.cpp's lazy grammar plus the budget sampler.
- **TensorRT-LLM:** guided decoding with xgrammar or llguidance. For one-model speculative decoding, the grammar advance and
  mask fill run as a `cudaLaunchHostFunc` callback on a dedicated stream inside the CUDA graph, ordered by events, with masks
  for every draft position and rollback after verification
  ([tech blog 12](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html)).

**What transfers to the rig:**
- The no-mask-while-unconstrained rule: (a), and in (b) for free.
- Rollback across draft positions: (b) gets it by checking positions in order.
- If (c) is ever built, the smaller-of-allowed/rejected id list.

Overlap does not pay here: the mask costs ~0.1 ms against a 27–31 ms token.

### 3.5 Comparison

| Design | Semantics vs llama-server today | Effort | Saved per token before the trigger | Saved per token after it | Rejection | Under speculation |
|---|---|---|---|---|---|---|
| (a) `-bs` until the trigger | Identical | ~1 day | 3.0–3.3 ms (est.) | 0 | As today | Host path once the call starts |
| **(b) GPU sample, host check, row on rejection** | **Identical** | **2–3 days** | **3.0–3.3 ms** | **3.0–3.3 ms** | **As today + 2.8 ms** | **(k + 1) × 2.82 ms saved per step** |
| (b) + llguidance fallback mask | Identical | + Lark emitter, Rust build (~1 week) | same | same | ~3 ms instead of 50–150 ms | same |
| (c) llguidance masks on the GPU | Mask first: differs where the chain's survivors include a disallowed token | 1–2 weeks | needs (a) | 2.8–3.1 ms | None | Masks per draft position; invalid drafts cut |

## 4. Experiments, ranked by value for effort

| # | Experiment | Where, effort | Expected | Decides |
|---|---|---|---|---|
| 1 | Tool-request A/B on the fork build: one `-bs` server, the same request as `tool_choice` "none" (GPU path, same prompt), "none" with `"backend_sampling": false` (host path, no grammar), and "auto" (host path with the lazy grammar); temperature 0, then 1.0 with fixed seeds | rig, 1 h | (ii) − (i) ≈ 3.0–3.3 ms per token; (iii) − (ii) ≈ 0 at temperature 0 | Whether the ~10 % is real on the fork (Q55); go/no-go for (b) |
| 2 | Rejection rate and pre-trigger share: count the resample branch (`common/sampling.cpp:671`, a five-line patch printed with the slot's timings) and the tokens before `<tool_call>`, over real agent sessions or BFCL tasks sent to `/v1/chat/completions` with `tools`, at temperature 1.0, top-p 0.95, top-k 20 | quant machine, 2–3 h (`bench/bfcl_flint` renders prompts itself, so it needs a variant that sends `tools`) | Rejections well under 1 % of tokens after the trigger; most tokens before it (claim to test) | Whether (b)'s host fallback is enough (≲ 1 %) or needs llguidance (≳ 2 %); how much (a) alone would give |
| 3 | Implement (b) in the fork (with #25919 and (a)'s tests), then repeat experiment 1 (built: §5; rig commands in §5.5) | fork, 2–3 days; rig 1 h | Variant (iii) within 1–2 % of (i): +9–11 % on tool requests | The new default for agent serving |
| 4 | Host-only benchmark on the Celeron: the §2.2 measurements (fill, chain, check, accept, full-row pass, clone) built CPU-only for x86-64-v2 in the rig's Docker image (glibc ≤ 2.38) | rig, 15 min, no GPU | Desktop × 4–6; the full-row pass × 4–10 | Replaces every "(est.)" in §2; the cost of a rejection |
| 5 | GPU cost of the sampler chain on a P106: `GGML_CUDA_PROFILE` on the `-bs` variant of experiment 1, top-k/argsort/softmax/cumsum/get_rows groups on the last card | rig, 15 min | 0.1–0.5 ms per token | Whether a fused top-k kernel is worth anything |
| 6 | Speculation with tools after Q52/Q53: `ngram-simple` drafts with fixed synthetic acceptance (`--spec-synth-len`) on a tool request whose `write` argument copies a file from the prompt; host path against (b) | rig, 1 h | (k + 1) × 2.8 ms per step saved | The verify-cost budget of Q53 on agent traffic |
| 7 | llguidance as (b)'s fallback mask, or (c) | fork, 1–2 weeks | Removes 50–150 ms rejections | Only if experiment 2 finds frequent rejections |

**Experiment 1 commands** (rig, fork build, S4-L0 on nine cards with the chat runs' settings):

```bash
BIN=/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin
LLAMA_NO_MOE_DOWN_RESCALE=1 $BIN/llama-server -m /mnt/models/flint-S4L0.gguf -ngl 99 -fa on \
    -c 262144 -b 512 -ub 128 -np 1 -bs --cache-ram 0 --ctx-checkpoints 4 \
    --no-webui --host 127.0.0.1 --port 8089 > /tmp/tools_ab.log 2>&1 &
until curl -sf 127.0.0.1:8089/health > /dev/null; do sleep 2; done
python3 - <<'EOF'
import json, urllib.request
def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": req}}}
TOOLS = [tool("bash", "Run a shell command", {"command": "string", "description": "string", "timeout": "number"}, ["command", "description"]),
         tool("read", "Read a file", {"filePath": "string", "offset": "number", "limit": "number"}, ["filePath"]),
         tool("write", "Write a file", {"filePath": "string", "content": "string"}, ["filePath", "content"]),
         tool("edit", "Replace text in a file", {"filePath": "string", "oldString": "string", "newString": "string"},
              ["filePath", "oldString", "newString"])]
MSGS = [{"role": "system", "content": "You are a coding agent. Use the tools."},
        {"role": "user", "content": "Create src/stats.py with a RunningStats class (Welford's algorithm: push, extend, "
                                    "variance, stdev, merge) and a short demo under __main__. Write it with the write tool."}]
for temp, seed in [(0.0, 1), (1.0, 11), (1.0, 12)]:
    for tc, bs in [("none", True), ("none", False), ("auto", True)]:
        body = {"messages": MSGS, "tools": TOOLS, "tool_choice": tc, "backend_sampling": bs, "temperature": temp,
                "top_p": 0.95, "top_k": 20, "seed": seed, "max_tokens": 8192, "cache_prompt": False}
        req = urllib.request.Request("http://127.0.0.1:8089/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(req, timeout=3600))["timings"]
        print(f"T {temp} seed {seed} tool_choice {tc:4s} backend_sampling {bs!s:5s}: "
              f"{t['predicted_n']} tok at {t['predicted_per_second']:.2f} t/s")
EOF
grep -c "not compatible with grammar" /tmp/tools_ab.log   # one per "auto" request
kill %1; rm -f /tmp/tools_ab.log
```

- At temperature 0 all three variants should generate the same tokens (same `predicted_n`) unless the grammar rejects one.
- Repeat on six cards (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`) to line up with llama-bench's 26.8 ms.
- For experiment 5, start the server with `GGML_CUDA_PROFILE=/tmp/prof.csv` and summarize with
  `scripts/cuda_profile_summary.py /tmp/prof.csv --tokens 24 --by op`. `GGML_CUDA_PROFILE_HOST=/tmp/host.csv` gives the
  host timeline: on the host path, the gap between tokens should show the 2.8 ms copy plus the host sampling.

## 5. Built: GPU sampling with a host grammar check (Q67)

Design (b) of §3.2 is in the fork, on branch `pascal-p106-v2-toolsample` (off `pascal-p106-v2`):
- `e2d587398` is the change and `972ccde20` a warning fix; the rig tarball is built from `972ccde20`;
- `f11ac1589` adds a switch for tests only.

The combination with the screened head (Q92) is on `pascal-p106-v2-toolsample-screen` (§5.3).

### 5.1 What changed

- **The switch-off is gone.** `common_sampler_init` keeps backend sampling for a sampler with a grammar or a
  reasoning budget. llama-server binds the request's chain to the context as for chat, because the request's
  `backend_sampling` field now survives sampler creation; the server's binding code is unchanged.
  `LLAMA_GRAMMAR_HOST_SAMPLING=1` restores the old switch-off, and a per-request `"backend_sampling": false` still
  selects the host path.
- **`common_sampler_sample`** takes the backend sampler's token in place of the host path's first draw:
  1. if the reasoning budget is forcing a token, the forced token replaces it (on the host path it is the only
     candidate; this is PR #25919's rule);
  2. while the grammar cannot reject anything (a lazy grammar before its trigger, or inside the reasoning), the token
     is returned as it is;
  3. otherwise the host checks that one token against the grammar, the same single-token check the host path makes;
  4. on a rejection the host fetches that position's full logits row from the device and runs the host path's
     resample: the reasoning budget, the grammar over all 150,144 tokens, then every sampler of the chain on the host.
- **`llama_get_logits_ith_fetch(ctx, i)`** returns an output's full logits row, also for an output a backend sampler
  sampled. The context records which output rows stayed on the device (every output of a ubatch whose sequences all
  have backend samplers) and copies one row (600 KB for Flint) when asked. The logits tensor is a graph output, which
  the sampler subgraph reads through a padded copy, so the device keeps it until the next decode. Rows of an earlier
  ubatch of the same decode are overwritten by later ubatches; for them the call returns NULL. Generation steps and
  verify steps fit one ubatch.
- **Counters (Q66).** Each sampler counts its tokens: drawn on the device, sampled while the grammar could reject them,
  rejected (and of those, first drawn on the device), forced by the reasoning budget. llama-server prints one line per
  request with a grammar after its timings and returns `sampled_on_device`, `grammar_n` and `grammar_rejected_n` in the
  response's `timings`.
- **Smaller pieces:** `llama_sampler_grammar_awaiting_trigger`, `common_sampler_grammar_active` and
  `common_reasoning_budget_get_forced`.
- **Test switch:** `LLAMA_GRAMMAR_TEST_REJECT=N` counts every Nth token the host checks as rejected, on both paths, so
  that the resample path runs at a known rate (§5.4).

### 5.2 Why the output is llama-server's

llama-server samples from the chain, checks the sampled token against the grammar, and resamples from the masked full
row if the grammar rejects it (§1.2). Each step of the new path matches one of these:

- **The first draw.** The backend `dist` sampler takes its uniform from the same Mersenne-twister stream as the host
  `dist` sampler, one draw per sampled token (`llama_sampler_dist_backend_set_input`, `src/llama-sampler.cpp`), and
  the backend chain runs the same top-k, top-p, min-p and temperature. The backend token is therefore the host path's
  first draw, up to the rounding of the GPU's softmax and cumulative sum against the host's double-precision sum. A
  draw within that rounding of a cumulative boundary can pick the neighbouring token, with or without a grammar. The
  no-grammar baseline in §5.4 measures how often.
- **The check** is the same call on the same token.
- **The resample** uses the full row, with the grammar and chain applied on the host exactly as on the host path. It
  takes the next draw of the stream, as the host path's resample does. In single-output decoding the backend sampler
  took its draw from the stream before the decode, so the next draw is the same one the host path would use, and the
  output is identical.
- **Several outputs per sequence** (a speculative verify step) use transactional draws: the backend sampler draws from
  a copy of the stream, and a draw is taken from the stream itself only when its token is accepted. Without care the
  resample would reuse the rejected token's uniform. Conditioned on the rejection, that uniform lies inside the
  intervals of disallowed tokens, so the resample would be biased. The new path therefore takes that draw from the
  stream before resampling. The verify step then stops at that row, because the later rows were drawn before the
  resample took its draw; they are sampled again from fresh draws in the next step. Each token still has its exact
  distribution, but after such a rejection the random stream runs one draw ahead of the host path's.
- **The reasoning budget's forced token** is the only candidate on the host path, so taking it is the same.
- **Callers that mask first** (`grammar_first`, not llama-server) get the full row and the host path's masked sampling.
  This costs one extra draw from the stream, and the semantics are unchanged.
- **Not changed:** the rejection-sampling verifier for probabilistic drafters
  (`common_sampler_sample_and_accept_n_rejection`) runs the same code, reading the backend chain's candidate
  probabilities as it already does for requests without a grammar.

### 5.3 With the screened head (Q92)

The screen computes exact logits for its top 256 candidates and gives every other token −∞. Q91 showed that the true
top 20 are among the candidates on every row it checked, so top-k ≤ 20 sampling is exact.

- **Accepted tokens.** A token the grammar accepts is the chain's own draw, exact under the same condition as for chat.
- **A rejection is different.** The resample needs the 20 best allowed tokens by their exact logits. A rejection
  happens exactly when the model's own top tokens are disallowed. The allowed ones (a tool name, an argument key,
  closing punctuation) can then rank anywhere, also below the 256th candidate, where the screened row holds −∞.
  Q91's evidence covers the unmasked top 20 only.
- **The fetched screened row is therefore not exact after a rejection.** Nothing bounds how far below the candidates
  an allowed token's true logit can sit. The certified variant would need one: the gap between the exact and the
  screen logit of token j is at most ‖w_j − ŵ_j‖·‖h‖ (Cauchy–Schwarz on the head row and its GPTQ copy). With that
  bound per token, a resample whose 20th allowed logit clears the 256th screen logit by the largest bound is exact,
  and any other resample would redo the position with the full head. Not built.
- **The rule that is built** (commit `45d243e80` on `pascal-p106-v2-toolsample-screen`) keeps the screen while the
  grammar cannot reject a token: before the trigger, and inside the reasoning. The token is then the chain's draw, so
  the screen is as exact as for chat. llama-server decides per batch, so the batch after `<tool_call>` is computed
  with the full head, and a rejection there resamples from exact logits. A verify step stops at the row where the
  grammar becomes active, so that the later rows are recomputed with the full head.
- **Cost.** The screen applies to the tokens before the call (reasoning and text: §2.4's pre-trigger share, measured
  in §5.4), and the full head computes the call itself. Switching costs one graph rebuild per tool call.

### 5.4 Checks

The checks ran on the CPU of the 7-GPU quant machine (its GPUs were busy with other work). A CPU-only build runs the
backend sampler on the CPU backend, which is the GPU path in every step that matters here: the chain runs in the
sampling graph, the logits stay in the backend's buffer, and a rejection fetches the row from there.

Setup ([grammar_sampling_check.py](../../../bench/tool_sampling/grammar_sampling_check.py)):
- llama-server with `-bs` and Flint UD-Q4_K_XL;
- one BFCL task per category (simple, multiple, parallel, parallel_multiple, live_simple, live_multiple, irrelevance,
  live_irrelevance), sent with its tools and `tool_choice` "auto", so that Flint's lazy grammar applies;
- the makers' sampling, at most 1,536 tokens per request;
- each request sent twice to the same server with the same seed: backend sampling on ("device") and off ("host",
  llama-server's host path).

| Grammar | Sampling | Requests per side | Tokens per side | Tokens while the grammar applied | Rejections | Result |
|---|---|---|---|---|---|---|
| tool grammar | T 0 | 8 | 1,575 | 301 | 2 | identical |
| tool grammar | T 1, seed 1 | 8 | 1,350 | 300 | 2 | identical |
| none (`tool_choice` "none") | T 0, T 1 | 8, 8 | 1,575, 1,350 | — | — | identical |
| tool grammar, every third checked token rejected (`LLAMA_GRAMMAR_TEST_REJECT=3`) | T 0 | 8 | 1,575 | 301 | 186 | identical |
| the same | T 1, seed 1 | 8 | 1,300 | 300 | 151 | identical |

- **What "identical" covers:** the reasoning, the content and the parsed tool calls.
- **Calls:** 7 of the 8 grammar requests per run made a call; the irrelevance task answered without one.
- **Rejections:** every rejection resampled from the fetched row to the token the host path picked. That covers the 4
  natural ones and the 337 forced ones. The test switch counts every token the host checks outside the reasoning, so
  it also forces resamples before the trigger.
- **The old switch-off:** llama-server logged no "not compatible with grammar" warning.
- **granite-3.1-3b-a800m** (40 tasks, five per category, T 0 and T 1) was identical as well. Its template gets no tool
  grammar in this build (it writes `<tool_call>[…]` as text), so it confirms only the chain without a grammar.

**Q66 from the same runs** (8 tasks, so only a first look):
- 20.5 % of the tokens were sampled while the grammar could reject them: 13.9 % on simple and 53 % on
  parallel_multiple, none on the irrelevance task.
- 4 of those 601 tokens were rejected (0.67 %). At that rate the host fallback costs 0.0067 × (50–150 ms + 2.8 ms),
  about 0.35–1.0 ms per grammar token, or 0.07–0.2 ms per token of the request.

**On CUDA** (one V100 of the 7-GPU machine, the fork's v2 release candidate with this change; Flint UD-Q4_K_XL, one
BFCL task per category at T 0 and T 1 with seed 1):
- Backend sampling on against off: 8 of 8 tasks identical at each temperature with the tool grammar, and 8 of 8 without
  it. All 2,620 tokens were sampled on the device, 542 of them while the grammar applied, with 5 rejections resampled
  from the fetched row and no switch-off.
- `LLAMA_GRAMMAR_HOST_SAMPLING=1` gave the host path's output for 8 of 8 tasks at each temperature, and logged its
  switch-off warning once per grammar request.
- `LLAMA_GRAMMAR_TEST_REJECT=3`: 385 forced rejections, each resampled from the fetched row; device against host
  identical for 8 of 8 tasks at each temperature.

**Not yet checked on a GPU:** Q66 over the 300 BFCL pilot tasks, and the screened-head combination (screen on against
off).

### 5.5 On the rig

Experiment 3 of §4: the A/B of experiment 1 on the new build, plus the host path with the grammar as a fourth variant.
Build the tarball with `scripts/build_llamacpp_docker.sh fork/pascal-p106-v2-toolsample` (for the screened head,
`fork/pascal-p106-v2-toolsample-screen`) and unpack it in `/opt/llama-builds` on the rig.

```bash
B=/opt/llama-builds/llama.cpp-20261024-972ccde20-pascal-p106-v2-toolsample/bin
# nine cards, then six (GPUS=0,1,2,3,4,5) to line up with llama-bench
ssh rig "BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf GPUS=0,1,2,3,4,5,6,7,8 TAG=q67-9 \
    LLAMA_NO_MOE_DOWN_RESCALE=tg bash -s" < bench/tool_sampling/tool_request_ab.sh
# the old switch-off on the same build, for a same-binary reference
ssh rig "BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q67-9-hostenv LLAMA_GRAMMAR_HOST_SAMPLING=1 \
    LLAMA_NO_MOE_DOWN_RESCALE=tg bash -s" < bench/tool_sampling/tool_request_ab.sh
# the screened head with a grammar, on the -screen build, with Q92's screen side file next to the models
B=/opt/llama-builds/llama.cpp-20261024-06a17d382-pascal-p106-v2-toolsample-screen/bin
ssh rig "BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q67-screen-9 LLAMA_SCREEN_HEAD=1 \
    LLAMA_SCREEN_HEAD_FILE=<dir>/flint-output-screen-gptq-q4_0.gguf LLAMA_NO_MOE_DOWN_RESCALE=tg bash -s" \
    < bench/tool_sampling/tool_request_ab.sh
ssh rig 'grep -o "screened output head [a-z]*" /tmp/tool_request_ab/q67-screen-9/server.log | uniq -c'
```

Read `summary.txt`, per temperature and seed:
- **"auto" with backend sampling** should run within 1–2 % of "none" with backend sampling, and **3.0–3.3 ms per token
  faster than "auto" without it** (+9–11 %). That decides Q67 for agent serving.
- The grammar counters on the "auto" lines: every token drawn on the GPU, the grammar tokens (the `write` call's
  arguments), and the rejections, each of which costs a row fetch plus the full-row grammar pass (§2.2).
- The warning count must be 0 on the first run and one per "auto" request with `LLAMA_GRAMMAR_HOST_SAMPLING=1`.
- At temperature 0 all four variants should generate the same tokens, unless the grammar rejected one.
- With the screen, the server log shows the screen "on" for the reasoning and "off" from the batch after
  `<tool_call>`.

## Sources

The fork's tree at upstream `d77256999` (P; line numbers are the fork's; master `36e36e6` is identical in the sampling
files):
- `common/sampling.cpp`: `common_sampler_init` (191–445; disables at 422–432; budget sampler at 314–327; llguidance branch
  at 217–222), `grammar_should_apply` (461), accept (476), `common_sampler_sample` (606–688), the trigger conversion
  (1065–1091).
- `common/chat-auto-parser-generator.cpp:66-92`, `common/chat.cpp:1744-1747`, `common/peg-parser.cpp:1606`,
  `common/llguidance.cpp`, `common/json-schema-to-grammar.cpp:994-1000`, `common/reasoning-budget.cpp:166-186`.
- `src/llama-grammar.cpp` (trigger regex 382–413, apply 1359–1400, accept 1402–1458, clone 1329–1357), `src/llama-sampler.cpp`
  (chain backend prefix 733–770, top-k 1469–1508, dist 1263–1342, grammar 2679–2700).
- `src/llama-context.cpp` (needs_raw_logits 1820–1835, logits copy 2065–2077, sampled outputs 2169–2177, output buffers
  2243–2392), `src/llama-graph.cpp:4204-4298`, `src/llama-model.cpp:1675-1699`.
- `tools/server/server-context.cpp` (2013–2036, 4743, 4801, 4821), `tools/server/server-common.cpp:1512-1527`,
  `tools/server/server-schema.cpp:187`, `tools/llama-bench/llama-bench.cpp:2198-2222`.

llama.cpp PRs and issues (P): #17004, #18547, #18550, #20970, #21209, #21784, #21870, #24655, #25532, #25919, #26885,
#26958, #27050, #28371, #29557, #29576, #29796, #29797, #30117, #30172, #30196, #30223, #30918, #4218 (all at
`https://github.com/ggml-org/llama.cpp/pull/<n>` or `/issues/<n>`); `docs/llguidance.md` (llguidance: 50 µs average, p99
0.5 ms per mask for a 128k tokenizer on JSONSchemaBench).

Other engines:
- XGrammar: [arXiv 2411.15100](https://arxiv.org/abs/2411.15100) (P),
  [engine integration docs](https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html) (P).
- vLLM: [#26866](https://github.com/vllm-project/vllm/pull/26866), [#29821](https://github.com/vllm-project/vllm/pull/29821),
  [structured output API docs](https://docs.vllm.ai/en/v0.22.0/api/vllm/v1/structured_output/) (P).
- SGLang: `python/sglang/srt/model_executor/model_runner.py` (`_preprocess_logits`),
  `python/sglang/srt/constrained/reasoner_grammar_backend.py` at <https://github.com/sgl-project/sglang> (P).
- TensorRT-LLM: [tech blog 12, guided and speculative decoding](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html) (P).
- llguidance: [repository](https://github.com/guidance-ai/llguidance), Python wheel 1.9.1 used for the mask timings (P).

Flint: `chat_template.jinja`, `tokenizer_config.json` and `generation_config.json` of
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (P).

This repository:
- [mining_rig.md](../../mining_rig.md): link rates.
- [speed-ups](../../benchmarks/2026-10-22-speedups.md): the pre-fork `-bs` gain.
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md).
- [HyperQwen lessons](hyperqwen_lessons.md) §4: the earlier estimate.
- [flint_speedups.md](flint_speedups.md): template and tool format.
- [quants/experiments.md](../../../quants/experiments.md): Q38, Q53, Q55, Q66, Q67, Q91, Q92.
- [bench/tool_sampling](../../../bench/tool_sampling/): `tool_request_ab.sh` (experiments 1 and 3) and
  `grammar_sampling_check.py` (§5.4, Q66).
