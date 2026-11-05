# SpecPilot and Qwen3.8-Flash-Next on the P106 rig (2026-11-04)

[SpecPilot](https://github.com/StargazerX0/SpecPilot) (Li et al., [arXiv 2611.02501](https://arxiv.org/abs/2611.02501),
Apache-2.0, 2026-11-02) is an agentic loop that searched for lossless speed-ups of speculative decoding, one deployment
at a time. Its repository ships the six SGLang engines the loop found: Qwen3.8-27B and Muse-Glimmer-30B with the DFlash2
and DSpark drafters on one H100 or one RTX PRO 6000. This note reads it for Qwen3.8-Flash-Next on nine P106-090 cards:
what each mechanism needs, which of them the rig's cost structure lets pay, and what to take. The repository (one
commit, `78c42af`) and the paper's HTML version were read on 2026-11-04; nothing was built or run. Numbers marked
**est.** are arithmetic from the linked figures.

## Summary

1. **SpecPilot's three single-request mechanisms do not transfer to the rig.** Wide best-first trees (32 nodes), W12
   (a lossless 12-bit BF16 format) and two-level verification (TLV) take its H100 engine from stock DFlash2's 3.95× over
   plain decoding to 8.86×, because there extra verify rows are nearly free: the target's GEMMs took 17.88 ms for one
   row and 18.58 ms for 64. On nine P106s a 5-token pass of Swift Q2_0 costs 2.63× a one-token pass (82.4 → 217.1 ms).
   That is the regime of the paper's batching runs, where the same mechanisms lost: 24-node trees −14.9 %, a
   24-position block −9.9 %, TLV −13 % (its §6.5).
2. **The rig already does what the paper's batching results point to: verify rows only where they pay.** The per-step
   draft length ([Q687](../../data/2026-11-04_adaptive_draft/README.md), `adapt5`) reached 1.52–1.53× over no draft
   against a fixed draft of 2's 1.30–1.39×. It is the one-request form of SpecPilot's per-request drafter block and
   shared node budget.
3. **Four things are worth taking, none of them a kernel:**
   - token forcing, a statistical test that every committed token is the target's own choice, adapted to the agent
     check's sampled decoding (Q1014, §4.1);
   - the cost model's three ceiling factors, recomputed after every round of rig work (Q1015, §4.2);
   - the promotion rule from paired runs, here on two rigs at once with a swap (Q1016, §4.3);
   - the drafter cut to 4 bits: every single-request final ran DFlash2's body in INT4, and the rig's DFlash2 GGUF is
     Q8_0, 652 MB against 362 MB at Q4_K_M (Q1017, §4.4).
4. **Its tree verify for Gated DeltaNet is a reference implementation for Q686** (§4.5), the recorded-input recurrent
   state: each leaf's root path is walked with the state in registers, per-node commit inputs replace per-node state
   snapshots, and a commit kernel replays the accepted path, bitwise equal to the stock kernel on chains. It is Triton
   for sm_90 and sm_120; a CUDA port for compute 6.1 is the fork's work.
5. **Three ideas are parked with their arithmetic** (§5): TLV (Q1018: the cheapest coarse view of a 2–3-bit MoE file
   still costs est. ≥ 0.78 of an exact pass), the certified exact-argmax head (Q1019: a screen tight enough to exclude
   rows needs 8 bits, more bytes than Flash-Next's Q5_K head), and drafting past the drafter's trained block (Q1020).
6. **The code was read, not run, and holds nothing malicious** (§6). It cannot run on the rig in any case: it needs
   torch 2.13 with CUDA 13.0, which has no compute 6.1 target, and its kernels use Hopper and Blackwell instructions.

## 1. What SpecPilot is

The loop runs in rounds. A cost model writes the speed-up over plain decoding as tokens per step over step time and
computes a ceiling from three factors:
- the **tokens factor**: how many more tokens per step the best tree on the recorded drafter lattices would accept, in
  exact replay at the best node budget;
- the **kernel factor**: the profiled step time over the sum of every kernel class's byte floor, the larger of its FLOP
  time at 60 % of peak and its byte time at 90 % of peak bandwidth;
- the **overhead factor**: the measured step time over the profiled one.

A master agent sends one direction per GPU at the largest factor. Sub-agents implement each direction as patches behind
switches whose defaults reproduce stock. Controller scripts that the agents cannot change time every patch against the
current best in paired runs on two GPUs, swapping the GPUs between runs, and promote it only if the gain's 95 % lower
bound exceeds 1 % plus twice the stock-against-stock noise (3 % if its outputs diverge from the current best's). Each
round ends with a sealed final on unseen requests, checked by token forcing (§4.1). Its mechanisms become entries of a
structured memory (140 entries, each a code delta with its correctness contract and evidence) that later deployments
start from. The agents were Claude Opus 5.5 in Claude Code. A single agent without the cost model reached 1.34× over
stock in three rounds, against SpecPilot's 2.13×: it kept adding to the most visible gap (tokens per step) and never
touched the weight reads that took 80 % of the step.

The repository holds the loop's results, not the loop: a pruned SGLang 0.5.20 (`python/`), an experiment plugin, and,
per setting, the 27–39 new modules and 46 `SPECPILOT_*` switches of its final engine (`settings/<setting>/files`,
`config.json`). The workload is the first turns of GSM8K, HumanEval, MATH500, MBPP and MT-Bench, greedy, thinking on,
2,048 new tokens; there is no multi-turn or agentic workload.

| Setting | Stock / plain | SpecPilot / plain | Tokens per exact pass |
|---|---:|---:|---|
| Qwen3.8-27B + DFlash2, H100, 1 request | 3.95× | 8.86× | 4.65 → 13.08 |
| Qwen3.8-27B + DSpark, H100, 1 request | 3.74× | 9.41× | 4.42 → 12.85 |
| Qwen3.8-27B + DFlash2, RTX PRO 6000, 1 request | 4.46× | 10.87× | 5.23 → 12.80 |
| Muse-Glimmer-30B + DFlash2, RTX PRO 6000, 1 request | 4.69× | 10.63× | 5.16 → 12.64 |
| Qwen3.8-27B FP8 + DFlash2, H100, 16–24 requests | 2.02× | 3.90× | 4.56 → 5.39 |
| Qwen3.8-27B FP8 + DSpark, H100, 16–24 requests | 1.94× | 3.93× | 4.37 → 5.42 |

Qwen3.8-27B is a dense hybrid (48 Gated DeltaNet and 16 attention layers): the layer family of Flash-Next's 36 + 12,
without the routed experts.

## 2. The rig is in the per-row-cost regime

SpecPilot's single-request gains rest on one fact: with one request a step reads every target weight once, so verifying
32 or 64 rows instead of 8 costs little. Its batching runs show the other case: once every verified row costs time,
mechanisms that verify more per exact pass lose.

On the rig every verified token costs a large share of a pass ([C266](../../data/2026-11-04_kpass_profile/README.md):
Swift Q2_0 on nine cards, ms per pass):

| Part | 1 token | 5 tokens | 5 / 1 |
|---|---:|---:|---:|
| routed experts (Q2_0) | 8.69 | 55.49 | 6.39 |
| attention and recurrent projections | 27.13 | 61.77 | 2.28 |
| hops between cards, host work and gaps | 11.6 | 40.5 | 3.50 |
| hyper-connections | 10.03 | 22.25 | 2.22 |
| Gated DeltaNet state and conv | 3.46 | 5.46 | 1.58 |
| the whole pass | 82.4 | 217.1 | 2.63 |

Each token routes its own 10 of 512 experts (~31 distinct experts per layer for 5 tokens of real agent text), a P106 has
5 SMs and dp4a only, and the residual of every verified token crosses eight card boundaries. SpecPilot's H100 verify
grew 4 % from 1 to 64 rows; the rig's grows 163 % from 1 to 5.

What SpecPilot found for that regime matches the rig's own result: choosing rows per request and per step (a shared
node budget of the batch's most probable nodes, 16, 12 or 8 drafter positions per request) gained, while wide trees and
TLV lost. Flash-Next's per-step draft length is that rule for one request.

## 3. Mechanism by mechanism

| SpecPilot mechanism | What it does there | On the rig |
|---|---|---|
| Best-first tree verify on the DFlash2 lattice | edges scored by the selector's score conditional on the parent, 32 nodes over 16 drafted positions; in exact replay 4.25 tokens per step (stock chain) → 5.38 at 32 nodes on an 8-position lattice → 5.85 on 16 positions; round one took 184 → 270 tok/s | every node is a column of a pass that costs 2.63× at 5 tokens. The [tree replay](../../data/2026-11-04_tree_replay/README.md) (Q688) scores the same best-first trees on the rig's DFlash2 lattices (`--mode markov`); token trees (Q614) wait on cheap columns (Q591) |
| Snapshot-free Gated DeltaNet tree verify with a replay commit | a path kernel walks each leaf's root path from the committed state; verify stores each node's commit inputs (normalised k, v, gate, β: ~33 KiB per node per layer there) instead of its 3 MiB state; the commit replays the accepted path, bitwise equal to the stock kernel on chains; a WY kernel (after TreeWY) solves all paths at once on tensor cores | the design of Q686, which replaces one 113 MiB state plane per drafted token. The recurrence itself costs only +2.0 ms at 5 tokens here, so its value is memory and trees (§4.5) |
| W12, lossless 12-bit BF16 | a 16-binade exponent window per output channel, 12.01–12.02 bits a weight, decoded in registers inside the GEMM; 1.15–1.26× plain BF16 decoding on H100 | measured: x12, the same idea, is slower than plain BF16 on a P106 ([C378](../../data/2026-11-04_lossless_bf16/README.md): 11.04 against 9.42 ms a step at one token), and the hyper-connections are Q8_0 since R28 |
| Two-level verification (TLV) | a Q pass over W12's nested 8-bit view (98.7 % top-1 agreement) checks the tree without committing; its path plus a second draft form one 32-node tree for one exact pass: 13.08 tokens per exact pass, round three 270 → 416 tok/s | parked, Q1018 (§5.1) |
| Certified exact-argmax head | an INT8 copy of the BF16 head screens every row, a per-row Cauchy–Schwarz bound on the INT8 error drops the rows that cannot be the argmax, and the survivors are rescored exactly: 878 → 482 µs on H100, slower than BF16 on the RTX PRO 6000 | parked, Q1019 (§5.2); Q619's shortlist head, which bounds the whole logit, is parked for the same reason |
| The drafter at low precision | in all four single-request finals the drafter's body runs in INT4 (asymmetric groups of 128, MSE-optimal clipping) and its candidate head in INT4 (symmetric groups of 256), on the Qwen settings over 98,304 vocabulary rows | the draft head is already cut to a shortlist (`LLAMA_DRAFT_SHORTLIST`); the body is Q8_0: Q1017 (§4.4) |
| Calibrated tree edges | a constant table of the chance that a candidate is the true token, by slot, sibling rank and drafter-probability bin, blended half and half with the drafter's log-probability: +0.55–0.75 % tokens per step on held-out requests | a small gain; evidence for Q734 and Q760 (the selector's probabilities calibrated) |
| The DSpark lattice | DSpark's Markov head conditions each drafted token on the previous one; a tree that ignores this accepts half the tokens | for Q568 (DSpark for Flash-Next as GGUF): its trees need the conditional edges |
| Drafting past the trained block | DFlash2 trained on blocks of 8 drafts 16 positions, which lifts the replay curve by ~0.5 tokens per step at 32 nodes | parked, Q1020 (§5.3) |
| Per-request drafter block, shared node budget, prompt admission | batching: drafter rows packed under a GEMM tile boundary, the batch's most probable nodes verified, short prompts fed into the verify step | the rig serves one agent; the per-step budget is Q687 |
| Fused glue and accept kernels, the trailing-step skip, PDL, TMA weight pre-staging, L2 prefetch, FP8 and stream-K GEMMs | launch and latency work for sm_90 and sm_120 and for SGLang's overlap scheduler | no counterpart on compute 6.1 or in llama-server's single slot |

## 4. What to take

### 4.1 Token forcing (Q1014)

SpecPilot accepts a final only if a teacher-forced scorer (the target on the stock engine, with an FP32 head) agrees
with its outputs as often as with stock's. A token is a mismatch if it is not the scorer's argmax, and its margin is the
log-probability gap to the argmax. Stock itself mismatches on 0.72 % of tokens, near-ties from kernel rounding. A
system is rejected if its mismatch rate exceeds stock's by more than 0.5 points at 95 % confidence, or if its
unexplained large-margin events (above a near-tie threshold set from stock's own first departures from plain decoding)
are more frequent than stock's. An injected fault (a logit bias of 2 on every 7th token id after 2,048 tokens) moved task
accuracy by −1.6 points, inside the noise; token forcing rejected it at 2.0–2.4 % mismatches.

The rig's lossless checks compare whole greedy replies with plain decoding on short smoke prompts, while the agent check
samples (temperature 1.0, top-p 0.95, top-k 20; [agent check](../../../bench/agent_check/README.md)). The test for the
rig:
- **greedy arms:** SpecPilot's two rules, against a no-draft arm of the same build;
- **sampled arms:** the scorer applies the request's sampler chain (top-k, top-p, min-p, temperature, and the grammar
  where one applied) at every position of every reply, and tests (a) tokens outside the sampler's support, which exact
  sampling never emits, beyond near-ties at the support's edge, and (b) the randomized probability integral transform
  u = F(x⁻) + V·p(x), with V uniform on [0, 1), which is uniform when x is drawn from p: a χ² test of u per arm, against
  the no-draft arm's. Q808's bias (random draws reused across a checkpoint replay, p = 6 × 10⁻⁹ in the CPU check) is the
  kind of fault this finds on real sessions;
- the scorer runs on the quant machine with the same GGUF file. Its kernels round differently from the P106's, and the
  no-draft arm's own mismatch rate calibrates that. It needs each reply's token ids, not its text.

### 4.2 The ceiling, recomputed per round (Q1015)

The rig's notes already hold two of the three factors. The [bandwidth floor](flash_next_floor_and_engine.md) (an
82.4 ms token against a 28.7 ms floor; the 5-token slope at 33.2 ms a token against est. 8–11) is the kernel factor, and
the unprofiled pass over the cards' GPU time (82.4 / 68.7 = 1.20 at 1 token, 217.1 / 174.3 = 1.25 at 5) is the overhead
factor. The tokens factor is the tree replay's best policy at its best node budget over the live tokens per step. One
table of the three factors after each rig round names the largest gap, as SpecPilot's master did, and keeps the work
from staying on the most visible one.

### 4.3 Paired runs on two rigs (Q1016)

SpecPilot times every patch against the current best on two GPUs at once and swaps them between runs. With two identical
rigs, an arm pair runs as A on rig 1 and B on rig 2, then swapped: half the wall-clock time of four arms in series on
one rig, with the rig-to-rig difference cancelled. The promotion rule carries over: the gain's 95 % lower bound above
1 % plus twice the A-against-A noise, measured once per pair of rigs.

### 4.4 A 4-bit DFlash2 drafter (Q1017)

The drafter only proposes, so any approximation of it keeps the output exact; it is judged by tokens per step against
the bytes it saves. All four single-request finals ran DFlash2's body in INT4. The rig's DFlash2 file is Q8_0 (652 MB);
the publisher's Q4_K_M is 362 MB and its Q5_K_M 432 MB, both made from the BF16 original. Est. saving: the 290 MB not
read in each draft pass, at most 1.7–2.4 ms a draft step at the cards' 122–173 GB/s (less where the draft pass is bound
by launches and hops), 290 MB free on the card that holds the drafter, and 290 MB less to load through the 4 GB host.
Measured on the rig ([C528](../../data/2026-11-05_dflash2_bits/README.md)): Q5_K_M keeps Q8_0's tokens per step
(2.597 against 2.600), saves 2.2 ms a draft pass at base clock and 210 MiB on the drafter's card, and takes 0.8 % less
time a token. Q4_K_M saves 3.4 ms and 276 MiB but accepts 1.3 % fewer tokens a step, which spends the saving. The
host's lowest free memory does not move with either.

### 4.5 The reference for Q686

Q686 records each verified token's inputs to the Gated DeltaNet and conv states and replays the accepted path after the
verify, instead of keeping a 113 MiB state plane per drafted token. SpecPilot's files implement that design:
- `settings/h100_c1_qwen38-27b_dflash2/files/python/sglang/srt/speculative/specpilot_tree_gdn.py`: the path verify
  kernel and the commit replay;
- `.../srt/speculative/specpilot_db_conv.py`: the conv commit rebuilt from time-major node inputs, a third of the bytes
  of per-node windows;
- `.../srt/speculative/specpilot_d8_gdn.py`: the WY solve for trees of up to 32 nodes, (I + X)⁻¹ as
  (I − X)(I + X²)(I + X⁴)(I + X⁸) for the strictly lower triangular X, in exact FP32.

The tests and the branch-wise reference that their docstrings cite are not in the repository, so a port to compute 6.1
brings its own check against plain decoding.

## 5. Parked

### 5.1 Two-level verification (Q1018)

TLV pays when a cheap view of the target agrees with it almost always and costs much less than an exact pass. W12's
8-bit view is free: it is the first two thirds of each W12 record. Flash-Next's 2–3-bit files hold no coarser view
inside their blocks, and a second, smaller copy does not fit beside the main file on nine 6 GB cards and a 4 GB host.
The cheapest view of an MoE is fewer experts per token (one pooled set for the pass, Q864, or a top-p cut, Q618). At 5
tokens it removes at most the routed experts' 46.8 ms of growth, so the Q pass costs est. ≥ 170 of an exact pass's
217 ms (≥ 0.78), while the hops, projections and hyper-connections stay. The paper's own break-even: with the Q pass at
0.80–0.88 of a step, a cycle must accept nearly twice the tokens of a one-pass step. Reopens with a view at ≤ ~0.4 of an
exact pass and ≥ ~95 % top-1 agreement (fewer layers or cards), or once k-token passes are nearly flat (Q591), where the
single-request regime applies.

### 5.2 The certified exact-argmax head (Q1019)

The certified head drops a row v when its screened logit plus the bound ‖h‖ · ‖W_v − Ŵ_v‖ cannot reach the top. For rows
of 2,560 near-Gaussian values with one absmax scale per row, a b-bit screen's error norm is est. 2.3 · ‖W_v‖ / (2^b − 1).
With ‖h‖ ≈ 156 and row norms up to 1.41 on Swift Q2_0's head (Q662), the bound is est. ~2 logits at 8 bits, ~8 at 6 bits
and ~34 at 4 bits (~21 with a scale per 32 values), against a ~10-logit margin between the top logit and the rest. Only an
8-bit screen excludes rows, and it reads 636 MB, more than the Q5_K head itself (437 MB, 3.58 ms a token). SpecPilot's
head pays because its exact head is BF16, twice the screen's bytes. The agent check's top-k 20 sampling would also need
a certified top 20, not one argmax. Reopens with a head stored at 8 bits or more.

### 5.3 Drafting past the trained block (Q1020)

SpecPilot drafts 16 positions with a DFlash2 trained on blocks of 8 and verifies trees up to depth 15. The rig's verify
width ends at 6 (`--spec-draft-n-max 5`), and long chains wait on cheap columns (Q617, Q624). Reopens with Q624.

## 6. How it was read, and what the code does

- The repository is one commit (`78c42af`, 2026-11-02), 3,584 files, 8.4 MiB packed: 2,023 Python files, ~400 CUDA
  and C++ files, JSON kernel configurations and six PNG figures; no binaries, no submodules, one symlink inside the
  tree. It was cloned with hooks, symlinks, LFS and submodules off, made read-only, and read with text tools only.
- `scripts/setup_env.sh` installs `uv` through its official installer, the ~200 packages of a pinned lock file from
  PyPI (SGLang's dependencies and projects of NVIDIA, Hugging Face, pydantic and vLLM; `httpx2` and `httpcore2` are
  pydantic's), and on drivers older than R580 NVIDIA's CUDA 13.0 forward-compatibility package.
  `scripts/download_models.sh` takes six checkpoints at pinned revisions from Hugging Face. The servers listen on
  127.0.0.1.
- A search of every file found no obfuscated or encoded payloads, no network calls beyond upstream SGLang's own clients
  and the local server, no reads of credentials or keys, and no persistence. The only hidden characters are zero-width
  spaces in an upstream SGLang comment copied from NVIDIA's documentation.
- `scripts/score.py` runs model-written HumanEval and MBPP code in a subprocess with resource limits and an empty
  environment, which is not a sandbox: run it in a container if at all.

## Sources

- The repository: [StargazerX0/SpecPilot](https://github.com/StargazerX0/SpecPilot), commit `78c42af`: `README.md`,
  `settings/*/README.md` and `config.json`, and the module docstrings under `settings/*/files`.
- The paper: [arXiv 2611.02501](https://arxiv.org/abs/2611.02501), HTML version v1: §3 (the loop, the ceiling, token
  forcing), §4 (the mechanisms), §6.1 (rounds and the single-agent baseline), §6.2 (injected faults), §6.4 (tree replay),
  §6.5 (regimes).
- The drafter files: [incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF)
  (file sizes).
- The rig's figures: [C266's k-pass profile](../../data/2026-11-04_kpass_profile/README.md),
  [the bandwidth floor](flash_next_floor_and_engine.md), [Q687](../../data/2026-11-04_adaptive_draft/README.md),
  [C378](../../data/2026-11-04_lossless_bf16/README.md), [multi-token verify](multitoken_verify_hybrid.md) and the
  registry rows named above ([quants/experiments.md](../../../quants/experiments.md)).
