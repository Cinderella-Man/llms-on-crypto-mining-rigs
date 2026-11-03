# Q583: what verifying k tokens costs Qwen3.8-Flash-Next on nine P106 cards (2026-11-03)

Speculative decoding pays only if checking k drafted tokens in one pass of the main model costs much less than k
one-token passes. Q558's MTP draft on Swift-1.5 Q2_0 gained only ~1.11–1.15× ([Q558's data](../2026-11-03_flash_next_mtp/README.md)):
a draft step (draft passes plus a 4-token pass of the main model) took ~347 ms against ~127 ms per token without the
draft. This experiment measures a k-token pass for k = 1–5 on Miner Llama v3's base, profiles where its time goes
(`GGML_CUDA_PROFILE`, as in [Q215](../2026-11-03_flash_next_baseline/README.md)), and profiles one drafting request
with Q558's line. The registry row is Q583 in [quants/experiments.md](../../../quants/experiments.md).

## Result: each extra verified token costs 0.4× a token, and the dense small-batch products decide whether drafts pay

| part of the model (GPU ms per pass, summed over the nine cards) | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | 4-token verify in the drafting request | draft passes per step |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed experts (Q2_0) | 27.2 | 29.9 | 44.3 | 58.3 | 70.7 | 45.1 | 2.9 |
| shared expert | 4.4 | 5.7 | 7.3 | 9.6 | 9.2 | 7.7 | 0.3 |
| BF16 hyper-connections | 44.6 | 63.3 | 83.4 | 101.8 | 123.0 | 77.2 | 2.9 |
| router (F32) | 2.5 | 3.4 | 4.5 | 3.2 | 6.0 | 2.7 | 0.2 |
| attention and recurrent projections (Q3_K, IQ4_XS, Q4_K) | 24.5 | 38.1 | 50.9 | 69.2 | 67.9 | 52.4 | 1.7 |
| recurrent state (delta net, convolution) | 4.3 | 4.8 | 5.3 | 5.8 | 6.2 | 9.8 | 0.0 |
| flash attention | 0.7 | 0.9 | 1.2 | 1.2 | 1.7 | 0.9 | 0.2 |
| output layer (Q5_K main, Q6_K draft; 248,320 tokens) | 3.5 | 3.5 | 3.6 | 3.6 | 3.6 | 10.4 | 18.5 |
| other operations without weights | 26.3 | 30.5 | 30.9 | 31.1 | 30.8 | 32.1 | 2.8 |
| other weights | 1.6 | 2.0 | 2.7 | 3.2 | 3.9 | 3.0 | 1.3 |
| **GPU total** | **139.6** | **182.0** | **234.1** | **286.9** | **323.0** | **241.5** | **30.9** |
| wall time per pass, profiler on | 178.7 | 220.2 | 279.8 | 339.2 | 381.9 | | |
| **wall time per pass, profiler off** | **136.0** | **185.8** | **244.6** | **305.4** | **359.9** | | |

The k columns are llama-bench passes of k tokens (median of five after a warm-up; random token ids, an empty
context, logits for the last token only). The last two columns are medians over the 101 drafting steps of one request
(312 tokens, 0.693 acceptance, 3.08 tokens per step) with the MTP draft: the main model's pass over the token and the
three drafts, with logits for all four, and the draft's four passes per step.

- **A k-token pass costs 136 + 50–59 ms per extra token** (wall, profiler off): two tokens 186 ms, four 305 ms, five
  360 ms. An extra verified token costs ~0.41× a one-token pass.
- **The drafting request's 4-token verify costs 241.5 ms of GPU time, 1.73× a one-token pass** (139.6), and the four
  draft passes 30.9 ms (medians 4.5, 8.8, 8.8 and 8.8 ms), 18.5 ms of which is the draft's own output layer over the
  full 248,320-token vocabulary. A step's wall time (profiler on) has a median of 327 ms: 272 ms on the cards and
  ~55 ms between the passes. Three steps of the 101 took ~2.7 s and one 1.1 s with normal GPU time, with no swap-out
  and no memory pressure on the host (Q590); without those four, steps average 349 ms, Q558's ~347.
- **Where a 4-token verify's extra 102 ms goes** (against a one-token pass): BF16 hyper-connections +32.6 ms,
  attention and recurrent projections +27.9, routed experts +17.9 (four tokens pick more distinct experts), the
  output layer +6.9 (logits for four rows), operations without weights +5.8, recurrent state +5.5 (the state is kept
  per drafted token for rollback), shared expert +3.3. The dense products, which read the same weights for every
  token, are 59 % of it: their small-batch kernels on Pascal read or compute nearly per column.
- **llama-bench's 4-token pass costs 45 ms more GPU time than the verify** (286.9 against 241.5): routed experts
  +13 ms (random ids spread over more experts than consecutive tokens of one answer), and the dense parts 20–30 % more
  for the same weights and the same four columns (Q589). Every pass was flat across its repetitions (k = 4:
  286.8–287.2 ms; the verify 237.6–241.5 ms over the first steps), so the gap is systematic. The verify column is the
  one a drafter pays; the k columns give the slope.

### What it means for drafts on this rig

A draft gains `tokens per step × time per token / time per step`. On v3's base: 3.08 × 127 / 347 ≈ 1.13×, which is
what Q558 measured. Taking the step apart with the table:

- **The BF16 hyper-connections are where a draft gains most today:** 77.2 ms per step for 3.08 tokens (25 ms per
  token) against 44.6 ms per token without the draft. Everything else in a step (270 ms for 3.08 tokens, 87.6 ms per
  token) already costs more than everything else in a one-token pass (82.4 ms).
- **So the BF16 kernel (Q546) and the retype (Q218) make drafts worth less, not more, unless they also flatten the
  k-curve.** With Q546's estimate of 11–14 ms per token for the BF16 tensors and 4 columns costing about 1 (est. 14 ms
  at k = 4), a token takes ~95 ms and an MTP step ~284 ms: est. 1.03× (10.9 against 10.5 tok/s). Both lines get
  faster; the draft's lead shrinks.
- **What makes drafts pay is the k-curve of the rest:** the Q3_K and IQ4_XS projections (×2.1 at four tokens), the
  routed experts (×1.66), the draft's own 248,320-token output layer (18.5 ms per step; a token shortlist for the draft's
  head, Q592 with Q400's method, removes most of it) and the verify's output layer (×3). Kernels for 2–8 columns that read each weight
  once (Q591) are the lever for every drafter, MTP, DFlash2 and pipelined speculation alike.
- **DFlash2 (Q561):** one draft pass of 5 layers through the target's shared Q5_K head for three positions (est.
  15–25 ms) instead of the MTP draft's 30.9 ms, and the same 241.5 ms verify: a step est. ~337 ms. At an acceptance
  length of 2.4 tokens per step that is est. 0.90×, at 2.83 1.07×, at 3.26 1.23×; break-even at ~2.65 tokens per
  step. The research note's 1.3–1.5× assumed a 200–230 ms verify and a negligible step overhead.

## Files

| File | What it holds |
|---|---|
| [bench_by_k.csv](bench_by_k.csv) | per k: GPU ms per pass by part (median of the repetitions), wall ms per pass with and without the profiler |
| [draft_request.csv](draft_request.csv) | per step of the drafting request: the main pass's and the draft passes' GPU ms by part, the step's wall time (host clock, profiler on); steps 0–1 are the prompt's two chunks |
| [passes.csv](passes.csv) | every pass of both traces: the cards it ran on and its GPU ms by part |
| [tables.md](tables.md) | `verify_split.py`'s printed output: the pass counts it found, the table above, the step times and the reply's timings |
| [bench-prof.json](bench-prof.json), [bench-wall.json](bench-wall.json) | llama-bench's JSON output, profiler on and off (every repetition's time) |
| [reply-draft.json](reply-draft.json) | the drafting request's reply, with its timings (`draft_n`, `draft_n_accepted`) |
| [run/verify_cost.sh](run/verify_cost.sh) | the run on the rig: three loads, each after a logged quiet check and under the host-memory guard |
| [run/verify_split.py](run/verify_split.py) | the tables from the traces (standard library only; the parts are regexes on the launch groups' labels) |

The traces themselves (18 MB and 4 MB of CSV) are not published; `passes.csv` keeps every pass.

## Commands

On the rig, with the desktop stopped and nothing else running; `mem_guard.sh` and the request are Q558's
([run/](../2026-11-03_flash_next_mtp/run/)), copied next to `verify_cost.sh`:

```
ssh $RIG "BIN=<v3 build>/bin MODEL=<Swift Q2_0 first shard> DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> OUT=<work dir>/out \
  setsid nohup bash <work dir>/run/verify_cost.sh > <work dir>/verify_cost.out 2>&1 < /dev/null &"
```

The three loads, in order (231 s, 228 s and 217 s to load; ~13 minutes in all):

```
GGML_CUDA_PROFILE=$OUT/prof-draft.csv llama-server -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 \
  --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui -b 2048 -ub 512 \
  -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 \
  --spec-draft-n-ubatch 128 --no-host -m $MODEL                  # then one request: Q558's smoke_request.json
GGML_CUDA_PROFILE=$OUT/prof-bench.csv llama-bench -m $MODEL -ngl 99 -fa on -b 2048 -ub 512 -ts 6/6/6/6/5/6/5/6/3 -lm dio \
  --no-host 1 -p 1,2,3,4,5 -n 0 -r 5 -o json
llama-bench <the same, without the profiler>
```

Then, on another machine: `python3 run/verify_split.py <work dir>/out` (it reads `<work dir>/verify_cost.out` for the
line where the request starts in the trace).

## Provenance and licence

On 2026-11-03, on the second of the owner's identical rigs (nine P106-090 cards, Celeron 3865U, one 4 GB SO-DIMM),
headless (`multi-user.target`), right after Q558 on the same rig. The quiet checks before each load (no llama
process, every card at 0 MiB, load below 2) are in the run's console. Miner Llama v3's base `37e717ff6`. The model:
Swift-1.5 GSQ-RCO Q2_0 from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`; the draft: `mtp-Qwen3.8-Flash-Next-draft.gguf` from
[agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)
`273edc1d2ffae5241fe3cab69568d682dad4c718`, as in Q558. The host-memory guard never fired (at most 385 MiB of swap-out
in 10 s, while the server loaded; pressure below 10 %). The profiler records CUDA events between launch groups and
synchronizes after every evaluation, which adds 3–4 ms of GPU time and ~40 ms of wall time to a one-token pass.

The tables hold timings and token counts. The reply is the model's text for a prompt written for Q558; the model's
weights are under the Swift Open License 1.0 and qwen-community-1.0, the draft's under qwen-community-1.0.
