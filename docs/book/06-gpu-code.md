# 6. Writing my own CUDA for a 2016 GPU

> **In short**
> - Stock llama.cpp is tuned for large modern GPUs. On a P106, with 5 SMs, no tensor cores and slow FP16 math, its
>   kernels left most of the memory bandwidth unused.
> - I rewrote the hot kernels one at a time, each behind an off switch and checked against the old path. With a newer
>   upstream base, they took a Flash-Next token on nine cards from 136.0 ms to 74.9 ms.
> - How a kernel is launched (its block size and memory budget) mattered as much as what it computes: one setting alone
>   saved 7.5 ms a token.
> - The bandwidth floor is 28.7 ms a token. Most of the gap is kernel speed, not launch overhead, so I rewrite kernels
>   before I would rewrite the engine.

The numbers are from 2026-11-04. The model is Qwen3.8 Flash-Next in its 2-bit Q2_0 file, spread over nine cards
([Chapter 5](05-nine-cards.md)), unless a section says otherwise. Kernel timings are diagnostics: the release itself is
judged on whole agent tasks ([Chapter 3](03-measuring.md)).

## A kernel, in backend terms

A GPU program is a **kernel**: one function that thousands of threads run at once, each on its own slice of the data.
The host **launches** it, much like an RPC. Each launch has a fixed cost before any work starts: 1.72 µs on a P106 when
the launches are recorded in advance as a CUDA graph, 6.43 µs when they are not.

Threads come in **blocks**, and each block runs on one **streaming multiprocessor** (SM), the GPU's version of a CPU
core. A block keeps its working data in **registers** (private to each thread) and in **shared memory**, a small
scratchpad on each SM: 96 KB per SM on this card. An SM runs several blocks at once if their registers and shared
memory fit side by side. That number, the **occupancy**, decides how many memory reads are in flight.

Writing a token is mostly **matrix-vector products** ([Chapter 2](02-how-an-llm-makes-text.md)): for each layer, a
large table of weights times one short vector of activations. It is a full table scan. Each weight is read once and
used for one multiply-add, so the scan runs at the speed of memory. A kernel's whole job is to keep enough reads in
flight that the memory bus never waits.

## Why stock kernels underuse a P106

| What the P106 has | What it means for code written for newer cards |
|---|---|
| 5 SMs, 1.5 MB of L2 cache, 173 GB/s measured | Kernels that split a product into many tiny blocks never put enough bytes in flight on 5 SMs |
| No tensor cores | The matrix units that modern kernels use for batches of tokens do not exist; all math is plain FP32 or 8-bit integer |
| FP16 at 1/64 of the FP32 rate | Half-precision math is very slow, and FP16's small range (largest value 65,504) can overflow |
| `dp4a`: four 8-bit multiply-adds in one instruction, at 4× the FP32 rate | The one fast path for quantized weights: unpack to 8-bit integers, multiply with `dp4a` |

Two examples show the size of the waste. Stock llama.cpp turned CUDA graphs off on cards older than Volta, the
generation after Pascal, so each of the thousands of small kernels in a token paid the full launch cost. And for
Flash-Next's BF16 weights, stock code launched 10,240 blocks of 160 threads for one product, each thread reading
4 bytes once. That kept 38,400 bytes in flight across the 5 SMs and read at 19.6 GB/s, about a ninth of what the card can
do.

## The first rewrites: one token at a time

**The idea.** In plain decoding the model handles one token per step. Every product is one weight table times one
vector, so a kernel for exactly that case can be much simpler than llama.cpp's general one.

**What I did.** For the first release (v1) I switched CUDA graphs on for Pascal and gave wide rows a launch shape of 2
warps per block (a warp is 32 threads that move in lockstep). New Pascal matrix-vector kernels read rows in 128-bit
loads staged through shared memory, with several rows sharing each activation load. And I wrote an event profiler,
because NVIDIA's own profilers refuse to run on mining cards.

**What happened.** On six cards with an earlier model, Flint, in Unsloth's UD-Q4_K_XL file, a single-request benchmark
went from 17.56 tokens a second on stock llama.cpp to 19.53 with CUDA graphs, 24.58 with the 2-warp shape for wide
rows and 34.06 with the Pascal kernels: 1.94× from code alone. In the v2 release the Pascal one-token kernels were still
worth +15.5 % (36.8 against 31.9 tok/s), the largest single switch.

**Why.** The weights did not change, only how they were read. Fewer, larger launches and lean decoding mattered more
than the bytes themselves.

## The FP16 sum that overflowed

**The idea.** The quantized kernels first squash the activation vector into blocks of 32 eight-bit integers with one
scale each (llama.cpp's `q8_1` blocks). Each block also stores the sum of its values, because several weight formats
need it. llama.cpp stores that sum in FP16.

**What went wrong.** FP16's largest finite value is 65,504. In Flint, a few channels in the last three layers reach
typical sizes (RMS) of up to 48,400, against ≤ 302 everywhere else. A block's sum then passes the limit, the stored sum
becomes infinite, and the output becomes NaN (not a number). llama.cpp works around this with a **rescale**: it
divides the input of each expert's down projection by a power of two and multiplies the result back. On Flint that is
470 extra launches a token, 2.58 ms on a P106. Without it, Flint's Q4_1 file gave NaN on every test chunk.

**What I did.** My Pascal kernels compute each block's sum themselves, in FP32, from the block's own int8 values. The
values are already in registers for the multiply, so this adds no memory traffic, and the rescale can go.

**What happened.** On the rig, Flint UD-Q4_K_XL on six cards went from 33.99 to 37.26 tokens a second (+9.6 %), with
one-token perplexity identical to the last digit (23.2625 both ways). On 2026-11-04 the same overflow turned up again,
this time in llama.cpp's own matrix-vector kernel: a test with activations of 250,000 in some channels gave infinities
for Q4_0 and Q5_0 at 8 tokens. Those five dot products (Q4_0, Q4_1, Q5_0, Q5_1, IQ1_S) now sum the int8 values with
`dp4a` too. The batch kernels used for prompts still keep FP16 sums.

**Why.** A cache of a derived value (the block sum) was stored in a type too small for it. Recomputing it from data
that is already in registers was free; the workaround was not.

## Kernels for Flash-Next: 2-bit experts and BF16

**The idea.** Flash-Next's file uses two kinds of weights that no Pascal kernel covered: 2-bit Q2_0 experts and BF16
(a 16-bit float format) for its hyper-connections, the products that mix its four parallel residual streams. In the
first profile on the rig the largest of them ran at 29.7 GB/s (the 2-bit gate and up products), 20.0 GB/s (the
2-bit down products) and 19.6 GB/s (the BF16 up projections), a sixth of the card's speed or less.

**Why stock was slow.** The rows are short: 640 or 2,560 weights. For the experts, stock code launched blocks of 128
threads in which only 80, or for the down projection 20, had work, and each down projection launched 6,400 blocks
that read 720 bytes each. The BF16 case was the 10,240-block launch above. The experts took 10.9 picoseconds per weight
where the bandwidth allows 1.63.

**What I did.** Both new kernels size a group of threads to the row, let several rows share each load of the
activations, and read weights in 128-bit loads. The 2-bit kernel turns each 2-bit code into an 8-bit value with a
byte-permute instruction and a 4-byte table, then multiplies with `dp4a`. The BF16 kernel turns BF16 into FP32 with a
16-bit shift, which is exact, and adds in FP32. Both passed llama.cpp's operator tests on the P106 (339 and 265 cases),
and a build with a planted bug failed exactly the cases each kernel takes.

**What happened.**

| Weights in one token | Stock (ms) | New kernel (ms) | Speed reached |
|---|---:|---:|---|
| Experts, Q2_0 (routed and shared) | 25.23 | 6.96 | 132.0 GB/s (gate and up), 81.5 GB/s (down) |
| Hyper-connections and the other BF16 weights, 1 token | 25.46 | 8.93 | 158.9 GB/s (up), 157.2 GB/s (down): 89 % and 88 % of the card's 179 GB/s streaming read |
| Hyper-connections and the other BF16 weights, 4 tokens | 71.07 | 14.17 | |

Together with the multi-token kernels below and a newer upstream base, a whole Flash-Next token went from 136.0 ms to
82.4 ms. One part lost: inside llama.cpp at 2–4 tokens, the new 2-bit kernel was slower than stock (0.81×, 0.71× and
0.65× its speed at 2, 3 and 4 tokens). That pushed the next piece of work.

**Why.** Short rows on few SMs need few, well-fed blocks. The fix was the same for both formats: match the thread
group to the row, and decode in registers with the integer instructions the card is fast at.

## Several tokens in one pass

**The idea.** Speculative decoding ([Chapter 8](08-guessing-ahead.md)) guesses a few tokens ahead and checks them all
in one pass. If each weight is read once and applied to all k tokens, a 5-token pass should cost little more than a
1-token pass. A P106 has the arithmetic for est. 8–16 tokens at memory speed. The target was a 5-token pass at most
1.3× a 1-token pass.

**What I did.** Two kernel families. The **k-column kernels** (one column per token) decode each weight block once
into 8-bit words and multiply it with `dp4a` against every token's activations. The **multi-token expert kernel**
groups the tokens that picked the same expert, reads that expert once and applies it to each of them. Both give
results bit for bit identical to the one-token kernels.

**What happened.** The whole model, ms per pass on nine cards:

| Tokens per pass | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Before (v3 base, 2026-11-03) | 136.0 | 185.8 | 244.6 | 305.4 | 359.9 |
| After (merged build, 2026-11-04) | 82.4 | 122.1 | 149.4 | 173.0 | 217.1 |

Every pass got cheaper, but a 5-token pass still cost 2.63× a 1-token pass, not 1.3×. Inside llama.cpp the new kernels
won clearly in places (2-bit products at 5, 6, 7 and 8 tokens: 2.89×, 3.47×, 2.84× and 4.11× faster) and lost in
others (the 3-bit attention products at one token: 0.77×, 0.78× and 0.75× at 6,144, 10,240 and 12,288 rows). The expert down projection never used the new kernel: its 180-byte rows failed an alignment
check meant for 16-byte units, so it ran token by token. That check is fixed but not yet timed (open).

**Why.** Two reasons, both found by the profiler below. Each token picks its own experts, so their bytes grow with k
whatever the kernel ([Chapter 7](07-experts.md)). And the new kernels' launch shapes were wrong for this card.

## Launch shapes: the 32 KB budget

**The idea.** A kernel's code is only half of its speed. The other half is its **launch shape**: how many rows each
group of threads handles, how much shared memory each block takes, and so how many blocks fit on one SM at once.

**How it should work.** The multi-token kernels stage the activations in shared memory, up to a budget. With the
default budget of 48 KB per block, two blocks fit in an SM's 96 KB. With 32 KB, three fit, and a third block means a
third more memory reads in flight.

**What I did.** The per-kernel benchmark on a P106 showed the default shape losing at almost every Flash-Next shape.
I tried a 32 KB budget on the rig through an environment variable, with no rebuild.

**What happened.**

| Tokens per pass | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Default shape (ms a pass) | 82.4 | 122.1 | 149.4 | 173.0 | 217.1 |
| 32 KB budget (ms a pass) | 74.9 | 107.7 | 138.5 | 162.8 | 205.3 |

One plain token went from 82.4 to 74.9 ms, 9.1 % faster, from one launch setting. The benchmark also showed where the
budget does not help: kernels for the grid-based IQ formats use 171–221 registers per thread, so only two blocks fit
per SM whatever the shared memory, and there the smaller tile is 5–19 % slower. So the new default rule asks the
compiled kernel for its register count and picks the 32 KB tile only where a third block really fits. That build is
checked for correctness but not yet timed on the rig: open as of 2026-11-04.

**Why.** On a 5-SM card, occupancy is the lever. The best shape depends on registers, row width and token count
together, so the rule reads them from the compiler instead of hard-coding one answer.

## Attention: the array that lived in memory

**The idea.** v2 had its own Pascal kernel for attention while decoding, nicknamed FAP. It shipped switched off, because
on Flint it was slower than stock: 36.8 against 40.3 tokens a second at short context, 15.0 against 25.4 after 32k
tokens.

**What I did.** I read its compiled machine code. The compiler reported zero register spills, but each thread kept a
32-float array in **local memory**: a per-thread slice of the card's main memory, reached through the L1 cache with a
latency of tens of cycles. The loop over that array could not be unrolled, so the array never fit in registers. The
code did est. 800 local loads and stores per tile of keys, against 2,048 useful multiply-adds. I made each step of the loop
a template on its width, so it unrolls fully, and the array now lives in registers: 168 of them, no stack frame, no
local memory, and still three blocks per SM.

**What happened.** On 640,000 emulated results the new code matches the old one in every bit. It has not run on a GPU
yet: open as of 2026-11-04, settled by its GPU check and a rig timing.

**Why.** "No spills" is not "no memory traffic". Look at the machine code for local loads and stores, not only at the
compiler's spill count.

## Written, checked, not yet timed

Three more pieces were written on 2026-11-03/04 and checked without a GPU. They sit in the merged build behind switches,
and their own effect on the rig is open.

| Piece | The problem | What it does | Estimate |
|---|---|---|---|
| Sparse attention | The attention layers need only 2,051 selected cache cells a token, but llama.cpp's gather needs a newer GPU, so a P106 reads the whole cache: 64 MiB per layer and token at 32k where 4 MiB would do | reads only each token's kept cells | −3 to −5 ms a token at 5k of context, −34 to −37 at 32k |
| Fused recurrent layers | 36 of the 48 layers are recurrent (a fixed-size state updated each token); each took 11 tiny launches, and its 3 MiB state was copied first | updates the state in place: 11 launches become 3, logits identical on the CPU | −5 to −7 ms a token (in the 82 ms build, not timed alone) |
| CUDA graphs per verify width | A CUDA graph is a recorded launch sequence, replayed like a prepared statement. llama.cpp keeps one, so each change of verify width re-records it: est. 10–20 ms per change | one recorded graph per width up to 8 (a port of upstream drafts) | +2–7 % per draft step |

## How I profile

**The problem.** NVIDIA's profilers (Nsight Systems, nvprof) refuse the P106. So the fork has its own profiler: it
records a CUDA event between every two groups of launches and reads the timestamps afterwards.

**What went wrong.** Each event costs 7.6 µs on a P106. A token has 2,520 of them, so the profiler added 19.2 ms and
made every small operation look 3.8 times its real size: a profiled token took 156.44 ms against ≈112–120 ms
without it. That inflation misled three estimates, including the first case for a megakernel (below).

**What I did.** I added three lighter modes: one event pair per layer, per sampled launch group, or per card's whole
work. They keep CUDA graphs on and never wait for the GPU; they read each timestamp a token later, when it is long
finished. The summary script subtracts the measured 7.6 µs per event.

**What happened.** The per-layer and whole-card modes agree within 0.3 ms, and the sampled mode comes 2.9 % and 1.3 %
below them at one and five tokens. The lightest costs 1.3 % of the wall time at one token and 2.7 % at five. On rig 1
they split a 1-token and a 5-token pass into parts:

| Part (ms a pass) | 1 token | 5 tokens | Growth |
|---|---:|---:|---:|
| Routed experts | 8.69 | 55.49 | 6.39× |
| Attention and recurrent projections | 27.13 | 61.77 | 2.28× |
| Hops between cards, host work, gaps | 11.60 | 40.54 | 3.50× |
| Hyper-connections (mixing Flash-Next's four parallel residual streams) | 10.03 | 22.25 | 2.22× |
| Everything else | 24.95 | 37.05 | 1.48× |
| **Whole pass, unprofiled** | **82.4** | **217.1** | **2.63×** |

The target had been a 5-token pass at most 1.3× a 1-token pass; it measured 2.63×. The table says where to look: the
experts (why they grow is [Chapter 7](07-experts.md)), the projections' launch shapes (the section above) and the hops
([Chapter 5](05-nine-cards.md)).

**A suspected trap in the benchmark tool.** llama-bench, llama.cpp's standard benchmark, fills its prompts with random
token ids. Uniform random routing would read 48.30 different experts per layer in a 5-token pass; five consecutive
tokens of real agent text use 30.86. So I expected llama-bench to overstate the expert cost of a multi-token pass. On
real text it did not: the routed experts took 8.62 ms at one token and 54.46 ms at five, against 8.69 and 55.49. A
5-token pass of real text took 208.1 ms against 217.1, and that difference is the output layer, which a pass over a
prompt skips. llama-bench's random tokens do not route uniformly either.

**Why.** A profiler is a program with its own cost. Measure that cost, subtract it, and check the sum of the parts
against an unprofiled run. And benchmark with the inputs the model really sees.

## The floor: 28.7 ms against 74.9

**The idea.** Before choosing the next change, compute the best possible token: the bytes it must read, divided by
the card's bandwidth.

**What I did.** I added up every tensor a Flash-Next token reads from the file's headers. Because the nine cards work
in turn (one card at a time holds the token), one card's 173 GB/s is the whole bandwidth.

**What happened.** A token reads 4,155 MB of weights: 24.0 ms at 173 GB/s. The recurrent state and the bytes crossing
the cards add 4.7 ms (1.4 and 3.3), so the floor is **28.7 ms a token, 34.8 tokens a second**. Against the 82.4 ms
token measured before the launch-shape fix, the gap of 53.7 ms split into four slices (estimates from the per-part timings):

| Slice of the gap | est. ms |
|---|---:|
| Weight products running below bandwidth (on average at half of it) | ≈24 |
| Outside the cards: host work, waits, the benchmark's logits copy | ≈13 |
| Operations without weights (norms, gates, state updates) | ≈9 |
| Extra time inside the model beyond each kernel's benchmark speed | ≈7 |

The biggest slice is kernel speed, and two thirds of it sat in one place: the attention and recurrent projections took
est. 23.8 ms for 8.05 ms of bytes. The 32 KB budget then took 7.5 ms off the token, largely in that slice. Launch
costs are spread over all four slices and total only est. 4.9–6.7 ms.

**Why.** The floor turns "make it faster" into a ledger. It said the next milliseconds are in the products, not in the
launches or the host.

## The megakernel and the engine question

**The idea.** A **megakernel** runs all of a card's work for a token inside one long-lived kernel, with the GPU itself
stepping through the operations. No launches, no gaps. Research groups have shown large gains this way on fast GPUs,
where a whole token takes about a millisecond.

**What I did.** I counted the launches in a Flash-Next token (est. 2,898 kernels in 2,520 launch groups) and timed a persistent kernel against a
CUDA graph on one recurrent layer's chain of 228 small operations, on a P106.

**What happened.** The best persistent kernel took 72 % of the graph's time per operation; I had set ≤ 50 % as the bar
to continue. Across a token that saves at most est. 2.3 ms of the 120.70 ms the profile then measured, 1.9 %. The track is
stopped. One persistent kernel did ship in v2: Flint's MoE layer as one launch instead of 13, which gave +5.6 % on nine
cards and +8.8 % on six and saved exactly the launches it removed, nothing more.

That raised a larger question: since the rig runs one model on one kind of card, why not drop llama.cpp and write a
dedicated engine? I priced three steps beyond the kernel work every option needs (estimates, one token, short context):

| Option | est. ms saved | est. weeks | Risk |
|---|---:|---:|---|
| Kernel work (launch shapes, a Q3_K one-token kernel, a 16-bit router) | 6–14 | 1–3 | low |
| (a) More fused kernels inside the fork, each behind a switch | 4–8 more | 3–5 | low |
| (b) A Flash-Next-only executor inside the fork (a fixed program per card) | 3–7 more | 6–10 | medium-high |
| (c) A standalone engine | 1–3 more | 20–40 | high |

**Why the engine rewrite is parked.** The largest slice is kernel speed, and a new engine would have to write the same
kernels again before it gained anything. The host-side generality a rewrite removes first cost est. 0.65–1.05 ms on Flint
once CUDA graphs ran. And a dedicated engine ported to another card without matrix units ran at 0.52–0.66× llama.cpp's
speed. So: kernels and fusions now; the executor only after two one-card benchmarks show the excess it would remove;
the standalone engine only if a measurement finds ≥ 5 ms of host time a pass that the fork cannot remove.

## What to remember

- On a small GPU, a kernel's speed is mostly about keeping enough bytes in flight: block size, registers and shared
  memory decide it as much as the math does.
- Rewrite one kernel at a time, behind a switch, and check it bit for bit against the old path before timing it.
- A value cached in too small a type (the FP16 block sum) fails silently on large inputs. Recompute it where the data
  already sits.
- Profile the profiler. Subtract its own cost and check that the parts add up to an unprofiled run.
- Benchmarks with random inputs can mislead: random tokens spread over experts more than real text does.
- Compute the floor first. It tells you which slice of the time is worth the next week.

## Where the details are

- What these kernels added up to in the v3 release, step by step: [chapter 10](10-v3.md#102-the-steps).
- Feature list of v2 with every check and rig result: [engine_features.md](../engine_features.md).
- v1 kernels and launch shape: registry rows Q34, Q38, Q46; [design notes](../research/notes/pascal_mmvq_integration.md).
- FP32 activation sums: Q76 (v2, switch `LLAMA_NO_MOE_DOWN_RESCALE`); the rescale and its NaN in
  [the Flint benchmark](../benchmarks/2026-10-22-flint.md); the same fix in llama.cpp's own kernel in the
  [kshape folder](../data/2026-11-04_kshape/README.md).
- 2-bit expert and BF16 kernels: Q223, Q546; [q2_0_kernel](../data/2026-11-03_q2_0_kernel/README.md),
  [bf16_kernel](../data/2026-11-03_bf16_kernel/README.md), rig timings in
  [kernel_timing](../data/2026-11-03_kernel_timing/README.md).
- Multi-token kernels: Q591 (k-column and multi-token, T(k) on the rig), Q825 (the down projection's alignment check);
  [kcol_dense](../data/2026-11-04_kcol_dense/README.md), [kcol_coverage](../data/2026-11-04_kcol_coverage/README.md),
  [iq_kcol](../data/2026-11-04_iq_kcol/README.md), [mtok_experts](../data/2026-11-04_mtok_experts/README.md); the
  "before" row is the v3 base in [verify_cost](../data/2026-11-03_verify_cost/README.md), the "after" row fork
  branch `pascal-p106-v3-b12076-ladder3` in [kcurve_p106](../data/2026-11-04_kcurve_p106/README.md).
- Attention: Q151 (FAP in v2, off), Q675 (the local-memory fix); [fap_spills](../data/2026-11-04_fap_spills/README.md).
  Sparse attention: Q230; [qsa_sparse_attn](../data/2026-11-04_qsa_sparse_attn/README.md).
- Recurrent fusions: Q436–Q438; [gdn_fusions](../data/2026-11-04_gdn_fusions/README.md). Verify graphs: Q52, Q701;
  [vgraphs](../data/2026-11-04_vgraphs/README.md). All merged in
  [ladder3_merge](../data/2026-11-04_ladder3_merge/README.md).
- Launch shapes: Q823 (the 32 KB budget, measured through `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`), Q824 (dispatch
  table), Q825; [kcurve_p106](../data/2026-11-04_kcurve_p106/README.md), [kshape](../data/2026-11-04_kshape/README.md).
  Branch `pascal-p106-v3-b12076-kshape` brought them; they are in the v3 release build, `pascal-p106-v3-b12076-ladder5`,
  where Flash-Next's preset sets `KCOL_SMEM=32768` and `KCOL_TABLE=0` ([chapter 10](10-v3.md)).
- Profiling: Q681 (light profiler, `GGML_CUDA_PROFILE_MODE`), Q822 (1- vs 5-token profile), Q827 (real-text passes,
  planned); [light_profiler](../data/2026-11-04_light_profiler/README.md),
  [kpass_profile](../data/2026-11-04_kpass_profile/README.md).
- The floor and the engine options: [flash_next_floor_and_engine.md](../research/notes/flash_next_floor_and_engine.md)
  (written against the 82.4 ms token, before the launch-shape fix), Q865–Q869, Q466 (standalone engine, parked).
- Megakernel: Q670, Q88 (Flint's persistent MoE kernel); [megakernel note](../research/notes/megakernel.md),
  [census](../data/2026-11-04_megakernel_census/README.md), [P106 bench](../data/2026-11-04_megakernel_rig/README.md).
