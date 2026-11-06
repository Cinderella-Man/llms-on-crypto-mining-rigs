# 4. Making models smaller

> **In short**
> - Quantization stores each weight in fewer bits. It works like lossy compression with a quality score: a smaller
>   file and fewer bytes read per token, at a measured cost in accuracy.
> - On this rig the format matters as much as the size. Simple formats decode fast on a 2016 GPU. Clever ones can be
>   slower than a file twice their size.
> - Where the bits go matters more than which format you pick. My best files keep the sensitive parts at 8 bits and
>   squeeze the rest.
> - For the 125B model I run now, the rig leaves est. 2.84 bits per expert weight. Two ways to win more room, cutting
>   rarely used experts and trellis formats, are still open.

The facts in this chapter are from 2026-11-04.

## Why shrink at all

[Chapter 2](02-how-an-llm-makes-text.md) showed that writing one token means reading every weight that takes part in
it. For one user, speed is roughly the card's memory bandwidth divided by the bytes read per token. The rig sets two
hard limits ([Chapter 1](01-the-rig.md)): 54 GB of card memory in total, and 173 GB/s of reading speed per card.

A weight at full 16-bit precision takes 2 bytes. Flint, the 39.8B-weight model I started with, would need 79.6 GB that
way. Qwen3.8-Flash-Next, the 125B model I run now, would need far more. Neither fits. So every file on the rig is a
compressed copy, and the compression decides two things at once: whether a model fits, and how fast it runs.

## Quantization: lossy compression for weights

**The idea.** Store each weight in fewer bits, the way a JPEG stores a photo in fewer bytes. Some detail is lost, and
a quality score tells you how much.

**How it works.** The weights are cut into small blocks, for example 32 in a row. Each block stores one scale factor in
16 bits, then one small integer per weight. To read a weight back, the GPU multiplies its integer by the block's scale.
Q8_0 uses 8-bit integers and Q4_0 4-bit ones. The scale is shared by the whole block, which is why Q8_0 costs 8.5 bits
per weight and not 8.

```
one Q4_0 block:   [ scale: 16 bits | 32 codes × 4 bits ]  = 144 bits for 32 weights = 4.5 bits per weight
reading it back:  weight[i] = scale × code[i]
```

Think of the weights as a huge read-only table that every request scans from start to end. Quantization is a codec for
that table: fewer bytes per page, a little decode work per value, and an error you can measure.

These are the formats this book mentions:

| Format | Bits per weight | How a block is stored |
|---|---|---|
| BF16, F16 | 16 | not compressed |
| Q8_0 | 8.5 | 32 weights, one scale, 8-bit codes |
| Q5_0 | 5.5 | 32 weights, one scale, 5-bit codes |
| Q4_0, Q4_1 | 4.5, 5.0 | 32 weights, one scale; Q4_1 adds a minimum |
| Q4_K, Q5_K, Q6_K | 4.5, 5.5, 6.56 | 256-weight superblocks, with a small extra scale per 32 weights |
| IQ4_XS, IQ4_NL | 4.25, 4.5 | 4-bit indices into a fixed table of 16 uneven values |
| IQ3_S, IQ3_XXS | 3.44, 3.06 | indices into a codebook of 4-weight patterns, plus signs |
| IQ2_XS | 2.31 | indices into a codebook of 8-weight patterns, plus signs |
| Q2_0 | 2.25 | 64 weights, one scale, 2-bit codes for −1, 0, 1 and 2 |

**K-quants** add a second level. A big block of 256 weights holds one main scale, and each group of 32 inside it gets
its own small scale on top. That follows the real spread of the weights more closely at the same size.

**IQ formats** go further. Instead of one code per weight, they store an index into a fixed codebook of common patterns
of 4 or 8 weights. It is a dictionary compressor: one index stands for a whole group of values.

**What happened on this rig.** The cleverer formats compress better, but decoding them costs work, and a P106 has only
5 streaming multiprocessors to do it ([Chapter 6](06-gpu-code.md)). On the first day, a Q4_K_M file was 6.5 % bigger
than Q4_0 but 22.9 % slower ([Chapter 1](01-the-rig.md)). The IQ formats need a table lookup every 4 to 8 weights: an
IQ3_XXS weight took 20.5 ps to decode in llama.cpp's kernel, a Q4_0 weight 4.2 ps in mine. On nine cards Flint writes
28.1 tok/s from the makers' simple-format Q4_1 (26.0 GB) and 19.0 from Unsloth's K-quant UD-Q4_K_XL (25.1 GB), 1.48
times as fast.

**Why.** "Speed equals bandwidth divided by bytes" holds only while the decoder keeps up with memory. On a small GPU,
below est. 3.5 bits per weight, the card runs out of instructions before it runs out of bandwidth. So a format is a
trade between bytes and decode work, and on a P106 the simple formats often win.

### Importance matrices: profile first, then compress

**The idea.** Not every weight matters equally. An importance matrix records which weights carried the most signal
while the model read some sample text, the calibration text. The quantizer then puts its rounding error where it hurts
least. This is profile-guided optimization: run a representative workload, record the hot paths, optimize for them.
Like a profile, it is only as good as the workload, so I build mine from agent sessions.

**What happened.** On Ornith-1.5-35B, a matrix from 102,400 tokens of agent sessions brought a Q4_K_M file 29.1 %,
24.9 % and 27.6 % closer to the reference on wiki, agent and code text. For plain Q4_0 experts it barely helped, and on
agent text it hurt by 8.6 % (KLD 0.0396 to 0.0430).

**Why.** K-quants have more freedom per block, so they gain most from a good profile. And in a mixture-of-experts model
each expert sees only its share of the calibration text, 3,200 tokens on average here, so rare experts get a thin profile.

### Scoring the loss

[Chapter 3](03-measuring.md) explains the quality score. The KL divergence (KLD) is a diff score between the outputs
of the original and the compressed file on the same text, where 0 means identical. "Same top token" counts how often
both pick the same most likely next token. There is a noise floor: the same Flint file on two types of GPU already
differs by a KLD of 0.065, because tiny rounding differences flip which expert the router picks.

### The quality bar

A faster file that answers worse is not a win, so I needed a fixed line. I chose the file people already download:
Unsloth's UD-Q4_K_XL of Flint. Against the full-precision model it scores a KLD of 0.120 and 87.7 % same top token.
Since 2026-10-23 a Flint file of mine counts only if it reaches about that: KLD at most 0.125, same top token at least
87.5 %. The makers' own Q4_1 file was much faster, but it agreed with the full-precision model on only 78.4 % of
tokens. Nobody gives up 9.3 points of agreement for speed.

Every speed claim for Flint is made against that file, on the same engine and the same cards.

## My own file mixes

A file doesn't have to use one format. Each tensor (each named block of weights, such as one layer's attention
weights) can have its own. A mix is the list of those choices. This is where I found the most quality per byte.

### Flint: where the bits go

**The idea.** Reach Unsloth's quality using only formats that decode fast on Pascal: Q4_1, Q5_0 and Q8_0.

**What I did.** I scored the makers' files and Unsloth's file. Then I changed one kind of tensor in one group of 8
layers at a time, and measured what each change cost in KLD and saved in bytes.

**What happened.**
- **Plain Q4_0 broke Flint:** KLD 0.565, perplexity 45 % worse. Seven tensors had a few huge channels. With only those
  seven kept at Q8_0, the KLD fell to 0.313.
- **The same change cost very different amounts in different places.** Lowering the attention output to Q4_1 cost
  +0.012, +0.017 and +0.018 of KLD in the three groups of layers 0–24, and +0.006, +0.001 and +0.002 in the three
  groups of layers 25–48. Per 100 MB saved, the cheapest group cost 0.001 of KLD and the dearest 0.11, 110 times as
  much.
- **The v1 file** used those costs. It scored KLD 0.123 and 88.7 % same top token at 32.5 GB, level with Unsloth's
  file. On real chat with 262k tokens of context it wrote 24.7 tok/s, against 20.4 for Unsloth's file.
- **The v2 file.** My own kernels later made Q5_0 nearly as cheap as Q4_1, 1.14 times the time per weight instead
  of 1.40. So I re-priced every mix. The result, Flint Pascal Q5 GGUF, scores KLD 0.1048 and 88.84 % at 29.39 GB,
  reading 2.091 GB per token. On the agent check it ran at 53.8 tok/s, against 50.7 for Unsloth's file.
- **Two small tricks.** GPTQ, a rounding that corrects each weight's error in the weights still to come, let the output
  layer drop to Q5_0 for 0.0023 of KLD instead of 0.023, saving 0.116 GB per token. The routers, the small tensors
  that pick experts, went from F32 to F16 and saved 49 MB per token for free.

**Why.** A model's sensitivity is not spread evenly. Flint's early layers carry it. Ornith-1.5-122B's sensitive layers
are the late ones, the reverse. So the costs have to be measured for each model, and a mix built from measured costs
beats any single format.

### A file next to Q8_0, with fewer bytes

**The idea.** My headline Flint file is plain Q8_0 on nine cards. A file within noise of Q8_0 that reads fewer bytes
would be pure speed.

**What I did.** I started from Q8_0 and defined 23 switches, each moving one kind of tensor in one layer group down to
Q5_0. I scored every switch alone. Then a small dynamic program, a knapsack solver, picked the best set of switches for
each byte budget. The idea of measuring each change and solving for the set came from a paper called SLQ.

**What happened.** The chosen file scored a KLD of 0.0656, against 0.0580 for the Q8_0 it started from, reading
11.0 % fewer bytes per token (a 40.43 GB file). The prediction landed 0.0011 from the real file's score. On the rig it
was 6.3 % faster per generated token than Q8_0 (21.58 against 22.92 ms). Whether it replaces Q8_0 in a release is still open.

**Why.** When the costs of separate changes simply add up, a cheap search finds the best file without building hundreds
of them. On Flint they added up well.

### Ornith: the same method at 2 to 3 bits

Ornith-1.5-122B fits only at 2 to 3 bits per weight. My two new Pascal formats, N2P and N3P, decode at Q4_0's speed
(an expert's gate weight in 4.18 and 3.80 ps, against 3.85; [Chapter 6](06-gpu-code.md)). Used for every expert, they did not beat Unsloth's UD-Q2_K_XL. Allocated per layer
group with the same switch method, they beat it on all three test texts and ran the agent check at 14.5 tok/s against
11.3, on different builds (14.9 against 12.6 on v3's). The formats gave the speed; the allocation gave the quality.

## Retyping the float tensors in Flash-Next

Qwen3.8-Flash-Next is a 125B mixture-of-experts model that uses about 6B weights per token. It also carries a
51B-weight n-gram table, a 28.8 GB lookup table that [Chapter 5](05-nine-cards.md) keeps on the SSD. The files I run
are quantized copies of Swift-1.5, a fine-tune of Flash-Next that thinks less. They were made with GSQ-RCO, a method
that learns how to round each weight instead of rounding to the nearest value.

**The idea.** These files kept two kinds of tensor in floats: the hyper-connections, small matrices that mix the
model's running state between layers, in BF16, and the routers in F32. They took a large share of each token's time.
Converting them to Q8_0 and F16 would cut bytes without touching the experts.

**What I did.** A script rewrote 290 hyper-connection tensors to Q8_0 and 96 routers to F16, byte for byte what
llama.cpp's own quantizer writes, and left every other tensor identical.

**What happened.** Each file lost 725.7 MB on the cards. The KLD against the unmodified Q2_0 file was 0.0166 on agent
text, 2.1 times the noise floor (0.0079). On the rig's agent check the retyped file passed 12 of 12 tasks at 9.0 tok/s,
against 7.8, and its GPU time per token fell from 135.96 to 120.70 ms.

Then the result turned around. A day later I wrote a Pascal kernel for BF16 math ([Chapter 6](06-gpu-code.md)). With
it, the original BF16 tensors became cheaper than the retyped Q8_0 ones, whose rows are only 320 weights wide: the
unmodified file took 74.9 ms per token, the retype 78.7 at best. And the F32 routers had held BF16 values all along,
with the low 16 bits of every one zero, so routers in BF16 are exact.

**Why.** A format's cost belongs to the kernel that reads it, not to the format. When the kernels change, every
"this type is cheaper" decision has to be measured again.

## Pascal-fast files: keep the experts, retune the rest

**The idea.** For Flint I made my own quants from the full model. For Flash-Next, the makers' learned-rounding experts
are hard to beat, and they are 33.98 of the 37.47 GB on the cards. But of the 4,155 MB a token reads, 3,492 MB are the
dense weights that every token uses: attention, the shared expert and the rest. So I keep the experts byte for byte
and give every other tensor the type a P106 reads fastest. The rule: no tensor gets fewer bits than the makers gave it.

**What happened.** The first surprise was in the speed table. The makers' Q3_K turned out to be the slowest type per
weight on a P106, slower than Q6_K with nearly twice its bits. Swapping Q3_K for IQ4_XS gives both more bits and more
speed. The first file, Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (`pmix-q2-fast`), changes 168 tensors and adds only 5 MiB on the cards. Its estimated GPU time per
token falls from 82.4 to 72.4 ms, which is 12.1 to 13.8 tok/s before any guessing ahead. Its agent-text KLD moved from
0.2866 to 0.2792.

**Why.** Raising 1,106 million dense weights from 3.44 to 4.25 bits barely changed the score. So the error of the Q2_0 file
lives in its routed experts, and the dense part can be chosen for speed. This is still open: the speed is an estimate
from kernel timings, the rig runs are queued, and the IQ3_XXS version is built but not yet scored.

## Choosing Flash-Next's file

**The budget.** Nine cards hold est. 47 GB of weights once the context cache and work buffers are paid for. Est.
4.1 GB of that goes to tensors other than the routed experts. The experts have 120.8B weights, so each bit per weight
costs 15.1 GB. That leaves the experts est. 2.84 bits per weight, or 2.70 with the MTP draft model on the cards too
([Chapter 8](08-guessing-ahead.md)). Every file worth considering sits right on that line.

| File | On the cards | KLD, agent text | KLD, code | Fits nine cards at 64k tokens | Speed on the rig |
|---|---|---|---|---|---|
| Swift Q2_0 | 35,730 MiB | 0.285 | 0.112 | yes, with room | measured: the file in use today |
| Swift IQ2_XS | 37,196 MiB | 0.225 | 0.075 | yes | est. 4.8 ms a token slower than Q2_0 (9.5 with the expected kernels) |
| Swift IQ3_XXS | 44,710 MiB | 0.176 | 0.054 | only just, with tiny batches or moved experts | est. 17–21 % slower than Q2_0 |
| Swift IQ3_S | 52,066 MiB | 0.121 | 0.032 | no | — |

The KLDs are against Swift-1.5's Q8_0, whose noise floor on agent text is 0.029. Q2_0's codes are just −1, 0, 1 and 2
times a scale. Rounded the ordinary way it is unusable: in a proxy test its error was 17.5 times IQ3_XXS's. GSQ-RCO's
learned rounding is what makes it work.

**What happened.** IQ3_XXS is the best file that fits. It cuts the agent-text KLD 1.62 times against Q2_0 (0.176 against
0.285), for an estimated 17–21 % slower decoding, with IQ2_XS as the fallback. Q2_0 is what the rig runs today: it fit first, and the
kernels were built for it. The agent check of IQ3_XXS on the rig is open.

**The n-gram table** stays at 28.8 GB in 4.5-bit IQ4_NL. At 4.1 bits it would save 2.6 GB but moved prose outside the
noise; at 3.1 bits it failed. Its weights look like Gaussian noise with no empty rows, so there is nothing to prune.

## Pruning experts nobody calls

**The idea.** Experts are handlers that a router picks for each token ([Chapter 7](07-experts.md)). If some handlers
almost never run, delete them. The file shrinks and most traffic sees no change. A token whose expert was deleted gets
the next expert on its router's list.

**How it should work.** Count how often each expert is chosen on representative traffic, then keep the top N in every
layer.

**What I did.** I counted 1,024,000 routings per layer over 102,400 tokens of agent sessions. A new tool removes
experts from a model file exactly and checks every byte it keeps. With it I cut Swift IQ3_S, the best file that does
not fit, from 512 experts per layer to 464, 448 and 416.

**What happened.**
- **Usage has a cheap tail.** In the median layer 54 experts carry half the routings, and 406 carry 99 %. Keeping 464
  per layer keeps 99.78 % of all routings, and keeping 416 keeps 99.08 %.
- **The cuts save 4.74, 6.32 and 9.48 GB.** At 416 experts the file fits 64k tokens of context on whole layers.
- **The first quality check is too small to decide.** On a CPU, with only 1,024 tokens of agent text, the three cuts
  scored a KLD of 0.060, 0.044 and 0.056 against the uncut file. That is about the size of IQ3_S's whole lead over IQ3_XXS (0.055),
  and the sample cannot rank the cuts.
- **Open:** a GPU run on all three texts decides whether a cut IQ3_S beats IQ3_XXS at the same size.

**ISTA's Coder cut.** ISTA, the lab that published the GSQ-RCO files of the original model, also published a Coder file
that keeps 256 of 512 experts per layer, chosen on code, agent, vision and spatial text. On my agent text those 256
carry only 66.8 % of the routings; the 256 most used would carry 90.8 %. Against the same file uncut, the cut alone
moves the KLD from 0.117 to 0.274 on agent text and from 0.161 to 0.846 on wiki text. Published pruning research agrees:
coding scores drop 1.9 % with a quarter of the experts removed and 6.9 % with half.

**Why.** Cut on the traffic you serve, and cut lightly. Halving the experts is far past the cheap tail.

## Trellis formats: fewer bits, more decode work

**The idea.** A trellis format stores a bit stream that drives a tiny state machine. At each step the current state is
hashed into a fixed codebook, and the codebook entry gives the next weights. Within a block, each value depends on the
state before it, like a stream decoder. This reaches fractional bit rates with good quality.

**The files.** agentionai's Gyro files store Flash-Next's routed experts in TQ2_T, at 2.125 bits per weight. A block of
128 weights holds one 16-bit scale and a 15-bit starting state. One step is a 32-bit multiply, a shift, a mask and one
read from a 2,048-entry codebook, which yields two weights. Everything else in the file is Q8_0. Gyro-Coder keeps 256
experts per layer and puts 19.29 GiB on the GPUs, 55 % of what Swift Q2_0 puts there.

**What happened.** agentionai report a KLD of 0.118 on C++ code and 0.579 on held-out text, against their own Q8_0 on
their own texts. Those numbers are not on my scale. My scoring of both Gyro files on the same three texts as every other
file here was set up on 2026-11-04 and is open.

**Can a P106 decode it?** Pascal has no single instruction for a 32-bit multiply, so the hash costs three. My
first estimate was 9–28 ps per weight. To pay, a trellis decode needs to come in near 5 ps per weight. On one P106,
with the codebook held in the card's fast on-chip memory, agentionai's kernels took 23.8 ps per weight for the experts'
gate and up products and 37.8 ps for the down product at one token (10.0 and 22.3 in their best layouts), against 3.3
and 3.9 for Miner Llama's Q2_0. They miss the line.

**Why it is worth the trouble.** If both answers come back good, half the expert bytes would buy room for better types
elsewhere, or for the MTP draft model, on the same nine cards.

## What didn't work

| Idea | What happened | Why |
|---|---|---|
| Hadamard rotation of Flint's weights before 4-bit quantization | Rotated Q4_1 was worse: KLD 0.308 against 0.283 | llama.cpp also squeezes activations to 8 bits per 32 values; rotation spreads Flint's few huge channels into every one of those blocks |
| Hadamard rotation of 2–3-bit experts | 9.1 % more error in a proxy test | Parked until a learned rotation or quantization-aware training |
| Fewer experts per token on Flint, 6 down to 5 or 4, for every token | 3.6 % or 8.6 % faster (25.13 or 26.35 against 24.26 tok/s), for 1.9 or 6.7 points less same top token | Attention, the shared expert and the output layer don't shrink, so the saving is small and the loss is not |
| Uniform 2–3-bit grids, as in Q2_K and Q3_K | 13.0 % (2-bit) and 14.9 % (3-bit) more error than codebook formats of the same size | Weights are not spread evenly; a codebook follows their shape |
| SLQ's own formats | Converted exactly to GGUF types, est. 27 % (with Q5_1) or 53 % (fast types only) more bytes than a GGUF at the same quality | Kept only its allocation idea |
| The makers' Flint tensors moved into my files | Worse: experts KLD 0.208 against 0.178 | The makers' tensors only work together, inside their own file |
| Methods below 1 bit per weight | Not started: the best 1-bit results raise perplexity 1.64 times (Llama-2-7B) to 5.6 times (Gemma-3-27B) on dense models, and none has run on an MoE | The evidence argued against it |
| A smaller n-gram table | 4.1 bits outside noise, 3.1 bits failed | Its weights hold no structure to exploit |

## What to remember

- Quantization is lossy compression with a score. Measure the loss against a noise floor, and fix a quality bar before
  you chase speed.
- On a small GPU, the decode cost of a format matters as much as its bytes. The simple formats often win.
- Where the bits go beats which format: measure what each tensor group costs, and spend bits where they are cheap.
- A format's cost belongs to its kernel. When the kernels change, measure the file choices again.
- Cut experts by measured usage, on the traffic you serve, and cut lightly.
- An estimate is not a result. The Pascal-fast files, the cut files and the trellis formats stay open until they are
  measured on the rig.

## Where the details are

- How to make and score quants: [quants/guide.md](../../quants/guide.md); Flint's mixes and recipes:
  [quants/README.md](../../quants/README.md), [models/flint](../../models/flint/README.md); every experiment: the
  [registry](../../quants/experiments.md).
- Noise floor: Q21. Quality bar: the registry's Plan section (decided 2026-10-23). Format decode speeds:
  [pascal_lowbit_format.md](../research/notes/pascal_lowbit_format.md),
  [kquant_decode_pascal.md](../research/notes/kquant_decode_pascal.md).
- Importance matrix effect: Q159, [2026-10-28_ornith35_kld](../data/2026-10-28_ornith35_kld/README.md).
- Flint: Q24 (Q4_0 fix), Q36 (v1 file S4-L0), Q73 and Q77 (re-pricing), Q56 and
  [2026-10-24_q56_gptq_head](../data/2026-10-24_q56_gptq_head/README.md) (GPTQ head), Q113 (F16 routers), Q41 (task
  check), Q44 (makers' tensors), Q74 and [2026-10-23_qat_check](../data/2026-10-23_qat_check/README.md) (the makers'
  Q4_1 is a plain rounding of their own 4-bit release, not a trained one). The v2 file, Flint Pascal Q5 GGUF, is K2h-r16.
- Flint Pascal Q8 GGUF (`p065`): Q168, [2026-10-27_flint_near_q8](../data/2026-10-27_flint_near_q8/README.md),
  [slq.md](../research/notes/slq.md).
- Ornith-1.5-122B: Q164, Q167, Q212, Q189; [2026-10-27_lowbit](../data/2026-10-27_lowbit/README.md),
  [2026-10-27_ornith_alloc](../data/2026-10-27_ornith_alloc/README.md),
  [2026-10-28_ornith_valid1](../data/2026-10-28_ornith_valid1/README.md).
- Retypes: Q218, [2026-11-03_retype](../data/2026-11-03_retype/README.md),
  [2026-11-03_retype_agent](../data/2026-11-03_retype_agent/README.md); the BF16 kernel that reversed it: Q546,
  [2026-11-03_bf16_kernel](../data/2026-11-03_bf16_kernel/README.md); the k-curve with both files: Q591,
  [2026-11-04_kcurve_p106](../data/2026-11-04_kcurve_p106/README.md). Hyper-connections below Q8_0: Q620, open.
- Pascal-fast files: Q880–Q889 and Q899, [2026-11-04_pascal_mix](../data/2026-11-04_pascal_mix/README.md),
  [2026-11-04_gguf_compose](../data/2026-11-04_gguf_compose/README.md).
- File choice: Q811, [2026-11-04_file_choice](../data/2026-11-04_file_choice/README.md); quality chain (FN1, FN2):
  Q216, Q217, [2026-11-03_flash_next_quality](../data/2026-11-03_flash_next_quality/README.md); the bit budget:
  [sub_1_bit_quants.md](../research/notes/sub_1_bit_quants.md); the Q2_0 kernel: Q223,
  [2026-11-03_q2_0_kernel](../data/2026-11-03_q2_0_kernel/README.md); uniform grids and mainline Q2_0: Q305.
- N-gram table bits: Q433, [2026-11-03_table_bits](../data/2026-11-03_table_bits/README.md).
- Expert usage and pruning: Q232, [2026-11-04_expert_usage](../data/2026-11-04_expert_usage/README.md); the cut tool
  and files: Q890, [2026-11-04_gguf_prune](../data/2026-11-04_gguf_prune/README.md),
  [scripts/gguf_prune.py](../../scripts/gguf_prune.py); the GPU score of the cuts: Q891, open. ISTA's Coder and
  published pruning results:
  [models_large_moe_and_compression.md](../research/notes/models_large_moe_and_compression.md),
  [flash_next_variants.md](../research/notes/flash_next_variants.md).
- Gyro and trellis formats: Q896 (scores, open), [2026-11-04_gyro_kld](../data/2026-11-04_gyro_kld/README.md); Q895
  (P106 decode timing, open), Q241 (the earlier estimate, parked);
  [flash_next_news_2026-11.md](../research/notes/flash_next_news_2026-11.md).
- What didn't work: Q13 and Q304 (Hadamard), Q6 (fewer experts per token; its per-token version for Flash-Next is
  Q618, open), Q305 (uniform grids), [2026-10-27_slq](../data/2026-10-27_slq/README.md) (SLQ formats), Q44,
  [sub_1_bit_quants.md](../research/notes/sub_1_bit_quants.md), Q433.
- Where sources differ, this chapter uses the latest measurement or estimate: Q74 for how the makers' Flint Q4_1 was
  made, and Q895's P106 measurement of a trellis decode with the codebook in shared memory
  ([gyro_p106](../data/2026-11-04_gyro_p106/README.md); Q241's estimate of 9–28 ps came before that design).
