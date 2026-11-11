# 1. The rig: nine GPUs, one slow link each, and 4 GB of RAM

> **In short**
> - The rig is a Manli mining box: nine 6 GB NVIDIA P106-090 cards, a 2-core Celeron, 4 GB of RAM and a 64 GB SSD.
> - Its strength is memory: 54 GB on the cards, each card reading its own memory at 173 GB/s.
> - Its weaknesses are everything else: tiny 5-SM chips, one PCIe 1.0 lane per card, no AVX on the CPU, very little
>   host RAM.
> - Each limit decides something: what software runs, how a model is split, which flags keep the server alive.
> - Free hardware settings gave almost nothing. The speed has to come from software.

Facts here are from 2026-11-04. Measurements were taken on the rig itself, not copied from spec sheets.

## The machine

The Manli M-P106L9-N6G is a complete mining computer in one case. A purpose-built motherboard carries nine graphics
cards, a small Intel CPU, an SSD and a wall of ten fans. It was sold for mining Ethereum. Ethereum stopped using
mining in September 2022, and since then these boxes have had no job. Thousands of them were built, and the product
is discontinued.

| Part | What it is | What it means for a language model |
|---|---|---|
| GPUs | 9 × NVIDIA P106-090, 6 GB each, 54 GB in all | Plenty of memory for big models, but in nine separate 6 GB pools |
| Per card | Pascal chip, 5 streaming multiprocessors, 173 GB/s memory bandwidth (measured), 75 W | Fast memory, small chip: the chip often cannot keep its memory busy |
| Links | One PCIe 1.0 lane per card, 189 MB/s; no direct card-to-card path | Data crosses between cards slowly, always through the host |
| CPU | Intel Celeron 3865U: 2 cores at 1.8 GHz, no AVX | Too slow to compute any part of a model |
| Host RAM | 4 GB, one channel | Every buffer the server keeps in RAM counts; the default settings crash it |
| Storage | 64 GB SATA SSD | Model files of 20.8 to 66.5 GB live on a USB drive; only small, hot files go on the SSD |

A *streaming multiprocessor* (SM) is one of the GPU's compute units, a bit like a CPU core that runs many threads at
once. This card has 5; the P106-100, a fuller version of the same chip, has 10. *Memory bandwidth* is how many bytes per second the
chip can read from its own memory. *PCIe* is the bus between a card and the rest of the computer.

## Why a mining rig at all

Ethereum's mining algorithm was limited by memory speed, not by arithmetic. So mining cards pair a mid-range chip with
fast memory. The P106-090 is a cut-down GTX 1060 chip, NVIDIA's 2016 generation, with no display outputs. Each one
reads its 6 GB at 173 GB/s. Together the nine read 1,557 GB/s.

[Chapter 2](02-how-an-llm-makes-text.md) explains why that is the number that matters. In one line: to write each
token, the GPU must read most of the model's weights from memory, and the arithmetic per weight is tiny. A rig built
for memory reading is built for the expensive part.

The catch is that a model does not get 1,557 GB/s. The nine pools are separate. A model is cut into nine slices, and
the cards take turns. [Chapter 5](05-nine-cards.md) is about living with that.

## Limit 1: the cards are small and old

### What they lack

The P106-090 is a Pascal GPU with compute capability 6.1. *Compute capability* is NVIDIA's version number for a chip's
features. Pascal is old enough that much of today's software skips it:

- **No tensor cores.** Newer GPUs have special units for the matrix maths of AI. Pascal has none.
- **Slow 16-bit maths.** Half-precision floating point runs at 1/64 of the 32-bit rate. Many modern engines are built
  on 16-bit maths, so they either refuse to run or crawl.
- **Fast 8-bit integer dot products.** Pascal has one useful trick: an instruction that multiplies and adds four 8-bit
  numbers at once. llama.cpp's compressed-weight code uses it. That is one reason llama.cpp runs well here.
- **The last driver and toolkit.** NVIDIA's driver 580 and CUDA 12.x are the last versions that support Pascal. CUDA 13
  cannot build code for these cards. Everything here is built with CUDA 12.9 for this exact chip.

A survey at the start found that only the llama.cpp family runs on this chip. Other popular engines need newer GPUs,
and one llama.cpp variant needs AVX2 on the CPU, which the Celeron lacks.

### Story: how much of the bandwidth does real code use?

**The idea.** Before tuning anything, find out what one card really does. Are the nine cards equal? How close does real
code get to the 173 GB/s?

**How it should work.** Run one small, well-known model on each card in turn. Divide the bytes read per token by the
time per token. That gives the *effective bandwidth*, the share of the memory speed the software actually uses.

**What I did.** I ran Llama-2-7B, llama.cpp's standard benchmark model, in three storage formats on each card. Then I
split it over two cards and over all nine.

**What happened.** All nine cards generated within 4.7 % of each other (18.78 to 19.67 tok/s on Q4_0). But real
code used far less than the memory could give:

| Weight format | Generation | Effective bandwidth | Share of 173 GB/s |
|---|---|---|---|
| 8-bit (Q8_0), on two cards because it does not fit one | 13.23 tok/s | 92.9 GB/s | 53.7 % |
| 4-bit, simple (Q4_0), the mean of the nine cards | 19.45 tok/s | 73.0 GB/s | 42.2 % |
| 4-bit, two-level blocks (Q4_K_M) | 14.99 tok/s | 60.1 GB/s | 34.7 % |

A P106-100, the same chip with 10 SMs instead of 5, reaches 114.9 GB/s on the same 4-bit file (from its 30.64 tok/s
on llama.cpp's public scoreboard).

**Why.** With only 5 SMs, the card cannot issue memory requests and unpack compressed weights fast enough to keep its
memory busy. The memory waits for the chip, not the other way round. Simple formats are cheap to unpack, so they come
closer to the limit. The Q4_K_M file is 6.5 % bigger than Q4_0 (4.08 against 3.83 GB) but 22.9 % slower. That one result shaped every file choice
that followed ([chapter 4](04-smaller-models.md)) and most of the GPU code ([chapter 6](06-gpu-code.md)).

## Limit 2: nine pools, one slow lane each

### The links

Each card sits on a single PCIe lane at 1.0 speed. A desktop graphics card gets 16 lanes of a newer generation:
at PCIe 3.0, est. 63 times as much on paper. Eight of the nine cards share two small switches, four cards per switch:

```
Celeron + chipset
├─ port 1 ─ switch ─┬─ GPU 0
│                   ├─ GPU 1
│                   ├─ GPU 2
│                   └─ GPU 3
├─ port 2 ───────────── GPU 4
├─ port 3 ───────────── network card
└─ port 4 ─ switch ─┬─ GPU 5
                    ├─ GPU 6
                    ├─ GPU 7
                    └─ GPU 8
```

Measured on the rig, with 64 MB transfers:

| Transfer | Speed |
|---|---|
| Host to one card, one at a time | 189 MB/s (GPU 4, on its own port: 200 MB/s) |
| One card to host | 213 MB/s |
| All nine cards loading at once | 715 MB/s in total, 90 to 120 MB/s per card behind a switch |

The cards cannot talk to each other directly. Anything that moves from card 3 to card 4 goes up to host memory and
back down, over two slow legs.

### Why it matters

For mining, the links didn't matter: the host sends a small job and gets a short answer back. For a language model
they set three things.

- **The model is split by layers.** A model is a stack of layers, and each card holds a run of consecutive layers. For
  each token, only a small vector of numbers passes from one card to the next. Splits that exchange large blocks of
  data on every step are out of the question.
- **Every hand-off costs time.** Llama-2-7B ran 3.8 % and 5.0 % faster on two cards than on one (20.19 and
  20.43 tok/s against 19.45), because each card worked only about half the time (46.3 % and 49.0 % on GPU 0 and 1)
  and stayed further under its power limit. On nine cards it ran at 17.67 tok/s, 12.5 % slower than the 20.19 on two:
  1.01 ms per extra card boundary, plus idle cards slowing their clocks. More cards add memory, not speed.
  [Chapter 5](05-nine-cards.md) measures the hand-offs in detail.
- **Loading is slow, once.** Filling one card's 6 GB takes est. 34 seconds, and all nine est. 81 seconds even in
  parallel. Flint's 42.3 GB file loads from a USB stick in 565 s on my engine (568 s on stock llama.cpp). After that, the weights never cross the
  link again, so loading time never slows generation.

### Story: can the links run faster?

**The idea.** The switches and the chipset ports can do PCIe 2.0, twice the speed. Maybe the cards can too.

**What I did.** I read each card's link capability register and the switch ports' registers.

**What happened.** The switches and ports offer 2.0 speed. The cards themselves advertise only 1.0 speed. No BIOS,
switch or register setting can train a link faster than the card allows, and Pascal's firmware is signature-checked.

**Why.** It is a hardware limit, and a dead end. Even if it worked, my estimate is about 1 % faster generation on nine
cards, because the hand-offs are small. Only loading would speed up.

## Limit 3: a weak host with 4 GB of RAM

### The CPU computes nothing

The Celeron has two cores and no AVX vector instructions. It cannot help with the model's maths at any useful speed,
so every weight must sit in GPU memory. That makes the 54 GB a hard ceiling.

No AVX has a practical side too. A program built on a modern PC usually contains AVX instructions, and on this rig it
crashes with "Illegal instruction". I build every engine on another PC, inside Docker, for this exact GPU and without
AVX. The rig only receives a finished archive.

One early worry was that two cores could not feed nine GPUs. A test on small models showed otherwise: on five models the
GPU was busy 95, 96, 96, 97 and 98 % of the time, and taking away a whole CPU core cost at most 6.3 %. The CPU is weak, but it is rarely the
brake.

### The 4 GB are the tightest limit

Linux sees 3,837 MiB of the 4,096 MiB. The desktop, the driver, nine GPU contexts and the server all share it. Some
buffers must be *pinned*: locked in RAM so the GPU can copy from them directly. Pinned memory cannot be swapped out.

**Story: the server that took the desktop down.**

**The idea.** Run a model with its full 262,144-token context, its longest input.

**What I did.** I started llama-server, the engine's HTTP server, with its default settings.

**What happened.** The kernel's out-of-memory killer ended the server, and parts of the desktop with it. At that
context, the default *micro-batch* of 512 tokens made llama.cpp pin est. 1 GiB of RAM for one buffer, the attention
mask (520 MiB at half that context). A micro-batch is the number of prompt tokens the engine processes in one GPU step. With a micro-batch of 128,
the buffer shrank to 258 MiB, and generation speed did not change.

**Why.** llama-server's defaults assume a machine with plenty of RAM. On this host three settings decide whether the
server lives:

| Setting | Default | On this rig | Why |
|---|---|---|---|
| Micro-batch (`-ub`) | 512 | 128 at long context | The attention mask is pinned in host RAM and grows with context × micro-batch |
| Prompt cache in RAM (`--cache-ram`) | 8 GiB | 0 | Saved prompts would live in host RAM |
| Saved checkpoints (`--ctx-checkpoints`) | 32 | 0 to 4 | Each one holds a copy of the model's running state in host RAM |

These flags are in every command line since. [Chapter 2](02-how-an-llm-makes-text.md) explains the checkpoints.
Later work moved the checkpoints off the host altogether, into card memory, which turned out faster as well
([chapter 14](14-reading-prompts.md)).

**Flash-Next squeezes even harder.** On 2026-11-03, a coding-agent session on Flash-Next left as little as 195 MiB of
RAM available, with up to 1,043 MiB swapped out. That is why a small extra model, which guesses tokens ahead, at first
could not load beside it ([chapter 8](08-guessing-ahead.md)).

**I keep the rig as sold.** The board can take a bigger memory module, and that would be the easy fix. I decided
against it. The point of this project is that anyone with the same rig can do the same, with no new parts. So the RAM
problem is solved in software: turning off the desktop freed 276 MB, and a kernel boot setting returns more.
[Chapter 5](05-nine-cards.md) tells that story.

## Limit 4: a small disk

The 64 GB SSD holds the operating system, the CUDA libraries and not much else. Model files are 20.8 to 66.5 GB each. They
live on a USB 3 drive instead. A tested USB 3 stick reads about 410 MB/s, faster than one card's link can accept, so it
does not slow loading.

**Story: where Flash-Next's lookup table can live.**

**The idea.** Flash-Next carries a 28.8 GB lookup table. Only a few rows of it are needed per token, so it doesn't need
to sit on a GPU. It can stay on disk and be read on demand.

**How it should work.** The engine maps the file into memory. The operating system reads the pages a token needs from
disk and keeps recent ones in its cache.

**What I did.** I measured random 4 KiB reads on the internal SSD and on a USB hard disk, the access pattern of the
table.

**What happened.** The SSD did 6,039 random reads per second, one at a time. The USB hard disk did 95. In a replay
of the agent check's sessions, each generated token needed 5.70 reads. On the hard disk that would add est. 37 to
59 ms to every token, est. 43 to 90 % on top of the rest of the token's work. On the SSD it costs est. 0.2 to 1.3 % of
a token's time.

**Why.** Latency, not bandwidth, decides a random read. This spinning disk waited 10.4 ms per read on average. So the table goes
on the SSD, and I cleared room for it by moving everything else to the USB disk. Disk size never rules a model out;
disk latency can decide where each piece goes.

## What I set up

The full guide is [setup.md](../setup.md). It set up a second, identical rig in one afternoon. The short version:

| Step | What and why |
|---|---|
| Ubuntu 24.04 | A desktop install works. The desktop holds only 4 MiB per card |
| NVIDIA driver 580 | The last driver for Pascal. No CUDA toolkit is needed just to run the engine |
| Check against the spec | A read-only script lists clocks, power limits, link speeds and firmware. Its output carries no serials, so it can be shared |
| Strip the install | A script removes every snap and the unneeded packages (128 on the first rig, 103 on the second), and turns off automatic upgrades |
| Hold the NVIDIA packages | A background driver upgrade breaks CUDA with a "version mismatch" until the next reboot |
| A model drive | An ext4 USB drive, mounted by label with `nofail`, so a missing drive never stops a boot |
| Test the network both ways | On one setup, a second router dropped traffic in one direction. `ping` and downloads did not show it; a two-way throughput test did |
| The engine and a check | The release check runs a model and reports any problem with the cards, memory or drive |

One detail shapes daily work: a reboot needs a passphrase typed at the rig's own keyboard, because the disk is
encrypted. So I never reboot a rig remotely, and any risky test waits until someone can sit at the console.

## Hardware tuning: free settings, small results

Miners tune their cards: lower power, higher memory clock. I checked each free setting for what it could give a
language model.

**Power limits.** While generating text with the model split over the cards, each card drew only 35 to 39 W of its
75 W. On nine cards the power cap never showed (0 of 4 decode phases); on six it showed in at most 3.6 % of a phase's
samples. The nine cards together drew about 320 W, 8.9 joules per token at 35.9 tok/s. Raising the limit is impossible, since 75 W is the maximum, and lowering it saves nothing here.

**"Prefer maximum performance".** The driver has a mode that is supposed to keep clocks high. It gained nothing and
was slightly slower in all three tests: 29.79 against 29.75 tok/s from idle, 35.07 against 34.31 after a 2,048-token
prompt and 33.30 against 33.25 served.

**Idle clocks.** This one mattered. When a model is split over nine cards, each card works for a short slice of every
token and waits the rest. Waiting cards drop from 1,867 MHz to their 1,354 MHz base clock. The same prompt ran
at 38.5 tok/s with the cards at full clock and 34.0 at base clock. No setting holds the clock on these cards. The only
cure is to keep them busy. Flint's release settings keep the cards at full clock 81 % of the time, because the idle
cards work on guessed tokens ([chapter 8](08-guessing-ahead.md)). Ornith-1.5-122B on the v2.0 engine, which decodes one
token at a time, sat at base clock half the time.

**Memory overclock.** This is the one setting that should raise generation speed. My estimate is +3–5 % at +10 % memory
clock, because only est. 55 % of a token's time (14.9 of 26.8 ms) is spent streaming weights. The cards have no error-correcting
memory, so "it didn't crash" proves nothing: a weak memory cell silently corrupts weights for the whole session. The
test is bit-exact output instead. With stable memory, the engine's output is identical to the stock clock's, token for
token. The test is planned and approved, but it needs someone at the console in case a card hangs. It is **open**.

**Why.** Hardware settings give little because the cards are rarely at their limits. They wait for each other, for
the links and for their own small chips. The lesson is the idle-clock result: part of what looks like a software
speed-up is simply keeping the cards busy enough to stay fast.

## What to remember

- The rig's one strength is memory: 54 GB that the cards read fast. Text generation needs exactly that.
- The 5-SM chips use only 34.7 % (Q4_K_M), 42.2 % (Q4_0) and 53.7 % (Q8_0) of their memory bandwidth on real code. Simple formats and good GPU code close that
  gap, not clock settings.
- One PCIe 1.0 lane per card means: split by layers, pass only small vectors, load once.
- The 4 GB host sets the safe flags: a small micro-batch, no prompt cache in RAM, few checkpoints. The rig stays as
  sold, so memory is freed in software.
- Waiting cards drop to a lower clock. Keeping them busy pays twice.
- Free hardware tuning gave almost nothing. The memory overclock is still open.

## Where the details are

- The hardware, measured: [mining_rig.md](../mining_rig.md). Setting up a rig: [setup.md](../setup.md).
- Single-card calibration (Llama-2-7B, effective bandwidth, two and nine cards):
  [benchmark page](../benchmarks/2026-10-22-single-card-calibration.md),
  [data](../data/2026-10-22_e0.8_calibration/README.md).
- Small models, the CPU and the GPU's busy time: [small MoE benchmark](../benchmarks/2026-10-22-small-moe-overhead.md).
- The 4 GB host and `-ub 128`: [Qwen baselines](../benchmarks/2026-10-22-qwen-baselines.md) and
  [setup.md](../setup.md#running-llama-server-on-a-4-gb-host). Flash-Next's host memory in an agent session:
  [host_ram_4gb.md](../research/notes/host_ram_4gb.md). Keeping the rig as sold: registry row Q86 (parked). Headless
  mode: Q83 ([data](../data/2026-11-03_headless/README.md)).
- The lookup table on the SSD: Q214 ([data](../data/2026-11-03_table_storage/README.md)) and Q228
  ([data](../data/2026-11-03_ngram_replay/README.md)).
- Stripping the install: [data](../data/2026-11-03_rig_debloat/README.md). The network fault:
  [data](../data/2026-11-03_rig_network/README.md).
- Hardware tuning: [hardware_tuning.md](../research/notes/hardware_tuning.md); PCIe 2.0 and power, Q86; the power log,
  Q143; "prefer maximum performance", Q141; idle clocks, Q154 ([data](../data/2026-10-27_clock_share/README.md)); the
  memory overclock, Q133 (approved, not run).
- Registry: [quants/experiments.md](../../quants/experiments.md).
