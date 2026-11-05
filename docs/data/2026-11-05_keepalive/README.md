# A keep-alive that holds the cards' boost clock (C533, C558, C559; 2026-11-05)

What this is: a keep-alive for the P106 cards, in Miner Llama (`GGML_CUDA_KEEPALIVE`) and in
[bench/persistent_pipe](../../../bench/persistent_pipe/README.md) (`--keepalive`): its code, its CPU checks, its first
run on a rig (the bench, then the engine's T(k) and plain decoding), and the answer to why two cards of the release line
already hold their clock (C559). The agent check is still to come; its command is below. Registry rows Q1021 (the
keep-alive), Q1045 (holding the cards between agent turns), Q1046 (cards 4 and 8), Q1065 (the pulse in the engine),
Q1066 (the back-off) and Q1075–Q1077 in [quants/experiments.md](../../../quants/experiments.md).

Why: on the release line the driver runs every card at its 1,354 MHz base clock while Flash-Next decodes, because each
of the nine cards computes only ~1/9 of a token; a busy card runs at up to 1,911 MHz
([2026-11-05_card_clocks](../2026-11-05_card_clocks/README.md)). Holding boost is estimated at 5.5–9 ms a token on the
served decode (−8 to −14 %).

## The result on a rig (C533, 2026-11-05)

**With `GGML_CUDA_KEEPALIVE=decode` every card decodes at boost, and decoding is 14–19 % faster: 51.1 ms a token
against 59.6–62.8 ms (19.6 against 15.9–16.8 tok/s).** That is the release line without the draft (pmix-q2-fast, the
k-column settings, the Q8_0 hop cast through the staged hop), the `-keepalive` build with the switch off and on, two
loads each, alternated, on one of the 4 GB rigs. llama-bench, 5 repetitions after a warm-up ([tk.tsv](tk.tsv)):

| | off, load 1 | off, load 2 | `decode`, load 1 | `decode`, load 2 | `decode` against off |
|---|---:|---:|---:|---:|---|
| tg128, ms a token | 62.75 ± 0.11 | 59.62 ± 0.02 | 51.06 ± 0.05 | 51.08 ± 0.04 | −8.5 to −11.7 ms (−14 to −19 %) |
| tg128, tok/s | 15.94 | 16.77 | 19.58 | 19.58 | +17 to +23 % |
| T(1), ms a pass | 66.55 ± 4.32 | 62.99 ± 3.22 | 65.27 ± 5.04 | 62.74 ± 4.21 | none: the cards are still ramping up (below) |
| T(2) | 94.48 ± 2.23 | 89.62 ± 1.72 | 82.72 ± 4.72 | 79.72 ± 4.26 | −6.9 to −14.8 ms (−8 to −16 %) |
| T(3) | 116.28 ± 3.14 | 109.88 ± 2.80 | 95.57 ± 2.81 | 90.68 ± 2.88 | −14.3 to −25.6 ms (−13 to −22 %) |
| T(4) | 131.48 ± 3.61 | 124.08 ± 2.96 | 101.17 ± 3.05 | 101.11 ± 2.97 | −23.0 to −30.4 ms (−19 to −23 %) |
| T(5) | 153.88 ± 3.78 | 146.28 ± 2.68 | 118.18 ± 2.76 | 118.29 ± 2.81 | −28.0 to −35.7 ms (−19 to −23 %) |

The last column runs from the closest to the farthest pair of loads; at 5 tokens a pass that is 23.6–23.7 ms a token
against 29.3–30.8.

- **The clocks** ([tk_clocks.tsv](tk_clocks.tsv), `nvidia-smi` every 2 s beside each load, per card and phase): while
  tg128 runs, off, every card sits at the 1,354 MHz base clock (19–20 of its 20–21 samples, the other one between);
  with `decode` every card runs at 1,873–1,911 MHz in all 17 of its samples. Power: 27.3–29.8 W a card off (252 W for
  the nine), 34.8–39.8 W with the keep-alive (323–327 W): +72 W for 17–23 % more tokens a second, so a token takes
  16.5–16.7 J against 15.1–15.8 J (+5 to +11 %).
- **The blocks' own clock log** (one line per card at each test's end, from 100 ms samples while the block runs): in
  T(1) the cards' medians are 1,325–1,475 MHz (card 8 1,925) and none of 0–7 reaches 1,800; in T(2) 1,675–1,875 MHz;
  from T(3) to tg128 1,875–1,925 MHz, every sample at 1,800 or more, on every card. T(1) is llama-bench's first test
  after the model load and lasts under half a second, so each card's block runs only ~0.3 s of it, while the card is
  still ramping up from base. tg128 (one token a pass, over 128 tokens) is the k = 1 result; Q1077 repeats T(1) after a
  warm-up.
- **The switch's mechanics:** the smoke (`-p 0 -n 16`) logged a line per card (3 runs each, all ended by the host).
  Of all 117 block reports none ended by its timeout. The off loads differ from each other by 3.1 ms a token (the
  driver's clock choice), the on loads by 0.02 ms.
- **Against C532's estimate:** plain decoding, est. ~7 ms a token, measured 8.5–11.7. The served MTP decode, est.
  5.5–9 ms a token: T(4)'s 23–30 ms a pass scaled to the 41 of 49 layer shares that sit at base on that line (the
  draft's two cards already hold their clock), at 2.5 tokens a pass, gives 7.7–10.2 ms a token.

**Classes** (CLAUDE.md's gain, negligible, loss; against the spread of repeated loads):

| Setting | Class | Why |
|---|---|---|
| `GGML_CUDA_KEEPALIVE=decode` | **gain** | −8.5 to −11.7 ms a token on plain decoding, −19 to −23 % on 4- and 5-token passes, far beyond the spread (3.1 ms between the off loads, 0.02 between the on loads); +8 W a decoding card |
| `request`, `idle` | not measured yet | they differ from `decode` only while llama-server takes a request or idles between requests; llama-bench does neither (each test frees its context, which ends the blocks). Their arms in the agent check decide (below) |
| bench `--keepalive resident` against off | **gain** | −12 to −15 % a pass at 1 token, −18 to −23 % at 4 |
| bench `--keepalive pulse` against off | **gain** | −14 to −16 % at 1 token, −19 to −23 % at 4 |
| bench `pulse` against `resident` | **gain** | −0.75 ms a pass at 1 token (−1.6 %; three runs of each, which agree within 0.005 ms), −0.43 ms at 4 (−0.5 %; one run each, their p10–p90 apart) |

### The bench: resident against pulse

`bench/persistent_pipe` on the nine cards with C417's stand-in products (4 a card, 210 MB of FP16 weights each), the
residual in Q8_0 through the staged hop (the release line's), 1,000 timed passes at 1 token and 250 at 4. Runs 2 and 3
idle 10 s before each step and load every kernel at the start (`CUDA_MODULE_LOADING=EAGER`, below). Median ms a pass
([bench.tsv](bench.tsv); per card and step [bench_cards.tsv](bench_cards.tsv), `nvidia-smi` every 500 ms from 10 s
into each step):

| Keep-alive | 1 token a pass | 4 tokens a pass | The cards while the passes run | W a card |
|---|---|---|---|---:|
| off | 53.75 (run 1), 52.28 (run 2) | 121.21 (run 1), 115.10 (run 2) | base; in run 2 some cards between (1,430–1,493 MHz) | 29 |
| `resident` (one thread a card, its word read every 50 µs) | 45.879, 45.878, 45.881 | 93.91 | 1,875 MHz on every card, every sample | 38–39 |
| `pulse` (busy 100 ms every 1,000 ms) | 45.133, 45.129, 45.128 | 93.48 | median 1,875 MHz on every card; 87–98 % of the samples at boost at 1 token, 100 % at 4 | 38–39 |

- **Both hold all nine cards at boost; the pulse costs the products less.** Each card's products alone, right after
  the keep-alive stopped (the card still at its clock): 4.90–4.94 ms (44.24–44.33 ms for the nine) against
  5.79–5.91 ms at base (52.63): −16 %, as C417 found. With the pulse a pass is those products plus 0.84 ms of hops
  (~105 µs a hop); the resident thread adds 0.75 ms to that, from its own work on the card (a read of mapped host
  memory every 50 µs, and one thread resident on an SM); Q1066's longer back-offs show which.
- **The stop:** a resident thread ends 158–195 µs after its word is cleared; the pulse's stop waits for the current
  spin (up to 17 ms seen, at most its 100 ms).
- The "products with the keep-alive" column of [bench.tsv](bench.tsv) is not the thread's cost: its calibration starts
  as the cards leave base (after the 10 s of idle), so it includes their ramp. Q1076 warms the cards first.
- Run 2 named its repeated steps alike, so its first `kaoff`, `karesident` and `kapulse` were overwritten by the
  second ones (`-2`; the clocks of both are in `bench_cards.tsv`); `run_rig.sh` now numbers repeats.

### Lazy module loading and a resident kernel

Run 1 used CUDA 12's default lazy module loading, under which the first launch of a kernel that is not loaded yet waits
until every kernel running on its card has ended. The first launch of a kernel the calibration had not used (the Q8_0
cast and uncast, the argmax) waited for the resident thread's own safety timeout (600 s in the calibration, 200 s in
the staged run). Each thread lived exactly that long, and the timed passes ran after it had ended, at base: 53.64 ms, as
off (`no` in [bench.tsv](bench.tsv)'s last column). The pulse was not affected, since each spin ends within 100 ms.
`run_rig.sh` now sets `CUDA_MODULE_LOADING=EAGER`, and runs 2 and 3 are valid.

In the engine, `decode` starts a card's block after each of its graphs, and the block ends itself 1 s after the card's
last graph. A kernel launched for the first time within that second waits up to 1 s. llama-bench showed none (every
block report: 0 ended by the timeout), because each test's warm-up loads its kernels before its first block starts.
llama-server's first requests may (a prompt's kernels right after a decode, or the reverse): Q1075 checks the agent
check's block reports for it.

## The engine switch: `GGML_CUDA_KEEPALIVE`

Fork branch `pascal-p106-v3-b12076-keepalive` (one commit on the cumulative branch `-ladder4`), off by default.

| Variable | Values | What it does |
|---|---|---|
| `GGML_CUDA_KEEPALIVE` | unset or `0` | off: nothing is launched or allocated |
| | `decode` (or `1`) | a card's block starts after a graph is computed on it and ends itself `GGML_CUDA_KEEPALIVE_MS` after the card's last graph |
| | `request` | `decode`, and when llama-server takes a request, every card that has computed a graph starts its block ahead of the prompt, so that the cards leave their idle state while the prompt is prepared (C558) |
| | `idle` | `decode` with a 120 s default timeout: the cards keep their clock while the server idles between an agent's turns (C558) |
| `GGML_CUDA_KEEPALIVE_MS` | 10–3,600,000 (default 1,000; 120,000 for `idle`) | how long a block runs after its card's last graph |
| `GGML_CUDA_KEEPALIVE_POLL_US` | 1–100,000 (default 50) | the block's back-off between two reads of its word: the longest a stop waits, and one PCIe read per back-off |

How it works:
- **The block:** one thread, launched on a stream of the card's lowest priority, reads a 64-bit word in mapped host
  memory (the run's generation and the card's beat), then spins on `clock64()` for the back-off. It ends when the host
  writes another generation, or when the beat has not changed for the timeout. While it runs, the driver counts the
  card as busy.
- **When it starts:** after every graph computed on the card, the host writes a new beat; a card whose block is not
  running gets a new one. A beat after a gap of about the timeout ends the old block and starts a fresh one, and every
  250 ms a beat also asks the stream whether the block has ended, so a block that ended unseen is replaced.
- **When it stops:** a resident kernel would make anything that waits for a whole device wait forever, so the host ends
  the card's block before a CUDA graph capture begins on it, and every card's block before a free (`cudaFree`, the
  pools, buffers, hop rings, mailboxes), a pinned host allocation or registration, a VMM unmap and the pool's device
  synchronize. Graph replays, peer copies and `cudaMemGetInfo` run beside it. The next graph starts the block again.
  The timeout bounds the wait of any synchronizing call the controller is not told about.
- **What it reports:** each block measures its card's SM clock every 100 ms (its `clock64()` against `%globaltimer`).
  When a context is freed, the log has a line per card: the runs, how they ended, and the clock's p10, median and p90
  with the share below 1,400 MHz and at 1,800 MHz or more.
- The settings, the block's decision at each poll and the start/stop controller are plain C++
  (`ggml/src/ggml-cuda/keepalive-policy.h`), so `tests/test-cuda-keepalive.cpp` checks them on a CPU, with a thread
  standing in for each card's block.

## The bench: `bench/persistent_pipe --keepalive`

`--keepalive resident` runs the same thread per card around the calibration and every host-driven variant (the word
polled every `--keepalive-poll-us`, default 50); `--keepalive pulse` launches a short spin on every card every period
instead (`--keepalive-pulse ON,PER`, default 100 ms every 1,000 ms), since a card keeps boost for a few seconds after a
busy spell. Each thread measures the clock every 50 ms, and the run prints per card and phase the clock and how long the
stop took. The calibration runs twice, with the keep-alive and right after it stopped, so the difference is what the
thread costs the products. `run_rig.sh`'s steps `kaoff karesident kapulse kak4off kak4` are the rig run.

## C559: what keeps cards 4 and 8 in P0 while the server idles

**Not a keep-alive the engine has.** Nothing in the MTP draft's path or the CUDA backend runs between requests: the
draft (`common_speculative_impl_draft_mtp`) launches work only inside a request's drafting passes and its catch-up
after a verify, and the server's idle loop waits on its task queue. The CUDA backend has no resident or repeating kernel
on S35's line. The persistent MoE kernel is Flint-shaped and ends with each layer, and the mailbox hop is off. The
model loader synchronizes and frees its pinned buffers and events at the end of each load. The staged hop's events
complete with each copy, and device sampling (`-bs`) runs per token.

Every change of each card's P-state, clock and clock-event reasons in C532's log ([transitions.tsv](transitions.tsv),
seconds from the server's start) shows that the two cards differ:

- **Card 4 is not held.** After its last work a card stays in P0 for ~19 s before the driver marks it idle (card 2:
  the target's warm-up at 283.5 s, the idle reason at 302.4 s, P8 at 305.3 s). Card 4's last work was the draft's
  set-up at ~306 s (its token table was uploaded at 287–294 s), 11.7 s before the request (317.6 s). Its P0 at base
  through the 10 s idle window is that normal hold; the window was too short to see it end.
- **Card 8 is held busy without work.** It reached 1,885 MHz while the draft's layer was uploaded (293 s) and stayed
  there, never marked idle, through the 11.7 s before the request, and through the 9 s from the reply (367.4 s) to the
  log's end, which came 3 s after the server process had ended. A card whose work stops drops from boost to base
  within a second in the same log (card 4 at 294.0 s). The driver counts card 8 as busy, and no code path of the
  engine explains it.
- **While decoding, both cards hold a fixed clock:** card 8 1,885 MHz without a change from 345.8 s to the log's end,
  card 4 1,797 MHz from 343.1 s to the server's stop, while the other seven, with similar shares, fell to base. Card 4
  runs only a few rows of the draft's token table a pass beyond its own layers. The two cards share one thing no other
  card has: they are the two ends of the draft's only hop between cards that are not neighbours (the draft's token rows,
  4 → 8, through the staged hop's pinned ring).
- A short rig check decides what holds them: `utilization.gpu` and the clock per card through 60 s of server idle and
  through a decode, with the draft and without, with the draft's table on card 8 (no `-otd`), without device sampling
  (`-bs`), without the staged hop, and after the process ends.

So C533's keep-alive is still needed: nothing in the engine holds the clock that could be copied to the other cards.

## The CPU checks

On the build machine, CPU only (no GPU used): Docker, `--cpu-shares=256`, `-j 8`, under the shared build lock.
Every check passed.

| Check | Verdict | What it compared |
|---|---|---|
| build-rig | OK | the rig tarball `llama.cpp-20261105-f6c1c6a4a-pascal-p106-v3-b12076-keepalive.tar.gz` (605,424,616 B, md5 `d18fccc9f85657f6a5040c83753ceeb1`; `TESTS=1`, sm_61, no AVX, GLIBC_2.38) |
| build-extra | OK | `test-llama-archs` and `test-cuda-keepalive` from the same tree (`extra-f6c1c6a4a.tar.gz`, md5 `a74f341188d355386c7d9a5b00d53412`) |
| cpubuild | OK | the branch without CUDA: `llama-server`, `test-llama-archs`, `test-backend-ops`, `test-cuda-keepalive` |
| identity-rig | OK | `test-llama-archs -b CPU`, the logits of every architecture: -ladder4's rig build writes 136 files, this one compares 143 cases, none differs |
| identity-cpu | OK | the same between the two CPU builds: none differs |
| switchon | OK | the rig build with `GGML_CUDA_KEEPALIVE=decode` against -ladder4's: none differs (without a CUDA device no block starts) |
| unit | OK | `test-cuda-keepalive`, 3 runs: 86 checks, 0 failed each |
| bench | OK | `bench/persistent_pipe` with `--keepalive` (md5 `b527b3a0563cb09d9e7b79751a656d55`; sm_61, compute_61 PTX and sm_86, no AVX, GLIBC_2.38; the two keep-alive kernels 32 and 40 registers, no spills) |

`test-llama-archs` reports 7 of its 148 tests failed in -ladder4 and in this branch alike (registry Q1050). The unit test
covers the settings, the poll decision (the stop winning over a beat and a timeout, a late beat resetting the timeout,
the beat wrapping), the clock statistics, the switch off (nothing launched or written), `decode` (one run across 20
graphs, the block following the beats, a capture ending it, the next graph starting another, a free ending every
card's), the timeout (a block that ends itself is replaced at the next graph), beats every 5 ms holding one run past the
timeout for 1.5 s, the 250 ms check finding a block that ended unseen, `request` (only cards that have computed a graph
start, a second kick only beats), `idle` (the block outlives a 200 ms gap), and graphs, captures and frees from three
threads at once with every run accounted for. The CUDA side (the kernel, the stream, the mapped words) is checked by the
rig run above.

## The rig run: what ran, and the agent check to come

On a rig with nothing else running, driven over ssh from a machine that runs nothing else, the bench binary built above
copied from the build machine (~30 min in all with the repeats):

```bash
# the bench (run 1 used the defaults then: lazy loading, no pause; runs 2 and 3 as below)
RIG=<rig> FROM='<ssh host of the build machine>' BIN=<build machine path>/bench/persistent_pipe/build/persistent_pipe \
  MD5_BIN=<its md5> PAUSE=10 STEPS='kaoff karesident kapulse kak4off kak4 kak4pulse kaoff karesident kapulse' \
  OUT=<local folder> bash docs/data/2026-11-04_persistent_pipe/run/rig_run.sh
RIG=<rig> ... PAUSE=10 STEPS='karesident kapulse karesident' OUT=<another folder> bash docs/data/2026-11-04_persistent_pipe/run/rig_run.sh
# on the rig, after the runs
SKIP=10 bash docs/data/2026-11-05_keepalive/run/bench_cards.sh <OUT>/out/smi.csv <OUT>/out/run.log <run> > bench_cards.tsv
```

Then the engine on S35's line without the draft, with the `-keepalive` tarball unpacked in `/opt/llama-builds` (the
smoke and T(k), ~21 min), and the tables on the rig:

```bash
ENVFILE=<machine file> MODEL=<pmix-q2-fast first shard on the rig> BUILD=<the -keepalive build's folder> \
  OUT=<local folder> STEPS='smoke tk' LOADS=2 SMI_MS=2000 bash docs/data/2026-11-05_keepalive/run/rig_keepalive.sh
bash docs/data/2026-11-05_keepalive/run/tk_tables.sh speed <OUT> > tk.tsv
bash docs/data/2026-11-05_keepalive/run/tk_tables.sh clocks <OUT> > tk_clocks.tsv
bash docs/data/2026-11-05_keepalive/run/tk_tables.sh blocks <OUT>    # the blocks' own clock lines
```

**The agent check (to come, ~80 min):** the arms `s35kaoff` (the same build and line, switch off: the control on the
same rig) and `s35ka` (`GGML_CUDA_KEEPALIVE=decode`), against each other and against S35's `s35stack` (18.8 tok/s,
time to first token median 7.42 s, on the other rig); `s35kareq` and `s35kaidle` add C558's two settings (+80 min):

```bash
ENVFILE=<machine file> RIG_PORT=8192 RELAY_PORT=8199 CLIENT_PORT=18493 PROXY_PORT=18494 \
  CLIENT_DIR=<client folder> CLIENT_OUT=<client results folder> \
  MODEL=<pmix-q2-fast first shard on the rig> BUILD=<the -keepalive build's folder> OUT=<local folder> \
  STEPS='prep arms report' ARMS_KA='s35kaoff s35ka' bash docs/data/2026-11-05_keepalive/run/rig_keepalive.sh
```

## Files

| File | What it is |
|---|---|
| [tk.tsv](tk.tsv) | the engine: T(k) at k = 1–5 (ms a pass) and tg128 (ms a token) per load, the switch off and `decode`, mean and standard deviation of 5 repetitions |
| [tk_clocks.tsv](tk_clocks.tsv) | the engine: per load, phase (the model load, the T(k) tests, tg128) and card, the `nvidia-smi` samples in P8, at base, between and at boost, the median SM clock and the mean power |
| [bench.tsv](bench.tsv) | the bench: every run's median, p10, p90 and mean ms a pass, the products' calibration with the keep-alive and right after it, the stop's time, and whether the run is valid |
| [bench_cards.tsv](bench_cards.tsv) | the bench: per run, step and card, the `nvidia-smi` samples (every 500 ms, from 10 s into the step) in P8, at base, between and at boost, the median SM clock and the mean power |
| [transitions.tsv](transitions.tsv) | C559: every change of a card's P-state, SM clock, power and clock-event reasons in C532's log, from the server's start to the log's end, with the run's marks |
| [run/transitions.sh](run/transitions.sh) | builds it from C532's `nvidia-smi` log and marks |
| [run/checks.sh](run/checks.sh) | the build and the CPU checks on the build machine (calls the -ladder4 merge's [checks.sh](../2026-11-04_ladder4_merge/run/checks.sh) for the build, the CPU build and the identity) |
| [run/rig_keepalive.sh](run/rig_keepalive.sh) | the engine's rig run: smoke, T(k) (with `SMI_MS`, a sparse clock log beside each load), the agent check |
| [run/tk_tables.sh](run/tk_tables.sh) | `tk.tsv`, `tk_clocks.tsv` and the blocks' lines from the engine run's files |
| [run/bench_cards.sh](run/bench_cards.sh) | `bench_cards.tsv` from the bench's `nvidia-smi` log and step log |

## Commands

On the build machine (CPU only), with the fork cloned in `$WORK/src` (remote `fork`), -ladder4's rig tarball unpacked
with its extra tests in `bin/`, and a static native CPU build of -ladder4:

```bash
WORK=<work dir> REPO=<this repository> BASE_BUILD=<-ladder4's unpacked tarball> BASE_CPU=<-ladder4's CPU bin/> \
  JOBS=8 LOCK=<shared build lock> DOCKER_ARGS='--label <the machine's label>' \
  bash docs/data/2026-11-05_keepalive/run/checks.sh
bash docs/data/2026-11-05_keepalive/run/transitions.sh <C532's smi.csv.gz> <C532's marks.tsv> > transitions.tsv
```

Provenance: the code is Miner Llama's (MIT, as llama.cpp) and this repository's; `transitions.tsv` is derived from the
`nvidia-smi` log of [2026-11-05_card_clocks](../2026-11-05_card_clocks/README.md); the other tables come from this
run's llama-bench, `persistent_pipe` and `nvidia-smi` output on the rig. The model is Swift's Qwen3.8-Flash-Next
GSQ-RCO Q2_0 as retyped by [the Pascal mixes](../2026-11-04_pascal_mix/README.md) (pmix-q2-fast).
