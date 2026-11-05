# A keep-alive that holds the cards' boost clock (C533, C558, C559; 2026-11-05)

What this is: the code and the CPU checks of a keep-alive for the P106 cards, in Miner Llama (`GGML_CUDA_KEEPALIVE`) and
in [bench/persistent_pipe](../../../bench/persistent_pipe/README.md) (`--keepalive`), and the answer to why two cards
of the release line already hold their clock (C559). The rig run that measures what the keep-alive gains is the next
step; its commands are below. Registry rows Q1021 (the keep-alive), Q1045 (holding the cards between agent turns) and
Q1046 (cards 4 and 8) in [quants/experiments.md](../../../quants/experiments.md).

Why: on the release line the driver runs every card at its 1,354 MHz base clock while Flash-Next decodes, because each
of the nine cards computes only ~1/9 of a token; a busy card runs at up to 1,911 MHz
([2026-11-05_card_clocks](../2026-11-05_card_clocks/README.md)). Holding boost is estimated at 5.5–9 ms a token on the
served decode (−8 to −14 %).

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
threads at once with every run accounted for. The CUDA side (the kernel, the stream, the mapped words) runs first on the
rig.

## The rig run (next)

The bench first (~5 minutes, no model, either rig idle), from the repository root with the binary built above:

```bash
RIG=<rig> FROM='<ssh host of the build machine>' BIN=<build machine path>/bench/persistent_pipe/build/persistent_pipe \
  MD5_BIN=<its md5> STEPS='kaoff karesident kapulse kak4off kak4' OUT=<local folder> \
  bash docs/data/2026-11-04_persistent_pipe/run/rig_run.sh
bash docs/data/2026-11-04_persistent_pipe/run/clock_table.sh <OUT>/out/smi.csv <OUT>/out/run.log > clocks_10s.tsv
```

It answers whether the resident thread holds every card at boost, what it costs the products on 5 SMs (the calibration
with the thread against right after it stopped, and C417's 4.90 ms a card at boost and ~5.85 ms at base) and what it
draws. Then the engine on S35's line, with the `-keepalive` tarball unpacked in `/opt/llama-builds` (smoke, T(k) at
k = 1–5 and tg128 with the switch off and on alternated over two loads each, then the agent check with
`GGML_CUDA_KEEPALIVE=decode` against S35's `s35stack`, ~1.5 h):

```bash
ENVFILE=<machine file> RIG_PORT=8192 RELAY_PORT=8199 CLIENT_PORT=18493 PROXY_PORT=18494 \
  CLIENT_DIR=<client folder> CLIENT_OUT=<client results folder> \
  MODEL=<pmix-q2-fast first shard on the rig> BUILD=<the -keepalive build's folder> OUT=<local folder> \
  bash docs/data/2026-11-05_keepalive/run/rig_keepalive.sh
```

`ARMS_KA="s35ka s35kareq s35kaidle"` adds C558's two settings as arms.

## Files

| File | What it is |
|---|---|
| [transitions.tsv](transitions.tsv) | C559: every change of a card's P-state, SM clock, power and clock-event reasons in C532's log, from the server's start to the log's end, with the run's marks |
| [run/transitions.sh](run/transitions.sh) | builds it from C532's `nvidia-smi` log and marks |
| [run/checks.sh](run/checks.sh) | the build and the CPU checks on the build machine (calls the -ladder4 merge's [checks.sh](../2026-11-04_ladder4_merge/run/checks.sh) for the build, the CPU build and the identity) |
| [run/rig_keepalive.sh](run/rig_keepalive.sh) | the engine's rig run: smoke, T(k), the agent check |

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
`nvidia-smi` log of [2026-11-05_card_clocks](../2026-11-05_card_clocks/README.md).
