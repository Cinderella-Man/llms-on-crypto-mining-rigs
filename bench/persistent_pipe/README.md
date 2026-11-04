# persistent_pipe: a token-persistent pipeline across the rig's cards, against the host-driven loop

When the nine-P106 rig decodes with a layer split, each token runs through the cards one after another: card 0's
layers, a hop through the host to card 1, card 1's layers, and so on to card 8, which picks the token; then the host
reads the token and starts the next one on card 0. The cards have no peer access, so each hop's residual crosses two
PCIe 1.0 x1 links through host memory, and the 2-core Celeron host drives every launch, copy and synchronize. A
token-persistent pipeline takes the host out of that loop: each card runs one kernel that stays resident for the whole
generation, polls its input mailbox in mapped host memory, runs its layers, and writes the residual into the next card's
mailbox; the last card picks the token and mails it back to card 0. The host only starts the kernels and reads the
results.

Building that inside llama.cpp means a per-card megakernel of every Flash-Next operation (registry Q670) plus the
mailboxes (Q682). This benchmark measures what it could save first, with a stand-in workload, on the rig's own cards,
links and host: the upper bound of what such an executor saves per token. Registry row Q684 in
[quants/experiments.md](../../quants/experiments.md); the design notes are
[megakernel.md](../../docs/research/notes/megakernel.md) §4 and
[flash_next_floor_and_engine.md](../../docs/research/notes/flash_next_floor_and_engine.md) (option (b), the executor).

## The stand-in workload

| Part | What it is | Default |
|---|---|---|
| Residual | n values per token, K tokens per pass (K = 1: decoding; 2, 4, 8: a verify pass's width) | n = 10,240: Flash-Next's four hyper-connection streams of 2,560 |
| A card's share | L matrix-vector products y = x / 2 + tanh(W x) / 2 over one FP16 weight matrix of n × n (210 MB at n = 10,240, read in full by every product; product s shifts the rows by s × 1,031) | L = 4, est. ~5.6 ms on a P106 (`--card-ms` picks L for a target time) |
| Card 0's input | an embedding of each token id: a hash of the id, the pass and the position | |
| The last card's output | the token ids: the argmax of each column of the residual | |
| The hop | the residual in F32 (40,960 bytes a token) or Q8_0 (ggml's `block_q8_0` and its rounding, 10,880 bytes), as the fork's `LLAMA_HOP_CAST=q8_0` sends it | F32 |

Every pass depends on the one before it (the ids feed the next pass), as decoding does, so the cards run one after
another and a pass's time is the cards' work plus every hop.

## The variants

| Variant | How a pass runs |
|---|---|
| `sync` | the host launches each card's products, then the hop: a copy into pinned host memory, a host synchronize, a copy onto the next card; at the end it reads the ids back and synchronizes |
| `peer` | every hop is one `cudaMemcpyPeerAsync` (the driver stages it through the host without peer access), with an event that the next card's stream waits on (ggml's copy between two CUDA backends); one host synchronize a pass, on the ids |
| `staged` | every hop is the fork's manual staged hop (`GGML_CUDA_HOP_STAGED=1`, [card hops](../../docs/data/2026-11-04_card_hops/README.md)): `--staged-chunks` chunks (default 2, the fork's default) through pinned slots, each chunk's two copies on the two cards' streams behind an event, so one chunk's second leg overlaps the next chunk's first; one synchronize a pass |
| `persist` | one cooperative kernel per card for every pass. Card i's block 0 polls card i's mailbox in mapped pinned host memory (slowly until the sender's hint that its last product has started, then every `--poll-us`), one sequence flag per chunk (`--mail-chunks`, default 4); each chunk's reader block copies it into the card's memory in 16-byte loads as soon as its flag is up; the products run behind a grid barrier; after the last product one block writes the residual into the next card's mailbox chunk by chunk in 16-byte stores (Q8_0 cast in shared memory first), raising each chunk's flag. In a mailbox a Q8_0 chunk is its column's `block_q8_0` run padded to 16 bytes (64 bytes more than the 10,880 a token at 4 chunks). The last card's argmax goes to card 0's mailbox and to the results in host memory |
| `persist-direct` | the same, but the last product writes its rows straight into the next card's mailbox while it runs, and the block that completes a chunk raises its flag, so the bytes cross the sender's link while it still computes |

Every variant runs the same device functions on the same blocks of work (one warp per 4 rows, a fixed reduction
order; the casts and the argmax shared), so every variant's token ids and final residual must be bit-identical to the
first variant's. The run prints `identical` per variant and a checksum (FNV-1a of the ids and the residual's bits), and
exits with code 2 when any variant differs or a mailbox wait times out (`--timeout-s`; the host's own watchdog also
stops the kernels when a run takes three times its estimate).

## What it reports

Per variant: the time of a pass (median, p10, p90, mean, over `--passes` after `--warmup`), tokens per second, the cards'
compute, the overhead (pass time minus compute) and the overhead per hop (a pass has N hops: N − 1 residual hops and the
ids back to card 0).

- **Compute**, host-driven variants: each card's L products alone, launched back to back and timed with events before
  the runs (`the products alone` line). Persistent variants: per pass and card, from the input in place to the last
  product done, on the card's own `%globaltimer`.
- **Pass time**, host-driven: the host clock between two passes' ends. Persistent: the last card's timer between two
  passes' ids posted.
- **Per boundary** (persistent only): from card i's last product done to card i + 1's input in. The cards' timers
  differ (two RTX 3090s in one machine: ~174 ms), so before and after each persistent run every pair of neighbours
  measures its offset with a ping-pong through mapped host memory (as NTP does: 200 rounds, the median), and each pass
  is aligned with the offsets interpolated between the two; the run prints the offsets and how far they moved.
- **The saving**: each persistent variant against each host-driven one, ms a pass, %, and µs a hop.

Not modelled: the host's own per-token work in llama.cpp (graph build or update, sampling, the server), which a
persistent pipeline also takes off the per-token path; the launch gaps between a real card's ~250–340 small kernels
(Q670's census bounds them at ≤ ~7 ms a token, ≤ 2.3 ms after barriers); CUDA graphs (each card's products here are 4
large launches, so the host enqueues a pass far ahead of the cards).

## Building

On a machine with Docker, from the repository root:

```bash
DOCKER_ARGS='--cpu-shares=256' bench/persistent_pipe/build.sh
```

The binary holds sm_61 code (the P106), compute_61 PTX (runs the Pascal path on newer cards through
`CUDA_FORCE_PTX_JIT=1`, for correctness only) and sm_86 code; host code without AVX for the rig's Celeron, the CUDA
runtime linked statically. `build/BUILDINFO.txt` records the toolkit and the md5, `build/ptxas.log` the registers.

## Running on a rig

With nothing else on the cards, ~15 minutes on nine cards (each card holds 210 MB of weights):

```bash
RIG=${RIG:-rig}
ssh "$RIG" 'mkdir -p /tmp/pp'
scp bench/persistent_pipe/build/persistent_pipe bench/persistent_pipe/run_rig.sh "$RIG":/tmp/pp/
ssh "$RIG" 'cd /tmp/pp && setsid nohup ./run_rig.sh > console.log 2>&1 < /dev/null &'
ssh "$RIG" 'test -e /tmp/pp/out/DONE && cat /tmp/pp/out/run.log'
scp -r "$RIG":/tmp/pp/out ./persistent_pipe_out
ssh "$RIG" 'rm -rf /tmp/pp'
```

`run_rig.sh` refuses to start while a compute process is on the cards (`FORCE=1` overrides), logs `nvidia-smi` every
500 ms into `smi.csv`, and runs its steps, each into `<step>.txt` and `<step>.csv`: `smoke` (two cards, every path
once), `f32` and `q8` (every card, every variant, 1,000 passes), `two` (two cards), `k4` (4 tokens a pass, Q8_0, 250
passes) and `polls` (the persistent variants with polls every 0.5 and 8 µs). Env: `OUT`, `BIN`, `CARDS`, `STEPS`,
`EXTRA` (options added to every run), `FORCE`.

## Options

| Option | Meaning |
|---|---|
| `--cards N`, `--devices LIST` | the first N visible devices, or these devices in pipeline order (2–16) |
| `--n N`, `--k K` | residual values per token (a multiple of 256, ≤ 16,384), tokens per pass (1, 2, 4, 8) |
| `--layers L`, `--card-ms T` | products per card, or as many as come closest to T ms on the first card |
| `--passes P`, `--warmup W`, `--smoke` | timed and untimed passes (default 1,000 and 20); `--smoke`: 20 after 2 |
| `--wire f32\|q8` | the residual between cards |
| `--variants LIST` | `sync,peer,staged,persist,persist-direct` (default all; the first is the reference of the checks) |
| `--staged-chunks C`, `--mail-chunks C` | chunks of the staged hop (default 2) and of a mailbox (default 4), 1–16 |
| `--grid G` | blocks of each persistent kernel (default: as many as the card holds at once) |
| `--poll-us X`, `--slow-poll-us X` | time between two polls of a flag in host memory after the sender's hint (default 2) and before it (default 50) |
| `--timeout-s S` | a mailbox wait longer than this stops every card (default 10) |
| `--sched auto\|spin\|yield\|blocking` | how the host thread waits on the devices (`cudaInitDevice` flags; default auto, as llama.cpp leaves it) |
| `--seed S`, `--csv FILE` | the first ids; `variant,cards,k,wire,layers,key,value,unit` |

## What decides what

- **The saving a pass** of `persist-direct` against `staged` (the release line's hop, with `--wire q8`) is what a
  token-persistent executor could save per token on this rig at most, beyond its per-card megakernel. Divided by the
  pass, it is the share of decode speed at stake. Q684 is worth building past Q670's stages only if this is large
  against the ~120 ms of a Flash-Next token on the release line; the k = 4 run says the same for verify passes.
- **Per hop**, `persist` against `persist-direct`: what writing the residual while the last product runs adds over
  writing it after.
- **The polls**: how much of a persistent hop is the poll spacing, and whether idle cards' polling slows the active
  hop on the shared switch links.

## Files

| File | What |
|---|---|
| `src/main.cu` | the benchmark |
| `build.sh` | the Docker build |
| `run_rig.sh` | the rig run: quiet check, `nvidia-smi` log, the steps |
