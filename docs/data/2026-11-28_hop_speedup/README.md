# Card-to-card hops on a P106 mining rig: what they cost, and every software path tried (2026-11-28)

What this measures: how long Qwen3.8-Flash-Next's hand-offs between cards take on nine P106-090 cards, where that time
goes, and whether any software path moves them faster than the v5 preset does. Everything here runs at stock clocks
with the PCIe links as the board trains them (no overclocking, no link-speed change). Registry rows
[Q1942](../../../quants/experiments.md) (the bytes lane), [Q682](../../../quants/experiments.md) (the mailbox),
[Q1943](../../../quants/experiments.md) (the staged ring's settings), [Q1941](../../../quants/experiments.md) (the host
writing into a card's video memory) and [Q752](../../../quants/experiments.md) (stream memory operations). Background:
[card hops](../../research/notes/card_hops_x1.md) and [the DIY interconnect note](../../research/notes/diy_card_interconnect.md).

## The problem, for a reader new to this rig

- **The board:** nine P106-090 cards (GP106, 6 GB, compute 6.1), each on a PCIe 1.0 x1 link. Cards 0–3 sit behind one
  ASMedia ASM1184e switch, cards 5–8 behind a second, card 4 alone on a chipset root port; the host is a two-core
  Celeron 3865U with 4 GB of RAM ([mining_rig.md](../../mining_rig.md)).
- **No card reaches another card's memory.** The switches pass a card-to-card write up to the chipset, which refuses it,
  and the chipset drops traffic between its own ports. CUDA peer-to-peer (also forced), a Vulkan dma-buf share
  ([DRMShare](https://github.com/Cinderella-Man/drmshare)) and NCCL's peer transports all fail here
  ([p2p and Gen2 trials](../../research/notes/p106_p2p_gen2_trials.md)).
- **So every hop goes card → host memory → card.** In a layer split one card works at a time; at its last layer it
  hands the residual (Flash-Next: four streams of 2,560 values a token) to the next card. A decode pass makes eight such
  hops, a four-token verify pass carries four tokens in each, and a prompt micro-batch of 256 tokens carries 256.
- **What the v5 preset does today:** it rounds the residual to Q8_0 at each card boundary (`LLAMA_HOP_CAST=q8_0`,
  10,880 B a token) and sends it through a ring of pinned host slots, in 2 chunks per hop
  (`GGML_CUDA_HOP_STAGED=1`): the sending card copies a chunk into a slot, the receiving card copies it out, ordered by
  events. Measured alone (C140): 126.7 µs for one token's hop, about 26 µs fixed plus 10 ns a byte (213 MB/s up, 189
  MB/s down, both near the PCIe 1.0 lane's packet limit).

## What runs (lanes on the deep lane, one rig and one boot each)

The [deep lane](../../../bench/deep_lane/README.md) loads `llama-server` on the preset's line, sends prompts of exactly
0, 8,192 and 32,768 tokens cut from recorded agent text and generates 256 tokens after each (twice at temperature 0,
once with the preset's sampler), for several settings whose loads alternate: a warm-up, then each setting twice, the
second round in reverse order. Per depth it reports each load's decode step (ms), verify pass and prompt reading speed,
and classes every setting against the control: gain, negligible or loss.

| Lane | Rig | Settings (the first is the control) | Question |
|---|---|---|---|
| `bytes` | 1 | `q8` the preset; `f16` F16 hops (20,480 B a token); `f32` no cast (40,960 B); `drv` Q8_0 through the driver's own staged copy | how a hop's cost grows with its bytes in the running engine: link-bound or not, so what a faster link could give |
| `mailbox` | 2 | `q8` the preset; `mbd` the mailbox for hops up to 64 KiB (decode, verify); `mba` the mailbox for every hop; `mbg` the mailbox with its gate (1 MiB buffers) for hops above 64 KiB: a hop larger than a buffer goes to the staged ring, so the gate takes the 64-token prompt micro-batches (696,320 B) and the 256-token ones (2,785,280 B) stay on the ring, as in `mbd` | whether a kernel-driven hop through mapped host memory, streamed in chunks with a flag each, beats the staged ring |
| `ring` | next free | `q8` the preset (2 chunks); `c4`, `c8` 4 and 8 chunks; `s1m` 1 MiB slots | the staged ring's own settings on the release line |
| `timing` | next free | on the hop-paths build: `q8` the preset, timing off; `q8t` the preset with `GGML_CUDA_HOP_TIMING=1`; `f32t` F32 hops with timing; `mbdt` the decode mailbox with timing | each hop's time in the engine per path and size class, and whether the host or the card waits (`q8` against `q8t` is the stamps' own cost) |
| `wc` | next free | on the hop-paths build: `q8` the preset; `wcr` the staged ring's slots write-combined (`GGML_CUDA_HOP_WC=1`); `mbd` the decode mailbox; `mbdw` the same with its slots write-combined (`GGML_CUDA_HOP_WC=2`) | whether write-combined host buffers move a hop faster |

The mailbox (`GGML_CUDA_HOP_MAILBOX`, Q682) puts one kernel on each card: the sender streams the bytes into a ring of
pinned host memory mapped into both cards and sets a sequence flag after each chunk; the receiver polls the flag and
reads each chunk as it lands, so the second leg starts after the first chunk instead of after the whole hop, and no host
thread or event sits between the cards. This is what NCCL's shared-memory transport does, minus its low-latency
protocol, which doubles the bytes on the wire.

## Standalone checks ([bench/hop_paths](../../../bench/hop_paths/README.md))

- `vram_write`: can the host write a hop's bytes into the receiving card's video memory (its BAR1 window) faster than
  the card reads them from pinned host memory? This is the one number behind sharing data through video memory by way of
  the host (the second leg as posted writes instead of the card's reads), against today's second leg.
- `memops_check`: do stream memory operations (`cuStreamWriteValue32` / `cuStreamWaitValue32` on a mapped host word)
  run on a P106 under driver 580, and how fast does their signal cross from one card's stream to another's against an
  event?

## The two new switches (fork branch `pascal-p106-v5-hop-paths`)

- `GGML_CUDA_HOP_TIMING=1` times each hop between two cards inside the running engine: a one-thread kernel on the
  sending card stamps that card's `%globaltimer` just before the hop, another on the receiving card just after the hop is
  ordered there, and the host notes when it enqueued the hop; each card's clock is related to the host's by a short
  calibration on first use and every 5 seconds. The summary per boundary, size class and path is logged when the server
  stops; `GGML_CUDA_HOP_TIMING_FILE` writes every hop as CSV (`%p` in the name becomes the process id). It uses no CUDA
  timing events, which change the token on these cards.
- `GGML_CUDA_HOP_WC` allocates the hop buffers write-combined: bit 0 the staged ring's slots, bit 1 the mailbox's slots
  and gate buffers. The host never reads them.
- Both are off by default. The build (`926fea4d4`, `TESTS=1`) adds no compiler warning, and on a build machine's two
  cards the fork's `test-hop-mailbox` passes every ring setting plain, with `GGML_CUDA_HOP_WC=3` and with timing on (the
  cards' clocks related to the host's to within 2.3 and 3.0 µs, 160 hops timed, none without both stamps).

## Results

Pending: the lanes started 2026-11-28 at 13:55. Each result is added here with every load's value.

## Files

| File | What |
|---|---|
| [run/hops.sh](run/hops.sh) | the lanes (`LANE=bytes`, `mailbox`, `ring`, `timing`, `wc`) on top of `bench/deep_lane/remote.sh` |
| [run/hop_paths.sh](run/hop_paths.sh) | the standalone checks (`vram_write`, `memops_check`) and C140's hop bench (`megakernel_p106 --hop`) on an idle rig, ~15 minutes |

## Commands

From the repository's root, on a machine that runs only ssh and tar (everything timed runs detached on the rig):

```bash
RIG=<rig> LANE=bytes BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<recorded agent text, proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-28_hop_speedup/run/hops.sh
# then STEP=start, STEP=wait, and STEP=collect OUT=<folder>; the same with LANE=mailbox or LANE=ring
python3 -I bench/deep_lane/deep_table.py <OUT> --control q8
```

The build was `pascal-p106-v5-cand-b12674` at `eee615cdb` (the fork [miner-llama](https://github.com/Cinderella-Man/miner-llama)),
built for compute 6.1 with `scripts/build_llamacpp_docker.sh`; the preset was
[release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) at this repository's commit
`60e20eb6`.

## Provenance and licence

Measured by this project on its own rigs; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here.
The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
