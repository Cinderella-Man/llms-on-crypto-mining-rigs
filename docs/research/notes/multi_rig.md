# Several rigs, one model: splitting a model across 2–5 mining rigs (2026-10-28)

Scope: whether two to five identical Manli M-P106L9-N6G rigs (nine P106-090 cards each, [hardware spec](../../mining_rig.md))
can run one model together over Ethernet, which way of splitting it works on this hardware, how fast it would be, which
models it pays for, what software exists, and what else several rigs can do together. This note builds on
[distributed inference](distributed_inference.md) (llama.cpp RPC between a rig and a PC, its protocol, the gigabit cost
model and security; not repeated here), [idle cards](idle_cards.md), [pipelined speculation](pipelined_speculation.md),
[multi-conversation serving](multi_conversation_serving.md), [multi-GPU physics](multigpu_pascal_performance.md),
[large MoE models](models_large_moe_and_compression.md) and [power](power_and_tuning.md). Conventions are those of the
other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, vendor or maintainer measurement) and secondary.
- A "hand-off" is the hidden state passing from one card to the next inside a rig (through host memory, no P2P); a
  "crossing" is the hidden state passing from one rig to the next over the network.
- GB are decimal (10⁹ bytes). "Rig budget" is the weight a rig holds with a working context: est. ~47 GB, from the
  Ornith-1.5-122B files of 44.5–48.2 GB that run on nine cards with 49k–123k tokens of context
  ([Ornith page](../../../models/ornith-1.5-122b/README.md)); nine cards have 9 × 6,019 MiB = 56.8 GB free. R rigs: est.
  ~94, ~141, ~188 and ~235 GB for 2, 3, 4 and 5 rigs.

Starting point, measured on one rig ([registry](../../../quants/experiments.md), [Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)):
- Flint (39.8B MoE, 2.5B active; 49 layers, hidden 2,048, 150,144-token vocabulary) in Unsloth's Q8_0 (42.3 GB, 2.70 GB
  read per token) on nine cards: 44.6–45.9 tok/s in the coding-agent check with pipelined speculation; plain decoding
  35.9 tok/s in llama-bench tg128 on the fork, ~28 ms per token ([power](power_and_tuning.md)).
- Ornith-1.5-122B (122B hybrid MoE, ~10B active; 48 layers, hidden 3,072, 248,320-token vocabulary) in UD-Q2_K_XL
  (44.5 GB): 9.7 tok/s in the agent check on v2.0 (103.4 ms per token), 11.3 tok/s with v3's IQ2_S/IQ3_XXS kernels
  (88.6 ms). Its v2.0 profile: Q8_0 dense tensors at ~150 GB/s per card, routed experts in IQ formats at 19–29 GB/s,
  attention, Gated DeltaNet and small ops ~28 ms per token at short context; 142.9 ms per token at 39k tokens of context.
- The hand-off between cards has not been measured directly (Q81). Estimates range from ~0.1 ms
  ([multi-GPU physics](multigpu_pascal_performance.md)) to ~0.55–0.65 ms ([idle cards](idle_cards.md), from Flint's pass
  time); this note uses 0.1–0.6 ms per hand-off.

## 1. The question

Each rig already runs its nine cards as a pipeline over PCIe 1.0 x1 links with no peer-to-peer access. Up to five such
rigs can be cabled together. Can one model run across them, and is it worth it?

**Short answer.** Yes, by layers, and the network is not what limits it:
- **The link is not the bottleneck.** A layer split sends one hidden state (8–16 KiB) per rig boundary per token.
  Gigabit Ethernet moves that in est. 0.2–0.4 ms. That is about one card-to-card hand-off, of which a nine-card rig
  already pays eight per token.
- **What limits it is the layer split itself.** One token still passes every card in turn, so a model twice as large
  per token is twice as slow, whatever the number of rigs.
- **Several rigs buy capacity and quality, not single-stream speed.** Ornith in 4–8-bit files on 2–3 rigs runs at
  est. 8–13 tok/s, about its 2-bit speed on one rig, much closer to the original model. 200–400B-class MoE models fit
  4–5 rigs at est. 4–9 tok/s.
- **For Flint, which fits one rig, splitting only slows it down.** Five rigs serve it best as five copies.
- **Tensor and expert parallelism between rigs lose at these latencies.**

## 2. The link between rigs

### What the rig has

- **Network:** one onboard Gigabit Ethernet controller (Realtek RTL8111/8168, driver `r8169`) on its own chipset root
  port. The port runs at PCIe 1.0 x1 (250 MB/s per direction), twice what gigabit needs
  ([hardware spec](../../mining_rig.md)).
- **No free PCIe slot:** all nine x16 slots carry GPUs, and the M.2 slot holds a SATA SSD.
- **USB:** two USB 3.0 (5 Gbit/s) and two USB 2.0 ports. No Thunderbolt or USB4, so no USB networking between hosts
  beyond Ethernet adapters: a PC's xHCI controller is host-only (est.).
- **Two ways to add a faster NIC:**
  - a USB 3.0 Ethernet adapter;
  - a NIC in place of one GPU (6 GB less). The slot's link runs at PCIe 2.0 x1, 500 MB/s or ~4 Gbit/s per direction,
    either direct on root port #4 or behind a switch shared with three GPUs.

### Gigabit Ethernet between two such hosts

**Latency.** The `r8169` patch that added interrupt-coalescing control measured two RTL8168d hosts through a
store-and-forward gigabit switch:

| Coalescing | Small packets, ping min / avg (ms) | 1.5 KB frames, ping min / avg (ms) |
|---|---|---|
| RX coalescing on (driver default then) | 0.274 / 0.485 | 0.330 / 0.566 |
| `ethtool -C … rx-usecs 0 rx-frames 0` | 0.036 / 0.051 | 0.112 / 0.125 |

- Since 2019 the driver turns RX coalescing off by default on RTL8168, because together with ASPM it raised packet
  latency considerably.
- A plain-LAN llama.cpp RPC setup measured 0.29 ms round trips. A WireGuard-based mesh VPN measured ~2 ms on this rig
  ([distributed inference](distributed_inference.md) §3), so traffic between rigs belongs on the plain LAN or a direct
  cable.

**Bandwidth.** ~117 MB/s of TCP goodput (941 Mbit/s, est. at the 94 % efficiency measured on 10 GbE). That is 0.6× one
card's host link (189–213 MB/s).

**One crossing, est.** One hidden state from the last card of rig i to the first card of rig i+1:

| Step | 8 KiB (Flint, F32) | 16 KiB (hidden 4,096, F32) | Basis |
|---|---|---|---|
| Card → host (D2H, ~213 MB/s + ~10 µs) | ~0.05 ms | ~0.09 ms | measured link rate |
| TCP send and receive on the Celeron | ~0.02–0.06 ms | ~0.02–0.06 ms | est. |
| Serialization at 1 Gbit/s | ~0.07 ms | ~0.13 ms | 117 MB/s |
| Wire, switch, NIC | ~0.02–0.03 ms | ~0.03 ms | half the 0.036–0.051 ms small-packet round trip |
| Host → card (H2D, ~189 MB/s + ~10 µs) | ~0.05 ms | ~0.10 ms | measured link rate |
| **Total** | **~0.2–0.25 ms** | **~0.3–0.4 ms** | (est.) |

- F16 activations halve the serialization and copy terms (the project's Q231 does this between cards for
  Qwen3.8-Flash-Next).
- If coalescing is on, add ~0.2–0.4 ms.
- A crossing therefore costs about what a hand-off between two cards of one rig costs (0.1–0.6 ms, est.).

**CPU.** Receiving and sending one 8–16 KiB message per token is negligible for the Celeron (est.). Encrypted
transports (SSH tunnels, VPNs) are a different matter: their cost on this CPU is unmeasured (claim to test).

### Faster links and their limits

| Option | Bandwidth | Latency | Cost on the rig | Worth it for |
|---|---|---|---|---|
| Onboard 1 GbE, direct cable (2 rigs) | ~117 MB/s | lowest; auto-MDI-X, any cable | none | everything below |
| Onboard 1 GbE + gigabit switch (3–5 rigs) | ~117 MB/s per port | +~12 µs per full frame (store-and-forward) | a switch | 3–5 rigs |
| USB 3.0 2.5GbE adapter (RTL8156B, `r8152` driver) | 2.32 Gbit/s measured on other hosts once the `r8152` driver binds instead of `cdc_ncm` | higher than PCIe (USB bulk transfers; unmeasured, claim to test) | a USB 3 port (shared with the model stick's controller) | network model loading, a second NIC for a ring |
| 2.5/5/10GbE NIC in a GPU slot | ~4 Gbit/s ceiling at PCIe 2.0 x1 (est.) | like the onboard NIC | one GPU (6 GB) per rig | network model loading |
| RoCEv2 NIC (ConnectX-class) in a GPU slot | as above | hardware RoCE ~3.7 µs against ~21 µs without RDMA (2010 measurement); llama.cpp RPC negotiates RDMA through `libibverbs` | one GPU per rig | nothing on this hardware (below) |
| Soft-RoCE (`rdma_rxe`) on the Realtek NIC | as 1 GbE | ~11.6 µs in the same 2010 study; CPU-driven | CPU time | nothing |

- **RDMA saves only the network's share of a crossing, ~0.05–0.1 ms (est.).** These cards have no GPUDirect RDMA:
  NVIDIA documents it for Tesla and Quadro GPUs. Every byte still goes card → host → NIC over the x1 links. The two
  PCIe legs (~0.1–0.2 ms) and the card hand-offs (0.1–0.6 ms each) stay.
- **Bandwidth beyond 1 GbE helps only bulk transfers.** Decode needs at most 0.8 MB/s per link (16 KiB × 50 tok/s,
  est.), well under 1 % of gigabit. Prompt micro-batches are larger, but the link still carries them several times
  faster than the cards compute them (section 3). What a faster link speeds up is model weights sent over the network.

**Tuning:**
- Plain LAN, no VPN overlay.
- RX and TX coalescing off (`ethtool -C`, verify with `ethtool -c`).
- `TCP_NODELAY` in the application.
- The NIC's interrupt on the core that does not run the decode loop.
- Jumbo frames only if both NICs and the switch support them (RTL8111 revisions differ; claim to test). They put a
  12–16 KiB hidden state into two frames instead of nine to eleven.
- Busy polling costs a core the 2-core host cannot spare (est.).

**Security.** `rpc-server` is unauthenticated (see [distributed inference](distributed_inference.md) §5). A direct
cable or a switch used only by the rigs isolates the traffic.

## 3. Traffic a model needs between rigs, per way of splitting

Shapes from the models' configs. Per-token bytes count weights read per token, est. from the configs as in
[large MoE models](models_large_moe_and_compression.md); for Ornith they match the profile (3.62 G routed-expert and
5.35 G dense weights per token).

| Model | Total / active | Layers, hidden | Experts (routed / top-k) | Attention and f16 KV per token | Files (GB) |
|---|---|---|---|---|---|
| Flint | 39.8B / 2.5B | 49, 2,048 | 256 / 6 + shared | 14 full + 35 sliding-513, 28 KiB | Q8_0 42.3 |
| Ornith-1.5-122B | 122B / 10B | 48, 3,072 | 256 / 8 + shared | 36 Gated DeltaNet + 12 full, 24 KiB | UD-Q2_K_XL 44.5 · UD-Q4_K_XL 79.7 · UD-Q5_K_XL 94.6 · UD-Q6_K_XL 114.2 · Q8_0 132.6 |
| Step-3.5-Flash | 199B / 11B | 45, 4,096 | 288 / 8 + shared | 12 full + 33 sliding-512, 48 KiB | Q4_0 113.4 · Q8_0 209.4 |
| MiniMax-M2.5 | ~230B / 10B | 62, 3,072 | 256 / 8 | 62 full (8 KV heads), 248 KiB | Q4_0 129.5 · Q8_0 243.1 |
| DeepSeek-V4-Flash | 284B / 13B | 43, 4,096 | 256 / 6 + shared | sliding-128 + sparse top-512 indexer, small | UD-IQ3_XXS 103 · UD-Q4_K_XL 155.1 · UD-Q8_K_XL 162 |
| Qwen3.5-397B-A17B | 397B / 17B | 60, 4,096 | 512 / 10 + shared | 45 Gated DeltaNet + 15 full, 30 KiB | UD-IQ2_XXS 114.9 · UD-IQ3_XXS 140.3 · UD-Q3_K_XL 178.7 · UD-Q4_K_XL 245.3 · Q8_0 421.5 |

Est. from the configs: Qwen3.5-397B reads 7.55 G routed-expert and ~8.7 G other weights per token, Step-3.5-Flash 5.3
and ~6.1 G, MiniMax-M2.5 7.0 and ~3.4 G, DeepSeek-V4-Flash 6.5 G of experts of its 13B active.

### 3.1 Pipeline (layer) split: each rig holds a block of layers

**Traffic.** One hidden state per crossing per token, forward. Rig 1 needs the sampled token back (4 bytes when the
last rig samples on its cards).

| | Flint (2,048) | Ornith, MiniMax (3,072) | Hidden 4,096 |
|---|---|---|---|
| Per crossing, decode, F32 / F16 | 8 / 4 KiB | 12 / 6 KiB | 16 / 8 KiB |
| Per crossing, a 512-token prompt micro-batch, F32 | 4 MiB, ~36 ms at 117 MB/s | 6 MiB, ~54 ms | 8 MiB, ~72 ms |
| Link throughput, prompt tokens per second | ~14,300 | ~9,500 | ~7,200 |
| Added per token per crossing (est., section 2) | 0.2–0.25 ms | 0.25–0.35 ms | 0.3–0.4 ms |

- **Prompt processing does not wait on the link.** On one rig prompts read at up to 723 tok/s with Flint (8,192
  tokens) and 209–263 tok/s with Ornith (13,700 tokens). The link carries 10–60× that (est.); each micro-batch only
  arrives 36–72 ms later per crossing.
- **Load:** each rig reads its own layers from its own drive in parallel (~5.5 min for 44.5 GB from a USB 3 stick,
  measured), if the software allows it (section 4).

**Time model (est.).** The basis is the measured rates above:

T ≈ weight bytes ÷ card read rate + ~0.6 ms of other work per layer + (cards − rigs) × hand-off + crossings × 0.2–0.4 ms

- Card read rates: Q8_0 and Q4_0 at 100–150 GB/s on the fork's kernels; K-quants at an est. 60–100; IQ formats at
  19–29 GB/s on llama.cpp's kernels, ~70 GB/s with v3's.
- Attention at depth comes on top: Ornith measured +~40 ms per token at 39k tokens of context.

Single conversation, short context:

| Model, file | Rigs (cards) | Bytes per token (est.) | Matrices | Other | Hand-offs | Network | Per token | tok/s |
|---|---|---|---|---|---|---|---|---|
| Ornith UD-Q2_K_XL, v2.0 (measured) | 1 (9) | 5.0 GB | – | – | – | – | 103.4 ms | 9.7 |
| Ornith UD-Q2_K_XL, v3 IQ kernels (measured) | 1 (9) | 5.0 GB | – | – | – | – | 88.6 ms | 11.3 |
| Ornith UD-Q2_K_XL on 18 cards (room for the full 262k context) | 2 (18) | 5.0 GB | as above | | +1–5 ms | 0.6 ms | ~90–110 ms | ~9–11 (est.) |
| Ornith UD-Q4_K_XL | 2 (18) | 5.9–7.8 GB | 46–73 ms | 28 ms | 2–10 ms | 0.6 ms | 77–112 ms | 9–13 (est.) |
| Ornith Q8_0 | 3 (27) | 9.5 GB | 64–77 ms | 28 ms | 2–14 ms | 0.9 ms | 95–120 ms | 8–10.5 (est.) |
| Step-3.5-Flash Q4_0 | 3 (27) | 6.4–6.9 GB | 43–69 ms | 27 ms | 2–14 ms | 0.9 ms | 73–111 ms | 9–13 (est.) |
| Step-3.5-Flash Q8_0 | 5 (45) | 12.1 GB | 81–101 ms | 27 ms | 4–24 ms | 1.5 ms | 114–154 ms | 6.5–8.8 (est.) |
| MiniMax-M2.5 Q4_0, short context | 3–4 (27–36) | 5.9 GB | 39–59 ms | 37 ms | 2–19 ms | 1 ms | 79–116 ms | 8.5–12.5 (est.) |
| MiniMax-M2.5 Q4_0, 32k context | 3–4 | + 8.1 GB of KV | | | | | 180–500 ms | 2–5.5 (est.) |
| DeepSeek-V4-Flash UD-Q8_K_XL | 4 (36) | ~10.4 GB | 75–115 ms | 26–34 ms | 3–19 ms | 1.2 ms | 105–170 ms | 6–9.5 (est.) |
| Qwen3.5-397B-A17B UD-IQ3_XXS, v3 IQ kernels | 4 (36) | 7.8–10 GB | 74–112 ms | 36 ms | 3–19 ms | 1.2 ms | 114–168 ms | 6–8.8 (est.) |
| Qwen3.5-397B-A17B UD-IQ3_XXS, llama.cpp IQ kernels | 4 (36) | 7.8–10 GB | 133–224 ms | 36 ms | 3–19 ms | 1.2 ms | 173–280 ms | 3.6–5.8 (est.) |

Notes on the rows:
- **Network crossings cost 0.5–1.5 % of a token at these sizes (est.).** The hand-offs inside the rigs cost 2–15 %,
  so Q81's measurement decides the spread.
- **Ornith on 2–3 rigs runs as fast as Ornith in 2 bits on one (est.).** The 2-bit file's experts run in IQ formats
  at a sixth of the card's read rate, while Q4–Q8 experts run on the fork's fast kernels. The quality gain is large:
  UD-Q2_K_XL sits at KLD 0.184 against the makers' Q8_0 on wikitext, and Q8_0 is the makers' own file.
- **Two rigs also give Ornith its full 262,144-token context:** est. 6 GB of KV plus recurrent states, against 49k–123k
  on one rig.
- **MiniMax-M2.5's full attention reads 248 KiB of KV per token of context.** At agent depths that outweighs the
  weights, so it suits these cards only for short prompts. Hybrid (DeltaNet) and sparse-attention models do not have
  this problem.
- **DeepSeek-V4-Flash's near-native file fits four rigs.** Its experts come in an FP4 format with no Pascal kernel in
  the fork, and its sparse attention is untested on compute 6.1 (claim to test).
- **Qwen3.5-397B UD-Q4_K_XL (245.3 GB) does not fit five rigs with a context (est.),** and the Q8_0 file would need
  about nine. UD-IQ3_XXS (140.3 GB) fits four rigs comfortably and three barely; UD-Q3_K_XL (178.7 GB) needs four to
  five.
- **Prompts:** time to first token grows with active parameters. The first token per agent turn takes a median 6.6–7.6
  s with Ornith on one rig; a 17B-active model on four rigs, est. 1.5–2× that.
- **Power:** a layer split keeps every card in P0 at ~35–39 W while it waits its turn (power note: 317–329 W of GPU
  power for nine cards in decode). Five rigs draw est. ~2 kW at the wall for one conversation of a 200–400B model.

**Flint, which fits one rig:**

| Flint Q8_0 | Plain decode (est.) | With pipelined speculation |
|---|---|---|
| One rig, nine cards (measured) | 35.9 tok/s (tg128, ~28 ms per token) | 44.6–45.9 tok/s (agent check) |
| Two rigs, same nine cards (5 + 4), stock RPC | 34–35 tok/s at short context, 32–33 at 32k: +2 crossings, 3 blocking remote copies, the remote cards' masks (~2 ms at 32k) | none through stock RPC (section 4) |
| Two rigs, 18 cards, a rig-to-rig pipeline link (section 6) | 30–34 tok/s: +9 hand-offs, +2 crossings | about as on one rig (S_eff ~2× but acceptance-bound) |
| Two rigs, 18 cards, stock RPC, 32k context | 25–31 tok/s, including est. 3–4 ms of attention masks sent over the network | none |
| Two rigs, 18 cards, stock RPC, 262k context | est. 15–19 tok/s: ~2.6–3.6 MB of masks per token cross the network unless built on the cards | none |

- **A higher-quality file buys nothing.** Q8_0 is already near-lossless, and BF16 (~80 GB) would read twice the bytes
  on cards without fast BF16/FP16 math.
- **More context buys nothing.** 262,144 tokens already fit one rig.
- **Fewer layers per card do not speed up one conversation.** A token still passes all layers in turn, and each extra
  card adds a hand-off.
- **Throughput is the same as two copies.** Halving the layers per card halves the pipeline cycle, so the ceiling for
  many conversations doubles to est. 500–590 tok/s. Two copies on two rigs reach the same 2 × 292 tok/s ceiling
  ([idle cards](idle_cards.md): 3.4 ms cycle) with no network at all.

**Pipelined speculation across a slow hop.**
- **A pipeline link costs no throughput.** If each rig sends its hidden state onward without waiting (section 6), a
  crossing adds only to the time a token needs to pass all cards (L), not to the cycle (c, set by the slowest card).
  The link's 7,000–14,000 tokens per second exceed what any card stage can produce.
- **More rigs do not raise single-stream speculation gains.** S_eff = L / c roughly doubles on two rigs (est. ~14 on 18
  cards), but at guess accuracy α ≈ 0.4 the gain S / (α + (1 − α) S) is 1.51× at S 7.2 and 1.57× at S 14 (est., PPSD's
  closed form). Speculation is accuracy-bound.
- **Idle cards drop to base clock.** With 18 cards each card idles ~95 % of a single stream, so more work in flight is
  needed to keep the clock up ([idle cards](idle_cards.md) §1.6).
- **The deeper pipeline does serve more conversations at once:** est. ~14 staggered conversations of Ornith on two rigs
  at near single-stream speed, against ~7 on one rig.

### 3.2 Tensor parallelism across rigs

**Traffic.** Two all-reduces per layer, each moving a hidden state each way between rigs:
- Flint 98, Ornith 96, Qwen3.5-397B 120 exchanges per token;
- 16–32 KiB per exchange: est. 1.6 MB per token for Flint, 2.4 MB for Ornith;
- each exchange pays PCIe on both ends and a network round trip: est. 0.3–0.6 ms.

**Added per token (est.):** Flint 29–59 ms, Ornith 29–58 ms, Qwen3.5-397B 36–72 ms.

**Saved:** at most half of the matrix time on two rigs. For Ornith Q4 that is ~25–35 ms of ~100, so two rigs break
even at best, and three or more lose. For Flint (~28 ms per token) it is a 2–3× slowdown.

**Measured on 1 GbE elsewhere:** distributed-llama's tensor parallelism on four Raspberry Pi 5 boards moved 636 kB out
and 1,057 kB in at the root per token for Qwen3-30B-A3B. It spent 14–37 ms per token in synchronization, of a 76.7 ms
token.

Tensor parallelism adds no capacity over a layer split. Rejected.

### 3.3 Expert parallelism, or one rig holding the experts of some layers

- **Expert parallelism** (each layer's experts spread across rigs): every MoE layer needs a dispatch and a combine with
  the rigs holding its token's experts. With 8 of 256 experts per token, nearly every layer touches every rig.
  - Cost: est. 47–60 round trips per token (Flint 47, Ornith 48, Qwen3.5-397B 60) at 0.3–0.6 ms, or 14–36 ms added.
  - Saving: at most the remote share of expert time, ~9–25 ms for Ornith on two rigs. Dense tensors, which are most of
    Ornith's bytes, are not parallelized at all.
  - Verdict: a loss.
- **Expert offload** (attention on rig A, the experts of a block of layers on rig B): two crossings per offloaded
  layer, est. 19–38 ms for Ornith's 48 layers, against 0.3 ms for moving the same layers whole. Always worse than a
  layer split.
- Both are rejected, consistent with the single-rig result ([idle cards](idle_cards.md) §2.9–2.10).

## 4. Software for multi-node inference on commodity Ethernet

| System | Runs on these cards and this host? | Split | Measured multi-node decode (primary unless marked) |
|---|---|---|---|
| **llama.cpp RPC** (`rpc-server`), built from the fork | yes (CUDA sm_61; one server exposes all nine cards) | layers; `-sm tensor` all-reduce between exactly two servers; prompt micro-batches overlap across RPC devices since PR #18626 (2026-08-26) | below |
| distributed-llama | CPU path optimized for AVX2 (the Celeron has none); Vulkan experimental, untested on Pascal | tensor parallel, 2ⁿ nodes, at most as many nodes as KV heads (Ornith and Qwen3.5-397B: 2; Flint: 4); one device per node (est.), so one card per rig | Qwen3-30B-A3B Q40, 4 × Raspberry Pi 5 on a gigabit switch: 13.04 tok/s, sync 14–37 ms per token |
| exo | no: GPU only on macOS (MLX); Linux runs on CPU, MLX CUDA needs compute ≥ 7.x | tensor and pipeline | 4 × M3 Ultra over Thunderbolt 5 RDMA: Qwen3-235B ~32 tok/s, 1.8× on 2 and 3.2× on 4 devices |
| prima.cpp | dense models only, a model copy per device, Q4_K_M/Q6_K/Q8_0 only (S) | piped ring | 70B Q4K 674 ms per token over Wi-Fi (paper) |
| Petals | PyTorch: Pascal only in CUDA 12.6 wheels, which end with PyTorch 2.14; a Python server per card does not fit 4 GB of RAM; no GGUF | pipeline of block servers | up to 6 tok/s Llama 2 70B, 4 tok/s Falcon 180B (README) |
| vLLM / SGLang pipeline parallel over Ray | no: vLLM requires compute capability 7.5 or higher | TP/PP/EP | – |
| PipeInfer | MPI on a 2023 llama.cpp | pipelined speculation across nodes | design reference, see [pipelined speculation](pipelined_speculation.md) |
| LocalAI P2P, GPUStack | wrappers over llama.cpp RPC | as RPC | – |

### llama.cpp RPC as it is today (master, read 2026-10-28)

**Status.** The README still calls it a fragile, insecure proof of concept.

**Transport and commands.**
- One `rpc-server` per host serves all its accelerators. TCP is the default. RDMA is negotiated at connect time on
  RoCEv2 NICs via `libibverbs` (or Thunderbolt 5 on macOS) and falls back to TCP.
- Sent asynchronously, with no reply: `SET_TENSOR`, `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` (an unchanged graph is re-run
  by its uid) and `GET_TENSOR` into a caller buffer. The server queues graph compute with
  `ggml_backend_graph_compute_async`.
- Blocking until a reply returns: `COPY_TENSOR` and the allocation commands.

**Boundaries between cards of one remote host.**
- `cpy_tensor_async` is not implemented (NULL in the backend interface), so the scheduler copies with `COPY_TENSOR`.
- The server synchronizes before any command that reads or writes buffer data, so each boundary costs the client one
  network round trip after the previous card finishes.
- Copies between two servers are not supported: the client falls back to a copy through its own host. The PR #18626
  author left server-to-server copies out as the hard part.
- `event_wait` is a no-op, since each server has a single stream.

**Measured, multi-node.**
- Two Strix Halo hosts over a 9.4 Gbit/s USB4 link:
  - Qwen3.5-397B-A17B UD-Q4_K_XL: tg128 11.76 tok/s, pp512 147.6.
  - MiniMax-M2.5-REAP-139B Q8_0: tg 15.35.
  - Qwen3-Coder-Next: ~26.3 tok/s on two hosts against 36.8–38.7 on one (−30 %).
- Four M3 Ultra Mac Studios over Thunderbolt 5 with TCP (S): Qwen3-235B fell from 20.4 tok/s on one node to 15.2 on
  four, while exo with RDMA rose from 19.5 to 31.9.
- Plain 1 GbE: Qwen3.5-35B-A3B at 51.7 tok/s on two GPUs, and 10 GbE: a dense 72B at 5.9 tok/s
  ([distributed inference](distributed_inference.md) §1).
- PR #18626's own prompt numbers: +30–70 % with two RPC servers, worse with three or four.

**Load path for multi-rig.**
- The client process reads the whole GGUF and streams every remote tensor over the network. The server buffers each
  tensor whole in host RAM.
- Five rigs' worth from one gigabit port takes est. ≥ 27 min (4 × 47 GB at 117 MB/s). The `-c` cache stores each
  rig's share on its own drive for later loads.
- The largest expert tensors of the models above are 0.9–1.6 GB at 4–8 bits (est.: Ornith Q8_0 0.86 GB, Qwen3.5-397B
  Q4 1.2 GB, Step-3.5-Flash Q8_0 1.6 GB). Each must fit next to the server process in a 4 GB host (claim to test;
  the board takes a second SO-DIMM, up to 32 GB).
- Loads of models over ~100 GiB across two hosts hung in `llama-server` until `-dio` (direct I/O) was set (issue
  #19745, ROCm hosts).

**What this means here (est.):**
- **Stock RPC runs a layer split across rigs, one token at a time.** The client host makes one round trip per remote
  card boundary, so a remote rig adds ~9 round trips per token (0.5–3 ms). That is tolerable for Ornith-class tokens.
- **Every per-token input crosses the network once per remote card.** The attention mask (n_kv × 2 bytes per card
  with full-attention layers) is the large one: est. 0.6 MB per token for nine remote cards at 32k, 2.3 MB at 131k.
  The fork's on-card mask removes it if its op runs inside the remote graph (claim to test).
- **The fork's pipelined executor and streams mode do not survive the RPC boundary.** Blocking copies and a no-op
  event wait mean the client cannot keep several tokens in flight on remote cards. Flint's agentic 45 tok/s would fall
  to its plain rate.
- **One Celeron schedules every card's split.** At ~0.3 ms per card per token ([multi-conversation
  serving](multi_conversation_serving.md)), 36–45 cards cost est. 11–14 ms of host time per token on the head rig.

## 5. Other ways five rigs can work together

**One model copy per rig behind a load balancer** gives the most agentic throughput for models that fit one rig:
- **Today:** five independent agents at Flint's ~45 tok/s each, est. ~225 tok/s in total.
- **Per rig with staggered streams** (built in `llama-pipespec`, not yet in llama-server, Q104/Q105): est. ~265 tok/s
  in steady state with eight conversations at ~33 tok/s each ([idle cards](idle_cards.md)). Five rigs: est. ~1,300
  tok/s for ~40 concurrent agent conversations.
- **Routing must be sticky per conversation.** The prompt cache lives in the rig's slot, so a 30k-token agent context
  moved to another rig is re-read: est. ~45 s with Flint (600–700 tok/s) and ~2 min with Ornith (209–263 tok/s).
  - A plain HTTP proxy does this: nginx `hash <conversation key> consistent` keeps a key on one server and remaps few
    keys when a rig joins or leaves; `least_conn` suits stateless requests.
  - Paddler, an LLM-aware balancer, ships its own llama.cpp engine, so it cannot front the fork's servers.

**Different models on different rigs, and agent teams:**
- Example split of five rigs: Ornith in Q4 on two (planning, review, hard steps) and Flint on three (parallel workers
  at 45 tok/s each). That needs only per-role endpoints in the agent harness, no engine work.
- Every 200–400B option above runs at 4–9 tok/s and holds four or five rigs for one conversation, so it suits a
  reviewer that answers rarely, not a worker that generates most tokens (est.).

**A draft model on another rig:**
- **The network is not the obstacle.** A draft round trip (est. 0.1–0.3 ms) is small against Ornith's ~90 ms tokens.
- **There is no reason to use another rig.** A small same-tokenizer drafter (the Qwen3.5 family shares Ornith's
  vocabulary) fits a spare card, or the spare memory of a two-rig split, on the same rig.
- **Flint has no same-tokenizer drafter.** Its guesser (copy lookup, answer store, the staircase head) reads data
  already on the rig.
- **Batched verification is the limit, not placement.** Ornith's MTP head gave stock llama.cpp +22 % and v2.0 nothing.
- **Drafting across models with different vocabularies** (Flint for Ornith) would need cross-tokenizer speculation.
  Not pursued.

**Moving a conversation's KV cache instead of re-reading it.** A 30k-token Ornith context holds est. ~0.9 GB of KV
and recurrent state (24 KiB per token plus the DeltaNet states). That crosses gigabit in ~8 s, against ~2.5 min to
re-read the prompt at ~200 tok/s; for Flint (28 KiB per token plus ~53 MiB of window cells, ~0.9 GB) ~8 s against
~45 s. llama-server's slot save and restore writes and reads such a state, so a balancer could move a conversation
between rigs when one is overloaded (claim to test: whether the saved state covers the recurrent layers).

**Prompt processing on another rig (disaggregated prefill)** needs a second copy of the weights and moves the same KV
state. It keeps long prompts from stalling the decoding rig's other conversations. With two copies of a model, though,
each can simply serve its own conversations. Worth a row only once two copies of a model are deployed anyway.

## 6. Verdict and first measurements

**Verdict:**
- **A single model across rigs is realistic as a layer split.** It is practical for 2–3 rigs now with stock llama.cpp
  RPC and slow single conversations, and fully practical only with a rig-to-rig pipeline link in the fork.
- **The network costs ~0.2–0.4 ms per rig boundary per token (est.)**, about one card hand-off, so 1 GbE is enough and
  faster links or RDMA change nothing measurable for decode.
- **It pays for models that do not fit one rig.**
  - Ornith-1.5-122B in Q4–Q8 on 2–3 rigs runs est. 8–13 tok/s: as fast as its 2-bit file on one rig, much closer to
    the original, and with the full 262k context.
  - Step-3.5-Flash Q4_0 on three rigs: est. 9–13 tok/s.
  - The 200–400B class on 4–5 rigs (Step-3.5-Flash Q8_0, DeepSeek-V4-Flash near-native, Qwen3.5-397B-A17B
    UD-IQ3_XXS): est. 4–9 tok/s.
- **It does not pay for Flint, or any model that fits one rig.** Split, it is slower (est. 0.8–0.95× plain), loses
  pipelined speculation through stock RPC, and gains no throughput over one copy per rig.
- **Tensor and expert parallelism between rigs are rejected:** 14–72 ms of added latency per token against savings of
  at most half the matrix time.
- **For most users, five rigs are five servers.** Sticky routing, per-role models, and at most one or two rigs joined
  for a larger model.

### What would have to be built in the fork

1. **A rig-to-rig pipeline link** (PipeInfer over TCP on the existing executor). Each rig runs its own process with a
   layer range on its nine cards and its own pipelined executor. A boundary message carries slot, position, sequence
   and the hidden state (F16 where exact enough, Q231's question).
   - KV and recurrent-state edits (dropped guesses, `seq_rm`, rollbacks of Q162) travel in order in the same stream,
     so each rig applies them between the same tokens.
   - The last rig samples on its cards and returns 4-byte token ids to the first, which runs the HTTP server and the
     guesser.
   - Each rig builds its own masks and inputs from positions, so per-token inputs never cross the network.
   - Messages are one-way: no round trip per token in steady state.
2. **Partial model loading:** a layer range of a GGUF loaded by each rig from its own drive, with the token embedding
   on the first rig and the output head on the last. Loading becomes parallel (~5.5 min per rig) instead of ≥ 27 min
   through one port. It also removes whole-tensor staging in 4 GB of RAM.
3. **If stock RPC is kept instead:**
   - `cpy_tensor_async` for cards of one server (a copy queued on the server with no reply);
   - event waits on the server's card streams;
   - server-to-server boundary copies;
   - the on-card mask inside remote graphs.

   This is more code in a protocol upstream still calls a proof of concept, and one Celeron still schedules every card.

### First measurements, cheapest first

0. **One-rig loopback RPC (no second rig, ~1 h on the rig).** Measures the protocol's own cost on the Celeron; the
   wire adds ~0.05 ms per round trip.
   - Setup: an `-DGGML_RPC=ON` build of the fork (built on the quant machine), `rpc-server` bound to 127.0.0.1
     exposing cards 5–8, `llama-server` on cards 0–4 plus that server, Flint Q8_0 at a 32k context.
   - Compare with the same nine cards in one process: plain decode, `--pipe-depth 3`, `GGML_RPC_DEBUG=3 -lv 5` for
     commands and bytes per token, host CPU and memory.
   - Expected (est.): plain −3 to −8 %; pipelined speculation drops to plain speed on the RPC part.
1. **Network baseline (once a second rig or any Linux PC is on the same switch, 15 min):**
   - `ethtool -c` on both ends;
   - `ping -i 0.2 -s 82` and `-s 1472` with coalescing on and off;
   - `iperf3` both directions;
   - CPU during both.
   - Expected: 0.04–0.3 ms round trips, ~940 Mbit/s, a few % of a core.
2. **Two rigs, same nine cards (5 + 4) over 1 GbE, Flint Q8_0 via stock RPC.** Against the nine local cards this
   isolates the crossing and the remote copies. Then 9 + 9 cards to price eight more hand-offs.
   - Expected (est.): 34–35 tok/s plain at 5 + 4 (short context), 25–31 at 9 + 9 (32k).
   - The pair calibrates every estimate in this note.
3. **Ornith-1.5-122B UD-Q4_K_XL (79.7 GB) on two rigs via stock RPC, on the agent check,** against UD-Q2_K_XL on one
   rig. Its KLD against the makers' Q8_0 is scored on the quant machine with the
   [Ornith KLD texts](../../data/2026-10-27_ornith_kld/README.md). This is the first run with a user-visible gain:
   quality at equal speed.
   - Expected (est.): 9–13 tok/s plain.
   - The KLD and top-token agreement against the makers' Q8_0 should improve several-fold.
4. Only if (3) holds: the pipeline link (build item 1) and Q8_0 on three rigs.

### Claims to test

- Hand-off cost per card boundary (Q81): 0.1 or 0.6 ms changes every multi-rig estimate by 2–15 %.
- A crossing costs 0.2–0.4 ms with coalescing off; the RTL8111's default coalescing state on Ubuntu 24.04.
- Remote-card masks and inputs per token over RPC, and whether the fork's on-card mask runs inside remote graphs.
- Host RAM for `rpc-server` with 0.9–1.6 GB tensors on a 4 GB host.
- Q4_K, Q5_K and Q6_K expert read rates on the fork's kernels (used above at an est. 60–100 GB/s).
- MXFP4 and DeepSeek-V4's sparse attention on compute 6.1.
- USB 2.5GbE adapter latency on this host.

## Sources

Hardware and network
- This project: [hardware spec](../../mining_rig.md) (NIC, root ports, link rates); [distributed inference](distributed_inference.md) (RTT on a plain LAN and over a mesh VPN, RPC protocol, security)
- F. Romieu, K. Smelkov, "r8169: Add support for interrupt coalesce tuning (ethtool -C)", Linux commit
  [5097083](https://github.com/torvalds/linux/commit/509708310cf917a05fbceb41ad67da1416b81bd0) (P; ping RTTs between two RTL8168d hosts)
- Linux stable patch "r8169: disable default rx interrupt coalescing on RTL8168", 2019-04 ([LKML](https://lkml.iu.edu/1904.1/07378.html)) (P)
- CNX Software, "Fixing performance issues with Realtek RTL8156B 2.5GbE USB dongle in Ubuntu", 2022-02-20 ([link](https://www.cnx-software.com/2022/02/20/fixing-performance-issues-with-realtek-rtl8156b-2-5gbe-usb-dongle-in-ubuntu/)) (S; 2.32 Gbit/s with `r8152`)
- LANL Team CYAN, "Implementation and Comparison of RDMA over Ethernet", 2010 ([slides](https://neno.lanl.gov/projects/national-security-education-center/information-science-technology/_assets/docs/2010-si-docs/Team_CYAN_Implementation_and_Comparison_of_RDMA_Over_Ethernet_Presentation.pdf)) (P; RoCE 3.7 µs, Soft-RoCE 11.6 µs, no RDMA 21.09 µs)
- NVIDIA, GPUDirect RDMA guide (CUDA 9.2): Tesla and Quadro GPUs ([docs](https://docs.nvidia.com/cuda/archive/9.2/gpudirect-rdma/index.html)) (P)
- nginx, `ngx_http_upstream_module`: `hash … consistent`, `least_conn` ([docs](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)) (P)

Models (configs and file listings read 2026-10-28)
- Qwen3.5-397B-A17B [config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json); [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) (UD-IQ2_XXS, UD-IQ3_XXS, UD-Q3_K_XL, UD-Q4_K_XL, Q8_0 folders)
- MiniMax-M2.5 [config](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/raw/main/config.json); [unsloth GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) (Q4_0, Q8_0)
- DeepSeek-V4-Flash [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json); unsloth, [DeepSeek V4 guide](https://unsloth.ai/docs/models/deepseek-v4) (284B / 13B active, GGUF sizes)
- Step-3.5-Flash [config](https://huggingface.co/stepfun-ai/Step-3.5-Flash/raw/main/config.json); [bartowski GGUF](https://huggingface.co/bartowski/stepfun-ai_Step-3.5-Flash-GGUF) (Q4_0, Q8_0)
- Ornith-1.5-122B [unsloth GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) (UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL folders); this project's [Ornith page](../../../models/ornith-1.5-122b/README.md) and [benchmark](../../benchmarks/2026-10-27-ornith-122b.md)

Software and multi-node measurements
- llama.cpp [tools/rpc/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) and
  [ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp), master read 2026-10-28 (P)
- llama.cpp [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) "rpc : implement event and async backend APIs", merged 2026-08-26 (P)
- llama.cpp [issue #19745](https://github.com/ggml-org/llama.cpp/issues/19745), large RPC loads and `-dio` (P)
- visorcraft, [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf), two-host RPC results, 2026-02/03 (P)
- J. Geerling, "1.5 TB of VRAM on Mac Studio – RDMA over Thunderbolt 5", 2025-12 ([blog](https://jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5)) (P); per-node figures as reported by [Gigazine](https://gigazine.net/news/20251222-mac-rdma-over-thunderbolt-1-5tb/) (S)
- b4rtaz, [distributed-llama](https://github.com/b4rtaz/distributed-llama) and [Discussion #255](https://github.com/b4rtaz/distributed-llama/discussions/255) (Qwen3-30B-A3B on 4 × Raspberry Pi 5, 2025-09-05) (P)
- [exo](https://github.com/exo-explore/exo) README (P); [Petals](https://github.com/bigscience-workshop/petals) README (P); Z. Li et al., "Prima.cpp", [arXiv 2504.08791](https://arxiv.org/abs/2504.08791) (P); prima.cpp quant support as reported in a [CSDN walkthrough](https://aicoding.csdn.net/6a3214b210ee7a33f27e627d.html) (S)
- vLLM, [GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html): compute capability 7.5+ (P)
- PyTorch dev-discuss, [Maxwell and Pascal removed from CUDA 12.8/12.9 builds](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) and [no CUDA 12.6 wheels from 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) (P)
- B. Butler et al., "PipeInfer", [arXiv 2407.11798](https://arxiv.org/abs/2407.11798) (P); R. Li et al., PPSD, [arXiv 2509.19368](https://arxiv.org/abs/2509.19368) (P)
- intentee, [Paddler](https://github.com/intentee/paddler) README (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q81, Q87, Q104, Q105, Q110, Q116, Q162, Q166, Q231, Q270); [idle cards](idle_cards.md); [pipelined speculation](pipelined_speculation.md); [multi-conversation serving](multi_conversation_serving.md); [multi-GPU physics](multigpu_pascal_performance.md); [power and tuning](power_and_tuning.md); [large MoE models](models_large_moe_and_compression.md)
