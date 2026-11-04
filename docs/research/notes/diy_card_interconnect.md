# Home-made ways to link the rig's cards: what a faster card-to-card path can buy, and every hack ranked (2026-11-04)

Scope: every do-it-yourself way to move data between the nine P106-090 cards faster than today's path (card → host
memory → card, each card on its own PCIe 1.0 x1 link, eight of them four-and-four behind two ASM1184e switches, no
peer-to-peer, a 2-core Celeron 3865U host), from software and driver tricks to firmware registers, cheap boards and
cables. It starts from the measured hop ([card hops on x1](card_hops_x1.md), C140 in
[2026-11-04_megakernel_rig](../../data/2026-11-04_megakernel_rig/README.md), C200r in
[2026-11-04_card_hops](../../data/2026-11-04_card_hops/README.md)) and asks first what any faster path could save.
The rig's own link facts are in [2026-11-04_diy_interconnect](../../data/2026-11-04_diy_interconnect/README.md).
Registry rows: Q990–Q1011 in [quants/experiments.md](../../../quants/experiments.md).

Conventions:
- "(est.)" marks this note's own arithmetic or inference, with its basis next to it. Numbers without it are measured
  on the rig or quoted from the linked source.
- A **hop** is one card boundary of the layer split: card i's residual (Flash-Next: four streams × 2,560 values; with
  today's Q8_0 hop cast 10,880 B per token) leaving card i and arriving on card i + 1. A token crosses eight.
- "As sold" means the stock rig with software changes only, the project's default. Everything that needs money, a
  physical change, a driver binary patch or a firmware register write is a proposal for the rig's owner and sits in
  the **hardware hacks** track, apart from the public setup.

## Summary

- **Within today's layer split, the whole interconnect is worth ≤ 1.7 % of a decoded token and est. 8–11 % of prompt
  time.** One token's Q8_0 hop takes 126.7 µs (8 × = 1.0 ms of a 58.6 ms token at 17.06 tok/s). C200r's F32 → Q8_0
  cast gave +22–31 % prompt speed, so hops were est. 25–32 % of prompt time with F32 and are est. 8–11 % now. An
  infinitely fast link would add at most ~1.7 % to decode and ~9–12 % to prompts; on an agent session est. ≤ 5 %
  (prompts take est. a fifth to a third of its time).
- **The intuitive model (a 250 MB/s link crossed twice, so a direct cable only has to beat ~125 MB/s) is right about
  today's path, but no cable can do better than the cards' own links.** Today's legs run one after the other:
  ~26 µs fixed plus ~10 µs per KB, ~86 MB/s effective at 10.9 KB and ~130–150 MB/s at 40–160 KB. But the two legs
  use *different* links (card i's and card i + 1's), and PCIe carries each direction on its own wires, so the host
  path is serialized, not halved: overlapping the legs recovers most of what a direct path would give. A direct path
  between two cards also crosses both links. The GPU has no other data port (no display, no SLI, no NVLink, I²C at
  400 kHz), so every "cable" ends up as PCIe through the same x1 links, and nothing outside the card can beat the
  card's own ~190–210 MB/s.
- **The ceiling per decode hop, with what reaches it** (est., section 1.3): legs overlapped through host memory
  ~75–85 µs (software, in progress as the streamed mailbox, Q682); true peer-to-peer inside a switch ~55 µs
  (blocked by the driver's policy; section 4); Gen2 links ~45–65 µs (firmware registers, the owner's decision); Gen2
  plus peer-to-peer ~30 µs. The best of these saves ~0.8 ms a token (1.3 %).
- **Peer-to-peer may be one driver option away.** The driver refuses it by policy, not for want of silicon: Pascal
  GeForce cards did PCIe peer-to-peer (two GTX 1080 Ti, 2019), the rig's mobile chipset is simply missing from NVIDIA's
  table, and NVIDIA's own test key `ForceP2P=0x11` skips every check. It gave verified peer copies on a CMP 170HX
  mining card that reported "GPU not supported". On the rig it needs a reload of the NVIDIA modules (the owner's
  decision) and checksummed tests, since forced P2P has written to the wrong place on other cards.
- **The one change that raises the physical ceiling is the card's own link speed, and it may be a volatile register,
  not hardware.** The cards advertise only 2.5 GT/s (Gen1) in `LnkCap` while the switches' downstream ports and the
  root ports advertise 5 GT/s (Gen2). On P104-100 and CMP mining cards the Gen1 cap is an override bit in the GPU's own
  PCIe block (`NV_XVE_PRIV_MISC_1`, BAR0 `0x8841C`) that the boot code sets; clearing it and retraining doubled a
  P104-100's bandwidth on Linux with the rig's exact driver (580.178.04). On a P106-090 it is unproven: the one public
  attempt dropped the card off the bus. Gen2 is the hard ceiling here (the Celeron's Base-U PCH and the ASM1184e are
  Gen2), and the four cards behind each switch still share one Gen2 x1 uplink.
- **Where a fast, low-latency link would change the design is tensor or expert parallelism**, where all nine cards work
  on every layer at once instead of one card at a time. It needs 96 small all-reduces a token; on today's links est.
  30–130 µs each (3–12 ms a token), so it pays only if splitting a layer nine ways speeds compute by ≥ 1.07–1.24×.
  Published MoE results lose or barely gain even on x8/x16 links (8 × V100: 57.9 against 73.8 tok/s; 4 × P100 with
  graphs: 1.15×), and prompts under tensor parallelism would be link-capped. Not ruled out on these slow-memory cards;
  one benchmark decides it (Q996).
- **Dead ends, and why:** FPGA bridges (the FPGA cannot write into a GeForce card, so it is a mailbox no better than
  host memory, and it needs a slot the rig does not have); ESP32 and other microcontrollers (≤ 10–75 MB/s, and they sit
  on the host's USB); a NIC inside the rig (no free slot, stages through host memory anyway); the GPU's display outputs
  (none on the board, frame-scale latency even where they exist); VBIOS cross-flashing, strap resistors and capacitor
  mods (none ever changed a P106's PCIe generation; the cards' slots are one lane anyway); PCIe switch cards (an x1 slot
  keeps them x1); Gen3 or wider links on this board (the PCH and the slots stop at Gen2 x1); nouveau (Pascal runs at
  boot clocks there).

### What to try first

| Rank | Idea | Decode hop (est.) | Prompt hops (est.) | Cost, risk | As sold? | Decides it | Row |
|---|---|---|---|---|---|---|---|
| 1 | **Legs overlapped through host memory** (the streamed mailbox in 4–16 chunks; the persistent pipeline) | 127 → 75–85 µs (−0.3 to −0.4 ms a token) | ~147 → ~185 MB/s | software; in progress | yes | the hop bench's streamed-mailbox columns | Q682, Q684 |
| 2 | **Read-only facts**: `nvidia-smi topo -p2p r` / `w` reason codes; whether the installed `nvidia.ko` holds the `ForceP2P` key; whether an IOMMU is active (the `lspci` facts are in: section 2) | — | — | none, a minute on an idle rig | yes | the outputs themselves | Q991 (Q990 done) |
| 3 | **`ForceP2P=0x11`**: NVIDIA's own override of the peer-to-peer checks, then checksummed peer copies on one pair behind a switch | 127 → ~55 µs (−0.57 ms a token) | one link's rate instead of two legs, off the shared uplink: est. +3–5 % prompts | no money; a module reload, and a hang or wrong copies are possible: **owner's decision** | hardware and files as sold, a driver option | checksums, then the hop bench's peer column | Q992 |
| 4 | **Two numbers for tensor/expert parallelism**: a 9-card star all-reduce through host memory (cards write partials, the CPU sums, cards read) and the 9-way compute speedup of one Flash-Next layer | — | — | software, a bench, ~1 h on a rig | yes | a ≤ 50 µs all-reduce and S ≥ 1.3× reopen the fork project | Q996 |
| 5 | **Gen2 on the cards' own links** (volatile BAR0 register writes, then a retrain from the switch port; possibly a 2-byte driver patch) | 127 → 45–65 µs (−0.5 to −0.65 ms); ~30 µs with peer-to-peer | hop time halved where one card per switch transfers: est. +4–6 % prompts | no money; risk: a card off the bus, which here needs a reboot at the console | hardware as sold, firmware state not: **owner's decision** | `LnkSta 5GT/s` on card 4 (its own root port), then the hop bench | Q994 |
| 6 | **USB 3 → 2.5/5 GbE adapters between two rigs** (the two-rig split, Q505) | rig-to-rig crossing: ~same (USB adds ~0.6 ms round trip) | rig-to-rig prompt crossing 2.5–3.7× faster than gigabit | ~€15–40 per adapter: **owner's purchase** | adds parts | iperf3 and ping between the rigs, then Q505's prompt speed | Q998 |
| 7 | **A peer window through I/O-memory registration** (card A's kernels write into card B's Vulkan-mapped BAR1 window) | as rank 3, without the driver's peer permission | as rank 3 | software; untested by anyone, a freeze is possible: **owner's decision** | yes | a checksummed write from A into B's window | Q993 |

## 1. What a faster path can buy

### 1.1 Today's hop, measured

C140 (rig 1, cards 0 → 1, median of 300 per cell; every pair within 3 %; C200r on rig 2 within a few µs):

| Bytes | What | Driver's peer copy | Staged, 2 chunks | Staged, 4 chunks | Effective rate, best |
|---:|---|---:|---:|---:|---:|
| 10,880 | one token, Q8_0 (today) | 141.4 µs | **126.7 µs** | 130.2 µs | 86 MB/s |
| 40,960 | one token, F32 | 455.1 µs | 356.2 µs | 316.9 µs | 129 MB/s |
| 43,520 | four tokens, Q8_0 (a verify pass) | 478.9 µs | 377.7 µs | 336.9 µs | 129 MB/s |
| 163,840 | four tokens, F32 | 1,684.6 µs | 1,291.1 µs | 1,113.2 µs | 147 MB/s |

- **The fit:** ~26 µs fixed plus ~10 ns per byte, which is 1/213 MB/s (card → host) plus 1/189 MB/s (host → card):
  the two legs in series. Chunking overlaps them partly.
- **Both legs run near the link's packet limit.** A PCIe 1.0 lane carries 250 MB/s after 8b/10b coding. The path
  already uses 256 B payloads (section 2): a card → host write carries 20–24 B of headers (~230 MB/s), and the link
  layer's acknowledgements and flow-control packets take est. 5–10 % more (213 measured). A host → card copy is the
  card reading host memory, answered in completions of 64–128 B with 20 B of headers each (~190–215 MB/s before the
  link layer; 189 measured). Nothing is left to win on the wire itself.
- **A token on the release line** (S35: llama-bench 17.06 tok/s, 58.6 ms a token) spends 8 × 126.7 µs = 1.0 ms in hops,
  1.7 %. C200r's decode gain from the cast (+10–15 %) is the F32 hops' share before it.
- **Prompts** (C200r, rig 2, `f32` → `q8` lines): +22 % at 1.4k tokens, +31 % at 7k, +22–23 % at 31–33k. If all of
  the gain is hop bytes, hops took H = (1 − 1/s) / 0.734 = 25–32 % of F32 prompt time (0.734 is the share of bytes
  the cast removes), and take 0.266 H / (1 − 0.734 H) = **8–11 %** now (est.).
- **Verify passes** (k tokens a pass, the agent check's MTP line): a 4-token Q8_0 hop is 337 µs, 2.7 ms a pass over
  eight boundaries; bytes scale with k, so bandwidth matters more as passes grow (8 tokens: est. 5–7 ms a pass).

### 1.2 Two crossings are two links, not one link halved

- The staged copy's first leg loads card i's link (upstream), the second card i + 1's link (downstream). Different
  wires, so nothing forces them to take turns; today they mostly do because each leg waits for the previous one to
  finish (the 26 µs fixed part is software: copy submission, events, a synchronize; a GPU reading host memory costs
  0.5–0.8 µs plus 0.3–0.9 µs per chipset or switch hop on other machines, [Chips and Cheese](https://chipsandcheese.com/p/microbenchmarking-chipsets-for-giggles),
  and a 64 B DMA read ~0.55 µs on a server, [pcie-bench](https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf)).
- **PCIe is full duplex**, but not perfectly: copies in both directions on one card give ~56–75 % of the one-way rate
  each on fast cards ([nvbandwidth](https://github.com/NVIDIA/nvbandwidth),
  [NVIDIA forum](https://forums.developer.nvidia.com/t/asymmetric-pcie-bandwidth-in-bidirectional-transfers-h2d-drops-56-while-d2h-maintains-performance/352186)),
  because a host → card copy sends its read requests upstream beside the card → host data. In a decode hop each card
  uses one direction, so this does not bite; in a prompt pipeline a card receives and sends at once (est. 1.3–1.5×
  aggregate, not 2×).
- **Behind a switch**, card i's upload and card i + 1's download cross the shared Gen2 x1 uplink in opposite
  directions, so they do not compete either. Only concurrent hops of several cards behind one switch share it
  (prompts with several micro-batches in flight).
- **A direct peer write would cross the same two links** (up card i's lane into the switch, down card i + 1's lane),
  forwarded packet by packet. Its gain over a well-pipelined host path is the root complex round trip and the
  software's fixed part, not bandwidth.

### 1.3 The ceiling per decode hop

10,880 B, est. from the measured rates (bytes ÷ rate plus the overheads named):

| Path | Per hop | Per token (8 hops) | Saved against today | What it needs |
|---|---:|---:|---:|---|
| Today: staged, 2 chunks | 126.7 µs (measured) | 1.01 ms | — | — |
| Host memory, legs fully overlapped (fine chunks or flags) | 75–85 µs: the slower leg 57.6 µs + one chunk + 10–20 µs | 0.6–0.7 ms | 0.3–0.4 ms (0.5–0.7 %) | software (Q682's streamed mailbox, Q684) |
| Peer write inside a switch | ~55 µs: 51.6 µs at 211 MB/s + 2–5 µs | 0.44 ms | 0.57 ms (1.0 %) | the driver's peer access (section 4) |
| Gen2 links, today's staged path | ~65–80 µs: 26 µs + 10,880 B × (1/427 + 1/386 MB/s) | 0.53–0.64 ms | 0.4–0.5 ms | section 3 |
| Gen2 links, legs overlapped | ~40–45 µs: 28 µs + 10–15 µs | 0.33–0.36 ms | 0.65 ms (1.1 %) | sections 3 and 5 |
| Gen2 links, peer write | ~30 µs | 0.24 ms | 0.77 ms (1.3 %) | sections 3 and 4 |
| Zero | 0 | 0 | 1.01 ms (1.7 %) | — |

Gen2 x1 rates are a CMP 100-210's measured 386 MB/s host → card and 427 MB/s card → host on an x1 riser
([CMP100-210 Gen3 x1](https://github.com/mintoriakamoto/CMP100-210xFULLTENSORSxGEN3x1)), the nearest published figure.

**The bar for any other link** (a cable, a NIC, a microcontroller): with latency L and bandwidth B it beats today's hop
only if L + 10,880 B / B < 127 µs, *and* it must not add the two PCIe crossings it cannot avoid. Since every byte still
leaves card i and enters card i + 1 over their own x1 links, an outside link adds to the path instead of replacing it.

### 1.4 Where a faster path would matter more

- **Prompts:** a 512-token micro-batch carries 5.57 MB per boundary in Q8_0: est. 38 ms at today's best staged rate,
  29.5 ms with legs overlapped, ~13–15 ms with Gen2 and overlap. With hops at est. 8–11 % of prompt time, halving them
  is est. +4–6 % prompt speed; removing them +9–12 %.
- **Longer verify passes:** 8 tokens a pass would carry est. 5–7 ms of hops; doubling the bandwidth saves est. 2.5–3.4 ms
  a pass.
- **Tensor or expert parallelism:** section 6.

## 2. The rig's link facts

From the rig (`lspci -vv` in the hardware inventory of 2026-10-22, the same on the second rig; summarized in
[2026-11-04_diy_interconnect](../../data/2026-11-04_diy_interconnect/README.md)) and the datasheets:

| Part | Link capability | Link status | Source |
|---|---|---|---|
| P106-090 (each of nine) | 2.5 GT/s (Gen1), x4 | 2.5 GT/s x1 ("downgraded") | `lspci -vv` on the rig |
| ASM1184e downstream ports (eight) | 5 GT/s (Gen2), x1 | 2.5 GT/s x1 (held down by the card) | `lspci -vv`; [ASMedia](https://www.asmedia.com.tw/product/556yQ9dSX7gP9Tuf/b7FyQBCxz2URbzg0.html): one Gen2 x1 upstream, four Gen2 x1 downstream |
| ASM1184e upstream ports (two) | 5 GT/s x1 | 5 GT/s x1 | `lspci -vv` |
| PCH root ports (four: two switches, card 4, the NIC) | 5 GT/s x1 | 5 GT/s to the switches; 2.5 GT/s to card 4 and the NIC | `lspci -vv` |
| Celeron 3865U PCH (Base-U) | 10 lanes, PCIe 2.0 only; groups of four lanes as 4 × x1, 2 × x2, x2 + 2 × x1 or x4 (soft straps in the flash descriptor); at most six root ports | — | [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/96507/intel-celeron-processor-3865u-2m-cache-1-80-ghz/specifications.html); [7th Gen U/Y I/O datasheet vol. 1, 334658-003](https://cdrdv2-public.intel.com/334658/7th-gen-core-family-mobile-u-y-processor-lines-i-o-datasheet-vol-1.pdf), Tables 1-1, 1-2, §3.1–3.2 |

The rest of the path (rig 1, `run/pcie_facts.sh` in the data folder, Q990):
- **The cards' PCIe capability is version 1**: they have no `LnkCap2`/`LnkCtl2` registers, so `LnkCap`'s 2.5 GT/s is
  the whole story; the switch ports and root ports already target 5 GT/s.
- **Payloads are 256 B on every card's path** (cards, switch ports and root ports capable of and set to 256 B; only the
  network controller's port runs 128 B); the cards' read requests are 512 B, completions on a 64 B boundary.
- **No ACS stands in the way of peer traffic:** the ASM1184e ports have no ACS capability, and the root ports have it
  with every control bit off.
- **Each card's BAR1 is 256 MB** (64-bit, prefetchable; no resizable BAR).
- **Four root ports are enabled, all in use** (switch, card 4, network controller, switch); the SSD sits on the PCH's
  SATA controller.

- **Gen1 is set by the card, Gen2 by everything else.** Every port above the cards already offers 5 GT/s; the cards
  offer 2.5. So a card that advertised Gen2 would train at Gen2 with no change to the board.
- **Gen3 and more lanes are out on this board.** Only Premium-U PCHs have PCIe 3.0 (12 lanes); the Base-U Celeron has
  10 Gen2 lanes. Every slot is wired one lane, and the ASM1184e is x1 on both sides, so a wider root port would reach
  no card.
- **The cards are x4 parts:** `LnkCap` says x4, as a vendor listing does (PCIe 1.1 x4,
  [TweakTown](https://www.tweaktown.com/news/60895/bitnand-prepares-nvidia-p106-090-6gb-mining-card-costs-389/index.html)).
  On a board with x4 slots each card would get est. ~800 MB/s at Gen1 (section 8, not as sold).
- **The Manli board itself** is undocumented in public beyond the product page: no teardown, slot wiring or BIOS
  options were found.

## 3. Raising the cards' link speed to Gen2

### 3.1 What caps the cards

- **Not a fuse and not the VBIOS file, as far as anyone has found.** On every NVIDIA mining card reverse-engineered so
  far the cap is a "CYA" override in the GPU's PCIe block, `NV_XVE_PRIV_MISC_1` at BAR0 offset `0x8841C`, which the
  card's boot code (devinit) sets so that `LnkCap`/`LnkCap2` advertise 2.5 GT/s only:
  - CMP 100-210 (GV100): four bits `CYA_GEN2/GEN3_SPEED_OVERRIDE_{EN,VAL}`; the fuse words that would make it permanent
    read 0; clearing the bits takes `LnkCap2` from 0x2 to 0xE and the card from Gen1 x1 to **Gen3 x1** (0.194 / 0.214 →
    0.806 / 0.845 GB/s) ([mintoriakamoto](https://github.com/mintoriakamoto/CMP100-210xFULLTENSORSxGEN3x1)).
  - CMP 170HX: bits 11/13 (enables) and 12/14 (values), plus a `DIS_G2` chicken bit at `0x8C2C0` bit 2 → Gen2
    ([cmp170hx wiki](https://github.com/Consensus-Protocol/cmp170hx/wiki/Unlock-PCIe-Gen2)).
  - **P104-100 (GP104, Pascal like the P106):** a group of volatile writes (`0x8872C`, `0x2241C`, `0x8841C`, `0x8C2C0`,
    `0x8C300`, `0x8C040`, `0x8C1C0`, `0x88610`), then the target speed in `LnkCtl2` on both ends and a retrain from the
    port above: `LnkCap2` 0x2 → 0x6, Gen1 x4 → Gen2 x4 ([skbskhud/p104-unlock-v3](https://github.com/skbskhud/p104-unlock-v3)).
- **On Pascal the closed driver re-clamps on top.** The Linux P104-100 project found a mode whitelist in the 580 driver's
  RM that keeps Gen1; a 2-byte patch of `nvidia.ko` (580.178.04, the rig's version) plus a helper module took the card
  from 0.779 / 0.855 to **1.561 / 1.708 GB/s**; it must be redone after every boot
  ([WJNKAC/nvidia-p104-unlock-gen2-for-linux](https://github.com/WJNKAC/nvidia-p104-unlock-gen2-for-linux)).
- **The P106-090 silicon can run Gen2:** one card trained at 5 GT/s x16 on a desktop board after a UEFI pre-boot
  register tool ([P106-90-Unlock-X16](https://github.com/DaZuiZZZZ/P106-90-Unlock-X16)), and its PCIe mirror registers
  sit at the P104's offsets. A Windows tool aimed at the P106-090 (10de:1c09) exists; its only public report (2026-11-02)
  latched the Gen2 declaration and then lost the card after the reset ([P106-Unlock-Gen2 issue #1](https://github.com/DaZuiZZZZ/P106-Unlock-Gen2/issues/1)).
- **The driver's registry keys only choose among advertised speeds.** `RMPcieLinkSpeed` has allow/forbid fields per
  generation (Gen2 in bits 1:0, Gen3 in 3:2…, [nvrm_registry.h](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/main/src/nvidia/interface/nvrm_registry.h));
  `RmForceEnableGen2` is a test key. None lifts the card's own clamp, and a wrong value pins every card at Gen1. No
  `nvidia-smi` option raises the generation.
- **`setpci` alone gains nothing:** setting the target speed in the switch port's `LnkCtl2` and retraining comes back at
  2.5 GT/s while the card advertises 2.5 GT/s; the bad outcome is a card that drops.
- **No VBIOS, strap or capacitor mod has changed a P106's or P104's PCIe generation.** Pascal VBIOS images are signed and
  the device ID cannot be changed by flashing; capacitor mods restore missing lanes on some cards' PCBs (widths, not
  speeds), which a one-lane slot cannot use.

### 3.2 What Gen2 would give here (est.)

- **Per card, one at a time:** ~2× (est. 386 MB/s host → card, 427 MB/s card → host, the CMP 100-210's Gen2 x1). The
  decode hop falls to ~65–80 µs on today's staged path and ~40–45 µs with overlapped legs (section 1.3): 0.4–0.65 ms a
  token.
- **Behind a switch, several at once:** the Gen2 x1 uplink (500 MB/s raw, ~400–430 usable per direction) is shared by
  four cards, so concurrent prompt hops gain less than 2×; card 4 on its own root port gets the full Gen2.
- **Prompts:** est. +4–6 % (hop time halved on hops that run alone, less where two cards behind one switch transfer at
  once).
- **Loading a model:** the source is a USB disk at ~120 MB/s, so no gain.

### 3.3 Risks and how a trial would run

- **The failure mode is a card that falls off the bus** (Xid 79, a "zombie" card, an unbind that hangs), recoverable
  only by a reboot or a power cycle. On a rig run headless a reboot needs someone at the machine.
- **Nothing is flashed:** the register writes are volatile and a power cycle returns the card to stock. A driver binary
  patch is a separate step that comes back with every driver install.
- **The register sequence for the P106 is not verified:** the P106 tool reuses an "unverified P104-100 flow" and omits
  writes the P104 sequence needs; user-space writes through `/dev/mem` while the link changes hung the P104 card twice in
  two tries, kernel-side writes did not.
- **A trial (the owner's decision; Q994):** card 4 first (its own root port, no switch, no neighbour to disturb),
  with someone at the machine, no model loaded and no timed task on the rig; read the card's `LnkCap` and `LnkSta` (it
  has no `LnkCap2`) before and after; if it holds at 5 GT/s, the hop bench with card 4 as one end, then a decode and prompt line.

## 4. Peer-to-peer without NVLink

### 4.1 What blocks it

- **Pascal GeForce did peer-to-peer over PCIe.** Two GTX 1080 Ti on driver 410.79 reported peer access "Yes"; the
  latency of a small copy fell from ~11–12 µs to ~1.2 µs at the same bandwidth
  ([Puget Systems, 2019](https://www.pugetsystems.com/labs/hpc/P2P-peer-to-peer-on-NVIDIA-RTX-2080Ti-vs-GTX-1080Ti-GPUs-1331/)).
  Consumer cards lost it with Turing. No report either way exists for the P106, P104, P102 or CMP 30–90HX.
- **The refusal is the driver's policy, decided per pair** in RM's `p2p_caps.c` (the open modules' copy, which shares its
  core with the closed driver): GNS when the GPU carries a "P2P disabled" property, CNS when the chipset is not in
  NVIDIA's table and the two cards share no switch the driver accepts, TNS for some Intel root-port layouts
  ([p2p_caps.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/nvidia/src/kernel/platform/p2p/p2p_caps.c)).
  `nvidia-smi topo -p2p r` and `-p2p w` print the code per pair.
- **The rig's chipset is not in the table.** The chipset list has Skylake and Kaby Lake desktop PCHs (Z170, Z270, …) but
  no Sunrise Point-LP (`0x9Dxx`) entry, and its last row defaults to "not supported"
  ([nvcst.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/nvidia/arch/nvalloc/common/inc/nvcst.h),
  [chipset_pcie.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/nvidia/src/kernel/platform/chipset/chipset_pcie.c)).
  A pair behind one ASM1184e escapes that only if the driver's list of accepted switches includes ASMedia, and the
  headers define no ASMedia vendor ID. Expected: CNS for every pair, possibly GNS for the SKU (est.).
- **The hardware path exists inside each switch.** A PCIe switch forwards a memory write from one downstream port to
  another unless ACS redirect sends it upstream; Linux sets the redirect bits only when an IOMMU asked for them
  ([p2pdma](https://docs.kernel.org/driver-api/pci/p2pdma.html), `pci_enable_acs` in `drivers/pci/pci.c`). On the rig
  nothing redirects: the ASM1184e ports have no ACS capability and the root ports' redirect bits are off (section 2).
  Across the two switches a peer write would pass the PCH's root ports, where forwarding between ports is not
  guaranteed on client chipsets.

### 4.2 The override NVIDIA ships

- **`ForceP2P`** (`NVreg_RegistryDwords="ForceP2P=0x11"`: force peer reads and writes) is checked before the chipset,
  topology and GPU tests and returns early ([nvrm_registry.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/nvidia/interface/nvrm_registry.h):
  "Overrides chipset-based P2P configurations", "only be used to test on internal issues").
- **It has worked on a mining card:** two CMP 170HX that reported GNS got checksum-verified peer copies and peer loads
  from kernels on driver 610.57.04, NCCL then used its P2P channels, and 8k prefill went from ~2,476 to 4,714 tok/s
  ([Level1Techs, 2026](https://forum.level1techs.com/t/couldnt-resist-grabbing-a-cmp-170hx-and-now-im-in-a-sticky-position/253947?page=3));
  two RTX A5000 on Ice Lake went from CNS to OK on driver 515
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/p2p-device-couldnt-access-another-one-with-p2pbandwidthlatencytest/220183)).
  Whether the closed 580 driver for Pascal reads the key is not shown anywhere (est. likely: the key and the early
  return sit in common code); `grep -a -c ForceP2P` on the installed `nvidia.ko` answers it without loading anything.
- **Failure modes to guard against:** forced P2P on hardware whose mailbox path does not work wrote to the wrong place on
  RTX 4090s ([tinygrad](https://raw.githubusercontent.com/tinygrad/open-gpu-kernel-modules/550.54.15-p2p/README.md)), and
  GeForce Pascal peer tests hung machines in 2018 ([NVIDIA forum](https://forums.developer.nvidia.com/t/simplep2p-and-p2pbandwidth-latencytest-hangs-ubuntu/62010)).
  Every test checksums the data, watches `dmesg` for Xid and AER errors, and starts on one pair behind one switch.
- **Setting it means reloading the NVIDIA modules or rebooting** (the option is read at load): the owner's decision on
  this rig (Q992). The order: the read-only checks (Q991); the option with someone at the machine and no task on
  the rig; checksummed `cudaMemcpyPeer` and peer stores on one pair behind a switch; then the hop bench's peer column
  and llama.cpp's own peer path (`GGML_CUDA_P2P`).
- **What it would give** (est., section 1.3): a decode hop of ~55 µs instead of 127 µs (−0.57 ms a token), prompt hops
  at one link's rate instead of two legs, and the shared switch uplink and host memory out of the path for pairs behind
  one switch. In the split's order (cards 0–3 behind one switch, card 4 alone, cards 5–8 behind the other) six of the
  eight boundaries are such pairs; 3 → 4 and 4 → 5 cross the PCH.

### 4.3 A peer window without the driver's permission (untested)

- `cuMemHostRegister` with `CU_MEMHOSTREGISTER_IOMEMORY` registers "memory-mapped I/O space … belonging to a third-party
  PCIe device" for a GPU's kernels ([CUDA runtime](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__MEMORY.html)).
  A GeForce GTX 580 wrote into an FPGA's memory window this way at 1.6 GB/s ([Bittner & Ruf, MSR 2012](https://microsoft.com/en-us/research/wp-content/uploads/2016/02/2012062520UCAA2012_Bittner_Ruf_Final-1.pdf)).
- **The idea (est.):** card B allocates a buffer from the Vulkan heap that is both device-local and host-visible (VRAM
  behind B's BAR1, ~256 MB without resizable BAR, [Sawicki](https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them)),
  shares it with CUDA through external memory, maps it for the CPU, and card A registers that CPU pointer as I/O memory.
  A's kernels then post writes straight into B's VRAM through the switch, with a flag written last after
  `__threadfence_system()`; B polls the flag in its own memory.
- **Risks:** the one detailed report of I/O-memory registration froze the machine in the driver's
  `os_lookup_user_io_memory` (driver 465, a fix promised; [NVIDIA forum](https://forums.developer.nvidia.com/t/cudahostregister-and-cudahostregisteriomemory-for-accessing-3rd-part-pcie-device/180337));
  the driver may refuse its own BAR mapping; an IOMMU in translating mode would block it. No one has published this with
  a second GPU. A freeze needs a reboot, so a trial needs someone at the machine (Q993).

### 4.4 What does not apply to Pascal

- **The tinygrad and aikitoria P2P patches** change the open kernel modules (Turing and newer, with GSP) and need
  resizable BAR; Pascal runs only the closed 580 driver and has neither.
- **GDRCopy and GPUDirect RDMA** (`nvidia_p2p_get_pages`) are for Data Center and RTX/Quadro boards; the call fails with
  −EINVAL on a GeForce RTX 3090 ([GDRCopy](https://github.com/NVIDIA/gdrcopy)).
- **nouveau** owns BAR1 itself, so peer writes would be possible, but Pascal stays at its boot clocks there forever
  (NVIDIA never released the PMU firmware, [Collabora](https://www.collabora.com/news-and-blog/news-and-events/nvk-enabled-for-maxwell,-pascal,-and-volta-gpus.html))
  and there is no CUDA: the cards would lose far more in memory clock than any link could give.
- **CUDA IPC, stream memory operations aimed at another card's memory and NCCL's P2P transport** all need the driver's
  peer mapping; without it they fail or stage through the host.

## 5. Host-side tricks

### 5.1 Overlapping the legs (in progress)

- **The streamed mailbox** (Q682): card i's last kernel writes the residual into mapped pinned host memory in 1–16
  chunks with a flag each; card i + 1's kernel polls and reads each chunk as it lands, so the second leg trails the
  first by one chunk instead of waiting for all of it. C140's unchunked mailbox took 436 µs at 40 KB, the same as a
  host-staged copy, because it did not overlap. Ceiling: section 1.3's 75–85 µs.
- **The persistent pipeline** (Q684): a hop written while the card's last product still runs leaves est. ~20 µs of
  it exposed (Q8_0), less than any link change in section 1.3 would.
- **A poll costs est. 1.2–2.5 µs on this rig** (a client CPU's 0.54–0.79 µs, a switch crossing, Gen1 serialization;
  [Chips and Cheese](https://chipsandcheese.com/p/microbenchmarking-chipsets-for-giggles)), so a mailbox signal floors
  at est. 3–6 µs against today's ~26 µs fixed part.

### 5.2 Write-combined memory

- Write-combined pinned memory is not snooped and can cross PCIe faster on some systems, but CPU reads of it are very
  slow ([cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)). A mailbox that only
  GPUs touch gains nothing measurable on the reports found; for a CPU-reduced all-reduce (section 6) only the buffer the
  CPU writes and the cards read is a candidate. An A/B in the hop bench settles it (part of Q682's bench).

### 5.3 Larger packets (MPS)

- The card → host leg is posted writes of at most the path's Max Payload Size, and the rig already runs the
  largest the devices allow: 256 B on every card's path (section 2). Raising it (Q995) gains nothing. The host → card
  leg is the card's reads, answered in completions on the root port's 64 B boundary, which no setting changes.

### 5.4 What the host itself costs

- The 2-core Celeron's part in a hop is the fixed ~26 µs (submission, events, the wake-up); the leg overlap of 5.1
  removes most of it from the critical path. A mailbox needs no host thread at all. Nothing in the host's memory
  bandwidth (one DDR4-2133 channel) limits a hop: 10 KB is ~1 µs of it.

## 6. What a fast link would unlock: all nine cards on every layer

In a layer split one card works at a time and eight wait, so a single conversation uses one card's ~173 GB/s of
memory bandwidth at any moment. Tensor parallelism (each card holds a ninth of every layer) and expert parallelism
(each card holds a ninth of each layer's experts) use all nine at once, and their price is communication on every
layer, where the interconnect becomes the design's limit.

- **Traffic:** two all-reduces per layer (after the mixer or attention, after the experts and shared expert), 96 a token
  for Flash-Next's 48 layers; each card contributes 2,560 values: 2,720 B in Q8_0, 5,120 B in F16, 10,240 B in F32
  ([PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) sends F32 for small reductions). An expert-parallel
  variant with the non-expert weights on every card needs only the combine, 48 a token, at the cost of VRAM the 6 GB
  cards may not have and of load imbalance (10 routed experts over 9 cards: the busiest holds ~2–3).
- **The shape that fits this rig is a star through host memory** (est.): every card writes its partial into pinned
  host memory, the CPU sums nine vectors (~23k additions, a few µs), every card reads the sum. A ring over nine cards
  would take 16 sequential steps. Per all-reduce in Q8_0: up ~13 µs a card, but four cards share each switch's uplink
  (~27 µs), the sum ~5–10 µs, down ~29 µs through the uplink, polls ~5 µs: **est. ~70 µs** (F16 ~110–130 µs). The best
  published small all-reduce through host memory is 7–9 µs (NCCL's shared-memory transport, 2 × RTX 4090 on PCIe 4.0
  x16, P2P off, [PyTorch forum](https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366)).
- **Break-even** (est.): with layer split's token T = C + 1.0 ms and tensor split's C / S + 96 a, tensor split wins when
  its compute speedup S exceeds C / (T − 96 a): 1.07× at a = 50 µs, 1.13× at 80 µs, 1.24× at 127 µs, 1.46× at 200 µs.
- **The evidence on S is discouraging for MoE models of this size:** 8 × V100 on PCIe 3.0 x8, Qwen3.5-35B-A3B: tensor
  57.9 against layer 73.8 tok/s, and the 122B-A10B wins (58.3 against 47.7) ([PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749));
  4 × P100 with CUDA graphs, an A3B model: 70.6 against 61.4 tok/s (1.15×), 39.6 without graphs
  ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)); dense models gain 1.4–2.8×. Splitting a layer nine
  ways leaves each 5-SM card ~3,000 tiny kernels a token whose fixed costs do not divide. On the other side, the P106's
  slow memory makes the weight reads a larger share of C than on V100 or P100, which favours splitting.
- **Prompts would suffer:** each all-reduce of a 512-token micro-batch carries 5.57 MB per switch per direction in Q8_0,
  est. 13–26 ms, 96 a micro-batch: a link cap of est. ~200–400 tok/s against today's layer-split hops of ~0.45 s a
  micro-batch even fully serialized. Speculative verify passes multiply all-reduce bytes the same way.
- **Software status:** llama.cpp's `-sm tensor` has no all-reduce kernel for compute 6.1 (Q879), its internal all-reduce
  takes exactly two Volta-or-newer cards, and vLLM's custom all-reduce turns off above two PCIe-only GPUs.
- **What a better link changes:** with Gen2 on every card the star all-reduce falls to est. ~35–45 µs (still the shared
  uplink); with a direct Gen2 lane per card to the host, est. 16–19 µs. Neither changes S, which decides the question.
- **The decision is two numbers (Q996):** the star all-reduce's latency on nine cards (median and p99, Q8_0 and
  F16, GPU-driven flags so it fits inside a CUDA graph), and S for one Flash-Next layer split nine ways (its kernels
  at a ninth of their rows, on one card, against the full layer). The fork project (Q997) is parked until both are
  in: it reopens at an all-reduce ≤ ~50 µs and S ≥ ~1.3×.

## 7. Hardware bridges

| Bridge | How it would work | Bandwidth, latency (sourced) | Why it does or does not help | Cost |
|---|---|---|---|---|
| FPGA board with a PCIe endpoint (LiteFury/NiteFury, Acorn CLE-215+, PCILeech-class cards, Kintex-7, ECP5, Tang Mega 138K) | card i writes into the FPGA's memory window, card i + 1 reads it | a GeForce GTX 580 wrote into an FPGA at 1.6 GB/s (40 µs for 4 B through the CUDA API), but read from it at only 0.514 GB/s, and the FPGA cannot write into a GeForce card: GPUDirect RDMA is Tesla/Quadro only ([Bittner & Ruf, MSR 2012](https://microsoft.com/en-us/research/wp-content/uploads/2016/02/2012062520UCAA2012_Bittner_Ruf_Final-1.pdf), [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)) | it is a mailbox exactly like pinned host memory, both cards still cross their own x1 links, and every slot holds a card. As root port for the GPUs it would become their host and run the driver; no GPU behind an FPGA root port was found | $25 (used CLE-215+ in 2020) to $100–350 |
| Microcontroller link (ESP32 SPI/SDIO, RP2040/RP2350, STM32) | a side channel between cards | ESP32 SDIO ≤ 25 MB/s theoretical, ~3–10 measured; RP2040 USB 12 Mbit/s; RP2350 HSTX → HDMI capture 75 MB/s ([Espressif](https://docs.espressif.com/projects/espressif-esp-faq/en/latest/software-framework/peripherals/sdio.html), [CNX](https://www.cnx-software.com/2024/11/19/high-speed-data-acquisition-raspberry-pi-pico-2-hstx-interface-cheaper-hdmi-to-usb-3-0-video-capture-dongle/)) | below one x1 lane, ends in host memory over USB, and a CUDA kernel cannot reach it; at most an out-of-band fan or power controller | $5–15 |
| A NIC inside the rig (ConnectX-3, 10/40 GbE, InfiniBand) | RDMA between cards | ~1 µs RDMA latency; but an x8 card in an x1 slot gets ~2 Gbit/s at Gen1, ~4 at Gen2 ([Pi CM4 report](https://github.com/geerlingguy/raspberry-pi-pcie-devices/issues/143)) | no free slot without removing a card, and without GPUDirect the data stages through host memory anyway | $25–55 used |
| USB 3 → 2.5/5 GbE adapters between rigs (RTL8156B/BG, AQC111U) | widen the two-rig link beyond the onboard gigabit | 2.35 Gbit/s send, 1.4–2.3 receive (RTL8156B with `r8152`); ~3.45 Gbit/s (AQC111U); ~0.6 ms idle round trip with `r8152`, ~3 ms with `cdc_ncm` ([CNX](https://www.cnx-software.com/2022/02/20/fixing-performance-issues-with-realtek-rtl8156b-2-5gbe-usb-dongle-in-ubuntu/), [netdev](https://lists.openwall.net/netdev/2022/05/02/14), [ServeTheHome](https://www.servethehome.com/?p=49814)) | USB 3 sits inside the PCH, off the cards' links, so it costs no card; helps the two-rig split's prompt crossings (5.6 MB: ~43 ms on gigabit, ~17 ms at 2.5 GbE, ~12 ms at 5 GbE), not a decode crossing | ~€15–40 each (two per link) |
| USB 3 host-to-host cable | rig-to-rig | ~1.5 Gbit/s ([PL27A1 patch](https://lore.kernel.org/all/56C9CC84.7040705@gmail.com/T/)) | slower than a 2.5 GbE adapter, odd drivers | ~€20 |
| PCIe switch card (PLX PEX8747, ASM2824, mining 1-to-4 ASM1184e boards) | more ports, peer traffic kept inside the switch | 100–150 ns cut-through on PLX parts | in an x1 slot its uplink is x1; the cards' x1 links stay; peer traffic needs the driver's peer access, which these cards lack | $33–57 (1-to-4), $157–300 (PLX, ASM2824) |
| Thunderbolt / USB4 networking | rig-to-rig | ~9–12 Gbit/s | needs a controller and a slot the rig does not have | — |

## 8. The cards' other I/O, and the topology

- **Display outputs:** the P106 boards have no connectors, and no hardware mod that adds them was found (the Chinese
  "headless card" mods are driver patches that render on the P106 and show the picture through the iGPU, e.g.
  [NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher)). Even on a card with outputs, data sent as video
  (hsdaoh moves 184 MB/s through a $5 HDMI capture stick, [hsdaoh](https://github.com/steve-m/hsdaoh)) arrives frame by
  frame in the receiver's host memory: milliseconds of latency, then a PCIe copy anyway.
- **I²C, GPIO, SLI, NVLink:** GPU I²C runs at ≤ 400 kHz through the driver, not CUDA; the GPIOs belong to the driver
  (fan, power); GP106 has no SLI connector and no NVLink.
- **Card order:** at Gen1, alternating cards between the two switches does not help (in a full prompt pipeline every
  card transfers once per stage, [card hops](card_hops_x1.md) 4.5). After a Gen2 unlock the shared uplinks bind
  sooner, and the split could put concurrent prompt hops behind different switches (parked, Q999).
- **Fewer boundaries** (Q964) and **fewer bytes per hop** (Q963, below 8 bits) shrink the same budget from the other
  side; both are in progress.
- **Other boards:** the cards are x4 parts, so a board with x4 slots (Gen1 x4, est. ~800 MB/s per card, 4×) is the only
  hardware route to much more per-card bandwidth. It is a different machine, not this rig as sold (Q1000).
- **The M.2 slot** holds the SATA SSD; no PCIe port is enabled for it (the tree shows four root
  ports, all in use). Only if the slot is wired for PCIe and the firmware can enable a fifth port, an M.2-to-PCIe riser would give a tenth device x2 or x4 at Gen2 (a GPU, a NIC),
  a physical change for the owner (Q1001).

## 9. Every idea

Decode hop: one token's 10,880 B (today 126.7 µs). Bandwidth: one hop's sustained rate (today ~147 MB/s at 160 KB).
Every value est. unless the row says measured.

| Idea | Decode hop | Bandwidth per hop | Cost | Difficulty, risk | As sold | The one measurement | Row |
|---|---|---|---|---|---|---|---|
| Legs overlapped through host memory (streamed mailbox, persistent pipeline) | 75–85 µs | ~185 MB/s (the slower leg) | none | code, in progress; a waiting kernel needs a timeout | yes | the hop bench's streamed-mailbox columns at 10,880 B and 5.57 MB | Q682, Q684 |
| The rig's PCIe facts in full (`lspci -vv`: `LnkCap2`, `LnkCtl2`, ACS, MPS, BARs) | — | — | none | none (read-only) | yes | done: section 2 | Q990 |
| The driver's peer verdict per pair (`nvidia-smi topo -p2p r` / `w`), the `ForceP2P` string in `nvidia.ko`, IOMMU on or off | — | — | none | none (read-only) | yes | the outputs | Q991 |
| `ForceP2P=0x11`: forced peer-to-peer | ~55 µs | ~211 MB/s, one link | none | a module reload; possible hang or wrong copies | a driver option | checksummed `cudaMemcpyPeer` and peer stores on one pair behind a switch | Q992 |
| A peer window by I/O-memory registration of card B's Vulkan BAR1 buffer | ~55 µs | ~211 MB/s | none | untested; a freeze is possible | yes | one checksummed write from A into B's window | Q993 |
| Gen2 on the cards' links (volatile BAR0 writes, retrain; maybe a 2-byte driver patch) | 45–65 µs (~30 with peer-to-peer) | ~386 / 427 MB/s per card; ~400 per switch uplink when shared | none | a card off the bus needs a reboot; the P106 sequence is unverified | firmware state changed | `LnkSta` 5 GT/s on card 4, then the hop bench | Q994 |
| MPS 256 B on every card's path | 0: already 256 B (measured) | — | none | — | yes | done: the `lspci` (section 2) | Q995 |
| A 9-card star all-reduce through host memory, and S for one layer split nine ways | (tensor split: 96 × ~70 µs a token) | — | none | a bench | yes | all-reduce median and p99 at 2,720 and 5,120 B; S on one card | Q996 |
| Tensor or expert parallel decode of Flash-Next on nine cards | token est. ≥ 20–30 ms if S is high, a loss if low | prompts link-capped at est. ~200–400 tok/s | none | weeks of fork work; no compute 6.1 all-reduce today | yes | Q996 first | Q997 |
| USB 3 → 2.5/5 GbE adapters between two rigs | rig-to-rig: ~same | 117 → ~290–430 MB/s rig-to-rig | ~€15–40 each, two a link | low | adds parts | iperf3 and ping between the rigs, then Q505's prompt speed | Q998 |
| The split's card order chosen so concurrent prompt hops sit behind different switches | — | matters only once the uplinks bind (after Gen2) | none | low | yes | prompt speed with two orders | Q999 |
| A board with x4 slots for the cards | ~20–40 µs | Gen1 x4: ~800 MB/s per card | a different board | a rebuild | no | — | Q1000 |
| The M.2 slot's PCIe lanes, if any, for a tenth device | — | a device at x2/x4 Gen2 | an M.2 riser, ~€10–20 | a physical change | no | look at the slot's key and the root ports `lspci` lists | Q1001 |
| An FPGA between the cards (mailbox, or as their root port) | ≥ today | ≤ the cards' own links | $25–350 | high; needs a slot the rig lacks; an FPGA cannot write into a GeForce card | no | — | Q1002 |
| A microcontroller link (ESP32, RP2040/RP2350, STM32) | milliseconds | ≤ 10–75 MB/s | $5–15 | low | no | — | Q1003 |
| A NIC inside the rig (ConnectX-3, 10/40 GbE, InfiniBand) | ≥ today | ≤ the card's link; stages through host memory without GPUDirect | $25–55 used | removes a card | no | — | Q1004 |
| A card's display output as a data link | ≥ a frame (16.7 ms at 60 Hz) | ~180–300 MB/s into host memory | — | no connectors on the board; no mod known | no | — | Q1005 |
| I²C, GPIO, SLI or NVLink between the cards | — | ≤ ~50 KB/s (I²C at 400 kHz); SLI and NVLink absent | — | — | — | — | Q1006 |
| VBIOS cross-flash, strap resistors or added capacitors for a faster link | no change | no change | ~€5 (a CH341A programmer) | bricking | no | — | Q1007 |
| A PCIe switch card (PLX, ASM2824, 1-to-4 boards) in a card's slot | ≥ today | its uplink stays x1 | $33–300 | removes a card; peer traffic still needs the driver | no | — | Q1008 |
| Gen3 or wider links on this board (BIOS, soft straps) | — | — | — | the Base-U PCH and the ASM1184e stop at Gen2 x1; slots are one lane | no | — | Q1009 |
| nouveau, which owns BAR1 and could do peer writes | — | — | — | Pascal stuck at boot clocks; no CUDA | no | — | Q1010 |
| The tinygrad/aikitoria P2P patches, GDRCopy, GPUDirect RDMA | — | — | — | open modules (Turing+), resizable BAR, Data Center or RTX boards | no | — | Q1011 |

## Gaps

- Whether the P106-090's Gen2 register sequence works behind an ASM1184e or on a Base-U root port: no report exists for
  any card in either place.
- The ASM1184e's forwarding latency, cut-through or store-and-forward, and multicast: no public datasheet.
- The poll latency of a GPU reading host memory on a Gen1 x1 link behind a switch (est. 1.2–2.5 µs): the streamed
  mailbox bench measures it.
- Tensor parallelism's compute speedup on nine 5-SM cards (Q996).
- Whether the board's M.2 slot is wired for PCIe at all (no port is enabled for it).
