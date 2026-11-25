# Peer-to-peer and Gen2 on the P106 rig: everything tried (2026-11-25)

On 2026-11-25 I tried, on one of the rigs, both ways of making the cards talk faster that the
[DIY interconnect note](diy_card_interconnect.md) left open. One was **peer-to-peer** (cards writing straight into each
other's memory instead of through host memory); the other was **Gen2** (running the cards' PCIe links at 5 GT/s instead
of 2.5). Neither works on this board. This page records what was tried, in what order, with which settings, what each
register read, what broke and how it was recovered. Whoever comes back with new knowledge can start from here
without repeating a step.

What either would have bought, for scale: a decode hop between two cards takes ~127 µs today. Peer-to-peer was est.
~55 µs and Gen2 est. 45–65 µs, or −0.5 to −0.65 ms a generated token, about 1 %. Prompt hops would gain more
([diy_card_interconnect.md](diy_card_interconnect.md), section 1).

## At a glance

| What | Setting | Result | Registry, data |
|---|---|---|---|
| Peer-to-peer, the driver as installed | read-only check | the driver refuses every pair: `CNS` (chipset not supported), the cards themselves not blocked | Q991, [2026-11-25_p2p_verdict](../../data/2026-11-25_p2p_verdict/README.md) |
| Peer-to-peer forced, two cards behind one switch | `ForceP2P=0x11`, `DmaRemapPeerMmio=0`, IOMMU translated | the driver says OK; **no word arrives** (copy, kernel writes, kernel reads); a peer read hangs | Q992, [2026-11-25_p2p_force](../../data/2026-11-25_p2p_force/README.md) |
| Peer-to-peer forced, across the chipset | `ForceP2P=0x11`, the writing card's IOMMU group in passthrough | **no word arrives**, no error anywhere | Q992, same folder |
| Peer-to-peer forced, two cards behind one switch, earlier run | `ForceP2P=0x11`, `iommu=pt` | the chipset port logs the write as an Unsupported Request; four cards needed a reboot | [P2P guide](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/pascal-and-volta.md#the-test-on-a-mining-box-the-driver-said-yes-the-hardware-said-no) |
| Memory sharing through a dma-buf (Vulkan), every path, earlier run | the Hidden GEM driver change | **0 of 16,777,216 words** arrive, no error | [Hidden GEM](https://github.com/Cinderella-Man/hidden-gem/blob/main/docs/volta-dmabuf.md#on-a-board-that-does-not-carry-peer-traffic) |
| A peer window through I/O-memory registration | not run | parked: its writes take the same path | Q993 |
| Gen2, first try, card 4 | 2-byte driver copy; the P104-100 push on a card the driver had initialized | the Gen2 declaration never latched; **a retry I added broke the link**: card off the bus, reboot | Q994, [2026-11-25_p106_gen2](../../data/2026-11-25_p106_gen2/README.md) |
| Gen2, second try, card 4 | fresh boot, card untouched since power-on, the same push once | **the card died at the speed write**: off the bus, reboot | Q994, same folder |
| Gen3 or wider links | — | impossible on this board: the chipset and switches stop at Gen2 x1, every slot is one lane | Q1009 |

**Verdict:** card-to-card traffic stays staged through host memory on this board. The remaining software levers for
the hop are in [diy_card_interconnect.md](diy_card_interconnect.md), section 5, and registry Q682.

## The hardware facts underneath

These decide every result below; each was read on the rig unless a source is linked.

- **The path between cards** ([mining_rig.md](../../mining_rig.md), "PCIe topology" and "Boards and connectors"):
  - Cards 0–3 sit behind one ASMedia ASM1184e switch, on chipset root port #3 (`00:1c.0`).
  - Card 4 sits alone on root port #4 (`00:1c.3`).
  - Cards 5–8 sit behind a second ASM1184e, on root port #6 (`00:1c.5`).
  - Every link is x1. The cards train at 2.5 GT/s; the switches and root ports offer 5 GT/s.
- **The ASM1184e has no ACS capability** (no "Access Control Services" in `lspci -vv`), so it can't be told to
  redirect traffic. It also **does not route a write from one of its cards to another**: it passes the write up to the
  chipset (the Unsupported Request below).
- **The chipset (Sunrise Point-LP, inside the Celeron 3865U):**
  - It neither sends traffic back down the port it came from (Unsupported Request), nor passes it from one root port to
    another (silently dropped).
  - Its root ports have ACS, and the kernel turns redirect on through its "Intel SPT PCH root port ACS workaround"
    (boot log). `lspci` reads the standard register offsets and shows every ACS control bit off, but this chipset
    keeps them elsewhere.
- **The IOMMU is on, translating,** by default in Ubuntu's kernel (`CONFIG_INTEL_IOMMU_DEFAULT_ON=y`, groups of type
  `DMA-FQ`). There are 22 groups, nine for the nine cards, and each card behind a switch shares its group only with its
  own switch port.
- **The cards** (P106-090, `10de:1c09`):
  - The PCIe capability is version 1, so config space has no `LnkCap2` or `LnkCtl2`. `LnkCap` is `0x00453c41`:
    2.5 GT/s, x4.
  - BAR0 is 16 MB and BAR1 256 MB.
  - BAR0 from `0x88000` mirrors the PCIe config registers (`0x88084` = `LnkCap`, `0x88088` = `LnkCtl`/`LnkSta`,
    `0x880A4` = the `LnkCap2` value), as on the P104-100; checked equal on the rig.
- **The driver:** NVIDIA 580.178.04, closed (the last branch for Pascal). Its `nvidia.ko` holds the registry keys
  `ForceP2P`, `RMForceP2PType`, `RMPcieP2PType`, `PeerMappingOverride` and `DmaRemapPeerMmio`, and `modinfo` lists
  the module parameters `NVreg_RegistryDwords`, `NVreg_RegistryDwordsPerDevice` and `NVreg_DmaRemapPeerMmio`.
- **NVIDIA's lists, in its open kernel modules (identical in release 580.196.11):**
  - The chipset table has the Skylake and Kaby Lake desktop chipsets, not the laptop-class `0x9Dxx` of this Celeron,
    and falls back to "not supported".
  - The accepted PCIe switches (`IS_SUPPORTED_PCIE_SWITCH`, [pcie_switch_ref.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/src/common/inc/swref/published/pcie_switch/pcie_switch_ref.h))
    are NVIDIA BR03/BR04, PLX 87xx/97xx, PMC-Sierra 85xx/40xx and Mellanox `0x1976`, no ASMedia.
- **Linux's list of host bridges that route peer traffic** (`pci_p2pdma_whitelist`,
  [p2pdma.c](https://github.com/torvalds/linux/blob/master/drivers/pci/p2pdma.c)) has Intel server parts and one Haswell
  client bridge, and no Skylake or Kaby Lake client part.

## Peer-to-peer

### Why the driver refuses

The driver decides per pair in `p2p_caps.c` ([source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/src/nvidia/src/kernel/platform/p2p/p2p_caps.c)):
- `CNS` when the chipset isn't in its table and the two cards share no switch it accepts;
- `GNS` when the GPU carries a "P2P disabled" property;
- `TNS` for some Intel root-port layouts.

`ForceP2P` is checked first and overrides all three. On the rig, `nvidia-smi topo -p2p r` and `-p2p w` reported `CNS`
for all 72 ordered pairs, including the pairs behind one switch (`topo -m` shows those as `PIX`). No pair showed
`GNS`, so the P106-090 itself does not block peer-to-peer. The same cards pass a peer-to-peer integrity test on a
B550 desktop board at 0.84 GB/s ([P2P guide, boards.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md)).

### Test 1: forced, inside one switch, plain bus addresses

- **Setting:** the NVIDIA modules reloaded with `NVreg_RegistryDwords=ForceP2P=0x11 NVreg_DmaRemapPeerMmio=0`, on the
  `modprobe` command line only, nothing in `/etc` ([reload_driver.sh](../../data/2026-11-25_p2p_force/run/reload_driver.sh)).
  - `ForceP2P=0x11` forces peer reads and writes on.
  - `DmaRemapPeerMmio=0` makes the driver hand a card the other card's plain bus address instead of mapping it
    through the IOMMU. With the IOMMU translating, a remapped address is one no switch port claims, so it could never
    stay inside the switch.
- **Driver view:** `topo -p2p r` turned to OK for every pair, and `cudaDeviceCanAccessPeer` said yes.
- **Data** ([bench/p2p_check](../../../bench/p2p_check/README.md), cards 0 → 1, 64 MiB, every word checked):

  | Test | Round 0 | Round 1 |
  |---|---|---|
  | `cudaMemcpyPeer` 0 → 1 | all 16,777,216 words wrong | all wrong |
  | kernel on 0 writes into 1 | all wrong | all wrong |
  | kernel on 0 reads from 1 | all wrong | hung until the 240 s timeout |

- **Afterwards:** the kernel log had no Xid, no AER and no IOMMU fault. No process was left stuck, and the cards
  stayed healthy.

### Test 2: forced, across the chipset, the writer in IOMMU passthrough

- **Setting:** `ForceP2P=0x11` with the default `DmaRemapPeerMmio` (1). Card 4 was alone in IOMMU group 15, so its
  group was switched to passthrough at runtime without a reboot: unload the driver (which unbinds the card),
  `echo identity > /sys/kernel/iommu_groups/15/type`, then load it again. Card 4 then writes with physical addresses
  and no translation. The hardware passthrough capability is present (the IOMMU's `ecap` PT bit).
- **Data** (16 MiB, every word checked, a control copy through host memory in every round):

  | Pair | Kernel writes, 2 rounds | `cudaMemcpyPeer`, 1 round | Control |
  |---|---|---|---|
  | 4 → 5 (chipset port to the second switch) | all wrong, every word still 0 | all wrong, every word still 0 | right |
  | 4 → 3 (chipset port to the first switch) | all wrong, every word still 0 | — | right |

- **Afterwards:** nothing in the kernel log. The group was set back to `DMA-FQ` and the stock driver reloaded.

### The earlier run, and the memory-sharing project

- **IOMMU passthrough, two cards behind one switch:** `ForceP2P=0x11` with `iommu=pt` on the kernel line, cards 0 → 1.
  - Root port `00:1c.0` logged an AER Unsupported Request with TLP header `0x40701002 0x030000ff 0xdc139068 …`: a
    memory write from card 0 to an address inside card 1's register window.
  - So the driver addressed it correctly; the switch passed it up, and the chipset refused to send it back down.
  - The four cards behind that port needed a reboot.
- **[Hidden GEM](https://github.com/Cinderella-Man/hidden-gem):** a dma-buf memory share through `nvidia-drm` (a small
  change, the same driver version). The import worked on every path, and 0 of 16,777,216 words arrived on any of them,
  with no error.

### What it means, and what would change it

The board carries no peer traffic:
- writes inside a switch go up to the chipset and are refused there;
- writes between chipset ports are dropped;
- nothing reports it, except the Unsupported Request when the IOMMU is in passthrough.

No driver setting changes the routing. A peer window that bypasses the driver's permission (Q993) sends the same
writes down the same path, so it is parked.

Peer-to-peer comes back only on hardware that routes it, which means none of the rigs as sold:
- cards behind a switch that routes peer traffic between its ports (any spec-compliant switch, with `ForceP2P` where
  the driver doesn't accept it);
- cards on CPU root ports;
- cards on a chipset that forwards peer traffic.

## Gen2 on the cards' links

### The three layers of the Gen1 cap

1. **The card advertises Gen1 only.** `LnkCap` speed 2.5 GT/s and the `LnkCap2` mirror `0x2`, at power-on and after.
2. **The driver's start-up sets override bits.** `PRIV_MISC_1` at BAR0 `0x8841C` reads `0x00340500` at power-on
   and `0xc0b46500` after the driver has initialized the card.
3. **The closed driver forces Gen1 on every CUDA start.** Its link-speed function (`_nv017105rm`, `set_link_speed(mode)`,
   mode = PCIe generation) is called with mode 1 from `rm_init_adapter`. This is the finding of
   [WJNKAC/nvidia-p104-unlock-gen2-for-linux](https://github.com/WJNKAC/nvidia-p104-unlock-gen2-for-linux) on a
   P104-100 with the same driver.

On top of that, the board allows Gen2 at most, x1.

The method that worked on that P104-100 under Linux (read, not run; my own implementations below):
- a 2-byte driver patch so the mode-1 call does nothing;
- right after the driver's probe and before anything opens the card: the declaration unlock (XP3G overrides at
  `0x8E110`–`0x8E12C`, bits 15/13/11 set and 12/14 cleared in `0x8841C`, bit 0 set and 12 cleared in `0x88610`), the
  Gen2-disable bit cleared (`0x8C2C0` bit 2), the target speed (`0x8C040` bits 19:18 = 1), the training trigger
  (`0x8872C` = 6), and a retrain from the port above.

They measured 1.561 / 1.708 GB/s against 0.779 / 0.855 at Gen1 x4.

### The driver copy

[patch_gen1_bail.py](../../data/2026-11-25_p106_gen2/run/patch_gen1_bail.py):
- **Where:** it finds a 32-byte sequence that must occur exactly once. In the rig's `nvidia.ko` (md5
  `e225b2a2f42328f055384f491eb6c11a`) it does, at file offset `0x12fe56`: `.text` at `0x2a90`, `_nv017105rm` at
  `.text+0x12d3c0`, so the function's start + 6.
- **What it changes:** `lea -0x1(%rdx),%eax` becomes `lea -0x2`, and `cmp $0x2,%eax` becomes `cmp $0x1`. So mode 1
  takes the existing early return, which returns 0 (success); modes 2 and 3 run as before. The disassembly before and
  after confirms it; `cmp -l` shows exactly two bytes changed.
- **Loading it:** the copy carries Canonical's signature, which no longer matches. Linux refuses a module whose
  signature fails ("Key was rejected by service") even when signatures are not enforced (`sig_enforce` N, Secure Boot
  off). So the script drops the appended signature (750 bytes plus the 40-byte trailer), and the unsigned module loads
  with `insmod`, marking the kernel tainted (`E`) until the next boot. The installed file was never touched.

### Card 4's registers in each state

| BAR0 register | Power-on, before any driver start-up (second try) | After the driver's start-up (first try, before the push) | After the first push |
|---|---|---|---|
| `0x88084` `LnkCap` mirror | `0x00453c41` | `0x00453c41` | `0x00453c41` |
| `0x88088` `LnkCtl`/`LnkSta` mirror | `0x10110140` | `0x10110140` | `0x10110140` |
| `0x880A4` `LnkCap2` mirror | `0x00000002` | `0x00000002` | `0x00000002` (no latch) |
| `0x8841C` `PRIV_MISC_1` | `0x00340500` | `0xc0b46500` | `0xc0b4ad00` |
| `0x88610` `VSEC_HIERARCHY` | `0x00000000` | `0x00000000` | `0x00000001` |
| `0x8C040` `LINK_CONFIG_0` | `0x20089000` (speed field 2) | `0x20085800` (speed field 2) | `0x20045800` (speed field 1) |
| `0x8C2C0` `CYA_0` | `0x060711b3` | `0x068711b3` | `0x068711b3` |
| `0x8C1C0` | — | `0x00110036` | `0x00110036` |
| `0x8C300` | — | `0x040a00c0` | — |
| `0x2241C` | — | `0x00000000` | — |

The P104 notes give `0xc0b40500` / `0xc0b48d00` for its `0x8841C` before the unlock and `0xc0b4ad00` after; the
P106's driver start-up sets bits 13 and 14 instead. The CMP 100-210 notes
([mintoriakamoto](https://github.com/mintoriakamoto/CMP100-210xFULLTENSORSxGEN3x1)) give `0x00340500` as the value
with the clamp cleared. The P106's power-on value already is that, and the card still advertises Gen1 only, so on the
P106 the Gen1 declaration doesn't come from these override bits alone.

### First try (card already initialized by the driver)

Card 4 had been through four driver loads that day.

1. **Load the copy.** With the stock modules unloaded, the copy went in by `insmod`. The driver's udev rule then
   loaded the display modules on its own, and they initialize every card; I unloaded them before the push.
2. **Push** ([gen2_push.py](../../data/2026-11-25_p106_gen2/run/gen2_push.py)). It refuses to write unless the BAR0
   mirror equals config space. Every write went in while the link was stable, then the trigger, then 0.15 s later a
   retrain from the port.
   - Every write landed (table above).
   - The `LnkCap2` mirror stayed `0x2` and the link came back at 2.5 GT/s.
   - This is the P104 project's "stuck state": the unlock bits written (`0x8841C & 0xA800` nonzero) but the
     declaration not latched. Their notes say every rescue fails in it and a reboot clears it.
3. **My mistake, the retry** ([retry_latch.py](../../data/2026-11-25_p106_gen2/run/retry_latch.py); don't run it).
   - It wrote `0x8872C` = 0, then 6. The 0 is a step the P104 sequence never takes.
   - It polled the card's BAR0 for 10 s while the link retrained. The P104 project reports user-space accesses during
     a link change killing its card in two tries out of two.
   - The declaration never latched. After the next retrain the card answered `ffffffff` to every config read, and the
     port's `LnkSta` read `0x1811` (training forever, data link down).
4. **Recovery, all failed:**
   - a retrain from the port;
   - removing the card from the PCI tree, a secondary bus reset on the port (`BRIDGE_CONTROL` bit 6), and a rescan (a
     bus reset is signalled over the link, so it can't reach a card whose link is down);
   - a link disable and re-enable on the port;
   - the port's target speed set to 2.5 GT/s with a retrain.
   
   The slot has no power control (`SltCap: PwrCtrl-`). The stock driver was reloaded for the other eight cards, and a
   normal reboot brought card 4 back.

### Second try (a fresh boot, card untouched since power-on)

1. **The boot.** A one-time GRUB entry ([gen2_fresh_boot.sh](../../data/2026-11-25_p106_gen2/run/gen2_fresh_boot.sh)
   `install-entry`) copied the default entry into `/boot/grub/custom.cfg`. GRUB's `41_custom` reads that file
   without `update-grub`. `grub-reboot` picked it for one boot. The entry added:
   - `modprobe.blacklist=nvidia`;
   - `module_blacklist=nvidia_drm,nvidia_modeset` (the kernel itself refuses the display modules, which the udev rule
     loads by name and which initialize every card);
   - `systemd.mask=nvidia-persistenced.service`.
2. **What the boot did:**
   - the stock `nvidia` and `nvidia_uvm` still loaded at ~21 s, by name, despite `modprobe.blacklist`;
   - the display modules were refused ("Module nvidia_modeset is blacklisted"), and the persistence daemon stayed off;
   - nobody opened a card, and card 4's registers were the power-on values in the table.
3. **The push, once.**
   - The stock modules were unloaded and the copy loaded; the display modules were refused again, and nobody opened
     a card.
   - Every check passed, and `LINK_CONFIG_0` read a valid value just before the speed write.
   - The read-back right after that write returned `ffffffff` from every register: the card died at the speed write
     or just after it.
   - Port `LnkSta` read `0x1811` again. No recovery was tried, since it had failed on this state once already.
4. **Afterwards:** card 4 was removed from the PCI tree and the stock driver loaded for the other eight. The entry was
   removed (`grub.cfg` was never changed). The card needs a normal reboot, as after the first try.

### Why it fails, as far as the evidence goes

There is no good window on this rig:
- **Before the driver's start-up,** the card's link block is not set up. `LINK_CONFIG_0` reads `0x20089000`, not
  `0x20085800`, and writing its speed field killed the link. The P104 notes warn that writing the speed field without
  its preparatory steps kills the card. Their card sat in a desktop whose firmware may have initialized it at boot; on
  this headless rig only the NVIDIA driver ever does.
- **After the driver's start-up,** the declaration did not latch: the stuck state.

The P106's start-up sets different override bits from the P104's (bits 13 and 14 of `0x8841C`), so the P104 unlock
may simply not apply. The one public P106-090 attempt, a Windows tool, also lost its card after the reset
([issue #1](https://github.com/DaZuiZZZZ/P106-Unlock-Gen2/issues/1)).

### Not tried, each costing a reboot when it fails

These are listed so nobody mistakes them for tried; Q994 is parked behind the first condition below.

- **The other P104 tool's sequence** ([skbskhud/p104-unlock-v3](https://github.com/skbskhud/p104-unlock-v3)), on a
  card the driver initialized once after a fresh boot: it clears bits 13 and 14 of `0x8841C` (`&= ~0x6000`) instead of
  setting 13. It also writes `0x2241C |= 0x81`, `0x8C300 &= ~0x10` and `0x8C1C0` bits 19:16 = 2, then the target speed
  on both link ends and a retrain from the root port.
- **The same writes from a kernel module** (`writel`, as the P104 project's `gen2kr`) instead of a user-space mmap.
- **A hook inside the driver** between the card's start-up and its first `set_link_speed`, if such a point exists. It
  would need the driver's internals traced on a P106.
- **The UEFI pre-boot tool** that showed a P106-090 at 5 GT/s x16 on a desktop board
  ([P106-90-Unlock-X16](https://github.com/DaZuiZZZZ/P106-90-Unlock-X16)): its register writes are not published, and
  running a third party's firmware tool is the owner's decision.

**Q994 reopens when a sequence is shown to work on a P106-090 elsewhere.**

## Operating notes for the rig, learned on the day

- **Reloading the driver:**
  1. stop `nvidia-persistenced`;
  2. `modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia`;
  3. `modprobe nvidia <options>`;
  4. load the other three modules and start the daemon.

  Options on the `modprobe` line are gone at the next boot. The NVIDIA udev rule loads `nvidia-modeset`, `nvidia-drm`
  and `nvidia-uvm` by name and starts the persistence daemon whenever the driver appears, also after an `insmod`.
- **Holding the driver back for one boot:** `modprobe.blacklist=nvidia` did not stop the driver from loading by name
  at boot. `module_blacklist=` (the kernel's own list) does refuse a module outright, and `systemd.mask=` keeps a unit
  off for one boot. A one-time GRUB entry in `/boot/grub/custom.cfg` plus `grub-reboot` needs no `update-grub`.
- **A modified driver module** must have its signature removed, even with signature enforcement off.
- **Reading a card's registers:** the sysfs `resource0` file can be mapped while the driver is bound.
- **Switching one card to IOMMU passthrough** at runtime works when its IOMMU group holds only that card: unbind, write
  `identity` to the group's `type`, rebind. Card 4 qualifies; the cards behind a switch share their group with their
  switch port.
- **A card whose link died** stays dead until a reboot: config reads give `ffffffff`, and the port shows `LnkSta`
  `0x1811`. Removing it from the PCI tree (`echo 1 > /sys/bus/pci/devices/<card>/remove`) lets the driver run the
  other eight cleanly. A normal warm reboot restored the card after the first try.
- **On these cards:** never write 0 to `0x8872C`, never touch BAR0 while a link retrains, and never retry a failed push.

## Tools made on the day

- [bench/p2p_check](../../../bench/p2p_check/README.md) checks every word of a peer copy, peer write or peer read
  against a control copy through host memory, counts words that stayed 0, and times peer and staged copies. With
  `--host-cards` it does the same for host ↔ card copies.
- [reload_driver.sh](../../data/2026-11-25_p2p_force/run/reload_driver.sh) reloads the driver as stock, with
  `ForceP2P` and `DmaRemapPeerMmio=0`, or with `ForceP2P` alone.
- [patch_gen1_bail.py](../../data/2026-11-25_p106_gen2/run/patch_gen1_bail.py),
  [gen2_push.py](../../data/2026-11-25_p106_gen2/run/gen2_push.py) and
  [gen2_fresh_boot.sh](../../data/2026-11-25_p106_gen2/run/gen2_fresh_boot.sh) are the Gen2 steps, with their
  refusals and checks.
- [p2p_verdict.sh](../../data/2026-11-25_p2p_verdict/run/p2p_verdict.sh) is the read-only check of the driver's
  verdict, the keys, the IOMMU and ACS.

Gen1 host-copy baseline for comparison (card 4, pinned memory, 64 MiB): 200 MB/s host → card and 213 MB/s
card → host. A 10 KB copy took 62.3 µs and 57.8 µs. Card 0, behind a switch: 189 / 213 MB/s, 66.5 / 61.1 µs.
