# Forced peer-to-peer on the rig: the driver says yes, nothing arrives (2026-11-25, Q992)

NVIDIA's driver refuses peer-to-peer on every card pair of the rig ("chipset not supported",
[2026-11-25_p2p_verdict](../2026-11-25_p2p_verdict/README.md)). This run forced it on with NVIDIA's own override and
checked whether the data gets through. It doesn't, on any path. The cards themselves can do it, so the board is the
limit. Result page: [diy_card_interconnect.md](../../research/notes/diy_card_interconnect.md), section 4; registry
Q992.

## Results (rig 1, driver 580.178.04, kernel 7.0, nine P106-090)

| Setting | `topo -p2p r` | Pair, path | Test | Words that arrived |
|---|---|---|---|---|
| stock | CNS everywhere | — | — | — (no peer access) |
| `ForceP2P=0x11`, `DmaRemapPeerMmio=0`, IOMMU translated | OK everywhere | 0 → 1, one ASM1184e switch | `cudaMemcpyPeer`, kernel writes, kernel reads, 64 MiB, 2 rounds | **none**; the second round's peer read hung until the 240 s timeout |
| `ForceP2P=0x11`, card 4 alone in passthrough (its IOMMU group set to `identity`) | OK everywhere | 4 → 5, chipset port to the second switch | kernel writes, 16 MiB, 2 rounds; `cudaMemcpyPeer`, 1 round | **none** (every word still 0) |
| the same | OK everywhere | 4 → 3, chipset port to the first switch | kernel writes, 16 MiB, 2 rounds | **none** (every word still 0) |
| every setting | — | every pair tested | the control copy through host memory | **all**, every round |

- **Nothing in the kernel log** in any test: no Xid, no AER, no IOMMU fault. The cards stayed healthy, with no stuck
  process and every card at idle afterwards. After the stock driver was loaded again, the control copy passed on pairs
  covering all nine cards.
- **My earlier run in the P2P guide project (same rig and driver) did the remaining case.** `ForceP2P=0x11` with
  `iommu=pt` on the kernel line, two cards behind one switch. The write reached the chipset's root port with the right
  address inside the other card's register window, and the port logged it as an Unsupported Request (AER). The switch
  passed the write up instead of across, and the chipset won't send it back down the port it came from. The four
  cards behind that port needed a reboot
  ([how-to-enable-p2p-on-nvidia-gpus, pascal-and-volta.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/pascal-and-volta.md#the-test-on-a-mining-box-the-driver-said-yes-the-hardware-said-no)).
  Its companion project Hidden GEM, sharing memory through a dma-buf, saw 0 of 16,777,216 words arrive on every path, again without
  an error.
- **The same cards do it elsewhere:** two P106-090 on a B550 board passed that project's integrity test at 0.84 GB/s
  ([boards.md](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/blob/main/docs/boards.md)).

What it means: the board carries no peer traffic.
- The ASM1184e switches send traffic between their own cards up to the chipset.
- The chipset neither sends it back down the port it came from (Unsupported Request) nor passes it from one of its
  ports to another (silently dropped).

So no setting helps: not the override key, IOMMU passthrough, plain bus addresses, nor a peer window that bypasses
the driver's permission (Q993), which uses the same path. Card-to-card traffic stays staged through host memory.

## How it was run

On rig 1, in about 15 minutes, with the day's first row paused for it. Only the driver options
changed, on the `modprobe` command line, and the stock driver was loaded again at the end. Nothing was written to
`/etc`.

```bash
bench/p2p_check/build.sh                                            # on the build machine, CUDA 12.9 container
ssh "$RIG" 'bash -s' -- remap0 < run/reload_driver.sh               # ForceP2P=0x11, DmaRemapPeerMmio=0
ssh "$RIG" 'timeout 240 ./p2p_check --pairs 0:1 --tests copy,store,load --rounds 3'
# card 4 alone in passthrough: driver unloaded, then
#   echo identity > /sys/kernel/iommu_groups/<card 4's group>/type
#   modprobe nvidia NVreg_RegistryDwords=ForceP2P=0x11   (and nvidia_uvm, nvidia_modeset, nvidia_drm)
ssh "$RIG" 'timeout 150 ./p2p_check --pairs 4:5,4:3 --tests control,store --rounds 2 --mb 16'
ssh "$RIG" 'timeout 90 ./p2p_check --pairs 4:5 --tests control,copy --rounds 1 --mb 16'
# back to stock: driver unloaded, the group's type back to DMA-FQ, modprobe nvidia without options
ssh "$RIG" 'timeout 150 ./p2p_check --pairs 0:1,2:3,4:5,6:7,8:0 --tests control --rounds 1 --mb 16'
```

The first pair test ran before the control test and the zero count were added to `p2p_check`; its wrong words are
counted the same way.

| File | What |
|---|---|
| [run/reload_driver.sh](run/reload_driver.sh) | reloads the NVIDIA modules as `stock`, `remap0` (`ForceP2P=0x11`, `DmaRemapPeerMmio=0`) or `iommu` (`ForceP2P=0x11`) |
| [../../../bench/p2p_check/](../../../bench/p2p_check/README.md) | the test program (binary md5 `e1b9f6b5b870bc5366fdbc80f766a6dd`, the final build) |

Provenance: the rig's own output (kept privately with the raw logs). The earlier `iommu=pt` result and the B550
result come from my published P2P guide, linked above.
