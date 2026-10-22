# Manli M-P106L9-N6G — 9× P106-090 GPU mining system

The Manli M-P106L9-N6G ("MANLI GPU Mining System P106-090 (6GB) X9") is an integrated GPU mining
system. It has a purpose-built motherboard carrying nine NVIDIA P106-090 6 GB cards, an Intel
Celeron host, an M.2 SSD and a wall of chassis fans, all in one enclosure. The product is
discontinued.

## Specifications

| Component | Specification |
|---|---|
| Model | Manli M-P106L9-N6G |
| GPUs | 9 × NVIDIA P106-090, 6 GB GDDR5 each (54 GB total) |
| GPU slots | 9 × PCIe x16 (mechanical), each wired as a single lane |
| PCIe switching | 2 × ASMedia ASM1184e 4-port PCIe 2.0 x1 packet switches |
| CPU | Intel Celeron 3865U (Kaby Lake), 2 cores / 2 threads, 1.8 GHz, 2 MB L3, 15 W |
| Chipset | Intel Sunrise Point-LP (on-package), PCIe root ports at 2.0 speed |
| Integrated graphics | Intel HD Graphics 610 |
| Memory | 1 × 4 GB DDR4 SO-DIMM at 2133 MT/s, single channel; 2 slots, 32 GB maximum |
| Storage | 64 GB M.2 SATA III SSD |
| Network | 1 × Gigabit Ethernet (Realtek RTL8111/8168) |
| Display output | 1 × HDMI 1.4 (from the integrated graphics) |
| USB | 2 × USB 3.0, 2 × USB 2.0 |
| GPU power input | 7 × 6-pin 12 V |
| Power supply | 1600 W, sold separately |
| Cooling | 10 × 90 mm 4800 rpm fans, 5 intake and 5 exhaust |
| Firmware | AMI UEFI BIOS A355M003 |
| Operating system | Windows 10 64-bit (optional at purchase); runs Linux as well |

## GPU: NVIDIA P106-090 6 GB

The P106-090 is a cut-down GP106 (the GTX 1060 die) made for mining. It has no display outputs and
a narrow PCIe interface. NVIDIA's reference P106-090 has 3 GB; Manli's board carries 6 GB.

| Property | Per card |
|---|---|
| GPU | GP106-090-A1, Pascal, 16 nm, PCI ID `10de:1c09` |
| CUDA compute capability | 6.1 (`sm_61`) |
| Streaming multiprocessors | 5 |
| CUDA cores | 640 |
| Memory | 6 GB GDDR5 (~6068 MiB usable by CUDA), 192-bit, 8 Gbps |
| Memory bandwidth | 192 GB/s theoretical, ~173 GB/s measured |
| L2 cache | 1.5 MB |
| Clocks | 1354 MHz base, 1531 MHz boost, up to 1911 MHz |
| FP32 | 1.96 TFLOPS at 1531 MHz |
| FP16 / FP64 | 1/64 and 1/32 of the FP32 rate |
| INT8 | `dp4a`, 4 × the FP32 rate |
| Tensor cores | none |
| ECC | none |
| Power limit | 75 W (adjustable 60–75 W); ~5 W idle |
| Thermal limits | 83 °C target, 98 °C slowdown, 101 °C shutdown |
| Host interface | PCIe 1.0 x4, running at PCIe 1.0 x1 in this system |
| Display outputs | none |
| VBIOS | 86.06.5B.00.0C |

The nine cards together provide 54 GB of VRAM in nine separate 6 GB pools, about 1.6 TB/s of combined
local memory bandwidth, and about 17.6 TFLOPS of FP32 compute.

## PCIe topology

Every GPU has a single PCIe lane. The four chipset root ports run at PCIe 2.0 x1. Two of them feed
ASM1184e switches, each of which fans out to four GPUs. A third port has one GPU of its own, and the
fourth serves the network controller. The GPUs link at PCIe 1.0 speed, which is the limit of the
P106-090.

```
Celeron 3865U + Sunrise Point-LP PCH
├─ root port #3  (Gen2 x1) ─ ASM1184e switch ─┬─ GPU0  03:00.0  Gen1 x1
│                                              ├─ GPU1  04:00.0  Gen1 x1
│                                              ├─ GPU2  05:00.0  Gen1 x1
│                                              └─ GPU3  06:00.0  Gen1 x1
├─ root port #4  (Gen2 x1) ──────────────────── GPU4  07:00.0  Gen1 x1
├─ root port #5  (Gen2 x1) ──────────────────── Realtek GbE NIC
└─ root port #6  (Gen2 x1) ─ ASM1184e switch ─┬─ GPU5  0b:00.0  Gen1 x1
                                               ├─ GPU6  0c:00.0  Gen1 x1
                                               ├─ GPU7  0d:00.0  Gen1 x1
                                               └─ GPU8  0e:00.0  Gen1 x1
```

GPU numbers follow CUDA and nvidia-smi order, which is PCI bus order.

Host↔GPU bandwidth (pinned memory, 64 MB transfers):

| Scenario | Per GPU | Total |
|---|---|---|
| One GPU at a time, host → GPU | ~189 MB/s (GPU4: ~200 MB/s) | — |
| One GPU at a time, GPU → host | ~213 MB/s | — |
| All nine at once, host → GPU | ~90–120 MB/s behind a switch, ~200 MB/s for GPU4 | ~715 MB/s |

PCIe 1.0 x1 carries 250 MB/s per direction before protocol overhead, and PCIe 2.0 x1 carries 500 MB/s.
Each switch uplink is shared by four GPUs.

The GPUs have no peer-to-peer access, so all GPU-to-GPU traffic passes through host memory. In
`nvidia-smi topo -m`, GPUs behind the same switch show as `PIX` and all other pairs as `PHB`.

PCIe power saving (ASPM) is disabled on every GPU link, every switch port and every root port; the root ports don't
support ASPM at all. The network controller's root port runs at PCIe 1.0 x1 (250 MB/s), still twice what Gigabit
Ethernet needs.

## Power and cooling

- **Power supply:** 1600 W, sold separately. Nine GPUs at their 75 W limit draw at most 675 W.
- **GPU power input:** 7 × 6-pin 12 V connectors.
- **Fans:** ten 90 mm, 4800 rpm chassis fans, five intake and five exhaust, cool the GPUs and the
  board. The fans are not monitored or controlled by the OS.
- **GPU temperatures:** 23–30 °C at idle; 46–57 °C peak per card during sustained single-card inference.
- **Clocks under load:** compute work runs in the P0 performance state with the full memory clock (4004 MHz,
  sampled as 4006 MHz). The only supported memory clocks are 4004 and 405 MHz, and application clocks are not
  supported.
- **Power limit under load:** a card generating text draws ~60–70 W on average and spends much of its time at the
  75 W limit, which pulls the core clock down to ~1700–1760 MHz from its 1911 MHz maximum
  ([single-card calibration](benchmarks/2026-10-22-single-card-calibration.md)).

## Workload characteristics

The x1 links are no problem for mining, where the host sends small jobs and gets hashes back. For
other workloads they set the limits:

- **Loading data onto the GPUs is slow.** Filling one card's 6 GB takes over 30 s, and filling all
  nine (54 GB) takes about 80 s even with transfers in parallel. The SSD (~485 MB/s sequential read),
  its 64 GB capacity and the 4 GB of system RAM further limit staging of large files.
- **Multi-GPU work should keep data local.** With no peer-to-peer access and 100–200 MB/s per GPU,
  splitting a job so that each GPU works on its own slice and passes only small results along suits
  this system. Schemes that synchronise large tensors on every step do not.
- **The host is weak.** Two Celeron cores and 4 GB of single-channel RAM bottleneck anything that does
  real work on the CPU. RAM can be expanded to 32 GB using the free second slot.

## Software support

- **Driver:** the NVIDIA 580 branch is the last driver line that supports Pascal GPUs. It continues as
  a legacy branch that gets occasional fixes, such as for newer Linux kernels.
- **CUDA:** CUDA 12.x is the last toolkit line that supports Pascal. Build for `sm_61` with CUDA 12.9,
  even though the 580 driver reports CUDA 13.0.
- **Tested configuration:** Ubuntu 24.04 LTS, NVIDIA driver 580.178.04, CUDA 12.9.2.
- **Usable GPU memory:** with the desktop session running, Xorg holds 4 MiB per card, and a CUDA program sees
  6019 MiB free of 6068 MiB on each card before allocating anything.
- **CUDA graphs:** llama.cpp disables CUDA graphs on Pascal GPUs at runtime, whatever its build options say
  ([small MoE benchmark](benchmarks/2026-10-22-small-moe-overhead.md)).

## Mining performance

Manli rated the system at:

| Coin | Hash rate | Power |
|---|---|---|
| Ethereum | 200 MH/s ± 5 % | 800 W |
| Ethereum Classic | 200 MH/s ± 5 % | 800 W |
| Monero | 3800 H/s ± 5 % | 600 W |

Ethereum moved to proof of stake in September 2022 and no longer uses mining. Monero has used the
CPU-oriented RandomX algorithm since November 2019.

## References

- Manli, M-P106L9-N6G product page (archived):
  <https://web.archive.org/web/20231001152427/https://manli.com/en/product-detail-MANLI_GPU_Mining_System_P106-090_(6GB)_X9-206.html>
- TechPowerUp GPU Database, Manli P106L Mining Card (F347G) 6 GB:
  <https://www.techpowerup.com/gpu-specs/manli-p106l-mining-card-f347g-6-gb.b6880>
- Phoronix, "NVIDIA Confirms 580 Linux Driver Is The Last For Maxwell / Pascal / Volta":
  <https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW>
