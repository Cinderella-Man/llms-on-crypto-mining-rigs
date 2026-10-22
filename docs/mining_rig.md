# Manli M-P106L9-N6G (9× P106-090) — hardware notes

The Manli **M-P106L9-N6G** ("MANLI GPU Mining System P106-090 (6GB) X9") is a turnkey GPU mining box: a
purpose-built motherboard with nine NVIDIA P106-090 mining cards, a low-power Intel Celeron host, a
small SSD and a fan wall, all in one chassis. Manli lists it as discontinued.

Everything below was checked on a real unit over SSH on **2026-10-22** (Ubuntu 24.04, NVIDIA driver
580.178.04), unless it is marked as a manufacturer spec. Where the manufacturer's spec sheet or the
public GPU databases disagree with the hardware, the hardware wins and the difference is called out.

Legend: ✅ verified on the unit · ⚠️ true only with a caveat · ❓ manufacturer spec, cannot be checked
from software (needs physical inspection)

## At a glance

| Component | Manli spec | Observed on this unit | |
|---|---|---|---|
| GPUs | 9 × P106-090 (6 GB) | 9 × NVIDIA P106-090 (GP106, PCI ID `10de:1c09`), 6 GB each, VBIOS 86.06.5B.00.0C | ✅ |
| GPU slots | 9 × PCI-E x16 | Every GPU link runs at **PCIe 1.0 x1**; 8 of 9 sit behind two PCIe 2.0 x1 switches ([topology](#pcie-topology)) | ⚠️ |
| CPU | Intel Celeron mobile series | Intel Celeron 3865U (Kaby Lake), 2 cores / 2 threads, 1.8 GHz, 2 MB L3, Intel HD 610 iGPU | ✅ |
| RAM | 1 × 4 GB DDR4-2133 SO-DIMM | 1 × 4 GB DDR4 SO-DIMM (rated 2400, running at 2133 MT/s), single channel; second slot empty, firmware reports 32 GB max | ✅ |
| Storage | 1 × 64 GB M.2 SATA III SSD | 64 GB M.2 SATA SSD (ATA form factor field = M.2), 6 Gb/s link, ~485 MB/s sequential read | ✅ |
| Network | 1 × Gigabit LAN | Realtek RTL8111/8168 (r8169 driver), 1 Gb/s full duplex | ✅ |
| Display | 1 × HDMI 1.4 | One HDMI connector, driven by the Intel iGPU (the P106 cards have no outputs) | ✅ |
| USB | 2 × USB 3.0, 2 × USB 2.0 | Intel USB 3.0 xHCI controller present; physical port count not checked | ❓ |
| GPU power input | 7 × 6-pin 12 V | — | ❓ |
| PSU | 1600 W, sold separately | — | ❓ |
| Cooling | 10 × 9 cm 4800 rpm fans (5 intake, 5 exhaust) | No fan sensors are exposed to the OS; all nine GPUs report 0 % fan, consistent with passive cards | ❓ |
| Firmware | — | AMI BIOS A355M003 (2017-07-28); SMBIOS product/vendor fields left as unfilled placeholders | ✅ |
| OS | Windows 10 64-bit (optional) | Reinstalled with Ubuntu 24.04.5 LTS | — |

## GPUs: NVIDIA P106-090 (Manli 6 GB variant)

The P106-090 is a cut-down GP106 (the GTX 1060 die) sold for mining only: no display outputs, a
narrow PCIe interface, and a mining-oriented driver ID. NVIDIA's reference P106-090 has 3 GB; Manli's
board (TechPowerUp lists it as "P106L Mining Card (F347G) 6 GB") doubles that to 6 GB.

Per card, as reported by the driver and CUDA on this unit:

| Property | Value |
|---|---|
| Die / part | GP106-090-A1 (Pascal, 16 nm) |
| CUDA compute capability | 6.1 |
| SMs / CUDA cores | **5 / 640** (see note below) |
| VRAM | 6 GB GDDR5 (6144 MiB; ~6068 MiB visible to CUDA), 192-bit bus |
| Memory bandwidth | 192 GB/s theoretical, **~173 GB/s measured** (device-to-device copy) |
| L2 cache | 1.5 MB |
| Clocks | 1531 MHz rated boost (CUDA), up to 1911 MHz max (nvidia-smi); memory 8 Gbps effective |
| FP32 | ~2.0 TFLOPS theoretical at 1531 MHz; ~1.55–1.75 TFLOPS in a simple FMA microbenchmark |
| FP16 / tensor cores | FP16 runs at 1/64 of FP32 rate; no tensor cores; INT8 `dp4a` is supported (cc 6.1) |
| Power limit | 75 W default and max, 60 W min; ~5 W at idle |
| Thermal limits | target 83 °C, slowdown 98 °C, shutdown 101 °C |
| ECC | none |
| Display outputs | none (PCI class "3D controller", not VGA) |
| Host interface | PCIe 1.0 x4 capable (the card itself is capped at Gen1), running x1 in this chassis |

**Core count:** TechPowerUp and similar databases list the P106-090 with 6 SMs / 768 CUDA cores. On
all nine cards in this unit the driver reports 5 SMs, and the hardware `%nsmid` register (read from
a CUDA kernel) also returns 5, so these cards have **640 CUDA cores**.

**Whole system:** 9 × 6 GB = 54 GB of VRAM in nine separate pools, 9 × ~173 GB/s of local memory
bandwidth, and roughly 14–18 TFLOPS FP32 in aggregate.

**Driver support:** NVIDIA's R580 driver branch is the last one that supports Pascal; it becomes a
legacy branch that only gets occasional fixes (for example, for newer Linux kernels). CUDA 12.x is the
last toolkit line that supports Pascal, so build for `sm_61` with CUDA 12.9 even though the 580 driver
reports CUDA 13.0.

## PCIe topology

The spec sheet's "9 × PCI-E x16" describes the physical slot size only. Electrically, every GPU has a
single lane. Everything hangs off four x1 root ports on the chipset (the Sunrise Point-LP PCH, on the
same package as the CPU), which on this Celeron are limited to PCIe 2.0 speed. Two ASMedia ASM1184e
packet switches (4-port, PCIe 2.0 x1 upstream) fan two of those ports out to eight of the GPUs.

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

GPU numbers follow CUDA / nvidia-smi order (PCI bus order). The GPUs negotiate Gen1 because the cards
are capped at PCIe 1.0, even though the switch ports support Gen2.

Measured host↔GPU bandwidth (pinned memory, 64 MB transfers):

| Scenario | Per GPU | Total |
|---|---|---|
| One GPU at a time, host → GPU | ~189 MB/s (GPU4: ~200 MB/s) | — |
| One GPU at a time, GPU → host | ~213 MB/s | — |
| All nine at once, host → GPU | ~90–120 MB/s behind a switch, ~200 MB/s for GPU4 | ~715 MB/s |

For reference, PCIe 1.0 x1 is 250 MB/s per direction before protocol overhead and PCIe 2.0 x1 is
500 MB/s, so each switch uplink is shared by four GPUs.

**Peer-to-peer:** CUDA reports no P2P access between any pair of GPUs. `nvidia-smi topo -m` shows
`PIX` (same switch) inside each group of four and `PHB` (through the host bridge) everywhere else.

### What this means for non-mining workloads

Mining barely uses the bus: the host sends a small job and gets hashes back, so x1 links cost
nothing. Other workloads feel it:

- **Loading data onto the GPUs is slow.** Filling one card's 6 GB takes over 30 s on its own; filling
  all nine (~54 GB) takes about 80 s even with transfers running in parallel. The SSD (~485 MB/s),
  the 64 GB disk size and the 4 GB of system RAM are further limits on staging large files.
- **Work that exchanges lots of data between GPUs scales badly.** With no P2P and ~100–200 MB/s per
  GPU, all GPU-to-GPU traffic goes through host memory over these links. Splitting a job so that each
  GPU works on its own slice and only passes small results along suits this machine; schemes that
  synchronise large tensors on every step do not.
- **The host is weak.** Two Celeron cores and 4 GB of single-channel RAM will bottleneck anything that
  does real work on the CPU. The RAM can be upgraded (second SO-DIMM slot free, 32 GB max per firmware).

## Power and cooling

Not observable from software, so these are Manli's figures:

- **PSU:** 1600 W, sold separately. For scale, nine GPUs at their 75 W limit draw at most 675 W.
- **GPU power input:** 7 × 6-pin 12 V connectors.
- **Fans:** 10 × 9 cm, 4800 rpm, five intake and five exhaust. They are not connected to anything the
  OS can read or control.

The GPUs report 0 % fan speed, which is consistent with passive heatsinks cooled by the chassis
fans. At idle, GPU temperatures were 23–30 °C.

## Mining history (for context)

Manli rated the box at about 200 MH/s on Ethereum / Ethereum Classic at 800 W and 3800 H/s on
Monero at 600 W. Neither still applies:

- **Ethereum** stopped proof-of-work mining entirely at the Merge (September 2022). Its DAG was still
  under 6 GB at that point, so these cards were not pushed out by DAG growth; the network simply
  stopped using GPU miners.
- **Monero** has used RandomX, which is designed for CPUs, since November 2019, so GPU mining it is no
  longer competitive.

Manli's own product page also lists AI computing, rendering and data processing as intended uses.

## How these were checked

Everything was read on the running system without opening the case:

- PCIe tree and link widths/speeds: `lspci -tv`, `lspci -vvv` (`LnkCap` / `LnkSta`)
- GPU properties, power, thermals, topology: `nvidia-smi -q`, `nvidia-smi topo -m`
- CPU, RAM, firmware: `lscpu`, `dmidecode -t processor -t memory -t bios`
- SSD: `lsblk`, `hdparm -I` (form factor), `hdparm -t` (read speed)
- SM count, P2P, memory and transfer bandwidth: small CUDA programs built with CUDA 12.9 for
  `sm_61` (`cudaGetDeviceProperties`, `%nsmid`, `cudaDeviceCanAccessPeer`, pinned `cudaMemcpy`
  timing)

Still to be confirmed by looking inside the box: PSU model and wattage, number of fans, GPU power
cabling, physical slot form factor, whether the GPU heatsinks are fanless, and the number of USB
ports.

## Sources

- Manli product page, M-P106L9-N6G specifications (archived copy; the live page now returns 404):
  <https://web.archive.org/web/20231001152427/https://manli.com/en/product-detail-MANLI_GPU_Mining_System_P106-090_(6GB)_X9-206.html>
- TechPowerUp GPU Database, Manli P106L Mining Card (F347G) 6 GB:
  <https://www.techpowerup.com/gpu-specs/manli-p106l-mining-card-f347g-6-gb.b6880>
- Phoronix, "NVIDIA Confirms 580 Linux Driver Is The Last For Maxwell / Pascal / Volta":
  <https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW>
