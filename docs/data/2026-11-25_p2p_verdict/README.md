# The driver's peer-to-peer verdict on the rig (2026-11-25, Q991)

Why the nine cards get no peer-to-peer access, read straight from the rig. This covers what the driver reports for each
card pair, whether the installed driver knows NVIDIA's override keys, and whether an IOMMU and ACS redirect stand
in the path. Read-only: nothing was loaded, set or written. The result page is
[diy_card_interconnect.md](../../research/notes/diy_card_interconnect.md), sections 2 and 4, and registry row Q991.

## Results (rig 1, driver 580.178.04, kernel 7.0, Ubuntu 24.04)

| Check | Result |
|---|---|
| `nvidia-smi topo -m` | `PIX` inside each switch (cards 0–3, cards 5–8), `PHB` for every other pair |
| `nvidia-smi topo -p2p r` | `CNS` (chipset not supported) for all 72 ordered pairs, the `PIX` pairs included |
| `nvidia-smi topo -p2p w` | `CNS` for all 72 ordered pairs |
| `GNS` (GPU not supported) anywhere | no: the cards themselves aren't blocked |
| Key names in the installed `nvidia.ko` | `ForceP2P` 3, `RMForceP2PType` 1, `RMPcieP2PType` 1, `PeerMappingOverride` 1, `DmaRemapPeerMmio` 6 occurrences |
| `nvidia` module options in `/etc/modprobe.d` | none about peer access |
| IOMMU | on: `iommu: Default domain type: Translated`, 22 groups, all `DMA-FQ` (the kernel builds `CONFIG_INTEL_IOMMU_DEFAULT_ON=y`, `CONFIG_IOMMU_DEFAULT_DMA_LAZY=y`) |
| ACS on the four PCH root ports | redirect on: `Intel SPT PCH root port ACS workaround enabled` for `00:1c.0`, `1c.3`, `1c.4` and `1c.5` in the boot log |
| ACS on the ASM1184e ports | no ACS capability (2026-11-04, [pcie_facts](../2026-11-04_diy_interconnect/README.md)) |
| IOMMU groups | nine groups for nine cards; each card behind a switch shares its group only with its own switch port |

What it means:
- **The refusal is the chipset rule, applied to every pair.** NVIDIA's chipset table has no Sunrise Point-LP entry.
  Its list of accepted switches (`IS_SUPPORTED_PCIE_SWITCH`) has no ASMedia, so the pairs behind one switch fall under
  the same rule.
- **The override is there.** The installed driver carries `ForceP2P`, NVIDIA's key that skips the chipset, topology and
  GPU checks.
- **The IOMMU changes how a forced test must run.** The root ports' ACS control registers sit at non-standard offsets
  on this PCH. `lspci` reads the standard ones and shows them off, while the kernel's workaround has turned redirect
  on. With the IOMMU translating, the driver maps a peer card's memory window through the IOMMU (`nv_dma_map_peer`
  calls `dma_map_resource`). A card would then write to an IOMMU address that no switch port claims, so even a pair
  behind one switch would go up to the PCH. Two settings keep such a pair inside its switch:
  - `DmaRemapPeerMmio=0` beside `ForceP2P=0x11` in `NVreg_RegistryDwords`: plain bus addresses and no IOMMU mapping.
    Pairs across the PCH would then fault, so it suits the six in-switch boundaries only.
  - `iommu=pt` on the kernel line: passthrough, which needs a reboot.

## How it was run

`ssh "$RIG" 'bash -s' < run/p2p_verdict.sh > p2p_verdict.txt`, on rig 1 at 13:42 on 2026-11-25, during a pause of the
day's first row. The commands ran as two `ssh` calls with the same commands as [run/p2p_verdict.sh](run/p2p_verdict.sh);
`DmaRemapPeerMmio` was counted in a third call. The raw output is not published (it holds firmware table addresses);
the table above has every result it showed.

| File | What |
|---|---|
| [run/p2p_verdict.sh](run/p2p_verdict.sh) | the read-only checks: `nvidia-smi topo`, the key names in the module, IOMMU, ACS and groups |

Provenance: NVIDIA's driver output and the Linux boot log on the rig; the source references are NVIDIA's open kernel
modules, release 580.196.11:
[p2p_caps.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/src/nvidia/src/kernel/platform/p2p/p2p_caps.c),
[pcie_switch_ref.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/src/common/inc/swref/published/pcie_switch/pcie_switch_ref.h),
[nv-dma.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/kernel-open/nvidia/nv-dma.c) and
[nv-reg.h](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.196.11/kernel-open/nvidia/nv-reg.h).
