# What a P106-090 holds back, and whether anything public unlocks it safely (2026-11-17)

Scope: the P106-090 is a cut-down GP106 (the GTX 1060 die) sold for mining. Against the full die it runs half the
shader multiprocessors, trains its PCIe link at the slowest rate, and has no display outputs. This note asks, for each
of those limits, whether any public tool or signed firmware image can lift it, and whether doing so would be safe on a
[Manli M-P106L9-N6G rig](../../mining_rig.md) (nine P106-090 6 GB cards, a 2-core Celeron, 4 GB of host RAM, driver 580,
CUDA 12.9). It is desk work: published reports were read, and the candidate tools were cloned and read at their source.
**Nothing was run from any clone, no binary was opened, and nothing was flashed or sent to a machine.** Any VBIOS flash
is the owner's decision alone, on one card.

Conventions follow the other notes:
- A **report** is someone else's result, not measured here; it is labelled as such.
- A **claim to test** is something this note infers or that no primary source settles; it is labelled as such and, where
  a rig run would decide it, pointed at its registry row.
- Source labels: **P** primary (vendor docs, datasheets, a tool's own source code), **S** secondary (forum and
  community reports). Each source is dated in the list at the end.

Known going in (so this note does not re-derive it): no PCIe-generation unlock exists for these cards and Pascal VBIOS
images are signature-checked ([hardware tuning](hardware_tuning.md) §4); the public unlock tools for mining cards target
Turing and Ampere CMP cards, not Pascal ([Pascal forks](pascal_forks_and_quants.md) §1, §6).

## The limits, and the verdict on each

| Item | What limits it | What could change it | Gone at reboot, or written to the card? | Risk on this rig | Verdict |
|---|---|---|---|---|---|
| **5 of the die's 10 SMs disabled** (640 of 1,280 cores) | On-die floorsweeping fuses set at manufacture; FECS/the firmware reads the enabled-SM mask at init | Nothing public. The CMP "SM unlocks" rewrite FECS issue-rate state through a **patched NVIDIA open kernel module** plus a GSP firmware exploit — neither exists on Pascal | n/a (physical fuse) | n/a — no method reaches it | **Nothing to unlock; do not pursue.** No Pascal SM-enable tool exists, and the mechanism the CMP tools use is Turing+ only |
| **PCIe link generation: Gen1 (2.5 GT/s)** | The card advertises 2.5 GT/s as its maximum (`LnkCap`); a link trains to the fastest rate both ends advertise | Nothing public for P106/P104. The CMP Gen2 unlocks use the GSP policy path + open modules (Turing+) | n/a | n/a | **Not available** (restates [hardware tuning](hardware_tuning.md) §4). Even if it existed: est. ~1 % decode on nine cards |
| **PCIe link width: this rig is one physical lane per card** | The Manli board wires every GPU slot as a single lane through ASM1184e switches; the card's `LnkCap` on this rig advertises width x4 | One tool (`P106-90-Unlock-X16`) runs a closed boot-time EFI binary that writes the width registers x4→x16, volatile per boot — but "EFI cannot create physical lanes that do not exist" | Volatile (re-run each boot; nothing written to the card) | Closed prebuilt binary; needs Secure Boot off and edits the ESP + boot order | **Do not use, and moot here.** The binary is unauditable, and there is one physical lane per card on this board regardless |
| **Cross-flash to a P106-100 / GTX 1060 image** (for the other 5 SMs or more VRAM) | VBIOS signature check (Falcon, since Maxwell) + nvflash GPU/Board-ID match | `nvflashk` forces nvflash's ID-mismatch bypass, but still needs a **signed** image and cannot disable the on-die signature check; the fused-off SMs stay off whatever image boots | Written to the card (persistent); reversible only by reflashing the backup | Bricking one of nine cards on a headless switch-fanout board; LUKS reboots are at the console | **The owner's decision, and pointless.** No compute gain (SMs are fused, the 6 GB is already present); real brick risk |
| **No 3D / display path in the driver** | The VBIOS reports no display outputs; the driver's device-ID list omits the mining SKU for graphics | `NVIDIA-patcher` adds the device ID to the driver and re-signs it (Windows) or byte-patches `nv-kernel.o` (Linux) | Driver files on the host, not the card; undone by reinstalling the driver | Windows path impersonates an "NVIDIA Corporation" signing cert and needs test-signing | **Not needed.** This rig already has full CUDA compute on Linux; the Windows path is a do-not-use |
| **A possible per-instruction throttle** (dp4a, FFMA, …) | Unknown — never measured on a P106-090 | n/a (this is a measurement, not an unlock) | n/a | one card, 1 h | **Measure it:** registry Q103 is still worth its one-hour calibration (below) |

### In short

On the compute side there is nothing to unlock on a Pascal P106 and no public tool that could. Every "mining-card
unlock" that actually raises throughput — the CMP 40HX/50HX/90HX projects — works by loading a **patched NVIDIA open
kernel module** and exploiting the **GSP** firmware to rewrite the card's protected SM issue-rate state. Pascal has no
GSP and is not supported by the open kernel modules at all (NVIDIA documents the open modules as Turing-and-newer),
so that entire class of tool cannot run against a P106-090 under driver 580. Cross-flashing is blocked by the VBIOS
signature and, even if bypassed, cannot re-enable fused-off SMs. The one tool that touches a P106 PCIe setting unlocks
link *width*, which is physically fixed at one lane per card on this board, and ships only a closed boot-time binary.
The single open question that could still find something is empirical, not a flash: whether the card throttles any
instruction class (Q103).

## 1. The five disabled SMs (question 1)

The full GP106 die has 10 SMs / 1,280 CUDA cores (2 GPCs × 5 SMs × 128 cores); the GTX 1060 6 GB and the P106-100 enable
all of them (**P**, GP106 die shot and the GTX 1060 launch specs). This rig's cards report **5 SMs / 640 cores** to CUDA
([hardware spec](../../mining_rig.md), measured). So half the die's shader multiprocessors are off.

**What turns them off is on-die floorsweeping, not a setting.** Across the GP10x line NVIDIA bins each die and permanently
disables defective or surplus SMs with fuses blown at manufacture; the firmware and FECS (the front-end context switcher)
read the enabled-SM mask at init. This is the industry-standard mechanism, and the CMP unlock work is the clearest
indirect evidence that it is a hardware gate, not a VBIOS field:

- The CMP "SM unlock" does **not** edit a VBIOS value. It forces the firmware's FECS SM-selector initialization state to a
  full-speed pattern (`SS0 = 0x88888888`, `SS1 = 0x8`, FECS PLM `0xFFFFFF8F`) through a patched open kernel module and a
  GSP Booter exploit (**P**, `xrip/cmp50hx-unlock` patch 01 and `Cyridd/cmpunlocker` README). That these cards need a
  firmware-state rewrite to reach full issue rate shows the throttle lives below the VBIOS.
- Where a count really is fused, even the CMP tools cannot move it. `xrip/cmp50hx-unlock` patch 02 sets the reported
  RT-core count to 56, and its own README is explicit: the verifier then prints 56, but "RT tests still fail at SM decode
  because of the physical RT fuse" — a report/API change only (**P**). The same holds for a shader fuse.
- The older AMD precedent matches: on an HD 6950, a shader-unlock flash reported success yet GPU-Z still showed the
  original count, the shaders being hard-locked on that card (**S**, 2011). A fuse-locked part accepts the flash and
  ignores it.

No tool, report, or forum thread claims a working shader unlock on any P106-090 (searched; none found). 

**Claim to test (minor):** this rig's 5-SM / 640-core figure is the measured value the project builds on; public GPU
databases list a *reference* 3 GB P106-090 as 6 SMs / 768 cores. The difference is a per-SKU floorsweep, and it underlines
that the SM count is a fixed hardware configuration, not a published knob. A one-line read-only `nvidia-smi`/CUDA query
confirms the rig's count whenever convenient; the kernels are already tuned for 5 SMs, so nothing rests on re-confirming it.

**Whether it is a fuse or a VBIOS strap, exactly, is not proven for the P106-090 in any primary source read here.** The
evidence above (firmware-state rewrite on CMP, fuse survives a flash on comparable parts) points to fuses/floorsweeping;
a VBIOS strap is not ruled out by a document, but no VBIOS edit that changes an SM count on any GP10x card was found.

## 2. Cross-flashing a P106-090 (question 2)

**Has anyone flashed a P106-090 with a signed image of another GP106 card, and what happened?** No clean, documented
success was found. What the record shows:

- **The signature wall.** Since Maxwell an on-die Falcon checks the VBIOS signature; edits to signed regions flash but the
  driver then rejects the card ([hardware tuning](hardware_tuning.md) §4, **S**). A full *other-card* image (a P106-100 or
  GTX 1060 BIOS) is itself NVIDIA-signed, so the signature check is not what stops it — the **ID match** is.
- **The ID wall, and the bypass.** Factory nvflash refuses an image whose GPU ID, Board ID and subsystem ID do not match
  the card. `notfromstatefarm/nvflashk` force-enables nvflash's built-in mismatch bypass so a mismatched *signed* image
  will flash; its README is candid that "whether or not it will actually work is a different story," and that a true
  cross-generation or cross-class image is often simply rejected by the GPU ("Nothing changed! … Invalid firmware image
  detected") (**P**, nvflashk README).
- **It buys nothing even if it takes.** The other 5 SMs are fused off (§1), so a 10-SM image cannot light them. The 6 GB is
  already the physical memory on the Manli board, so there is no VRAM to "unlock" (unlike the P104-100 4→8 GB cards, whose
  boards carry 8 GB behind a 4 GB BIOS — not this card). The P106-090 has no display connectors wired on the board, so a
  gaming BIOS adds no output either.
- **What the one P106-090 flash thread shows.** A folding-at-home user flashed a Zotac P106-090 with a community BIOS and
  then hit driver crashes and `nvidia-smi: GPU is lost` on one host; another user on the same thread said the same BIOS ran
  fine (**S**, 2020). The lesson is variance and risk, not a working upgrade.

**Recovery on a host whose console is on integrated graphics.** This rig's display is HDMI from the Intel HD Graphics 610
iGPU, which is the *good* case for recovery: a bad GPU flash leaves the iGPU driving the screen, so the card can be
reflashed from its backup without a second machine (nvflashk lists "an integrated GPU and output on the motherboard" as
the easiest recovery, **P**). The hazards specific to this rig are different: nine cards sit behind ASM1184e switches, a
card that drops off the bus can take its switch siblings with it until a power cycle, and every reboot needs the LUKS
passphrase typed at the console. Net: recoverable in principle, but it is a console-side, one-card-at-a-time operation
with no upside. **Verdict: the owner's decision, and not worth making.**

## 3. Signed VBIOS versions for this card (question 3)

The rig's cards report VBIOS **86.06.5B.00.0C** ([hardware spec](../../mining_rig.md)). Public VBIOS databases (TechPowerUp)
hold entries for GP106 gaming and mining boards — P106-100 6 GB, the GTX 1060 6 GB family (`86.06.xx`), and a 3 GB
P106-090 — but **no entry for the Manli P106-090 6 GB `86.06.5B.00.0C` image was found** (searched TechPowerUp's VGA BIOS
collection by model and by device ID; the Manli mining SKU was sold only inside the pre-built rig and is not individually
listed, confirmed in a TechPowerUp forum thread, **S**). What that means in practice:

- There is no alternative *signed* P106-090 6 GB image published to flash, so there is no "newer version" to move to and
  nothing to diff against for this exact card. Any image on the public databases is for a different board (different
  board ID), i.e. a cross-flash (§2), not an update.
- The differences that *do* exist between GP106 images in those databases are clock/fan/power tables and the memory-vendor
  support list (Hynix/Micron/Samsung together, so the VBIOS version does not even identify the memory vendor — already
  noted in [hardware tuning](hardware_tuning.md) §1.1). None changes the SM count or the link generation.

**Verdict: there is one signed image for this card — the one on it. There is no published P106-090 6 GB alternative, and
none would change what §1 and the link limits fix.**

## 4. The unlock tools, read from source (question 4)

Every candidate was cloned read-only and read. Grouped by what they actually are:

| Tool | Target card(s) | Mechanism | Could it act on a P106 under driver 580? |
|---|---|---|---|
| `xrip/cmp50hx-unlock` | CMP 50HX (TU102, Turing) | Patched NVIDIA **open** kernel module `610.43.03` + a GSP Booter exploit (UEFI or kernel) that rewrites FECS SM issue-rate state and PCIe policy | **No.** Pascal has no GSP and no open-module support |
| `Cyridd/cmpunlocker` | CMP 40HX (TU106, Turing) | Patched open modules (`580.173.02`–`620.70.06`) or a pre-driver CPU-RM module with GSP OFF; same FECS/GSP state rewrite; optional Gen2 + ReBAR + a userspace `libnvidia-glcore` patch | **No.** Same reason; the "580" here is the *open-module* 580, which still requires GSP/Turing |
| `Wh1stle05/cmp90hx` | CMP 90HX (GA102, Ampere) | Open-module / firmware unlock (FP32 0.72→18.24 TFLOPS reported) | **No** |
| `Rhonstin/beellama-cmp90hx` | CMP 90HX (Ampere) | A llama.cpp-family fork that *measures* the CMP throttle (dp4a 29×, FFMA 14×) and substitutes instructions; not a card unlock | **No** (and it is an engine, not a flash) |
| `WifiLast/torch-cmp-patch` | CMP (Turing/Ampere) | A PyTorch/CUDA kernel patch working around CMP throttles | **No** |
| `notfromstatefarm/nvflashk` | Any NVIDIA GPU | Patched **closed** nvflash: forces the ID-mismatch bypass to flash a mismatched *signed* BIOS | Mechanically yes (it is architecture-agnostic), but see §2: no useful image exists and it cannot re-enable fused SMs |
| `DaZuiZZZZ/P106-90-Unlock-X16` | **P106-090** (`10DE:1C09`) | A closed prebuilt EFI (`NVPermissiveEFI.efi`) run from UEFI Shell at each boot, writing PCIe link-**width** registers x4→x16; volatile | Targets a P106 by name, but unlocks width only — physically fixed at one lane per card on this board |
| `dartraiden/NVIDIA-patcher` | P106/P104/P102 + CMP + others | Adds the mining device ID to the NVIDIA **display** driver and re-signs it (Windows) or byte-patches `nv-kernel.o` (Linux) to expose 3D/NVENC | Yes, but it adds a *graphics* path; it changes no SM, link or firmware, and this rig already has full CUDA compute |
| `DualCoder/vgpu_unlock` | Maxwell–Turing (Pascal listed) | Intercepts the GRID driver's ioctls to spoof the device ID so a consumer card reports vGPU capability (VM partitioning) | Yes for vGPU, but it needs the GRID/vGPU driver and enables virtualization, not compute speed — not applicable here |

The two load-bearing facts:

1. **The compute unlocks are GSP + open-kernel-module work, and both are Turing-and-newer.** NVIDIA's open GPU kernel
   modules "cannot support GPUs before Turing because they depend on the GPU System Processor (GSP)" introduced in Turing;
   on a pre-Turing GPU the installer falls back to the proprietary modules (**P**, NVIDIA open-kernel-module README and
   datacenter driver docs). Every CMP compute unlock reads above builds patched *open* modules and exploits GSP. A P106-090
   has neither. This is why there is no Pascal counterpart and cannot be one with these methods.
2. **The one P106-named tool writes nothing to the card and fixes nothing physical.** `P106-90-Unlock-X16` runs
   `NVPermissiveEFI.efi pcie-gen2-x16` from a UEFI Shell `startup.nsh` on every boot, then chainloads Windows; the width it
   sets is volatile and its own README states "EFI cannot create physical lanes that do not exist." This board wires one
   lane per GPU through the switches, so there is no width to recover.

## 5. Per-tool: publisher, licence, binaries, what it writes, safety (question 5)

| Tool | Publisher | Licence | Ships prebuilt binaries? | Writes to the system / card | Safety verdict for this rig |
|---|---|---|---|---|---|
| `xrip/cmp50hx-unlock` | GitHub user `xrip` | MIT (repo) + copyleft-next (one file); redistributes NVIDIA firmware blobs | **Yes** — release `.run`/`.zip`, and in-repo Windows BYOVD drivers `ThrottleStop.sys`, `WinRing0x64.sys` | Builds/loads a patched kernel module; UEFI boot entry; writes GPU PCIe/FECS registers | **Do not use.** Wrong architecture; and it installs a modified kernel module plus known-vulnerable signed drivers |
| `Cyridd/cmpunlocker` | GitHub user `Cyridd` (fork of `PZH1gdmu/CMP40HX-Unlock`) | MIT repo, but redistributes a GPL-3 driver, a proprietary signed driver and NVIDIA firmware, so "not distributable under MIT alone" | **Yes** — Windows ZIP + unsigned EFI app | Patched modules; UEFI app; register writes; optional userspace library patch | **Do not use.** Wrong architecture; unsigned EFI, mixed redistribution |
| `Wh1stle05/cmp90hx` | GitHub user `Wh1stle05` | GPL-2.0 | (not inspected in depth; Ampere target) | Patched modules / firmware state | **Do not use** (wrong architecture) |
| `Rhonstin/beellama-cmp90hx` | GitHub user `Rhonstin` | not stated in a top-level file | It is an engine fork, not a flasher | n/a (software engine) | **Not an unlock.** Useful only as evidence that CMP cards throttle per instruction |
| `WifiLast/torch-cmp-patch` | GitHub user `WifiLast` | not stated | — | Userspace CUDA/PyTorch patch | **Not applicable** (CMP, software) |
| `notfromstatefarm/nvflashk` | GitHub user `notfromstatefarm` / "kefinator" | Closed — "base … copyright nVIDIA … provided in a modified fashion"; binary-only | **Yes** — binary only (no source) | **Writes the card's VBIOS** by bypassing ID checks | **The owner's decision only, and discouraged.** A closed binary that flashes firmware, for zero gain here (§2) |
| `DaZuiZZZZ/P106-90-Unlock-X16` | GitHub user `DaZuiZZZZ` | No top-level licence; only a TianoCore BSD on a template file. Core `NVPermissiveEFI.efi` is **not in the repo** (release-only, closed) | **Yes** — release EFI + UEFI Shell, closed | Copies to the ESP, adds a firmware boot entry, writes GPU PCIe width registers at each boot (volatile) | **Do not use.** Unauditable boot-time binary; needs Secure Boot off; and width is physically fixed here |
| `dartraiden/NVIDIA-patcher` | GitHub user `dartraiden` | No top-level licence file found | **Yes** — bundles Windows tooling (`signtool.exe`, `7z`, `PEChecksum`) and a cert archive | Windows: generates a self-signed cert naming "NVIDIA Corporation", re-signs the driver, needs test-signing. Linux: byte-patches `nv-kernel.o` and reinstalls | **Not needed; do not use.** Adds a graphics path the rig does not use; the Windows path impersonates an NVIDIA signing identity |
| `DualCoder/vgpu_unlock` | Jonathan Johansson | MIT | No (Python + C source) | Hooks the GRID driver's ioctls and makes `nv-kernel.o`'s rodata writable to spoof vGPU capability | **Not applicable.** Enables vGPU virtualization, needs the GRID driver; no bearing on single-host inference speed |

General safety notes that apply to all of the above: every one needs Secure Boot off and either a patched kernel module,
a re-signed or impersonated driver, a closed boot-time binary, or a firmware flash. None is "fine to try" on a production
rig. The only ones that even target a P106 (`nvflashk` by way of a cross-flash, `P106-90-Unlock-X16`, `NVIDIA-patcher`)
deliver nothing this rig can use, for the physical reasons in §1–§4.

## What this leaves for the plan

- **Nothing to unlock on the compute path.** Close the "is there a hardware unlock" question for the P106-090: SMs are
  fused, the link generation and width are fixed, the VBIOS is signed with no useful alternative image, and every public
  tool either targets Turing/Ampere through GSP + open modules or writes something with no gain here.
- **Q103 is still worth running.** The one thing not yet *measured* is whether the card throttles any instruction class
  (dp4a, FFMA, integer, shared-memory, conversions). The CMP precedent proves NVIDIA does throttle mining cards per
  instruction, but no such measurement exists for any Pascal P106, and single-token decode exercises dp4a at only ~3 % of
  its rate, so a moderate throttle would not show in normal runs ([Pascal forks](pascal_forks_and_quants.md) §6). A
  one-hour, one-card calibration (one kernel per class at full occupancy) is the direct test and the empirical floor under
  this whole question. Keep **Q103** queued; if it finds a throttle, *that* would be the first real "held-back" finding —
  though with no open-module path on Pascal there would still be no tool to lift it, only an engine-side workaround like
  the CMP instruction substitution.
- **No new unlock idea emerged** that is not already tracked. The read-only `LnkCap`/`LnkCap2` width-and-speed check is
  already [hardware tuning](hardware_tuning.md) §4's P1; the SM-count re-confirmation is a trivial read the kernels already
  assume. The search closed doors rather than opening them.

## Sources

NVIDIA and vendor (P)
- NVIDIA Open GPU Kernel Modules README (open modules require Turing+ via GSP; pre-Turing falls back to proprietary),
  <https://download.nvidia.com/XFree86/Linux-x86_64/560.35.03/README/kernel_open.html>; datacenter driver kernel-modules
  guide (open modules "supported only on Turing and newer"), <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/610/kernel-modules.html>
- GP106 die configuration (2 GPC × 5 SM × 128 cores = 1,280; GTX 1060 6 GB / P106-100 full), TechPowerUp GPU database
  GP106 and P106-100 entries; GTX 1060 launch specs (read via search excerpts, the pages 403-block direct fetches)
- Phoronix, "NVIDIA Confirms 580 Linux Driver Is The Last For Maxwell / Pascal / Volta", <https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW>

Tools, read at source after a read-only clone (P)
- [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock) (README, `patches/cmp50hx/01–04`, `efi-unlock/README.md`;
  FECS SS0/SS1 state, RT-count report-only, BYOVD drivers)
- [Cyridd/cmpunlocker](https://github.com/Cyridd/cmpunlocker) (README; TU106, open-module versions `580.173.02`–`620.70.06`, CPU-RM GSP-OFF)
- [notfromstatefarm/nvflashk](https://github.com/notfromstatefarm/nvflashk) (README; ID-mismatch bypass, "Nothing changed!/Invalid firmware image", recovery via iGPU)
- [DaZuiZZZZ/P106-90-Unlock-X16](https://github.com/DaZuiZZZZ/P106-90-Unlock-X16) (README + `startup.nsh`; `NVPermissiveEFI.efi pcie-gen2-x16`, width-only, release-only closed binary)
- [dartraiden/NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher) (README + `Presets.txt` + `docs/README-PATCHER-LINUX.md`; device-ID add, re-sign/byte-patch, "NVIDIA Corporation" generated cert)
- [DualCoder/vgpu_unlock](https://github.com/DualCoder/vgpu_unlock) (README; ioctl spoof of vGPU capability, MIT)
- [Wh1stle05/cmp90hx](https://github.com/Wh1stle05/cmp90hx), [Rhonstin/beellama-cmp90hx](https://github.com/Rhonstin/beellama-cmp90hx), [WifiLast/torch-cmp-patch](https://github.com/WifiLast/torch-cmp-patch)

Reports (S, dated)
- Folding@Home forum, "P106-090 Troubles" (a Zotac P106-090 flashed with a community BIOS; driver crashes and "GPU is lost"
  on one host, same BIOS fine on another), 2020-11-13/15, <https://forum.foldingathome.org/viewtopic.php?p=345802>
- TechPowerUp forum, P106-90 6 GB from a Manli mining rig; the SKU is not individually listed and a modified nvflash may be
  needed for a board-ID mismatch, <https://www.techpowerup.com/forums/goto/post?id=5355163> (2026)
- Tom's Guide forum, HD 6950 shader-unlock flash "succeeded" but GPU-Z still showed the original count (fuse-locked),
  <https://forums.tomsguide.com/threads/problem-flashing-bios-on-hd-6950-to-unlock-shaders.250415/post-1101232> (2011)
- VideoCardz, "Inno3D launches P106-090 Mining Cards with 768 CUDA cores" (reference variant core count),
  <https://videocardz.com/73139/inno3d-launches-p106-090-mining-cards-with-768-cuda-cores>
- Linus Tech Tips, "P106 NOW SUPPORT DirectX (Not Official)" (nvflash cannot flash a BIOS built for a different GPU; signed
  BIOS images resist tampering), <https://linustechtips.com/topic/1001580-p106-now-support-directx-not-official/>

This project
- [hardware tuning](hardware_tuning.md) §1.1, §4 (PCIe cap, signature check, memory vendor not in the VBIOS);
  [Pascal forks](pascal_forks_and_quants.md) §1, §6 (CMP unlocks are Turing/Ampere; Q103's role);
  [hardware spec](../../mining_rig.md); registry [Q103](../../../quants/experiments.md), [Q1531](../../../quants/experiments.md)
