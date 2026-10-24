# Free hardware settings on the P106 rig: memory and core clocks, power, the PCIe link and host settings, and how to prove each one safe (2026-10-24)

Scope: settings the owner of a Manli M-P106L9-N6G (nine P106-090 cards, [hardware spec](../../mining_rig.md)) can change
without buying anything, which could make Flint decode or prompt processing faster on this project's llama.cpp fork: a
memory overclock, power and core clocks, the PCIe link generation, and host settings of the same kind (ASPM, CPU
frequency and idle states, interrupts, persistence mode, huge pages, the desktop, BAR sizes). For each: what it would buy
on this rig's measured numbers, how to apply it on driver 580 with Xorg running, and a test protocol that proves it safe
on cards without ECC, where a reboot needs the disk passphrase typed at the rig's keyboard. Conventions are those of the
other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (vendor documentation, code, datasheets, measurements in a repository) and
  secondary. Overclocking figures from miners are reports: each is labelled with its card, memory vendor and units.
- Memory offsets are given in **transfer-rate units**, the units of `nvidia-settings`' `GPUMemoryTransferRateOffset`
  and of the "Linux" column in mining profiles: the P106-090's 8,008 MT/s, which `nvidia-smi` reports as a 4,004 MHz
  memory clock. +800 is therefore +10 % (8,808 MT/s, `nvidia-smi` 4,404 MHz). Windows tools show half these numbers.
- "The fork" is branch `pascal-p106-v2`; token times are llama-bench tg128.

Starting point, measured ([registry](../../../quants/experiments.md), [calibration](../../benchmarks/2026-10-22-single-card-calibration.md),
read-only inventory E0.1, and the `nvidia-smi` logs recorded with the Q38 rig runs on 2026-10-23):
- **Decode:** UD-Q4_K_XL 37.3 tok/s on six cards (26.8 ms per token, 2.21 GB of weights per token), ~31 tok/s on nine;
  S4-L0 37.4 tok/s on six. At the ~150 GB/s the Pascal kernels stream, the weights take ~14.9 ms of a six-card token;
  the other ~12 ms are kernel boundaries and small kernels, page walks (Q99), hops and the host.
- **Bandwidth:** a plain streaming read on one card reaches 179 GB/s (93 % of the 192 GB/s the 8 Gbps GDDR5 provides);
  the fork's kernels reach 138–165 GB/s on large shapes in [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md),
  less inside a real token.
- **Clocks and power:** compute work runs in P0 at the full 4,004 MHz memory clock; the only supported memory clocks are
  4,004 and 405 MHz, application clocks are not supported, and the power limit is adjustable from 60 to 75 W. Maximum
  boost is 1,860–1,911 MHz depending on the card.
- **PCIe:** every card trains at 2.5 GT/s x1; each card's Link Capabilities register advertises 2.5 GT/s x4 at most,
  while the ASM1184e switch ports and the root ports advertise 5 GT/s. ASPM is disabled on every link.

## Verdict

- **A memory overclock is the one setting that raises decode speed, but by about half its clock gain, not all of it.**
  Only ~55 % of a fork token is weight streaming, and part of that is latency-bound. A typical GP106 memory margin of
  +800 (+10 %) should give est. **+3–5 % decode today** on six or nine cards, +2–7 % across the plausible range of margins
  and scaling, and est. +5–9 % once the persistent kernel (Q88) and page-walk fixes (Q99) shrink the fixed part of a
  token. Prompt processing is compute-bound and gains est. 0–3 %.
- **It can be applied at runtime through NVML, without touching Xorg.** `nvmlDeviceSetMemClkVfOffset` (documented for
  Maxwell and newer, root only) sets one card's memory offset from a ten-line Python script on libnvidia-ml; offset 0
  restores stock at once, and a reboot or driver reload always clears it. The Coolbits route needs an NVIDIA X screen on
  every card, which this rig's desktop, with its display on the Intel graphics, most likely does not have. The NVML route
  works headless too, with persistence mode on.
- **Safety is proved by bit-exact output, not by "no crash".** GDDR5's CRC protects only the data bus: a weak cell, an
  address or command error, or a bit flipped while weights load leaves wrong weights in VRAM for the whole session.
  The fork's decode is deterministic, so with stable memory a one-token KLD run against a stock base file gives exactly
  the statistics of a stock rerun and the same top token everywhere. The protocol: one card, +200 steps, a streaming-bandwidth and kernel sweep per
  step, a CUDA pattern test, the exact-KLD check on Flint, an hour's soak, then one step back for margin, card by card.
- **Power and core clocks have nothing to give the fork's layer split today.** Busy cards run at their top boost bin
  (1,835–1,903 MHz) at 33–38 W average and ≤ 36 °C; the 75 W cap is active in under 4 % of busy samples. A higher power
  limit is impossible (75 W maximum), a lower one saves nothing, `nvidia-smi -lgc` is Volta+, and "prefer maximum
  performance" has no clock-down left to remove. A positive core offset is the only core lever: est. +0.5–2 % decode,
  +3–5 % prefill, with the same validation as memory. The classic mining tune (power down, core offset up, memory up)
  matters only once the pipelined executor keeps every card busy (Q87, Q104).
- **PCIe 2.0 is not available.** The cards themselves advertise 2.5 GT/s, so no BIOS, switch or `setpci` setting can
  train a faster link; Pascal VBIOS images are signature-checked. Even if it were possible it would buy ~1 % decode on
  nine cards and roughly halve load times.
- **Host settings are worth minutes of testing, not more:** ASPM is already off; CPU energy preference, idle states and
  thread/IRQ placement are est. 0–2 % on served tokens and belong in Q118's follow-up; huge pages, resizable BAR and the
  IOMMU give nothing measurable here. The host setting that matters most is a quiet machine: no apt, snap or relay work
  during timings, and the NVIDIA packages held, since a background driver upgrade breaks CUDA until the next reboot.

## Ranked settings

Gains are single-stream Flint decode on the fork (six or nine cards) unless the row says otherwise. "Owner effort" counts
the owner's own time; every change needs the owner's go-ahead.

| # | Setting | Expected gain (est.) | Risk | Reversible without reboot? | Owner effort | First experiment | Kill criterion |
|---|---|---|---|---|---|---|---|
| 1 | Memory clock offset per card through NVML, validated card by card | +3–5 % decode at +10 % memory clock (range +2–7 %); +5–9 % after Q88/Q99; prefill 0–3 % | Silent weight corruption; driver hang that needs a reboot at the console | Yes (offset 0; any reboot or driver reload clears it), unless the card hangs | Approve; be at the console for the first card (~1.5 h); ~20–30 min per further card | M0 range query, then M1–M3 on GPU0 (section 7) | GPU0's streaming read gains < 4 % at the last clean step minus margin, or six-card tg128 gains < 2 % with all cards set |
| 2 | Quiet host during timings: apt/snap timers stopped, relays paused, NVIDIA packages held | No mean gain; removes outliers and the risk of a driver/library mismatch mid-session | None | Timers: yes (runtime stop). Hold: yes (`apt-mark unhold`) | Approve the package hold | H0: list timers and running jobs before each timed block | — (hygiene) |
| 3 | Headless mode (Q83) with persistence mode on | Host RAM +0.5–0.8 GB (est. from the idle inventory); 0–2 % on served tokens; keeps NVML offsets and power settings between processes | Low; the console stays usable for the passphrase | Yes (`systemctl isolate graphical.target`) | Approve; nothing at the console | Q83 as planned, plus offsets read back after a process exits | No RAM or speed gain worth the desktop |
| 4 | CPU energy preference, idle-state limit, thread and IRQ placement | 0–2 % on served tokens (est.), mostly streamed replies | Low | Yes (sysfs writes; all reset at boot) | Approve | H1: Q118's served run with EPP `performance` and deep C-states disabled | < 1 % on served tok/s and no change in spread |
| 5 | Core clock offset through NVML | Decode +0.5–2 %; prefill +3–5 %; more under the pipelined executor's power cap | Core instability: wrong results or Xid errors | Yes (offset 0), unless the card hangs | Approve; at the console | C1 after row 1 is settled | < 3 % on a one-card prompt test at the last clean step |
| 6 | "Prefer maximum performance" (PowerMizer mode 1) | 0 % measured: busy cards already sit at their top boost bin | Low; higher idle power | Yes | — | None needed; re-check only if serving logs show clock-down | — |
| 7 | Power limit | 0 %: 75 W is the maximum and is rarely reached; 60 W clips only rare peaks | Low | Yes | — | None | — |
| 8 | PCIe 2.0 on the GPU links | Would be ~1 % decode on nine cards, ~2× load and slot-restore speed; **not available** | VBIOS flashing: bricked card | — | — | P1: read `LnkCap2` once (read-only) | `LnkCap2` lists only 2.5 GT/s (expected) |
| 9 | ASPM, resizable BAR / BAR1, huge pages, IOMMU, GNOME animations | 0 % (ASPM already off; Pascal has no resizable BAR; the rest is below noise at these copy sizes) | — | — | — | None | — |

## 1. Memory overclock on the P106-090 under Linux (question 1)

### 1.1 What memory the cards carry, and how to tell

- **Type and speed:** 6 GB of GDDR5 on a 192-bit bus at 8,008 MT/s (192 GB/s): six 8 Gb chips, presumably of the 8 Gbps
  grade the GTX 1060 6 GB uses (est.), so already at their rated speed; any overclock is beyond the vendor's guarantee.
- **Vendor:** nothing on Linux reports it. `nvidia-smi` gives the VBIOS version, NVML the bus width
  (`nvmlDeviceGetMemoryBusWidth`) but no vendor call, and NV-CONTROL (what `nvidia-settings -q` reads) has no vendor or
  memory-type attribute (checked in `nvml.h` and `NVCtrl.h`). NVIDIA's forum answers say it is not exposed on Linux;
  GPU-Z gets it on Windows from NVAPI. VBIOS images for GP106 mining and gaming cards list GDDR5 support for Hynix,
  Micron and Samsung together (TechPowerUp's database for P106-100 and GTX 1060 images), so the VBIOS version does not
  identify the vendor either. The ways that work are reading the chip markings with a card out of the chassis or booting
  Windows with GPU-Z, both hardware-level owner tasks.
- **Report:** owners of Manli's 6 GB P106-090 on a HiveOS forum thread (2019–2021) state Samsung memory on their cards.
  Not verified for this rig's cards.

### 1.2 Offsets miners report (reports, not measurements here)

| Card | Memory | Algorithm | Memory offset (transfer-rate units) | Result | Source |
|---|---|---|---|---|---|
| P106-100 | not stated | Ethash | +500 (shown as +250 for Windows), core +75 | 18.0 → 22.8 MH/s | Kryptex profile, 2023 |
| P106-100 | not stated | Ethash | +1,000 (+500 Windows), core +100 | 18.0 → 24.0 MH/s | Kryptex profile, 2023 |
| GTX 1060 6 GB | Samsung | Etchash | +800 (+400 Windows), core +50 | 18.7 → 22.0 MH/s | Kryptex profile, 2023 |
| GTX 1060 6 GB (ASUS) | Micron | Ethash | +800 (+400 Windows) stable; +880 gave artifacts on the desktop | 20.0 → 22.4 MH/s | Kryptex profile, 2021 |
| P106-090 6 GB (Manli) | Samsung (owners' statement) | Ethash | up to "+2000" tried, no gain | 15–16.5 MH/s whatever the memory offset; best 19 MH/s with core +300 and memory +0 | HiveOS forum thread |

What these say for this rig:
- **The usable margin of 8 Gbps GP106 memory is +500 to +1,000, i.e. +6 to +12.5 %,** with Samsung at the high end and
  card-to-card spread (est. from the rows above). Aged mining memory may have less margin; each card has to be measured.
- **Ethash is the wrong proxy for a 5-SM card.** The P106-090 owners found Ethash limited by the core, not the memory:
  memory offsets changed nothing, a core offset helped, and the card mined 15–19 MH/s against 18 (stock) to 24
  (overclocked) for the 10-SM P106-100. Its random 128-byte reads plausibly need more loads in flight than five SMs keep
  (est.). The fork's matrix kernels stream contiguous rows and already reach 77–92 % of the streaming ceiling, so they
  are memory-bound in a way Ethash on this card was not.

### 1.3 How to apply it on driver 580

**Route A, NVML (recommended).**
- `nvmlDeviceSetMemClkVfOffset(device, offset)`: "It works on Maxwell onwards GPU architectures", "requires elevated
  privileges" (NVML reference). The newer `nvmlDeviceSetClockOffsets` takes a P-state and an `nvmlClockOffset_t`; on
  Pascal the driver applies an offset to all performance levels anyway (driver README, Coolbits entry), so the older
  call is enough. Getters: `nvmlDeviceGetMemClkVfOffset` and `nvmlDeviceGetMemClkMinMaxVfOffset` (the allowed range).
- **Units are not documented**; one forum script doubles its argument before calling the newer function. The first
  step therefore reads `clocks.mem` from `nvidia-smi` under load after a small offset and fixes the units on this rig.
- **Support on Pascal is reported, not documented for this card:** a user reporting that Maxwell overclocking fails on
  the 580 branch states that it works on Pascal and newer. M0's range query answers it in seconds: `NOT_SUPPORTED` means
  route B.
- **No X needed, no persistent change:** the offset lives in the driver until the driver unloads or the machine
  reboots. With Xorg running the driver never unloads; headless, persistence mode keeps it loaded (section 1.4).
- **Per card:** the call takes one device handle; NVML numbers devices in PCI bus order, the same order as `nvidia-smi`
  and the project's `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **No install:** Python's `ctypes` calls `libnvidia-ml.so.1`, which the driver installs:

```python
#!/usr/bin/env python3
# nvml_memoc.py GPU mem|core [OFFSET]: print, or set (as root), one card's memory or core clock VF offset
import ctypes, sys
nv = ctypes.CDLL("libnvidia-ml.so.1")
nv.nvmlErrorString.restype = ctypes.c_char_p
def ok(r, what):
    if r != 0:
        sys.exit(f"{what}: {nv.nvmlErrorString(r).decode()}")
dom = {"mem": "MemClk", "core": "GpcClk"}[sys.argv[2]]
ok(nv.nvmlInit_v2(), "init")
h = ctypes.c_void_p()
ok(nv.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(sys.argv[1])), ctypes.byref(h)), "handle")
if len(sys.argv) > 3:
    ok(getattr(nv, f"nvmlDeviceSet{dom}VfOffset")(h, ctypes.c_int(int(sys.argv[3]))), "set")
cur, lo, hi = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
ok(getattr(nv, f"nvmlDeviceGet{dom}VfOffset")(h, ctypes.byref(cur)), "get")
ok(getattr(nv, f"nvmlDeviceGet{dom}MinMaxVfOffset")(h, ctypes.byref(lo), ctypes.byref(hi)), "range")
print(f"gpu {sys.argv[1]} {dom} offset {cur.value} (allowed {lo.value}..{hi.value})")
nv.nvmlShutdown()
```

**Route B, Coolbits with `nvidia-settings` (fallback).**
- `Option "Coolbits" "8"` enables the offset attributes `GPUMemoryTransferRateOffset[level]` and, on GTX 1000-series
  and later, `GPUMemoryTransferRateOffsetAllPerformanceLevels` and `GPUGraphicsClockOffsetAllPerformanceLevels`
  (NV-CONTROL attributes 410, 425, 424): `nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=800'`.
- **The catch on this rig:** the option "is set for an X screen" (driver README), and a GPU used through PRIME offload
  has no X screen of its own, so Coolbits is ignored there (NVIDIA forum, driver 440–460; an internal bug was filed, no
  fix confirmed). The rig's desktop runs on the Intel graphics with the nine cards attached to Xorg (4 MiB each), which
  matches that case (est.: the Xorg configuration was not dumped).
- **What route B needs:** an xorg.conf with one NVIDIA screen per card. `AllowEmptyInitialConfiguration` (on by
  default) lets the NVIDIA X driver start without displays, and `nvidia-xconfig --enable-all-gpus --cool-bits=8`
  writes such a file. That replaces the desktop's configuration and needs an Xorg restart: a persistent change with a
  small risk of a black login screen, recoverable from a text console but awkward on a machine whose console is also
  where the disk passphrase goes. Alternatively a second, bare X server for the NVIDIA cards only (`X :1 -config …`),
  whose offsets last only while it runs (est.).
- Route B is worth it only if M0 finds the NVML calls unsupported on the P106.

**Nine cards.** Each card gets its own offset from its own validation. In a layer split each card's gain is its share of
the token, so cards left at stock just don't speed up their share.

**Across reboots.** Nothing persists by default, which is the safe state: any reboot returns all cards to stock. After a
long clean soak, a systemd one-shot unit can re-apply the validated per-card offsets after the driver loads. It should
start a few minutes after boot and check for a disable file first, so that a card that hangs at the new setting can be
stopped from the console before the unit runs.

### 1.4 Headless (Q83 interplay)

- **Route A works without any X server.** Without Xorg holding the cards, the driver tears down a GPU's state when its
  last client exits unless persistence mode is on ("whether the GPU driver software is torn down after the last client
  exits", NVML reference); power limits are documented to need it for the same reason. Headless mode should therefore
  turn persistence mode on (`nvidia-smi -pm 1`) and re-apply offsets after enabling it, then read them back after a
  CUDA process has exited.
- On this install `nvidia-persistenced` runs but `nvidia-smi` reports persistence mode disabled (inventory), which fits
  the `--no-persistence-mode` flag Ubuntu's unit is reported to ship with. With Xorg running this does not matter.
- Route B needs an X server, so headless means a bare X server just for Coolbits, or giving up route B.

## 2. What it buys (question 2)

### 2.1 Decode

A six-card token (26.8 ms) is ~14.9 ms of weight streaming at ~150 GB/s plus ~11.9 ms that a memory clock does not touch
(boundaries and small kernels, page walks, hops, host). If a fraction f of the streaming time scales with the memory clock
and the clock rises by s, the token falls by 14.9 · f · s / (1 + s) ms (est.):

| Memory clock gain s | f = 0.7 (part latency-bound) | f = 1.0 (fully bandwidth-bound) | Nine cards (32.3 ms), f 0.7–1.0 | Six cards after Q88 (est. 21 ms, f 0.85–1.0) |
|---|---|---|---|---|
| +5 % (+400) | +1.9 % | +2.7 % | +1.5–2.2 % | +3.0–3.5 % |
| +10 % (+800) | +3.7 % | +5.3 % | +3.0–4.3 % | +5.8–7.0 % |
| +12.5 % (+1,000) | +4.5 % | +6.6 % | +3.7–5.3 % | +7.2–8.6 % |

- **Why f < 1.** The fork's kernels run at 77–92 % of the streaming ceiling in the microbenchmark and slower inside a
  token. A kernel that is short of loads in flight (Little's law) gains only the part of its latency that DRAM clocks
  set; the walk of a missed page (Q99, est. 1.3–2.0 µs each) is mostly latency too. M1 measures f per kernel directly:
  GB/s at each offset against the clock ratio.
- **The streaming read itself should scale:** at 179 GB/s five SMs hold ~21 KB each in flight (est., ~0.6 µs loaded
  latency); +10 % needs ~24 KB, within what 2,048 threads per SM with 128-bit loads can issue. M1's `STREAM_READ_PLAIN`
  line confirms it.
- **Issue-bound kernels gain nothing.** K-quant and IQ4_XS kernels on llama.cpp's stock path (12–50 instructions per byte,
  [K-quant note](kquant_decode_pascal.md)) are limited by the core; the fork's Pascal kernels (Q8_0 ~1.1, Q4_1 ~2.35,
  Q5_0 ~4.1 slot-equivalents per byte against ~5.3 available) stay memory- or latency-bound up to ~+25 % bandwidth
  (est.).
- **At long context** attention reads the KV cache at 57–87 GB/s ([attention note](pascal_attention_long_context.md)),
  well below the ceiling, so its share gains less than the weights.
- **Several conversations** (Q104's streams mode) are bound by each card's cycle, which splits the same way: the same
  percentages apply to aggregate throughput.
- **The gain grows as the engine improves.** Every fixed millisecond that Q88 (persistent kernel), Q99 (page walks),
  Q100 (norm writes Q8_1) and Q116/Q120 (host) remove raises the streaming share, toward ~70 % of a token at ~21 ms.

### 2.2 Prompt processing, power and temperature

- **Prefill:** MMQ is bound by `dp4a` issue and shared memory ([prefill note](prefill_pascal.md) §2.2): est. ~0–1 % today.
  After PR #31047 (Q108) the expert bytes (~1.5 ms of ~4–5 ms per prompt token at `-ub 128`, streamed at 120–140 GB/s)
  become a real share: est. +2–3 % at +10 %.
- **Power:** a higher memory clock at fixed memory voltage raises board power by est. 1–3 W. Busy cards average 33–38 W
  in the fork's layer split (section 3), so the 75 W cap stays out of reach; a fully loaded card would trade those watts
  against core clock.
- **Temperature:** GPU temperatures stay at 28–36 °C under the fork's layer split and reached 46–57 °C on single cards at
  full duty (calibration). The cards report no memory temperature; the chassis fans run at a fixed 4,800 rpm. GPU
  temperature is the only proxy, and the soak's limit is set on it (section 5).

## 3. Power and core clocks (question 3)

What the cards do now:

| Run | Duty per card | Core clock while busy | Board power while busy, mean / peak | Power cap active | Peak GPU temperature |
|---|---|---|---|---|---|
| Stock llama.cpp, Llama-2-7B on one card (calibration) | ~100 % | 1,710–1,760 MHz | ~69 W / 75 W | 50–90 % of samples | 46–57 °C |
| Stock, the same model on two cards | ~45 % | ~1,830 MHz | — | 14–22 % | — |
| Fork, Flint S4-L0 and UD-Q4_K_XL on six cards (Q38 runs) | layer split | 1,835–1,903 MHz per card | 34–37 W / 67–72 W | 0–4 % of busy samples | 30–36 °C |
| Fork, Flint Q8_0 on nine cards (Q38 run) | layer split | 1,841–1,900 MHz per card | 33–35 W / 40–70 W | ≤ 1 % of busy samples | 28–34 °C |

"Busy" = `nvidia-smi` samples with utilization ≥ 5 % while the model is loaded. Each card's top boost bin is 1,860–1,911
MHz (inventory), so under the fork's layer split a busy card already runs at its maximum clock, at half its power limit.

- **Power limit (`nvidia-smi -pl`):** 60–75 W; NVML's `nvmlDeviceSetPowerManagementLimit` "Kepler or newer", not
  persistent. 75 W cannot be raised. Lowering it saves nothing in layer-split decode, where cards draw ~35 W on average;
  it would cost speed on any fully loaded card (single-card runs, long prompts on few cards, the pipelined executor).
  E2.16 stays n/a.
- **Locked clocks:** `-lgc`/`nvmlDeviceSetGpuLockedClocks` are "Supported on Volta+"; `-lmc` is Ampere+ in NVML;
  application clocks are not supported on this card (measured). None is available, and none is needed while busy cards
  sit at the top bin.
- **PowerMizer "prefer maximum performance"** (`GPUPowerMizerMode=1`, NV-CONTROL attribute 334: "raise GPU clocks to favor
  maximum performance") was proposed against clock-down on lightly loaded cards
  ([dense 27B note](dense_27b_speedups.md), stock builds). On the fork's logs busy cards show no clock-down, and the
  attribute needs an NVIDIA X screen like Coolbits. Nothing to gain; re-check only if serving logs show low clocks.
- **Core offset** (`nvmlDeviceSetGpcClkVfOffset`, or Coolbits `GPUGraphicsClockOffsetAllPerformanceLevels`): shifts the
  boost table, so a busy card's top bin rises by the offset (+100 MHz ≈ +5 %).
  - Decode: the fork's kernels are not issue-bound (Q5_0 uses ~50 % of issue at 138 GB/s, est.), and of the ~12 ms of
    fixed time only the GPU-side part (small kernels, ramp-up, tails, ~6–8 ms) scales with the core clock: est.
    +0.5–2 % for +5 %.
  - Prefill: MMQ scales almost fully with the core clock (shared memory and `dp4a` run on it): est. +3–5 % for +5 %.
  - Risk: an unstable core produces wrong results or Xid errors; the validation is the same as for memory.
- **The classic mining tune** (power limit down, core offset up so each voltage step gives more MHz, memory up) buys
  efficiency under a binding power cap. The fork's layer split never binds the cap, so the tune has nothing to work on
  today. It becomes relevant when the pipelined executor (Q87) or the streams mode (Q104) keeps every card busy: then the
  cards meet the 75 W cap as in the single-card calibration (1,710–1,760 MHz), extra memory watts come out of the core
  clock, and a positive core offset recovers clock at the same power. That regime should be measured with the executor's
  rig runs (power, clocks and cap reasons logged) before any core offset is chosen.

## 4. PCIe link generation (question 4)

- **Measured:** each P106-090 advertises "Speed 2.5GT/s, Width x4" in Link Capabilities and trains at 2.5 GT/s x1. The
  ASM1184e downstream ports and the root ports advertise 5 GT/s; the switch uplinks run at 5 GT/s, GPU4's and the network
  controller's root ports at 2.5 GT/s because their devices do.
- **Why nothing upstream can change it:** a link trains to the highest rate both ends support. Linux's PCIe bandwidth
  controller, for example, writes "the highest speed supported by the Port and the PCIe device" into the downstream
  port's Target Link Speed and retrains; it cannot pick a rate the card does not advertise. BIOS options for root-port
  speed act on the same register. `setpci` writes to Link Control 2 plus a retrain end at 2.5 GT/s at best, and a failed
  retrain can drop the card from the bus until a reboot.
- **The cap is in the card.** GP106 silicon runs PCIe 3.0 x16 on the GTX 1060, so the mining SKU's limit is set by
  VBIOS straps or fuses (which one is not documented). Since Maxwell an on-die Falcon checks the VBIOS signature; a
  forum report says edits to signed regions of a Pascal image flash but the driver then rejects the card, and the public
  bypass tools are described as working on Turing and older without Pascal details. A P104-100 owner reports the same
  2.5 GT/s lock in that card's VBIOS, and no Gen2 unlock for P104/P106 cards exists
  ([Pascal forks](pascal_forks_and_quants.md), [community reports](quant_speed_d_community.md) X8, X9). The CMP unlocks
  target Turing and Ampere cards through GSP-era open kernel modules, which do not support Pascal.
- **The one read-only check left (P1):** `lspci -vvv -s 03:00.0 | grep -E 'LnkCap2|LnkCtl2'`. Software is supposed to read
  supported speeds from Link Capabilities 2; if it lists 5 GT/s despite the 2.5 GT/s maximum, a retrain test becomes
  thinkable (owner at the console). If it lists only 2.5 GT/s, the question is closed.
- **What Gen2 would have bought (est.):**
  - Hops: each staged 8 KB hop is two ~43 µs legs at ~190 MB/s plus fixed latency; halving the transfer part saves
    ~40 µs per hop, ~0.3 ms per token on nine cards (~1 %).
  - The host mask copies at depth (~3 ms per decode token at 32k on nine cards, 3.2 ms per prompt token): halved, but
    Q110/Q116 remove them anyway.
  - Loading and slot restores: one card at a time up to ~2× (~380 MB/s against ~190), all cards at once capped by the two
    5 GT/s switch uplinks shared by four cards each: Flint's 199 s load would fall to est. 100–120 s.
  - The 2.8 ms logits copy of tool requests on the host-sampling path: 1.4 ms.

## 5. Proving it safe without ECC (question 5)

### 5.1 What can go wrong

- **What GDDR5 protects:** the DRAM computes a CRC-8 over each byte lane's burst (8 DQ + DBI, 72 bits), "securing all DQ
  and DBI# lines", and returns it on dedicated EDC pins; "the command that caused the error can be repeated" (Qimonda's
  GDDR5 white paper; Micron's GDDR5 briefs describe CRC-8 on read and write data). That covers transmission errors on the
  data bus. It does not cover stored bits (no ECC on these cards), address and command transfers, or anything inside the
  GPU. Whether GP106's controller enables EDC replay is not documented by NVIDIA.
- **So a too-fast memory clock can fail silently.** A flipped bit in a weight written during load, or lost by a weak cell,
  stays wrong for the whole session and shifts logits by an amount no crash reveals; transient read errors corrupt one
  token. Bandwidth that stops rising (replays, if EDC is on) is an early sign, not a guarantee.
- **The fork makes silent errors visible.** Its decode is deterministic: the same build, file, card set and inputs give
  byte-identical logits (shown in the [prefill](prefill_pascal.md) and [pipelined speculation](pipelined_speculation.md)
  checks). With stable memory, a run against a stock KLD base file gives exactly the statistics of a stock rerun against
  the same file, with `same top p` 100 %. llama-perplexity stores the base log-probabilities in 16 bits, so identical
  logits still show `Mean KLD` 0.000000 and a maximum of ~1e-5, not 0. Any difference from the stock rerun's statistics
  is an error on some card in the run. One-token steps (`-ub 1`) exercise the decode kernels; a few thousand tokens
  route through nearly every expert, so most of the card's weights are read.
- **Hangs:** a memory clock far past the margin can stop the card (Xid 79 "fallen off the bus", or 13/31/43 in a CUDA
  process). `nvidia-smi --gpu-reset` needs every process off the card, Xorg included, and its support on this GeForce-class
  card is not documented, so a hung card means a reboot, i.e. the owner at the keyboard for the passphrase. A link drop
  behind an ASM1184e switch could also stall the host (est.).

### 5.2 Protocol per card

All steps on one card at a time, with nothing else on the rig, `nvidia-smi` logging clocks, power and temperature, and the
kernel log scanned for Xid after every step (`journalctl -k --since '-10 min' | grep -i 'NVRM: Xid'`).

| Stage | What | Time | Pass |
|---|---|---|---|
| 0 | Read-only: NVML offset range (M0), `LnkCap2` (P1), CPU frequency driver and timers (H0) | 1 min | the set and get calls exist (not `NOT_SUPPORTED`) |
| 1 | Stock baselines on the card: `bench/mmvq_pascal` with its correctness check and timing (streaming read, Q8_0/Q5_0/Q4_1 kernels); two stock Flint `-ub 1` KLD runs on six cards, the second against the first's base file | ~25 min (one model load) | the second stock run gives KLD 0 and 100 % same top token (determinism proved on the rig) |
| 2 | Offset sweep on the card: +200, +400, … up to the reported range (+1,000, at most +1,200), each with the bench (check + timing) | ~5 min per step | correctness check passes; no Xid; streaming GB/s rises with the clock |
| 3 | Pattern test at the highest clean step: `cuda_memtest` (tests 0–8 and 10: walking ones on the address lines, own-address, moving inversions, random patterns), one pass, built from source for `sm_61` | est. 10–20 min | no error |
| 4 | Flint `-ub 1` KLD at that step against the stock base (six cards, the card in the set) | ~6 min + load | statistics identical to the stock rerun's, 100 % same top token |
| 5 | Margin: set one step lower (−200, ~2.5 %); 1-hour soak repeating stage 4 in a loop | 1 h | every run identical to the stock rerun; GPU temperature < 70 °C; no Xid |
| 6 | After every card: the nine-card `-ub 1` KLD run and tg128 against stock on six and nine cards | ~30 min | identical to stock; tg128 gain ≥ 2 % (else restore stock everywhere) |

- **Step size and margin.** +200 is ~2.5 % of the memory clock; the final setting is one step below the last step that
  passed stages 2–4, because rare errors at the edge can take hours to show (memtest_vulkan's authors note errors that
  appear only after warm-up and recommend ≥ 6 minutes, rarely 2–3 hours).
- **Failure at any stage** marks the previous step as the card's limit. Bandwidth that stops rising, or falls, between two
  steps is treated as a failure too.
- **Rollback without a reboot:** `python3 nvml_memoc.py N mem 0` restores stock immediately; nothing else was changed. Only a
  hang needs the reboot, and then every card is back at stock by construction.
- **In service:** each session that uses offsets starts with one `-ub 1` KLD canary against the stock base of the same
  build and file (~2 min after the load). This also catches a card whose margin has drifted with age or temperature.
- **Tools.** `cuda_memtest` (NCSA licence, maintained fork) builds with CMake and `-DCMAKE_CUDA_ARCHITECTURES=61` on the
  quant machine; select the card with `CUDA_VISIBLE_DEVICES`. memtest_vulkan (prebuilt Vulkan binary, zlib licence, aimed
  at overclock validation and address-bus faults) is a second opinion if the owner accepts a downloaded binary.
  gpu-burn is a compute stress test and adds little for memory. NVIDIA's nvbandwidth measures memcpy patterns between
  host and devices, not the in-kernel streaming read that `bench/mmvq_pascal` already reports.
- **Results stay publishable.** A stable offset changes no output bit, so every quality result stands. Published speeds
  should stay at stock clocks, with overclocked numbers as separate, labelled rows.

## 6. Other settings in the same class (question 6)

| Setting | State on this rig | Effect on our numbers (est.) | Action |
|---|---|---|---|
| PCIe ASPM | Disabled on every GPU link, switch port and root port (inventory); root ports don't support it | none left | none (E2.18 n/a) |
| CPU frequency (Celeron 3865U, 1.8 GHz, no turbo) | Not recorded | The submitting thread yield-spins and keeps its core at full clock; the server and HTTP threads on the other core may run at lower P-states between bursts. EPP `performance` (`energy_performance_preference`, or the `performance` governor, which writes EPP 0 under `intel_pstate` with HWP): 0–1 % on streamed replies | H1, runtime sysfs write, reset at boot |
| CPU idle states | Not recorded | Deep C-state exits on the second core delay the HTTP thread; disabling states with exit latency above ~50 µs (`cpuidle/stateN/disable`, or holding `/dev/cpu_dma_latency` open) costs a few watts: 0–1 % | H1 |
| IRQ affinity, thread placement | irqbalance default (not checked) | CUDA's default yield wait polls, so GPU interrupts are rare; network interrupts and the HTTP thread compete with the submitting thread. Placement is change #1 of the [host note](host_decode_loop.md): 0–2 % with it | in Q118's follow-up |
| `nvidia-persistenced` / persistence mode | Daemon active, mode reported disabled; Xorg keeps the driver loaded | none while Xorg runs; needed headless to keep offsets and power limits and to skip driver initialization per process | with Q83 |
| Huge pages / THP | Ubuntu default | Pinned buffers come from the NVIDIA driver, and host-side copies are KB per token; nothing measurable | none |
| IOMMU | Kernel command line has no IOMMU options (inventory) | Translation on staged copies is negligible at 190–213 MB/s per link; changing it needs a reboot | none; a read-only check (`/sys/class/iommu`) if hop costs (Q81) look high |
| Resizable BAR / BAR1 | Not supported on Pascal GeForce | llama.cpp maps no device memory through BAR1 and peer access is off | none |
| GNOME animations, compositor | The desktop draws on the Intel graphics | CPU wake-ups only when the screen is in use; headless mode covers it | Q83 |
| Background jobs (apt/snap timers, relays, `tailscaled`) | apt and snap timers active by default on Ubuntu; a relay drives `tailscaled` to ~90 % of a core | Minutes-long full-core jobs on a 2-core host are the largest source of outliers; an unattended NVIDIA driver upgrade breaks CUDA until the next reboot | H0 before every timed block: stop the timers for the session, pause relays; `apt-mark hold` the NVIDIA packages (owner's call); serve LAN clients on the LAN address rather than the tailnet |
| Memory timings (straps) | In the signed VBIOS | Timing tweaks for GP106 need VBIOS edits | none |

## 7. Plan (question 7)

### 7.1 First experiments

[bench/mem_offset](../../../bench/mem_offset/README.md) packs M0–M3 and the soak (stages 0–5) into one kit: the NVML
script with a read-only probe and a measured unit for the offset, the step script with a reset trap, the mmvq and
`cuda_memtest` binaries for sm_61, and the exact check. The commands below show the same steps by hand.

`B=/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin`, `M` the six-card Flint file used in the session
(UD-Q4_K_XL or S4-L0), `WIKI` wikitext-2's `wiki.test.raw`, `E="LLAMA_NO_MOE_DOWN_RESCALE=1"` (`=tg` for UD-Q4_K_XL),
`nvml_memoc.py` from section 1.3 and the `bench/mmvq_pascal` binary copied to `/tmp/oc` on the rig.

**M0, H0, P1 (read-only, ~1 min):**

```bash
ssh rig 'for g in 0 1 2 3 4 5 6 7 8; do python3 /tmp/oc/nvml_memoc.py $g mem; python3 /tmp/oc/nvml_memoc.py $g core; done'
ssh rig "lspci -vvv -s 03:00.0 | grep -E 'LnkCap|LnkCtl2|LnkSta'"
ssh rig 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor \
    /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference 2>/dev/null; \
    grep . /sys/devices/system/cpu/cpu0/cpuidle/state*/{name,latency}; systemctl list-timers --no-pager | head -20'
```

Without an offset argument the script only reads. Kill for route A: the getters return `NOT_SUPPORTED`.

**M1, one-card sweep (owner at the console, ~45 min):**

```bash
ssh rig 'nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 500 > /tmp/oc/smi.csv 2>&1 &'
for off in 0 200 400 600 800 1000; do
  ssh rig "python3 /tmp/oc/nvml_memoc.py 0 mem $off && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /tmp/oc/mmvq_pascal --types q8_0,q5_0,q4_1 --csv /tmp/oc/bench_$off.csv; echo exit=\$?; \
      journalctl -k --since '-10 min' | grep -i 'NVRM: Xid' || echo no-xid"
done
ssh rig 'python3 /tmp/oc/nvml_memoc.py 0 mem 0; pkill -f "nvidia-smi -i 0"'
```

Read: `clocks.mem` under load against the offset (fixes the units: +200 should read ~4,104 MHz if NVML uses
transfer-rate units); `STREAM_READ_PLAIN` and each kernel's GB/s against the clock ratio (f of section 2.1); the bench's
exit status (2 = a correctness check failed). Stop the sweep at the first failure.

**M2, pattern test (~20 min):** at the highest clean step, `CUDA_VISIBLE_DEVICES=0 ./cuda_memtest --num_passes 1
--exit_on_error`, then offset 0.

**M3, exact output on Flint (~25 min with loads):**

```bash
PPL="-m $M -ngl 99 -fa on -f $WIKI -c 512 --chunks 8 -b 512 -ub 1"   # 4,096 one-token steps, ~2 min on six cards
ssh rig "$E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $B/llama-perplexity $PPL --kl-divergence-base /tmp/oc/base.kld"
ssh rig "$E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $B/llama-perplexity $PPL --kl-divergence-base /tmp/oc/base.kld --kl-divergence"  # stock: expect 0
ssh rig "python3 /tmp/oc/nvml_memoc.py 0 mem <step> && $E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $B/llama-perplexity $PPL \
    --kl-divergence-base /tmp/oc/base.kld --kl-divergence; python3 /tmp/oc/nvml_memoc.py 0 mem 0"
# afterwards: rm -rf /tmp/oc on the rig (the base file is ~0.6 GB)
```

Pass: the offset run's statistics block equals the stock rerun's, digit for digit, with "Same top p" 100 % (section 5.1).
GPU0 holds the first ~8 layers in the six-card split.

**M5, the payoff (~15 min):** with the validated offsets, `llama-bench -p 0 -n 128 -r 3` on six and nine cards against the
same run at stock; the kill criterion of row 1 applies here.

**H1 (with Q118's served run, ~20 min):** the same non-streamed and streamed requests with EPP `performance` on both cores
and deep idle states disabled, against the defaults; served tok/s and its spread.

**C1 (later):** the M1 sweep with `nvml_memoc.py 0 core <offset>` in +25 MHz steps (one boost bin is ~13 MHz), a one-card
prompt test (llama-bench pp512 with a model that fits one card, as in the calibration) and the same M2–M5 gates.

### 7.2 What can run when

- **In today's rig session, after the timed blocks and with the owner's OK:** M0/H0/P1 (read-only), then M1–M3 and M5
  on GPU0 (~1.5 h together, owner within reach of the console). Nothing in them needs a reboot, a package or a
  configuration file. H1 fits into the Q118 follow-up.
- **Next sessions:** the other eight cards (~20–30 min each, the same stages), the nine-card check and the one-hour soak;
  the executor's power regime (Q87/Q104 rig runs with power logging), then C1 if that regime is power-bound.
- **Only after a ≥ 24 h clean record:** the boot-time unit that re-applies offsets (a persistent change).
- **Needs a reboot or hardware:** none of the recommended steps. A hang needs a reboot with the passphrase. PCIe 2.0 is a
  hardware limit of the cards. Route B (Coolbits) needs an Xorg restart, not a reboot. A second SO-DIMM (Q86) is a
  power-down.

## What this changes in the plan

- **Q86's memory overclock is a runtime NVML change on one card, with an exact-output protocol,** not an xorg.conf edit.
  Its expected value is +3–5 % decode now (not +x % for +x % clock), rising toward +5–9 % as Q88/Q99/Q100 land; it stays
  the owner's decision.
- **E2.16 (power-limit sweep) stays n/a, and the clock-mode experiments of the dense-model note expect nothing on the
  fork:** busy cards already run at their top boost bin at half their power limit.
- **The executor's rig runs (Q87, Q104) should log power, clocks and cap reasons**: they are the first workload that will
  meet the 75 W cap, which is where memory watts and core offsets start to trade.
- **PCIe 2.0 is closed** pending the one read-only `LnkCap2` check; **ASPM is closed**.
- **A quiet-host checklist belongs before every timed block:** apt/snap timers, relays, and the NVIDIA packages held.
- Suggested registry rows (not added here): M0–M5 under Q86 (one card, then all); C1 as its own row after the executor's
  power data; H0/H1 under Q118; P1 as a one-line check.

## Already covered

| Finding in this research | Where it is |
|---|---|
| Memory overclock as the only lever on the decode ceiling; mining profiles; P2 state; clock locking; persistence; ASPM | [multi-GPU physics](multigpu_pascal_performance.md) Q8, E12–E15; survey C18–C22, E2.15–E2.19 |
| Memory clock P0 at 4,004 MHz under compute, application clocks N/A, ASPM disabled, link capabilities, power-limit range | E0.1 inventory; [hardware spec](../../mining_rig.md); [calibration](../../benchmarks/2026-10-22-single-card-calibration.md) |
| Headless mode, persistence mode, RAM freed | [operations note](ops_and_benchmarking.md) §3; Q83 |
| PowerMizer mode, core offset under the cap, `-lgc` Volta+ | [dense 27B note](dense_27b_speedups.md) §5, experiments 6, 14, 17 |
| Core-clock sensitivity of an issue-bound card; a display costs ~13 % (a card driving a monitor); the P104/P106 PCIe lock reports | [community reports](quant_speed_d_community.md) §1.13, K2, X8–X10 |
| No PCIe or instruction-rate unlock for Pascal mining cards; CMP unlocks need GSP-era modules; Q103's throughput calibration | [Pascal forks](pascal_forks_and_quants.md) §1, §6; Q103 |
| Page walks as a fixed cost per token | Q99; [Pascal forks](pascal_forks_and_quants.md) §3 |
| `-t 1`, the OpenMP spinner, synchronization policy, thread placement, page faults on the mapped embedding table | [host note](host_decode_loop.md); Q118 |
| Second SO-DIMM | Q86, E2.19 |
| Thermal steady state and power-log validity | E0.12 |

## Not verified

- That NVML's memory and core VF offsets are accepted on the P106-090 under driver 580, their units, and that offsets
  persist after the setting process exits while Xorg holds the driver.
- That this rig's Xorg attaches the cards as offload GPUs without X screens (the Xorg configuration was not dumped), and
  so that Coolbits would be ignored as it stands.
- The memory vendor of this rig's cards and their overclock margin; all margins above are other cards' reports.
- That GP106's memory controller enables GDDR5 EDC replay, and so whether bandwidth flattens before errors appear.
- The scaling fraction f of the fork's kernels with the memory clock (M1 measures it) and the streaming read's scaling.
- Memory power at a higher clock (1–3 W is an estimate; the cards report board power only).
- `LnkCap2` of the cards, and whether the 2.5 GT/s cap is a strap, a fuse or the VBIOS.
- `nvidia-smi --gpu-reset` on this card, and the host's behaviour when one card drops off the bus behind a switch.
- The CPU frequency driver, governor, idle states and IRQ distribution on the rig (H0 reads them).
- `cuda_memtest`'s run time for 6 GB on a P106-090.
- That Ubuntu's `nvidia-persistenced` unit carries `--no-persistence-mode` on this install (forum reports; the inventory
  shows the effect).

## Sources

NVIDIA documentation and code (P)
- NVIDIA Linux driver README (580 series), "X Config Options": `Coolbits` (bit 3 offsets, "On GPUs based on the Pascal
  architecture the offset is applied to all performance levels", applied per X screen), `AllowEmptyInitialConfiguration`,
  <https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/xconfigoptions.html>
- NV-CONTROL header `NVCtrl.h` (nvidia-settings): `NV_CTRL_GPU_POWER_MIZER_MODE` (334), `NV_CTRL_GPU_NVCLOCK_OFFSET` (409),
  `NV_CTRL_GPU_MEM_TRANSFER_RATE_OFFSET` (410), `…_ALL_PERFORMANCE_LEVELS` (424, 425), no memory vendor attribute,
  <https://github.com/NVIDIA/nvidia-settings/blob/main/src/libXNVCtrl/NVCtrl.h>
- NVML API reference, device commands: `nvmlDeviceSetMemClkVfOffset`, `nvmlDeviceSetGpcClkVfOffset` ("Maxwell onwards"),
  `nvmlDeviceSetGpuLockedClocks` (Volta+), `nvmlDeviceSetMemoryLockedClocks` (Ampere+), `nvmlDeviceSetPowerManagementLimit`,
  `nvmlDeviceSetPersistenceMode`, <https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceCommands.html>; `nvml.h`
  (signatures, `nvmlClockOffset_v1_t`, getters, no vendor query), <https://github.com/NVIDIA/go-nvml/blob/main/gen/nvml/nvml.h>
- nvidia-smi documentation: `-lgc`/`-rgc` "Supported on Volta+", `-pl`, `-ac`, `-pm`, clock event reasons,
  <https://docs.nvidia.com/deploy/nvidia-smi/index.html>

Memory (P)
- Qimonda, "GDDR5 – White Paper" (2007), section 4.2 "Error detection mechanism",
  <https://xdevs.com/doc/_Datasheets/RAM.DDR5/Qimonda_GDDR5_whitepaper.pdf>
- Micron, 4Gb and 8Gb GDDR5 SGRAM product briefs (CRC-8 on read/write data), <https://www.mouser.com/datasheet/2/671/8gb_gddr5_sgram_brief-1285024.pdf>
  (read through a search excerpt)

Linux kernel (P)
- `intel_pstate` (HWP, `performance`/`powersave`, `energy_performance_preference`), <https://docs.kernel.org/admin-guide/pm/intel_pstate.html>
- CPU idle time management (`stateN/disable`, `latency`, `/dev/cpu_dma_latency`), <https://docs.kernel.org/admin-guide/pm/cpuidle.html>
- PCIe bandwidth controller, "[PATCH v9 7/9] PCI/bwctrl: Add API to set PCIe Link Speed" (speed = highest supported by
  port and device), <https://lkml.iu.edu/hypermail/linux/kernel/2410.2/04672.html>; `drivers/pci/pcie/bwctrl.c`

Test tools (P)
- [ComputationalRadiationPhysics/cuda_memtest](https://github.com/ComputationalRadiationPhysics/cuda_memtest) (tests 0–10, `--stress`, CMake)
- [GpuZelenograd/memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan) (overclock validation, address-bus faults, run ≥ 6 min)
- [wilicc/gpu-burn](https://github.com/wilicc/gpu-burn); [NVIDIA/nvbandwidth](https://github.com/NVIDIA/nvbandwidth);
  [nvidia_oc](https://docs.rs/nvidia_oc) (NVML overclocking tool, boot-time service example)

Reports (S, labelled where used)
- Kryptex overclocking profiles: [P106-100 low](https://www.kryptex.com/overclocking/nvidia-p106-100-low-overclock-2),
  [P106-100 high](https://kryptex.com/cs/overclocking/nvidia-p106-100-high-overclock-2),
  [GTX 1060 Samsung](https://www.kryptex.com/overclocking/nvidia-gtx-1060-samsung-medium-overclock),
  [ASUS GTX 1060 Micron](https://www.kryptex.com/overclocking/asus-dual-geforce-gtx-1060-oc-6gb)
- HiveOS forum, "Nvidia p106-090 slow hash rate after OC", 2019–2021,
  <https://hiveon.com/forum/t/nvidia-p106-090-slow-hash-rate-after-oc-only-15mh-s-help/11034>
- NVIDIA forums: Coolbits ignored under PRIME render offload, <https://forums.developer.nvidia.com/t/option-coolbits-is-not-used-optimus-enabled-laptop-running-an-rtx-2070-manjaro-linux/111771>;
  `nvmlDeviceSetClockOffsets` per-P-state behaviour and doubled memory offsets, <https://forums.developer.nvidia.com/t/nvmldevicegetminmaxclockofpstate-nvmldevicesetclockoffsets-issues/318332>;
  overclocking on Maxwell versus Pascal under 580, <https://forums.developer.nvidia.com/t/overclocking-doesnt-work-on-maxwell-gpus/337911>;
  memory vendor not exposed on Linux, <https://forums.developer.nvidia.com/t/memory-brand-type/167451>;
  `nvidia-persistenced` unit flags, <https://forums.developer.nvidia.com/t/setting-up-nvidia-persistenced/47986>
- TechPowerUp forum on signed regions of Pascal VBIOS images, <https://techpowerup.com/forums/goto/post?id=5088818>;
  Igor's Lab on signature-bypass tools (Turing and older), <https://www.igorslab.de/en/durchbruch-bei-nvidia-bios-signature-lock-neue-tools-ermoeglichen-vbios-modding-und-crossflash/>;
  TechPowerUp VGA BIOS database entries for P106-100 and GTX 1060 images (GDDR5 Hynix/Micron/Samsung support)

This project
- [Quant registry](../../../quants/experiments.md) (Q38, Q76, Q81, Q83, Q86, Q87, Q88, Q99, Q100, Q103, Q104, Q108, Q110,
  Q116, Q118), [survey tracker](../../experiments.md) (E0.1, E0.12, E2.15–E2.19), [hardware spec](../../mining_rig.md),
  [single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md), [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md)
  (streaming read, correctness check, `--spread`, `--tlb-probe`), the notes linked above, and the `nvidia-smi` logs of the
  Q38 rig runs (raw logs kept outside the repository)
