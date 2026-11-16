# Can the driver hold a P106 at its top clock, so the engine needs no keep-alive? (2026-11-17)

**No setting was found that makes driver 580 hold a Pascal card at its top clock: every documented control either
picks the power state, which the cards already hold while they decode, or locks the clock to its base, and the
controls that lock a chosen or boosted clock are written for newer cards. So there is nothing for the owner to do
at the console: the three controls still worth one try are run-time commands, undone by one command each, with no X
screen, module reload or reboot, and the one module-parameter route, the PowerMizer registry keys, left the driver
at version 530. The test that would prove a route is the one-token decode on nine cards with the keep-alive off and
a 100 ms clock log: the route holds the clock if every card reads 1,800 MHz or more in every sample and a token
takes no more than 1 ms longer than with the keep-alive.**

That "no" rests on NVIDIA's documents and on owners' reports, not on a rig run: the three one-command checks of
registry row Q1553 would turn it into a measurement.

Scope: registry row Q1530 in [quants/experiments.md](../../../quants/experiments.md). A P106-090 falls to its base
clock of 1,354 MHz between bursts of work. Miner Llama holds it at 1,873–1,911 MHz with a keep-alive: a small block
of kernel launches on each card. That block has a price: Flint reads prompts 16.5 % slower with it, and deep into a
context long prompts read 2.3 % faster without it on Flash-Next and 2.6 % on Ornith, while a 64-token prompt reads
9.0 % slower without it on Flash-Next and 17.8 % on Ornith
([R165](../../data/2026-11-16_r165_flint_stack_split/README.md),
[R167](../../data/2026-11-16_r167_keepalive_depth/README.md)). This note asks whether a driver setting could hold
the clock instead, on a host whose cards have no display outputs.

It is desk work. NVIDIA's documentation, the public source of its kernel modules and of `nvidia-settings`, the NVML
header and owners' reports were read. Three repositories were cloned and read; nothing was built, run or installed
from them, and no command went to any machine.

Conventions follow the other notes:
- **Measured** means measured on this project's rig, with the folder linked.
- A **report** is someone else's result. A **claim to test** is anything this note infers, or that a document says
  and no rig run has confirmed; every such statement carries the label.
- Sources: **P** primary (NVIDIA's documentation and source), **S** secondary (forum reports), each dated in the list
  at the end.

## What the rig has already measured

| Fact | Numbers | Source |
|---|---|---|
| A card that decodes one token at a time sits at base | all nine cards in P0 at 1,354 MHz in every sample, the memory clock full (4,006 MHz), no clock-event reason set | [card clocks](../../data/2026-11-05_card_clocks/README.md) |
| The clock follows load, slowly | after a prompt the boost lasts 4–7 s of light load; a sleeping card reaches 1,650–1,860 MHz within about a second | the same folder |
| The keep-alive holds the top clock | every card at 1,873–1,911 MHz; a plain token 62.75 and 59.62 ms off against 51.06 and 51.08 on; 27–30 W a card off against 35–40 W on (+72 W for nine) | [keep-alive](../../data/2026-11-05_keepalive/README.md) |
| PowerMizer mode 1, set through the X server, holds the power state and not the clock | an idle card: P0, 1,354 MHz, 26.0–26.3 W, against P8, 139 MHz, 5.7–6.0 W in the default mode; a card's first busy sample 1,354–1,493 MHz in mode 1; tg128 from idle 29.75 tok/s against 29.79 | [Q141](../../data/2026-10-24_rig_q141/README.md), driver 580.178.04 |
| Application clocks do not exist on the card | `nvidia-smi` lists its supported clocks as N/A | registry Q154 |

So two things are settled before any reading: the cards are already in their top power state (P0) while they
decode, and the loss is inside P0, between the base clock and the boost above it. A setting that only chooses the
power state cannot help.

## The routes

| # | Route | Documented | Needs an X screen | Pascal on driver 580 | What it would hold | Verdict |
|---|---|---|---|---|---|---|
| 1 | PowerMizer mode 1, "prefer maximum performance" | yes: the NV-CONTROL and NVML headers (P) | no longer: NVML has the call, and `nvidia-settings` 580 uses it first (claim to test on a P106 with no X client) | accepted through the X server (measured) | P0 while idle, at the base clock (measured) | **not a clock hold.** It is a way to keep cards awake between turns (Q1554) |
| 2 | PowerMizer mode 3, "prefer consistent performance" | yes, the same headers: "lock to GPU base clocks" (P) | as route 1 | claim to test | 1,354 MHz always: the opposite of the goal (claim to test) | not for speed; a tool for repeatable timings (Q1555) |
| 3 | The PowerMizer registry keys as a module parameter (`PowerMizerEnable`, `PerfLevelSrc`, `PowerMizerLevel`, `PowerMizerDefault`, `PowerMizerDefaultAC`) | never by NVIDIA; the parameter that carries them is described in the kernel-interface source (P) | no | reports: no effect from 530.41.03 on, the names gone from the driver (S) | a performance level, on the drivers that had them | **gone.** Also a level is a power state, not a clock |
| 4 | `OverrideMaxPerf`, `RMForcePstate` as module parameters | no; both called unsupported by a reply written as from inside NVIDIA (S) | no | `OverrideMaxPerf`: removed at 465.24.02 (S). `RMForcePstate`: reported to exist still (S); on a P106 a claim to test | a power state | not a clock hold |
| 5 | Locked clocks, `nvidia-smi -lgc 1911,1911` | yes, "Supported on Volta+" (P) | no | reports of refusal on a GTX 1060, a Quadro P1000, a Quadro P2000 and a GTX 1080 Ti (S); never tried on the rig | the exact clock, if accepted | **est. refused**; one command settles it (Q1553) |
| 6 | Application clocks, auto boost | yes; on Pascal auto boost is "controlled through application clocks" (P) | no | N/A on the card (measured) | — | closed |
| 7 | The CUDA clock override, `nvidia-smi -cc 1` | yes, with no list of cards (P) | no | claim to test; asked for on a Titan Xp in 2020 without an answer (S) | lifts a limit the driver sets for CUDA work; the P106 already decodes in P0 | est. no change; one command (Q1553) |
| 8 | A core clock offset through NVML | yes (P) | no | claim to test (registry Q135) | nothing: it moves the whole curve, base included | not a hold; it stays Q135's question |
| 9 | The driver's own control calls: the P-state boost and the rated-clock control | in the public control headers only (P); no tool or manual | no | claim to test; NVIDIA's profiler offers the boost lock "on Ampere and newer" (P) | the boost point, if a Pascal card took the request | **est. refused**; parked (Q1556) |

## 1. The PowerMizer mode

**What it is.** One attribute per card, with four values in NVIDIA's headers: 0 adaptive (clocks follow
utilization), 1 prefer maximum performance, 2 auto (the default on the rig), 3 prefer consistent performance, which
the header describes as "lock to GPU base clocks".

**It no longer needs an X screen.** The NVML header has `nvmlDeviceGetPowerMizerMode_v1` and
`nvmlDeviceSetPowerMizerMode_v1`, marked "For Maxwell or newer fully supported devices". In the `nvidia-settings`
source at tag 580.95.05, a set on a GPU target goes to NVML first and reaches the X server only when NVML does not
have the attribute; with no X connection the NVML result is returned. So these should work from a shell with no
display (claim to test):

```bash
# read, set mode 1 on card 0, set the default back
ssh rig "nvidia-settings -q '[gpu:0]/GPUPowerMizerMode'"
ssh rig "nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'"
ssh rig "nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=2'"
```

That a P106-090 accepts the NVML call with no X client is a claim to test: Q141 set the mode with the X server
running, and the P106 is not one of NVML's "fully supported" products.

**What mode 1 does on the rig (measured, Q141).** It keeps an idle card in P0 at 1,354 MHz and 26.0–26.3 W, where
the default lets it sleep in P8 at 5.7–6.0 W. It does not raise a busy card's clock and does not make a card start
its work at a higher clock. So it holds the power state and leaves the clock inside it to the driver.

- Undone by: mode 2 on each card.
- After a reboot: gone (claim to test for the NVML path; it is driver state, not a file).
- Idle power: about +20 W a card that would otherwise sleep (measured), est. +180 W for nine.

**What mode 3 would do (claim to test).** Pin the clock at 1,354 MHz under any load. That is a loss for speed. It
is what NVIDIA's profiler did by default until its 2026.1 release, to make timings repeatable, and it could do the
same for this project's diagnostic A/B runs (Q1555).

### No route here waits on an X screen on the cards

- The mode was set on the rig with the stock driver: `nvidia-settings` reached all nine cards as GPU targets through
  the desktop's X server, whose display is on the integrated graphics (measured, Q141). No card carried an X screen
  of its own.
- Through NVML the mode needs no X server at all (claim to test).
- The registry keys, `nvidia-smi` and the driver's control calls never involve X.

So the verdict would be the same if a P106 could carry an X screen. Where a screen driven by the card would matter
is a setting the driver ties to one, such as the Coolbits clock offsets ([hardware tuning](hardware_tuning.md) §1.3,
route B), and those have an NVML route too. The stock driver treats a P106 as a compute-only card and refuses it the
3D path (a report); whether it would still let the card carry an X screen is not known here. The owner named
[dartraiden/NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher), which patches NVIDIA's Linux installer so
that mining cards, the P106-090 among them, get that path. It is the possible way to an X screen on these cards
(claim to test). [What a P106-090 holds back](p106_090_holds_back.md) §6 reads the patch and its safety. It replaces
the host's driver files, which is the owner's decision, and no route in this note needs it.

## 2. Registry keys as a kernel-module parameter

**The carrier is real and documented in source.** The kernel-interface source shipped with the driver
(`nv-reg.h`) describes the module parameter `NVreg_RegistryDwords`: a semicolon-separated list of `key=value`
pairs, and pairs that match none of the module's own options are passed on to the driver core. A second parameter,
`NVreg_RegistryDwordsPerDevice`, takes a `pci=` address first, so a key can be set for one card. Both are read
when the module loads: setting or removing one needs the module reloaded or a reboot, which on this rig is the
owner's, at the console. A parameter kept in a `modprobe.d` file survives every reboot until the file is removed.

**The PowerMizer keys are gone.** The string seen in forum posts since at least 2015 is

```
options nvidia NVreg_RegistryDwords="PowerMizerEnable=0x1;PerfLevelSrc=0x2222;PowerMizerLevel=0x3;PowerMizerDefault=0x3;PowerMizerDefaultAC=0x3"
```

- NVIDIA never documented these keys. The 580 README's list of X configuration options has no `RegistryDwords` and
  no PowerMizer entry (P).
- Reports (S): on a GTX 1080 Ti the keys worked as a module option on 520.56.06 and had no effect on 530.41.03; a
  second owner names 525.116.04 as the last driver on which they worked, and found no `PowerMizerEnable` in the
  unpacked 535.98 driver; NVIDIA filed bug 4282064 on 2023-09-11, and the thread shows no fix up to its last post
  on 2025-08-05.
- The public registry header at tag 580.178.04 defines 221 key names. None contains PowerMizer, a performance
  level, a P-state or a clock setting (P). That header belongs to the open modules, which drive Turing and newer
  cards, so it is not the full list of keys the closed driver core reads (claim to test: a search of the rig's own
  driver files for the names, read-only, row Q1533).
- Even where they worked, the keys chose a performance level. In 2017 the owner of two GTX 1080 Ti cards could move
  the idle level with them and could not move a CUDA load out of P2 (S).

**`OverrideMaxPerf` and `RMForcePstate`.** A 2025-07-25 reply, written as from inside NVIDIA, says
`OverrideMaxPerf` was unofficial and was removed, and that `RMForcePstate`, also unsupported, still exists:
`NVreg_RegistryDwords="RMForcePstate=8"` forces P8 (S). An owner's report of 2021-04-19 puts the removal of
`OverrideMaxPerf` between 460.67 and 465.24.02 (S). Forcing P0 on a P106 this way (claim to test, value 0) would
give what mode 1 gives, a card awake at the base clock, with a module reload to set it and another to undo it.
Neither name is in the public header.

## 3. `nvidia-smi` and NVML clock controls

| Control | The documentation says | On Pascal |
|---|---|---|
| `-lgc min,max` (`nvmlDeviceSetGpuLockedClocks`) | "Supported on Volta+"; lost at a reboot or driver reload; undone with `-rgc` | reports (S): refused with "Setting locked GPU clocks is not supported" on a GTX 1060 (465.24.02, 2021-04-18), a Quadro P1000 and a Quadro P2000 (460.89, 2021-01-15), and unsupported on a GTX 1080 Ti (537.13, 2024-01-19) |
| The same call with the two symbolic values | lock to the TDP clock, or keep it as a floor and let the card boost | the TDP clock is the base clock, so on a P106 neither form asks for more than it already gets (claim to test) |
| `-ac mem,core` (application clocks) | Maxwell GeForce, and Kepler and newer Tesla, Quadro and Titan; deprecated | N/A on the P106 (measured); refused on GTX 1080 and GTX 1070 cards (S) |
| Auto boost | Kepler-era Tesla and Maxwell GeForce; on Pascal and newer it is driven through application clocks | not available: no application clocks |
| `-cc 1` (CUDA clock override) | "Overrides or restores default CUDA clocks"; no list of cards; undone with `-cc 0` | the public source has a "Cuda limit" control, clocks limited for CUDA work; that `-cc` switches it is a claim to test. Owners of GeForce and Titan Pascal cards report compute held in P2, and one asked for this switch on a Titan Xp without an answer (S). A P106 decodes in P0 (measured), so est. nothing to lift (claim to test) |
| `nvmlDeviceSetGpcClkVfOffset` (core offset) | Maxwell and newer; root | moves every point of the curve, base included; it is an overclock with its own validation (registry Q135), not a hold |

`-lgc 1911,1911` is the one command that would be exactly the wanted setting. It has never been sent to a P106 on
this rig. It costs seconds, changes nothing if refused, and is undone with `-rgc` if accepted (Q1553).

## 4. The driver's own control calls

The public source of NVIDIA's kernel modules (tag 580.178.04, the rig's driver version) carries the control
interface that NVML and `nvidia-settings` sit on. The open modules themselves drive Turing and newer cards only;
the closed driver that runs a P106 is built from the same branch, and that it answers these calls for a Pascal card
is a claim to test.

**The P-state boost** (`NV2080_CTRL_CMD_PERF_BOOST`). It raises the P-state by one level or to the highest, for up
to 3,600 seconds or until cleared. It has a flag for CUDA clients, so est. it is the request a CUDA program already
makes (claim to test). It chooses a P-state, so it cannot lift a card that is already in P0.

**The rated-clock control** (`NV2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL`). This is where the driver's own words
for base and boost are:
- the base clock is the "rated TDP" point, and boost is anything above it;
- a client can ask to allow boost, to limit to the rated point, to lock the clocks at it, or to keep it as a floor;
- a second point type, turbo boost, exists; where profiling is restricted to administrators, the kernel side
  refuses it to other clients;
- the driver's own vote limits a card to the rated point "when no power controllers are active", and a status
  structure reports which client holds the card there.

NVIDIA's profiler, Nsight Compute, is the tool with a clock lock of this kind, as `--clock-control` (that it uses
this control is a claim to test). The option arrived in the version shipped with CUDA 10.2, the last one for
Pascal: support for the architecture was removed in 2020.1 (P). It locked to base by default, and a forum report
says that version locks a GTX 1080-class card to its base frequency (S) [8][9]. The lock to boost came in 2025.4,
"on Ampere and newer GPU", and became the default in 2026.1 (P). So NVIDIA's own tool does not offer a boost lock
for a Pascal card.

Trying it anyway means a small program that calls the driver's control device directly (est. 150 lines against the
MIT-licensed headers). Its read-only half would say whether the rated-point limiter is what holds a lightly loaded
P106 at exactly 1,354 MHz with no event reason set. Its other half would ask for the lock at the boost point on one
card. Both are undocumented calls into a closed driver; a card that hangs needs a reboot, and the disk passphrase
at the console. Est. the request is refused or ignored. Parked as Q1556.

## 5. What owners of Pascal cards report

| Date | Card, driver, system | What they tried | Result |
|---|---|---|---|
| 2015-04-27 | Quadro K4000 or GTX Titan, 340.29, no X server | `GpuPowerMizerMode=1`; `-ac`; the PowerMizer keys on the kernel command line | the mode needed an X server; `-ac` refused; the keys "had no effect" [1] |
| 2016-11-26 | three GTX 1080, 375.20, Ubuntu 16.04 | wanted a fixed clock to compare kernels; `-ac` | supported clocks N/A, `-ac` refused; a relative offset was the only thing accepted [2] |
| 2017-05-30 | two GTX 1080 Ti, 381.22, Ubuntu 16.04 | the PowerMizer keys in the X configuration | the idle level moved; a CUDA load stayed in P2 [3] |
| 2020-08-06 | Titan Xp | the registry keys suggested in the thread (`RMForcePstate`) | the default P-state became P0; CUDA work still ran in P2 [4] |
| 2021-01-15 | Quadro P1000, 460.89, Windows 10 | `-ac`, then `--lock-gpu-clocks` | `-ac` accepted without effect; the lock refused; the same refusal on a Quadro P2000 [5] |
| 2021-04-18 | GTX 1060 notebook, 465.24.02, Arch Linux | `OverrideMaxPerf`; `-lgc 139,500` | the key stopped working at this driver and worked again on 460.67; the lock refused [6] |
| 2023-03-27 to 2025-08-05 | GTX 1080 Ti, Arch Linux; a second owner on Gentoo | the PowerMizer keys as a module option | worked on 520.56.06 (the second owner: up to 525.116.04); no effect on 530.41.03; NVIDIA bug 4282064, no fix [7] |
| 2024-01-22 | GTX 1080 Ti, 537.13, Windows 10 | inference with gaps between bursts; `-lgc`, an overclocking tool | the clock fell in the gaps; no lock was available; a small kernel launched every 10 µs held it, and gaps above about 50 µs let it fall [8] |

A regular of NVIDIA's forum answered such threads the same way in 2021 and in 2024: the automatic clock management
has no user-settable controls, and clock locking belongs to other product lines [5][8]. The owner in the last row
ended where this project did, with a keep-busy kernel. Those two threads are about Windows; the rig's logs show the
same behaviour under Linux.

## 6. Why nothing holds it (my reading, a claim to test)

On Pascal the top clock is not a state the card can be put in. It is boost above the rated point, and the driver
grants boost to a card that is busy enough. Every user control sits one level up: it picks the P-state (routes 1,
3, 4, the P-state boost) or it takes boost away (mode 3, the lock to base). The controls that set a clock or lock
at boost (`-lgc` with numbers, the profiler's boost lock) are documented for Volta and Ampere and newer.

The measurements fit this: with no event reason set, a card that works one ninth of the time reads exactly 1,354
MHz, the card's base clock; a card that is kept busy reads 1,873–1,911 MHz; and mode 1 changes the idle power state
and nothing else.

## 7. The test for any route

One rig, idle and quiet, with the 100 ms clock log of
[2026-11-05_card_clocks](../../data/2026-11-05_card_clocks/README.md) (`run/clocks_run.sh`) and the bench line of
[2026-11-05_keepalive](../../data/2026-11-05_keepalive/README.md) with the keep-alive off.

1. **Accepted or refused**, on one idle card, seconds each:

   ```bash
   ssh rig 'nvidia-smi -i 0 -lgc 1911,1911; nvidia-smi -i 0 --query-gpu=clocks.sm,pstate,power.draw --format=csv; nvidia-smi -i 0 -rgc'
   ssh rig 'nvidia-smi -i 0 -cc 1; nvidia-smi -i 0 -cc 0'
   ssh rig "nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=3'; nvidia-settings -q '[gpu:0]/GPUPowerMizerMode'; nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=2'"
   ```

2. **For a control the card accepted:** three arms on the same day, two loads each, tg128 at one token a pass:
   the keep-alive off, the keep-alive on, and the control set on all nine cards with the keep-alive off. The
   control holds the clock if both are true:
   - every card reads 1,800 MHz or more in every sample of its timed windows;
   - its token time is no more than 1 ms above the keep-alive arm's. On 2026-11-05 the two loads with the keep-alive
     differed by 0.02 ms (51.06 and 51.08), the two without it by 3.13 ms (62.75 and 59.62), and the arms by 11.69
     and 8.54 ms, so a 1 ms band is wider than the keep-alive arm's error and far narrower than the effect.

   The token time is the judge. Driver 580 may report a clock from before idle slowdowns (registry Q1551).
3. **Its idle price:** watts a card with the control set and nothing loaded, against 5.7–6.0 W asleep and
   26.0–26.3 W awake at base.
4. **Only then the deep lane:** the preset with the route in place of its keep-alive against the preset as it is,
   two loads an arm, as R167 ran it.
5. **Undo** with the command beside each control, and read the setting back.

Step 1 changes run-time driver settings, so it needs the owner's word; it reloads nothing and leaves nothing
behind. Registry Q1553.

## What this changes in the plan

- **The keep-alive stays, and its cost is engine work.** The rows that cut it are unchanged: the pulse keep-alive
  (Q1065: 45.133, 45.129 and 45.128 ms a pass against the resident block's 45.879, 45.878 and 45.881) and the block
  paused during long prompt calls (Q1461, with R167c).
- **Q1553 (new, a rig, est. 10 min):** the three one-command checks above. Est. it closes the documented routes by
  measurement.
- **Q1554 (new):** mode 1 as the driver's own way to keep cards awake between agent turns, an arm beside the
  `idle` keep-alive of Q1045. It has no block for a request to wait behind; it holds the base clock, not boost.
- **Q1555 (new):** mode 3 for repeatable diagnostic timings, if Q1553 shows it pins the clock.
- **Q1556 (new, parked):** the rated-clock control asked directly.
- **Q1533** (the unlisted switches) gains four names to search the rig's driver files for: `PowerMizerEnable`,
  `PerfLevelSrc`, `OverrideMaxPerf`, `RMForcePstate`.

## Claims to test, in one list

1. `nvidia-smi -lgc` is refused on a P106-090 under 580.178.04 (Q1553).
2. `nvidia-smi -cc 1` is refused or changes no clock (Q1553).
3. The PowerMizer mode can be set through NVML with no X client on a P106, and is gone after a reboot (Q1553).
4. Mode 3 pins a P106 at 1,354 MHz under load (Q1553).
5. The PowerMizer key names are absent from the rig's 580.178.04 driver files (Q1533).
6. `RMForcePstate=0` holds a P106 in P0 at the base clock (not queued: it needs a module reload and would give what
   mode 1 gives).
7. The closed driver answers the rated-clock control for a Pascal card, and its limiter is what holds 1,354 MHz
   (Q1556).
8. A driver-held top clock, if any route gave one, would cost less than the keep-alive's +72 W for nine cards
   (step 3 of the test).
9. A P106 on a driver patched with NVIDIA-patcher can carry an X screen, and settings tied to one then apply (not
   queued: no route here needs an X screen on the cards).

## Already covered

| Finding | Where |
|---|---|
| Mode 1 measured: no speed gain, +20 W an idle card | [power and tuning](power_and_tuning.md) §3.1; [Q141's data](../../data/2026-10-24_rig_q141/README.md) |
| Application clocks N/A, locked clocks documented for Volta and newer, the 75 W limit | [hardware tuning](hardware_tuning.md) §3; registry Q154 |
| Compute forced to P2 on GeForce Pascal cards, application-clock refusals | [multi-GPU physics](multigpu_pascal_performance.md), Q8 |
| The core and memory offsets and their validation | [hardware tuning](hardware_tuning.md) §1, §5; registry Q133, Q135 |
| The keep-alive: what it holds, what it draws, where it costs | [keep-alive](../../data/2026-11-05_keepalive/README.md); [R165](../../data/2026-11-16_r165_flint_stack_split/README.md); [R167](../../data/2026-11-16_r167_keepalive_depth/README.md); [hardware and host review](v5_review_hardware_host_2026-11.md) |

## Sources

NVIDIA documentation and source (P)
- NVIDIA kernel modules, public source at tag 580.178.04, <https://github.com/NVIDIA/open-gpu-kernel-modules>:
  `kernel-open/nvidia/nv-reg.h` (`RegistryDwords`, `RegistryDwordsPerDevice`);
  `src/nvidia/interface/nvrm_registry.h` (221 key names); `src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080perf.h`
  (the P-state boost, the rated-clock control and its status structure);
  `src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080perf.h` (the CUDA limit);
  `src/nvidia/src/kernel/gpu/perf/kern_perf_pwr.c` and `kern_perf_boost.c` (the kernel side of both)
- `nvidia-settings` source at tag 580.95.05, <https://github.com/NVIDIA/nvidia-settings>:
  `src/libXNVCtrl/NVCtrl.h` (attribute 334 and its four modes), `src/libXNVCtrlAttributes/NvCtrlAttributes.c` and
  `NvCtrlAttributesNvml.c` (a GPU attribute is set through NVML first)
- NVML header `nvml.h`, API version 13, as carried by <https://github.com/NVIDIA/go-nvml> (read 2026-11-17):
  the PowerMizer mode calls, locked clocks and their two symbolic values, auto boost on Pascal
- `nvidia-smi` documentation, <https://docs.nvidia.com/deploy/nvidia-smi/index.html> (read 2026-11-17): `-ac`,
  `-lgc`, `-rgc`, `-cc`, auto boost
- NVIDIA Linux driver README 580.95.05, X configuration options,
  <https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/xconfigoptions.html>
- Nsight Compute release notes, <https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html> (read 2026-11-17):
  the boost lock in 2025.4, the default in 2026.1; the 2020.1 notes,
  <https://docs.nvidia.com/nsight-compute/2020.1/ReleaseNotes/index.html>: Pascal support removed

A tool named by the owner, not read for this note
- [dartraiden/NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher): read in
  [what a P106-090 holds back](p106_090_holds_back.md) §6

Reports (S), NVIDIA developer forum unless named
1. "GeForce/Quadro power management on a headless Linux machine (without X server)", 2015-04-27,
   <https://forums.developer.nvidia.com/t/geforce-quadro-power-management-on-a-headless-linux-machine-without-x-server/37761>
2. "1080FE powersave frequency scaling", 2016-11-26,
   <https://forums.developer.nvidia.com/t/1080fe-powersave-frequency-scaling/46401>
3. "How to force performance level 3 (driver 381.22, with GTX 1080 Ti)", 2017-05-30,
   <https://forums.developer.nvidia.com/t/how-to-force-performance-level-3-driver-381-22-with-gtx-1080-ti/50353>
4. "--cuda-clock nvidia-smi Support for Titan XP etc", 2020-07-04,
   <https://forums.developer.nvidia.com/t/cuda-clock-nvidia-smi-support-for-titan-xp-etc/140337>
5. "Very fast ramp-down from high to low clock speeds leading to increased time repeatedly ramping up", 2021-01-15,
   <https://forums.developer.nvidia.com/t/very-fast-ramp-down-from-high-to-low-clock-speeds-leading-to-increased-time-repeatedly-ramping-up/165974>
6. "OverrideMaxPerf and PowerMizerEnable tricks don't work on 465.24.02 driver", 2021-04-16, with the reply of
   2025-07-25, <https://forums.developer.nvidia.com/t/overridemaxperf-and-powermizerenable-tricks-dont-work-on-465-24-02-driver/175494>
7. "Kernel module option NVreg_RegistryDwords for PowerMizerEnable doesnt work on 530.41.03", 2023-03-27,
   <https://forums.developer.nvidia.com/t/kernel-module-option-nvreg-registrydwords-for-powermizerenable-doesnt-work-on-530-41-03/247610>
8. "Stability Issues with GPU Inference on Older GPUs (e.g., 1080Ti)", 2024-01-19,
   <https://forums.developer.nvidia.com/t/stability-issues-with-gpu-inference-on-older-gpus-e-g-1080ti/279474>
9. "Nsight Compute Clock Speed During Profiling", 2022-03-16 (clocks locked to base by default),
   <https://forums.developer.nvidia.com/t/nsight-compute-clock-speed-during-profiling/208646>
