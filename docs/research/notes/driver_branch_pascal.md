# Is NVIDIA driver branch 580 the right one for a P106-090's compute work? (2026-11-17)

**Verdict.** Branch 580 is confirmed: it is the only branch that supports Pascal and still gets fixes, and I found no
measurement and no report of an older branch running CUDA work faster on a Pascal card. No older branch is worth a
test: what is published points the other way (newer drivers hand a CUDA graph to the card with less host work), and
the test would take two package changes and two reboots at the console and leave the rig, for its length, on a branch
with a known security hole that gets no fix. What the reading did turn up sits inside 580: a newer point release with
a fix for `nvidia-smi` polling, a change in what `nvidia-smi`'s clock means, and one interrupt setting that differs
from branch 570, each now a registry row (Q1550 to Q1552).

The question ([registry](../../../quants/experiments.md) Q1532): one decode step of Miner Llama is thousands of small
kernel launches driven by a 2-core Celeron ([hardware spec](../../mining_rig.md)), so the driver's cost per launch, how
it waits and its clock policy all count. Eight host-side switches already took a Flash-Next step from 119.15 to
111.94 ms ([step review](v5_review_flash_step_2026-11.md)). The tested configuration is Ubuntu 24.04 with a 7.0
kernel, driver 580.178.04 and CUDA 12.9.2.

Conventions:
- "est." marks arithmetic or inference of this note, with its basis beside it.
- A **report** is one owner's or one site's statement, not checked by me. Each has its date and link.
- Nothing ran on any machine for this note, and no source measured a P106. Every statement about what a branch would
  do on the rig is therefore a **claim to test**, and the note says so where it matters.
- "Userspace driver" is `libcuda.so` and the libraries beside it, which come with the driver package and turn CUDA
  calls into commands for the card. The CUDA 12.9 runtime the engine is built with (`libcudart`) sits on top of it.
- Sources were read on 2026-11-17.

## 1. Which branches run Pascal with the CUDA 12 runtime, and until when each gets fixes

A program built with any CUDA 12.x toolkit needs a driver of 525.60.13 or newer on Linux (CUDA 12.9 Update 1 release
notes, the table of minimum drivers for minor-version compatibility). Branch 580 is the last that supports Pascal:
"The release 580 series will be the last to support GPUs based on the Maxwell, Pascal, and Volta architectures"
(NVIDIA's deprecation schedule, updated 2025-07-01). So the candidates are the branches from 525 to 580.

| Branch | The CUDA toolkit released with it (first driver) | Last release in NVIDIA's archive | Fixes |
|---|---|---|---|
| 525, 530 | 12.0 (525.60.13), 12.1 (530.30.02) | not looked up | ended |
| 535 | 12.2 (535.54.03) | 535.309.01, 2026-04-28 | ended June 2026 (it was a long-term branch) |
| 545 | 12.3 (545.23.06) | not looked up | ended |
| 550 | 12.4 (550.54.14) | 550.163.01 | ended |
| 555, 560 | 12.5 (555.42.02), 12.6 (560.28.03) | not looked up | ended |
| 565 | none in the release notes' table | 565.77 | ended |
| 570 | 12.8 (570.26) | 570.211.01, 2026-01-13 | ended; Ubuntu called it end-of-life on 2026-04-01 |
| 575 | 12.9 (575.51.03) | 575.64.05 | ended |
| **580** | 13.0 | **580.196.07, 2026-10-21** | **until June 2028** (a long-term branch) |
| 590 and newer | 13.x | | no Pascal |

- **Kinds and their length** (NVIDIA's driver lifecycle page, updated 2026-11-04): a production branch gets bug fixes
  and security updates "for up to 1 year", a long-term branch for 3 years, a new-feature branch for no stated time.
- **Which branches are alive** (NVIDIA's table of supported drivers, updated 2026-11-04): only R580 (long-term, end of
  life June 2028), R595 (production, March 2027) and R620 (long-term, September 2029). Every branch older than 580 is
  off the table, which is what "ended" means above. June 2026 for 535 is from NVIDIA's data-center driver guide, read
  through a search excerpt.
- **The P106-090 itself** is in 580.196.07's list of supported products ("NVIDIA P106-090", device 1C09), among the
  current cards, not the legacy ones.
- **For GeForce cards** the press reports security-only updates for Maxwell, Pascal and Volta every quarter until
  October 2028 (reports, 2025; I did not find NVIDIA's own page for that date).
- **What "runs CUDA 12" means on an older branch.** CUDA 12.9 was released with driver 575.51.03 (its updates with
  575.57.08); on anything older a 12.9 program runs in compatibility mode, without whatever its calls need from a
  newer driver. Which of the engine's calls that would hit on 535, 550 or 570 is not checked. One planned path is
  known to need more than 535: conditional nodes in CUDA graphs are tested for with a driver of CUDA 12.3 or newer
  ([host note](host_decode_loop.md), section 2.3).

Ubuntu 24.04 follows NVIDIA. Its `nvidia-graphics-drivers-570` source package carries "Transition EOL 570 drivers to
580" since 570.211.01-0ubuntu1.24.04.1 (published 2026-04-27, Launchpad bug 2147038). Its 535 package last took a new
upstream release on 2026-05-19, and in August 2026 an owner found that an update had removed 535 and installed 580 (a
report, Ubuntu Discourse, 2026-08-04). A site's check of 2026-11-12 still lists 550, 570 and 575 as real packages on
24.04 (a report), so what `apt install nvidia-driver-570` gives on a given day is not verified. On a rig set up by
[the checklist](../../setup.md), an older branch may no longer be one `apt install` away.

## 2. What changed between the branches that touches compute on Pascal

### 2.1 Launch and submission cost: newer is cheaper, and on Pascal less so

Two sources compare driver generations on the cost of launching a CUDA graph, the shape of a decode step here. Both
measured Ampere cards.

**NVIDIA's CUDA driver team** (Hoffman and Oh, 2024-09-11) compared toolkit 11.8 with driver 520.61.05 against toolkit
12.6 with driver 560.28.03, on a Xeon Silver 4208 at 2.10 GHz and an RTX 3060, for a graph that is one chain of
kernels:

| Per graph | 11.8, driver 520 | 12.6, driver 560 |
|---|---|---|
| Building the graph, 100 nodes | 168 µs | 127 µs |
| Building the graph, 1,025 nodes | 2,143 µs | 1,526 µs |
| First launch, host time, 100 nodes | 25 µs | 15 µs |
| First launch, host time, 1,025 nodes | 278 µs | 175 µs |
| Repeat launch, host time | 2 µs + 200 ns a node | ~2.5 µs + ~1 ns a node (Ampere) |
| Time on the card, 100 nodes | 61 µs | 53 µs |
| Time on the card, 1,025 nodes | 629 µs | 567 µs |

- **The line that matters for a P106:** "Launch still takes O(n) time on pre-NVIDIA Ampere architecture hardware." The
  near-constant repeat launch is for Ampere and newer. On Pascal every node of a graph costs host time at every
  launch, on every branch.
- **The other gains are not split by card generation.** Building a graph, the first launch and "inter-kernel latency"
  on the card (up to 60 ns a node) all improved, and the post does not say on which cards. It also does not say which
  release brought which gain, only that all were there by driver 560. Branches 535 and 550 lie in between.
- **Nothing in it got slower.** The post reports no regression.

**A study of the driver's command stream** (Yan, Karlin and Grant, arXiv 2604.26889, 2026-04-29) traced what the
userspace driver writes to an A40, with toolkit 11.8 on userspace driver 520.61.07 against toolkit 13.0 on
580.105.08, a Xeon Gold 6338, for chains of 1 to 2,000 identical kernels:

| A chain of 2,000 kernels, one launch | 11.8, driver 520 | 13.0, driver 580 |
|---|---|---|
| Launch time | 209 µs | 5.9 µs |
| Command bytes written | 45,476 | 2,216 |
| Doorbell writes | several, growing with the chain | one |

At one kernel the two are equal (1.8 and 1.9 µs; 328 and 340 bytes). This is the Ampere path again, and the method
needs NVIDIA's open kernel modules, which do not support Pascal, so it cannot be repeated on a P106.

**What this means for the rig (est.).**
- The direction is one way: from 520 to 580 the userspace driver does less host work per graph. I found no source in
  which an older branch launches faster.
- How much of it a P106 gets is unknown, since both sources measured Ampere. The claim to test would be "580 is no
  slower than 570 at launching a P106's graphs", and the sources give no reason to doubt it.
- The prize is bounded. A Flash-Next token is ~2,520 launch groups and ~2,900 kernels over nine cards, and removing
  every launch at no cost would save 6.7 ms of a token, about 6 % ([megakernel note](megakernel.md)). A branch change
  could move only a part of that, in either direction. A fifth of it would be est. 1.3 ms, est. 1.2 % of a token; I
  have no evidence for even that.

Two owners' reports go the other way, both on Ampere cards and neither with versions:
- An A100 owner measured about 1 µs between two dependent kernels before "upgrading driver and CUDA toolkit" and
  2.6 µs after (NVIDIA forum, 2024-10-11). NVIDIA's reply questions the measurement (event timing includes other
  work) and advises a microbenchmark; the thread ends without one.
- An RTX 3090 owner on driver 545.23.08 found short runs "about 1/3" slower than on an earlier driver they could not
  name (NVIDIA forum, 2024-10-22); the one reply suspects clock-down while idle.

### 2.2 How the host waits, and interrupts

- **The wait policy is the program's, not the branch's.** Whether a waiting thread spins, yields or sleeps is set with
  `cudaSetDeviceFlags` ([host note](host_decode_loop.md), section 2.4). I found no release note from 535 to 580 that
  changes it.
- **One interrupt setting differs between 570 and 580.** `RMIntrLockingMode` came with branch 570 as an experimental
  option, off by default (a report of 570.124.04's release highlights, 2025-02), and is on by default from 580.65.06
  (a report of its highlights, 2025-08-04). It lets display interrupts be served while the driver is busy, and the
  highlights name stutter in VR. `NVreg_RegistryDwords=RMIntrLockingMode=0` as a module parameter turns it off. Whether
  it costs or saves anything for CUDA work is not stated anywhere. Est. nothing on decode: CUDA's default wait polls,
  so few interrupts should reach the host in a token. It is the one kernel-side difference I found that is not display
  work alone, and it can be tried inside 580 (Q1552, parked behind Q1551's interrupt count).

### 2.3 The launch queue: a fault fixed in the release the rig runs

580.178.04 (2026-08-03) lists: "Fixed a bug that could cause Xid 32 errors and application crashes when scaling via
CUDA_SCALE_LAUNCH_QUEUES" at 2x or 4x, seen with CUDA graphs that have many chained kernel nodes (NVIDIA's release
notes for that version).

- A decode step here is exactly such graphs. The rig leaves the queue at its default, because 4x costs about 120 MB
  of host memory a card ([host memory note](host_ram_4gb.md)), and Q693 tests 0.5x and 0.25x; the fix names 2x and 4x
  only.
- The tested configuration is the first release with this fix. Branch 570's last release is from 2026-01-13, almost
  seven months earlier, so est. no older branch has it. A larger queue on fewer cards, or on a host with more memory,
  would be unsafe there.

### 2.4 Clocks and power state

- **What `nvidia-smi` reports changed in 580.** From 580.65.06, clock values in nvidia-settings, NVML and `nvidia-smi`
  show clocks "before thermal and idle slowdowns", to match Windows (a report of its highlights, 2025-08-04). The
  drivers' log here reads `clocks.sm` and the active throttle reasons (Q1464). If the report is right, that clock
  cannot show a card that clocked down while idle or hot, and the throttle reasons are the only sign. What the field
  shows on a P106 is a one-minute read-only check (Q1551).
- **The clock policy itself:** I found no release note from 535 to 580 that changes how a Pascal card picks its clock
  under CUDA work, and no owner's report of it. Holding the clocks is its own question (Q1530).

### 2.5 Host memory

- No release note and no report gives host memory per card by branch. The sizes of the loaded kernel modules by branch
  are not published either; Q1551 reads them for 580.
- **One change may matter on a full card.** 580.159.03 (2026-04-29) has "Improved support for falling back to system
  memory when available video memory is low, to help prevent Wayland desktop freezes" (a report of its highlights).
  A rig's cards are filled close to their 6 GB with weights, and a desktop session keeps a 4 MiB context on each.
  Whether that session's allocations move to the 4 GB of host memory when a model is loaded is a claim to test
  (Q1551); it argues for [a rig without the desktop](host_ram_4gb.md), not for another branch.

### 2.6 Stability, security and the kernel

- **580.196.07 (2026-10-21), newer than the tested 580.178.04,** lists among its fixes (NVIDIA's release notes): NVML
  and `nvidia-smi` called a library for extended PCI fields of each card, which "under unique conditions" could "result
  in GPU fallen off bus errors in low frequency", so the calls were removed from paths queried often; and "GPU
  utilization reported 100% following SIGINT-terminated CUDA workloads". Neither is stated for Pascal. Every timed run
  here polls `nvidia-smi`, and a card off the bus means a reboot at the console, so the release is worth taking at the
  owner's next reboot (Q1550).
- **Kernel 7.0.** The same release fixes wake from hibernation and a black console after a switch on kernel 7.0 and
  newer (a report, 2026-10). Branch 570's last release predates kernel 7.0. Ubuntu's archive lists 570 module packages
  built for one of its 7.0 kernels, so Ubuntu's patched 570 builds there; NVIDIA's own installer for 570 or 535 on a
  7.0 kernel is not verified.
- **Security.** NVIDIA's bulletin of 2026-09-30 covers CVE-2026-47591, a write to read-only memory by an unprivileged
  local user in the driver's kernel layer. The fixed release for R580 is 580.178.04, the tested one; for R570, R575
  and older, "no fix listed" (a report, 2026-11-12). In May 2026 the 570 series was already left out of a bulletin as
  end-of-life (a report).

## 3. What owners report after changing branch

| Date | Where, card | Branches | What is reported | Bears on the rig? |
|---|---|---|---|---|
| 2026-11-16 | llama.cpp issue 32375, Tesla P100 (Pascal, compute 6.0) | 550.163.01 and 580.178.04 | a long-context fault of hybrid models gives the same wrong output "byte-for-byte" on both; no speeds | yes: one owner's evidence that results do not depend on the branch |
| 2026-02-19, 2026-03-14 | NVIDIA forum, two GTX 1080 owners, kernels 6.18 and 6.17 | 580.119.02 and 580.126.09; the second owner says 535 and 570 "were stable on previous installs" | the card falls off the bus (Xid 79) under light desktop use; no reply from NVIDIA, no fix confirmed | weakly: a display on the card, a desktop compositor, and no side-by-side test |
| 2026-08-04 | Ubuntu Discourse, GeForce 840M laptop (Maxwell) | 535 to 580 by Ubuntu's update, then back to 535 | games jitter on battery under 580; 535 restored it | no: graphics and battery policy on a hybrid laptop |
| 2024-10-11, 2024-10-22 | NVIDIA forum, A100 and RTX 3090 | not named | launches slower after an upgrade (section 2.1) | no: Ampere, versions unknown |

- **Not found:** any Pascal owner's speed before and after a branch change, for any CUDA program. The llama.cpp
  threads by Pascal owners that I read name one driver or none: eight Tesla P40s with no driver stated (2026-02-01),
  and the P102-100, P40 and P100 figures in [upstream Pascal patches](upstream_pascal_patches.md) and
  [the llama.cpp family](engines_llamacpp_family.md) on 550 or on 580.
- **The pattern in what exists:** complaints about 580 on old cards are about displays, desktops and battery. The rig
  uses none of them for its speed.

## 4. Verdict, and what a test would have taken

**580 is confirmed.** In order of weight:
1. It is the only branch with Pascal support that gets fixes (to June 2028), and the tested 580.178.04 holds two that
   concern this rig: the launch-queue fault with long graph chains and CVE-2026-47591.
2. The two published comparisons of driver generations show less host work per graph launch in the newer one, and no
   source shows an older branch ahead on any card.
3. The most a branch could move is a part of est. 6.7 ms a token, and nothing says which way.
4. One Pascal owner ran 550 and 580 and got identical output.

**The test I do not recommend, for the record.** The branch would have been 570 (570.211.01): the newest older line
with a long life, a CUDA 12.8 userspace driver, and builds for a 7.0 kernel in Ubuntu's archive. Not 535 or 550, whose
userspace predates driver 560, the point by which NVIDIA's post has all its graph gains.
- **What it would compare,** same binary, file and day, two loads an arm: Flash-Next's decode step against the
  step review's 111.94 ms; one agent check a branch; the host's memory floor from
  [`rig_host_ram.sh census`](../../../scripts/rig_host_ram.sh); clocks and throttle reasons of busy and idle cards.
- **What it would take,** all at the console and all the owner's: the 570 packages, if Ubuntu still carries them as
  570, or NVIDIA's installer in place of 580, a reboot, the runs, then 580 back and a second reboot.
- **What it would risk:** a rig that runs for that time on a branch with a known hole and no fix, and a module build
  that may fail on the 7.0 kernel.
- **What would reopen it:** a published before-and-after on a Pascal card that shows an older branch ahead in CUDA
  work, or a 580 release that breaks the rig.

**What to do instead, inside 580:**

| Row | What | Cost | Expected |
|---|---|---|---|
| Q1551 | read-only: what `nvidia-smi`'s clock shows on an idle and a busy card; the kernel modules' size; the desktop's memory with cards empty and full; GPU interrupts a token | one rig, est. 10 min | says what the clock log can see; prices Q1552 |
| Q1550 | 580.196.07 at the owner's next reboot, then the release check | the owner at the console; est. 30 min of rig time | est. no speed change; one less way to lose a card |
| Q1552 | `RMIntrLockingMode=0`, the one interrupt difference between 570 and 580 | a module reload (the owner's), two loads an arm | est. none; parked behind Q1551 |

## 5. New registry rows

- **Q1550:** the newest point release inside branch 580 (the owner's decision).
- **Q1551:** what `nvidia-smi`'s clock shows on driver 580, with three more read-only numbers.
- **Q1552:** parked, the driver's interrupt locking mode off.

## Not verified

- Anything on a P106: no source measured one, and nothing ran for this note.
- NVIDIA's own changelog for the 580 display driver. It ships inside the installer and is not on NVIDIA's download
  site as a page; the release highlights here are quoted from news sites' copies.
- Whether the gains in NVIDIA's 2024 post other than the near-constant repeat launch apply to Pascal.
- Which CUDA calls of the engine need a driver newer than 535, 550 or 570.
- Whether branch 570 or 535 lacks the launch-queue fix (est. from dates), and whether the fault exists below 2x.
- That NVIDIA's installer for 570.211.01 or 535.309.01 builds on a 7.0 kernel, and whether Ubuntu 24.04's 570
  packages still install 570 (its changelog and a site's check disagree).
- The October 2028 date for security updates on GeForce cards (press reports), and June 2026 for 535 (a search
  excerpt of NVIDIA's guide).
- The exact wording of 570.124.04's and 580.196.07's display-driver highlights (search excerpts of news pages).
- That the `nvidia-smi` fix in 580.196.07 concerns the query the drivers here use (`--query-gpu` in a loop); NVIDIA
  names `nvidia-smi -q`.

## Sources

NVIDIA (primary)
- Supported drivers and CUDA toolkit versions, updated 2026-11-04,
  <https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html>; driver
  lifecycle, updated 2026-11-04, <https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html>; data-center
  driver guide (535's end of life; a search excerpt), <https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Datacenter_Drivers.pdf>
- UNIX graphics feature deprecation schedule (the 580 sentence, added 2025-07-01),
  <https://forums.developer.nvidia.com/t/unix-graphics-feature-deprecation-schedule/60588>
- CUDA Toolkit 12.9 Update 1 release notes (minimum and released drivers per toolkit; Pascal "feature-complete"),
  <https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html>
- The driver archive's list of releases per branch (its `latest.txt` dated 2026-11-03),
  <https://download.nvidia.com/XFree86/Linux-x86_64/>; supported products of 580.196.07,
  <https://download.nvidia.com/XFree86/Linux-x86_64/580.196.07/README/supportedchips.html>
- Data Center GPU driver release notes: 580.196.07 (2026-10-21),
  <https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v13.0.pdf>;
  580.178.04 (2026-08-03), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-178-04/index.html>;
  580.173.02 (2026-06-29), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-173-02/index.html>;
  580.159.04 (2026-05-14), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-04/index.html>;
  580.126.16 (2026-02-09), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-16/index.html>;
  570.211.01 (2026-01-13), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-211-01/index.html>;
  535.309.01 (2026-04-28; a search excerpt), <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-309-01/index.html>
- H. Hoffman and F. Oh, "Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements",
  2024-09-11, <https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements>

Research (primary)
- Y. Yan, I. Karlin and R. Grant, "Revealing NVIDIA Closed-Source Driver Command Streams for CPU-GPU Runtime Behavior
  Insight", arXiv 2604.26889, 2026-04-29, <https://arxiv.org/abs/2604.26889>

Distribution packaging
- Ubuntu 24.04, `nvidia-graphics-drivers-570` changelog, <https://ubuntuupdates.org/package/core/noble/restricted/updates/nvidia-graphics-drivers-570>;
  Launchpad bug 2147038, "Transition EOL 570 drivers to 580", 2026-04-01,
  <https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-570/+bug/2147038>; `nvidia-graphics-drivers-535`
  changelog, <https://ubuntuupdates.org/package/core/noble/restricted/updates/nvidia-graphics-drivers-535>; 570 module
  packages for a 7.0 kernel, <https://www.ubuntuupdates.org/package/core/noble/restricted/updates/linux-modules-nvidia-570-open-nvidia-7.0>
  (a search excerpt)

Reports
- llama.cpp issue 32375, 2026-11-16 (Tesla P100, drivers 550.163.01 and 580.178.04),
  <https://github.com/ggml-org/llama.cpp/issues/32375>; discussion 19248, 2026-02-01 (eight Tesla P40s),
  <https://github.com/ggml-org/llama.cpp/discussions/19248>
- NVIDIA forum: "Xid 79, 154: GPU has fallen off the bus - GTX 1080 + Driver 580.119.02", 2026-02-19,
  <https://forums.developer.nvidia.com/t/xid-79-154-gpu-has-fallen-off-the-bus-gtx-1080-driver-580-119-02-kde-wayland/361007>;
  "Successive kernels switch latency", 2024-10-11,
  <https://forums.developer.nvidia.com/t/kernel-switch-latency-successive-kernels-switch-latency/309504>; "What could
  be possible reasons for affecting the kernel launch overhead for fast small kernels?", 2024-10-22,
  <https://forums.developer.nvidia.com/t/what-could-be-possible-reasons-for-affecting-the-kernel-launch-overhead-for-fast-small-kernels/310718>
- Ubuntu Discourse, "Update Removed Preferred NVidia 535 Driver", 2026-08-04,
  <https://discourse.ubuntu.com/t/update-removed-preferred-nvidia-535-driver/85877>
- GamingOnLinux: 580.65.06's release highlights, 2025-08-04,
  <https://www.gamingonlinux.com/2025/08/nvidia-beta-driver-580-65-06-released-for-linux-with-bug-fixes-and-expanded-wayland-support>;
  580.159.03's, 2026-04-29,
  <https://www.gamingonlinux.com/2026/04/nvidia-580-159-03-driver-released-for-linux-with-some-essential-fixes/>;
  570.124.04's, 2025-02 (a search excerpt),
  <https://gamingonlinux.com/2025/02/nvidia-stable-driver-570-124-04-released-for-linux-with-vrr-for-multiple-displays-and-various-bug-fixes>;
  the May 2026 security bulletin and 570's end of life (a search excerpt),
  <https://www.gamingonlinux.com/2026/05/nvidia-reveal-more-gpu-driver-security-flaws-for-may-2026/>
- Comss, 580.196.07's fixes for kernel 7.0, 2026-10 (a search excerpt), <https://www.comss.ru/page.php?id=22530>
- ComputingForGeeks, CVE-2026-47591 and the fixed release per branch, updated 2026-11-12,
  <https://computingforgeeks.com/nvidia-nvroot-forgeptr-linux-driver-fix>
- TechRadar, security updates for Maxwell, Pascal and Volta until October 2028 (a search excerpt),
  <https://www.techradar.com/pro/nvidia-drops-linux-590-driver-support-for-gtx-900-and-10-series-signaling-the-end-of-game-ready-updates-for-older-gpus>;
  Phoronix, "NVIDIA Confirms 580 Linux Driver Is The Last For Maxwell / Pascal / Volta",
  <https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW>

This project
- [Quant registry](../../../quants/experiments.md) (Q693, Q1464, Q1530, Q1532, Q1550 to Q1552), the
  [hardware spec](../../mining_rig.md), [setup](../../setup.md), the [host note](host_decode_loop.md), the
  [hardware tuning note](hardware_tuning.md), the [megakernel note](megakernel.md), the
  [host memory note](host_ram_4gb.md) and the [step review](v5_review_flash_step_2026-11.md).
