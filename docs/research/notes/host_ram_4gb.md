# Host RAM on a 4 GB nine-GPU host: where it goes and what gives it back without new parts (2026-11-04)

Scope: the rigs run as sold: one 4 GB DDR4 SO-DIMM (the board takes a single stick), a Celeron 3865U, nine P106-090
cards, no new parts ([hardware spec](../../mining_rig.md); a bigger module is parked, Q86 and Q339). Host RAM is what keeps
the MTP drafter and larger drafts from loading ([Q226 and Q558](../../data/2026-11-03_flash_next_mtp/README.md)). This note
asks how much *usable* host RAM software can give back, from primary sources: what a CUDA context costs on the host and what
shrinks it, what llama.cpp pins and the alternatives, Linux memory policy and boot parameters, and llama.cpp's own host
buffers. It starts from the rig's measurements and from [the draft-memory switches](../../data/2026-11-04_draft_mem/README.md)
and does not repeat what they settle ("Already covered"). Nothing ran on a rig for this note: every rig-side check below is a
read, an environment variable or a setting that reverts at once, unless the text says it needs a reboot.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (vendor and kernel documentation, kernel and llama.cpp source and pull
  requests, measurements in this repository) and secondary (forum and issue reports).
- "Usable" memory is what pinned buffers and the kernel's own allocations can take: MemAvailable minus CmaFree (§3.1).
- Registry ids are rows of [quants/experiments.md](../../../quants/experiments.md). Risk labels in the closing table: **R0**
  an environment variable or a read, undone at once; **R1** a runtime setting or flag, undone at once; **R2** a reboot at
  the console (the disk passphrase) or engine code that has not run on a rig; **R3** firmware.

Starting point, measured (rig 7.0 kernel, driver 580, CUDA 12.9, headless, debloated):
- **An agent session on Flash-Next Q2_0** (nine cards, three shards, the 28.8 GB n-gram table on the SSD) leaves 140–460 MiB
  of MemAvailable and 0.3–1.5 GB of swap in use; minus CmaFree, 154–205 MiB at the median
  ([session data](../../data/2026-11-03_flash_next_mtp/session_memory.csv)).
- **Pinned buffers hold ~735 MiB**: the main model's 379 MiB compute buffer at `-ub 512`, its 260 MiB Q3_K token embedding
  (unless `--no-host`), the rest not itemised (Q549). The direct-I/O loader stages through 4 × 64 MiB more during a load
  (Q580). Four context checkpoints hold 4 × 113 MiB of ordinary memory.
- **The firmware holds ~73 MiB** (~45 MiB of it the integrated graphics' block); Linux sees MemTotal 3,837 MiB of 4,096
  (Q570).
- **Going headless** returned ~280 MB of MemAvailable (Q83), but only ~76 MiB at the lowest point of a load, because most of
  it was CMA-marked pages ([Q558 data](../../data/2026-11-03_flash_next_mtp/README.md)).
- **The host cost of each of the nine CUDA contexts is not measured** (Q565).

## Summary

- **The largest block is a boot setting: 970 MiB of the 3,837 MiB are marked CMA with no CMA area, which is what Kexec
  HandOver (KHO) scratch looks like.** Ubuntu's 7.0 kernels build with KHO on by default (reported on 7.0.0-31 and 7.0.0-38). KHO reserves scratch
  regions at boot, by default twice what the kernel had reserved by then, and releases them to the allocator as CMA
  pageblocks, so they appear in `CmaFree` and in `/proc/pagetypeinfo` but not in `CmaTotal`. Only movable pages may live
  there; long-term pinned buffers (CUDA's) and the kernel's unmovable allocations cannot. `kho=off` on the kernel command
  line removes the reservation (one reporter: `CmaFree` 0 kB afterwards, pinned allocations no longer stalling). Est. **+410–970
  MiB usable** for one reboot, no run-time cost, and nothing lost here, since the rigs use neither kexec nor kdump. This
  is the likely answer to Q560; a read of the rig's kernel config and `kho` debugfs confirms it (§3.1).
- **CUDA's documented per-context knobs free almost nothing, except one that has a published price: the launch queue.**
  Lazy module loading is already the default on CUDA 12.9 with driver 580; `cudaDeviceSetLimit`'s limits are device-side; the
  driver's module parameters hold nothing for per-GPU host memory; persistence mode changes nothing while a process runs.
  `CUDA_SCALE_LAUNCH_QUEUES=4x` costs ~120 MB of system RAM per GPU (llama.cpp PR #19042), so the default queue is est. ~40
  MB per GPU, ~360 MiB over nine contexts, and `0.25x` would return est. up to ~270 MiB at a prompt-speed cost nobody has
  measured on x1 links (§1).
- **Pinned memory is no speed problem over PCIe 1.0 x1 and a stability problem.** The link moves ~190 MB/s a card; staging
  a pageable copy through the driver costs est. ≤ 10 % more CPU on bulk copies and nothing measurable on per-token inputs.
  What unpinning loses is asynchrony, which the fork's packed uploads use. Pinned pages are the ones the kernel can neither
  swap nor move into CMA pageblocks, so the lever is fewer pinned bytes (the embedding on a card, shared draft tensors,
  smaller staging and host scratch), not `cudaHostRegister` (it pins the same pages) and not `GGML_CUDA_NO_PINNED` as a
  default (§2).
- **zram and zswap do not free RAM here; they turn RAM into swap-in speed.** What is swapped is llama-server's float state
  (est. 1.1–1.4× with lz4) and cold daemon pages. SSD swap holds 0 MiB of RAM, a compressed pool holds up to its size.
  A 5 % zswap pool would cut a swap-in from ~76 µs a page (a checkpoint restore: 2.8 s against 0.63 s) to est. ≤ 10 µs, for
  ~190 MiB; worth it only if the rig still swaps after the first item (§3.2).
- **What stops a stall is policy, not size.** MGLRU's `min_ttl_ms` turns a multi-minute thrash into an OOM kill of the process
  the guard kills anyway; `memory.high` on the daemons' slices and `memory.low` on llama-server push cold daemon pages to
  swap first (est. 50–150 MiB resident); `systemd-oomd` is a second killer to check for (§3.3).
- **Smaller items**, each under ~60 MiB: a 64 MiB swiotlb pool (est., if present), services, modules and the journal
  (Q571), THP, the console and DRM modules, the BIOS graphics size (Q570, parked) (§3.5).
- **llama.cpp**: the host buffers that can move to the cards are the token embedding (260 MiB, Q563), the checkpoints (up to
  452 MiB for est. 38–64 MiB of VRAM per card, Q440) and the draft's tensors (Q564). `-lm mlock` would try to lock the
  CPU-resident tensors, which on Flash-Next likely include the 28.8 GB table: never on this model. The table's page cache is
  not worth protecting: beyond 512 MiB it saves ≤ 0.5 read per token (Q228) (§3.4, §4).
- The ranked list, by MiB per unit of risk, closes the note.

## Where the 4,096 MiB goes

| Block | MiB | Basis |
|---|---:|---|
| DRAM module | 4,096 | |
| Held below the kernel by the firmware (reserved and ACPI ranges, ~45 of it the graphics block) | ~73 | Q570, `/proc/iomem` |
| The kernel's early allocations: memory map (64 B per 4 KiB page, est. 64 MiB), image and tables, and a swiotlb pool if allocated (§3.5) | ~186 | est.: 259 MiB gap to MemTotal minus the firmware's 73 |
| **MemTotal** | **3,837** | `/proc/meminfo` |
| of it KHO scratch, CMA-marked, movable pages only | 970 | 485 pageblocks of 2 MiB, [pageblocks.csv](../../data/2026-11-03_flash_next_mtp/pageblocks.csv) |
| **Usable by pinned buffers and kernel allocations** | **~2,867** | 3,837 − 970 |
| Pinned (Shmem): compute buffer, token embedding, rest | ~735 | Q549, Q558 |
| Context checkpoints (ordinary memory, swappable) | 452 | 4 × 112.6 MiB, Q215 |
| Nine CUDA contexts: driver, queues, UVM, pools | not measured | Q565; est. 360–630 at the default queue (§1.2) |
| Daemons, slab, page tables, page cache | not measured | Q571 for the daemons |

At the low point of an agent session MemAvailable (140–460 MiB) therefore describes a host whose pinned and kernel memory
has 2.87 GiB to live in, not 3.8: the allocator takes CMA pageblocks for movable pages only once their free share passes
half of the zone's free memory (est.: the allocator's CMA balance), so most of the CMA quarter sits idle, which is why both
of Q226's draft-line OOMs found 955–967 MiB of CMA pages free.

## 1. CUDA contexts and the NVIDIA driver (question 1)

### 1.1 The knobs, one by one

| Knob | Effect on host RAM | Source | On the rig |
|---|---|---|---|
| `CUDA_MODULE_LOADING=LAZY` | NVIDIA's 12.2.1 guide: "This reduces memory overhead, both on GPU memory and host memory." Default on Linux since CUDA 12.2 (all platforms 12.3), driver ≥ 515; `CUDA_MODULE_DATA_LOADING` also defaults to LAZY. Libraries built before 11.7 and modules with managed variables still load eagerly | [lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html), [12.2.1 guide](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-programming-guide/lazy-loading.html), [environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) | Already on (CUDA 12.9, driver 580). Nothing to gain unless a service unit sets `EAGER`: read `/proc/<pid>/environ`. The fork's fat binary is sm_61 machine code, so no JIT and `CUDA_CACHE_*` (an on-disk cache) do not matter |
| `CUDA_FORCE_PRELOAD_LIBRARIES` | Default 0; 1 preloads NVVM and JIT libraries and raises the footprint | same page | Keep 0 |
| `CUDA_SCALE_LAUNCH_QUEUES` | Scales the CUDA command buffer (default 1x; 0.25x, 0.5x, 2x, 4x). PR #19042 reports "~120 MB higher system RAM usage per GPU" at 4x and no VRAM change; llama.cpp set 4x by default, then reverted that (#19227, hangs on one board) and now only documents it | [environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html), [PR #19042](https://github.com/ggml-org/llama.cpp/pull/19042), [PR #19227](https://github.com/ggml-org/llama.cpp/pull/19227) | Keep unset or lower (§1.2). 4x would cost est. 9 × 120 ≈ 1.08 GiB here |
| `CUDA_DEVICE_MAX_CONNECTIONS` (default 8, 1–32) | Hardware work queues per device; the documentation gives no memory figure | environment variables | Unmeasured; one stream per card is what ggml uses, so 1–2 may be safe (est.). A Q565 arm |
| `cudaDeviceSetLimit` | Stack size, printf FIFO, device `malloc` heap, device-runtime depth and pending launches, L2 fetch granularity and persisting size: all concern the device | [runtime API, device management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) | No host lever |
| Streams and cuBLAS handles | ggml creates a stream on first use of a slot (`GGML_CUDA_MAX_STREAMS` is 8) and a cuBLAS handle on the first cuBLAS call of a device; the handle's workspace (4 MiB below Hopper) is a `cudaMalloc` | [ggml-cuda `common.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh) | Lazy already; the first prompt per card pays the handle (host cost unmeasured) |
| Persistence mode, `nvidia-persistenced` | The daemon holds the device files open so the driver keeps GPU state between jobs; NVIDIA calls the overhead minimal and gives no memory figure. While a CUDA process runs the state is held either way | [driver persistence](https://docs.nvidia.com/deploy/driver-persistence/persistence-daemon.html) | 0 MiB while serving. Keep it on headless for NVML offsets ([hardware tuning](hardware_tuning.md) §1.4) |
| `NVreg_*` module parameters | The 580 registry list has no per-GPU host-memory parameter. `InitializeSystemMemoryAllocations` (default 1) zeroes system-memory allocations (CPU time per pinned allocation, not bytes); `ExcludedGpus` hides cards; `TemporaryFilePath` is for suspend | [`nv-reg.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/arch/nvalloc/unix/include/nv-reg.h) | Nothing to set. No `nvidia-uvm` parameter for memory was found either |
| `nvidia-drm` / `nvidia-modeset` | NVIDIA's README calls DRM KMS disabled by default; distribution packages may enable it. Each loaded GPU registers a DRM device | [DRM KMS chapter](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/kms.html) | Est. ≤ 10 MiB; check `/sys/module/nvidia_drm/parameters/modeset`. A boot-time blacklist needs the reboot (no module unloads on a rig without the owner) |

### 1.2 What a context costs

- **Virtual, not resident:** the first `cudaSetDevice` raised one process from ~300 MB to 107 GB of address space, 200–300 GB
  with several devices; the one reply, hedged as outside its author's area, put it down to the unified address space
  reserving room for all host and device memory
  ([forum thread](https://forums.developer.nvidia.com/t/issues-using-cuda-on-a-realtime-redhat-system/35813)).
  It rules out `vm.overcommit_memory=2` (a HiveOS thread on 12–13 AMD GPUs tried it with partial relief only).
- **First use, resident:** one recent report on two A6000s and CUDA 12.9 shows RSS rising from 106 MB to 246 MB over the
  first 200 contexts, the jump attributed to CUDA initialisation, then ~100 kB a context
  ([OpenMM #5536](https://github.com/openmm/openmm/issues/5536)); an order of magnitude with the application's libraries in
  it, not a per-context figure. Library pages are file-backed and shared by all nine contexts.
- **The launch queue, the one priced item:** 4x adds ~120 MB per GPU, so if the queue scales linearly the default is est. 40
  MB per GPU and 0.25x est. 10 MB. Over nine contexts: ~360 MiB at the default, ~90 at 0.25x, ~1.08 GiB at 4x; **est. up to
  270 MiB back at 0.25x, 180 at 0.5x**. Linear scaling and "system RAM" meaning resident pages are both assumed.
  Its speed side: the PR measured prompt gains of 10–25 % at 1k–8k tokens for 4x on two large cards (a 70B model under
  pipeline parallelism) and none at decode. The rig decodes by CUDA graph (about one launch per card per token, est.), so the
  cost would fall on prompt micro-batches, which launch eagerly, and only where the CPU outruns the queue: on a 2-core host
  feeding x1 links that is possible, so it is an A/B on the agent check's prompt tok/s, not an assumption.
- **Everything else per context** (the driver's pushbuffers, UVM's per-GPU state, kernel-side allocations per GPU): no
  published figure; NVIDIA says only that driver memory is not reported on Linux. Mining-rig threads give none for NVIDIA
  either: the HiveOS reports (AMD, 12–13 GPUs on 4–16 GB) end in OOM kills, with more RAM as the one fix
  ([thread](https://hiveon.com/forum/t/ttm-failed-allocating-swap-storage/17802)).
- **Est. total** for nine contexts at the default queue: ~360 MiB of launch queue plus an unknown driver share, so 360–630 MiB
  if the other state is 0–30 MiB a card. This is the largest unmeasured block of the table above.

### 1.3 Measuring one context (Q565, no new tooling)

For 1…9 visible cards (`CUDA_VISIBLE_DEVICES`), start an idle llama-server on a tiny model with `-ngl 99`, wait for the load, and
read `/proc/<pid>/smaps_rollup` (Rss, Pss_Anon, Pss_File, Pss_Shmem) and the `/proc/meminfo` lines Shmem, AnonPages, Slab,
SUnreclaim, VmallocUsed, PageTables, KernelStack and Percpu before and after. The slope per card is the context's host
cost, the offset the process's. Repeat with `CUDA_SCALE_LAUNCH_QUEUES` at 0.25x, 1x and 4x, `CUDA_MODULE_LOADING` at EAGER
and LAZY, and `CUDA_DEVICE_MAX_CONNECTIONS` at 1, 2 and 8. A card count whose slope exceeds ~100 MiB would justify hiding idle
cards on files that fit fewer than nine (Q2_0 needs ≥ 7 cards at 64k, Q213).

## 2. Pinned memory (question 2)

### 2.1 What llama.cpp pins, and why

- **CPU-side weights.** `make_cpu_buft_list` puts the first GPU's host buffer type ahead of the plain CPU type unless `--no-host`
  (PR #16310); CUDA's host buffer type is `cudaMallocHost` with no flags, wrapped as a CPU buffer, falling back to ordinary
  memory if the allocation fails or `GGML_CUDA_NO_PINNED` is set, with a 4 GiB cap per buffer
  ([`ggml-cuda.cu`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu),
  [`llama-model.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp)). The input layer is
  always kept on the CPU ("very little benefit to offloading"), so the token embedding is pinned by default.
- **The output buffer** (logits, embeddings, backend-sampling rows) uses the host type of the output device: n_vocab ×
  n_outputs_max floats, plus 2 × n_vocab floats and (1 + n_vocab) ids per output row with backend sampling, and terms that
  scale with `-b` for layer-tap embeddings ([`llama-context.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-context.cpp)).
  At n_vocab 248,320 and one slot that is est. 4 MiB; the load log's `output buffer size` line shows it.
- **The CPU backend's compute buffer and the scheduler's input copies** (379 MiB at `-ub 512`, Q549; four copies under
  pipeline parallelism, Q89), **the fork's loader staging** (Q580) and **its packed upload buffers** (Q120).
- **Why pinned at all:** asynchronous host-to-device copies need page-locked memory, and the fork's packed uploads stage from
  pinned buffers ([host in the decode loop](host_decode_loop.md) §6). NVIDIA's runtime documentation warns that "excessive"
  pinned memory degrades the system because it leaves less to page
  ([runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)).

### 2.2 The alternatives and their price on a x1 link

| Alternative | Effect on pinned bytes | Speed price over PCIe 1.0 x1 |
|---|---|---|
| Pageable memory (`GGML_CUDA_NO_PINNED=1`) | Same bytes, but swappable, migratable and CMA-eligible. A pageable copy is staged through the driver and, per NVIDIA, the call returns once the data is staged while the DMA may still run ([API synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)) | Link ~190 MB/s per card against a staging memcpy at memory speed (est. ≥ 2 GB/s on this host): est. ≤ 10 % more CPU on bulk copies, same bulk rate; per-token inputs (kilobytes) unmeasurable. Loses asynchrony: the fork's packed uploads stage from pinned buffers (host submit 2.91 → 1.65 ms a token, Q120). The variable is process-wide, so it cannot spare them. An A/B, not a default |
| `--no-host` | The CPU weights (the 260 MiB embedding) leave pinned memory for ordinary memory; with mmap the buffer can be file-backed and evictable, with `-lm dio` it is anonymous. Also lets CPU repack buffers be used (PR #16310) | Q558 ran it; first-pass page faults on mmap'd embedding rows are ~0.23 per token (Q118) |
| The embedding on a card (`-ot token_embd=CUDA<n>`, `LLAMA_INPUT_ON_GPU`) | −260 MiB host | −0.43 ms a token (Q150's B2), +260 MiB of VRAM on one card (Q563) |
| Smaller staging (`LLAMA_DIO_STAGING=2x16`) | 256 → 32 MiB during each buffer load (Q580) | Load time from the stick: S31's `stage` arm |
| Smaller host scratch (`-ub`, `--spec-draft-n-ubatch`) | Compute buffers scale with the micro-batch (Q549, Q558) | `-ub 256` costs prompt speed on Flint (154 tok/s against 580 at `-ub 128`, Q83); the draft's `128` costs nothing measured |
| `cudaHostRegister` on existing pages | None: it page-locks the same range and tracks it like `cudaHostAlloc`; `cudaHostRegisterReadOnly` needs a device attribute for read-only mappings | No gain for the 28.8 GB table, whose reads are random 4 KiB on the CPU. master's `ggml-cuda.cu` reads a `GGML_CUDA_REGISTER_HOST` switch; its use was not confirmed |

### 2.3 Why pinned memory is what runs out

A long-term pin (the kernel's `FOLL_LONGTERM` path) migrates a page out of movable zones and CMA pageblocks before pinning it
and fails if it cannot ([patch series](https://lore.kernel.org/r/20201217185243.3288048-6-pasha.tatashin@soleen.com)).
Pinned buffers and the kernel's unmovable allocations therefore share the ~2,867 MiB outside the KHO scratch, which is the
mechanism behind Q558's finding that pinned memory, not MemAvailable, ran out.

## 3. Linux (question 3)

### 3.1 The CMA-marked quarter is KHO scratch (the cause behind Q560)

- **What the rig shows:** 485 CMA pageblocks (970 MiB: 206 in DMA32, 279 in Normal) with `CmaTotal: 0 kB`, idle and with the
  desktop stopped (Q560); `CmaFree` 411 MiB idle and 955–967 MiB at both OOMs.
- **What the source does:** mainline's `kexec_handover.c` reserves KHO's boot-memory ("scratch") regions with memblock at
  boot (low, global and per node), sizes them by default at 200 % of the memory the kernel had reserved so far, and
  `kho_init()` releases each pageblock to the allocator with `init_cma_reserved_pageblock` as `MIGRATE_CMA`. That is why they
  count in `CmaFree` and the page-type table but in no CMA area (the Strata reporter found `/sys/kernel/mm/cma/` empty).
  The kernel documentation says the regions are declared CMA so the memory stays usable, and that CMA admits only movable pages
  ([KHO concepts](https://docs.kernel.org/next/core-api/kho/concepts.html)).
- **Ubuntu's 7.0 kernels:** two independent reports show `CONFIG_KEXEC_HANDOVER=y`, `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y` and
  `CONFIG_MEMBLOCK_KHO_SCRATCH=y` with no `kho=` or `cma=` on the command line (7.0.0-31, 17.2 GiB of scratch on a 128 GB
  host), and `CmaTotal: 0 kB` with 8.7 GiB of `CmaFree` inflating `MemAvailable` (7.0.0-38, the first rig's kernel)
  ([Strata #3698](https://github.com/Niko1221/Strata/issues/3698), [Gufo #468](https://github.com/gufo-org/gufo/pull/468)).
  On a 4 GB host the same rule gives 970 MiB, a quarter of the module.
- **The switch:** `kho=` accepts 0/off/n and 1/on/y; `kho_bootmem=` (earlier trees: `kho_scratch=`) sets the sizes, as a
  percentage or three sizes ([kernel parameters](https://raw.githubusercontent.com/torvalds/linux/master/Documentation/admin-guide/kernel-parameters.txt)).
  The reporter who booted with `kho=off` saw `CmaFree` drop to 0 kB and the pinned-memory stall disappear.
- **What KHO is for:** preserving state across a live-update `kexec`. The rigs boot cold and Q570 found no crash kernel.
- **Consequences:** (1) +410–970 MiB usable (the idle and OOM `CmaFree` readings bracket it), and `MemAvailable` stops
  over-reporting; (2) Q83's headless gain becomes ~280 MB usable instead of ~76 MiB; (3) the guards that subtract `CmaFree`
  (Q559, `mem_guard.sh`) keep working, reading 0; (4) it needs one reboot, so the passphrase at the console, and the owner.

### 3.2 Swap, zram and zswap

- **What they are:** zram is a compressed block device in RAM that assumes ~2:1 and can write idle or incompressible pages
  to a backing partition; same-filled pages take no memory ([zram](https://docs.kernel.org/admin-guide/blockdev/zram.html)).
  zswap is a compressed cache in front of an existing swap device, capped by `max_pool_percent`, evicting least-recent
  pages to the swap device when full ([zswap](https://docs.kernel.org/admin-guide/mm/zswap.html)). The rig already swaps to
  a 3.7 GiB file on the SATA SSD, so zswap is the fit: the SSD stays the backstop.
- **Speed on the Celeron:** lz4 compresses 770 MB/s and decompresses 4,950 MB/s at ratio 2.11 on a 4.9 GHz Core i7-9700K;
  zstd -1 510 and 1,570 MB/s at 2.90 ([lz4](https://github.com/lz4/lz4)). Scaled by clock to 1.8 GHz (est.): lz4 ~285 and
  ~1,800 MB/s, zstd -1 ~190 and ~580 MB/s. A 4 KiB page then decompresses in est. 2 µs (lz4) or 7 µs (zstd), against 138
  µs median for a 4 KiB read at depth 1 on the rig's SSD ([Q214](../../data/2026-11-03_table_storage/README.md)), which also
  serves the n-gram table.
- **What it would buy:** a checkpoint restore from swap is 2.8 s against 0.63 s resident (113 MiB = 28,928 pages, ~76 µs
  each); from a zswap hit est. 0.8–1.0 s. It would also keep swap traffic off the SSD that serves the table.
- **What it costs:** RAM. Recurrent state and KV are float data (est. 1.1–1.4× with lz4); daemon heaps compress 2.5–3× (est.);
  zero pages are free. A 5 % pool is ~190 MiB of a host with 154–205 MiB usable, so zswap helps latency and hurts capacity;
  the reclaim itself spends the second core. It does not free MiB, so it is low in the list and only for a rig that still
  swaps after §3.1. Per-cgroup `memory.zswap.max` can cap the pool llama-server uses.

### 3.3 Reclaim policy and protecting llama-server

- **`vm.swappiness`** (default 60, range 0–200): the kernel documentation defines 100 as equal I/O cost for file and swap pages,
  and Chris Down puts an SSD near it ([sysctl vm](https://docs.kernel.org/admin-guide/sysctl/vm.html),
  [In defence of swap](https://chrisdown.name/2018/01/02/in-defence-of-swap.html)). Here the file cache of the table is worth
  little (Q228: 5.70 reads a token at ≥ 2 GiB of cache, 6.23 at 512 MiB, 16.3 with none, so ~0.08 ms a token between 0.5 and
  2 GiB), while a swapped checkpoint costs 2.2 s. That argues against raising it; no measurement supports lowering it either,
  because it also keeps cold daemon pages resident. Leave 60; test 30 only with the agent check.
- **`vm.watermark_scale_factor`** (default 10 = 0.1 % of memory between watermarks, ~4 MiB here): 50–100 starts kswapd
  earlier at the cost of est. 15–35 MiB of idle headroom, for fewer direct-reclaim stalls. `vm.min_free_kbytes` stays at its
  default. `vm.overcommit_memory` stays 0: mode 2 contradicts CUDA's address-space reservation (§1.2), mode 1 frees nothing.
- **cgroup v2 and systemd:** `memory.low` is best-effort protection (reclaimed only when unprotected cgroups have nothing left),
  `memory.high` the throttle and "main mechanism", `memory.swap.max` a swap cap
  ([cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html), [systemd.resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)).
  Run llama-server in its own service with `MemoryLow=` near its expected resident share, and put `system.slice` and
  `user.slice` under `MemoryHigh=` of ~192M: cold daemon pages go to swap before the server's. Est. +50–150 MiB resident
  (the daemons' resident set is est. 150–250 MiB, part of it already swapped). Cost: SSH and journald slow under pressure.
- **MGLRU's `min_ttl_ms`** (`/sys/kernel/mm/lru_gen/`, default 0): protects the working set of the last N ms and, in the
  documentation's words, "The OOM killer is triggered if this working set cannot be kept in memory"; 1000 is the suggested
  starting value ([multi-gen LRU](https://docs.kernel.org/admin-guide/mm/multigen_lru.html)). Against Q226's eight minutes at
  100 % system time, with the guard's loop starved, an in-kernel trigger acts without user-space CPU. The victim is llama-server
  (`oom_score_adj` 1000). Check `/sys/kernel/mm/lru_gen/enabled` first. Cost: a legitimate load burst can be killed.
- **`systemd-oomd`**, if installed, kills a cgroup on swap use or memory pressure by its own thresholds
  ([ManagedOOM*](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)); on a session
  scope it would be a second, unlogged killer. Check `systemctl is-active systemd-oomd`.

### 3.4 Page cache, mmap and `posix_fadvise`

- **What llama.cpp does on Linux** ([`llama-mmap.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-mmap.cpp)):
  `MAP_SHARED`; `posix_fadvise(SEQUENTIAL)` over the whole file; `MADV_RANDOM` on every lazy range (the table); `MADV_WILLNEED`
  and `MAP_POPULATE` only when prefetching and no lazy range exists (Q547's double read); `unmap_fragment` uses `munmap`, not
  `MADV_DONTNEED`.
- **With `-lm dio`** the weights bypass the page cache; what the cache holds is the table's 4 KiB reads (11–14 per token,
  [Q214](../../data/2026-11-03_table_storage/README.md)). Q228's replay shows the cache reaches its floor at ~2 GiB, a 32–256
  MiB row cache reaches the same floor, and 0.5 GiB costs 0.5 read a token. So the kernel's default (drop clean file pages
  first) is the right policy, and no `fadvise` call frees MiB: those pages are already free to reclaim.
- **One check:** the table's mapping should show `rr` (random read expected) in `VmFlags` of `/proc/<pid>/smaps`; without it every
  fault reads around the page, wasting cache and SSD time on a 4 KiB lookup.
- **`MADV_PAGEOUT` or `MADV_COLD`** on a saved checkpoint would only move it to the swap file: the swap-in price of §3.2 comes
  back at the next restore.

### 3.5 Services, journal, modules and boot parameters

| Item | What is known | Est. MiB | Note |
|---|---|---:|---|
| Services and modules | Q571's list (rsyslog beside journald, udisks2, upower, polkit, power-profiles-daemon, kerneloops, fwupd, packagekit, irqbalance; sound, Bluetooth and Wi-Fi modules); a Go-based mesh-VPN agent is a candidate, with `GOGC` and `GOMEMLIMIT` as its soft limits ([Go GC guide](https://go.dev/doc/gc-guide)) | 50–100 | Q571. Limiting the agent's heap trades CPU on a host whose cores are the bottleneck; low priority |
| journald | `RuntimeMaxUse=` defaults to 10 % of the runtime file system (a tmpfs, so RAM), capped at 4 GiB; `Storage=auto` is persistent when `/var/log/journal` exists ([journald.conf](https://www.freedesktop.org/software/systemd/man/latest/journald.conf.html)) | 0–40 | `RuntimeMaxUse=16M`; read `du -sh /run/log/journal` first |
| swiotlb | The default bounce-buffer pool is 64 MiB, allocated at boot below 4 GiB; `swiotlb=<slots>` (2 KiB each) sets it, and with `CONFIG_SWIOTLB_DYNAMIC` the pool grows on demand ([swiotlb](https://docs.kernel.org/core-api/swiotlb.html)). On x86-64 it is allocated when RAM extends above 4 GiB and no IOMMU takes over (est.; the document gives the pool and its size, not this trigger): the rig's DMA32 zone is ~2.2 GiB of the module, so RAM does sit above the line | ≤ 56 | `dmesg \| grep -i 'software IO TLB'` and `/sys/kernel/debug/swiotlb`; `swiotlb=4096` (8 MiB) rides the §3.1 reboot. A 32-bit-only device then bounces in a small pool |
| THP | `enabled` is `always`, `madvise` or `never`; `max_ptes_none` lets khugepaged back sparse regions with 2 MiB pages, "more memory usage" ([THP](https://docs.kernel.org/admin-guide/mm/transhuge.html)) | 0–60 | Ubuntu's default is `madvise` (check); `transparent_hugepage=madvise` on the command line if not |
| `crashkernel`, `cma=` | `crashkernel=` reserves a region unavailable to the system; `cma=` sizes the global CMA area ([kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html)) | 0 | Q570: none set; `cma=0` would not touch KHO's pageblocks (est.: KHO declares no CMA area) |
| Initramfs | Freed once the root is mounted | 0 | |
| Integrated graphics' stolen memory | Pre-allocated by the BIOS, locked for video and invisible to the OS ([Intel FAQ](https://intel.com/content/www/us/en/support/articles/000006532.html)); i915 reports it | ≤ 45 | Q570, parked: the console is the passphrase prompt, and the setting may be absent from the menu |
| `nvidia-drm`, `nvidia-modeset`, `i915` | §1.1; `nomodeset` or an `i915` blacklist leaves the console to the firmware framebuffer | 10–25 | Not recommended: a blank console at the passphrase needs the console to fix |

## 4. llama.cpp (question 4)

### 4.1 Host buffers that can move to the cards, and the loader

| Host buffer | Where it can go | MiB of host | Where it stands |
|---|---|---:|---|
| Main model's token embedding (Q3_K) | `-ot token_embd=CUDA<n>` or `LLAMA_INPUT_ON_GPU=1`; card 4 has ~1.3–1.4 GB free | 260 | Q563, planned |
| The draft's embedding and head | Shared with the main model (`LLAMA_DRAFT_SHARE=embd[,head]`) or on a card (`-otd`) | 0 host once `-otd` (644 MiB of a card) | Q564, built, CPU-checked |
| Context checkpoints | `--ctx-checkpoints-vram N`: each layer's state copied inside its card; est. 38–64 MiB of VRAM per card for four (a state is 9.4–15.9 MiB per card, Q550); the fullest card has ~519 MiB free | up to 452 | Q440, built, CPU-checked ([data](../../data/2026-11-04_ckpt_cards/README.md)) |
| Compute buffer (CPU backend) | A smaller micro-batch (Q549); the mask is already built on the cards (Q110, Q116) | 379 at `-ub 512` | Q549 |
| Loader staging | `LLAMA_DIO_STAGING=2x16`; or load the draft before the main context | 224 at a load | Q580 |

### 4.2 Load modes and `--mlock`

- `-lm` modes (server help): `auto`, `none` ("slower load but may reduce pageouts if not using mlock"), `mmap`, `mlock`
  ("keep model in RAM rather than swapping or compressing"), `mmap+mlock`, `dio`. Against 4 GB, `dio` keeps weights out of
  the page cache and out of Q547's double read, at the price of pinned staging (Q580); `mmap` with `--no-host` leaves the CPU
  tensors evictable.
- **`mlock` modes lock what the loader maps for the CPU.** With layers on the cards that is the 260 MiB embedding, and, if the
  loader also locks lazily read tensors (not confirmed), the 28.8 GB table: refused by the memlock limit or fatal. Never on
  Flash-Next. For the embedding alone, mlocked pages stay movable (`compact_unevictable_allowed` defaults to 1), unlike
  pinned buffers, so they could sit in CMA pageblocks; the embedding on a card is cleaner.

### 4.3 The server's own buffers

Defaults ([server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)): `--cache-ram` 8192
MiB (the prompt cache), `--ctx-checkpoints` 32 **per slot**, `-cms` 8192 tokens between checkpoints, `-np` auto (four slots,
unified KV), `-b` 2048, `-ub` 512, `--cache-idle-slots` on (needs `--cache-ram`). The rig's line already has
`--cache-ram 0`, `-np 1` and `--ctx-checkpoints 4` ([setup](../../setup.md)). What is left is checkpoints (452 MiB), the
output buffer (est. 4 MiB, more with layer taps at `-b 2048`: read the `output buffer size` line) and the sampler's rows
(on the card with `-bs`).

## What this changes in the plan

- **Q560 has a likely cause and a fix**: KHO scratch, `kho=off`. It is the largest single block and the cheapest per MiB.
- **Q565's first arm is the launch queue**, since it has a published price; the per-card slope of §1.3 decides whether hiding
  cards or `CUDA_DEVICE_MAX_CONNECTIONS` is worth anything.
- **Q83's headless result and Q558's "pinned memory runs out" are one fact** seen twice: usable memory is MemAvailable minus
  CmaFree until the reboot.
- **A reboot is worth bundling**: `kho=off`, a smaller `swiotlb=` (if the log shows the pool) and a THP check go in one edit.
- Suggested registry rows (not added here): KHO scratch (`kho=off`); the launch-queue arm of Q565; swiotlb; cgroup protection
  and `memory.high` on the daemons' slices; MGLRU `min_ttl_ms`; zswap as latency.

## Already covered

| Direction | Where it is |
|---|---|
| Headless boot (+280 MB), the debloat (no RAM), automatic upgrades off | Q83, Q506, Q507, Q569; [setup](../../setup.md) |
| `-ub`, `--cache-ram`, `--ctx-checkpoints` for a 4 GB host | [setup](../../setup.md) |
| Pinned compute buffer and mask copies; pipeline copies stay at four; the card-built mask | Q89, Q110, Q116 |
| The draft's embedding and head shared, the shortlist, `--no-host`, `-otd`, `--spec-draft-n-ubatch` | Q558, Q564, Q635, Q592; [draft data](../../data/2026-11-04_draft_mem/README.md) |
| Direct-I/O staging size and count | Q580 |
| Checkpoints in VRAM and at intervals | Q440, Q448, Q548 |
| The memory guard (swap-out, pressure, CmaFree) | Q559 |
| Services, modules, journald cap | Q571 |
| Firmware reserve and the BIOS graphics size | Q570 (parked) |
| A bigger module | Q86, Q339 (parked: "the point of this exercise is to run this on 4GB of RAM") |
| The table's page cache against its reads | Q228 |

## Not verified

- That the rig's 485 CMA pageblocks are KHO scratch: the match (CMA pageblocks, `CmaTotal` 0, Ubuntu 7.0 kernels with KHO on by
  default) is strong, but `/sys/kernel/debug/kho/out/scratch_len`, the kernel config and the command line were not read on the rig.
- Whether the rig's kernel names the size parameter `kho_bootmem` or `kho_scratch`.
- The launch queue's size and its scaling: 40 MB a GPU is a linear extrapolation from one PR's 4x figure, and "system RAM" may
  include pages that are not resident.
- Any other per-context host cost (Q565), and the host cost of a cuBLAS handle on the first prompt.
- Whether swiotlb is allocated on the rig, and whether any device there needs bounce buffers.
- That the allocator's CMA balance explains the idle CMA pages at the OOMs; the CPU cost of zswap on two Celeron cores; the
  compressibility of the recurrent state and KV; every "est." speed of a pageable copy.
- What `GGML_CUDA_REGISTER_HOST` registers, and whether the loader locks the lazily read table under `mlock`.
- That Ubuntu enables MGLRU and `nvidia-drm` modeset by default; whether `systemd-oomd` runs on the rigs.
- Whether CUDA's pinned (Shmem) pages are charged to llama-server's cgroup, which decides how much `MemoryLow=` protects.

## Sources

NVIDIA (P)
- CUDA Programming Guide, [lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)
  and [environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html);
  the [12.2.1 guide's lazy-loading page](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-programming-guide/lazy-loading.html).
- CUDA Runtime API: [memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html),
  [API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
  [device management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html).
- Driver: [persistence daemon](https://docs.nvidia.com/deploy/driver-persistence/persistence-daemon.html),
  [580 README, DRM KMS](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/kms.html),
  [`nv-reg.h` at 580.95.05](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/arch/nvalloc/unix/include/nv-reg.h).

llama.cpp (P): PR [#19042](https://github.com/ggml-org/llama.cpp/pull/19042) (launch queue: ~120 MB a GPU),
[#19227](https://github.com/ggml-org/llama.cpp/pull/19227) (its revert), [#16310](https://github.com/ggml-org/llama.cpp/pull/16310)
(`--no-host`); master's [`ggml-cuda.cu`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
(`GGML_CUDA_NO_PINNED`, host buffer type), [`common.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh),
[`llama-model.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp),
[`llama-context.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-context.cpp),
[`llama-mmap.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-mmap.cpp), the
[server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md).

Linux (P)
- KHO: [concepts](https://docs.kernel.org/next/core-api/kho/concepts.html),
  [`kexec_handover.c`](https://raw.githubusercontent.com/torvalds/linux/master/kernel/liveupdate/kexec_handover.c),
  [kernel parameters](https://raw.githubusercontent.com/torvalds/linux/master/Documentation/admin-guide/kernel-parameters.txt)
  (`kho=`, `kho_bootmem=`).
- Memory: [zram](https://docs.kernel.org/admin-guide/blockdev/zram.html), [zswap](https://docs.kernel.org/admin-guide/mm/zswap.html),
  [sysctl vm](https://docs.kernel.org/admin-guide/sysctl/vm.html), [cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html),
  [multi-gen LRU](https://docs.kernel.org/admin-guide/mm/multigen_lru.html), [THP](https://docs.kernel.org/admin-guide/mm/transhuge.html),
  [swiotlb](https://docs.kernel.org/core-api/swiotlb.html), [kernel parameters](https://docs.kernel.org/admin-guide/kernel-parameters.html),
  the long-term-pin series [mm/gup: migrate pinned pages out of movable zone](https://lore.kernel.org/r/20201217185243.3288048-6-pasha.tatashin@soleen.com).
- systemd: [resource control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html),
  [journald.conf](https://www.freedesktop.org/software/systemd/man/latest/journald.conf.html).
- [lz4](https://github.com/lz4/lz4) (lzbench, Core i7-9700K); [Go GC guide](https://go.dev/doc/gc-guide).

Reports (S): [Strata #3698](https://github.com/Niko1221/Strata/issues/3698) (KHO scratch as CMA on Ubuntu 7.0.0-31, `kho=off` works),
[Gufo #468](https://github.com/gufo-org/gufo/pull/468) (`CmaTotal` 0 and `CmaFree` 9 GB on 7.0.0-38),
[CUDA address-space reservation](https://forums.developer.nvidia.com/t/issues-using-cuda-on-a-realtime-redhat-system/35813),
[OpenMM #5536](https://github.com/openmm/openmm/issues/5536) (RSS after CUDA initialisation),
[HiveOS 12–13 GPU rigs](https://hiveon.com/forum/t/ttm-failed-allocating-swap-storage/17802),
[Chris Down, In defence of swap](https://chrisdown.name/2018/01/02/in-defence-of-swap.html),
[Intel on pre-allocated graphics memory](https://intel.com/content/www/us/en/support/articles/000006532.html).

This project (P)
- [Quant registry](../../../quants/experiments.md): Q83, Q86, Q89, Q110, Q116, Q118, Q120, Q213, Q214, Q215, Q226, Q228, Q339, Q440,
  Q448, Q506, Q507, Q547–Q549, Q558–Q560, Q563–Q565, Q569–Q571, Q580, Q592, Q635.
- Data: [Q226 and Q558](../../data/2026-11-03_flash_next_mtp/README.md) (`pageblocks.csv`, `session_memory.csv`),
  [draft memory](../../data/2026-11-04_draft_mem/README.md), [headless](../../data/2026-11-03_headless/README.md),
  [table storage](../../data/2026-11-03_table_storage/README.md), [n-gram replay](../../data/2026-11-03_ngram_replay/README.md),
  [checkpoints on the cards](../../data/2026-11-04_ckpt_cards/README.md).
- Notes: [slim fork](pascal_slim_fork.md), [host in the decode loop](host_decode_loop.md), [hardware tuning](hardware_tuning.md),
  [operations](ops_and_benchmarking.md); [setup](../../setup.md).

## Ideas for the rig

Ranked by MiB freed per unit of risk (risk labels in the conventions). MiB are est. unless a measurement is named.

| # | What to do | Est. MiB | Speed or stability cost | How to verify | Registry |
|---|---|---:|---|---|---|
| 1 | Boot with `kho=off` (edit `GRUB_CMDLINE_LINUX_DEFAULT`, `update-grub`, one reboot with someone at the console) **[R2]** | +410–970 usable | None at run time; KHO only serves kexec live update, which the rigs do not use. A reboot needs the passphrase | `/proc/cmdline` has it; `CmaFree` 0 kB; `/proc/pagetypeinfo` shows no CMA blocks; Q558's step-3 line loads without swap | Q560 |
| 2 | Check `CUDA_SCALE_LAUNCH_QUEUES` is unset in every launch path, then A/B `0.5x` and `0.25x` **[R0]** | 0 if unset; up to 180 and 270 | Possible prompt-speed loss under pipeline parallelism; decode (CUDA graphs) unaffected (est.); never `4x` (≈ 1.08 GiB) | `/proc/<pid>/environ`; Shmem and AnonPages after load at each value; the agent check's prompt tok/s | Q565 (new arm) |
| 3 | Put the main token embedding on a card: `-ot token_embd=CUDA4` or `LLAMA_INPUT_ON_GPU=1` **[R1]** | 260 | +260 MiB on a card with ≥ 1.3 GB free; −0.43 ms a token (Q150) | Load log's CPU buffer sizes; MemAvailable minus CmaFree once loaded; output identical | Q563 |
| 4 | `LLAMA_DIO_STAGING=2x16` on every load **[R1]** | 224 at a load (peak) | Load time from the stick unknown | S31's `stage` arm: no swap burst at the draft's load, load seconds | Q580 |
| 5 | A smaller swiotlb pool, `swiotlb=4096`, in the same reboot as #1, only if the pool is 64 MiB **[R2, shares #1's reboot]** | ≤ 56 | A 32-bit-only device bounces in a small pool; dynamic pools grow on demand | `dmesg` line and `/sys/kernel/debug/swiotlb` before; MemTotal and usable memory after | none |
| 6 | Checkpoints on the cards: `--ctx-checkpoints-vram 4` **[R2, code not yet on a rig]** | up to 452 | 38–64 MiB of VRAM per card (est.); restore from card memory instead of 0.63 s; off by default | S30's rig run: host memory, `ckpt_dev_n`, identical text | Q440, Q548 |
| 7 | Daemons under `MemoryHigh=192M` (`system.slice`, `user.slice`), llama-server in its own unit with `MemoryLow=` **[R1]** | 50–150 resident | SSH and journald slow under pressure | `systemd-cgtop -m`; MemAvailable at the session's low point | none |
| 8 | Q571's service and module stops; `RuntimeMaxUse=16M` for journald **[R1]** | 50–100 (+ ≤ 40 journal) | None measured | MemAvailable before and after on an idle rig | Q571 |
| 9 | MGLRU `min_ttl_ms=1000` **[R1, reverts at reboot]** | 0 (stability) | A load burst may be killed early; the victim is llama-server | `lru_gen/enabled`; a forced overload ends in an OOM kill within seconds, not minutes | none |
| 10 | Confirm THP is `madvise`; if `always`, set `madvise` **[R1]** | 0–60 | None | `/sys/kernel/mm/transparent_hugepage/enabled`; AnonPages after a session | none |
| 11 | `watermark_scale_factor=50` **[R1]** | −15…−35 (headroom) | Fewer direct-reclaim stalls | `allocstall_*` and PSI in a session | none |
| 12 | zswap, lz4, `max_pool_percent=5`, only after #1 if the rig still swaps **[R1]** | −190 at most (a latency tool) | Second-core CPU in reclaim; checkpoint restore est. 0.8–1.0 s against 2.8 s | `/sys/kernel/debug/zswap` stored and written-back pages; restore time in the responses' `cache_restore_ms` | none |
| 13 | A/B `GGML_CUDA_NO_PINNED=1`, never a default **[R1]** | 0 freed; up to 735 made swappable and CMA-eligible | Loses the fork's asynchronous uploads | Host submit time per token and agent-check speed | none |
| 14 | If Q565's slope per card exceeds ~100 MiB, hide idle cards on files that fit fewer than nine **[R1]** | 100 per card hidden | Fewer cards for the layer split | Q565's per-card slope | Q565 |
| 15 | BIOS graphics pre-allocation to its minimum **[R3]** | ≤ 45 | Console risk; the option may be absent | `MemTotal` before and after | Q570 (parked) |
