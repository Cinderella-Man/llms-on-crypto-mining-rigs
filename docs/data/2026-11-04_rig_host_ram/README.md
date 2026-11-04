# Host RAM on the rigs: the census tool and the run order (S32)

[scripts/rig_host_ram.sh](../../../scripts/rig_host_ram.sh) reads where a 4 GB rig's host RAM goes and applies, one at a
time and with an undo, the runtime settings that the research note ranks as giving some back. This folder holds what the
tool was checked against, the order of a run on a rig with the exact commands, and the numbers the note expects at each
step. The reasoning behind every setting, with sources, is in
[host_ram_4gb.md](../../research/notes/host_ram_4gb.md); the registry rows are Q692-Q700, Q563, Q565, Q571 and Q725-Q729 in
[quants/experiments.md](../../../quants/experiments.md). The tool was checked on the 7-GPU quant machine (below); on the
rigs, its census has run on the first rig at idle after the `kho=off` boot.

| File | What it is |
|---|---|
| [census_quant_machine.tsv](census_quant_machine.tsv) | the census of the 7-GPU quant machine (127 GB, kernel 7.0.0-42): what one block looks like |
| [expected.tsv](expected.tsv) | every number the note expects at each step of the run: census key, before, expected, basis |
| [run/s32_helpers.sh](run/s32_helpers.sh) | shell functions that make each step below one command (`rhr`, `measure`, `pull_all`, ...) |
| [census_rig1_after_boot.tsv](census_rig1_after_boot.tsv) | the first rig's census at idle after the permanent `kho=off` (2026-11-04, ~6 minutes after boot, headless, no model loaded) |

## The first rig after `kho=off` (2026-11-04)

`rhr census after_boot` on the first rig, idle and headless, ~6 minutes after a boot with `kho=off` on the kernel command
line (kernel 7.0.0-38). It is the idle baseline every later step on this rig compares against
([the full block](census_rig1_after_boot.tsv)):

| Row | Value |
|---|---|
| `mem.MemTotal_kB` / `mem.MemAvailable_kB` | 3,929,068 kB (3,837 MiB) / 3,286,296 kB (**3,209 MiB**) |
| `mem.CmaTotal_kB`, `mem.CmaFree_kB`, `pageblocks.CMA` | 0, 0, 0: no free memory is held as CMA any more, so `mem.usable_kB` = MemAvailable |
| `cmdline.kho`, `kho.scratch_len` | `off`, NA (the debugfs scratch file is absent once KHO is off) |
| `swiotlb.pool_MiB`, `thp.enabled`, `lru_gen.enabled` | 64, `madvise`, `0x0007` (MGLRU on) |
| `mem.swap_used_kB`, `vmstat.pswpout`, `vmstat.allocstall` | 0, 0, 0 |
| `cg.system.slice.current_kB` / `cg.user.slice.current_kB` | 391,564 kB (343,044 of it page cache) / 20,108 kB |

## The tool

Run it as root on the rig, read from stdin, so nothing is installed there:
`ssh "$RIG" 'bash -s -- census before' < scripts/rig_host_ram.sh`. Output files stay on the rig in `/var/tmp/rig_host_ram`
(a reboot keeps them); `apply` keeps what it changed in `/run/rig_host_ram` (a reboot clears it, and every setting is
runtime-only).

| Mode | What it does |
|---|---|
| `census [NAME]` | Read-only. One `key<TAB>value` row per fact, in a fixed order, so two runs diff. Memory (MemTotal, MemAvailable, MemFree, CmaTotal, CmaFree, Shmem, AnonPages, Slab, ...), `mem.usable_kB` (MemAvailable minus CmaFree), the CMA pageblocks of `/proc/pagetypeinfo`, the kernel command line, the kernel's KHO configuration and debugfs scratch regions, THP, the swiotlb pool from the kernel log, zswap and zram, swap, MGLRU, `vm` sysctls, reclaim counters and memory pressure, journald's disk and RAM use, the system and user slices' cgroup memory, the CUDA variables of the shell, and for a running llama-server its RSS and PSS, VmLck and VmPin, Shmem and anonymous pages, its cgroup's memory and the CUDA variables of its environment |
| `delta BEFORE AFTER` | The rows that differ between two census files, with the change and, for `_kB` rows, in MiB |
| `apply SETTING [VALUE]`, `revert SETTING` | `thp` (madvise), `mglru` (`min_ttl_ms` 1000), `watermark` (`vm.watermark_scale_factor` 50), `cgroup` (`MemoryHigh` 192M on `system.slice` and `user.slice`, `systemctl set-property --runtime`), `journald` (`RuntimeMaxUse` 16M, a drop-in under `/run`), `zswap` (5 % pool, lz4). Each prints the census delta. `cgroup` refuses while llama-server runs inside either slice, since the limit would throttle it |
| `boot-howto` | Prints the one-time GRUB steps (`kho=off`, and `swiotlb=4096` when the log shows the 64 MiB pool) and the permanent version. Text only: the script never edits GRUB and never reboots |
| `contexts` | Q565's protocol: an idle llama-server per card count (1, 3, 5, 9) and per arm (`CUDA_SCALE_LAUNCH_QUEUES` unset, `0.5x`, `0.25x`; `4x` is refused), with the host memory change over each load, and the per-card slope |

`contexts` needs `BIN` (llama.cpp's `bin/`) and `MODEL` (a small file that fits one card), and refuses to run while a
llama-server runs. Other arms are one `VAR=value` each, e.g. `ARMS="unset CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_DEVICE_MAX_CONNECTIONS=2"` (Q725) or `ARMS="unset CUDA_MODULE_LOADING=EAGER"` (Q726).

## Parsing check on the 7-GPU quant machine

The census ran there read-only (127 GB of RAM, 7 NVIDIA cards, kernel 7.0.0-42, which one of the rigs also runs); `delta` and the
`contexts` tables ran on synthetic census files, and a stand-in process named like the server exercised the per-process
rows. `apply`, `revert` and `contexts` did not run. What the census showed, in the note's terms
([the full block](census_quant_machine.tsv)):

| Row | Value |
|---|---|
| `kho.config` | `KEXEC_HANDOVER=y`, `KEXEC_HANDOVER_DEBUGFS=y`, `KEXEC_HANDOVER_ENABLE_DEFAULT=y`, `MEMBLOCK_KHO_SCRATCH=y`, `CMA=y`, `SWIOTLB_DYNAMIC=y` |
| `cmdline.kho`, `cmdline.cma` | unset, unset |
| `mem.CmaTotal_kB`, `cma.areas` | 0 kB, none |
| `pageblocks.CMA_MiB` | 17,842 MiB (8,921 pageblocks of 2 MiB: 409 in DMA32, 8,512 in Normal), 14 % of the memory |
| `kho.scratch_len` | three regions, 0x33200000, 0x314800000 and 0x113800000 bytes |
| `kho.scratch_MiB` | 17,842 MiB, the sum of the three: equal to `pageblocks.CMA_MiB` |
| `swiotlb.pool_MiB` | 64 |

On this kernel with its default command line the CMA-marked memory without a CMA area is the KHO scratch, to the
megabyte. That is the note's reading of Q560 on a machine other than a rig; a rig's census before the reboot checks it on
the rig itself (step 1 below).

## The run on a rig (S32)

Needs the rig idle (no timed task: the rig takes no other command during a run), booted headless, with the owner at its
console for step 2. About two hours, most of it model loads of ~4 minutes. Every step is one command; the values come
from [the S12 environment file](../2026-11-03_flash_next_mtp/run/s12.env.example) (`REL_BIN`, `MODEL`, `RIG_PORT`, `CTX`,
`ARGS_nodraft`), and `SMALL` is any model file of at most 2 GB on the rig.

```bash
# from the repository root; RIG is the rig's ssh alias
set -a; . s12.env; set +a
. docs/data/2026-11-04_rig_host_ram/run/s32_helpers.sh
```

**1. Before the reboot** (read-only). The gate: `kho.in_kernel` is `yes` and `kho.scratch_MiB` equals `pageblocks.CMA_MiB`;
if `kho.in_kernel` is `no`, stop here, since `kho=off` would change nothing.

```bash
rhr census before_boot
rhr boot-howto                    # this rig's own numbers, and whether the line gets swiotlb=4096
measure kho_unset                 # the same server load before and after the boot (loaded_kho_unset, vram_kho_unset.csv)
```

**2. The owner boots once with the new line** (the steps `boot-howto` printed: hold Shift or Esc at the firmware logo, `e`,
append `kho=off` and, if the pool is 64 MiB, `swiotlb=4096` to the `linux` line, Ctrl-X, the disk passphrase). Nothing is
saved: the next boot uses the old line.

**3. After the boot.**

```bash
rhr census after_boot
rhr delta before_boot after_boot          # expected: cmdline.kho off, CmaFree 0, pageblocks.CMA 0, usable +410 to +970 MiB
measure base                              # the new reference for every arm below
rhr delta loaded_kho_unset loaded_base
```

If the gain is there, the owner makes it permanent by hand (`boot-howto` prints the `GRUB_CMDLINE_LINUX_DEFAULT` edit and
`update-grub`).

**4. Q565 and Q693: the launch queue.** First the idle per-card slope on a small model (about 15 minutes), then the real
load at `0.5x` and `0.25x`; the prompt tok/s of each setting is a separate agent-check arm.

```bash
rhr_env "BIN=${REL_BIN%/*} MODEL=$SMALL" contexts      # slope.tsv: MiB_per_card per arm
pull_contexts
measure q05 'CUDA_SCALE_LAUNCH_QUEUES=0.5x'
measure q025 'CUDA_SCALE_LAUNCH_QUEUES=0.25x'
rhr delta loaded_base loaded_q05; rhr delta loaded_base loaded_q025
```

**5. One setting at a time**, each undone before the next (`measure` loads the server, takes the census and stops it):

```bash
rhr apply thp;       measure thp;    rhr delta loaded_base loaded_thp;    rhr revert thp         # Q697
rhr apply mglru;     measure mglru;  rhr delta loaded_base loaded_mglru;  rhr revert mglru       # Q696
rhr apply watermark; measure wm;     rhr delta loaded_base loaded_wm;     rhr revert watermark   # Q698
measure emb '' '-ot token_embd=CUDA4'; rhr delta loaded_base loaded_emb                           # Q563
measure nopin 'GGML_CUDA_NO_PINNED=1'; rhr delta loaded_base loaded_nopin                         # Q700
rhr apply journald;  rhr revert journald                                                          # Q571 (idle delta)
# Q695: the daemons' limit with the server in its own slice (MEMLOW: a little under loaded_base's llama0.cg.current_kB)
rhr apply cgroup; MEMLOW=1G measure_slice cgroup; rhr delta loaded_base loaded_cgroup; rhr revert cgroup
# Q699, only if loaded_base or an agent session shows swap use (mem.swap_used_kB, vmstat.pswpout)
rhr apply zswap;     measure zswap;  rhr delta loaded_base loaded_zswap;  rhr revert zswap
```

Reading each result: the delta's `mem.usable_kB`, `mem.Shmem_kB` and `mem.AnonPages_kB` rows against
[expected.tsv](expected.tsv), `vmstat.allocstall`, `vmstat.pswpout` and `psi.mem.*` for stalls (an agent session between two
censuses gives their growth), and `llama0.cg.shmem_kB` against `llama0.RssShmem_kB` to see whether CUDA's pinned pages are
charged to the server's cgroup (Q727). A setting stays only if its gain holds on the agent check's speed
([bench/agent_check](../../../bench/agent_check/README.md)); the result decides it for all five rigs.

**6. Collect and clean up.**

```bash
pull_all; pull_contexts
clean_rig
```

## Provenance

The tool and this folder were written on 2026-11-04. The expected numbers are the note's own estimates and measurements
(the note names its basis in each row); none was measured by this folder. The census block is the output of the script on
the quant machine; its rows carry no host name, address or serial.
