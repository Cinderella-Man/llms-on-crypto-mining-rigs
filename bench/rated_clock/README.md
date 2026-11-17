# rated_clock: ask the driver's rated-clock control directly (Q1556)

A P106-090 falls to its base clock, 1,354 MHz, between bursts of work, and Miner Llama holds it at boost with a
keep-alive kernel that costs Flint's prompts 16.5 %. Every documented driver control was tried on a rig and none held
boost ([R188](../../docs/data/2026-11-17_r186_r188_clocks/README.md)). NVIDIA's public driver headers describe one more
control that no NVIDIA tool sends to a Pascal card: the **rated-clock control** (`RATED_TDP`). This program sends it
directly, through the driver's control device, the way NVIDIA's own libraries talk to the driver. Background and
sources: [Can the driver hold a P106 at its top clock](../../docs/research/notes/clock_hold_by_driver.md), section 4;
registry row Q1556 in [quants/experiments.md](../../quants/experiments.md).

## The control, in the driver's words

The driver calls the base clock the "rated TDP" point; boost is anything above it. Five clients may each ask for an
action, and the driver obeys the first client, in this priority order, whose action is not `DEFAULT`:

| Priority | Client | Meant for |
|---|---|---|
| 0 | `RM` | the driver itself: it asks to limit a card to the rated point "when no power controllers are active"; no program can set it |
| 1 | `WAR_BUG_1785342` | a workaround that limits a card at boot |
| 2 | `GLOBAL` | an end-user tool's global switch |
| 3 | `OS` | the operating system, to lock clocks for profiling |
| 4 | `PROFILE` | application profiles |

| Action | Meaning |
|---|---|
| `DEFAULT` | no request of its own: boost above the rated point is allowed unless a client above asks otherwise |
| `FORCE_EXCEED` | allow boost and overrule lower clients that limit |
| `FORCE_LIMIT` | limit to the rated point |
| `FORCE_LOCK` | fix the clocks at a point |
| `FORCE_FLOOR` | fix the minimum at a point and let the maximum boost |

The point is either `RATED_TDP` (the base clock) or `TURBO_BOOST`. Where profiling is restricted to administrators,
the driver refuses `TURBO_BOOST` to other users, so the tool runs as root.

## What each mode does

`<card>` is an index in PCI bus order, the order of `nvidia-smi -i`, or a bus id such as `00000000:01:00.0`. Each
mode first prints the card's bus id and device file, to compare with
`nvidia-smi -i <card> --query-gpu=pci.bus_id --format=csv`.

| Mode | Sends | Changes anything |
|---|---|---|
| `status <card>` | the get for each of the five clients, then the status call | no |
| `lock-boost <card> <seconds> [os\|global]` | the reads, then `FORCE_LOCK` at `TURBO_BOOST` for the client (`os` by default), the reads again; holds the card open for `<seconds>` (1–3,600; Ctrl-C ends the hold early); then the default action and the reads | for `<seconds>` |
| `reset <card> [os\|global]` | the reads, then the default action (`DEFAULT` at `RATED_TDP`) for the client, the reads again | undoes a lock |

- **The reads.** Every client's action and point, and which client wins the arbitration (`holder:`), with a note
  when the winner holds the card at the rated point. Then the status call, whose fields name the driver's own active
  power controllers (`rm.clientActiveMask`; zero means the driver itself asks to limit), a registry switch
  (`rm.bRegkeyLimitRatedTdp`), the arbitrated action and point, and every client's input. The 580.178.04 headers
  define the status structure but no command for it; the tool sends the command id the structure's message id gives
  (`0x2080206d`), and a driver that does not know it answers `NV_ERR_NOT_SUPPORTED`, which the tool reports and
  accepts.
- **The lock lasts only while the tool holds the card.** With persistence mode off, the driver tears a card's state
  down when the last process closes it, so `lock-boost` keeps the card open for the hold and sends the default action
  itself at the end. `reset` is for a hold that was killed (`kill -9`, a lost ssh session) while another process,
  such as llama-server, kept the card open.
- **Errors.** Every mode frees what it allocated. A failed call prints the driver's status code with its name and the
  tool exits 1 without repeating the call. A failed read stops `lock-boost` before it sends the lock; once the lock was
  sent, whatever fails, the default action follows. `reset` sends the default action even after a failed read.
- **Version.** The tool refuses to run unless the driver's version string equals the headers' (580.178.04), using the
  driver's own strict version check.

## The risk, and the order

These are undocumented calls into the closed driver that runs Pascal cards. The headers are public, but the code
that answers for a P106 is not, and NVIDIA's tools do not send the boost lock to Pascal. Most likely the driver refuses
it or ignores it. If it misbehaves, a card or the whole driver can hang, and only a reboot clears that; on a rig with
an encrypted disk the reboot needs someone at its console.

1. `status` first, on one idle card, then while it decodes and while it reads a prompt, with the clock read beside
   each (`nvidia-smi -i <card> --query-gpu=clocks.sm,pstate,clocks_event_reasons.active --format=csv`). It only reads.
2. `lock-boost` only with someone at the rig's console, on one card, with a 0.5 s clock log
   (`nvidia-smi -i <card> --query-gpu=timestamp,clocks.sm,pstate,power.draw --format=csv,noheader -lms 500`).
3. `reset` right after, then the clock read back.

## Build

```bash
bench/rated_clock/build.sh
```

It compiles `rated_clock.c` in a container with no network (default image `nvidia/cuda:12.9.2-devel-ubuntu24.04`,
Ubuntu 24.04 like the rigs) into one static binary, `build/rated_clock`, that needs no library on the rig and uses
no AVX in its own code, and writes `build/BUILDINFO.txt`. With `NVSRC` set to a checkout of
[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) at tag `580.178.04` it also compiles
`abi_upstream.c`, which checks the same sizes, offsets and values (`abi_check.h`) against NVIDIA's own headers. Two
builds in that image on 2026-11-17 gave the same binary: md5 `48df2248786cd6068966f409eeed8805`, 965,280 bytes, gcc
13.3.0, glibc 2.39, statically linked, no program interpreter, 0 AVX instructions in the tool's own code; the upstream
check passed.

## Where the definitions come from

`nvidia/` holds MIT-licensed files from NVIDIA's open-gpu-kernel-modules at tag `580.178.04` (commit `c8e6998`):
`nvstatuscodes.h` as it is, and `nvrm_abi.h`, which copies only the definitions the tool uses, each block under the
name of its file:
- `kernel-open/common/inc/nv-ioctl.h` and `nv-ioctl-numbers.h`: the card list and the version check;
- `src/nvidia/arch/nvalloc/unix/include/nv_escape.h`: the alloc, control and free escapes;
- `src/common/sdk/nvidia/inc/nvtypes.h`, `nvstatus.h`, `nvos.h`: the base types and the alloc, control and free
  parameters;
- `class/cl0000.h`, `cl0080.h`, `cl2080.h`: the root client, device and subdevice classes;
- `ctrl/ctrl0000/ctrl0000gpu.h`: the GPU id to device instance call;
- `ctrl/ctrl2080/ctrl2080perf.h`: the rated-clock section.

How the driver handles these calls was read in the same tag's `kernel-open/nvidia/nv.c` (the ioctl entry, card list
and version check), `src/nvidia/arch/nvalloc/unix/src/escape.c` and `osapi.c`,
`src/nvidia/src/kernel/gpu/perf/kern_perf_pwr.c` (the set control and its administrator check) and
`src/nvidia/generated/g_subdevice_nvoc.c` (the exported controls: the get `0x2080206e` and the set `0x2080206f`, both
open to non-privileged clients; no `0x2080206d`). Nothing from that repository was built or run.

## Results

None yet: the runs are rows (a) and (b) of registry row Q1556.
