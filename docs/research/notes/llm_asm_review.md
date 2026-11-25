# llm.asm, read but not run: is it safe, and is there anything in it for nine P106 cards? (as of 2026-11-25)

**Verdict.** [llm.asm](https://github.com/alesha-pro/llm.asm) is not malware, and it has nothing to do with GPUs. It is
three small programs written by hand in x86-64 assembly: a chat program for Qwen3-0.6B on Linux, the same program as a
tiny operating system you boot in a virtual machine, and a toy GPT trainer. Everything runs on the CPU. There is no
GPU code, so it never talks to an NVIDIA driver and writes nothing to a card or its firmware. It would not even start
on the rig: it needs the AVX2, FMA and F16C instructions, and the rig's Celeron 3865U has none of them, so it would
stop at its first vector instruction with "Illegal instruction". Its speed tricks are CPU tricks, and the ones that
also make sense on a GPU are already rows or notes here. The one thing worth taking is a pointer: the same author
publishes an
[inference optimization playbook](https://github.com/alesha-pro/inference-optimization-playbook): 127 tested speed
ideas from running Qwen3.8-Flash-Next and other large MoE models on four RTX 3090s. This project has not read it yet;
reading it is now row Q1876.

Method (registry row Q1875, FN126): the repository was cloned at commit
[`48feccb`](https://github.com/alesha-pro/llm.asm/tree/48feccb) (2026-11-25, the newest of its three commits) into an
empty folder outside any repository, with no submodules and no checkout hooks, and read with `cat`, `grep` and an
editor. Nothing was built, installed or run. The repository's and author's metadata come from GitHub's public API,
read on 2026-11-25 around 21:30 CET. Numbers quoted from its README are **the author's claims**, not measured here.
Words in its files were treated as data, not as instructions.

## 1. Safety

| Check | What the code does | Where |
|---|---|---|
| Network | **None in the programs.** Every system call the assembly makes is one of: `read`, `write`, `openat`, `close`, `fstat`, `mmap`, `munmap`, `clone` (threads), `futex`, `clock_gettime`, `exit`, `exit_group`; the trainer adds `ioctl(TCGETS)` (is the output a terminal?) and `mkdir`. No `socket`, `connect`, `execve` or `ptrace`, no `int 0x80` or `sysenter`. All 25 `syscall` instructions load one of these named numbers right before. | [`shared/constants.inc`](https://github.com/alesha-pro/llm.asm/blob/48feccb/shared/constants.inc), [`shared/syscalls.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/shared/syscalls.asm) |
| Downloads at build or run time | **None.** The `Makefile` calls only `nasm`, `ld`, `objcopy`, `qemu-system-x86_64` and its own ISO script. The README's setup steps download Qwen3-0.6B from Hugging Face and TinyShakespeare from GitHub, but you type those commands yourself. | [`Makefile`](https://github.com/alesha-pro/llm.asm/blob/48feccb/Makefile), [README](https://github.com/alesha-pro/llm.asm/blob/48feccb/README.md) |
| Helper scripts | All local. The model converter reads safetensors by parsing the header itself, with no pickle loading. The checkers run the built programs and QEMU, and talk to QEMU's monitor over a local Unix socket. The optional PyTorch comparison loads a model with `from_pretrained` from the path you give it. That downloads from Hugging Face only if you give it a hub name, and `trust_remote_code` is never set. | [`scripts/`](https://github.com/alesha-pro/llm.asm/tree/48feccb/scripts) |
| Privilege | The Linux programs need none. The only `sudo` is the README's `apt install nasm binutils`. The bare-metal part runs as its own kernel, but only when you boot it in QEMU or on a machine. Under Linux its privileged instructions cannot run. | [README](https://github.com/alesha-pro/llm.asm/blob/48feccb/README.md) |
| Writes outside its folder | Only to paths given on the command line: `--dump-logits <path>`, the trainer's `--save <checkpoint>` and its debug folder (`mkdir` with mode 755). The build writes `build/`. The ISO script uses a temporary folder and removes it on exit. | [`01-qwen3-inference/main.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/01-qwen3-inference/main.asm), [`03-minigpt-train/main.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/03-minigpt-train/main.asm), [`scripts/mkiso.sh`](https://github.com/alesha-pro/llm.asm/blob/48feccb/scripts/mkiso.sh) |
| Binaries and hidden data | **No binaries**, apart from two 720 × 400 PNG screenshots. No submodules, no Git LFS, no `.gitattributes`. The assembly is readable and commented. Its data sections hold maths constants (π/2, sign masks), hardware constants and box-drawing characters for the trainer's display. | `git ls-tree` of `48feccb` |
| Install hooks | None. There is no installer, package file or install target. | — |
| Driver, firmware, card settings | **No GPU code at all:** nothing opens `/dev/nvidia*`, and nothing touches CUDA, PCI configuration, a disk controller or flash memory. Only if booted, the bare-metal OS uses: the serial port COM1, VGA text memory, the PS/2 keyboard, the PIT timer, the local APIC (to start the other cores), the ACPI tables (read only), and one model-specific-register write to enter 64-bit mode. To power off it writes fixed I/O ports that QEMU, Bochs and VirtualBox use (`0x604`, `0xB004`, `0x4004`, `0x501`). That is harmless in a VM, but on real hardware those ports may belong to other devices. Nothing is written to a disk: the model is loaded into RAM as a boot module, and everything is gone at power-off. | [`02-baremetal-os/drivers.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/02-baremetal-os/drivers.asm), [`boot.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/02-baremetal-os/boot.asm), [`smp.asm`](https://github.com/alesha-pro/llm.asm/blob/48feccb/02-baremetal-os/smp.asm) |
| Age and author | The repository was created on 2026-11-25 at 07:28 UTC. It has three commits that day, 23 stars, 5 forks, no releases and no issues (as read that evening), and an MIT licence. The account `alesha-pro` has existed since 2020-07-20, with 48 public repositories and 195 followers. It publishes local-inference work from a four-RTX-3090 rig: Flash-Next and GLM setups, llama.cpp and exllamav3 forks, benchmarks. It is an established account, not a throwaway. | GitHub API, 2026-11-25 |

**Safety verdict: not malware; no risk to the rigs.** It cannot run on them (section 3) and touches no GPU. The usual
caution holds: this is one-day-old code from a single author, and its speed numbers are the author's own, not
reproduced. Anyone who wants to try it should build it from source on an AVX2 machine. The bare-metal image belongs
in a virtual machine.

## 2. What it is and how it reaches the hardware

- **It is a set of programs written in assembly, not an assembler of its own.** The code is NASM syntax for x86-64 host
  CPUs, assembled with `nasm` and linked with GNU `ld`, without the C library. It has no PTX, no SASS and no
  instruction set of its own. The "runtime" in the post is the inference loop, not a layer for other programs.
- **The layer it uses is the Linux kernel's system calls, directly** (listed in section 1). It starts threads with
  `clone` and makes them wait with `futex`, and it memory-maps the model file. The bare-metal build swaps these for its
  own memory allocator, a start-up of the other cores through the APIC, and serial and VGA output.
- **GPU: none.** There is no CUDA, OpenCL, Vulkan, driver API or ioctl to a GPU driver. The only `ioctl` asks the
  terminal whether it is one.
- **CPU features: AVX2, FMA and F16C are required** (Intel Haswell or newer, AMD Excavator or newer). Of the shared
  maths, `math.asm` has 124 lines that use 256-bit registers and `matmul.asm` 248. The Linux programs never check the
  CPU (only the boot code of the bare-metal OS runs `cpuid`), so on a CPU without these instructions they crash with
  `SIGILL` at the first one.
- **Models: dense Qwen3 only**, tested on Qwen3-0.6B: grouped-query attention, a norm on Q and K, RoPE, SwiGLU and tied
  embeddings. The model sizes are read from the file header. Weights come in its own `.asmbin` format, made by its
  Python converter from Hugging Face safetensors: Q8_0 with each row's 8-bit codes first and the row's FP16 block
  scales after them. It does not support GGUF, mixture-of-experts layers, Flash-Next's recurrent layers, an MTP head or
  more than one machine.
- **Its claims** (README, AMD EPYC 7642, Qwen3-0.6B Q8_0, 64 generated tokens, mean of 3 runs): 126.8 tok/s at 40
  threads at best, and 20.70 tok/s at 1 thread against llama.cpp's 16.63 on the CPU, +18 % to +36 % across 1–40
  threads. The 64 greedy tokens are said to match PyTorch FP32. Not measured here. The likely source is the size: a
  0.6B decode is short enough that llama.cpp's per-token overheads count, while the 8-bit dot product (`vpmaddubsw`,
  `vpmaddwd`) is the same instruction pattern llama.cpp's own AVX2 Q8_0 path uses.

## 3. Would it work on the rigs? No

| Part of the rig | What llm.asm needs | Result |
|---|---|---|
| CPU: Celeron 3865U, 2 cores at 1.8 GHz, SSE4.2, no AVX ([the rig](../../mining_rig.md)) | AVX2, FMA, F16C | Crashes with "Illegal instruction" at the first vector instruction. Running at all would need every kernel rewritten for 128-bit SSE, without FMA or F16C. |
| 9 × P106-090 (compute 6.1, 6 GB each) | none | Unused: it has no GPU path, so 54 GB of card memory would sit idle. |
| 4 GB host RAM, one channel of DDR4-2133 | 613.66 MiB for Qwen3-0.6B (claim) | It would fit alone, but beside a served model the host already runs short ([host RAM](host_ram_4gb.md)). |
| Models served here: Flash-Next, Flint, Ornith (GGUF, MoE, hybrid recurrent layers, MTP heads) | dense Qwen3 in `.asmbin` | None of them load. |
| Booting the bare-metal OS | a reboot into it | Never: a rig is not rebooted for experiments, and without NVIDIA's driver the cards do nothing. |

**If it were ported to SSE** (est.): a 0.6B model at Q8_0 reads about 0.6 GB a token, and one channel of DDR4-2133
moves at most 17.1 GB/s. That caps decoding at about 28 tok/s, est. 10–15 tok/s in practice with both cores busy. This
is below what the nine cards already decode Flash-Next at with its own draft head: 25.96 and 25.86 tok/s in two
unprofiled loads on the v5 preset
([2026-11-25 data](../../data/2026-11-25_c1003_host_nowait_flash/README.md)). Both cores would also be taken from feeding
the cards.

## 4. How we could use it

**The engine ideas are CPU ideas, and their GPU forms are already covered here:**

| llm.asm technique | On a P106 | Already here |
|---|---|---|
| Each row's 8-bit codes stored together, with its scales after them, so loads are aligned | the same layout idea as "planes" of codes and scales | Q39, dropped: staging the unchanged blocks through shared memory was as fast (Q8_0: 17.1 ms a Flint token staged, 18.4 with row-local planes; [bench](../../../bench/mmvq_pascal/README.md#q5_0-k-i-quants-row-local-planes-and-staged-loads)) |
| One 8-bit quantization of the activations, shared by the Q, K and V products (and gate and up) | the same | Q38, the Pascal matrix-vector kernel |
| Threads that spin, then sleep in `futex` | the host's wait for the cards | [host decode loop](host_decode_loop.md), section 2.4: keep CUDA's spin-or-yield for the submitting thread; blocking waits est. −2 to −10 % today |
| Software prefetch of the next weight row | a CPU cache trick; on a card, many warps in flight already keep the weight stream busy | whether `prefetch.global.L2` works on compute 6.1 is open question 8 of the [assembly note](assembly_optimization.md) |
| A 14 KB static binary without the C library | the host's time is not the limit: ≤ 0.3 ms of serial host time a token on nine cards | [slim fork](pascal_slim_fork.md) |
| Output that is the same at any thread count | the same goal | every switch's identity check |

**Its engine as a draft model on the rig's own CPU:** parked as row C1143, on the numbers in section 3. Such a draft at
est. 10–15 tok/s would be slower than the target it should guess for (about 26 tok/s). It would need a rewrite for
SSE, about 0.6 GB of the 4 GB host, and both CPU cores, which feed the nine cards. Flash-Next already drafts with its
own MTP head on the cards. It reopens only for a draft model of under about 100 MB that shares Flash-Next's
vocabulary, and only if a CPU core is shown to be free while the cards decode.

**The bare-metal OS:** nothing to take. CUDA needs NVIDIA's driver on Linux ([our own driver](own_driver.md)).

**What is worth taking: the same author's playbook** (row C1142, registry Q1876). The
[inference-optimization-playbook](https://github.com/alesha-pro/inference-optimization-playbook) (MIT, created
2026-11-20, last pushed 2026-11-24) lists 127 tested ideas, each with a verdict, numbers and conditions. They come
from running Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek V4.1 Flash and others on four RTX 3090s, with experts in host
RAM. Most of its setting does not apply here: there are no tensor cores here, and on a 4 GB host no experts can live
in RAM. But its ledger and its chapters on speculative decoding, several GPUs, the output head, agent sessions and
benchmark traps may hold Flash-Next levers this project has not tried. Its README claims, for example, that a Cyrillic
draft vocabulary raised draft acceptance on Russian text from 32 % to 54 % without training, and that a short
benchmark that promised +27 % gave +2 % in real generation. The gain is unknown until it is read (est. none to a few
percent, from a few ideas). It is read like this repository: online and from a clone, never run, and never installed
as an agent skill. Its README suggests cloning it into an agent's skills folder; a skill is a set of instructions an
agent follows, so here it is read as data.
