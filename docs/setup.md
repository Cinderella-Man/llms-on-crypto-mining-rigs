# Setting up the rig for LLM inference

This is the software configuration every benchmark in this repository was measured on, with the pitfalls found on
the way. It assumes the stock rig: nine P106-090 cards, the Celeron 3865U with 4 GB of RAM and the 64 GB SSD
([hardware spec](mining_rig.md)).

## What the hardware dictates

| Constraint | Consequence |
|---|---|
| Pascal GPUs (compute capability 6.1) | NVIDIA driver **580** is the last driver branch and **CUDA 12.x** the last toolkit line that support them. CUDA 13 cannot build code for these cards. Build everything for `sm_61` with CUDA 12.9. |
| No tensor cores, FP16 at 1/64 of the FP32 rate | Engines built around FP16 or tensor cores don't run or crawl. [llama.cpp](https://github.com/ggml-org/llama.cpp) runs well: its quantized kernels use the cards' fast 8-bit integer dot product (`dp4a`). |
| Celeron without AVX, 4 GB RAM | The CPU can't do useful model computation, so every weight has to fit in the 54 GB of GPU memory. Software must not be compiled with AVX: a binary built with AVX on another machine crashes with "Illegal instruction". |
| 64 GB SSD | Disk space is the first limit you hit. A desktop install plus the full CUDA toolkit already takes ~25 GB; model files are 4–40 GB each. |

## Operating system and driver

- **OS:** Ubuntu 24.04 LTS (desktop install works; the desktop session holds only 4 MiB of memory per GPU).
- **Driver:** the `nvidia-driver-580` package from Ubuntu's repositories. `nvidia-smi` should list nine
  "NVIDIA P106-090" cards.
- **Driver updates:** when `apt upgrade` updates the NVIDIA driver, `nvidia-smi` fails with "Driver/library version
  mismatch" until the next reboot. `/var/run/reboot-required` shows that a reboot is pending.

## CUDA toolkit

The tested install is CUDA 12.9.2 (`cuda-toolkit-12-9`) from NVIDIA's "local" `.deb` installer for Ubuntu 24.04.
Three things are worth knowing on a 64 GB disk:

- **The local installer leaves a 4.4 GB copy of all its packages** in `/var/cuda-repo-ubuntu2404-12-9-local`. After the
  toolkit is installed, `sudo apt purge cuda-repo-ubuntu2404-12-9-local` deletes it and keeps the installed CUDA.
  Later CUDA package installs then need NVIDIA's online repository instead.
- **The full toolkit is ~7 GB plus ~2 GB of profilers** (Nsight Compute and Nsight Systems under `/opt/nvidia`).
  llama.cpp links only against `libcudart`, `libcublas` and `libcublasLt`; the cuFFT, cuSPARSE, cuSOLVER, NPP, cuRAND and
  nvJPEG libraries (~3.7 GB with their development files) are unused.
- **`nvcc` is not on `PATH`** after installation. Use `/usr/local/cuda-12.9/bin/nvcc`, or pass it to CMake as below.

`sudo apt clean` frees the downloaded package cache (~0.5 GB) as well.

## Building llama.cpp

Build on the rig itself, so the CPU part is compiled for the Celeron and can't contain AVX instructions:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build --config Release -j 2
```

The tested build is commit `85d5703a3` (2026-09-04). Its CMake cache records `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`,
`GGML_CUDA_GRAPHS=ON`, `GGML_NATIVE=ON` and `CMAKE_BUILD_TYPE=Release`, with all explicit AVX options off. Check the
result with:

```bash
build/bin/llama-server --list-devices
```

It should list `CUDA0` to `CUDA8`, each an "NVIDIA P106-090" with 6068 MiB.

Even though the build enables CUDA graphs, llama.cpp switches them off at runtime for Pascal GPUs; the
[small MoE benchmark](benchmarks/2026-10-22-small-moe-overhead.md) shows what that costs.

### Building newer versions on another PC

The rig's 2-core CPU builds slowly. [scripts/build_llamacpp_docker.sh](../scripts/build_llamacpp_docker.sh) builds any
llama.cpp version on another x86-64 Linux PC with Docker, inside NVIDIA's CUDA 12.9.2 image for Ubuntu 24.04: code for
compute capability 6.1 only, no AVX in the CPU part, and llama.cpp's own libraries linked statically, so the result
only needs the rig's CUDA 12.9 runtime libraries. It produces a ~160 MB tarball to unpack on the rig. With
`--cuda-graphs-pascal` it also applies [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721), which re-enables
CUDA graphs on Pascal and speeds up generation by 7–16 % in the benchmarks here. New model architectures need a recent
build: Flint, for example, is supported only from 2026-10-21.

## Running llama-server on a 4 GB host

Several llama-server defaults assume a machine with plenty of RAM. On this rig they can exhaust the 4 GB and trigger
the kernel's out-of-memory killer, which may take parts of the desktop with it:

| Setting | Default | On this rig | Why |
|---|---|---|---|
| `-ub` (micro-batch) | 512 | `-ub 128` (with `-b 512`) above ~100k tokens of context | The attention mask lives in pinned host memory, four copies of context × micro-batch; at 262,144 tokens the default needs ~1 GiB |
| `--cache-ram` | 8192 MiB | `--cache-ram 0` | Prompt cache kept in host RAM |
| `--ctx-checkpoints` | 32 per conversation | 0–4 | Each checkpoint of a hybrid model (Qwen3.6, Qwen3.8) holds the recurrent state of its linear-attention layers in host RAM |

Bind the server to `127.0.0.1` unless you mean to expose it, and keep an eye on `MemAvailable` in `/proc/meminfo`;
[scripts/server_bench.sh](../scripts/server_bench.sh) shows a watchdog that stops the server when it drops below
400 MB.

## Storing models

Model files don't fit comfortably on the 64 GB SSD next to the OS and CUDA. The benchmarks here load models from a
USB 3 flash drive formatted as ext4 and mounted at `/mnt/models`. The drive tested (a 64 GB SanDisk Ultra Fit,
using the kernel's `uas` driver at 5 Gbit/s) reads at ~410 MB/s and writes at ~120 MB/s. That is faster than the
~190 MB/s each GPU's PCIe link can accept, so it doesn't slow model loading. Format it once with:

```bash
sudo mkfs.ext4 -L models -m 0 /dev/sdX1     # check the device name with lsblk first: this erases the drive
sudo mkdir -p /mnt/models && sudo mount -o noatime LABEL=models /mnt/models
```

To mount it at every boot without risking the boot when the drive is missing, add this line to `/etc/fstab`:

```
LABEL=models  /mnt/models  ext4  defaults,noatime,nofail,x-systemd.device-timeout=5s  0  0
```

`nofail` makes systemd treat the mount as optional, the device timeout limits the wait for a missing drive to 5
seconds, and the final `0` skips the boot-time filesystem check. Check the line with `sudo findmnt --verify` and run
`sudo systemctl daemon-reload` afterwards. If the drive is plugged in after boot, mount it with
`sudo mount /mnt/models`. This matters on rigs with an encrypted root filesystem: a boot that stops at a missing drive
needs someone at the keyboard.

## Checking your rig

[scripts/rig_inventory.sh](../scripts/rig_inventory.sh) collects CPU flags, memory, driver and CUDA versions,
per-GPU clocks and power limits, PCIe link states and the llama.cpp build in one read-only pass. Its output contains no
hostname, MAC address or GPU serial, so you can share it when comparing rigs.
