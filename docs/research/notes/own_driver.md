# Could we write our own driver to switch off the power saving?

**Short answer: no.** A driver of our own could not change a P106's clocks, and it would lose CUDA, which the whole
engine runs on. The one route left is asking NVIDIA's own driver for a clock setting its tools never send to a Pascal
card ([`bench/rated_clock`](../../../bench/rated_clock/README.md)). Registry rows: Q1560 (our own driver, closed) and
Q1556 (the direct request).

## The power saving we would like to switch off

A P106 lowers its clock as soon as it has nothing to do. Measured on rig 1 after a reply
([R186](../../data/2026-11-17_r186_r188_clocks/README.md)): it holds its busy clock for about 2 s, reads 1,354 MHz
(its base clock) by 5 s, about 800 MHz at 17 s, and is in its deepest idle state by 24 s.

In a layer split each of the nine cards works only about a ninth of the time and waits for the others, so to the
driver every card looks mostly idle. Short bursts of work then start from a low clock and never climb all the way up.

## How the pieces fit

| Layer | What it is | Who writes it |
|---|---|---|
| The engine | llama.cpp with this project's kernels | us |
| CUDA libraries | `libcuda`, `libnvidia-ml` (NVML) | NVIDIA, closed |
| Kernel module | `nvidia.ko`, which talks to the card | NVIDIA, closed for Pascal |
| Firmware on the card | small built-in processors on the GPU itself; one of them manages power, clocks and voltage | NVIDIA, signed |
| VBIOS | the card's own startup code and tables: clock steps, the 75 W power cap | NVIDIA and the board maker, signed |

The clock is set at the bottom of this stack, by the power-management firmware running on the card, not by the
driver.

## Why a driver of our own cannot do it

1. **The card checks firmware, not drivers.** Any driver may talk to the card. But from the second Maxwell generation
   (GM20x) on, which includes Pascal (GP10x), the card lets only firmware signed by NVIDIA reach the settings that
   change clocks and voltage. NVIDIA has not released such firmware for Pascal to anyone else, so a driver of our own
   has nothing it is allowed to load.
2. **The same signature check guards the VBIOS,** where the 75 W cap and the clock table live. An edited VBIOS does not
   start, and the only signed image for this card is the one already on it
   ([what a P106-090 holds back](p106_090_holds_back.md)).
3. **CUDA runs only on NVIDIA's driver.** Every kernel this project has written is CUDA. A different driver means
   starting over on Vulkan, and still at the low clock.

## How the open-source driver (nouveau) works, then

nouveau is the Linux kernel's own driver for NVIDIA cards, written by volunteers by studying how the cards behave, with
OpenGL and Vulkan (NVK) on top in Mesa. It runs a P106 for graphics and Vulkan, but at the clock the card starts with:
- **Older cards** (Kepler, the first Maxwell generation GM10x): nouveau loads its own power firmware and can change
  clocks.
- **GM20x, Pascal and Volta:** it cannot. In the project's own words, there is "little hope" of that, because the
  firmware "needs to be signed by NVIDIA to have the necessary access"
  ([nouveau](https://nouveau.freedesktop.org/); its [power-management table](https://nouveau.freedesktop.org/PowerManagement.html)
  lists clock changes for these generations as "TODO").
- **Turing and newer:** these cards carry a GSP, a processor that runs NVIDIA's own firmware, and nouveau hands power
  management to it.

NVIDIA's own open-source kernel modules work the same way: they need the GSP firmware and run only on "Turing or
later" GPUs ([open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules)). A P106 has no GSP.

## What works instead

| Way | What it does on a P106 | Status |
|---|---|---|
| Keep-alive kernel (`GGML_CUDA_KEEPALIVE`) | a tiny kernel keeps each card looking busy, so it holds 1,847 MHz or more | in Ornith's and Flash-Next's presets; costs Flint's prompts 16.5 %, so Flint goes without |
| Clock offset through NVML | raises the boost clocks; the 1,354 MHz floor stays | being measured on one card (Q133, Q135) |
| "Prefer maximum performance" mode through NVML | the card stays awake, but only at 1,354 MHz | measured ([R188](../../data/2026-11-17_r186_r188_clocks/README.md)) |
| Locked clocks (`nvidia-smi -lgc`), CUDA clock override (`-cc`) | refused: "not supported" | measured (R188) |
| The driver's "rated clock" control, asked directly | its headers define a lock at a "turbo boost" point that NVIDIA's tools offer only on Ampere and newer | the tool is built ([`bench/rated_clock`](../../../bench/rated_clock/README.md)); a read-only status first, then one try on one card with someone at the console. Est. refused |

If the last row works, the cards would hold their top clock with no keep-alive, and a clock offset would raise exactly
the clocks the work runs at.
