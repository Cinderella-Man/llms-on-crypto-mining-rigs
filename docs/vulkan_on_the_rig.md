# llama.cpp's Vulkan backend on the rig

llama.cpp can drive NVIDIA cards through two backends: CUDA, used for every result in this repository so far, and
Vulkan, which runs through the graphics driver. Both read the same GGUF files. This page covers building the Vulkan
backend for the rig, what the rig needs to run it, whether Flint's graph runs on it, and how to measure it against
CUDA on the same cards.

Source references are to llama.cpp commit `d77256999` (2026-10-22), the commit of the CUDA builds measured here.
Paths are relative to `ggml/src/ggml-vulkan/` unless they name another directory.

## Why it is worth a test

Generation on a P106-090 is limited by how fast its 5 SMs can unpack weights, not by memory bandwidth
([single-card calibration](benchmarks/2026-10-22-single-card-calibration.md)). Vulkan has its own matrix-vector
shaders with a different instruction mix from CUDA's. Owners of other Pascal cards have measured Vulkan generating as
fast as or faster than CUDA, including a same-machine A/B on a GTX 1060, which uses the same GP106 chip: +11 % on a
dense 7B model and +47 % on a small MoE model, measured before llama.cpp enabled CUDA graphs on Pascal
([community survey](research/notes/quant_speed_d_community.md), section 4). Nobody has published a Vulkan result for a
5-SM card or for a model split across five or more Pascal cards.

The risk specific to this rig is the host: the Vulkan backend records every operation of every token on the CPU,
which here is a 2-core 1.8 GHz Celeron ([below](#risk-the-2-core-host-records-every-command)).

## The build

[scripts/build_llamacpp_vulkan_docker.sh](../scripts/build_llamacpp_vulkan_docker.sh) builds on any x86-64 Linux PC
with Docker, in the same way as the CUDA script:

```bash
scripts/build_llamacpp_vulkan_docker.sh d77256999
# optional second build with the rescale switch (see "Vulkan settings worth trying"):
PATCHES=patches/moe-down-rescale-env.diff scripts/build_llamacpp_vulkan_docker.sh d77256999
```

| Item | Value |
|---|---|
| Container | Ubuntu 24.04 with GCC 14, so the binaries match the rig's glibc and libstdc++ (highest symbols used: `GLIBC_2.38`, `GLIBCXX_3.4.32`) |
| Vulkan headers and shader compiler | LunarG Vulkan SDK 1.4.341.0 (downloaded once, ~290 MB, kept in the work directory). Ubuntu 24.04's own `glslc` (shaderc 2023.8) cannot compile `GL_EXT_integer_dot_product`; without it llama.cpp leaves out the shaders that use the cards' 8-bit dot product (`dp4a`) and reports `int dot: 0`. The script stops if that extension is missing |
| CPU code | No AVX, AVX2, FMA, F16C or AVX-512. The script counts VEX/EVEX-encoded instructions in every binary and stops if any is found; all nine binaries have none |
| Binaries | `llama-bench`, `llama-server`, `llama-cli`, `llama-completion`, `llama-perplexity`, `llama-batched-bench`, `llama-gguf`, `test-backend-ops`, `test-export-graph-ops`; ggml and all shaders are linked in |
| Run-time dependencies | `libvulkan.so.1` (the Vulkan loader), the NVIDIA driver's Vulkan driver, `libgomp`, `libstdc++` |
| Output | `llama.cpp-<date>-d77256999-vulkan.tar.gz`, 145 MB, 412 MB unpacked, with `bin/`, `lib/libvulkan.so.1` (Ubuntu 24.04's loader, for rigs without it) and `BUILDINFO.txt` |
| Build time | ~4 minutes on a 16-core PC including the SDK download, ~2 minutes for a rebuild with a warm ccache |

## What the rig needs

- **The NVIDIA driver's Vulkan driver.** The 580 driver includes it: the package `libnvidia-gl-580` installs
  `/usr/share/vulkan/icd.d/nvidia_icd.json`, which points the Vulkan loader at the driver.
- **The Vulkan loader** `libvulkan.so.1` (package `libvulkan1`). A desktop install usually has it, because GNOME's
  libraries depend on it; the NVIDIA packages don't. If it is missing, either install it
  (`sudo apt install libvulkan1`) or run the binaries with `LD_LIBRARY_PATH=<build>/lib`, which uses the copy in the
  tarball.
- **Nothing else.** No Vulkan SDK, shader compiler or CUDA toolkit is needed for the Vulkan build.

Read-only checks:

```bash
ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null     # nvidia_icd.json: NVIDIA's Vulkan driver
dpkg -S /usr/share/vulkan/icd.d/nvidia_icd.json                # expect libnvidia-gl-580
dpkg -l libvulkan1 mesa-vulkan-drivers vulkan-tools 2>/dev/null | grep '^ii'
ldconfig -p | grep 'libvulkan.so.1'                            # loader installed?
ls /usr/share/vulkan/implicit_layer.d/ 2>/dev/null             # VkLayer_MESA_device_select.json reorders devices
grep -H . /sys/class/drm/card*/device/boot_vga 2>/dev/null     # 1 = the GPU that drives the display
ldd /opt/llama-builds/llama.cpp-20261022-d77256999-vulkan/bin/llama-bench | grep vulkan   # "not found" = no loader
```

`vulkaninfo --summary` lists every Vulkan device, but only if `vulkan-tools` is installed.

## Installing the build

Copy the tarball to the rig (145 MB, e.g. `scp ~/llama-builds/llama.cpp-20261022-d77256999-vulkan.tar.gz rig:/tmp/`)
and unpack it next to the CUDA builds:

```bash
sudo tar xzf /tmp/llama.cpp-20261022-d77256999-vulkan.tar.gz -C /opt/llama-builds/
rm /tmp/llama.cpp-20261022-d77256999-vulkan.tar.gz
NODEVICE_SELECT=1 /opt/llama-builds/llama.cpp-20261022-d77256999-vulkan/bin/llama-cli --list-devices
```

The list should show nine `NVIDIA P106-090` devices. Each card compiles the shaders it needs the first time it uses
them, during the first tokens; `llama-bench`'s warm-up run absorbs this, so it does not enter the measured speed.

## Device numbering

The Vulkan backend numbers devices differently from CUDA, and two things on a desktop install can put another GPU
into the list:

- If Mesa's Vulkan drivers (`mesa-vulkan-drivers`) are installed, the Celeron's Intel HD Graphics 610 is a Vulkan
  device too. llama.cpp uses every discrete *and integrated* GPU by default (`ggml-vulkan.cpp:5314-5323`), so without
  a device list it would put model layers on the Intel GPU.
- Mesa's device-select layer moves the GPU that drives the display to the front of the list. On a PC with an AMD
  integrated GPU and an NVIDIA card, `llama-cli --list-devices` showed the integrated GPU as `Vulkan0`; with
  `NODEVICE_SELECT=1`, which switches the layer off, the Vulkan loader's own order applies, with discrete GPUs first.

So for every Vulkan run:

- set `NODEVICE_SELECT=1`;
- choose the cards with `GGML_VK_VISIBLE_DEVICES`, which takes indices into that list
  (`ggml-vulkan.cpp:5290-5306`); [scripts/bench.sh](../scripts/bench.sh) sets it from `GPU_SETS`, like
  `CUDA_VISIBLE_DEVICES`;
- check which physical cards ran in the `smi.csv` that `bench.sh` writes (utilisation and power per CUDA index), and
  use the same cards for the CUDA run.

## Which of Flint's operations run on Vulkan

All of them. The table maps each part of Flint's graph (`src/models/flint.cpp`, `src/llama-graph.cpp`) to the
condition in `ggml_backend_vk_device_supports_op` (`ggml-vulkan.cpp:15426`) that accepts it.

| Part of Flint | ggml operations | Accepted at `ggml-vulkan.cpp` | Notes |
|---|---|---|---|
| Weight matrix-vector products: attention, shared expert, dense layers, output head (Q4_1); router (F32) | `MUL_MAT` | 15503-15583 | For one token on NVIDIA the integer-dot path is used only when the inner dimension k is above 4096 (6612-6648). In Flint only the dense layers' down-projection (k = 6144) qualifies; every other matrix uses the float path, which dequantizes weights and reads F32 activations (6739, 7751-7772) |
| Routed experts, 6 of 256 | `MUL_MAT_ID` | 15503-15583, tile check 15515-15518 | The down-projection is fused with the routing-weight multiply (`MUL_MAT_ID_MUL`, 14473-14477) |
| RMS norms, including the per-head q/k norms | `RMS_NORM`, `MUL` | 15772, 15779-15785 | Fused with the following multiply, RoPE and the K-cache write (14478-14530) |
| RoPE on the 35 sliding-window layers | `ROPE` (NeoX) | 15764-15765 | |
| KV-cache writes | `SET_ROWS` (F16) | 15671-15693 | |
| Attention with learned sinks | `FLASH_ATTN_EXT` with a sink tensor | 15584-15631 | Sinks must be F32, which Flint's are (15592). Without cooperative matrices (Pascal has none) the shader needs subgroup shuffle and vote (15626). Sink handling: `vulkan-shaders/flash_attn.comp:711-725`, `flash_attn_split_k_reduce.comp:79-110` |
| Attention with flash attention off | `SOFT_MAX` with sinks, `MUL_MAT` | 15868-15870 | Sinks in `vulkan-shaders/soft_max.comp:71,153` |
| Sliding-window attention (window 513) | none of its own: a separate KV cache and an F16 mask built on the host | 15601 (F16 mask) | |
| Sigmoid top-k routing with normalisation | `UNARY` (sigmoid), `ARGSORT`, `GET_ROWS`, `SUM_ROWS`, `CLAMP`, `DIV` | 15470, 15812-15823, 15632-15668, 15876-15878, 15796-15799, 15779-15785 | Not fused. Vulkan's fused sigmoid top-k kernel (`TOPK_MOE_SIGMOID_NORM_BIAS`, `ggml-vulkan-types.h:492-495`) requires an expert-bias add, and Flint has no expert bias. This costs six small dispatches per MoE layer; CUDA fuses the same pattern into one kernel |
| SwiGLU | `GLU` | 15487-15502 | Not fused with the gate/up products, as it is on CUDA |
| MoE down-projection rescale (`build_amax_src1_f16`, `src/llama-graph.cpp:1670-1705`, applied at 2513-2524) | `UNARY` abs, `POOL_1D`, `SCALE` ×3, `LOG`, `UNARY` ceil, `CLAMP`, `UNARY` exp, `DIV`, `MUL` | 15473, 15958-15959, 15860-15861, 15807-15811, 15477, 15796-15799, 15459, 15779-15785 | `POOL_1D` needs a contiguous F32 input, which the abs output is. Its shader (`vulkan-shaders/pool1d.comp`) runs one thread per output value and loops over all 512 inputs, the same design as CUDA's `ggml/src/ggml-cuda/pool1d.cu`; in a per-operation profile it is the slowest of the small operations |
| Residual adds, sum over the 6 experts | `ADD` | 15779-15785 | Fused into multi-adds and into mat-vec + add (14445-14461) |
| Token embedding (F32) | `GET_ROWS` | runs on the CPU | Same as with CUDA: the table stays in host memory |

Unary operations are accepted for F32 or F16 input and output of the same type (15481-15483).

### Checked on another PC

`test-export-graph-ops` writes every distinct operation of a model's prompt and generation graphs, with its exact
shapes and parameters, to a file that `test-backend-ops` reads with `--test-file`. It loads the model on the CPU
only, so any PC with the Vulkan build can run it:

```bash
test-export-graph-ops -m flint-int4-q4_1.gguf -c 4096 -fa on  -o flint-ops-fa-on.txt
test-export-graph-ops -m flint-int4-q4_1.gguf -c 4096 -fa off -o flint-ops-fa-off.txt
```

For the makers' Q4_1 file the generation graph has 3,576 nodes (1,105 of them views), and there are 100 distinct
operations with flash attention on and 114 with it off. On an NVIDIA RTX 5070 laptop GPU with the features Pascal
lacks switched off (`GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_F16=1
GGML_VK_DISABLE_BFLOAT16=1`, after which the device reports `fp16: 0 | bf16: 0 | int dot: 1 | matrix cores: none`):

- `test-backend-ops support` accepts all 100 and all 114 operations;
- `test-backend-ops test` matches the CPU backend on all of them (and on the flash-attention set again with
  `GGML_VK_FORCE_MMVQ=1`), except two `LOG` cases: the test feeds random numbers in [-1, 1], both backends return NaN
  for the negative ones and the check flags the differing NaN sign. In Flint's graph `LOG` only sees non-negative
  values, and llama.cpp's own `LOG` tests pass;
- the whole model on that GPU (`-ngl 99`, with `GGML_VK_ALLOW_SYSMEM_FALLBACK=1` placing what does not fit in its
  8 GB in system memory) ran as two graph splits: the token-embedding lookup on the CPU and everything else on Vulkan.
  24 greedy tokens were identical to the CPU backend's. Perplexity on wikitext-2, one token at a time (`-b 1`, the
  generation path), matched the CPU backend:

  | Context | Vulkan | CPU |
  |---|---|---|
  | 2 × 512 tokens (inside the 513-token window) | 22.63 | 22.55 |
  | 1 × 1,024 tokens (sliding window in effect) | 8.311 | 8.305 |

A per-operation profile of one generated token shows ~38 Vulkan dispatches per MoE layer, of which 11 are the
rescale and 6 the routing. For the whole model that is about 1,800 dispatches per token, close to the ~1,900 kernel
launches of the CUDA graphs build.

### Checking on the rig

Device support depends on the driver, so repeat the support check on a P106-090 before the speed runs. Copy the two
operation files to the rig; the check loads no model and takes seconds:

```bash
VK=/opt/llama-builds/llama.cpp-20261022-d77256999-vulkan/bin
NODEVICE_SELECT=1 GGML_VK_VISIBLE_DEVICES=0 $VK/test-backend-ops support -b Vulkan0 --test-file flint-ops-fa-on.txt  | grep -c "NOT SUPPORTED"
NODEVICE_SELECT=1 GGML_VK_VISIBLE_DEVICES=0 $VK/test-backend-ops support -b Vulkan0 --test-file flint-ops-fa-off.txt | grep -c "NOT SUPPORTED"
```

Both counts should be 0. The device line it prints first (`ggml_vulkan: 0 = NVIDIA P106-090 ... | int dot: 1 | ...`)
shows whether the driver offers the integer dot product. If `FLASH_ATTN_EXT` is not supported, use `-fa off` for the
Vulkan runs. `test-backend-ops perf -b Vulkan0 --test-file flint-ops-fa-on.txt -o MUL_MAT` times Flint's own
matrix shapes on one card.

## The A/B test

Use the same commit, model file, cards and `llama-bench` arguments for both backends. The commands below run on the
rig from a copy of this repository's `scripts/`; through SSH, put the variables in the remote command:
`ssh rig "NODEVICE_SELECT=1 BIN=... MODEL=... TAG=... GPU_SETS=... BENCH_ARGS='...' bash -s" < scripts/bench.sh`.
Results land in `/tmp/bench/<TAG>/`.

```bash
CUDA=/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61/bin   # CUDA build with CUDA graphs on Pascal
VK=/opt/llama-builds/llama.cpp-20261022-d77256999-vulkan/bin
export NODEVICE_SELECT=1
```

### One card

Any model that fits on one card; Llama-2-7B Q4_0 connects the result to the
[single-card calibration](benchmarks/2026-10-22-single-card-calibration.md) and to other GPUs' scoreboard entries. In
Llama-2-7B the down-projection has k = 11,008 and uses the integer-dot path by default, the other matrices
(k = 4,096) the float path, so the two `MMVQ` variants separate the two shader families.

```bash
MODEL=/mnt/models/llama-2-7b.Q4_0.gguf
ARGS="-ngl 99 -fa 0,1 -p 512 -n 128 -r 5"
BIN=$CUDA MODEL=$MODEL TAG=vk-ab1-cuda GPU_SETS=0 BENCH_ARGS="$ARGS" bash scripts/bench.sh
BIN=$VK   MODEL=$MODEL TAG=vk-ab1-vk   GPU_SETS=0 BENCH_ARGS="$ARGS" bash scripts/bench.sh
GGML_VK_FORCE_MMVQ=1   BIN=$VK MODEL=$MODEL TAG=vk-ab1-vk-mmvq   GPU_SETS=0 BENCH_ARGS="$ARGS" bash scripts/bench.sh
GGML_VK_DISABLE_MMVQ=1 BIN=$VK MODEL=$MODEL TAG=vk-ab1-vk-nommvq GPU_SETS=0 BENCH_ARGS="$ARGS" bash scripts/bench.sh
BIN=$CUDA MODEL=$MODEL TAG=vk-ab1-cuda2 GPU_SETS=0 BENCH_ARGS="$ARGS" bash scripts/bench.sh   # drift check
```

`GPU_SETS=0` is CUDA device 0 for the CUDA build and Vulkan device 0 for the Vulkan build; confirm in `smi.csv` that
both ran on the same card.

### Flint on five cards

```bash
MODEL=/mnt/models/RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf
ARGS="-ngl 99 -fa on -p 512 -n 128 -r 3"
TG="-ngl 99 -fa on -p 0 -n 128 -r 3"
BIN=$CUDA MODEL=$MODEL TAG=vk-abf-cuda GPU_SETS=0,1,2,3,4 BENCH_ARGS="$ARGS" bash scripts/bench.sh
BIN=$VK   MODEL=$MODEL TAG=vk-abf-vk   GPU_SETS=0,1,2,3,4 BENCH_ARGS="$ARGS" bash scripts/bench.sh
GGML_VK_FORCE_MMVQ=1      BIN=$VK MODEL=$MODEL TAG=vk-abf-vk-mmvq   GPU_SETS=0,1,2,3,4 BENCH_ARGS="$TG" bash scripts/bench.sh
GGML_VK_DISABLE_FUSION=1  BIN=$VK MODEL=$MODEL TAG=vk-abf-vk-nofuse GPU_SETS=0,1,2,3,4 BENCH_ARGS="$TG" bash scripts/bench.sh
# optional, with the second build (generation only, see LLAMA_NO_MOE_DOWN_RESCALE below):
VKR=/opt/llama-builds/llama.cpp-20261022-d77256999-vulkan-moe-down-rescale-env/bin
LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$VKR MODEL=$MODEL TAG=vk-abf-vk-norescale GPU_SETS=0,1,2,3,4 BENCH_ARGS="$TG" bash scripts/bench.sh
```

Each run loads the 26 GB file again (a few minutes over the x1 links). If the Vulkan build runs out of memory on
five cards, use six for both backends; the number of cards does not change Flint's generation speed
([Flint benchmark](benchmarks/2026-10-22-flint.md)). The CUDA reference from that benchmark is 27.92 ± 0.04 tokens/s
(tg128, five cards).

While a Vulkan generation run is going, watch the host from a second shell:

```bash
top -H -b -d 2 -n 10 -p "$(pgrep -n llama-bench)" | grep -A3 PID
```

### Vulkan settings worth trying

| Variable | What it does | Why it may matter here |
|---|---|---|
| `GGML_VK_VISIBLE_DEVICES=0,1,...` | Devices to use, as indices into Vulkan's list (5290-5306) | Required, see [Device numbering](#device-numbering) |
| `GGML_VK_FORCE_MMVQ=1` | Quantized matrix-vector products quantize the activations to Q8_1 and use 8-bit integer dot products (4925-4930, 6612-6648) | The card is instruction-bound and `dp4a` runs at four times the FP32 rate; by default Flint uses the float path for almost every matrix |
| `GGML_VK_DISABLE_MMVQ=1` | Float path for every matrix-vector product | The opposite end, for the single-card comparison |
| `GGML_VK_DISABLE_FUSION=1` | Turns off all Vulkan operation fusions (4915) | Shows what the fusions are worth; expected to be slower |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | Keeps llama.cpp's node order instead of grouping independent nodes to save barriers (4036, 14786) | Diagnostic |
| `GGML_VK_MAX_NODES_PER_SUBMIT=N` | Hands recorded work to the GPU at least every N nodes, default 100 (4271-4277) | Changes how CPU recording and GPU execution overlap. In generation a FLOP-based rule (14360-14378) usually submits first, so expect little effect |
| `GGML_VK_PERF_LOGGER=1` | Prints the GPU time of every operation for every graph (5271) | Profiling only: it waits for the GPU after every graph, so don't use it for speed numbers |
| `LLAMA_NO_MOE_DOWN_RESCALE=1` | Skips the 11-operation rescale per MoE layer; needs the build with `patches/moe-down-rescale-env.diff` | The rescale exists because Q8_1 activation blocks store their sums as FP16, which Flint's activations overflow. Vulkan's Q8_1 blocks are FP16 too (`vulkan-shaders/types.glsl:242-246`), but in generation the default float path never builds them, so generation-only runs (`-p 0`) are valid without the rescale: Flint's one-token-at-a-time perplexity over 2 × 512 tokens was 22.43 without it and 22.63 with it, with no NaN. Prompt processing and `GGML_VK_FORCE_MMVQ=1` do build them, so never combine those with the switch |

The settings are read when the backend starts, so put them in front of the command, as above.

### Reading the result

- Compare tg128 at the same `-fa` setting. Adopt Vulkan for Flint only if it is clearly faster than the CUDA graphs
  build (≥ 5 %) and stays so with a model of another format, such as the UD-mirror file.
- Prompt processing is expected to be slower on Vulkan; owners of other Pascal cards report 2–5× slower.
- Check the device line for `int dot: 1`. With `int dot: 0` the `MMVQ` variants make no difference.

## Risk: the 2-core host records every command

With CUDA graphs, the CPU launches one recorded graph per card per token. The Vulkan backend has no equivalent: for
every token it walks the graph again and records each dispatch into command buffers (pipeline and descriptor binds,
push constants, the dispatch, and a barrier before a node that reads an earlier node's output;
`ggml_backend_vk_graph_compute`, 14282). It hands the recorded work to the GPU in batches while it keeps recording
(14360-14378), so recording is hidden as long as it is faster than the GPU's work.

The budget: at the CUDA build's 27.9 tokens/s a token takes 35.8 ms. Spread over ~1,800 dispatches that is ~20 µs of
GPU time per dispatch, and the cards run one after another in a layer split, so one CPU thread has to record each
card's ~360 dispatches within that card's ~7 ms. A Celeron 3865U has no turbo and runs at 1.8 GHz; if recording one
dispatch (llama.cpp's own work plus the driver's) costs more than ~20 µs there, the host sets the speed. Signs of it:

- the `llama-bench` main thread close to 100 % in `top -H` while the GPU columns in `smi.csv` show low utilisation;
- with `GGML_VK_PERF_LOGGER=1` in a short `llama-completion` run, the summed `Total time` of the five cards' graphs
  for one token well below the wall time per token.

Copies between cards go through host memory with either backend, since the cards have no peer-to-peer access.

## The CUDA side: two switches that cost nothing

Two run-time switches of the CUDA backend can be measured in the same session. Both exist at `d77256999`
(`ggml/src/ggml-cuda/ggml-cuda.cu`):

| Variable | What it does | Conditions |
|---|---|---|
| `GGML_CUDA_GRAPH_OPT=1` | Inside a CUDA graph, finds a node named `attn_norm` with exactly three single-row consumers (the Q, K and V projections) and runs the three branches on concurrent streams up to the attention (5020-5023, 5088-5100, 5073) | Only while CUDA graphs are in use (returns at 5032 otherwise), so on Pascal only with the CUDA-graphs build; only in generation (one row); the value must be exactly `1`. A file with a fused QKV tensor has one consumer and gains nothing |
| `GGML_CUDA_DISABLE_FUSION=1` | Turns off every CUDA fusion: gate/up products with the GLU, shared + routed experts, top-k routing, RMS norm + multiply (+ add, RoPE, cache write), multi-add, the MoE weighted sum, product + bias (171-174, 3777-3781, 4901) | Any non-zero integer. On a P40, turning fusion off made Q4_0 4.7 % faster and Q4_K_M 1.2 % slower ([llama.cpp #30334](https://github.com/ggml-org/llama.cpp/issues/30334)) |

[scripts/env_ab.sh](../scripts/env_ab.sh) runs `llama-bench` with neither, each, both, and neither again:

```bash
BIN=/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61/bin \
MODEL=/mnt/models/RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf TAG=flint-cuda-env bash scripts/env_ab.sh
```
