# Vulkan preparation: Flint's operation lists and one token's dispatches (2026-10-23)

[docs/vulkan_on_the_rig.md](../../vulkan_on_the_rig.md) checks llama.cpp's Vulkan backend for Flint before the rig
runs it: which operations Flint's graphs need, whether the backend supports them on a Pascal-class device, and how
many dispatches one generated token takes. This folder holds the two operation lists that check uses and the
per-operation profile of one token.

| File | What |
|---|---|
| `flint-q4_1-ops-faon.txt` | every distinct operation of the makers' Flint Q4_1 prompt (512-token) and generation (one-token) graphs at 4,096 tokens of context, flash attention on: 100 lines in `test-export-graph-ops` format, for `test-backend-ops --test-file` |
| `flint-q4_1-ops-faoff.txt` | the same with flash attention off: 114 lines |
| `vulkan-perf-logger-decode.txt` | `GGML_VK_PERF_LOGGER=1` for one generated token with 11 MoE layers and the output head on the Vulkan device (`-ngl 12`): count, mean µs and total per operation and shape; 416 dispatches, 3.13 ms |

**The export needs about 26 GB of RAM.** `test-export-graph-ops` loads the model on the CPU: the flash-attention-on
export of the 26 GB file peaked at 25.7 GB of resident memory (25,742,528 KB), so the 4 GB rig cannot make these lists
and they were made on another PC. Each graph has 3,576 nodes, 1,105 of them views, and 50 distinct operations.

The profile shows where a token's Vulkan dispatches go: per MoE layer, out of ~38, the rescale of the expert
down-projection input takes 11 (`ABS`, `POOL_1D`, three `SCALE`, `LOG`, `CEIL`, `CLAMP`, `EXP`, `DIV`, `MUL`) and the
routing 6 (`SIGMOID`, `ARGSORT`, `GET_ROWS`, `SUM_ROWS`, `CLAMP`, `DIV`).

Made on 2026-10-23 on a PC with a newer NVIDIA GPU (not the rig), with upstream llama.cpp `d77256999` built by
[scripts/build_llamacpp_vulkan_docker.sh](../../../scripts/build_llamacpp_vulkan_docker.sh), the makers' Q4_1 file
([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf), `flint-int4-q4_1.gguf`):

```bash
VK=<vulkan build>/bin
M=flint-int4-q4_1.gguf
/usr/bin/time -f "%e s, %M KB maxrss" $VK/test-export-graph-ops -m $M -c 4096 -fa on  -o flint-q4_1-ops-faon.txt
$VK/test-export-graph-ops -m $M -c 4096 -fa off -o flint-q4_1-ops-faoff.txt
# the profile: a short generation run of this form, with the GPU's features that Pascal lacks switched off
NODEVICE_SELECT=1 GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_BFLOAT16=1 GGML_VK_PERF_LOGGER=1 \
    $VK/llama-completion -m $M -ngl 12 -n 4 -p "Hello" -no-cnv
```

The perf logger prints one block per graph evaluation; `vulkan-perf-logger-decode.txt` is the block of one generation
step. Device
timings on that GPU say nothing about a P106-090's speed; the counts and the share per operation carry over. The
test-backend-ops and perplexity logs from that PC are summarized in the Vulkan doc and not published.
