# Q92: the screened output head in the engine, correctness checks (2026-10-24)

The check scripts behind registry row Q92 ([experiments.md](../../../quants/experiments.md)) and "The screened head in
the engine" in [quants/gptq_head/README.md](../../../quants/gptq_head/README.md), which gives the results: on G3 (screen
in the file, with Miner Llama's Pascal kernels forced and with llama.cpp's own) and on G3 and UD-Q4_K_XL with the
screen from a 173 MB side file, all 262,144 candidate logits are bit-identical to the full head's and greedy and seeded
tokens are identical; `llama-server` with backend sampling answered 24 requests byte-identically with the screen
requested and not, switching it on for the 16 greedy and sampled requests and off for the 8 that ask for token
probabilities. The prompts both checks use are [quants/gptq_head/prompts.txt](../../../quants/gptq_head/prompts.txt)
(8 prompts of prose, code, maths, chat and SQL written for this project); the server check is
[quants/gptq_head/screen_server_ab.sh](../../../quants/gptq_head/screen_server_ab.sh).

| File | What |
|---|---|
| `run/make_g3_screen.sh` | G3 with the screen tensor added (`gptq_head.py screen`), checked byte for byte (`verifyscreen`) |
| `run/screen_check.sh` | `llama-screen-check` on G3 + screen (Pascal kernels forced, then llama.cpp's own) and on G3 with the side file |

Made on 2026-10-24 on two RTX 3090s with an sm_86 build of the fork branch `pascal-p106-v2-screenhead` (`024f13df4`),
run from the repository root:

```bash
export MODELS=... BIN=<build>/bin GPUS=... PYTHONPATH=<llama.cpp>/gguf-py
python quants/gptq_head/gptq_head.py quant flint-BF16.gguf screen-gptq-search-mix.q4_0 --mode gptq --hessian H-mix.npy \
  --scale search --type q4_0                     # H-mix as in docs/data/2026-10-24_q56_gptq_head/run/after_dump.sh
S=screen-gptq-search-mix.q4_0 bash docs/data/2026-10-24_q92_screen_head/run/make_g3_screen.sh
python quants/gptq_head/gptq_head.py screenfile $MODELS/flint-q73-G3.gguf screen-gptq-search-mix.q4_0 \
  $MODELS/flint-output-screen-gptq-q4_0.gguf --top-m 256
bash docs/data/2026-10-24_q92_screen_head/run/screen_check.sh       # and the same with UD-Q4_K_XL + the side file
docker run --rm --gpus "\"device=$GPUS\"" -e LLAMA_NO_MOE_DOWN_RESCALE=1 -e BIN=$BIN -v $PWD:$PWD -v $MODELS:$MODELS \
  -v $BIN:$BIN nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  bash $PWD/quants/gptq_head/screen_server_ab.sh $MODELS/flint-q73-G3-screen.gguf $PWD/server_ab
# the screened head's ops against the CPU backend, with GGML_CUDA_MMVQ_PASCAL=2 (Pascal kernels forced) and =0
$BIN/test-backend-ops -o MUL_MAT_ID -b CUDA0 -p "n_mats=32768,n_used=256"
$BIN/test-backend-ops -o TOP_K -b CUDA0 -p "ne=\[150144,1,1,1\],k=256"
$BIN/test-backend-ops -o SET_ROWS -b CUDA0; $BIN/test-backend-ops -o FILL -b CUDA0
```

The scripts are this project's (MIT).
