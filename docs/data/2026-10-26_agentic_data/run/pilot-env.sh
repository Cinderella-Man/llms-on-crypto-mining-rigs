# The Elixir pilot's settings (2026-10-26, 14:36-15:57, two V100s): source this, then run bench/agentic_data/run.sh.
#   BENCH=<this repository's bench/ folder> WORK=<out> MODEL=<flint-Q8_0.gguf> HF_DIR=<RecursiveSI/flint small files> \
#   Q51=<Q51 prompts.jsonl> LLAMA_SERVER=<llama-server, CUDA build> MOUNTS="-v <folder>:<folder>:ro ..." \
#   . pilot-env.sh && bash "$BENCH/agentic_data/run.sh"
# Flint Q8_0 (approved writer flint-q8_0) in llama-server, as the Python pilot; 8 slots of 64k tokens (the Python
# pilot found Flint needs 64k contexts and 20-32k tokens per turn). The Python texts of that day (now the retired
# texts_v1.json) are kept as they are; the texts step writes only the Elixir ones. MOUNTS lists every folder the
# server container reads (the model, the binary). The pilot's stage 1 ran with 24 turns per session (run.sh's
# default); the file was changed to MAX_TURNS=48 during the pilot, the setting the long run used.
export WORK=${WORK:?output folder}
export HF_DIR=${HF_DIR:?RecursiveSI/flint small files}
export Q51=${Q51:?Q51 prompts.jsonl}
export MODEL=${MODEL:?Flint Q8_0 GGUF}
export LLAMA_SERVER=${LLAMA_SERVER:?llama-server binary}
export MOUNTS=${MOUNTS:?docker -v options for the model and binary folders}
export GPUS=3,4 PORT=18641 NP=8 CTX_SLOT=65536 MAX_GEN=32768 TIME_CAP=45
export TEXTS=$WORK/texts.json TEXTS_FROM=${BENCH:?the bench/ folder of this repository}/agentic_data/texts_v1.json
export EX_MODULES=option_parser,version,uri,base
export EX_NOTICE=$BENCH/agentic_data/notices/elixir-1.18.4-NOTICE
export PER_MODULE=2 IMPLEMENT=1 TOOLS=mixed MAX_TURNS=48
export STEPS="ex-image server texts ex-workspaces s1 s2 gate export stop"
