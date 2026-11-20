# R217 on rig 1 (2026-11-19): source this in the kit folder (/var/tmp/oc/mem_offset-kit-20261024).
# The kit settings every step loads; each step sets its own WORK (env-chain.sh for the ladders).
export WORK=/var/tmp/r217
export BIN=/opt/llama-builds/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin   # R200's engine
export CHECK_MODEL=/mnt/models/check/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf          # R200's check model, 2.02 GB, md5 c8e8b8bc7890057906d99643dbf8063b
export MEM_STEPS="200 400 600 800 900 1000 1100 1200 1300 1400"                         # to +1,400, as R200
# CORE_PEAK_MAX=0 was for step 1 only (R217, env-r217-step1.sh); steps 2 to 5 use the default (2250 MHz), which the core climb needs
