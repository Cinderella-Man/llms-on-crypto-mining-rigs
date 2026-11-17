# R200 sitting on rig 2: source this in the kit folder (/var/tmp/oc/mem_offset-kit-20261024).
# Same as C871's rig-1 env.sh except the check model, which is chosen from what is on rig 2's drives.
export WORK=/var/tmp/climb
export BIN=/opt/llama-builds/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin
export CHECK_MODEL=/mnt/models/check/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf   # 2.02 GB, md5 c8e8b8bc7890057906d99643dbf8063b; relayed 19:06 (no 2-4.5 GB model was on rig 2)
export MEM_STEPS="200 400 600 800 900 1000 1100 1200 1300 1400"   # the owner, 2026-11-17 ~18:50: the full climb to +1,400
