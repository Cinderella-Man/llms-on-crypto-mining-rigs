# R201 on rig 2: the soak ladder's settings. soak_chain.sh sources this in the kit folder
# (/var/tmp/oc/mem_offset-kit-20261024, left on rig 2 by R200 with env.sh and ornith-v4.env).
. ./env.sh
K=/var/tmp/oc/mem_offset-kit-20261024
export WORK=/var/tmp/r201
export PROFILE=$K/r201-rung1.json
export LADDER="$K/r201-rung2.json $K/r201-rung3.json $K/r201-rung4.json"
export MODEL=/mnt/models2/ornith/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
export SETTINGS=$K/ornith-v4.env
export OUT=/var/tmp/r201-ornith
export MINUTES=30 MODEL_MINUTES=30 END_BY=21:50
