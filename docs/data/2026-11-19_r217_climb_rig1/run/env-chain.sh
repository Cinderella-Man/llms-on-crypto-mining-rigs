# The first rig's memory soak ladder (R218), from R217's climb: soak_chain.sh sources this in the kit folder
# (/var/tmp/oc/mem_offset-kit-20261024, staged by R217 with env.sh) on a fresh boot.
# Start with: setsid nohup bash soak_chain.sh > /var/tmp/r218.console 2>&1 < /dev/null &
. ./env.sh
K=/var/tmp/oc/mem_offset-kit-20261024
export WORK=/var/tmp/r218
export PROFILE=$K/rung1.json
export LADDER="$K/rung2.json $K/rung3.json"
export FAILURES=$K/failures.json        # the rig's failures: Q133, R193, R194 and R217's climb
export MODEL=/mnt/models/ornith/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf   # Ornith-1.5-122B Pascal N2P GGUF, the whole-memory check
export SETTINGS=$K/ornith-v4.env        # release/ornith-v4/settings.env, as in R201
export OUT=/var/tmp/r218-ornith
export MINUTES=30 MODEL_MINUTES=30 END_BY=21:50
