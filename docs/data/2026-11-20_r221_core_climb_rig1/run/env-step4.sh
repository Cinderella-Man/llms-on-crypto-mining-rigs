# The first rig's core soak ladder (step 4), from R221's core climb: soak_chain.sh sources this in the kit folder
# (/var/tmp/oc/mem_offset-kit-20261024, staged by R221 with env.sh and the three rungs) on a fresh boot.
# Start with: ENV_FILE=./env-step4.sh setsid nohup bash soak_chain.sh > /var/tmp/oc-step4.console 2>&1 < /dev/null &
. ./env.sh
K=/var/tmp/oc/mem_offset-kit-20261024
export WORK=/var/tmp/oc-step4
export PROFILE=$K/step4-rung1.json
export LADDER="$K/step4-rung2.json $K/step4-rung3.json"
export FAILURES=/var/tmp/r221/failures/rung1.json   # steps 1 to 3's failures: R217's and R218's memory, R221's core
export MODEL=/mnt/models/ornith/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf   # Ornith-1.5-122B Pascal N2P GGUF, the whole-memory check
export SETTINGS=$K/ornith-v4.env        # release/ornith-v4/settings.env, as in R218
export OUT=/var/tmp/oc-step4-model
export MINUTES=30 MODEL_MINUTES=30 END_BY=21:50
