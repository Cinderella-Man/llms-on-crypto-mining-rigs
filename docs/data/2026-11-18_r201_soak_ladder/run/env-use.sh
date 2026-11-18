# R201's second boot on rig 2: the use profile held for 30 min, then Ornith with it for 30 min (no ladder).
# Start with: ENV_FILE=./env-use.sh setsid nohup bash soak_chain.sh > /var/tmp/r201b/chain.console 2>&1 < /dev/null &
. ./env.sh
K=/var/tmp/oc/mem_offset-kit-20261024
export WORK=/var/tmp/r201b
export PROFILE=$K/r201-use.json     # made from R201's soaked.json: each card at most two steps under its lowest failure
export LADDER=
export MODEL=/mnt/models2/ornith/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
export SETTINGS=$K/ornith-v4.env
export OUT=/var/tmp/r201b-ornith
export MINUTES=30 MODEL_MINUTES=30 END_BY=21:50
