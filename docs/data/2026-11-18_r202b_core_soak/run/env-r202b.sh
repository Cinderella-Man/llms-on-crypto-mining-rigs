# R202b on rig 2: the core soak ladder for all nine cards, every card's memory at r201-use.json's value.
# soak_chain.sh sources this in the kit folder (/var/tmp/oc/mem_offset-kit-20261024) when started with
#   ENV_FILE=./env-r202b.sh setsid nohup bash soak_chain.sh > /var/tmp/r202b.console 2>&1 < /dev/null &
# MODEL is empty on purpose: no served-model check tonight, so the chain fits before 22:00.
. ./env.sh
K=/var/tmp/oc/mem_offset-kit-20261024
export WORK=/var/tmp/r202b
export PROFILE=$K/r202b-rung1.json
export LADDER="$K/r202b-rung2.json $K/r202b-rung3.json"
export MODEL= SETTINGS= OUT=
export MINUTES=30 END_BY=21:45
