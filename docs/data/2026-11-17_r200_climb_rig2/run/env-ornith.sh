# R200: the Ornith check and the one-boot chain on the rig: the kit settings plus the Ornith file, its release
# settings and the folder for its hashes. Source it in the kit folder; WORK names the try's fresh work folder.
. ./env.sh
export WORK=${WORK:-/var/tmp/climb}
export MODEL=/mnt/models2/ornith/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
export SETTINGS=$PWD/ornith-v4.env   # a copy of release/ornith-v4/settings.env
export OUT=/var/tmp/r200-ornith
