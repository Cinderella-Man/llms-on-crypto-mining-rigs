#!/usr/bin/env bash
# Q99 on one P106 (card 0, nothing else on the rig): the TLB probe and the weight-spread sweep of bench/mmvq_pascal,
# the commands of bench/mmvq_pascal/README.md ("Rig commands", Q99 section).
#   RIG=<ssh alias> BIN=<path to a bench/mmvq_pascal build> bash run.sh    # results in ./out/
set -euo pipefail
RIG=${RIG:-rig}
BIN=${BIN:?path to bench/mmvq_pascal/build/mmvq_pascal}
mkdir -p out
ssh "$RIG" 'mkdir -p /tmp/q99'
ssh "$RIG" 'cat > /tmp/q99/mmvq_pascal && chmod +x /tmp/q99/mmvq_pascal' < "$BIN"
ssh "$RIG" 'cd /tmp/q99 && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
    ./mmvq_pascal --tlb-probe --csv tlb.csv > tlb.txt 2>&1
    for gb in 0.5 1 2 3 4 5; do
        ./mmvq_pascal --types q5_0,q8_0,q4_1 --variants st_r1c2,st_r2c2,st_r2c4 \
            --shapes moe_gate_up,moe_down,attn_q,attn_output,output --spread-gb $gb --csv spread$gb.csv > spread$gb.txt 2>&1
    done'
for f in tlb.csv tlb.txt spread0.5.csv spread1.csv spread2.csv spread3.csv spread4.csv spread5.csv; do
    ssh "$RIG" "cat /tmp/q99/$f" > "out/$f"
done
ssh "$RIG" 'rm -rf /tmp/q99'
