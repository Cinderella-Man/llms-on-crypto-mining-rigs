#!/usr/bin/env bash
# Q671: checks p106_timing.sh and timing_table.py without a GPU: three fake arms whose test-backend-ops and bench binaries
# print the lines of the real ones (a made-up time per arm), a full run of the script with them, and the tables. Runs in a
# slim Python container (no GPU, no nvidia-smi: the script's quiet check then sees no process on the card).
# Usage: [DOCKER_ARGS=...] p106_timing_dryrun.sh
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
PYIMAGE=${PYIMAGE:-python:3.12-slim-bookworm}
docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$HERE:/r:ro" "$PYIMAGE" bash -c '
    set -e
    R=/tmp/root
    for arm in base a b; do
        us=$(case $arm in base) echo 200.00 ;; a) echo 180.00 ;; b) echo 220.00 ;; esac)
        mkdir -p $R/$arm/bin $R/bench/$arm
        cat > $R/$arm/bin/test-backend-ops <<STUB
#!/bin/sh
printf "  MUL_MAT_ID(type_a=iq3_xxs,type_b=f32,n_mats=64,n_used=8,b=0,m=3072,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0): 3974 runs - $us us/run - 50.33 MFLOP/run - \033[1;34m194.87 GFLOPS\033[0m\n"
STUB
        cat > $R/bench/$arm/mmvq_pascal <<STUB
#!/bin/sh
while [ \$# -gt 0 ]; do [ "\$1" = --csv ] && f=\$2; shift; done
printf "device,code_arch,shape,K,N,tokens,calls_per_token,variant,G,R,bytes_per_call,us_median,us_min,gbs,regs,grid_blocks,err_over_tol,check\nfake,61,hc_up,320,10240,4,1,pascal,128,8,1,$us,1,1,1,1,0,PASS\n" > \$f
STUB
        chmod +x $R/$arm/bin/test-backend-ops $R/bench/$arm/mmvq_pascal
    done
    ROOT=$R ARMS="base a b" REPS=2 OUT=/tmp/out LOAD_MAX=1000 CASES="iq_down bench_bf16" bash /r/p106_timing.sh
    cat /tmp/out/console.txt
    python3 -I /r/timing_table.py /tmp/out base
'
