#!/usr/bin/env bash
# sass_counts.sh SASS PTXAS_LOG: per probe kernel, how often its timed instruction occurs in the sm_61 SASS (64 per
# loop iteration, times the compiler's unrolling of the loop; f2f 128, imad 3 XMAD per step), and the registers ptxas gave each kernel (32 or fewer keeps
# 8 blocks of 256 threads on an SM). Called by build.sh inside the CUDA container.
set -euo pipefail
sass=$1; log=$2
echo "kernel  timed-instruction  count-in-kernel  (64 per loop iteration, times the compiler's unrolling; f2f 128 and"
echo "imad 3 XMAD per step: a count below 64 means steps were merged)"
for kv in ffma:FFMA dfma:DFMA hfma2:HFMA2 iadd:IADD imad:XMAD lop3:LOP3 shf:SHF bfe:BFE prmt:PRMT dp4a:IDP \
          popc:POPC i2f:I2F f2f:F2F shfl:SHFL lds:LDS; do
    k=${kv%%:*}; op=${kv#*:}
    n=$(awk -v f="probe_$k" -v o="$op" '
        /Function : / { cur = $3 }
        cur == f && /\/\*[0-9a-f]+\*\// {
            line = $0; sub(/^.*\*\/ +/, "", line); sub(/^@!?U?P[0-9T] +/, "", line)
            split(line, w, /[ .;]/); if (w[1] == o) n++ }
        END { print n + 0 }' "$sass")
    printf "%-6s %-6s %s\n" "$k" "$op" "$n"
done
echo
echo "registers per kernel (sm_61):"
awk '/Compiling entry function/ { match($0, /probe_[a-z0-9]+/); k = substr($0, RSTART, RLENGTH); a = ($0 ~ /sm_61/) }
     a && /Used [0-9]+ registers/ { match($0, /Used [0-9]+ registers/); print "  " k ": " substr($0, RSTART, RLENGTH) }' "$log"
