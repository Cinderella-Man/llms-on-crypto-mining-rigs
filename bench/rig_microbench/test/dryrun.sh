#!/usr/bin/env bash
# dryrun.sh: runs run_block8.sh against stub programs (mmvq_pascal, insn_probe, persistent_pascal, test-backend-ops,
# nvidia-smi) on any Linux machine without a GPU: checks that every step runs, a failing step does not stop the block,
# the summary parses the CSVs and logs, and nothing is left running (~10 s).
# Usage: bench/rig_microbench/test/dryrun.sh
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
mm=$(cd "$here/../mmvq_pascal" && pwd)
d=$(mktemp -d)
mkdir -p "$d/kit/bin" "$d/q6k" "$d/stub"
cp "$here/run_block8.sh" "$mm/multitoken_perf.sh" "$mm/multitoken_perf_table.py" "$d/kit/"
cat > "$d/kit/bin/mmvq_pascal" <<'SH'
#!/usr/bin/env bash
csv=""; mode=gate
while [ $# -gt 0 ]; do case $1 in --csv) csv=$2 ;; --tlb-probe) mode=tlb ;; --spread-gb) mode=spread; gb=$2 ;; esac; shift; done
h="device,code_arch,shape,K,N,experts,used,calls_per_token,type,variant,bytes_per_call,us_median,us_min,us_max,gbs,pct_of_ref,ref_gbs,regs,blocks_per_sm,threads_per_block,grid_blocks,err_over_tol,check,weight_sets,swept_bytes"
case $mode in
gate)   { echo "$h"
          echo '"P106-090",61,attn_q,2048,4096,0,0,49,q6_K,st_r1c1,6881280,52.1,51,53,132.10,76,173,40,6,128,1024,0.1,PASS,1,0'
          echo '"P106-090",61,attn_kv,2048,512,0,0,98,q6_K,st_r1c1,860160,7.1,7,7.2,121.20,70,173,40,6,128,128,0.1,PASS,1,0'
          echo '"P106-090",61,moe_down,512,2048,256,6,47,q6_K,st_r1c1,1290240,10.0,9.9,10.1,129.00,74,173,40,6,128,512,0.1,PASS,1,0'
          echo '"P106-090",61,STREAM_READ_PLAIN,0,0,0,0,0,-,-,268435456,,,,178.90,103,173,,,,,,,,'; } > "$csv"
        echo "correctness: all variants match the CPU reference" ;;
tlb)    echo "footprint_mib,stride,ns" > "$csv"; echo "16,2,300" >> "$csv"; echo "tlb probe done" ;;
spread) { echo "$h"; echo "\"P106-090\",61,moe_gate_up,2048,512,256,6,47,q5_0,st_r1c2,0,$(awk -v g="$gb" 'BEGIN { print 40 + 3 * g }'),1,1,120,70,173,,,,,,PASS,4,0"; } > "$csv"
        [ "$gb" = 3 ] && { echo "out of memory"; exit 1; } ;;
esac
exit 0
SH
cat > "$d/kit/bin/insn_probe" <<'SH'
#!/usr/bin/env bash
printf 'device 0: P106-090\n\nclass  sass   ops/clk/SM\nffma   FFMA       126.00\ndp4a   IDP        125.00\n\nops/clk/SM: ...\n'
SH
cat > "$d/kit/bin/persistent_pascal" <<'SH'
#!/usr/bin/env bash
echo "all checks PASS"; echo "persistent (dynamic queue): 160.0 us per layer, speed-up 1.19x, saving 1.4 ms per token"
SH
cat > "$d/q6k/test-backend-ops" <<'SH'
#!/usr/bin/env bash
t=q6_K; [ "${GGML_CUDA_MMVQ_PASCAL:-}" = 0 ] && us=80 || us=52
echo "  MUL_MAT(type_a=$t,type_b=f32,m=4096,n=1,k=2048,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):   1000 runs -  $us.00 us/run - 6720 kB/run - 129.23 GB/s"
SH
cat > "$d/stub/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in
    *query-compute-apps*) exit 0 ;;
    *-lms*) while :; do echo "2026/10/24 12:00:00.000, P0, 1880, 4004, 40.00, 41, 0x0"; sleep 0.5; done ;;
    *) echo "index, name"; echo "0, P106-090" ;;
esac
SH
chmod +x "$d"/kit/bin/* "$d/q6k/test-backend-ops" "$d/stub/nvidia-smi"
(cd "$d/kit" && md5sum bin/* *.sh *.py > MD5SUMS)
PATH="$d/stub:$PATH" Q6K_BIN="$d/q6k" OUT="$d/out" bash "$d/kit/run_block8.sh" > "$d/console.log" 2>&1
rc=$?
fail=0
check() { if "${@:2}"; then echo "  ok: $1"; else echo "  FAILED: $1"; fail=1; fi; }
s=$d/out/summary.txt
check "exit 0 (got $rc)" [ "$rc" = 0 ]
check "gate parsed: two dense K=2048 rows, not met (121.2 < 130)" bash -c "grep -q 'attn_kv .*121.2' '$s' && grep -q 'gate: NOT met' '$s' && ! grep -q moe_down '$s'"
check "fork tables have the one-token row" grep -q 'attn_q 2048->4096 | 1 |' "$s"
check "spread rows for all 6 sizes, the failed 3 GB step logged" bash -c "[ \$(grep -c 'moe_gate_up' '$s') = 6 ] && grep -q 'q99_spread3: exit 1' '$d/out/block8.log'"
check "insn table and Q88 lines in the summary" bash -c "grep -q 'dp4a   IDP' '$s' && grep -q 'speed-up 1.19x' '$s'"
check "all 14 steps timed" [ "$(grep -c ": exit " "$d/out/block8.log")" = 14 ]
check "nothing left running" bash -c "! pgrep -f '$d/[s]tub' > /dev/null"
[ $fail = 0 ] || { tail -30 "$d/console.log"; }
cat "$s"
rm -rf "$d"
exit $fail
