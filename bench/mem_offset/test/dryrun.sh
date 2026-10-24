#!/usr/bin/env bash
# dryrun.sh: exercises memoc_steps.sh's control flow without a GPU, NVML or root, on any Linux machine: every program
# the kit calls (nvml_memoc.py, mmvq_pascal, cuda_memtest, the llama.cpp tools, nvidia-smi, dmesg, id) is replaced by a
# stub that keeps the "offset" in a file. Scenarios:
#   clean      the bench fails from +800 on: BEST +600, memtest, exact and a 1-minute soak pass, offset ends at 0
#   exactfail  as clean, but the logits differ at +600: `all` steps back to +400
#   xid        an Xid appears at +400: BEST +200
#   interrupt  TERM during the memtest: the trap kills the memtest and puts the offset back to 0
#   toolfail   cuda_memtest fails at +600 without reporting a memory error: the run stops, BEST stays +600
# Usage: bench/mem_offset/test/dryrun.sh [scenario...]   (default: all five; ~5 min, mostly the soak minute)
#        KEEP=1 keeps each scenario's folder (console.log, out/) and prints its path
set -uo pipefail
kit=$(cd "$(dirname "$0")/.." && pwd)
scenarios=${*:-clean exactfail xid interrupt toolfail}
fail=0

setup() {  # setup DIR SCENARIO
    local d=$1
    rm -rf "$d"; mkdir -p "$d/kit/bin" "$d/stub" "$d/llama"
    cp "$kit/memoc_steps.sh" "$kit/exact_check.sh" "$d/kit/"
    echo 0 > "$d/offset"; : > "$d/dmesg"; echo "$2" > "$d/scenario"
    export DRY=$d
    cat > "$d/kit/nvml_memoc.py" <<'PY'
#!/usr/bin/env python3
import os, sys, time
d = os.environ["DRY"]; f = os.path.join(d, "offset")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
cmd = args[0]
def cur(): return int(open(f).read())
def line(): return f"{time.strftime('%F %T')} gpu 0, P0, sm 1880 MHz, mem {4004 + cur() // 2} MHz, 40.0 W, 40 C, MemClk vf offset {cur()}"
if cmd == "set":
    open(f, "w").write(args[2]); print("after:", line())
elif cmd == "reset":
    open(f, "w").write("0"); print("after:", line()); open(os.path.join(d, "resets"), "a").write("x")
elif cmd == "units":
    print("  NVML raw unit: MT/s (transfer-rate units); stored in stub")
elif cmd in ("clocks", "probe", "get"):
    print(line())
PY
    cat > "$d/kit/bin/mmvq_pascal" <<'SH'
#!/usr/bin/env bash
off=$(cat "$DRY/offset"); sc=$(cat "$DRY/scenario"); csv=""
while [ $# -gt 0 ]; do [ "$1" = --csv ] && csv=$2; shift; done
[ "$sc" = xid ] && [ "$off" = 400 ] && echo "NVRM: Xid (PCI:0000:03:00): 13, Graphics SM Warp Exception" >> "$DRY/dmesg"
check=PASS; [ "$off" -ge 800 ] && check=FAIL
gbs=$(awk -v o="$off" 'BEGIN { printf "%.2f", 179 * (1 + o / 8008) }')
[ -n "$csv" ] && {
    echo "device,code_arch,shape,K,N,experts,used,calls_per_token,type,variant,bytes_per_call,us_median,us_min,us_max,gbs,pct_of_ref,ref_gbs,regs,blocks_per_sm,threads_per_block,grid_blocks,err_over_tol,check,weight_sets,swept_bytes"
    echo "\"P106-090\",61,attn_q,2048,4096,0,0,49,q8_0,llama,0,1,1,1,140.00,80,173,32,8,128,1024,0.0,$check,1,0"
    echo "\"P106-090\",61,STREAM_READ_LDG,0,0,0,0,0,-,-,268435456,,,,$gbs,90,173,,,,,,,,"
    echo "\"P106-090\",61,STREAM_READ_PLAIN,0,0,0,0,0,-,-,268435456,,,,$gbs,90,173,,,,,,,,"
} > "$csv"
sleep 1
[ "$check" = FAIL ] && exit 2; exit 0
SH
    cat > "$d/kit/bin/cuda_memtest" <<'SH'
#!/usr/bin/env bash
case "$*" in *--list_tests*) exit 0 ;; esac
[ "$(cat "$DRY/scenario")" = toolfail ] && [ "$(cat "$DRY/offset")" != 0 ] && { echo "ERROR: cannot allocate any memory from GPU"; exit 1; }
sleep "${DRY_MEMTEST_S:-2}"; echo "main thread: Program exits"
SH
    cat > "$d/llama/llama-perplexity" <<'SH'
#!/usr/bin/env bash
off=$(cat "$DRY/offset"); sc=$(cat "$DRY/scenario"); kld=0; base=""; prev=""
for a in "$@"; do [ "$a" = --kl-divergence ] && kld=1; [ "$prev" = --kl-divergence-base ] && base=$a; prev=$a; done
sleep 1
if [ $kld = 0 ]; then echo "base logits" > "$base"; echo "0.01.000.000 I Final estimate: PPL = 7.1234 +/- 0.1"; exit 0; fi
max=0.000043; [ "$sc" = exactfail ] && [ "$off" = 600 ] && max=0.002100
printf 'Mean PPL(Q)                   :   7.123456 ±   0.100000\nMean    KLD:   0.000000 ±   0.000000\nMaximum KLD:   %s\nSame top p: 100.000 ± 0.000 %%\n' "$max"
SH
    cat > "$d/llama/llama-completion" <<'SH'
#!/usr/bin/env bash
echo "the same greedy text"
SH
    cat > "$d/llama/llama-bench" <<'SH'
#!/usr/bin/env bash
while :; do sleep 1; done
SH
    cat > "$d/stub/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in *query-compute-apps*) exit 0 ;; esac
while :; do echo "2026/10/24 12:00:00.000, P0, 1880, $((4004 + $(cat "$DRY/offset") / 2)), 40.00, 41, 0x0"; sleep 0.5; done
SH
    printf '#!/usr/bin/env bash\ncat "$DRY/dmesg"\n' > "$d/stub/dmesg"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$d/stub/journalctl"
    printf '#!/usr/bin/env bash\necho 0\n' > "$d/stub/id"
    chmod +x "$d"/kit/*.sh "$d"/kit/nvml_memoc.py "$d"/kit/bin/* "$d"/llama/* "$d"/stub/*
}

nothing_left() {  # no process runs a program from the scenario's folders ([k]it etc. keep pgrep from matching this test)
    ! pgrep -f "$1/([k]it|[s]tub|[l]lama)/" > /dev/null
}

expect() {  # expect DESCRIPTION CONDITION...
    local what=$1; shift
    if "$@"; then echo "  ok: $what"; else echo "  FAILED: $what"; fail=1; fi
}

for sc in $scenarios; do
    d=$(mktemp -d)/dry
    setup "$d" "$sc"
    echo "== $sc"
    envs=(PATH="$d/stub:$PATH" B="$d/llama" OUT="$d/out" SOAK_MIN=1 MEMTEST_MIN=1 STEP=200 MAX=1200)
    case "$sc" in
    interrupt)
        env "${envs[@]}" DRY_MEMTEST_S=60 bash "$d/kit/memoc_steps.sh" all > "$d/console.log" 2>&1 &
        pid=$!
        for _ in $(seq 120); do grep -q 'memtest: cuda_memtest at' "$d/out/steps.log" 2>/dev/null && break; sleep 1; done
        sleep 2; kill -TERM "$pid"; wait "$pid"; rc=$?
        sleep 1
        expect "exit status 130 after TERM (got $rc)" [ "$rc" = 130 ]
        expect "offset back to 0" [ "$(cat "$d/offset")" = 0 ]
        expect "no stub left running" nothing_left "$d"
        ;;
    *)
        env "${envs[@]}" bash "$d/kit/memoc_steps.sh" all > "$d/console.log" 2>&1; rc=$?
        best=$(cat "$d/out/best" 2>/dev/null)
        case "$sc" in
        clean)     expect "BEST +600 (got $best), exit 0 (got $rc)" [ "$best" = 600 -a "$rc" = 0 ] ;;
        exactfail) expect "BEST stepped back to +400 (got $best), exit 0 (got $rc)" [ "$best" = 400 -a "$rc" = 0 ] ;;
        xid)       expect "BEST +200 (got $best)" [ "$best" = 200 ] ;;
        toolfail)  expect "stopped with exit 1 (got $rc), BEST still +600 (got $best)" [ "$rc" = 1 -a "$best" = 600 ] ;;
        esac
        expect "offset back to 0" [ "$(cat "$d/offset")" = 0 ]
        expect "no stub left running" nothing_left "$d"
        ;;
    esac
    [ $fail = 0 ] || { echo "-- console of $sc:"; tail -40 "$d/console.log"; }
    if [ -n "${KEEP:-}" ]; then echo "  kept: $d"; else rm -rf "$(dirname "$d")"; fi
done
exit $fail
