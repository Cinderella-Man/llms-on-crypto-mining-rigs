#!/usr/bin/env bash
# cuda_preflight_test.sh: checks scripts/cuda_preflight.sh and its probe without a GPU. It compiles a stand-in
# libcuda.so.1 (scripts/cuda_preflight/mock_libcuda.c) that can fail any driver call on any device, corrupt a copy or
# hang, runs the real probe against it in a container, and feeds the script a synthetic kernel log. Needs Docker and
# the image (IMAGE, default nvidia/cuda:12.9.2-devel-ubuntu24.04; any image with gcc); no GPU and no driver.
# Usage: scripts/test/cuda_preflight_test.sh   (PASS/FAIL per check; exit 0 when all pass)
# Environment: IMAGE, PREFLIGHT_DOCKER_ARGS (extra docker run arguments, e.g. a label), TMPDIR (where the scratch folder
# goes; containers mount it at the same path).
set -u
here=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$here/.." && pwd)
PRE=$S/cuda_preflight.sh
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
export IMAGE
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/mock" "$T/cache"
export PREFLIGHT_CACHE=$T/cache
: > "$T/klog_none"
export PREFLIGHT_KERNEL_LOG=$T/klog_none     # the host's own kernel log must not decide a check
fails=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); echo "      output: $out"; fi; }

read -r -a EXTRA <<< "${PREFLIGHT_DOCKER_ARGS:-}"
docker run --rm ${EXTRA[@]+"${EXTRA[@]}"} -e NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" \
    -v "$S/cuda_preflight:$S/cuda_preflight:ro" -v "$T/mock:$T/mock" "$IMAGE" \
    gcc -shared -fPIC -O1 -o "$T/mock/libcuda.so.1" "$S/cuda_preflight/mock_libcuda.c" > "$T/mock/build.log" 2>&1 \
    || { echo "FAIL  the mock libcuda does not build"; cat "$T/mock/build.log"; exit 1; }

# mock VAR=value ...: the probe against the mock driver, with those variables set inside the container
mock() {
    local a=() v
    for v in "$@"; do a+=(-e "$v"); done
    "$PRE" -q -- -e LD_LIBRARY_PATH="$T/mock" -v "$T/mock:$T/mock:ro" "${a[@]}"
}

cat > "$T/klog" <<'EOF'
2026-01-01T10:00:00+00:00 box kernel: NVRM: Xid (PCI:0000:09:00): 79, GPU has fallen off the bus.
2026-01-01T10:00:00+00:00 box kernel: NVRM: Xid (PCI:0000:09:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)
2026-01-01T10:00:01+00:00 box kernel: NVRM: Xid (PCI:0000:04:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
2026-01-01T10:00:01+00:00 box kernel: NVRM: Xid (PCI:0000:09:00): 154, GPU recovery action changed from 0x1 (GPU Reset Required) to 0x2 (Node Reboot Required)
2026-01-01T10:00:01+00:00 box kernel: NVRM: nvGpuOpsReportFatalError: uvm encountered global fatal error 0x60, requiring os reboot to recover.
EOF
printf '%s\n' '2026-01-01T10:00:00+00:00 box kernel: NVRM: Xid (PCI:0000:04:00): 13, Graphics Exception' > "$T/klog_quiet"

out=$(mock MOCK_DEVICES=2); rc=$?
check "two working devices: exit 0, one line, the OK verdict" '[ $rc = 0 ] && [ "$(echo "$out" | wc -l)" = 1 ] && echo "$out" | grep -q "^PREFLIGHT OK: 2 devices"'
out=$(mock MOCK_DEVICES=1); rc=$?
check "one device: singular" '[ $rc = 0 ] && echo "$out" | grep -q "PREFLIGHT OK: 1 device,"'

out=$(mock MOCK_FAIL=cuMemAlloc:1=2); rc=$?
check "allocation fails on card 1: exit 1, names card 1, its bus, the call and the cause" \
    '[ $rc = 1 ] && echo "$out" | grep -q "card 1 at bus 0000:05:00.0.*(Mock GPU 1): cuMemAlloc returned CUDA_ERROR_OUT_OF_MEMORY (2)" && echo "$out" | grep -q "memory is full"'
check "...and card 0 is not blamed" '! echo "$out" | grep -q "card 0"'
out=$(mock MOCK_FAIL=cuMemcpyDtoH:0=719); rc=$?
check "a copy fails on card 0 with a launch failure: names card 0 and the Xid hint" \
    '[ $rc = 1 ] && echo "$out" | grep -q "card 0 at bus 0000:04:00.0" && echo "$out" | grep -q "Xid lines in the kernel log"'
out=$(mock MOCK_CORRUPT=0); rc=$?
check "a copy that reads back different data fails" '[ $rc = 1 ] && echo "$out" | grep -q "card 0 .*found a data mismatch" && echo "$out" | grep -q "different data"'

out=$(mock MOCK_FAIL=cuInit=999); rc=$?
check "cuInit returns 999: exit 1, the code, and the missing device nodes (no GPU request mounts any here)" \
    '[ $rc = 1 ] && echo "$out" | grep -q "cuInit returned CUDA_ERROR_UNKNOWN (999); open(/dev/nvidiactl) fails" && echo "$out" | grep -q "GPU request did not reach it"'
out=$(PREFLIGHT_TIMEOUT=3 mock MOCK_HANG=1); rc=$?
check "a hung cuInit is reported after the timeout, and the container is gone" \
    '[ $rc = 1 ] && echo "$out" | grep -q "did not finish within 3 s" && [ -z "$(docker ps -q --filter name=cuda-preflight)" ]'
out=$("$PRE" -q -- -e UNUSED=1); rc=$?
check "no driver mounted: libcuda cannot be loaded" '[ $rc = 1 ] && echo "$out" | grep -q "libcuda.so.1 cannot be loaded"'
out=$("$PRE" -q -- --device /dev/cuda-preflight-no-such-device); rc=$?
check "docker cannot start the container: exit 1, says so" '[ $rc = 1 ] && echo "$out" | grep -q "docker could not start the probe container"'

out=$(PREFLIGHT_KERNEL_LOG=$T/klog_quiet mock); rc=$?
check "a log with only an unrelated Xid: no warning" '[ $rc = 0 ] && ! echo "$out" | grep -q WARNING'
out=$(PREFLIGHT_KERNEL_LOG=$T/klog mock); rc=$?
check "probe passes but the log has Xid 79 / 154 / UVM: exit 0 with a warning naming the bus" \
    '[ $rc = 0 ] && echo "$out" | grep -q "WARNING, the kernel log of this boot says: Xid 79 (fell off the bus) on PCI 0000:09:00" && echo "$out" | grep -q "UVM global fatal error at 10:00:01"'
out=$(PREFLIGHT_KERNEL_STRICT=1 PREFLIGHT_KERNEL_LOG=$T/klog mock); rc=$?
check "PREFLIGHT_KERNEL_STRICT=1: the same log fails the check" '[ $rc = 1 ] && echo "$out" | grep -q "PREFLIGHT_KERNEL_STRICT=1"'
out=$(PREFLIGHT_KERNEL_LOG=$T/klog mock MOCK_FAIL=cuInit=999); rc=$?
check "probe fails and the log explains it: verdict, log and the reboot advice in one line" \
    '[ $rc = 1 ] && [ "$(echo "$out" | wc -l)" = 1 ] && echo "$out" | grep -q "Kernel log of this boot: Xid 79" && echo "$out" | grep -q "Node Reboot Required) on 2 GPU(s)" && ! echo "$out" | grep -q "Reset Required" && echo "$out" | grep -q "reboot or power cycle"'

out=$(PREFLIGHT_SKIP=1 "$PRE" -q -- -e UNUSED=1); rc=$?
check "PREFLIGHT_SKIP=1 succeeds without running anything" '[ $rc = 0 ] && echo "$out" | grep -q "^PREFLIGHT SKIPPED"'
out=$("$PRE" 2>&1); rc=$?
check "no GPU request: usage, exit 2" '[ $rc = 2 ] && echo "$out" | grep -q "cuda_preflight.sh \[-i IMAGE\]"'
out=$(mock MOCK_DEVICES=2); rc=$?
[ -z "$(docker ps -aq --filter name=cuda-preflight)" ] || { echo "FAIL  a probe container is left behind"; fails=$((fails + 1)); }

echo
[ $fails = 0 ] && echo "all checks passed" || echo "$fails check(s) failed"
exit $((fails > 0))
