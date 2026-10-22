#!/usr/bin/env bash
# Read-only inventory of a mining rig: CPU, RAM, disk encryption speed, NVIDIA driver and CUDA, per-GPU state,
# PCIe link and ASPM state, and the llama.cpp build. It changes nothing. Run it as root (dmidecode, lspci -vvv).
# The output contains no hostname, MAC address or GPU serial, so it is safe to share.
#
#   sudo LLAMA_DIR=/path/to/llama.cpp bash scripts/rig_inventory.sh > inventory.txt
#   ssh root@rig 'LLAMA_DIR=/path/to/llama.cpp bash -s' < scripts/rig_inventory.sh > inventory.txt
set -u

LLAMA=${LLAMA_DIR:-$HOME/llama.cpp}
CUDA=${CUDA_DIR:-/usr/local/cuda-12.9}

run() {
    echo
    echo "===== $* ====="
    bash -c "$*" 2>&1
}

run 'date -Is; uname -r; lsb_release -ds; uptime'
run 'systemctl get-default; systemctl is-active gdm3 nvidia-persistenced 2>&1'

# CPU, RAM, swap
run 'lscpu | grep -E "Model name|^CPU\(s\)|MHz|L2|L3"'
run 'grep -m1 -o -w -E "sse4_2|aes|avx|avx2|fma|f16c|bmi2|avx512f" /proc/cpuinfo | sort -u | tr "\n" " "; echo'
run 'free -m; swapon --show'
run 'dmidecode -t memory | grep -E "^\s+(Size|Speed|Configured Memory Speed|Locator|Type):" | grep -v "Type Detail"'
run 'ps -eo rss,comm --sort=-rss | head -12'

# Storage and disk encryption (if any)
run 'df -h / /boot; lsblk -o NAME,SIZE,TYPE,TRAN,FSTYPE,MOUNTPOINTS | grep -v loop'
run 'for d in $(lsblk -rno NAME,TYPE | awk "\$2==\"crypt\"{print \$1}"); do cryptsetup status $d | grep -E "type|cipher|keysize"; done'
run 'command -v cryptsetup >/dev/null && cryptsetup benchmark -c aes-xts-plain64 -s 512'

# NVIDIA driver, CUDA, libraries
run 'cat /proc/driver/nvidia/version; cat /proc/cmdline'
run 'grep -E "EnablePCIeGen3|RegistryDwords|EnableMSI|DynamicPowerManagement" /proc/driver/nvidia/params'
run "python3 -c 'import json; d=json.load(open(\"$CUDA/version.json\")); [print(k, d[k][\"version\"]) for k in (\"cuda\", \"cuda_cudart\", \"libcublas\", \"cuda_nvcc\")]'"
run "ls -l $CUDA/targets/x86_64-linux/lib/ | grep -E 'libcudart.so|libcublas.so|libcublasLt.so'"
run 'ldconfig -p | grep -E "libcuda\.so|libnccl|libvulkan\.so"'
run 'dpkg -l | grep -E "^ii\s+(libnccl|nccl|vulkan-tools|nvtop|iperf3|nfs-common|python3-venv)" | awk "{print \$2, \$3}"'

# GPU state at idle
run 'nvidia-smi --query-gpu=index,pci.bus_id,name,vbios_version,pstate,persistence_mode,memory.used,memory.total,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks.applications.graphics,clocks.applications.memory,power.draw,power.limit,power.min_limit,power.max_limit,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv'
run 'nvidia-smi --query-gpu=index,clocks_event_reasons.active --format=csv || nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv'
run 'nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv; nvidia-smi | sed -n "/Processes/,\$p"'
run 'nvidia-smi topo -m'
run 'nvidia-smi -q -i 0 -d PERFORMANCE,CLOCK,POWER | grep -v "^$" | grep -v -E "Timestamp|Serial|UUID"'
run 'nvidia-smi -q -i 0 -d SUPPORTED_CLOCKS | grep "Memory"'
run 'nvidia-smi -q -d CLOCK | grep -E -A4 "Memory Clock Samples" | grep -E "Max|Min"'
run 'journalctl -k --no-pager | grep -i -E "NVRM: Xid|nvidia.*error" | tail -20; echo "(end of Xid scan)"'

# PCIe links and ASPM
run 'cat /sys/module/pcie_aspm/parameters/policy'
run 'for d in $(lspci -D -d 10de: | awk "{print \$1}") $(lspci -D -d 1b21: | awk "{print \$1}") $(lspci -D | grep -i "PCI bridge" | grep -i intel | awk "{print \$1}"); do echo "--- $d $(lspci -s $d | cut -d" " -f2-)"; lspci -vvv -s $d | grep -E "LnkCap:|LnkCtl:|LnkSta:|L1SubCap|L1SubCtl1" | sed "s/^\s*/  /"; done'

# Network link speed (no addresses)
run 'for i in $(ls /sys/class/net | grep -v -E "^(lo|tailscale|docker|br-|veth)"); do echo "$i: $(cat /sys/class/net/$i/operstate) $(cat /sys/class/net/$i/speed 2>/dev/null) Mb/s"; done'

# llama.cpp build
run "git -c safe.directory=$LLAMA -C $LLAMA log -1 --format='%H %ci %s'"
run "grep -E '^(GGML_[A-Z0-9_]+|CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE|CMAKE_CUDA_COMPILER):' $LLAMA/build/CMakeCache.txt | grep -E 'CUDA|NATIVE|AVX|FMA|F16C|SSE|RPC|VULKAN|BUILD_TYPE|BLAS'"
run "$LLAMA/build/bin/llama-cli --version"
run "$LLAMA/build/bin/llama-server --list-devices"
run "grep -n -B1 -A4 'cc < GGML_CUDA_CC_VOLTA' $LLAMA/ggml/src/ggml-cuda/ggml-cuda.cu | grep -E 'VOLTA|disable_due_to_gpu_arch'"
run "objdump -d $LLAMA/build/bin/libggml-cpu.so | grep -c ymm"
