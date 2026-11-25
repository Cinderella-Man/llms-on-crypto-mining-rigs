#!/usr/bin/env bash
# Reloads the NVIDIA kernel modules with one of three peer-to-peer settings, for the forced peer-to-peer test.
# Nothing is written to /etc: the options go on the modprobe command line, so the next boot is stock again.
#   stock    the driver as installed
#   remap0   NVreg_RegistryDwords="ForceP2P=0x11" NVreg_DmaRemapPeerMmio=0: peer access forced on, peer addresses
#            as plain bus addresses (no IOMMU mapping), for pairs behind one PCIe switch
#   iommu    NVreg_RegistryDwords="ForceP2P=0x11": peer access forced on, peer addresses mapped through the IOMMU
# Needs root, no process on the cards and no display server on them.
# Usage: ssh "$RIG" 'bash -s' -- <mode> < reload_driver.sh
set -u
mode=${1:?stock, remap0 or iommu}
case "$mode" in
  stock) opts=() ;;
  remap0) opts=(NVreg_RegistryDwords=ForceP2P=0x11 NVreg_DmaRemapPeerMmio=0) ;;
  iommu) opts=(NVreg_RegistryDwords=ForceP2P=0x11) ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "p2p test: reload $mode" > /dev/kmsg
if timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "a process is on the cards; not reloading"; exit 1
fi
systemctl stop nvidia-persistenced
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  if lsmod | grep -q "^$m "; then
    timeout 60 modprobe -r "$m" || { echo "unloading $m failed"; lsmod | grep '^nvidia'; exit 1; }
  fi
done
timeout 120 modprobe nvidia "${opts[@]}" || { echo "loading nvidia failed"; exit 1; }
for m in nvidia_uvm nvidia_modeset nvidia_drm; do timeout 60 modprobe "$m" || echo "loading $m failed"; done
systemctl start nvidia-persistenced
echo "== mode $mode"
grep -E 'RegistryDwords:|DmaRemapPeerMmio' /proc/driver/nvidia/params
timeout 60 nvidia-smi -L | wc -l
timeout 60 nvidia-smi topo -p2p r | sed -n 1,11p
