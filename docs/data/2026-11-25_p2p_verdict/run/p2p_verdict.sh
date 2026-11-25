#!/usr/bin/env bash
# The driver's peer-to-peer verdict per card pair, the override keys in the installed driver, and the IOMMU and ACS
# state, read-only: nothing is loaded, set or written. No serial numbers, UUIDs or addresses beyond PCI slots.
#
# Run it on the rig as root, with no timed task running there (each step is under a timeout):
#   ssh "${RIG:-rig}" 'bash -s' < run/p2p_verdict.sh > p2p_verdict.txt
set +e

echo "== driver"; head -1 /proc/driver/nvidia/version
echo "== topo -m"; timeout 30 nvidia-smi topo -m
echo "== topo -p2p r"; timeout 30 nvidia-smi topo -p2p r
echo "== topo -p2p w"; timeout 30 nvidia-smi topo -p2p w

echo "== key names in the installed module"
M=$(modinfo -n nvidia); echo "$M"
case "$M" in
  *.zst) D="zstd -dc" ;;
  *.xz) D="xz -dc" ;;
  *) D="cat" ;;
esac
for k in ForceP2P RMForceP2PType RMPcieP2PType PeerMappingOverride DmaRemapPeerMmio; do
  printf '%s: ' "$k"; timeout 60 $D "$M" | grep -a -c "$k"
done
echo "== nvidia module options"; grep -rh -i "options nvidia" /etc/modprobe.d/ 2>/dev/null

echo "== IOMMU and ACS"
dmesg 2>/dev/null | grep -i -e "iommu: Default domain" -e "ACS workaround" -e "DMAR: Intel(R) Virtualization"
echo "iommu groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"
cat /sys/kernel/iommu_groups/*/type 2>/dev/null | sort | uniq -c
for d in /sys/bus/pci/devices/*; do
  case "$(cat "$d/vendor")" in
    0x10de|0x1b21) echo "$(basename "$d") $(cat "$d/vendor") group $(basename "$(readlink "$d/iommu_group")")" ;;
  esac
done
grep -E "CONFIG_INTEL_IOMMU_DEFAULT_ON|CONFIG_IOMMU_DEFAULT_(PASSTHROUGH|DMA_LAZY|DMA_STRICT)" "/boot/config-$(uname -r)"
