#!/usr/bin/env bash
# The rig's PCIe link facts, read-only: one `lspci -vvv` over every NVIDIA GPU, ASMedia switch port and Intel PCI
# bridge, filtered to the fields that decide a faster card-to-card path (link speed and width, the supported and
# target speeds, payload and read-request sizes, ACS, BAR sizes, the driver bound). Serial numbers and other
# identifying fields are never printed.
#
# Run it on the rig as root (lspci needs root for the capability list), with nothing else running there:
#   ssh "${RIG:-rig}" 'bash -s' < run/pcie_facts.sh > pcie_facts.txt
set -u

lspci -D -nn -vvv 2>/dev/null | awk '
  /^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]/ {
    keep = ($0 ~ /\[10de:/ || $0 ~ /\[1b21:/ || ($0 ~ /\[8086:/ && $0 ~ /PCI bridge/))
    if (keep) print "--- " $0
    next
  }
  keep && /(LnkCap:|LnkSta:|LnkCtl:|LnkCap2:|LnkCtl2:|LnkSta2:|DevCap:|DevCtl:|MaxPayload|MaxReadReq|ACSCap:|ACSCtl:|Access Control Services|Region [0-9]|Kernel driver in use|Bus: primary)/ {
    sub(/^[ \t]+/, "  "); print
  }
'
echo "--- tree"
lspci -tv 2>/dev/null
