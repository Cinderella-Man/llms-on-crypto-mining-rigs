# The rig's PCIe link facts, for faster card-to-card paths (C416, 2026-11-04)

What this holds: the link capability and status of every device between the host and the nine P106-090 cards (the
cards, the two ASM1184e switches' ports and the PCH root ports), read with `lspci -vv` on the first rig, and the
read-only script that reads the rest of what the research note needs. The result page is the research note
[Home-made ways to link the rig's cards](../../research/notes/diy_card_interconnect.md); registry rows Q990 and
Q991 in [quants/experiments.md](../../../quants/experiments.md). The hop timings it builds on are C140's
([2026-11-04_megakernel_rig](../2026-11-04_megakernel_rig/README.md)) and C200r's
([2026-11-04_card_hops](../2026-11-04_card_hops/README.md)).

## The links (rig 1, the hardware inventory of 2026-10-22; the second rig's setup check of 2026-11-03 matched it)

| Device | Count | `LnkCap` (what it can do) | `LnkSta` (what it runs) |
|---|---:|---|---|
| P106-090 (`10de:1c09`) | 9 | 2.5 GT/s, x4 | 2.5 GT/s, x1 (downgraded) |
| ASM1184e downstream port (`1b21:1184`), above each card behind a switch | 8 | 5 GT/s, x1 | 2.5 GT/s, x1 |
| ASM1184e upstream port | 2 | 5 GT/s, x1 | 5 GT/s, x1 |
| PCH root port to a switch | 2 | 5 GT/s, x1 | 5 GT/s, x1 |
| PCH root port to card 4 | 1 | 5 GT/s, x1 | 2.5 GT/s, x1 |
| PCH root port to the network controller | 1 | 5 GT/s, x1 | 2.5 GT/s, x1 |

ASPM is disabled on every GPU link and switch port; the root ports do not support it.

What it shows:
- **The cards set the Gen1 limit.** Every port above a card offers 5 GT/s (PCIe 2.0); the cards offer 2.5 GT/s. A card
  that advertised 5 GT/s would train at Gen2 with nothing else changed.
- **Nothing on the board goes beyond Gen2 x1:** the switches are x1 on both sides and the root ports are x1 at 5 GT/s.
- **The cards are x4 parts** in one-lane slots.

Not yet read: the cards' `LnkCap2` and `LnkCtl2`, ACS on the switch ports, MaxPayload and MaxReadReq on each device, BAR
sizes (Q990; the command below).

## Run

```bash
# read-only, on an idle rig (no timed task, no model loading): one lspci -vvv filtered to the fields above, plus the tree
ssh "${RIG:-rig}" 'bash -s' < run/pcie_facts.sh > pcie_facts.txt
```

`run/pcie_facts.sh` prints, for every NVIDIA device, ASMedia switch port and Intel PCI bridge: `LnkCap`, `LnkSta`,
`LnkCtl`, `LnkCap2`, `LnkCtl2`, `LnkSta2`, `DevCap` and `DevCtl` (MaxPayload, MaxReadReq), the ACS capability and
control, BAR sizes and the bound driver; serial numbers are never printed. Its raw output is not published; the tables
here summarize it.

## Files

| File | What |
|---|---|
| [run/pcie_facts.sh](run/pcie_facts.sh) | the read-only reader of the link facts |

Provenance: `lspci` (pciutils) output on the rig; no third-party data.
