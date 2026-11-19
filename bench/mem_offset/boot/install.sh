#!/bin/bash
# install.sh: puts the boot-time clock service on a rig (bench/mem_offset/README.md, "Overclocked at every boot").
# It never enables or starts the service: `systemctl enable miner-llama-clocks` does that, per rig, by hand.
#
#   sudo bash bench/mem_offset/boot/install.sh                       # from a checkout of this repository
#   sudo MMVQ_BIN=<the kit tarball's bin/mmvq_pascal> bash bench/mem_offset/boot/install.sh
#
# What it does:
#   - copies the kit's scripts (climb.py, nvml_memoc.py, greedy_check.py, memoc_units.json, model_check.sh,
#     nvml_watchdog.py, boot/) to /opt/miner-llama/mem_offset, replacing an older copy there;
#   - copies mmvq_pascal there from MMVQ_BIN, or from the kit's own bin/ (a kit tarball has one); without either it
#     keeps one already installed, or says to set MMVQ_JOBS=0 (apply then skips its kernel checks);
#   - copies miner-llama-clocks.service to /etc/systemd/system and runs `systemctl daemon-reload`;
#   - writes /etc/miner-llama/clocks.env from boot/clocks.env.example only if there is none (never overwrites it);
#   - makes /var/lib/miner-llama.
# ROOT=<folder> installs under that folder instead of / and runs no systemctl (the tests use it).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
kit=$(dirname "$here")
ROOT=${ROOT:-}
if [ -z "$ROOT" ] && [ "$(id -u)" != 0 ]; then echo "install.sh: run as root (or ROOT=<folder> for a test)" >&2; exit 2; fi
dest=$ROOT/opt/miner-llama/mem_offset
if [ -d "$dest" ] && [ "$(cd "$dest" && pwd -P)" = "$(cd "$kit" && pwd -P)" ]; then
  echo "install.sh: run it from a checkout of the repository or an unpacked kit, not from $dest" >&2; exit 2
fi
wd=$kit/nvml_watchdog.py
[ -f "$wd" ] || wd=$kit/../../scripts/nvml_watchdog.py
for f in "$kit/climb.py" "$kit/nvml_memoc.py" "$kit/greedy_check.py" "$kit/memoc_units.json" "$kit/model_check.sh" \
         "$wd" "$here/clocks_at_boot.py" "$here/miner-llama-clocks.service" "$here/clocks.env.example"; do
  [ -f "$f" ] || { echo "install.sh: $f is missing" >&2; exit 1; }
done

install -d -m 0755 "$dest" "$dest/boot" "$dest/bin" "$ROOT/etc/systemd/system" "$ROOT/etc/miner-llama" \
  "$ROOT/var/lib/miner-llama"
install -m 0644 "$kit/climb.py" "$kit/nvml_memoc.py" "$kit/greedy_check.py" "$kit/memoc_units.json" \
  "$kit/model_check.sh" "$wd" "$dest/"
install -m 0755 "$here/clocks_at_boot.py" "$dest/boot/"
install -m 0644 "$here/miner-llama-clocks.service" "$here/clocks.env.example" "$here/install.sh" "$dest/boot/"
[ -f "$kit/README.md" ] && install -m 0644 "$kit/README.md" "$dest/"

if [ -n "${MMVQ_BIN:-}" ]; then
  install -m 0755 "$MMVQ_BIN" "$dest/bin/mmvq_pascal"; mmvq="from $MMVQ_BIN"
elif [ -f "$kit/bin/mmvq_pascal" ]; then
  install -m 0755 "$kit/bin/mmvq_pascal" "$dest/bin/mmvq_pascal"; mmvq="from $kit/bin"
elif [ -f "$dest/bin/mmvq_pascal" ]; then
  mmvq="the one already installed"
else
  mmvq="NONE: give MMVQ_BIN=<the kit tarball's bin/mmvq_pascal>, or set MMVQ_JOBS=0 in clocks.env"
fi

install -m 0644 "$here/miner-llama-clocks.service" "$ROOT/etc/systemd/system/miner-llama-clocks.service"
if [ -f "$ROOT/etc/miner-llama/clocks.env" ]; then
  cfg="kept as it was"
else
  install -m 0644 "$here/clocks.env.example" "$ROOT/etc/miner-llama/clocks.env"; cfg="new, from clocks.env.example: fill it in"
fi
[ -z "$ROOT" ] && systemctl daemon-reload

echo "installed (not enabled):"
echo "  kit      $dest"
echo "  mmvq     $mmvq"
echo "  unit     $ROOT/etc/systemd/system/miner-llama-clocks.service"
echo "  config   $ROOT/etc/miner-llama/clocks.env ($cfg)"
(cd "$dest" && md5sum climb.py nvml_memoc.py greedy_check.py memoc_units.json model_check.sh nvml_watchdog.py \
  boot/clocks_at_boot.py boot/miner-llama-clocks.service | sed 's/^/  /')
echo "next: the profile to /etc/miner-llama/clocks-profile.json, then BIN, CHECK_MODEL and DRIVER in clocks.env;"
echo "      on a fresh boot 'systemctl start miner-llama-clocks' tries it once; 'systemctl enable miner-llama-clocks'"
echo "      sets it at every boot (README, 'Overclocked at every boot')"
