#!/usr/bin/env bash
# The second Gen2 try, on a fresh boot where nothing has initialized the cards yet. Every step is its own subcommand,
# run by hand in order; nothing retries a failed push (a retry broke card 4's link on the first try).
#
# The boot: a one-time GRUB entry (install-entry, then grub-reboot) adds to the default kernel line
#   modprobe.blacklist=nvidia                        the driver does not load itself at boot
#   module_blacklist=nvidia_drm,nvidia_modeset       the kernel refuses the display modules, which udev loads by name
#                                                    as soon as the driver appears and which initialize every card
#   systemd.mask=nvidia-persistenced.service         the persistence daemon stays off
# so no card is initialized until `load` adds the patched driver by hand.
#
# Usage (root, on the rig; files in $DIR):
#   gen2_fresh_boot.sh install-entry   write /boot/grub/custom.cfg from the default entry, check it, pick it for one boot
#   gen2_fresh_boot.sh remove-entry    delete it again and clear the one-time choice
#   gen2_fresh_boot.sh check           after the boot: the kernel line, no NVIDIA module, the card's registers before any driver
#   gen2_fresh_boot.sh load            insmod the patched driver (nvidia.ko.gen1bail), nothing else may open the cards
#   gen2_fresh_boot.sh push            gen2_push.py on the card: read-only checks, then the writes and one retrain
#   gen2_fresh_boot.sh test [N]        N rounds (default 5) of host <-> card copies checked and timed, the link read after each
set -u
DIR=${DIR:-/var/tmp/gen2}
CARD=${CARD:-0000:07:00.0}
CUDA_INDEX=${CUDA_INDEX:-4}
EXTRA="modprobe.blacklist=nvidia module_blacklist=nvidia_drm,nvidia_modeset systemd.mask=nvidia-persistenced.service"
ENTRY_ID=gen2-hold
mark() { echo "gen2 fresh boot: $*" > /dev/kmsg; }

case "${1:-}" in
  install-entry)
    cfg=/boot/grub/grub.cfg
    # the first "menuentry 'Ubuntu'" block, renamed, with the extra parameters on its linux line
    awk -v extra="$EXTRA" -v id="$ENTRY_ID" '
      /^menuentry / && !done { f = 1; print "menuentry \x27Ubuntu, Gen2 test boot (NVIDIA driver held back)\x27 --class ubuntu --id " id " {"; next }
      f && /^[ \t]*linux[ \t]/ { sub(/\$vt_handoff/, extra " $vt_handoff") }
      f { print }
      f && /^}/ { f = 0; done = 1 }' "$cfg" > "$DIR/custom.cfg.new"
    grep -q "$EXTRA" "$DIR/custom.cfg.new" || { echo "the linux line was not changed"; exit 1; }
    grub-script-check "$DIR/custom.cfg.new" || { echo "grub-script-check failed"; exit 1; }
    [ -e /boot/grub/custom.cfg ] && { echo "/boot/grub/custom.cfg exists already; not overwriting"; exit 1; }
    cp "$DIR/custom.cfg.new" /boot/grub/custom.cfg
    grub-reboot "$ENTRY_ID"
    echo "== /boot/grub/custom.cfg"; cat /boot/grub/custom.cfg
    echo "== grubenv"; grub-editenv list
    ;;
  remove-entry)
    rm -f /boot/grub/custom.cfg
    grub-editenv /boot/grub/grubenv unset next_entry
    echo "== grubenv"; grub-editenv list; ls /boot/grub/custom.cfg 2>&1
    ;;
  check)
    mark check
    echo "== kernel line"; cat /proc/cmdline
    for p in $EXTRA; do grep -q -- "$p" /proc/cmdline && echo "  has $p" || echo "  MISSING $p"; done
    echo "== NVIDIA modules (none expected)"; lsmod | grep -E "^nvidia" || echo "  none"
    echo "== persistence daemon: $(systemctl is-active nvidia-persistenced)"
    echo "== card $CARD"; lspci -s "$CARD" -nn
    echo "== registers before any driver (read only)"; python3 -I "$DIR/gen2_push.py" "$CARD"
    ;;
  load)
    mark load
    lsmod | grep -q "^nvidia " && { echo "nvidia is loaded already: the cards may be initialized; stopping"; exit 1; }
    insmod "$DIR/nvidia.ko.gen1bail" NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var || exit 1
    sleep 3
    echo "== modules"; lsmod | grep -E "^nvidia" | awk '{print $1}' | tr '\n' ' '; echo
    lsmod | grep -q -E "^nvidia_(drm|modeset) " && echo "WARNING: a display module loaded; the cards may be initialized"
    echo "== who has the cards open (nobody expected)"; fuser -v /dev/nvidia* 2>&1 | grep -v "USER" || echo "  nobody"
    dmesg | grep -E "module verification failed|blacklisted" | tail -3
    ;;
  push)
    mark push
    python3 -I "$DIR/gen2_push.py" "$CARD" push
    echo "push exit $?"
    lspci -s "$CARD" -vv | grep -E "LnkCap:|LnkSta:"
    ;;
  test)
    n=${2:-5}
    for i in $(seq 1 "$n"); do
      mark "test round $i"
      echo "== round $i"
      "$DIR/p2p_check" --pairs none --host-cards "$CUDA_INDEX" --tests time --rounds 1 --reps 100 2>&1 | sed -n '/== card/,$p'
      echo "link after the round: $(lspci -s "$CARD" -vv | grep -o 'LnkSta:.*' | head -1)"
    done
    echo "== kernel errors"; dmesg | grep -i -E "xid|aer|fallen" | tail -5
    ;;
  *)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
