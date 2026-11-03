#!/usr/bin/env bash
# Strips an Ubuntu 24.04 desktop install down to what an LLM rig uses (Q506, Q507): the desktop apps, every snap and
# snapd, idle services (ModemManager, CUPS, Avahi, Bluetooth, crash reporters, PackageKit, fwupd) and automatic
# upgrades go. The NVIDIA driver, CUDA, the kernel, SSH, Tailscale, NetworkManager and Python stay, and so does the
# desktop session (GDM, GNOME Shell, Xorg): NV-CONTROL needs it until headless mode is measured (Q83).
#
# Run as root on the rig, with no model running. The default is a dry run that prints the plan and everything apt
# would remove, and stops if any of it is protected. APPLY=1 does it.
#   bash rig_debloat.sh
#   APPLY=1 bash rig_debloat.sh
set -uo pipefail
APPLY=${APPLY:-0}
LOG=${LOG:-/root/rig_debloat.log}

# Installed packages whose names match these go (anchored at the start of the name).
APP_PATTERNS='thunderbird|firefox|libreoffice|rhythmbox|totem|shotwell|transmission-|remmina|simple-scan|deja-dup|gnome-calendar|gnome-clocks|gnome-characters|evince|eog$|baobab|gnome-text-editor|yelp|usb-creator|seahorse|gnome-snapshot|gnome-system-monitor|gnome-disk-utility|gnome-font-viewer|gnome-logs|file-roller|gnome-software|update-manager|update-notifier|cups|printer-driver-|hplip|spice-vdagent|aisleriot|gnome-mahjongg|gnome-mines|gnome-sudoku'
SERVICE_PATTERNS='modemmanager$|avahi-daemon$|avahi-autoipd$|bluez$|bluez-cups$|apport|whoopsie|kerneloops|packagekit|fwupd|unattended-upgrades$|snapd$'
# apt may remove none of these; matched against every package in the simulated removal.
PROTECT='^(nvidia-|libnvidia-|cuda|libcu|libnv|linux-|openssh-|tailscale|network-manager|python3$|python3\.[0-9]+$|python3-minimal|libpython3|fio$|gdm3$|gnome-shell$|gnome-session|mutter|xserver-xorg|xorg$|systemd|udev|sudo$|apt$|dpkg$|cryptsetup|lvm2|libc6|libcups2|ubuntu-minimal$|ubuntu-standard$|curl$|wget$|rsync$)'
# The desktop session and what the rig runs on are marked manual, so no later autoremove takes them.
KEEP_MANUAL='gdm3 gnome-shell gnome-session xserver-xorg-core xserver-xorg network-manager openssh-server python3 fio rsync curl'

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
state() {
  log "$1: $(free -m | awk '/Mem:/{print "RAM used "$3" MB, available "$7" MB"}'); root free $(df -h --output=avail / | tail -1 | xargs)"
  log "$1: running services $(systemctl list-units --type=service --state=running --no-legend | wc -l); snaps $(snap list 2>/dev/null | tail -n +2 | wc -l)"
}

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
pgrep -f '/bin/llama-' >/dev/null && { echo "a llama process is running: stop it first"; exit 1; }
log "rig_debloat start (APPLY=$APPLY) on $(. /etc/os-release; echo "$PRETTY_NAME")"
state before

# 1. Packages: one simulated transaction, checked against PROTECT before anything is removed.
mapfile -t PKGS < <(dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' | awk '$1=="ii"{print $2}' |
  sed 's/:amd64$//' | grep -E "^(${APP_PATTERNS}|${SERVICE_PATTERNS})" | sort -u)
log "packages selected (${#PKGS[@]}): ${PKGS[*]}"
SIM=""; SIM_RC=0
[ "${#PKGS[@]}" -gt 0 ] && { SIM=$(apt-get -s purge "${PKGS[@]}" 2>&1); SIM_RC=$?; }
[ "$SIM_RC" = 0 ] || { log "STOP: the simulation failed ($SIM_RC): $(echo "$SIM" | tail -3 | tr '\n' ' ')"; exit 2; }
REMV=$(echo "$SIM" | awk '/^Purg|^Remv/{print $2}' | sort -u)
log "apt would remove ($(echo "$REMV" | grep -c .)): $(echo "$REMV" | tr '\n' ' ')"
BAD=$(echo "$REMV" | grep -E "$PROTECT" || true)
if [ -n "$BAD" ]; then log "STOP: the removal would take protected packages: $(echo "$BAD" | tr '\n' ' ')"; exit 2; fi

# 2. Snaps, leaves first, then runtimes, bases and snapd itself.
SNAPS=$(snap list 2>/dev/null | awk 'NR>1{print $1}')
ORDER="firefox thunderbird snap-store firmware-updater desktop-security-center prompting-client snapd-desktop-integration"
ORDER="$ORDER $(echo "$SNAPS" | grep -v -E '^(core[0-9]*|bare|snapd)$' | grep -v -x -E "$(echo "$ORDER" | tr ' ' '|')" | tr '\n' ' ')"
ORDER="$ORDER $(echo "$SNAPS" | grep -E '^core[0-9]+$' | tr '\n' ' ') bare snapd"
log "snaps to remove in order: $(for s in $ORDER; do echo "$SNAPS" | grep -q -x "$s" && printf '%s ' "$s"; done)"

if [ "$APPLY" != 1 ]; then log "dry run: nothing changed (APPLY=1 to remove)"; exit 0; fi

apt-mark manual $KEEP_MANUAL >> "$LOG" 2>&1
# Stop the services first, sockets and paths included: a daemon still running when its package goes keeps its user in
# use, its postrm fails and apt exits 100 with the package half removed (first run, 2026-11-03).
for u in avahi-daemon.socket avahi-daemon.service cups.socket cups.path cups.service cups-browsed.service \
         ModemManager.service bluetooth.service fwupd.service packagekit.service apport.service whoopsie.service \
         kerneloops.service; do
  systemctl list-unit-files "$u" --no-legend 2>/dev/null | grep -q . && systemctl disable --now "$u" >> "$LOG" 2>&1
done
for s in $ORDER; do echo "$SNAPS" | grep -q -x "$s" && { snap remove --purge "$s" >> "$LOG" 2>&1 || log "WARN: snap remove $s failed"; }; done
[ "${#PKGS[@]}" -gt 0 ] && { DEBIAN_FRONTEND=noninteractive apt-get purge -y "${PKGS[@]}" >> "$LOG" 2>&1 || log "WARN: apt-get purge exit $?"; }
# snapd must not come back with a later install.
printf 'Package: snapd\nPin: release a=*\nPin-Priority: -10\n' > /etc/apt/preferences.d/no-snapd.pref

# 3. Automatic upgrades off: upgrades become a manual step between runs (Q507).
for t in apt-daily.timer apt-daily-upgrade.timer; do systemctl disable --now "$t" >> "$LOG" 2>&1; systemctl mask "$t" >> "$LOG" 2>&1; done
dpkg --audit >> "$LOG" 2>&1 && log "dpkg --audit: clean" || log "WARN: dpkg --audit reports half-removed packages"

state after
log "rig_debloat done; the desktop session is kept (Q83); nothing was autoremoved"
