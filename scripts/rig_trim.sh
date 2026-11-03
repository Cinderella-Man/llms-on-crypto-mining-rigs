#!/usr/bin/env bash
# After rig_debloat.sh, on a rig with the CUDA 12.x toolkit installed: two more trims of the 64 GB SSD (Q506).
#   1. Purge Nsight Compute and Nsight Systems (~2.3 GB): they cannot profile the P106 (Nsight Systems and nvprof refuse
#      the card). The purge also takes the file-less metapackages cuda-toolkit-12-N, cuda-tools-12-N and
#      cuda-visual-tools-12-N, so the toolkit's own packages are marked manual first, and no later autoremove takes them.
#      It stops unless the simulated removal is exactly Nsight plus those three metapackages.
#   2. Lower the root filesystem's ext4 reserve from 5 % to 1 % (tune2fs -m 1, online): ~2.3 GB on the 64 GB SSD.
# The CUDA toolkit, nvcc and the runtime libraries stay. Run as root with no model running; the default is a dry run:
#   bash rig_trim.sh
#   APPLY=1 bash rig_trim.sh
# BIN=<engine folder>/bin also checks that the engine still finds its libraries and the cards afterwards.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
APPLY=${APPLY:-0}
CUDA_RE='^(cuda-|libcu(blas|fft|file|rand|solver|sparse)|libnpp|libnv|gds-tools)'   # not libcups*
ROOTDEV=$(findmnt -n -o SOURCE /)
free_gb() { df -B1 --output=avail / | tail -1 | awk '{printf "%.1f GB free", $1 / 1e9}'; }

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
[ -z "$(ps -eo comm | grep -x -E 'llama-server|llama-cli' || true)" ] || { echo "a llama process runs: stop it first"; exit 1; }
echo "== $(date -Is) APPLY=$APPLY; root: $(free_gb)"

NS=$(dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' '*nsight*' 2>/dev/null | awk '$1=="ii"{print $2}' | tr '\n' ' ')
if [ -n "$NS" ]; then
  REMV=$(apt-get -s purge $NS 2>/dev/null | awk '/^Purg|^Remv/{print $2}' | sort -u)
  OTHER=$(echo "$REMV" | grep . | grep -v -E 'nsight|^cuda-toolkit-12-[0-9]+$|^cuda-tools-12-[0-9]+$|^cuda-visual-tools-12-[0-9]+$' || true)
  echo "-- Nsight: the purge would remove $(echo "$REMV" | grep -c .): $(echo "$REMV" | tr '\n' ' ')"
  [ -z "$OTHER" ] || { echo "it would also remove: $(echo "$OTHER" | tr '\n' ' '); stopping"; exit 2; }
else
  echo "-- Nsight: not installed"
fi
RES=$(tune2fs -l "$ROOTDEV" | awk -F: '/^Reserved block count/{print $2+0}'); ALL=$(tune2fs -l "$ROOTDEV" | awk -F: '/^Block count/{print $2+0}')
echo "-- ext4 reserve on $ROOTDEV: $RES of $ALL blocks ($(awk -v r="$RES" -v a="$ALL" 'BEGIN{printf "%.1f", r * 100 / a}') %)"
[ "$APPLY" = 1 ] || { echo "dry run: nothing changed (APPLY=1 to apply)"; exit 0; }

if [ -n "$NS" ]; then
  AUTO=$(apt-mark showauto | grep -E "$CUDA_RE" | grep -v -i nsight || true)
  [ -n "$AUTO" ] && apt-mark manual $AUTO > /dev/null && echo "-- marked manual: $(echo "$AUTO" | grep -c .) CUDA packages"
  apt-get -o DPkg::Lock::Timeout=300 purge -y $NS > /dev/null && echo "-- Nsight purged"
fi
tune2fs -m 1 "$ROOTDEV" > /dev/null && echo "-- ext4 reserve set to 1 %"

echo "== checks"
dpkg --audit && echo "dpkg --audit: clean"
NVCC=$(ls -d /usr/local/cuda-12*/bin/nvcc 2>/dev/null | tail -1)
[ -n "$NVCC" ] && echo "nvcc: $($NVCC --version | tail -1)"
echo "cards: $(nvidia-smi -L | wc -l)"
echo "CUDA packages autoremove would take: $(apt-get -s autoremove 2>/dev/null | awk '/^Remv/{print $2}' | grep -c -E "$CUDA_RE" || true)"
if [ -n "${BIN:-}" ] && [ -x "$BIN/llama-server" ]; then
  echo "engine: $(ldd "$BIN/llama-server" | grep -c 'not found') libraries missing, $("$BIN/llama-server" --list-devices 2>&1 | grep -cE '^\s*CUDA[0-9]+:') CUDA devices"
fi
echo "== $(date -Is) root: $(free_gb)"
