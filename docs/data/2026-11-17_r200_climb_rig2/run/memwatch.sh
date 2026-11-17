# R200: host memory every 2 s while the climb runs (MemAvailable in kB), ends by itself after 3 h.
end=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "$(date +%T) MemAvailable $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  sleep 2
done
