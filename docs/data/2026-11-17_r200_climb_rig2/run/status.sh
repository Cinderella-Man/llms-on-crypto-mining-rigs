# R200: one short status read of the climb on the rig (ssh rig2 'bash -s' < status.sh). Read-only.
W=${WORK:-/var/tmp/climb}
echo "now $(date +%T), $(grep MemAvailable /proc/meminfo | tr -s ' ')"
[ -f $W/CLIMB_DONE ] && echo "CLIMB_DONE: $(cat $W/CLIMB_DONE)"
[ -f $W/APPLY_DONE ] && echo "APPLY_DONE: $(cat $W/APPLY_DONE)"
[ -f $W/TRIPPED ] && echo "TRIPPED: $(tail -n 2 $W/TRIPPED)"
[ -f $W/CARD_TRIPS ] && echo "CARD_TRIPS: $(tail -n 3 $W/CARD_TRIPS)"
echo "-- climb.log, last lines"; tail -n 4 $W/out/climb.log 2>/dev/null | cut -c1-260
echo "-- failures and stops"; grep -E "FAIL card|STOP|REFUSED|RESET FAILED" $W/out/climb.log 2>/dev/null | cut -c1-260
H=$(grep -m1 '^time,' $W/wd.csv 2>/dev/null)
if [ -n "$H" ]; then
  echo "-- hottest reading per card since the watchdog started (C), and over the last ~10 min"
  awk -F, -v hdr="$H" 'BEGIN{n=split(hdr,h,","); for(i=1;i<=n;i++) if(h[i]~/^g[0-9]+_c$/) c[i]=h[i]}
    /^#/ || $1=="time" {next} {for(i in c) if($i!="" && $i+0>m[i]) m[i]=$i+0}
    END{for(i in c) printf "%s=%s ", c[i], m[i]; print ""}' $W/wd.csv
  tail -n 1200 $W/wd.csv | awk -F, -v hdr="$H" 'BEGIN{n=split(hdr,h,","); for(i=1;i<=n;i++) if(h[i]~/^g[0-9]+_c$/) c[i]=h[i]}
    /^#/ || $1=="time" {next} {for(i in c) if($i!="" && $i+0>m[i]) m[i]=$i+0}
    END{for(i in c) printf "%s=%s ", c[i], m[i]; print ""}'
fi
[ -f /var/tmp/r200-mem.log ] && echo "-- lowest MemAvailable so far: $(awk '{print $3}' /var/tmp/r200-mem.log | sort -n | head -n 1) kB"
