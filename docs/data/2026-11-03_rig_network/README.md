# Q511: why copies into the rigs crawled, and the fix (2026-11-03)

After a router change, copies into both rigs ran at 20–80 kB/s while copies out of them ran normally, so a 66 GB model
file would have taken days. The rigs then sat behind a small consumer router (the "second router") whose own uplink went
to the main router: double NAT. This page records the measurements that located the fault, and the same paths after the
rigs moved onto a gigabit switch on the main network. Each path was measured for 10–20 s with `iperf3` (rig to rig) or a
one-shot `nc` stream of zeros read through `dd` (the other paths), with both rigs idle.

## Result: the second router dropped traffic toward the rigs; a switch on the main network fixes it

| Path | Behind the second router | On a switch on the main network |
|---|---:|---:|
| Rig → rig | 941 Mbit/s each way | 941 Mbit/s each way |
| Rig → the main network, plain TCP | 105 MB/s | 16–47 MB/s (the other end was busy with a download) |
| Rig → the main network, Tailscale | 98 MB/s | 47 MB/s (the same download) |
| The main network → rig, plain TCP | **21 kB/s** | **95 MB/s** |
| The main network → rig, Tailscale | **28 kB/s** | **85 MB/s** |
| Ping loss, rig → main network, 1,200–1,500-byte packets | **10–60 %** | 0 % |

- **The fault was on the second router's link to the main network, in the direction toward the rigs.** The rigs'
  NICs (Realtek r8169, gigabit, full duplex) showed no errors. Traffic between the two rigs through that router's own
  switch ran at full speed. Traffic out of the rigs ran at full speed. Only traffic in, over plain TCP and over Tailscale
  alike, collapsed with heavy packet loss.
- **Moving both rigs onto a gigabit switch on the main network fixed it.** That also removed the double NAT, so the
  rigs and the other machines now reach each other directly on the LAN, and Tailscale finds direct paths at 1–2 ms.
- **The first test after the move was misleading:** one rig received at 405 kB/s for a few seconds, then 85 MB/s on a
  rerun with 0 % loss. Wait a minute after re-cabling before measuring.
- **What it means for the rest of the project:** copies to the rigs no longer need a disk carried by hand, and an agent
  check driven from another machine measures the model rather than the network. Run `iperf3` in both directions first on
  any new rig network; a fault that only hurts one direction stays invisible to `ping` and to downloads in the other.

## How it ran

```bash
# rig to rig: a server on one rig, the client on the other
iperf3 -s -B <rig2 address>                       # on rig 2
iperf3 -c <rig2 address> -t 10; iperf3 -c <rig2 address> -t 10 -R   # on rig 1
# any other path: a listener on the receiving machine, 2 GiB of zeros from the sender, 20 s cap
timeout 25 nc -l <receiver address> <port> | dd of=/dev/null bs=1M     # receiver
timeout 20 sh -c 'head -c 2147483648 /dev/zero | nc -N <receiver address> <port>'   # sender
ping -c 10 -M do -s 1472 <other machine>        # loss and path MTU
```
