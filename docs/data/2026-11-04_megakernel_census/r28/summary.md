# r28

- trace: 1152 evaluations on 9 cards; 94 decoded tokens, 89 of them directly after another (wall time measurable)
- per token (medians): GPU 113.33 ms, launch groups 2517, hops between cards 14.27 ms, host before card 0 11.11 ms, wall 139.75 ms (profiler on); GPU busy 81.1 % of the wall
- groups per token: 2519 (est. kernels 3188); without weights 1658, 30.98 ms
- group durations, µs: min 4.10, p1 7.17, p10 8.19, median 12.29; without weights: p10 7.17, median 10.24, p90 22.53
- groups per token by duration: <5 µs 0, 5-10 µs 626, 10-20 µs 1058, 20-50 µs 319, >=50 µs 516
