# r13

- trace: 1152 evaluations on 9 cards; 94 decoded tokens, 89 of them directly after another (wall time measurable)
- per token (medians): GPU 129.06 ms, launch groups 2517, hops between cards 15.66 ms, host before card 0 11.54 ms, wall 155.38 ms (profiler on); GPU busy 83.1 % of the wall
- groups per token: 2519 (est. kernels 2898); without weights 1658, 30.90 ms
- group durations, µs: min 4.10, p1 7.17, p10 8.19, median 12.29; without weights: p10 7.17, median 10.24, p90 22.53
- groups per token by duration: <5 µs 0, 5-10 µs 619, 10-20 µs 1075, 20-50 µs 215, >=50 µs 612
