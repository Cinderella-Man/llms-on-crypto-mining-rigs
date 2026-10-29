# Q371 + Q372: two training changes against a control

Both arms start from the 1,480-minute nine-card head (`head.pt`) and train 1,100 optimizer steps on the same data in
the same order (`--max-steps 1100`, seed 1, 5,112 sequences, ~145 minutes each on one RTX 3090), with the head's
training command otherwise unchanged (`cmd-control.txt`, `cmd-exp.txt`). The warm start gets a fresh schedule: warm-up to
the learning rate 2 × 10⁻⁴, cosine decay to 2 × 10⁻⁵.

- **Control:** the head's own window draw (`--kinds 0.6,0.15,0.25`: staircase, pre-card, after-miss).
- **Experiment:** the after-miss window drawn twice as often (`--kinds 0.4,0.1,0.5`, Q371) and the loss weighted by what
  the server's lookups would guess at each generated position (`--source-labels`, `--source-weights 0.6,0.6,3,3,1`: copy
  lookup, 2-token continuation, 1-token continuation, none, other; Q372). The labels (`source_labels.py`, 1,177 sessions):
  of the generated positions the copy lookup takes 48 %, the 2-token continuations 26 %, the 1-token continuations 21 %,
  none 5 %. The weights move the head's positions (the last two) from 26 % to ~64 % of the loss.

Scored like Q384 and Q385 (`score.py --only-windows g9,g9miss`, α 3 as served and α 1), the paired difference by
`paired_diff.py` (the same positions in both files; 95 % intervals from resampling whole sessions).

## Experiment minus control (α 3)

| | Held-out sessions | 95 % | Agent check's text | 95 % |
|---|---|---|---|---|
| g9, all positions | −0.003 | −0.005 – −0.002 | +0.001 | −0.002 – +0.003 |
| g9, head positions | +0.001 | −0.000 – +0.002 | +0.004 | −0.001 – +0.006 |
| g9miss, all positions | +0.007 | +0.004 – +0.010 | +0.014 | +0.013 – +0.016 |
| g9miss, head positions | **+0.012** | +0.011 – +0.013 | **+0.016** | +0.014 – +0.019 |

At α 1 the same pattern: held-out +0.000 (g9) and +0.010 (g9miss) at the head's positions.

| Arm, held-out, α 3 | g9, head positions | g9miss, head positions |
|---|---|---|
| start (1,480-minute head, Q385) | 0.393 | 0.378 |
| control | 0.397 | 0.383 |
| experiment | 0.398 | 0.395 |

- **Q371 (after-miss window):** the window it trains gains +0.012 at the head's positions (+0.016 on the check's text),
  but less than the +0.03–0.06 expected, and it stays 0.003 below the staircase window there (0.395 against 0.398). Killed
  by its line (after-miss still below the staircase).
- **Q372 (loss by guess source):** right after card 1's stage (g9) the head's positions gain +0.001, under the +0.005
  line; the after-miss gain cannot be split between the two changes in this design. Killed by its line.
- **Together:** the head's positions gain +0.001–0.004 at g9 and +0.012–0.016 after a miss, at about −0.005 on positions the
  server does not give the head. Served, with part of the head's guesses following a miss, that is under +0.01 of
  acceptance (est.), well under 1 % of speed.
- **The fresh schedule:** validation (g9) fell from 0.557 to 0.488 in the first 130 steps of both arms and climbed back
  to 0.558 (control) and 0.549 (experiment) by step 1,100 (`train_log_*.jsonl`). The control ends within +0.004 of the
  start (held-out, the head's positions, α 3): a warm restart spends its steps recovering, as Q370's
  warm-up–stable–decay schedule is meant to avoid.

Files in `q371/`: `labels.txt` (the label shares), `smoke.txt` (10 steps of the experiment: the warm start loads every
weight, the labels cover all 1,177 records), `cmd-*.txt`, `train_log_*.jsonl` (both arms' logs with validation every
~15 minutes), `scores-<arm>-<text>-a<α>.md`, `breakdown-<arm>-<text>-a<α>.md`, `paired.md` (`paired_diff.py`, both
texts and α). `$RUN` is the training run's folder (`$RUN3` its part on a second disk), `$ARMS` the arms' folder, `$WORK`
the work folder.
