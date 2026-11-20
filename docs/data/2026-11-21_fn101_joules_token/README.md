# FN101: joules a token beside tok/s in the agent check's tables (2026-11-21)

Registry row [Q1518](../../../quants/experiments.md). Tool: [bench/agent_check/energy.py](../../../bench/agent_check/energy.py),
read by `report.py`, `agent_tables.py` and `class_arms.py`
([how the columns are computed](../../../bench/agent_check/README.md#the-cards-energy-joules-a-task-and-a-token)).

**In plain words.** Every agent check since 2026-11-17 logs each card's power twice a second. The agent check's tables
now turn that log into energy: how many joules the nine cards spent on a task and on each generated token, and how
many watts they drew while generating, while reading prompts and in the gaps between turns. This folder gives the first
tables, for twelve runs that already had the log: R220's four, R227's four and R209's stock and overclocked pairs. No
rig time was used.

**What they show.**

- **The cards draw about the same watts in every run.** On the second rig with its clock profile, the nine cards drew
  357.6, 356.2, 355.8 and 357.2 W while generating in R220's four runs and 356.1, 355.0, 353.6 and 354.6 W in R227's
  ([energy.tsv](energy.tsv)): about 40 W a card against a 75 W limit. So energy follows time: a faster run spends
  fewer joules.
- **A session's joules a token move with the work, as its minutes do.** Every run writes its own replies, and a run
  that generates more spreads its prompt reading and its gaps over more tokens. R220's two control runs, the same
  preset on the same boot, spent 18.17 and 16.88 J a generated token, and R227's two 16.24 and 14.89. The joules a
  task at equal work, which price one reference run's work at each run's speeds and watts, hold the work fixed:
  24,620 and 25,431 J a task for R220's controls, 29,798 and 29,135 J for R227's (each table on its own first run's
  work). The class tables judge energy first on those.
- **R220's draft ratio 0.75 and R227's prompt taper at 1,400 tokens are negligible on energy**, as on time
  ([class_r220.tsv](class_r220.tsv), [class_r227.tsv](class_r227.tsv)). The ratio: 24,201 and 24,521 J a task at
  equal work against the control's 24,620 and 25,431 (−2.7 %, inside the control's spread of 811 J). The taper: 30,015
  and 29,753 against 29,798 and 29,135 (+1.4 %, inside its spread of 663 J). The taper draws more while reading prompts,
  396.0 and 391.7 W against 368.9 and 372.3 W, and reads them no slower, so its prompt joules rise; that is too
  small a part of a session to move the joules at equal work.
- **The overclock buys its speed for about the same energy on the same work** (R209, one boot of the second rig, one
  run an arm; [r209_pairs.tsv](r209_pairs.tsv)). Flash-Next with every card's offsets drew 8.04 % more watts while
  generating (356.1 against 329.6 W) and generated 10.32 % faster (30.69 against 27.82 tok/s): 1.64 % fewer joules a
  task at equal work (30,460 against 30,968 J) and 3.91 % fewer joules a generated token (16.23 against 16.89).
  Ornith drew 8.26 % more while generating (373.4 against 344.9 W) and generated 7.78 % faster: 1.52 % more joules a
  task at equal work (35,012 against 34,489 J) and 9.28 % fewer joules a generated token (26.50 against 29.21); its
  session's joules a task rose 28.73 % (44,662 against 34,695) because the overclocked run generated 20,222 tokens
  against 14,253. Two runs of one Flash-Next preset moved +3.29 % and −2.22 % apart in joules at equal work (R220's
  and R227's controls), so neither model's change is beyond what a rerun moves. With one run an arm R209 gets no
  class here. For a "with rig settings" column: the overclock does not raise the energy a task costs.
- **The gaps between turns are short, and their watts are those of short pauses.** In c10a the agent's tools took
  33.1 of the session's 826.7 s (4.00 %), in 64 gaps: about half a second a turn (every run's `idle_s` is in
  [energy.tsv](energy.tsv)). That is about one card sample a gap, and NVML's reading lags behind the load, so c10a's
  327.3 W in the gaps is the cards just after a reply, not a rig left idle.
  Flash-Next's preset also runs its keep-alive block on each card between turns, which holds the clocks up. As context
  only: in the 14 s c10a's log ran on after its last reply, the nine cards drew 246.1 W, against 51.83 W asleep (P8)
  before the server loaded; how much a loaded server draws while it waits for minutes between sessions is registry
  row Q1658 (FN122).
- **Card 8 draws the most while generating** in every Flash-Next run: in c10a 43.62 W, against 39.65, 39.51, 39.03,
  38.97, 39.00, 38.86, 39.30 and 39.67 W for cards 0 to 7 ([energy_cards.tsv](energy_cards.tsv), by card index).
- **The client's and the rig's clocks line up.** The shift that best lines the requests up with the card power was
  +0.05 s in seven runs, +0.10 s in four and +0.15 s in one ([energy.md](energy.md)), against requests of 12 s on
  average. Moving the proxy's times half a second either way changes the watts while generating by at most 0.7 W and
  while reading prompts by at most 3.7 W ([shift.tsv](shift.tsv)); the gaps, a sample or two each, move by up to
  19.4 W.

## The runs

| Run | Row | Model, settings | Its folder |
|---|---|---|---|
| `c10a`, `c10b` | R220 | Flash-Next's v5 preset (control) | [2026-11-20_r220_flash_ratio075](../2026-11-20_r220_flash_ratio075/README.md) |
| `r075a`, `r075b` | R220 | the same with the sparse attention's ratio at 0.75 | same |
| `c600a`, `c600b` | R227 | Flash-Next's v5 preset (control) | [2026-11-20_r227_flash_taper1400_agent](../2026-11-20_r227_flash_taper1400_agent/README.md) |
| `t1400a`, `t1400b` | R227 | the same with the prompt taper at 1,400 tokens | same |
| `f5oc0`, `f5oc1` | R209 | Flash-Next at stock, then with every card's offsets | [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md) |
| `o5oc0`, `o5oc1` | R209 | Ornith at stock, then with every card's offsets | same |

All twelve ran on the second rig. R220's and R227's runs had its clock profile set; R209's stock runs had every offset
at 0.

## Files

| File | What it holds |
|---|---|
| [energy.md](energy.md), [energy.tsv](energy.tsv) | One row a run (`energy.py --fit --tsv`): tasks, turns, tokens generated, the session window and the seconds the log covers, the card energy, joules a task and a generated token, the mean watts and the watts and seconds of each part; `energy.md` adds the best clock shift |
| [energy_cards.tsv](energy_cards.tsv) | One row a run and card (by index): joules, and mean watts over the session, while generating, reading prompts and between turns |
| [shift.tsv](shift.tsv) | Each part's watts with the proxy's times moved −0.5, 0 and +0.5 s |
| [agent_r220.tsv](agent_r220.tsv), [agent_r227.tsv](agent_r227.tsv), [agent_r209_flash.tsv](agent_r209_flash.tsv), [agent_r209_ornith.tsv](agent_r209_ornith.tsv) | `agent_tables.py`'s tables with the energy columns; the reference (the equal-work columns' base) is each table's first run |
| [class_r220.tsv](class_r220.tsv), [class_r227.tsv](class_r227.tsv) | `class_arms.py`: the change against its control, two runs an arm, with the energy measures last |
| [r209_pairs.tsv](r209_pairs.tsv) | R209's stock and overclocked runs side by side, with how far R220's and R227's two controls moved apart |
| [check.txt](check.txt) | Each agent table's first 12 columns against the published row of the same run, its energy columns against `energy.tsv`, and its turns against the proxy's requests: all the same |
| [run/tables.sh](run/tables.sh) | Writes everything above |

## Commands

The tables folder holds, per run, the card log, the proxy's rows and the server log from that run's raw output
(the agent check's `rig/cards.csv`, `proxy.jsonl` and `rig/server-<arm>.log`), as `rig_<run>/cards.csv`,
`proxy_<run>/proxy.jsonl` and `rig/server-<run>.log`. R209's two Ornith runs shared one client folder, so `o5oc1`'s
card log and proxy rows hold `o5oc0`'s too; the window and the arm pick each run's own (`o5oc0` from its own log and
from the shared one give the same row). Then, from the repository root, with Python 3 and no packages:

```bash
bash docs/data/2026-11-21_fn101_joules_token/run/tables.sh TABLES
bash bench/agent_check/selftest_energy.sh /tmp/energy-selftest      # the energy columns' own test
```

The published tables come out as before without a card log: `report.py compare`, `table` and `gonogo` on R209's
`f5oc1` give the same files with the new code as with the code before it, and
[C855's check](../2026-11-17_c855_class_work_counts/run/check.sh) rebuilds its 19 tables and classes byte for byte.

## Provenance and licence

Measured on the project's own rig on 2026-11-19 and 2026-11-20 with the repository's own scripts; the tables were made
from the runs' logs at midnight between 2026-11-20 and 2026-11-21. The card logs and proxy rows themselves are not here (they are raw output); every
figure here is per card index or summed over the cards. No model file or task text is in this folder. The tables and
scripts here are under the repository's licence.
