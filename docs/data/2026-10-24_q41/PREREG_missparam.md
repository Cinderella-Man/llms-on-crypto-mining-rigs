# Pre-registration: G3 against UD-Q4_K_XL on fresh BFCL v4 miss_param tasks (confirmatory)

Written 2026-10-24 08:09 CEST, before any confirmatory slice started and before any of its results existed.
Decided after the exploratory result on 47 miss_param tasks (UD 24, G3 14; discordant 12 vs 2,
exact McNemar p 0.013, picked after seeing the pilot).

- **Hypothesis:** G3 passes fewer miss_param tasks than UD-Q4_K_XL.
- **Tasks:** the 139 BFCL v4 `multi_turn_miss_param` tasks in `e2_missparam_ids.txt` (sha256 `0f3be5e1fc323374…`).
  - These are all miss_param tasks outside the 47 already analysed (pilot 25, new-scenario slices 1–2: 22) and
    outside the mixed slice 3 that was running when this was written.
  - None is the excluded `multi_turn_miss_func_49`.
- **Runs:** attempt 1 (seed 1) for both files with the Q41 harness unchanged. UD-Q4_K_XL runs without the rescale
  variable, G3 with `LLAMA_NO_MOE_DOWN_RESCALE=1`.
  - The 139 tasks are split into 12 parts of 11–12 tasks (`main.sh ... IDS=/v4/e2_missparam_ids.txt`, shuffle 1, part
    k/12).
  - V100s (`-np 8 -c 262144`): parts 1–6 in order. GPUs 0,1,2 (`-np 4 -c 139264`): parts 12–7 in order.
  - On each GPU set every part runs UD, then G3, so each part ends as a pair on one GPU set.
- **Primary test:** one-sided exact McNemar on the pairs complete at 12:00, α = 0.05.
  - A pair is a task with an attempt-1 verdict from both files.
  - b = tasks only UD passes, c = tasks only G3 passes; p = P(X ≥ b) for X ~ Binomial(b + c, 1/2).
- **Stop rule:** test whatever pairs are complete at 12:00, when the GPUs are due for other work. No extension, no
  further attempts, no extra tasks.
- **Reported alongside (descriptive, not tested):**
  - pass rates and the paired difference with its 90 % interval;
  - BFCL's failure types per file (instance state mismatch, empty turn, execution response mismatch);
  - **the ask rate per file.** In the turn whose ground truth is empty (the user leaves a needed value out),
    "asked" means no decodable tool call in any step of that turn, "acted" means at least one. A secondary
    indicator is whether the model's text in that turn contains a question mark.
- **Kept apart:** the 47 exploratory tasks are reported separately and are not pooled into the test. The overall
  E2 tally (all categories) continues as before.
