# Python pilot: one row per session (2026-10-26)

`report.py` over the pilot's three session folders (stage 1, stage 2 round 1, stage 2 round 2): turns, generated tokens,
the longest turn, the final sequence length, how the session ended (`answer`, `turns` cap, context `limit`, `time` cap),
whether its check passed, tool calls, failed calls, seconds spent in tools and wall seconds. Totals below the table.

| session | type | turns | gen | longest_turn | seq | end | verified | calls | failed_calls | tool_s | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1-colorsys-wt | write_tests | 11 | 24852 | 8428 | 31476 | time | True | 15 | 2 | 2.3 | 1500.6 |
| s1-fnmatch-wt | write_tests | 15 | 28016 | 7956 | 32768 | limit | False | 23 | 7 | 2.8 | 1554.6 |
| s1-fractions-wt | write_tests | 12 | 18484 | 6225 | 32768 | limit | False | 15 | 3 | 1.7 | 1306.1 |
| s1-getopt-wt | write_tests | 13 | 17976 | 8837 | 22575 | answer | True | 17 | 4 | 2.4 | 1279.6 |
| s1-graphlib-wt | write_tests | 16 | 19375 | 7232 | 24061 | answer | True | 19 | 4 | 2.8 | 1345.7 |
| s1-netrc-wt | write_tests | 20 | 14136 | 2267 | 19600 | answer | True | 25 | 4 | 3.8 | 1034.4 |
| s1-reprlib-wt | write_tests | 24 | 10421 | 6214 | 14785 | turns | False | 30 | 2 | 3.7 | 758.1 |
| s1-shlex-wt | write_tests | 24 | 16153 | 4541 | 21757 | turns | False | 29 | 4 | 3.8 | 1170.5 |
| s2-colorsys-a0-fix_bug-0 | fix_bug | 7 | 6100 | 5329 | 12857 | answer | True | 7 | 0 | 1.3 | 177.5 |
| s2-getopt-a0-fix_bug-0 | fix_bug | 7 | 1615 | 867 | 6956 | answer | True | 7 | 0 | 1.3 | 97.2 |
| s2-graphlib-a0-fix_bug-0 | fix_bug | 10 | 1549 | 1096 | 8358 | answer | True | 9 | 0 | 1.7 | 95.0 |
| s2-netrc-a0-fix_bug-0 | fix_bug | 7 | 2070 | 866 | 7130 | answer | True | 7 | 0 | 1.4 | 112.9 |
| s2b-colorsys-a0-fix_bug-0 | fix_bug | 7 | 2304 | 1300 | 8551 | answer | True | 7 | 0 | 1.3 | 262.4 |
| s2b-colorsys-a0-fix_bug-1 | fix_bug | 8 | 4701 | 1824 | 11514 | answer | True | 8 | 0 | 1.5 | 418.8 |
| s2b-getopt-a0-fix_bug-0 | fix_bug | 7 | 2364 | 1476 | 8542 | answer | True | 7 | 0 | 1.3 | 268.9 |
| s2b-getopt-a0-fix_bug-1 | fix_bug | 8 | 7531 | 6501 | 15267 | answer | True | 10 | 0 | 1.9 | 555.0 |
| s2b-getopt-a0-implement-0 | implement | 16 | 14550 | 8112 | 21234 | answer | True | 18 | 5 | 2.9 | 822.5 |
| s2b-graphlib-a0-fix_bug-0 | fix_bug | 8 | 3756 | 2305 | 9740 | answer | True | 8 | 0 | 1.3 | 361.5 |
| s2b-graphlib-a0-fix_bug-1 | fix_bug | 18 | 18096 | 8281 | 26215 | answer | True | 24 | 0 | 3.9 | 890.4 |
| s2b-graphlib-a0-implement-0 | implement | 7 | 5162 | 4557 | 10071 | answer | True | 7 | 0 | 1.2 | 441.7 |
| s2b-netrc-a0-fix_bug-0 | fix_bug | 10 | 11492 | 7569 | 16976 | answer | True | 11 | 0 | 1.9 | 721.6 |
| s2b-netrc-a0-fix_bug-1 | fix_bug | 7 | 3966 | 3427 | 8953 | answer | True | 7 | 0 | 1.4 | 374.8 |

```json
{
 "sessions": 22,
 "verified": 18,
 "turns": 262,
 "gen": 234669,
 "seq": 372154,
 "calls": 310,
 "failed_calls": 35,
 "unknown_tool": 18,
 "problems": 2,
 "end_time": 1,
 "end_limit": 2,
 "end_answer": 17,
 "end_turns": 2,
 "generated_share_of_sequence": 0.631,
 "calls_by_tool": {
  "bash": 92,
  "run_command": 70,
  "read_file": 32,
  "read": 30,
  "write_file": 27,
  "list_dir": 23,
  "shell": 16,
  "edit": 15,
  "edit_file": 3,
  "terminal": 1,
  "shell_command": 1
 },
 "per_type": {
  "write_tests": {
   "sessions": 8,
   "verified": 4,
   "median_gen": 18230.0,
   "median_turns": 15.5,
   "median_wall_s": 1292.85
  },
  "fix_bug": {
   "sessions": 12,
   "verified": 12,
   "median_gen": 3861.0,
   "median_turns": 7.5,
   "median_wall_s": 315.2
  },
  "implement": {
   "sessions": 2,
   "verified": 2,
   "median_gen": 9856.0,
   "median_turns": 11.5,
   "median_wall_s": 632.1
  }
 }
}
```
