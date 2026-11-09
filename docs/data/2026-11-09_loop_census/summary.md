64 proxy logs, 4713 replies in 80 arms (1724 more rows are copies of those in a later log), 828 attempts with a result.

## Per group and model

| group | model | replies | tokens | repeat_token_pct | rule1_replies | rule2_replies | rule2_finished | class3_replies | rule1_token_pct | rule2_fire_pct | rule2_token_pct | rule1_time_pct | rule2_time_pct | attempts | runaway_attempts | runaway_time_pct | cut_replies | cut_rule1 | cut_rule2 | cut_class3 | length_stops | longest_tail_same | longest_tail_char_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| earlier | flint | 1590 | 1078648 | 2.60 | 1 | 45 | 40 | 0 | 0.01 | 4.05 | 14.12 | 0.01 | 10.49 | 252 | 10 | 19.31 | 10 | 0 | 5 | 0 | 0 | 1 | 3 |
| earlier | ornith | 371 | 110595 | 1.39 | 0 | 1 | 0 | 0 | 0.00 | 0.44 | 1.27 | 0.00 | 1.56 | 79 | 5 | 19.97 | 5 | 0 | 1 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 1248 | 405575 | 1.72 | 0 | 12 | 10 | 0 | 0.00 | 1.90 | 7.25 | 0.00 | 4.64 | 231 | 11 | 16.37 | 8 | 0 | 2 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 412 | 157837 | 2.18 | 0 | 8 | 8 | 0 | 0.00 | 5.26 | 15.18 | 0.00 | 8.03 | 78 | 3 | 13.11 | 2 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flint | 686 | 467228 | 2.25 | 0 | 16 | 15 | 0 | 0.00 | 2.54 | 10.37 | 0.00 | 8.40 | 108 | 3 | 12.42 | 3 | 0 | 1 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 399 | 118141 | 1.34 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 80 | 3 | 14.03 | 3 | 0 | 0 | 0 | 0 | 1 | 3 |

## Per arm

| group | model | folder | arm | replies | tokens | repeat_token_pct | rule1_replies | rule2_replies | rule2_finished | class3_replies | rule1_token_pct | rule2_fire_pct | rule2_token_pct | rule1_time_pct | rule2_time_pct | attempts | runaway_attempts | runaway_time_pct | cut_replies | cut_rule1 | cut_rule2 | cut_class3 | length_stops | longest_tail_same | longest_tail_char_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | t1base | 65 | 22942 | 2.24 | 0 | 1 | 1 | 0 | 0.00 | 7.07 | 19.26 | 0.00 | 11.94 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | t1skip | 58 | 32256 | 2.48 | 0 | 2 | 2 | 0 | 0.00 | 6.39 | 6.39 | 0.00 | 4.30 | 12 | 1 | 32.02 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | t1neg | 63 | 26722 | 2.88 | 0 | 2 | 2 | 0 | 0.00 | 7.59 | 28.40 | 0.00 | 19.36 | 12 | 1 | 34.81 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | t1swift | 66 | 25995 | 3.22 | 0 | 2 | 2 | 0 | 0.00 | 8.71 | 32.19 | 0.00 | 21.33 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | stockq2 | 47 | 16952 | 1.47 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | stockq2t | 53 | 16511 | 0.62 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 10 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flash_next | 2026-11-05_release_flashnext/rig/agent-report | stockpmix | 60 | 16459 | 1.00 | 0 | 1 | 1 | 0 | 0.00 | 1.97 | 9.22 | 0.00 | 7.30 | 11 | 1 | 25.79 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| v3-release | flint | 2026-11-05_release_regression/agent_flint/stock64k | rel | 67 | 43881 | 1.37 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 2 | 45.63 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/agent_flint/v3add | rel | 75 | 46720 | 1.56 | 0 | 1 | 1 | 0 | 0.00 | 2.22 | 2.60 | 0.00 | 2.06 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/agent_flint/v3base | rel | 77 | 40683 | 3.15 | 0 | 2 | 2 | 0 | 0.00 | 2.34 | 20.95 | 0.00 | 16.34 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/agent_flint_p065/v3base | rel | 78 | 48689 | 1.62 | 0 | 1 | 1 | 0 | 0.00 | 2.48 | 3.95 | 0.00 | 3.09 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/r1a-flint-q5/agent_pq5/v3base | rel | 99 | 67931 | 2.60 | 0 | 1 | 1 | 0 | 0.00 | 0.77 | 14.91 | 0.00 | 10.16 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/r1a-flint-q5/agent_q5stock/stock64k | rel | 68 | 46470 | 2.34 | 0 | 3 | 2 | 0 | 0.00 | 6.21 | 16.75 | 0.00 | 15.86 | 12 | 1 | 19.14 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-05_release_regression/r1a-flint-q5/agent_q5v3/v3base | rel | 70 | 51273 | 2.30 | 0 | 2 | 2 | 0 | 0.00 | 1.48 | 11.05 | 0.00 | 8.58 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-06_flint_side_rig2/agent_flint/reread | rel | 77 | 56059 | 2.19 | 0 | 2 | 2 | 0 | 0.00 | 1.35 | 2.23 | 0.00 | 1.76 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | flint | 2026-11-06_flint_side_rig2/agent_flint/v3base | rel | 75 | 65522 | 2.86 | 0 | 4 | 4 | 0 | 0.00 | 5.70 | 18.28 | 0.00 | 14.09 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-05_release_regression/agent_ornith/valid1 | rel | 62 | 17231 | 1.32 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-05_release_regression/agent_ornith_ka/valid1 | rel | 63 | 19113 | 1.77 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-05_release_regression/agent_ornith_ud-stock/ud-stock | stock | 56 | 15394 | 0.77 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 2 | 42.51 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-05_release_regression/agent_ornith_ud-v3/ud-v3 | rel | 54 | 13451 | 1.21 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-06_ornith_mtp_rig2/r2a/agent_ka/valid1 | rel | 63 | 16972 | 0.86 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-06_ornith_mtp_rig2/r2a/agent_mtpka/valid1-mtp | rel | 44 | 19165 | 2.24 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 1 | 30.41 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| v3-release | ornith | 2026-11-06_ornith_mtp_rig2/r2a/agent_mtpka-oom/valid1-mtp | rel | 0 | 0 |  | 0 | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v3-release | ornith | 2026-11-06_ornith_mtp_rig2/t3d-mtpadapt/agent_mtpadapt/valid1-mtp | rel | 57 | 16815 | 0.97 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_flash_next_baseline/agent | stock | 37 | 10825 | 1.75 | 0 | 1 | 1 | 0 | 0.00 | 9.27 | 16.55 | 0.00 | 12.60 | 8 | 1 | 30.70 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_flash_next_baseline/agent | rel | 53 | 15238 | 1.68 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 1 | 35.33 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-03_flash_next_mtp/agent | nodraft | 51 | 14766 | 1.16 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-03_flash_next_mtp/s15/agent-mtp-oom | mtp | 12 | 3548 | 0.20 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-03_flash_next_mtp/s15/agent-mtp2-report | mtp2 | 64 | 15649 | 1.18 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11 | 1 | 27.46 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_flash_next_dflash2/agent-dflash2b-report | dflash2 | 1 | 1 | 0.00 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |  |  | 0 | 0 |  | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-03_flash_next_dflash2/agent-dflash2b-report | dflash2b | 60 | 15609 | 1.71 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_flash_next_pipelined/r29a_rig1/agent | pipe0 | 41 | 14278 | 1.04 | 0 | 1 | 1 | 0 | 0.00 | 7.05 | 7.91 | 0.00 | 6.10 | 8 | 1 | 28.39 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_flash_next_pipelined/r29a_rig1/agent | pipe3 | 2 | 37 | 0.00 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |  |  | 0 | 0 |  | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-03_flash_next_pipelined/r29a_rig1/agent | pipe3b | 40 | 9170 | 0.65 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 8 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-03_retype_agent/agent | hcq8 | 62 | 15757 | 1.14 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/s33/agent-report-d | mtp2k | 0 | 0 |  | 0 | 0 | 0 | 0 |  |  |  |  |  | 0 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/s33/agent-report-d | mtp2d | 61 | 20820 | 1.77 | 0 | 1 | 0 | 0 | 0.00 | 0.37 | 30.62 | 0.00 | 20.43 | 12 | 1 | 25.32 | 1 | 0 | 1 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/s34/agent-report | s34nd | 64 | 30076 | 2.91 | 0 | 1 | 1 | 0 | 0.00 | 3.31 | 11.01 | 0.00 | 8.76 | 12 | 1 | 25.37 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/s34/agent-report | s34mtp | 60 | 25961 | 3.87 | 0 | 1 | 0 | 0 | 0.00 | 5.71 | 21.91 | 0.00 | 15.92 | 12 | 1 | 27.98 | 1 | 0 | 1 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/s34/agent-report | s34mtps | 69 | 16809 | 1.37 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/c134r/rig/agent-report | fix2 | 56 | 19010 | 2.35 | 0 | 1 | 1 | 0 | 0.00 | 2.54 | 9.73 | 0.00 | 5.29 | 10 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig2_runner/c134r/rig/agent-report | adapt5 | 62 | 22948 | 1.92 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 1 | 25.89 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-04_rig1_runner/s35/agent-report | s35stack | 67 | 22897 | 2.11 | 0 | 2 | 2 | 0 | 0.00 | 5.74 | 17.36 | 0.00 | 10.56 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-05_s36_stack/s36/agent-report | s36adapt | 68 | 24881 | 0.90 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 1 | 34.24 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-05_s36_stack/s36/agent-report | s36ref | 66 | 20777 | 1.32 | 0 | 1 | 1 | 0 | 0.00 | 1.15 | 1.15 | 0.00 | 0.70 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-05_keepalive/rig_engine/agent-report | s35kaoff | 65 | 21453 | 1.57 | 0 | 1 | 1 | 0 | 0.00 | 4.71 | 6.56 | 0.00 | 4.34 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| flash-next-series | flash_next | 2026-11-05_keepalive/rig_engine/agent-report | s35ka | 63 | 21897 | 1.61 | 0 | 1 | 1 | 0 | 0.00 | 0.49 | 16.51 | 0.00 | 9.68 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-05_keepalive/rig_engine/agent-report | s35kareq | 67 | 18640 | 1.07 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| flash-next-series | flash_next | 2026-11-05_keepalive/rig_engine/agent-report | s35kaidle | 57 | 24528 | 1.15 | 0 | 1 | 1 | 0 | 0.00 | 0.02 | 0.02 | 0.00 | 0.01 | 12 | 2 | 58.71 | 2 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | flint | 2026-10-26_release_setup/agent/cache | rel | 82 | 55858 | 2.02 | 0 | 3 | 3 | 0 | 0.00 | 3.14 | 20.47 | 0.00 | 14.28 | 10 | 1 | 40.75 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/d2 | rel | 60 | 49843 | 2.68 | 0 | 4 | 4 | 0 | 0.00 | 8.38 | 30.00 | 0.00 | 23.46 | 10 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/d4 | rel | 81 | 57882 | 3.42 | 0 | 1 | 1 | 0 | 0.00 | 3.00 | 10.53 | 0.00 | 7.86 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/k2h | rel | 63 | 54789 | 2.58 | 0 | 3 | 3 | 0 | 0.00 | 5.92 | 16.00 | 0.00 | 13.74 | 10 | 1 | 48.20 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/k2h-d4 | rel | 69 | 41443 | 2.78 | 0 | 2 | 2 | 0 | 0.00 | 9.95 | 21.34 | 0.00 | 14.36 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/noguess | rel | 55 | 36709 | 2.53 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 10 | 1 | 47.09 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/page | rel | 77 | 50736 | 2.49 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | flint | 2026-10-26_release_setup/agent/q8-d3b | rel | 63 | 52624 | 3.26 | 0 | 2 | 1 | 0 | 0.00 | 2.82 | 7.51 | 0.00 | 6.02 | 10 | 1 | 46.56 | 1 | 0 | 1 | 0 | 0 | 1 | 3 |
| earlier | flint | 2026-10-26_release_setup/agent/q8-d4b | rel | 77 | 64346 | 1.89 | 0 | 1 | 1 | 0 | 0.00 | 0.19 | 0.71 | 0.00 | 0.59 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/ud-d4 | rel | 70 | 62201 | 3.16 | 0 | 4 | 3 | 0 | 0.00 | 7.38 | 23.44 | 0.00 | 18.98 | 11 | 1 | 44.69 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/ud5-d4 | rel | 68 | 49287 | 3.09 | 0 | 2 | 2 | 0 | 0.00 | 7.83 | 35.41 | 0.00 | 27.07 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/stock | stock | 55 | 28735 | 2.04 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 1 | 31.45 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/stock-k2h | stock | 62 | 31232 | 2.13 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11 | 1 | 30.99 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/stock-ud | stock | 52 | 27592 | 4.49 | 0 | 2 | 1 | 0 | 0.00 | 16.64 | 16.94 | 0.00 | 16.13 | 9 | 1 | 31.94 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-26_release_setup/agent/v1-q8 | stock | 68 | 36162 | 1.73 | 0 | 2 | 1 | 0 | 0.00 | 1.63 | 5.01 | 0.00 | 4.64 | 12 | 1 | 39.63 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_answer_store/agent/base | rel | 70 | 51902 | 3.23 | 0 | 2 | 2 | 0 | 0.00 | 2.88 | 29.43 | 0.00 | 22.88 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_answer_store/agent/store | rel | 66 | 46431 | 2.05 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_flint_near_q8/rig/live | rel | 93 | 55596 | 2.35 | 0 | 5 | 5 | 0 | 0.00 | 4.55 | 4.96 | 0.00 | 3.67 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_mtp_head/agent/base | rel | 73 | 46581 | 1.89 | 0 | 1 | 1 | 0 | 0.00 | 2.59 | 2.59 | 0.00 | 1.74 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_mtp_head/agent/gen | rel | 57 | 45203 | 3.03 | 0 | 4 | 3 | 0 | 0.00 | 5.42 | 44.66 | 0.00 | 39.81 | 9 | 1 | 39.82 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-27_mtp_head/agent/nine | rel | 74 | 38770 | 1.71 | 1 | 0 | 0 | 0 | 0.17 | 0.00 | 0.00 | 0.16 | 0.00 | 10 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-28_reply_rereads/agent/tr | rel | 82 | 52035 | 2.63 | 0 | 5 | 5 | 0 | 0.00 | 5.70 | 22.15 | 0.00 | 17.82 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | flint | 2026-10-28_reply_rereads/agent/trtail | rel | 73 | 42691 | 2.68 | 0 | 2 | 2 | 0 | 0.00 | 6.49 | 19.54 | 0.00 | 15.20 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/stock | stock | 31 | 8730 | 2.20 | 0 | 1 | 0 | 0 | 0.00 | 5.59 | 16.09 | 0.00 | 13.18 | 7 | 1 | 33.80 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/stock-mtp | stock | 45 | 12939 | 1.57 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/ud-stock | stock | 42 | 10367 | 1.83 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 1 | 30.99 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/ud-v2 | rel | 59 | 14178 | 0.78 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/ud-v3 | rel | 59 | 17282 | 1.22 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/v2 | rel | 37 | 17567 | 1.61 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 2 | 56.43 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| earlier | ornith | 2026-10-27_ornith/rig/agent/v2-mtp | rel | 40 | 12351 | 1.09 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 9 | 1 | 33.79 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| earlier | ornith | 2026-10-28_lowbit_rig/agent/valid1 | rel | 58 | 17181 | 1.23 | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12 | 0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |

