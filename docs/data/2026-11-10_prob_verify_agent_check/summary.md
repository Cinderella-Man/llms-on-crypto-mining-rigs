Runs in the order they started: pv0-1, pv1-1, pv0-2, pv1-2; 12 tasks in every run.

Per run:

| run | arm | tasks | passed | failed_tasks | turns | requests | generated | tool_calls | tool_errors | malformed_calls | length_stops | runaways | infra_tries | drafted | accepted | share_accepted | tokens_per_step | generated_per_turn | median_reply | largest_reply | replies_1000_plus | first_reply | second_reply | later_replies | drawn_on_device | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pv0-1 | pv0 | 12 | 12 | - | 92 | 92 | 34945 | 87 | 0 | 0 | 0 | - | 0 | 35229 | 23129 | 0.6565 | 2.957 | 379.8 | 190.5 | 4694 | 7 | 792 | 11826 | 22327 | 34964 | 4 |
| pv1-1 | pv1 | 12 | 12 | - | 64 | 64 | 35406 | 59 | 1 | 0 | 0 | - | 0 | 31644 | 24874 | 0.7861 | 3.362 | 553.2 | 149.5 | 17785 | 5 | 801 | 25457 | 9148 | 35486 | 22 |
| pv0-2 | pv0 | 12 | 12 | - | 67 | 67 | 21936 | 60 | 2 | 0 | 0 | - | 0 | 21399 | 14752 | 0.6894 | 3.053 | 327.4 | 165 | 4071 | 2 | 797 | 10783 | 10356 | 21952 | 2 |
| pv1-2 | pv1 | 12 | 12 | - | 60 | 60 | 36683 | 56 | 0 | 0 | 0 | - | 0 | 33114 | 25665 | 0.775 | 3.329 | 611.4 | 174.0 | 9445 | 6 | 861 | 27222 | 8600 | 36763 | 26 |

`pv1` against `pv0` (difference of the means over runs; the interval is a 95 % bootstrap over tasks of the difference in the run's sum):

| measure | pv0-1 | pv1-1 | pv0-2 | pv1-2 | pv0_mean | pv1_mean | difference | difference_pct | pv0_2_minus_1 | pv1_2_minus_1 | within_ref_spread | ci95_low | ci95_high | tasks_higher | tasks_lower | tasks_equal | pv0_2_vs_1_higher | pv0_2_vs_1_lower | pv0_2_vs_1_equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passed | 12 | 12 | 12 | 12 | 12.0 | 12.0 | 0.0 | 0.0 | 0.0 | 0.0 | yes | 0.0 | 0.0 | 0 | 0 | 12 | 0 | 0 | 12 |
| turns | 92 | 64 | 67 | 60 | 79.5 | 62.0 | -17.5 | -22.01 | -25.0 | -4.0 | yes | -45.5 | -1.0 | 1 | 7 | 4 | 2 | 4 | 6 |
| generated | 34945 | 35406 | 21936 | 36683 | 28440.5 | 36044.5 | 7604.0 | 26.74 | -13009.0 | 1277.0 | yes | -9418.5 | 29236.5 | 7 | 5 | 0 | 6 | 6 | 0 |
| tool_calls | 87 | 59 | 60 | 56 | 73.5 | 57.5 | -16.0 | -21.77 | -27.0 | -3.0 | yes | -42.5 | 0.0 | 3 | 7 | 2 | 2 | 6 | 4 |
| tool_errors | 0 | 1 | 2 | 0 | 1.0 | 0.5 | -0.5 | -50.0 | 2.0 | -1.0 | yes | -2.0 | 1.0 | 1 | 2 | 9 | 2 | 0 | 10 |
| malformed_calls | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.0 | yes | 0.0 | 0.0 | 0 | 0 | 12 | 0 | 0 | 12 |
| first_reply | 792 | 801 | 797 | 861 | 794.5 | 831.0 | 36.5 | 4.59 | 5.0 | 60.0 | no | -21.0 | 109.5 | 6 | 4 | 2 | 6 | 5 | 1 |
| second_reply | 11826 | 25457 | 10783 | 27222 | 11304.5 | 26339.5 | 15035.0 | 133.0 | -1043.0 | 1765.0 | no | 1712.0 | 36570.0 | 9 | 3 | 0 | 4 | 8 | 0 |
| later_replies | 22327 | 9148 | 10356 | 8600 | 16341.5 | 8874.0 | -7467.5 | -45.7 | -11971.0 | -548.0 | yes | -17321.5 | -813.5 | 3 | 9 | 0 | 7 | 5 | 0 |
| generated_per_turn | 379.8 | 553.2 | 327.4 | 611.4 | 353.6 | 582.3 | 228.7 | 64.68 | -52.4 | 58.2 | no |  |  |  |  |  |  |  |  |
| median_reply | 190.5 | 149.5 | 165 | 174.0 | 177.75 | 161.75 | -16.0 | -9.0 | -25.5 | 24.5 | yes |  |  |  |  |  |  |  |  |
| replies_1000_plus | 7 | 5 | 2 | 6 | 4.5 | 5.5 | 1.0 | 22.22 | -5.0 | 1.0 | yes |  |  |  |  |  |  |  |  |
| share_accepted | 0.6565 | 0.7861 | 0.6894 | 0.775 | 0.67295 | 0.78055 | 0.1076 | 15.99 | 0.0329 | -0.0111 | no |  |  |  |  |  |  |  |  |
| tokens_per_step | 2.957 | 3.362 | 3.053 | 3.329 | 3.005 | 3.3455 | 0.3405 | 11.33 | 0.096 | -0.033 | no |  |  |  |  |  |  |  |  |

Per task:

| task_id | passed_pv0-1 | passed_pv1-1 | passed_pv0-2 | passed_pv1-2 | turns_pv0-1 | turns_pv1-1 | turns_pv0-2 | turns_pv1-2 | generated_pv0-1 | generated_pv1-1 | generated_pv0-2 | generated_pv1-2 | tokens_per_step_pv0-1 | tokens_per_step_pv1-1 | tokens_per_step_pv0-2 | tokens_per_step_pv1-2 | first_reply_pv0-1 | first_reply_pv1-1 | first_reply_pv0-2 | first_reply_pv1-2 | second_reply_pv0-1 | second_reply_pv1-1 | second_reply_pv0-2 | second_reply_pv1-2 | turns_pv1_minus_pv0 | turns_pv0_2_minus_1 | turns_pv1_2_minus_1 | generated_pv1_minus_pv0 | generated_pv0_2_minus_1 | generated_pv1_2_minus_1 | first_reply_pv1_minus_pv0 | first_reply_pv0_2_minus_1 | first_reply_pv1_2_minus_1 | second_reply_pv1_minus_pv0 | second_reply_pv0_2_minus_1 | second_reply_pv1_2_minus_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-javascript-complex-numbers | 1 | 1 | 1 | 1 | 29 | 5 | 8 | 5 | 9427 | 1034 | 760 | 742 | 2.614 | 3.447 | 3.22 | 3.419 | 52 | 83 | 44 | 72 | 42 | 428 | 36 | 246 | -13.5 | -21.0 | 0.0 | -4205.5 | -8667.0 | -292.0 | 29.5 | -8.0 | -11.0 | 298.0 | -6.0 | -182.0 |
| bugfix-javascript-grade-school | 1 | 1 | 1 | 1 | 7 | 5 | 4 | 5 | 890 | 544 | 628 | 625 | 3.248 | 3.532 | 3.034 | 3.434 | 57 | 53 | 63 | 55 | 102 | 119 | 283 | 100 | -0.5 | -3.0 | 0.0 | -174.5 | -262.0 | 81.0 | -6.0 | 6.0 | 2.0 | -83.0 | 181.0 | -19.0 |
| bugfix-javascript-phone-number | 1 | 1 | 1 | 1 | 5 | 5 | 5 | 5 | 633 | 715 | 763 | 687 | 3.23 | 3.405 | 3.127 | 3.401 | 51 | 51 | 51 | 56 | 203 | 110 | 171 | 38 | 0.0 | 0.0 | 0.0 | 3.0 | 130.0 | -28.0 | 2.5 | 0.0 | 5.0 | -113.0 | -32.0 | -72.0 |
| bugfix-javascript-space-age | 1 | 1 | 1 | 1 | 5 | 5 | 5 | 5 | 662 | 791 | 656 | 646 | 3.378 | 3.595 | 3.296 | 3.511 | 51 | 57 | 61 | 49 | 189 | 381 | 184 | 161 | 0.0 | 0.0 | 0.0 | 59.5 | -6.0 | -145.0 | -3.0 | 10.0 | -8.0 | 84.5 | -5.0 | -220.0 |
| bugfix-javascript-triangle | 1 | 1 | 1 | 1 | 5 | 5 | 5 | 4 | 540 | 669 | 662 | 600 | 3.273 | 3.484 | 3.294 | 3.55 | 65 | 58 | 53 | 61 | 229 | 253 | 174 | 270 | -0.5 | 0.0 | -1.0 | 33.5 | 122.0 | -69.0 | 0.5 | -12.0 | 3.0 | 60.0 | -55.0 | 17.0 |
| bugfix-javascript-variable-length-quantity | 1 | 1 | 1 | 1 | 8 | 7 | 5 | 6 | 8859 | 19770 | 4877 | 11578 | 3.139 | 3.441 | 3.095 | 3.47 | 74 | 91 | 85 | 81 | 2872 | 17785 | 4071 | 9445 | 0.0 | -3.0 | -1.0 | 8806.0 | -3982.0 | -8192.0 | 6.5 | 11.0 | -10.0 | 10143.5 | 1199.0 | -8340.0 |
| bugfix-python-affine-cipher | 1 | 1 | 1 | 1 | 6 | 5 | 8 | 5 | 1895 | 1721 | 2602 | 1941 | 2.938 | 3.106 | 2.933 | 3.156 | 75 | 75 | 83 | 83 | 378 | 750 | 272 | 687 | -2.0 | 2.0 | 0.0 | -417.5 | 707.0 | 220.0 | 0.0 | 8.0 | 8.0 | 393.5 | -106.0 | -63.0 |
| bugfix-python-list-ops | 1 | 1 | 1 | 1 | 5 | 5 | 5 | 5 | 2295 | 2514 | 1431 | 2137 | 3.036 | 3.231 | 3.051 | 3.273 | 75 | 75 | 45 | 75 | 1739 | 1863 | 158 | 1580 | 0.0 | 0.0 | 0.0 | 462.5 | -864.0 | -377.0 | 15.0 | -30.0 | 0.0 | 773.0 | -1581.0 | -283.0 |
| bugfix-python-phone-number | 1 | 1 | 1 | 1 | 6 | 5 | 5 | 5 | 969 | 903 | 1085 | 994 | 2.867 | 3.369 | 3.1 | 3.358 | 47 | 47 | 99 | 84 | 112 | 439 | 238 | 177 | -0.5 | -1.0 | 0.0 | -78.5 | 116.0 | 91.0 | -7.5 | 52.0 | 37.0 | 133.0 | 126.0 | -262.0 |
| bugfix-python-pig-latin | 1 | 1 | 1 | 1 | 5 | 5 | 6 | 5 | 2214 | 1494 | 2481 | 5753 | 2.936 | 3.192 | 3.015 | 3.149 | 80 | 80 | 77 | 80 | 723 | 753 | 598 | 4699 | -0.5 | 1.0 | 0.0 | 1276.0 | 267.0 | 4259.0 | 1.5 | -3.0 | 0.0 | 2065.5 | -125.0 | 3946.0 |
| bugfix-python-scale-generator | 1 | 1 | 1 | 1 | 5 | 7 | 5 | 5 | 1905 | 3249 | 4740 | 7726 | 2.944 | 3.112 | 2.951 | 3.239 | 78 | 78 | 83 | 78 | 1294 | 2012 | 4069 | 7145 | 1.0 | 0.0 | -2.0 | 2165.0 | 2835.0 | 4477.0 | -2.5 | 5.0 | 0.0 | 1897.0 | 2775.0 | 5133.0 |
| bugfix-python-variable-length-quantity | 1 | 1 | 1 | 1 | 6 | 5 | 6 | 5 | 4656 | 2002 | 1251 | 3254 | 3.286 | 3.348 | 3.241 | 3.422 | 87 | 53 | 53 | 87 | 3943 | 564 | 529 | 2674 | -1.0 | 0.0 | 0.0 | -325.5 | -3405.0 | 1252.0 | 0.0 | -34.0 | 34.0 | -617.0 | -3414.0 | 2110.0 |

Run `pv0-1`, per task:

| task_id | status | passed | tests_passed | tests_total | tests_untouched | turns | requests | generated | drafted | accepted | share_accepted | tokens_per_step | tool_calls | tool_errors | malformed_calls | length_stops | runaway | infra_tries | context_last | first_reply | second_reply | later_replies | largest_reply | drawn_on_device | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-javascript-complex-numbers | scored | 1 | 40 | 40 | 1 | 29 | 29 | 9427 | 10743 | 5821 | 0.5418 | 2.614 | 28 | 0 | 0 | 0 |  | 0 | 22492 | 52 | 42 | 9333 | 1158 | 9431 | 2 |
| bugfix-javascript-grade-school | scored | 1 | 20 | 20 | 1 | 7 | 7 | 890 | 807 | 616 | 0.7633 | 3.248 | 6 | 0 | 0 | 0 |  | 0 | 5332 | 57 | 102 | 731 | 192 | 892 | 1 |
| bugfix-javascript-phone-number | scored | 1 | 18 | 18 | 1 | 5 | 5 | 633 | 576 | 437 | 0.7587 | 3.23 | 4 | 0 | 0 | 0 |  | 0 | 4792 | 51 | 203 | 379 | 203 | 634 | 0 |
| bugfix-javascript-space-age | scored | 1 | 9 | 9 | 1 | 5 | 5 | 662 | 576 | 466 | 0.809 | 3.378 | 4 | 0 | 0 | 0 |  | 0 | 4170 | 51 | 189 | 422 | 195 | 663 | 0 |
| bugfix-javascript-triangle | scored | 1 | 21 | 21 | 1 | 5 | 5 | 540 | 486 | 375 | 0.7716 | 3.273 | 5 | 0 | 0 | 0 |  | 0 | 4212 | 65 | 229 | 246 | 229 | 542 | 0 |
| bugfix-javascript-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 8 | 8 | 8859 | 8445 | 6037 | 0.7149 | 3.139 | 8 | 0 | 0 | 0 |  | 0 | 14155 | 74 | 2872 | 5913 | 4694 | 8860 | 0 |
| bugfix-python-affine-cipher | scored | 1 | 16 | 16 | 1 | 6 | 6 | 1895 | 1926 | 1250 | 0.649 | 2.938 | 6 | 0 | 0 | 0 |  | 0 | 6331 | 75 | 378 | 1442 | 541 | 1898 | 0 |
| bugfix-python-list-ops | scored | 1 | 25 | 25 | 1 | 5 | 5 | 2295 | 2256 | 1539 | 0.6822 | 3.036 | 5 | 0 | 0 | 0 |  | 0 | 5989 | 75 | 1739 | 481 | 1739 | 2296 | 0 |
| bugfix-python-phone-number | scored | 1 | 21 | 21 | 1 | 6 | 6 | 969 | 999 | 631 | 0.6316 | 2.867 | 5 | 0 | 0 | 0 |  | 0 | 4836 | 47 | 112 | 810 | 276 | 970 | 0 |
| bugfix-python-pig-latin | scored | 1 | 23 | 23 | 1 | 5 | 5 | 2214 | 2250 | 1460 | 0.6489 | 2.936 | 5 | 0 | 0 | 0 |  | 0 | 5729 | 80 | 723 | 1411 | 1011 | 2215 | 0 |
| bugfix-python-scale-generator | scored | 1 | 17 | 17 | 1 | 5 | 5 | 1905 | 1929 | 1258 | 0.6522 | 2.944 | 5 | 0 | 0 | 0 |  | 0 | 6383 | 78 | 1294 | 533 | 1294 | 1906 | 1 |
| bugfix-python-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 6 | 6 | 4656 | 4236 | 3239 | 0.7646 | 3.286 | 6 | 0 | 0 | 0 |  | 0 | 8915 | 87 | 3943 | 626 | 3943 | 4657 | 0 |

Run `pv1-1`, per task:

| task_id | status | passed | tests_passed | tests_total | tests_untouched | turns | requests | generated | drafted | accepted | share_accepted | tokens_per_step | tool_calls | tool_errors | malformed_calls | length_stops | runaway | infra_tries | context_last | first_reply | second_reply | later_replies | largest_reply | drawn_on_device | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-javascript-complex-numbers | scored | 1 | 40 | 40 | 1 | 5 | 5 | 1034 | 906 | 734 | 0.8102 | 3.447 | 5 | 0 | 0 | 0 |  | 0 | 7024 | 83 | 428 | 523 | 428 | 1041 | 1 |
| bugfix-javascript-grade-school | scored | 1 | 20 | 20 | 1 | 5 | 5 | 544 | 465 | 390 | 0.8387 | 3.532 | 4 | 0 | 0 | 0 |  | 0 | 4797 | 53 | 119 | 372 | 158 | 550 | 1 |
| bugfix-javascript-phone-number | scored | 1 | 18 | 18 | 1 | 5 | 5 | 715 | 624 | 505 | 0.8093 | 3.405 | 4 | 0 | 0 | 0 |  | 0 | 4921 | 51 | 110 | 554 | 255 | 718 | 1 |
| bugfix-javascript-space-age | scored | 1 | 9 | 9 | 1 | 5 | 5 | 791 | 675 | 571 | 0.8459 | 3.595 | 4 | 0 | 0 | 0 |  | 0 | 4475 | 57 | 381 | 353 | 381 | 801 | 3 |
| bugfix-javascript-triangle | scored | 1 | 21 | 21 | 1 | 5 | 5 | 669 | 573 | 477 | 0.8325 | 3.484 | 4 | 0 | 0 | 0 |  | 0 | 4565 | 58 | 253 | 358 | 253 | 673 | 1 |
| bugfix-javascript-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 7 | 7 | 19770 | 17250 | 14024 | 0.813 | 3.441 | 8 | 1 | 0 | 0 |  | 0 | 25316 | 91 | 17785 | 1894 | 17785 | 19781 | 4 |
| bugfix-python-affine-cipher | scored | 1 | 16 | 16 | 1 | 5 | 5 | 1721 | 1665 | 1167 | 0.7009 | 3.106 | 5 | 0 | 0 | 0 |  | 0 | 6199 | 75 | 750 | 896 | 750 | 1727 | 1 |
| bugfix-python-list-ops | scored | 1 | 25 | 25 | 1 | 5 | 5 | 2514 | 2331 | 1736 | 0.7447 | 3.231 | 5 | 0 | 0 | 0 |  | 0 | 6221 | 75 | 1863 | 576 | 1863 | 2518 | 2 |
| bugfix-python-phone-number | scored | 1 | 21 | 21 | 1 | 5 | 5 | 903 | 813 | 635 | 0.7811 | 3.369 | 4 | 0 | 0 | 0 |  | 0 | 4346 | 47 | 439 | 417 | 439 | 911 | 2 |
| bugfix-python-pig-latin | scored | 1 | 23 | 23 | 1 | 5 | 5 | 1494 | 1413 | 1026 | 0.7261 | 3.192 | 5 | 0 | 0 | 0 |  | 0 | 4945 | 80 | 753 | 661 | 753 | 1502 | 2 |
| bugfix-python-scale-generator | scored | 1 | 17 | 17 | 1 | 7 | 7 | 3249 | 3132 | 2205 | 0.704 | 3.112 | 7 | 0 | 0 | 0 |  | 0 | 8005 | 78 | 2012 | 1159 | 2012 | 3256 | 2 |
| bugfix-python-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 5 | 5 | 2002 | 1797 | 1404 | 0.7813 | 3.348 | 4 | 0 | 0 | 0 |  | 0 | 5709 | 53 | 564 | 1385 | 1174 | 2008 | 2 |

Run `pv0-2`, per task:

| task_id | status | passed | tests_passed | tests_total | tests_untouched | turns | requests | generated | drafted | accepted | share_accepted | tokens_per_step | tool_calls | tool_errors | malformed_calls | length_stops | runaway | infra_tries | context_last | first_reply | second_reply | later_replies | largest_reply | drawn_on_device | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-javascript-complex-numbers | scored | 1 | 40 | 40 | 1 | 8 | 8 | 760 | 690 | 524 | 0.7594 | 3.22 | 7 | 1 | 0 | 0 |  | 0 | 4581 | 44 | 36 | 680 | 171 | 762 | 0 |
| bugfix-javascript-grade-school | scored | 1 | 20 | 20 | 1 | 4 | 4 | 628 | 612 | 421 | 0.6879 | 3.034 | 3 | 0 | 0 | 0 |  | 0 | 4890 | 63 | 283 | 282 | 283 | 629 | 1 |
| bugfix-javascript-phone-number | scored | 1 | 18 | 18 | 1 | 5 | 5 | 763 | 720 | 519 | 0.7208 | 3.127 | 4 | 0 | 0 | 0 |  | 0 | 4893 | 51 | 171 | 541 | 290 | 764 | 0 |
| bugfix-javascript-space-age | scored | 1 | 9 | 9 | 1 | 5 | 5 | 656 | 588 | 457 | 0.7772 | 3.296 | 4 | 0 | 0 | 0 |  | 0 | 4199 | 61 | 184 | 411 | 184 | 658 | 0 |
| bugfix-javascript-triangle | scored | 1 | 21 | 21 | 1 | 5 | 5 | 662 | 591 | 461 | 0.78 | 3.294 | 4 | 0 | 0 | 0 |  | 0 | 4449 | 53 | 174 | 435 | 208 | 663 | 0 |
| bugfix-javascript-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 5 | 5 | 4877 | 4716 | 3301 | 0.7 | 3.095 | 5 | 0 | 0 | 0 |  | 0 | 9792 | 85 | 4071 | 721 | 4071 | 4878 | 0 |
| bugfix-python-affine-cipher | scored | 1 | 16 | 16 | 1 | 8 | 8 | 2602 | 2640 | 1715 | 0.6496 | 2.933 | 8 | 1 | 0 | 0 |  | 0 | 7218 | 83 | 272 | 2247 | 989 | 2603 | 1 |
| bugfix-python-list-ops | scored | 1 | 25 | 25 | 1 | 5 | 5 | 1431 | 1395 | 962 | 0.6896 | 3.051 | 4 | 0 | 0 | 0 |  | 0 | 5150 | 45 | 158 | 1228 | 930 | 1432 | 0 |
| bugfix-python-phone-number | scored | 1 | 21 | 21 | 1 | 5 | 5 | 1085 | 1038 | 735 | 0.7081 | 3.1 | 5 | 0 | 0 | 0 |  | 0 | 6109 | 99 | 238 | 748 | 365 | 1086 | 0 |
| bugfix-python-pig-latin | scored | 1 | 23 | 23 | 1 | 6 | 6 | 2481 | 2457 | 1658 | 0.6748 | 3.015 | 6 | 0 | 0 | 0 |  | 0 | 5952 | 77 | 598 | 1806 | 867 | 2483 | 0 |
| bugfix-python-scale-generator | scored | 1 | 17 | 17 | 1 | 5 | 5 | 4740 | 4806 | 3134 | 0.6521 | 2.951 | 5 | 0 | 0 | 0 |  | 0 | 9385 | 83 | 4069 | 588 | 4069 | 4741 | 0 |
| bugfix-python-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 6 | 6 | 1251 | 1146 | 865 | 0.7548 | 3.241 | 5 | 0 | 0 | 0 |  | 0 | 5769 | 53 | 529 | 669 | 529 | 1253 | 0 |

Run `pv1-2`, per task:

| task_id | status | passed | tests_passed | tests_total | tests_untouched | turns | requests | generated | drafted | accepted | share_accepted | tokens_per_step | tool_calls | tool_errors | malformed_calls | length_stops | runaway | infra_tries | context_last | first_reply | second_reply | later_replies | largest_reply | drawn_on_device | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-javascript-complex-numbers | scored | 1 | 40 | 40 | 1 | 5 | 5 | 742 | 660 | 525 | 0.7955 | 3.419 | 5 | 0 | 0 | 0 |  | 0 | 6715 | 72 | 246 | 424 | 246 | 750 | 3 |
| bugfix-javascript-grade-school | scored | 1 | 20 | 20 | 1 | 5 | 5 | 625 | 549 | 443 | 0.8069 | 3.434 | 4 | 0 | 0 | 0 |  | 0 | 4796 | 55 | 100 | 470 | 220 | 631 | 2 |
| bugfix-javascript-phone-number | scored | 1 | 18 | 18 | 1 | 5 | 5 | 687 | 612 | 485 | 0.7925 | 3.401 | 4 | 0 | 0 | 0 |  | 0 | 4120 | 56 | 38 | 593 | 257 | 694 | 1 |
| bugfix-javascript-space-age | scored | 1 | 9 | 9 | 1 | 5 | 5 | 646 | 561 | 462 | 0.8235 | 3.511 | 4 | 0 | 0 | 0 |  | 0 | 4259 | 49 | 161 | 436 | 182 | 654 | 2 |
| bugfix-javascript-triangle | scored | 1 | 21 | 21 | 1 | 4 | 4 | 600 | 519 | 431 | 0.8304 | 3.55 | 3 | 0 | 0 | 0 |  | 0 | 4428 | 61 | 270 | 269 | 270 | 608 | 3 |
| bugfix-javascript-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 6 | 6 | 11578 | 10014 | 8241 | 0.8229 | 3.47 | 6 | 0 | 0 | 0 |  | 0 | 15844 | 81 | 9445 | 2052 | 9445 | 11585 | 3 |
| bugfix-python-affine-cipher | scored | 1 | 16 | 16 | 1 | 5 | 5 | 1941 | 1839 | 1326 | 0.721 | 3.156 | 5 | 0 | 0 | 0 |  | 0 | 6495 | 83 | 687 | 1171 | 823 | 1944 | 1 |
| bugfix-python-list-ops | scored | 1 | 25 | 25 | 1 | 5 | 5 | 2137 | 1968 | 1484 | 0.7541 | 3.273 | 5 | 0 | 0 | 0 |  | 0 | 5887 | 75 | 1580 | 482 | 1580 | 2145 | 2 |
| bugfix-python-phone-number | scored | 1 | 21 | 21 | 1 | 5 | 5 | 994 | 903 | 698 | 0.773 | 3.358 | 5 | 0 | 0 | 0 |  | 0 | 6056 | 84 | 177 | 733 | 449 | 1004 | 3 |
| bugfix-python-pig-latin | scored | 1 | 23 | 23 | 1 | 5 | 5 | 5753 | 5478 | 3926 | 0.7167 | 3.149 | 5 | 0 | 0 | 0 |  | 0 | 9181 | 80 | 4699 | 974 | 4699 | 5757 | 2 |
| bugfix-python-scale-generator | scored | 1 | 17 | 17 | 1 | 5 | 5 | 7726 | 7152 | 5341 | 0.7468 | 3.239 | 5 | 0 | 0 | 0 |  | 0 | 12715 | 78 | 7145 | 503 | 7145 | 7730 | 2 |
| bugfix-python-variable-length-quantity | scored | 1 | 31 | 31 | 1 | 5 | 5 | 3254 | 2859 | 2303 | 0.8055 | 3.422 | 5 | 0 | 0 | 0 |  | 0 | 7728 | 87 | 2674 | 493 | 2674 | 3261 | 2 |

