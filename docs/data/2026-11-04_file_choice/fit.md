Largest `-ub` that leaves 150 MiB on every card (free MiB on the fullest card; `no (x)`: the best of any `-ub` falls short), nine P106-090 cards. Whole layers / with routed-expert tensors moved by `-ot` (how many).

### Draft: none

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| ISTA IQ3_S | f16 | no (-472) / no (-153) | no (-573) / no (-214) | no (-674) / no (-281) | no (-983) / no (-533) | no (-1498) / no (-1181) |
| ISTA IQ3_S | q8_0 | no (-451) / no (-163) | no (-540) / no (-237) | no (-627) / no (-303) | no (-978) / no (-582) | no (-1564) / no (-1114) |
| Swift IQ3_S | f16 | no (-487) / no (-166) | no (-588) / no (-227) | no (-689) / no (-296) | no (-998) / no (-548) | no (-1513) / no (-1194) |
| Swift IQ3_S | q8_0 | no (-466) / no (-176) | no (-555) / no (-250) | no (-642) / no (-316) | no (-993) / no (-597) | no (-1579) / no (-1129) |
| Swift IQ3_XXS | f16 | ub 256: 156 / ub 256: 580 (11 moved) | ub 128: 189 / ub 256: 499 (8 moved) | ub 64: 155 / ub 256: 387 (8 moved) | no (-89) / ub 128: 159 (10 moved) | no (-598) / no (-297) |
| Swift IQ3_XXS | q8_0 | ub 128: 264 / ub 256: 567 (11 moved) | ub 128: 177 / ub 256: 460 (11 moved) | ub 64: 193 / ub 256: 355 (11 moved) | no (-138) / ub 128: 152 (8 moved) | no (-670) / no (-322) |
| Swift IQ3_XXS-hcq8 | f16 | ub 256: 227 / ub 256: 643 (11 moved) | ub 128: 289 / ub 256: 562 (8 moved) | ub 128: 187 / ub 256: 459 (8 moved) | no (11) / ub 128: 233 (10 moved) | no (-526) / no (-211) |
| Swift IQ3_XXS-hcq8 | q8_0 | ub 256: 214 / ub 256: 630 (11 moved) | ub 128: 248 / ub 256: 523 (11 moved) | ub 128: 166 / ub 256: 418 (11 moved) | no (-38) / ub 128: 224 (8 moved) | no (-570) / no (-239) |
| Swift IQ2_XS | f16 | ub 256: 1075 / ub 256: 1384 (13 moved) | ub 256: 975 / ub 256: 1295 (11 moved) | ub 512: 569 / ub 512: 879 (11 moved) | ub 256: 453 / ub 512: 224 (17 moved) | no (10) / ub 128: 230 (16 moved) |
| Swift IQ2_XS | q8_0 | ub 256: 1100 / ub 256: 1390 (15 moved) | ub 256: 971 / ub 256: 1264 (13 moved) | ub 512: 499 / ub 512: 813 (13 moved) | ub 256: 372 / ub 512: 172 (11 moved) | ub 64: 219 / ub 128: 259 (17 moved) |
| ISTA Coder IQ1_M | f16 | ub 256: 2249 / ub 256: 2440 (6 moved) | ub 256: 2135 / ub 256: 2333 (6 moved) | ub 512: 1719 / ub 512: 1922 (5 moved) | ub 512: 1124 / ub 512: 1255 (5 moved) | ub 256: 581 / ub 256: 937 (11 moved) |
| ISTA Coder IQ1_M | q8_0 | ub 256: 2274 / ub 256: 2427 (6 moved) | ub 256: 2145 / ub 256: 2320 (6 moved) | ub 512: 1673 / ub 512: 1869 (6 moved) | ub 512: 1012 / ub 512: 1216 (5 moved) | ub 256: 656 / ub 256: 796 (7 moved) |
| Swift Q2_0-hcq8 | f16 | ub 256: 1319 / ub 256: 1626 (11 moved) | ub 256: 1238 / ub 256: 1545 (11 moved) | ub 512: 822 / ub 512: 1112 (10 moved) | ub 512: 200 / ub 512: 393 (14 moved) | ub 64: 298 / ub 128: 502 (16 moved) |
| Swift Q2_0-hcq8 | q8_0 | ub 256: 1306 / ub 256: 1613 (11 moved) | ub 256: 1199 / ub 256: 1506 (11 moved) | ub 512: 748 / ub 512: 1055 (11 moved) | ub 256: 652 / ub 512: 402 (10 moved) | ub 128: 217 / ub 128: 410 (14 moved) |
| Swift Q2_0 | f16 | ub 256: 1234 / ub 256: 1555 (11 moved) | ub 256: 1153 / ub 256: 1474 (11 moved) | ub 512: 737 / ub 512: 1034 (10 moved) | ub 256: 664 / ub 512: 307 (14 moved) | ub 64: 221 / ub 128: 416 (16 moved) |
| Swift Q2_0 | q8_0 | ub 256: 1221 / ub 256: 1542 (11 moved) | ub 256: 1114 / ub 256: 1435 (11 moved) | ub 512: 663 / ub 512: 984 (11 moved) | ub 256: 567 / ub 512: 320 (11 moved) | ub 64: 430 / ub 128: 324 (14 moved) |

### Draft: dflash2

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| ISTA IQ3_S | f16 | no (-509) / no (-223) | no (-610) / no (-324) | no (-711) / no (-398) | no (-1029) / no (-704) | no (-1536) / no (-1325) |
| ISTA IQ3_S | q8_0 | no (-488) / no (-208) | no (-577) / no (-291) | no (-664) / no (-378) | no (-1015) / no (-708) | no (-1610) / no (-1330) |
| Swift IQ3_S | f16 | no (-524) / no (-238) | no (-625) / no (-339) | no (-726) / no (-411) | no (-1045) / no (-719) | no (-1551) / no (-1338) |
| Swift IQ3_S | q8_0 | no (-503) / no (-220) | no (-592) / no (-306) | no (-679) / no (-393) | no (-1030) / no (-715) | no (-1626) / no (-1337) |
| Swift IQ3_XXS | f16 | ub 128: 244 / ub 256: 409 (11 moved) | ub 64: 202 / ub 256: 299 (9 moved) | no (101) / ub 256: 202 (9 moved) | no (-145) / ub 64: 189 (13 moved) | no (-635) / no (-493) |
| Swift IQ3_XXS | q8_0 | ub 128: 227 / ub 256: 432 (10 moved) | ub 64: 222 / ub 256: 303 (10 moved) | no (148) / ub 256: 184 (11 moved) | no (-194) / no (140) | no (-726) / no (-392) |
| Swift IQ3_XXS-hcq8 | f16 | ub 256: 190 / ub 256: 504 (11 moved) | ub 128: 243 / ub 256: 383 (11 moved) | ub 64: 201 / ub 256: 287 (9 moved) | no (-45) / ub 128: 153 (13 moved) | no (-554) / no (-421) |
| Swift IQ3_XXS-hcq8 | q8_0 | ub 256: 177 / ub 256: 518 (10 moved) | ub 128: 211 / ub 256: 390 (11 moved) | ub 64: 227 / ub 256: 274 (11 moved) | no (-94) / ub 64: 226 (9 moved) | no (-626) / no (-319) |
| Swift IQ2_XS | f16 | ub 256: 1002 / ub 256: 1268 (15 moved) | ub 256: 921 / ub 256: 1187 (15 moved) | ub 512: 505 / ub 512: 762 (11 moved) | ub 256: 411 / ub 256: 659 (8 moved) | no (-32) / ub 128: 156 (10 moved) |
| Swift IQ2_XS | q8_0 | ub 256: 989 / ub 256: 1255 (15 moved) | ub 256: 882 / ub 256: 1148 (15 moved) | ub 512: 431 / ub 512: 697 (15 moved) | ub 256: 335 / ub 256: 592 (11 moved) | ub 64: 177 / ub 64: 420 (8 moved) |
| ISTA Coder IQ1_M | f16 | ub 256: 2169 / ub 256: 2336 (6 moved) | ub 256: 2088 / ub 256: 2236 (7 moved) | ub 512: 1634 / ub 512: 1820 (9 moved) | ub 512: 881 / ub 512: 1170 (10 moved) | ub 256: 462 / ub 256: 759 (17 moved) |
| ISTA Coder IQ1_M | q8_0 | ub 256: 2172 / ub 256: 2324 (6 moved) | ub 256: 2049 / ub 256: 2217 (6 moved) | ub 512: 1598 / ub 512: 1760 (6 moved) | ub 512: 916 / ub 512: 1113 (9 moved) | ub 256: 413 / ub 256: 702 (10 moved) |
| Swift Q2_0-hcq8 | f16 | ub 256: 1236 / ub 256: 1515 (12 moved) | ub 256: 1115 / ub 256: 1434 (12 moved) | ub 512: 659 / ub 512: 1018 (12 moved) | ub 256: 447 / ub 512: 356 (12 moved) | no (4) / ub 128: 421 (11 moved) |
| Swift Q2_0-hcq8 | q8_0 | ub 256: 1261 / ub 256: 1512 (8 moved) | ub 256: 1132 / ub 256: 1395 (12 moved) | ub 512: 660 / ub 512: 944 (12 moved) | ub 256: 478 / ub 512: 311 (12 moved) | ub 64: 213 / ub 128: 373 (12 moved) |
| Swift Q2_0 | f16 | ub 256: 1150 / ub 256: 1430 (12 moved) | ub 256: 1029 / ub 256: 1349 (12 moved) | ub 512: 573 / ub 512: 933 (12 moved) | ub 256: 361 / ub 512: 270 (12 moved) | no (-82) / ub 128: 335 (11 moved) |
| Swift Q2_0 | q8_0 | ub 256: 1175 / ub 256: 1441 (8 moved) | ub 256: 1046 / ub 256: 1313 (13 moved) | ub 512: 574 / ub 512: 859 (12 moved) | ub 256: 392 / ub 512: 226 (12 moved) | no (127) / ub 128: 287 (12 moved) |

### Draft: mtp

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| ISTA IQ3_S | f16 | no (-636) / no (-354) | no (-725) / no (-415) | no (-786) / no (-476) | no (-1175) / no (-788) | no (-2084) / no (-1451) |
| ISTA IQ3_S | q8_0 | no (-615) / no (-371) | no (-704) / no (-438) | no (-791) / no (-504) | no (-1273) / no (-848) | no (-2028) / no (-1568) |
| Swift IQ3_S | f16 | no (-651) / no (-367) | no (-740) / no (-428) | no (-801) / no (-489) | no (-1183) / no (-806) | no (-2099) / no (-1464) |
| Swift IQ3_S | q8_0 | no (-630) / no (-384) | no (-719) / no (-451) | no (-806) / no (-517) | no (-1281) / no (-853) | no (-2043) / no (-1573) |
| Swift IQ3_XXS | f16 | ub 128: 167 / ub 256: 246 (13 moved) | ub 64: 163 / ub 128: 289 (18 moved) | no (41) / ub 128: 219 (18 moved) | no (-357) / no (-32) | no (-1063) / no (-706) |
| Swift IQ3_XXS | q8_0 | ub 128: 161 / ub 256: 271 (13 moved) | no (134) / ub 128: 283 (13 moved) | no (47) / ub 128: 177 (13 moved) | no (-304) / no (-53) | no (-960) / no (-641) |
| Swift IQ3_XXS-hcq8 | f16 | ub 128: 243 / ub 256: 346 (13 moved) | ub 64: 212 / ub 256: 187 (13 moved) | no (100) / ub 128: 291 (18 moved) | no (-271) / no (40) | no (-977) / no (-606) |
| Swift IQ3_XXS-hcq8 | q8_0 | ub 128: 261 / ub 256: 323 (13 moved) | ub 128: 152 / ub 128: 376 (13 moved) | no (147) / ub 128: 212 (13 moved) | no (-204) / no (35) | no (-874) / no (-555) |
| Swift IQ2_XS | f16 | ub 256: 905 / ub 256: 1104 (14 moved) | ub 256: 784 / ub 256: 1023 (14 moved) | ub 512: 328 / ub 512: 564 (11 moved) | ub 256: 152 / ub 256: 425 (10 moved) | no (-291) / ub 64: 154 (13 moved) |
| Swift IQ2_XS | q8_0 | ub 256: 930 / ub 256: 1091 (14 moved) | ub 256: 801 / ub 256: 984 (14 moved) | ub 512: 329 / ub 512: 490 (11 moved) | ub 128: 423 / ub 256: 372 (11 moved) | no (-82) / ub 64: 177 (11 moved) |
| ISTA Coder IQ1_M | f16 | ub 256: 1937 / ub 256: 2147 (15 moved) | ub 256: 1856 / ub 256: 2050 (15 moved) | ub 512: 1248 / ub 512: 1248 (8 moved) | ub 256: 1116 / ub 256: 1116 (11 moved) | ub 128: 483 / ub 128: 483 (15 moved) |
| ISTA Coder IQ1_M | q8_0 | ub 256: 1942 / ub 256: 2138 (9 moved) | ub 256: 1817 / ub 256: 1972 (14 moved) | ub 512: 1100 / ub 512: 1100 (8 moved) | ub 256: 878 / ub 256: 878 (15 moved) | ub 64: 677 / ub 64: 677 (11 moved) |
| Swift Q2_0-hcq8 | f16 | ub 256: 1219 / ub 256: 1333 (10 moved) | ub 256: 1098 / ub 256: 1243 (12 moved) | ub 512: 627 / ub 512: 793 (10 moved) | ub 256: 430 / ub 256: 691 (12 moved) | no (-13) / ub 128: 181 (12 moved) |
| Swift Q2_0-hcq8 | q8_0 | ub 256: 1244 / ub 256: 1311 (12 moved) | ub 256: 1115 / ub 256: 1204 (12 moved) | ub 512: 628 / ub 512: 719 (10 moved) | ub 256: 446 / ub 256: 611 (10 moved) | ub 64: 181 / ub 64: 406 (14 moved) |
| Swift Q2_0 | f16 | ub 256: 1134 / ub 256: 1273 (10 moved) | ub 256: 1013 / ub 256: 1172 (12 moved) | ub 512: 557 / ub 512: 707 (10 moved) | ub 256: 345 / ub 256: 591 (12 moved) | no (-98) / ub 64: 310 (13 moved) |
| Swift Q2_0 | q8_0 | ub 256: 1159 / ub 256: 1260 (10 moved) | ub 256: 1030 / ub 256: 1133 (12 moved) | ub 512: 542 / ub 512: 633 (10 moved) | ub 256: 360 / ub 256: 525 (10 moved) | no (95) / ub 64: 320 (11 moved) |

### Draft: mtp-shr

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| ISTA IQ3_S | f16 | no (-636) / no (-354) | no (-725) / no (-415) | no (-786) / no (-476) | no (-1116) / no (-788) | no (-1928) / no (-1451) |
| ISTA IQ3_S | q8_0 | no (-615) / no (-371) | no (-704) / no (-438) | no (-791) / no (-504) | no (-1079) / no (-769) | no (-1840) / no (-1415) |
| Swift IQ3_S | f16 | no (-651) / no (-367) | no (-740) / no (-428) | no (-801) / no (-489) | no (-1131) / no (-806) | no (-1943) / no (-1464) |
| Swift IQ3_S | q8_0 | no (-630) / no (-384) | no (-719) / no (-451) | no (-806) / no (-517) | no (-1094) / no (-782) | no (-1848) / no (-1420) |
| Swift IQ3_XXS | f16 | ub 128: 197 / ub 256: 351 (13 moved) | ub 64: 195 / ub 256: 230 (13 moved) | no (101) / ub 128: 268 (13 moved) | no (-145) / no (-2) | no (-955) / no (-699) |
| Swift IQ3_XXS | q8_0 | ub 128: 177 / ub 256: 376 (13 moved) | ub 64: 172 / ub 256: 247 (16 moved) | no (106) / ub 128: 285 (13 moved) | no (-194) / no (32) | no (-960) / no (-635) |
| Swift IQ3_XXS-hcq8 | f16 | ub 128: 268 / ub 256: 422 (13 moved) | ub 128: 207 / ub 256: 301 (13 moved) | ub 64: 201 / ub 256: 175 (16 moved) | no (-72) / no (96) | no (-884) / no (-599) |
| Swift IQ3_XXS-hcq8 | q8_0 | ub 128: 261 / ub 256: 447 (16 moved) | ub 128: 161 / ub 256: 299 (16 moved) | ub 64: 177 / ub 128: 356 (16 moved) | no (-94) / no (68) | no (-874) / no (-555) |
| Swift IQ2_XS | f16 | ub 256: 941 / ub 256: 1200 (13 moved) | ub 256: 820 / ub 256: 1079 (13 moved) | ub 512: 328 / ub 512: 584 (14 moved) | ub 256: 152 / ub 256: 443 (10 moved) | no (-291) / ub 64: 212 (11 moved) |
| Swift IQ2_XS | q8_0 | ub 256: 966 / ub 256: 1195 (13 moved) | ub 256: 837 / ub 256: 1078 (12 moved) | ub 512: 329 / ub 512: 533 (14 moved) | ub 128: 423 / ub 256: 403 (15 moved) | no (-82) / ub 64: 209 (10 moved) |
| ISTA Coder IQ1_M | f16 | ub 256: 2052 / ub 256: 2243 (15 moved) | ub 256: 1931 / ub 256: 2145 (15 moved) | ub 512: 1440 / ub 512: 1664 (8 moved) | ub 512: 561 / ub 512: 561 (11 moved) | ub 128: 835 / ub 128: 980 (15 moved) |
| ISTA Coder IQ1_M | q8_0 | ub 256: 2071 / ub 256: 2241 (14 moved) | ub 256: 1857 / ub 256: 2119 (16 moved) | ub 512: 1366 / ub 512: 1591 (8 moved) | ub 512: 331 / ub 512: 331 (8 moved) | ub 128: 594 / ub 128: 594 (11 moved) |
| Swift Q2_0-hcq8 | f16 | ub 256: 1219 / ub 256: 1434 (12 moved) | ub 256: 1098 / ub 256: 1313 (12 moved) | ub 512: 642 / ub 512: 848 (10 moved) | ub 256: 430 / ub 512: 174 (15 moved) | no (-13) / ub 128: 208 (11 moved) |
| Swift Q2_0-hcq8 | q8_0 | ub 256: 1244 / ub 256: 1459 (12 moved) | ub 256: 1115 / ub 256: 1330 (12 moved) | ub 512: 643 / ub 512: 760 (9 moved) | ub 256: 461 / ub 256: 657 (12 moved) | ub 64: 196 / ub 128: 171 (12 moved) |
| Swift Q2_0 | f16 | ub 256: 1134 / ub 256: 1348 (12 moved) | ub 256: 1013 / ub 256: 1227 (12 moved) | ub 512: 557 / ub 512: 771 (10 moved) | ub 256: 345 / ub 256: 632 (11 moved) | no (-98) / ub 64: 341 (9 moved) |
| Swift Q2_0 | q8_0 | ub 256: 1159 / ub 256: 1373 (12 moved) | ub 256: 1030 / ub 256: 1244 (12 moved) | ub 512: 558 / ub 512: 684 (10 moved) | ub 256: 376 / ub 256: 581 (12 moved) | no (111) / ub 64: 388 (8 moved) |
