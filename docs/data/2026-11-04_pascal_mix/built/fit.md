Largest `-ub` that leaves 150 MiB on every card (free MiB on the fullest card; `no (x)`: the best of any `-ub` falls short), nine P106-090 cards. Whole layers / with routed-expert tensors moved by `-ot` (how many).

### Draft: none

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| pmix IQ3-FAST | f16 | ub 256: 166 / ub 256: 585 (11 moved) | ub 128: 203 / ub 256: 504 (8 moved) | ub 64: 169 / ub 256: 395 (8 moved) | no (-75) / ub 128: 167 (10 moved) | no (-590) / no (-285) |
| pmix IQ3-FAST | q8_0 | ub 256: 153 / ub 256: 572 (11 moved) | ub 128: 187 / ub 256: 465 (11 moved) | ub 64: 203 / ub 256: 360 (11 moved) | no (-124) / ub 128: 160 (8 moved) | no (-656) / no (-314) |
| Swift IQ3_XXS-C5 | f16 | ub 256: 156 / ub 256: 580 (11 moved) | ub 128: 189 / ub 256: 499 (8 moved) | ub 64: 155 / ub 256: 387 (8 moved) | no (-89) / ub 128: 159 (10 moved) | no (-598) / no (-297) |
| Swift IQ3_XXS-C5 | q8_0 | ub 128: 264 / ub 256: 567 (11 moved) | ub 128: 177 / ub 256: 460 (11 moved) | ub 64: 193 / ub 256: 355 (11 moved) | no (-138) / ub 128: 152 (8 moved) | no (-670) / no (-322) |
| pmix Q2-SPINE6 | f16 | ub 256: 1141 / ub 256: 1446 (11 moved) | ub 256: 1060 / ub 256: 1365 (11 moved) | ub 512: 644 / ub 512: 944 (12 moved) | ub 256: 593 / ub 512: 191 (11 moved) | ub 64: 164 / ub 128: 322 (11 moved) |
| pmix Q2-SPINE6 | q8_0 | ub 256: 1128 / ub 256: 1433 (11 moved) | ub 256: 1021 / ub 256: 1326 (11 moved) | ub 512: 570 / ub 512: 875 (11 moved) | ub 256: 474 / ub 512: 226 (12 moved) | ub 64: 373 / ub 128: 208 (11 moved) |
| pmix Q2-FAST | f16 | ub 256: 1234 / ub 256: 1552 (11 moved) | ub 256: 1153 / ub 256: 1471 (11 moved) | ub 512: 737 / ub 512: 1029 (10 moved) | ub 256: 672 / ub 512: 303 (14 moved) | ub 64: 229 / ub 128: 411 (16 moved) |
| pmix Q2-FAST | q8_0 | ub 256: 1221 / ub 256: 1539 (11 moved) | ub 256: 1114 / ub 256: 1432 (11 moved) | ub 512: 663 / ub 512: 981 (11 moved) | ub 256: 567 / ub 512: 321 (14 moved) | ub 64: 438 / ub 128: 320 (14 moved) |
| Swift Q2_0-C5 | f16 | ub 256: 1234 / ub 256: 1555 (11 moved) | ub 256: 1153 / ub 256: 1474 (11 moved) | ub 512: 737 / ub 512: 1034 (10 moved) | ub 256: 664 / ub 512: 307 (14 moved) | ub 64: 221 / ub 128: 416 (16 moved) |
| Swift Q2_0-C5 | q8_0 | ub 256: 1221 / ub 256: 1542 (11 moved) | ub 256: 1114 / ub 256: 1435 (11 moved) | ub 512: 663 / ub 512: 984 (11 moved) | ub 256: 567 / ub 512: 320 (11 moved) | ub 64: 430 / ub 128: 324 (14 moved) |

### Draft: dflash2

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| pmix IQ3-FAST | f16 | ub 128: 257 / ub 256: 417 (11 moved) | ub 128: 157 / ub 256: 309 (11 moved) | no (115) / ub 256: 214 (9 moved) | no (-131) / ub 64: 198 (13 moved) | no (-621) / no (-483) |
| pmix IQ3-FAST | q8_0 | ub 128: 237 / ub 256: 439 (10 moved) | ub 128: 150 / ub 256: 310 (10 moved) | ub 64: 162 / ub 256: 192 (11 moved) | no (-180) / ub 64: 153 (9 moved) | no (-712) / no (-384) |
| Swift IQ3_XXS-C5 | f16 | ub 128: 244 / ub 256: 409 (11 moved) | ub 64: 202 / ub 256: 299 (9 moved) | no (101) / ub 256: 202 (9 moved) | no (-145) / ub 64: 189 (13 moved) | no (-635) / no (-493) |
| Swift IQ3_XXS-C5 | q8_0 | ub 128: 227 / ub 256: 432 (10 moved) | ub 64: 222 / ub 256: 303 (10 moved) | no (148) / ub 256: 184 (11 moved) | no (-194) / no (140) | no (-726) / no (-392) |
| pmix Q2-SPINE6 | f16 | ub 256: 1033 / ub 256: 1348 (8 moved) | ub 256: 912 / ub 256: 1238 (10 moved) | ub 512: 456 / ub 512: 822 (10 moved) | ub 256: 244 / ub 256: 690 (9 moved) | no (-199) / ub 128: 260 (10 moved) |
| pmix Q2-SPINE6 | q8_0 | ub 256: 1058 / ub 256: 1373 (8 moved) | ub 256: 929 / ub 256: 1244 (8 moved) | ub 512: 457 / ub 512: 772 (8 moved) | ub 256: 275 / ub 256: 652 (10 moved) | no (10) / ub 128: 166 (9 moved) |
| pmix Q2-FAST | f16 | ub 256: 1146 / ub 256: 1419 (12 moved) | ub 256: 1025 / ub 256: 1338 (12 moved) | ub 512: 569 / ub 512: 922 (12 moved) | ub 256: 357 / ub 512: 266 (12 moved) | no (-86) / ub 128: 340 (11 moved) |
| pmix Q2-FAST | q8_0 | ub 256: 1171 / ub 256: 1443 (8 moved) | ub 256: 1042 / ub 256: 1314 (8 moved) | ub 512: 570 / ub 512: 848 (12 moved) | ub 256: 388 / ub 512: 215 (12 moved) | no (123) / ub 128: 283 (12 moved) |
| Swift Q2_0-C5 | f16 | ub 256: 1150 / ub 256: 1430 (12 moved) | ub 256: 1029 / ub 256: 1349 (12 moved) | ub 512: 573 / ub 512: 933 (12 moved) | ub 256: 361 / ub 512: 270 (12 moved) | no (-82) / ub 128: 335 (11 moved) |
| Swift Q2_0-C5 | q8_0 | ub 256: 1175 / ub 256: 1441 (8 moved) | ub 256: 1046 / ub 256: 1313 (13 moved) | ub 512: 574 / ub 512: 859 (12 moved) | ub 256: 392 / ub 512: 226 (12 moved) | no (127) / ub 128: 287 (12 moved) |

### Draft: mtp

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| pmix IQ3-FAST | f16 | ub 128: 170 / ub 256: 262 (13 moved) | ub 64: 166 / ub 256: 150 (18 moved) | no (44) / ub 128: 228 (18 moved) | no (-350) / no (-23) | no (-1053) / no (-692) |
| pmix IQ3-FAST | q8_0 | ub 128: 176 / ub 256: 287 (13 moved) | no (149) / ub 128: 299 (13 moved) | no (62) / ub 128: 179 (13 moved) | no (-289) / no (-45) | no (-953) / no (-631) |
| Swift IQ3_XXS-C5 | f16 | ub 128: 167 / ub 256: 246 (13 moved) | ub 64: 163 / ub 128: 289 (18 moved) | no (41) / ub 128: 219 (18 moved) | no (-357) / no (-32) | no (-1063) / no (-706) |
| Swift IQ3_XXS-C5 | q8_0 | ub 128: 161 / ub 256: 271 (13 moved) | no (134) / ub 128: 283 (13 moved) | no (47) / ub 128: 177 (13 moved) | no (-304) / no (-53) | no (-960) / no (-641) |
| pmix Q2-SPINE6 | f16 | ub 256: 1029 / ub 256: 1184 (10 moved) | ub 256: 908 / ub 256: 1103 (10 moved) | ub 512: 452 / ub 512: 627 (12 moved) | ub 256: 240 / ub 256: 465 (13 moved) | no (-203) / ub 64: 200 (9 moved) |
| pmix Q2-SPINE6 | q8_0 | ub 256: 1054 / ub 256: 1171 (10 moved) | ub 256: 925 / ub 256: 1064 (12 moved) | ub 512: 453 / ub 512: 527 (11 moved) | ub 256: 271 / ub 256: 429 (11 moved) | no (6) / ub 64: 231 (11 moved) |
| pmix Q2-FAST | f16 | ub 256: 1125 / ub 256: 1266 (10 moved) | ub 256: 1004 / ub 256: 1178 (12 moved) | ub 512: 548 / ub 512: 701 (10 moved) | ub 256: 336 / ub 256: 595 (12 moved) | no (-107) / ub 64: 304 (13 moved) |
| pmix Q2-FAST | q8_0 | ub 256: 1150 / ub 256: 1253 (10 moved) | ub 256: 1021 / ub 256: 1139 (12 moved) | ub 512: 543 / ub 512: 627 (10 moved) | ub 256: 361 / ub 256: 519 (10 moved) | no (96) / ub 64: 321 (14 moved) |
| Swift Q2_0-C5 | f16 | ub 256: 1134 / ub 256: 1273 (10 moved) | ub 256: 1013 / ub 256: 1172 (12 moved) | ub 512: 557 / ub 512: 707 (10 moved) | ub 256: 345 / ub 256: 591 (12 moved) | no (-98) / ub 64: 310 (13 moved) |
| Swift Q2_0-C5 | q8_0 | ub 256: 1159 / ub 256: 1260 (10 moved) | ub 256: 1030 / ub 256: 1133 (12 moved) | ub 512: 542 / ub 512: 633 (10 moved) | ub 256: 360 / ub 256: 525 (10 moved) | no (95) / ub 64: 320 (11 moved) |

### Draft: mtp-shr

| File | Cache | 32k | 48k | 64k | 128k | 262k |
|---|---|---|---|---|---|---|
| pmix IQ3-FAST | f16 | ub 128: 206 / ub 256: 360 (13 moved) | ub 64: 204 / ub 256: 239 (13 moved) | no (115) / ub 128: 277 (13 moved) | no (-133) / no (10) | no (-945) / no (-692) |
| pmix IQ3-FAST | q8_0 | ub 128: 186 / ub 256: 385 (13 moved) | ub 64: 181 / ub 256: 256 (16 moved) | no (115) / ub 128: 294 (16 moved) | no (-180) / no (35) | no (-953) / no (-626) |
| Swift IQ3_XXS-C5 | f16 | ub 128: 197 / ub 256: 351 (13 moved) | ub 64: 195 / ub 256: 230 (13 moved) | no (101) / ub 128: 268 (13 moved) | no (-145) / no (-2) | no (-955) / no (-699) |
| Swift IQ3_XXS-C5 | q8_0 | ub 128: 177 / ub 256: 376 (13 moved) | ub 64: 172 / ub 256: 247 (16 moved) | no (106) / ub 128: 285 (13 moved) | no (-194) / no (32) | no (-960) / no (-635) |
| pmix Q2-SPINE6 | f16 | ub 256: 1029 / ub 256: 1254 (11 moved) | ub 256: 908 / ub 256: 1133 (11 moved) | ub 512: 452 / ub 512: 677 (10 moved) | ub 256: 240 / ub 256: 548 (12 moved) | no (-203) / ub 64: 247 (9 moved) |
| pmix Q2-SPINE6 | q8_0 | ub 256: 1054 / ub 256: 1279 (11 moved) | ub 256: 925 / ub 256: 1150 (12 moved) | ub 512: 453 / ub 512: 613 (10 moved) | ub 256: 271 / ub 256: 494 (12 moved) | no (6) / ub 64: 319 (8 moved) |
| pmix Q2-FAST | f16 | ub 256: 1125 / ub 256: 1343 (11 moved) | ub 256: 1004 / ub 256: 1222 (12 moved) | ub 512: 548 / ub 512: 766 (10 moved) | ub 256: 336 / ub 256: 637 (11 moved) | no (-107) / ub 64: 336 (9 moved) |
| pmix Q2-FAST | q8_0 | ub 256: 1150 / ub 256: 1368 (12 moved) | ub 256: 1021 / ub 256: 1239 (12 moved) | ub 512: 549 / ub 512: 688 (10 moved) | ub 256: 367 / ub 256: 581 (12 moved) | no (102) / ub 64: 389 (11 moved) |
| Swift Q2_0-C5 | f16 | ub 256: 1134 / ub 256: 1348 (12 moved) | ub 256: 1013 / ub 256: 1227 (12 moved) | ub 512: 557 / ub 512: 771 (10 moved) | ub 256: 345 / ub 256: 632 (11 moved) | no (-98) / ub 64: 341 (9 moved) |
| Swift Q2_0-C5 | q8_0 | ub 256: 1159 / ub 256: 1373 (12 moved) | ub 256: 1030 / ub 256: 1244 (12 moved) | ub 512: 558 / ub 512: 684 (10 moved) | ub 256: 376 / ub 256: 581 (12 moved) | no (111) / ub 64: 388 (8 moved) |
