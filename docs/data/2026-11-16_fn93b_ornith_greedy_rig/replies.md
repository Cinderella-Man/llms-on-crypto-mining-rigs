| Depth (tokens) | Arm | Reply (md5 of the token ids) | Pass 2 | Against the reference `v4` | Against the reference `v4s` |
|---:|---|---|---|---|---|
| 0 | `v4` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `b4096` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `taper` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `incall` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `ckvram` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `hop` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 0 | `graphs` | `b8279751702b64bd93868f577002931f` | same | same | same |
| 4,096 | `v4` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 4,096 | `b4096` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 4,096 | `taper` | `c45afe38fb72afa2b59d2b385c0dd3d1` | same | differs at token 21 | differs at token 21 |
| 4,096 | `incall` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 4,096 | `ckvram` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 4,096 | `hop` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 4,096 | `graphs` | `5eb4b34a7837668b5097f3181056ccfc` | same | same | same |
| 8,192 | `v4` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 8,192 | `b4096` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 8,192 | `taper` | `5e66d7864ef0b3f0347988eb1e258772` | same | differs at token 11 | same |
| 8,192 | `incall` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 8,192 | `ckvram` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 8,192 | `hop` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 8,192 | `graphs` | `eea8bef31dda2a8da383789a89b35205` | same | same | differs at token 11 |
| 16,384 | `v4` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |
| 16,384 | `b4096` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |
| 16,384 | `taper` | `a15fa437298413e414caa53fa4d4fb01` | same | differs at token 6 | same |
| 16,384 | `incall` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |
| 16,384 | `ckvram` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |
| 16,384 | `hop` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |
| 16,384 | `graphs` | `d7e7c76f960356629ad21b74f5f8b0d4` | same | same | differs at token 6 |

Checks:

- decode prompt at 0: the reference run's (1 prompt md5 here)
- prime prompt at 4096: the reference run's (1 prompt md5 here)
- decode prompt at 4096: the reference run's (1 prompt md5 here)
- prime at 4096: 0 cached, 4032 read in every load here; the same as the reference run's (0 cached, 4032 read)
- prime prompt at 8192: the reference run's (1 prompt md5 here)
- decode prompt at 8192: the reference run's (1 prompt md5 here)
- prime at 8192: 3516 cached, 4612 read in every load here; the same as the reference run's (3516 cached, 4612 read)
- prime prompt at 16384: the reference run's (1 prompt md5 here)
- decode prompt at 16384: the reference run's (1 prompt md5 here)
- prime at 16384: 3516 cached, 12804 read in every load here; the same as the reference run's (3516 cached, 12804 read)
