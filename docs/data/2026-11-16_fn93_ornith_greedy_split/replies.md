Greedy pass 1 of each arm's first load: "same", or the first token that differs (counted from 1), against this machine's `v4` and `v4s` and the rig's (R162's first loads).

| Arm | Depth | vs `v4` | vs `v4s` | vs rig `v4` | vs rig `v4s` | ids md5 |
|---|---:|---|---|---|---|---|
| `v4s` | 0 | same | same | token 7 | token 7 | `70538883` |
| `v4s` | 4,096 | same | same | token 6 | token 6 | `0d12768b` |
| `v4s` | 8,192 | token 2 | same | token 3 | token 3 | `0cb4701b` |
| `v4s` | 16,384 | token 4 | same | token 6 | after char 160 | `ca3d8c65` |
| `v4s` | 32,768 | same | same | token 8 | token 17 | `c6330512` |
| `v4s` | 48,640 | same | same | same | same | `d340d86e` |
| `v4` | 0 | same | same | token 7 | token 7 | `70538883` |
| `v4` | 4,096 | same | same | token 6 | token 6 | `0d12768b` |
| `v4` | 8,192 | same | token 2 | token 2 | token 2 | `5834318a` |
| `v4` | 16,384 | same | token 4 | token 4 | token 4 | `21053999` |
| `v4` | 32,768 | same | same | token 8 | token 17 | `c6330512` |
| `v4` | 48,640 | same | same | same | same | `d340d86e` |

Checks:

- `step1`: 51 replies; greedy pass 2 unlike its pass 1 in 0; arm, kind, pass and depth with more than one reply over its loads: 0
- the rig (R162): arm, kind, pass and depth with more than one reply over its loads: 0
