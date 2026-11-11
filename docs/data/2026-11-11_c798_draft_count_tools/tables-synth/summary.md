# Draft length by position (run_draft_count.sh)

## Loads

| label | n_max | requests | generated | drafted | accepted | tokens_a_step | share_accepted | verify_steps | accepted_per_verify_step | restores | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d7s-1 | 7 | 6 | 6144 | 8200 | 4962 | 5.1980 | 0.6051 | 1175 | 4.2230 | 0 | 0 |
| d4s-1 | 4 | 6 | 6144 | 6081 | 4614 | 4.0157 | 0.7588 | 1522 | 3.0315 | 0 | 0 |

## Arms (pooled over their loads; each load by value)

| arm | loads | tokens_a_step_each | tokens_a_step_pooled | spread | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d7s | 1 | 5.198 | 5.1980 | nan | 0.8987 | 0.8075 | 0.7118 | 0.6063 | 0.5060 | 0.4046 | 0.2979 |
| d4s | 1 | 4.016 | 4.0157 | nan | 0.9054 | 0.8101 | 0.7163 | 0.6028 |  |  |  |

## Conditional acceptance by position (pooled)

| arm | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| d7s | 0.8987 | 0.8986 | 0.8808 | 0.8523 | 0.8350 | 0.7990 | 0.7373 |
| d4s | 0.9054 | 0.8948 | 0.8838 | 0.8418 |  |  |  |

## Against the reference arm

| arm | ref | diff | diff_pct | lo_requests | hi_requests | lo_traces | hi_traces | spread_ref | spread_arm |
|---|---|---|---|---|---|---|---|---|---|
| d4s | d7s | -1.1823 | -22.7451 | -1.2517 | -1.1145 | -1.2135 | -1.1529 | nan | nan |

## By the request's context length (prompt tokens, cached or not)

| arm | context | requests | verify_steps | tokens_a_step | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d7s | 0-4095 | 6 | 1175 | 5.1980 | 0.8987 | 0.8075 | 0.7118 | 0.6063 | 0.5060 | 0.4046 | 0.2979 |
| d4s | 0-4095 | 6 | 1522 | 4.0157 | 0.9054 | 0.8101 | 0.7163 | 0.6028 |  |  |  |

## Checks

12 of 12 checks OK.

