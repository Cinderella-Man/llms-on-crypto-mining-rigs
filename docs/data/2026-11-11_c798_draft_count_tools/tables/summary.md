# Draft length by position (run_draft_count.sh)

## Loads

| label | n_max | requests | generated | drafted | accepted | tokens_a_step | share_accepted | verify_steps | accepted_per_verify_step | restores | grammar_rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d3-1 | 3 | 6 | 768 | 2250 | 0 | 1.0000 | 0.0000 | 756 | 0.0000 | 0 | 0 |
| d5-1 | 5 | 6 | 768 | 3720 | 0 | 1.0000 | 0.0000 | 756 | 0.0000 | 0 | 0 |
| d7-1 | 7 | 6 | 768 | 5166 | 0 | 1.0000 | 0.0000 | 756 | 0.0000 | 0 | 0 |
| d7x-1 | 7 | 6 | 768 | 5166 | 0 | 1.0000 | 0.0000 | 756 | 0.0000 | 0 | 0 |
| d3-2 | 3 | 6 | 768 | 2250 | 0 | 1.0000 | 0.0000 | 756 | 0.0000 | 0 | 0 |

## Arms (pooled over their loads; each load by value)

| arm | loads | tokens_a_step_each | tokens_a_step_pooled | spread | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d3 | 2 | 1.000 1.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| d5 | 1 | 1.000 | 1.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |
| d7 | 1 | 1.000 | 1.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| d7x | 1 | 1.000 | 1.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Conditional acceptance by position (pooled)

| arm | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| d3 | 0.0000 | nan | nan |  |  |  |  |
| d5 | 0.0000 | nan | nan | nan | nan |  |  |
| d7 | 0.0000 | nan | nan | nan | nan | nan | nan |
| d7x | 0.0000 | nan | nan | nan | nan | nan | nan |

## Against the reference arm

| arm | ref | diff | diff_pct | lo_requests | hi_requests | lo_traces | hi_traces | spread_ref | spread_arm |
|---|---|---|---|---|---|---|---|---|---|
| d5 | d3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| d7 | d3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| d7x | d3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan |

## By the request's context length (prompt tokens, cached or not)

| arm | context | requests | verify_steps | tokens_a_step | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d3 | 0-4095 | 12 | 1512 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| d5 | 0-4095 | 6 | 756 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |
| d7 | 0-4095 | 6 | 756 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| d7x | 0-4095 | 6 | 756 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Checks

34 of 34 checks OK.

