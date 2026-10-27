24 chunks, 98280 scored tokens, reference for ratios: ud-v100

| File | KLD | Same top | Ratio to ud-v100 (95 % bootstrap over chunks) | Mean paired difference ± SE |
|---|---|---|---|---|
| ud-v100 | 0.4713 | 87.81 % | 1.000 (1.000-1.000) | +0.0000 ± 0.0000 |
| g3-v100 | 0.4812 | 88.08 % | 1.021 (1.000-1.042) | +0.0099 ± 0.0060 |
| g3r-v100 | 0.4812 | 88.08 % | 1.021 (0.997-1.042) | +0.0099 ± 0.0060 |

By position in the chunk (KLD):

| File | 4096-5119 | 5120-6143 | 6144-7167 | 7168-8191 |
|---|---|---|---|---|
| ud-v100 | 0.7883 | 0.5721 | 0.3150 | 0.2102 |
| g3-v100 | 0.8039 | 0.5887 | 0.3290 | 0.2035 |
| g3r-v100 | 0.8039 | 0.5887 | 0.3290 | 0.2035 |

By token class (KLD; share of scored tokens; ratio to ud-v100):

| Class | Tokens | ud-v100 | g3-v100 | g3r-v100 | g3-v100/ud-v100 | g3r-v100/ud-v100 |
|---|---|---|---|---|---|---|
| tools/system | 39166 (39.9 %) | 0.9515 | 0.9821 | 0.9821 | 1.032 | 1.032 |
| user | 1622 (1.7 %) | 0.1108 | 0.0919 | 0.0919 | 0.829 | 0.829 |
| tool output | 10941 (11.1 %) | 0.6972 | 0.6951 | 0.6951 | 0.997 | 0.997 |
| reasoning | 38071 (38.7 %) | 0.0174 | 0.0146 | 0.0146 | 0.838 | 0.838 |
| call | 2311 (2.4 %) | 0.0266 | 0.0261 | 0.0261 | 0.981 | 0.981 |
| answer | 4427 (4.5 %) | 0.0386 | 0.0332 | 0.0332 | 0.860 | 0.860 |
| markup | 1742 (1.8 %) | 0.2025 | 0.1790 | 0.1790 | 0.884 | 0.884 |
| after </think> | 230 (0.2 %) | 0.0516 | 0.0333 | 0.0333 | 0.646 | 0.646 |
| long-range copy | 29052 (29.6 %) | 1.1558 | 1.1797 | 1.1797 | 1.021 | 1.021 |

Read against written, paired by chunk (chunks with tokens of the group):

| Group | Tokens | ud-v100 KLD | g3-v100 KLD | g3r-v100 KLD | g3-v100/ud-v100 (95 % bootstrap) | g3r-v100/ud-v100 (95 % bootstrap) |
|---|---|---|---|---|---|---|
| read (tools, user, tool output) | 51729 (52.6 %) | 0.8713 | 0.8935 | 0.8935 | 1.020 (0.996-1.046) | 1.020 (0.998-1.046) |
| written (reasoning, calls, answers) | 44809 (45.6 %) | 0.0200 | 0.0170 | 0.0170 | 0.874 (0.835-0.918) | 0.874 (0.834-0.919) |

Same top token by class:

| Class | ud-v100 | g3-v100 | g3r-v100 |
|---|---|---|---|
| tools/system | 80.20 % | 79.94 % | 79.94 % |
| user | 86.99 % | 88.72 % | 88.72 % |
| tool output | 80.24 % | 81.77 % | 81.77 % |
| reasoning | 96.16 % | 96.52 % | 96.52 % |
| call | 99.05 % | 99.00 % | 99.00 % |
| answer | 96.16 % | 96.32 % | 96.32 % |
| markup | 88.69 % | 89.95 % | 89.95 % |
| after </think> | 94.35 % | 94.35 % | 94.35 % |
| long-range copy | 77.83 % | 77.94 % | 77.94 % |

15 clusters of chunks sharing a prefix into the scored half: [0,8,19] [1,10,20] [2,22] [3,7,21] [4,11,23] [5] [6] [9] [12] [13] [14] [15] [16] [17] [18]

Token-pooled ratios to ud-v100 (sum over the group's tokens), 95 % bootstrap over 24 chunks / over 15 clusters:

| Group | Tokens | g3-v100/ud-v100 | g3r-v100/ud-v100 |
|---|---|---|---|
| read | 51729 | 1.025 (1.003-1.047 / 0.995-1.052) | 1.025 (1.002-1.046 / 0.996-1.052) |
| written | 44809 | 0.852 (0.823-0.885 / 0.824-0.881) | 0.852 (0.823-0.883 / 0.825-0.881) |
| tool schemas and system text | 39166 | 1.032 (1.008-1.053 / 0.994-1.063) | 1.032 (1.008-1.053 / 0.995-1.063) |
| tool output | 10941 | 0.997 (0.968-1.067 / 0.975-1.059) | 0.997 (0.968-1.067 / 0.974-1.055) |
| user | 1622 | 0.829 (0.686-0.945 / 0.670-0.952) | 0.829 (0.680-0.944 / 0.666-0.956) |

Paired differences in units of ud-v100's KLD on the group, sum(A - B) / sum(ud-v100), ± SE over 24 chunks / over 15 clusters:

| A - B | read | written | tool schemas and system text | tool output | user |
|---|---|---|---|---|---|
| g3r-v100 - g3-v100 | +0.0000 ± 0.0000 / 0.0000 | +0.0000 ± 0.0000 / 0.0000 | +0.0000 ± 0.0000 / 0.0000 | +0.0000 ± 0.0000 / 0.0000 | +0.0000 ± 0.0000 / 0.0000 |

Non-finite KLD values: {'ud-v100': 0, 'g3-v100': 0, 'g3r-v100': 0}
