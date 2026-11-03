#!/usr/bin/env bash
# registers.tsv from bench/mmvq_pascal's build/res-usage_kcol.txt (cuobjdump -res-usage of build/mmvq_kcol, which
# build_kcol.sh writes): per k-column product kernel of the sm_61 code, its type, columns, rows per lane group,
# registers, stack and local memory. TYPES: the type names to keep (default the three Q630 adds).
# Usage: [TYPES="q4_0 q5_0 iq4_nl"] run/registers.sh <bench>/build/res-usage_kcol.txt > registers.tsv
set -euo pipefail
TYPES=${TYPES:-q4_0 q5_0 iq4_nl} perl -ne '
    BEGIN {
        @n = qw(q8_0 q2_0 q3_K q4_K q5_K q6_K iq4_xs q4_0 q5_0 iq4_nl);
        %keep = map { $_ => 1 } split " ", $ENV{TYPES};
        print "kernel\ttype\tcolumns\trows_per_group\tregisters\tstack\tlocal\n";
    }
    $arch = $1 if /arch = (\S+)/;
    if (/Function \S*mul_mat_kcolILi(\d+)ELi(\d+)ELi(\d+)E/) { ($t, $c, $r) = ($1, $2, $3); $f = 1; next }
    if ($f && /REG:(\d+) STACK:(\d+).*LOCAL:(\d+)/) {
        push @rows, [$n[$t], $c, $r, $1, $2, $3] if $arch eq "sm_61" && $keep{$n[$t]};
        $f = 0;
    }
    END { print join("\t", "mul_mat_kcol", @$_), "\n" for sort { $a->[0] cmp $b->[0] || $a->[1] <=> $b->[1] || $a->[2] <=> $b->[2] } @rows }
' "$1"
