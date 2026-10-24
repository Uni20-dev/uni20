#!/usr/bin/env python3
"""Verify Airy asymptotic join bounds using exact rational arithmetic.

DLMF 9.7.iii bounds the real-variable remainder by its first omitted term.
The printed logarithm is descriptive; all acceptance checks use integers.
"""
from fractions import Fraction
from math import isqrt, log2

for digits, join in [(24, 8), (53, 12), (64, 14), (113, 20)]:
    # floor(2*join**(3/2)/3), without a floating-point square root.
    zeta_lower = isqrt(4 * join**3 // 9)
    term = Fraction(1)
    target = Fraction(1, 2**(digits + 8))
    for order in range(1, 4 * digits + 64):
        previous = term
        ratio = Fraction((6 * order - 5) * (6 * order - 1),
                         72 * order * zeta_lower)
        term *= ratio
        assert term < previous, "Terms grew before meeting the join target"
        if term < target:
            print(f"digits={digits} join={join} zeta>={zeta_lower} "
                  f"order={order} log2(term)={log2(term):.6f}")
            break
    else:
        raise AssertionError("Iteration cap exhausted")
