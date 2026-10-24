#!/usr/bin/env python3
"""Print native Euler-Maclaurin coefficients and exact remainder bounds.

Only Python's standard library is needed. Coefficient prefixes come from exact
Bernoulli rationals; no special-function provider or floating oracle is used.
"""

from fractions import Fraction
from math import comb, factorial, log2


def bernoulli(order):
    values = [Fraction(1)]
    for n in range(1, order + 1):
        values.append(-sum(Fraction(comb(n + 1, k)) * values[k] for k in range(n)) / (n + 1))
    return values


def binary_prefix(value):
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2)**exponent:
        exponent -= 1
    prefix = int(value * Fraction(2)**(127 - exponent))
    assert Fraction(prefix) * Fraction(2)**(exponent - 127) <= value
    assert value < Fraction(prefix + 1) * Fraction(2)**(exponent - 127)
    return prefix >> 64, prefix % 2**64, exponent


def main():
    values = bernoulli(32)
    coefficients = [values[2*k] / factorial(2*k) for k in range(1, 17)]
    assert coefficients[:3] == [Fraction(1, 12), Fraction(-1, 720), Fraction(1, 30240)]
    print("// B_(2k)/(2k)!, k=1,...,16; 128-bit binary prefixes")
    for coefficient in coefficients:
        high, low, exponent = binary_prefix(coefficient)
        sign = "-" if coefficient < 0 else ""
        print(f"{sign}(R(0x{high:016x}ULL) * R(0x1p-63L) + "
              f"R(0x{low:016x}ULL) * R(0x1p-127L)) * R(0x1p{exponent:+d}L),")
    print("\n// Absolute integral-remainder bounds, s in [0, bits+1]")
    for bits, shift, terms in [(24, 8, 4), (53, 16, 8), (64, 16, 10), (113, 32, 16)]:
        maximum = Fraction(0)
        for left in range(bits + 1):
            # On [j,j+1], rising(s,2M-1) <= rising(j+1,2M-1),
            # while N^(1-s-2M) <= N^(1-j-2M). Both bounds are exact.
            rising = 1
            for k in range(2*terms - 1):
                rising *= left + 1 + k
            bound = abs(coefficients[terms-1]) * rising * Fraction(shift)**(1-left-2*terms)
            maximum = max(maximum, bound)
        print(f"// bits={bits}, N={shift}, M={terms}: bound={maximum}, "
              f"log2(bound)={log2(maximum):.9f}, bound/epsilon={float(maximum * 2**(bits-1)):.9g}")


if __name__ == "__main__":
    main()
