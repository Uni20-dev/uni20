#!/usr/bin/env python3
"""Print exact Bernoulli coefficients and digamma truncation bounds.

This offline generator needs only Python's standard library. It does not
write source files or use an approximate special-function provider.
"""

from fractions import Fraction
from math import comb, log2


def bernoulli(order):
    values = [Fraction(1)]
    for n in range(1, order + 1):
        values.append(-sum(Fraction(comb(n + 1, k)) * values[k] for k in range(n)) / (n + 1))
    return values


def main():
    values = bernoulli(34)
    assert values[2] == Fraction(1, 6)
    assert values[4] == Fraction(-1, 30)
    assert values[6] == Fraction(1, 42)
    print("// B_(2k)/(2k), k=1,...,16")
    for k in range(1, 17):
        coefficient = values[2 * k] / (2 * k)
        print(f"R({coefficient.numerator}LL) / R({coefficient.denominator}LL),")
    print("\n// First omitted term at each recurrence threshold")
    for bits, threshold, terms in [(24, 8, 4), (53, 16, 8), (64, 16, 10), (113, 32, 16)]:
        order = 2 * terms + 2
        bound = abs(values[order] / order) / threshold**order
        print(f"// bits={bits}, threshold={threshold}, terms={terms}: "
              f"tail <= 2^({log2(bound):.9f}), tail/epsilon={float(bound * 2**(bits - 1)):.9g}")


if __name__ == "__main__":
    main()
