#!/usr/bin/env python3
"""Print exact small-shape Stirling and zero-endpoint fraction bounds.

Offline investigation aid using only Python's standard library. It does not
write C++ source or call a numerical special-function provider.
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
    print("Small-shape log-Gamma quotient: |a| <= 1/2")
    for bits, shift, terms in [(24, 16, 4), (53, 32, 8), (64, 32, 10), (113, 64, 16)]:
        order = 2 * terms + 2
        bound = abs(values[order] / order) / (Fraction(shift) - Fraction(1, 2))**order
        print(f"bits={bits} shift={shift} terms={terms} derivative remainder <2^({log2(bound):.9f})")
    print("Contracted fraction: relative truncation error at x=0, a=-b")
    for bits, b in [(24, 8), (53, 16), (64, 16), (113, 32)]:
        depth = 8
        while Fraction(1, comb(b + depth + 1, b)) > Fraction(1, 2**(bits + 3)):
            depth *= 2
        error = Fraction(1, comb(b + depth + 1, b))
        print(f"bits={bits} b={b} depth={depth} error=1/{error.denominator} <2^({log2(error):.9f})")


if __name__ == "__main__":
    main()
