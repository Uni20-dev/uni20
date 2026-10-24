#!/usr/bin/env python3
"""Print Stirling/fraction bounds and guard-precision constant prefixes.

Offline investigation aid using only Python's standard library. It does not
write C++ source or call a numerical special-function provider.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb, log2


def bernoulli(order):
    values = [Fraction(1)]
    for n in range(1, order + 1):
        values.append(-sum(Fraction(comb(n + 1, k)) * values[k] for k in range(n)) / (n + 1))
    return values


def atan_interval(q, terms=128):
    value = sum((Fraction((-1)**k, (2*k+1)*q**(2*k+1)) for k in range(terms)), Fraction(0))
    return value, value + Fraction(1, (2*terms+1)*q**(2*terms+1))


def constant_prefixes():
    terms = 180
    lo = sum((Fraction(2, (2*k+1)*3**(2*k+1)) for k in range(terms)), Fraction(0))
    hi = lo + Fraction(2, (2*terms+1)*3**(2*terms+1)) * Fraction(9, 8)
    low = (lo.numerator << 256) // lo.denominator
    high = (hi.numerator << 256) // hi.denominator
    assert low == high
    print(f"log(2) fractional prefix: {low:064x}")
    a, b = atan_interval(5), atan_interval(239)
    pi_bounds = (16*a[0]-4*b[1], 16*a[1]-4*b[0])
    prefixes = []
    for digits in (160, 240):
        with localcontext() as context:
            context.prec = digits
            for bound in pi_bounds:
                pi_value = Decimal(bound.numerator) / Decimal(bound.denominator)
                prefixes.append(int(((2*pi_value).ln()-1) * Decimal(2)**256))
    assert len(set(prefixes)) == 1
    print(f"log(2*pi) integer part 1, fractional prefix: {prefixes[0]:064x}")


def main():
    values = bernoulli(34)
    assert values[2] == Fraction(1, 6)
    print("Small-shape log-Gamma quotient: |a| <= 1/2")
    for bits, shift, terms in [(24, 16, 4), (53, 32, 8), (64, 32, 10), (113, 64, 16)]:
        order = 2 * terms + 2
        bound = abs(values[order] / order) / (Fraction(shift) - Fraction(1, 2))**order
        print(f"bits={bits} shift={shift} terms={terms} derivative remainder <2^({log2(bound):.9f})")
    print("Guarded complete log-Gamma: first omitted Stirling term")
    for bits, shift, terms in [(24, 32, 4), (53, 64, 8), (64, 64, 10), (113, 128, 16)]:
        order = 2 * terms + 2
        bound = abs(values[order] / (order * (order-1))) / shift**(order-1)
        print(f"bits={bits} shift={shift} terms={terms} remainder <2^({log2(bound):.9f})")
    constant_prefixes()
    print("Contracted fraction: relative truncation error at x=0, a=-b")
    for bits, b in [(24, 8), (53, 16), (64, 16), (113, 32)]:
        depth = 8
        while Fraction(1, comb(b + depth + 1, b)) > Fraction(1, 2**(bits + 3)):
            depth *= 2
        error = Fraction(1, comb(b + depth + 1, b))
        print(f"bits={bits} b={b} depth={depth} error=1/{error.denominator} <2^({log2(error):.9f})")


if __name__ == "__main__":
    main()
