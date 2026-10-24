#!/usr/bin/env python3
"""Generate 4/(3*pi) using Machin intervals and exact integer arithmetic."""
import argparse
from pathlib import Path

BITS = 24832
GUARD = 64
SCALE = 1 << (BITS + GUARD)


def atan_interval(denominator):
    power, total, index = denominator, 0, 0
    while True:
        term = SCALE // ((2 * index + 1) * power)
        if not term:
            break
        total += term if index % 2 == 0 else -term
        index += 1
        power *= denominator * denominator
    # Each rounded term costs <1 integer unit, plus <1 for the remaining
    # alternating tail. Use a symmetric interval before Machin combination.
    return total, index + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    destination = parser.parse_args().output
    a, da = atan_interval(5)
    b, db = atan_interval(239)
    pi = 16 * a - 4 * b
    error = 16 * da + 4 * db
    # Reciprocal interval endpoints, in the final binary scale.
    lower = (4 * (1 << BITS) * SCALE) // (3 * (pi + error))
    upper = (4 * (1 << BITS) * SCALE) // (3 * (pi - error))
    assert lower == upper, 'Interval does not determine the complete prefix'
    words = [(lower >> shift) & 0xffffffff for shift in range(BITS - 32, -1, -32)]
    lines = [
        '// Generated coefficients; see scripts/generate-airy-phase.py and',
        '// scripts/generate-airy-constants.cpp, and docs/tensor/native_airy.md.',
        '#pragma once', '#include <cstdint>',
        'namespace uni20::detail::native_math::airy_detail {',
        f'inline constexpr int phase_constant_bits = {BITS};',
        'inline constexpr std::uint32_t phase_constant[] = {',
    ]
    for index in range(0, len(words), 6):
        lines.append('  ' + ', '.join(f'0x{word:08x}u' for word in words[index:index + 6]) + ',')
    lines.extend(['};', '}'])
    destination.write_text('\n'.join(lines) + '\n')
    print('4/(3*pi) bits', BITS, 'prefix', hex(words[0]),
          'interval widths in guard units', error, 'Machin term budgets', da, db)


if __name__ == '__main__':
    main()
