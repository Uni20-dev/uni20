#pragma once

#include "native_airy_phase_data.hpp"
#include "native_math.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math::airy_detail
{
template <int Capacity> struct UInt
{
    // Capacity follows the scalar exponent bound; only size limbs are active.
    std::array<std::uint32_t, Capacity> w{};
    int size = 0;
    void trim()
    {
      while (size && !w[size - 1])
        --size;
    }
    int bits() const { return size ? 32 * (size - 1) + 32 - std::countl_zero(w[size - 1]) : 0; }
    bool bit(int k) const { return k >= 0 && k / 32 < size && ((w[k / 32] >> (k % 32)) & 1); }
    void shift(int n)
    {
      std::uint64_t c = 0;
      for (int i = 0; i < size; ++i)
      {
        auto t = (std::uint64_t(w[i]) << n) | c;
        w[i] = std::uint32_t(t);
        c = t >> 32;
      }
      if (c)
      {
        assert(size < Capacity);
        w[size++] = c;
      }
    }
    void add(unsigned n)
    {
      std::uint64_t c = n;
      for (int i = 0; i < size; ++i)
      {
        auto t = std::uint64_t(w[i]) + c;
        w[i] = std::uint32_t(t);
        c = t >> 32;
        if (!c) return;
      }
      if (c)
      {
        assert(size < Capacity);
        w[size++] = c;
      }
    }
    std::uint32_t trial_word(int i) const
    {
      return (i < size ? w[i] << 2 : 0u) | (i > 0 && i - 1 < size ? w[i - 1] >> 30 : 0u) | (i == 0 ? 1u : 0u);
    }
    bool subtract_trial(UInt const& root)
    {
      int n = root.size + (root.size && root.w[root.size - 1] >> 30 ? 1 : 0);
      if (!n) n = 1;
      bool ge = size >= n;
      if (size == n)
      {
        for (int i = size - 1; i >= 0; --i)
        {
          auto b = root.trial_word(i);
          if (w[i] != b)
          {
            ge = w[i] > b;
            break;
          }
        }
      }
      if (!ge) return false;
      std::uint64_t borrow = 0;
      for (int i = 0; i < size; ++i)
      {
        std::uint64_t b = std::uint64_t(root.trial_word(i)) + borrow;
        auto a = std::uint64_t(w[i]);
        w[i] = std::uint32_t(a - b);
        borrow = a < b;
      }
      assert(!borrow);
      this->trim();
      return true;
    }
    static UInt mul(UInt const& a, UInt const& b)
    {
      UInt r;
      r.size = a.size + b.size;
      assert(r.size <= Capacity);
      for (int i = 0; i < a.size; ++i)
      {
        std::uint64_t c = 0;
        for (int j = 0; j < b.size; ++j)
        {
          auto t = std::uint64_t(a.w[i]) * b.w[j] + r.w[i + j] + c;
          r.w[i + j] = std::uint32_t(t);
          c = t >> 32;
        }
        r.w[i + b.size] = c;
      }
      r.trim();
      return r;
    }
};
// Restoring binary square root of n*2^(2*shift), exactly rounded downward.
// Reading zero bit pairs avoids allocating the shifted radicand.
template <int Capacity> UInt<Capacity> root_scaled(UInt<Capacity> const& n, int shift)
{
  UInt<Capacity> root, rem;
  int pairs = (n.bits() + 2 * shift + 1) / 2;
  for (int i = pairs - 1; i >= 0; --i)
  {
    rem.shift(2);
    rem.add(unsigned(n.bit(2 * i + 1 - 2 * shift)) * 2 + unsigned(n.bit(2 * i - 2 * shift)));
    bool up = rem.subtract_trial(root);
    root.shift(1);
    if (up) root.add(1);
  }
  return root;
}
// Compute 4*x^(3/2)/(3*pi) modulo four from the original dyadic input.
// Integer-root and constant truncation lose <2^(-digits-31) phase units.
// The final native sin/cos sees only a reduced angle in [-pi/4,pi/4].
template <NativeReal R, int Capacity> std::pair<R, R> phase_impl(R x)
{
  using Integer = UInt<Capacity>;
  using L = uni20::numeric_limits<R>;
  int exponent;
  R f = std::frexp(x, &exponent);
  Integer m;
  for (int k = 0; k < L::digits(); ++k)
  {
    f *= 2;
    m.shift(1);
    if (f >= 1)
    {
      m.add(1);
      f -= 1;
    }
  }
  Integer n = Integer::mul(Integer::mul(m, m), m);
  int e = 3 * (exponent - L::digits());
  if (e % 2 != 0)
  {
    n.shift(1);
    --e;
  }
  e /= 2;
  constexpr int guard = L::digits() + 32;
  int shift = std::max(0, e + guard);
  Integer root = root_scaled(n, shift);
  int constant_bits = ((3 * exponent / 2 + guard + 16 + 31) / 32) * 32;
  assert(constant_bits > 0 && constant_bits <= phase_constant_bits);
  Integer c;
  c.size = constant_bits / 32;
  for (int i = 0; i < constant_bits / 32; ++i)
    c.w[i] = phase_constant[constant_bits / 32 - 1 - i];
  Integer prod = Integer::mul(root, c);
  int binary_point = shift - e + constant_bits;
  int q = (unsigned(prod.bit(binary_point)) + 2 * unsigned(prod.bit(binary_point + 1)) + 1) % 4;
  R frac = 0;
  for (int k = L::digits() + 16; k >= 1; --k)
    frac = (frac + R(prod.bit(binary_point - k))) / 2;
  R angle = (frac - R(.5)) * (uni20::pi<R> / 2);
  R s = std::sin(angle), co = std::cos(angle);
  switch (q)
  {
    case 0:
      return {s, co};
    case 1:
      return {co, -s};
    case 2:
      return {-s, -co};
    default:
      return {-co, s};
  }
}

// Return sin(zeta+pi/4), cos(zeta+pi/4), zeta=2*x^(3/2)/3.
// The small capacity avoids clearing a large frame for ordinary arguments.
template <NativeReal R> std::pair<R, R> phase(R x)
{
  static_assert(uni20::numeric_limits<R>::radix == 2 && uni20::numeric_limits<R>::digits() <= 113 &&
                uni20::numeric_limits<R>::max_exponent <= 16384);
  assert(x >= 4 && std::isfinite(x));
  int e;
  std::frexp(x, &e);
  if (e <= 128) return phase_impl<R, 32>(x);
  constexpr int max_bits = 3 * uni20::numeric_limits<R>::max_exponent / 2 + uni20::numeric_limits<R>::digits() + 64;
  return phase_impl<R, 2 * ((max_bits + 31) / 32) + 8>(x);
}
} // namespace uni20::detail::native_math::airy_detail
