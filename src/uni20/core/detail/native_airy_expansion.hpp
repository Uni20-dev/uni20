#pragma once

#include "native_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace uni20::detail::native_math::airy_detail
{
// Four nonoverlapping native words protect central-series cancellation.
// This private arithmetic has no independent scalar API or dynamic storage.
template <NativeReal R> class Expansion {
    static constexpr int words_ = 4;
    std::array<R, 2 * words_ * words_ + 2> terms_{};
    int size_ = 0;
    void add(R v)
    {
      int n = 0;
      for (int i = 0; i < size_; ++i)
      {
        R s = v + terms_[i], p = s - v, e = (v - (s - p)) + (terms_[i] - p);
        if (e != 0) terms_[n++] = e;
        v = s;
      }
      if (v != 0 || n == 0) terms_[n++] = v;
      size_ = n;
    }
    void compress()
    {
      R carry = this->high();
      int bottom = size_;
      for (int i = size_ - 2; i >= 0; --i)
      {
        R s = carry + terms_[i], e = terms_[i] - (s - carry);
        if (e != 0)
        {
          terms_[--bottom] = s;
          carry = e;
        }
        else
          carry = s;
      }
      int n = 0;
      for (int i = bottom; i < size_; ++i)
      {
        R s = terms_[i] + carry, e = carry - (s - terms_[i]);
        if (e != 0) terms_[n++] = e;
        carry = s;
      }
      terms_[n++] = carry;
      size_ = n;
      if (size_ > words_)
      {
        std::copy(terms_.begin() + size_ - words_, terms_.begin() + size_, terms_.begin());
        size_ = words_;
      }
    }

  public:
    Expansion(R x = 0) { this->add(x); }
    R high() const { return terms_[size_ - 1]; }
    Expansion scaled(int n) const
    {
      Expansion r = *this;
      for (int k = 0; k < size_; ++k)
        r.terms_[k] = std::ldexp(r.terms_[k], n);
      return r;
    }
    R rounded() const
    {
      R r = 0;
      for (int i = 0; i < size_; ++i)
        r += terms_[i];
      return r;
    }
    friend Expansion operator+(Expansion a, Expansion const& b)
    {
      for (int i = 0; i < b.size_; ++i)
        a.add(b.terms_[i]);
      a.compress();
      return a;
    }
    friend Expansion operator-(Expansion a)
    {
      for (int i = 0; i < a.size_; ++i)
        a.terms_[i] = -a.terms_[i];
      return a;
    }
    friend Expansion operator-(Expansion a, Expansion const& b) { return a + -b; }
    friend Expansion operator*(Expansion const& a, Expansion const& b)
    {
      Expansion r;
      for (int i = 0; i < a.size_; ++i)
        for (int j = 0; j < b.size_; ++j)
        {
          R p = a.terms_[i] * b.terms_[j];
          r.add(std::fma(a.terms_[i], b.terms_[j], -p));
          r.add(p);
        }
      r.compress();
      return r;
    }
    friend Expansion operator/(Expansion a, Expansion const& b)
    {
      Expansion q;
      for (int i = 0; i < words_; ++i)
      {
        R next = a.high() / b.high();
        Expansion v(next);
        q = q + v;
        a = a - v * b;
      }
      return q;
    }
    friend Expansion operator/(Expansion a, R b)
    {
      Expansion q;
      for (int i = 0; i < words_; ++i)
      {
        R next = a.high() / b;
        Expansion v(next);
        q = q + v;
        a = a - v * Expansion(b);
      }
      return q;
    }
};
} // namespace uni20::detail::native_math::airy_detail
