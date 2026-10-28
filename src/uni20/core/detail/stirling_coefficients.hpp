#pragma once
#include <array>
#include <cstddef>

namespace uni20::detail
{
// B_(2k)/(2k*(2k-1)), k=1..16. Keep numerator/denominator conversion in R;
// no binary64 literal or MPFR dependency enters fixed-precision callers.
template <class R> constexpr std::array<R, 16> native_stirling_coefficients()
{
  return {{
      R(1LL) / R(12LL),
      R(-1LL) / R(360LL),
      R(1LL) / R(1260LL),
      R(-1LL) / R(1680LL),
      R(1LL) / R(1188LL),
      R(-691LL) / R(360360LL),
      R(1LL) / R(156LL),
      R(-3617LL) / R(122400LL),
      R(43867LL) / R(244188LL),
      R(-174611LL) / R(125400LL),
      R(77683LL) / R(5796LL),
      R(-236364091LL) / R(1506960LL),
      R(657931LL) / R(300LL),
      R(-3392780147LL) / R(93960LL),
      R(1723168255201LL) / R(2492028LL),
      R(-7709321041217LL) / R(505920LL),
  }};
}
} // namespace uni20::detail
