#pragma once

#include <uni20/random/transforms.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace uni20::random::test
{
// Use the exact axis and a one-grid-step rotation so the long-double reference
// retains the small component even when the original angle is nearly 2 pi.
template <class Evaluate> void check_box_muller_near_axes(Evaluate evaluate)
{
  constexpr std::array axis_cos{1, 0, -1, 0};
  constexpr std::array axis_sin{0, 1, 0, -1};
  long double const radius = std::sqrt(2 * std::numbers::ln2_v<long double>);
  for (unsigned axis = 0; axis < 4; ++axis)
    for (int step : {-1, 0, 1})
    {
      SCOPED_TRACE(::testing::Message() << "quarter turns = " << axis << ", grid steps = " << step);
      // u1 = 1/2. The angle word wraps modulo a full turn at axis zero.
      auto const angle_word = (std::uint64_t{axis} << 62) + static_cast<std::uint64_t>(step * 2048);
      Philox4x32Block const block{{UINT32_MAX, 0x7fffffff, static_cast<std::uint32_t>(angle_word),
                                   static_cast<std::uint32_t>(angle_word >> 32)}};
      long double const delta = step * 0x1p-52L * std::numbers::pi_v<long double>;
      auto const c = std::cos(delta);
      auto const s = std::sin(delta);
      std::array const expected{radius * (axis_cos[axis] * c - axis_sin[axis] * s),
                                radius * (axis_sin[axis] * c + axis_cos[axis] * s)};
      ValueBlock<double, 2> actual{};
      ASSERT_NO_FATAL_FAILURE(evaluate(block, actual));
      for (unsigned i = 0; i < 2; ++i)
        EXPECT_NEAR(actual.values[i], expected[i], 8 * std::numeric_limits<double>::epsilon() * std::abs(expected[i]));
    }
}
} // namespace uni20::random::test
