#pragma once

#include <gtest/gtest.h>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

namespace uni20::test
{
// GTest can print _Float128 through an integer conversion. Format scalar
// diagnostics explicitly, preserving all digits in the tested type.
template <class S> void expect_equal(S const& actual, S const& expected)
{
  EXPECT_TRUE(actual == expected) << "actual=" << format_scalar(actual) << " expected=" << format_scalar(expected);
}

// GTest's EXPECT_NEAR takes double. Keep subtraction, magnitude, and tolerance
// in the tested real type, and format only the diagnostic strings.
template <class S, class R> void expect_error_at_most(S const& actual, S const& expected, R const& tolerance)
{
  using std::abs;
  auto error = abs(actual - expected);
  EXPECT_TRUE(error <= tolerance) << "actual=" << format_scalar(actual) << " expected=" << format_scalar(expected)
                                  << " error=" << format_scalar(error) << " bound=" << format_scalar(tolerance);
}

} // namespace uni20::test
